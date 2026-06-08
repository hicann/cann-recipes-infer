# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2024-2026 LatenceAI. All rights reserved.
# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma4 text-only decoder model for cann-recipes-infer framework.

Covers only the Language MoE Decoder path. Vision/Audio towers are skipped.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu

from transformers.activations import ACT2FN

from module.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization import QuantizeMethodBase
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.model_loader.weight_utils import default_weight_loader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            # with_scale=False：gamma 恒为 1。预分配 ones buffer 避免每次 forward 新建临时
            # 张量；persistent=False 不进 state_dict，dtype/device 随 module 与 input 对齐。
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, hidden_states):
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.eps)[0]
        return result

    def forward_add(self, x1, x2):
        """Fused add + rms_norm: computes rms_norm(x1 + x2) and (x1 + x2).

        Returns:
            (normed, residual_sum) where normed = rms_norm(x1+x2), residual_sum = x1+x2
        """
        normed, _, residual_sum = torch_npu.npu_add_rms_norm(
            x1, x2, self.weight, self.eps
        )
        return normed, residual_sum


# ---------------------------------------------------------------------------
# Rotary Embedding - dual mode (sliding + full attention)
# ---------------------------------------------------------------------------

class Gemma4RotaryEmbedding(nn.Module):
    """Dual RoPE: sliding (full head_dim) and full (partial_rotary_factor)."""

    def __init__(self, config, device=None, max_position_embeddings=4096):
        super().__init__()
        self.config = config
        self.max_position_embeddings = max_position_embeddings

        sliding_params = config.rope_parameters["sliding_attention"]
        sliding_theta = sliding_params["rope_theta"]
        sliding_dim = config.head_dim
        inv_freq_sliding = 1.0 / (
            sliding_theta ** (torch.arange(0, sliding_dim, 2, dtype=torch.float32, device=device) / sliding_dim)
        )

        full_params = config.rope_parameters["full_attention"]
        full_theta = full_params["rope_theta"]
        partial_rotary_factor = full_params.get("partial_rotary_factor", 0.25)
        full_dim = config.global_head_dim
        rotary_dim = int(full_dim * partial_rotary_factor)
        self.rotary_dim_full = rotary_dim
        inv_freq_full = 1.0 / (
            full_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )

        # Precompute cos/sin to avoid .item() in graph mode.
        cos_sliding, sin_sliding = self._compute_cos_sin(
            max_position_embeddings, inv_freq_sliding, device, torch.bfloat16
        )
        cos_full, sin_full = self._compute_cos_sin(
            max_position_embeddings, inv_freq_full, device, torch.bfloat16
        )
        self.register_buffer("cos_sliding", cos_sliding, persistent=False)
        self.register_buffer("sin_sliding", sin_sliding, persistent=False)
        self.register_buffer("cos_full", cos_full, persistent=False)
        self.register_buffer("sin_full", sin_full, persistent=False)

    @staticmethod
    def _compute_cos_sin(seq_len, inv_freq, device, dtype, scaling=1.0):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.to(device) if device else inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * scaling).to(dtype)
        sin = (emb.sin() * scaling).to(dtype)
        return cos, sin

    def forward(self, position_ids, layer_type="sliding_attention"):
        """Returns (cos, sin) shaped [T, 1, D_rope] for TND broadcast over heads."""
        if layer_type == "sliding_attention":
            cos, sin = self.cos_sliding, self.sin_sliding
        else:
            cos, sin = self.cos_full, self.sin_full

        cos_out = cos.index_select(0, position_ids).unsqueeze(1)
        sin_out = sin.index_select(0, position_ids).unsqueeze(1)
        return cos_out, sin_out


# ---------------------------------------------------------------------------
# Scaled Word Embedding
# ---------------------------------------------------------------------------

class Gemma4ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma4ScaledVocabParallelEmbedding(nn.Module):
    """VocabParallelEmbedding wrapper with Gemma4 scaling."""
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0,
                 tp_size=1, tp_rank=0):
        super().__init__()
        self.embed_scale = embed_scale
        self.embedding = VocabParallelEmbedding(
            num_embeddings, hidden_size=embedding_dim, padding_idx=padding_idx,
            params_dtype=torch.bfloat16, tp_size=tp_size, tp_rank=tp_rank,
        )

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.embed_scale


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------

class Gemma4MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        # npu_fast_gelu == gelu(approximate='tanh'), graph-safe.
        return self.down_proj(torch_npu.npu_fast_gelu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE: Router and SparseMoeBlock (EP-capable)
# ---------------------------------------------------------------------------

class Gemma4Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size ** -0.5
        self.eps = config.rms_norm_eps
        self.top_k_experts = config.top_k_experts
        self.num_experts = config.num_experts

        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states):
        """Returns (topk_idx [T, K], topk_weight [T, K])."""
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        top_k_weights, top_k_index, _ = torch_npu.npu_moe_gating_top_k_softmax(
            expert_scores, None, k=self.top_k_experts
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_index, top_k_weights


class _GegluMoEMethod(QuantizeMethodBase):
    """MoE expert method using GEGLU (GELU_tanh) instead of SiLU.

    Prefill / eager: fused npu_geglu. Graph-mode decode: manual gelu * up
    (npu_geglu has no graph converter / has broken torch.compile meta).
    """

    def __init__(self, base_method, use_manual_geglu=False):
        self._base = base_method
        self.use_manual_geglu = use_manual_geglu

    def create_weights(self, *args, **kwargs):
        return self._base.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, *args, **kwargs):
        return self._base.process_weights_after_loading(*args, **kwargs)

    def apply(self, layer, x, expert_tokens, group_list_type, **kwargs):
        is_prefill = kwargs.get("is_prefill", False)
        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [layer.w13_weight],
            group_list=expert_tokens, group_type=0,
            group_list_type=group_list_type, split_item=3,
        )[0]
        if is_prefill or not self.use_manual_geglu:
            mm1_mm3, _ = torch_npu.npu_geglu(mm1_mm3, -1, 1)
        else:
            gate, up = mm1_mm3.chunk(2, dim=-1)
            mm1_mm3 = torch_npu.npu_fast_gelu(gate) * up
        out = torch_npu.npu_grouped_matmul(
            [mm1_mm3], [layer.w2_weight],
            group_list=expert_tokens, group_type=0,
            group_list_type=group_list_type, split_item=3,
        )[0]
        return out


class Gemma4GegluMoEGMM(FusedMoEGMM):
    """FusedMoEGMM with GEGLU activation."""

    def __init__(self, *args, **kwargs):
        use_manual_geglu = kwargs.pop("use_manual_geglu", False)
        super().__init__(*args, **kwargs)
        self.quant_method = _GegluMoEMethod(self.quant_method, use_manual_geglu=use_manual_geglu)

    def forward(self, x, expert_tokens, group_list_type=0, is_prefill=False, **kwargs):
        return self.quant_method.apply(
            layer=self, x=x, expert_tokens=expert_tokens,
            group_list_type=group_list_type, is_prefill=is_prefill, **kwargs,
        )


class Gemma4SparseMoeBlock(nn.Module):
    """MoE block with EP support using FusedMoEGMM + NPU routing operators."""

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.moe_intermediate_size = config.moe_intermediate_size

        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.experts_per_rank = self.num_experts // self.moe_ep_size

        self.batch_size_decode = infer_config.scheduler_config.batch_size_per_dp_rank

        self.ep_rank = comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0

        # Graph mode needs manual GEGLU (npu_geglu has no graph converter / broken meta).
        self.exe_mode = infer_config.model_config.exe_mode
        use_manual_geglu = self.exe_mode in ("ge_graph", "npugraph_ex")
        self.experts = Gemma4GegluMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            bias=False,
            quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.ep_rank,
            use_manual_geglu=use_manual_geglu,
        )
        self.router = Gemma4Router(config)

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        self.dispatch_kwargs = {
            "x_active_mask": None,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "scales": None,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": global_rank // self.moe_tp_size,
            "group_tp": moe_ep_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": global_rank % self.moe_tp_size,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "quant_mode": 0,
            "comm_alg": "fullmesh_v2",
        }
        self.combine_kwargs = {
            "x_active_mask": None,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": global_rank // self.moe_tp_size,
            "group_tp": moe_ep_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": global_rank % self.moe_tp_size,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "comm_quant_mode": 0,
        }

    def forward(self, hidden_states, is_prefill=False, topk_idx=None, topk_weight=None):
        if topk_idx is None or topk_weight is None:
            topk_idx, topk_weight = self.router(hidden_states)
        topk_idx = topk_idx.to(torch.int32)

        if self.moe_ep_size <= 1:
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, is_prefill)
        if is_prefill:
            return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, is_prefill=True)
        return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight)

    def moe_infer_tp(self, hidden_states, topk_idx, topk_weight, is_prefill=False):
        """Single-rank / TP-only MoE."""
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_idx,
            active_num=topk_idx.shape[0] * topk_idx.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )

        hidden_states_ordered_by_experts = self.experts(
            expanded_x,
            tokens_per_expert,
            group_list_type=1,
            is_prefill=is_prefill,
        )

        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_tp_group"))
        return hidden_states

    def dispatch_double_routing(self, tokens_per_expert, expanded_x):
        """AllToAll dispatch for EP double-routing."""
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)

        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, input_splits, output_splits

    def moe_infer_double_routing(self, hidden_states, topk_ids, topk_weight, is_prefill=False):
        """EP double-routing (prefill or eager decode)."""
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=None,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )

        tokens_per_expert_group, gathered_tokens, input_splits, output_splits = \
            self.dispatch_double_routing(tokens_per_expert, expanded_x)

        hidden_states_ordered, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1))

        hidden_states_ordered = self.experts(
            hidden_states_ordered,
            tokens_per_local_expert,
            group_list_type=1,
            is_prefill=is_prefill,
        )

        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        new_x = torch.index_select(hidden_states_ordered, 0, gathered_ids_unsort.float().argsort().int())
        gathered_back = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_back, new_x, input_splits, output_splits, group=moe_ep_group)

        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_back, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_back.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )
        return hidden_states

    def moe_infer_dispatch_combine(self, hidden_states, topk_ids, topk_weight):
        """EP decode: MC2 dispatch_v2 + combine_v2."""
        # kwargs 全为静态标量，已在权重加载后预构建；guard 避免每次 forward 重复构建
        # （含 dist.get_rank 等 host 调用），并兜底未预构建的路径。
        if getattr(self, "dispatch_kwargs", None) is None:
            self.set_mc2_kwargs()

        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            **self.dispatch_kwargs,
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        hidden_states_ordered = self.experts(expand_x, expert_token_num, group_list_type=1, is_prefill=False)

        combine_args = {
            "expand_x": hidden_states_ordered,
            "expert_ids": topk_ids,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32),
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            **self.combine_kwargs,
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)
        return hidden_states

# ---------------------------------------------------------------------------
# Attention (dual mode: sliding + full)
# ---------------------------------------------------------------------------


@dataclass
class _AttnInputs:
    """Per-call FA helper inputs (bundles QKV + PA context + forward metadata)."""
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    is_prefill: bool
    attention_mask: Optional[torch.Tensor]
    slot_mapping: dict
    block_table: dict
    forward_metadata: ForwardMetaData
    q_len: int


class Gemma4Attention(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = (self.layer_type == "sliding_attention")

        if self.is_sliding:
            self.head_dim = config.head_dim
            self.num_kv_heads = config.num_key_value_heads
        else:
            self.head_dim = config.global_head_dim
            self.num_kv_heads = config.num_global_key_value_heads or config.num_key_value_heads

        self.num_heads = config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        else:
            self.v_proj = None
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # QK norm absorbs the 1/sqrt(d) scaling.
        self.scale_fa = 1.0

        # Paged KV cache; framework binds the allocated tensor via tensor_setter.
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.attn_type = "SlidingWindow" if self.is_sliding else "FullAttention"
        self.block_size = infer_config.scheduler_config.block_size
        cache_dtype = torch.bfloat16
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_kv_heads,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
                sliding_window=self.sliding_window if self.sliding_window else None,
            ),
            CacheEntry(
                cache_name="v_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_kv_heads,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
                sliding_window=self.sliding_window if self.sliding_window else None,
            ),
        ]

        self.batch_size = infer_config.scheduler_config.batch_size
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_gegraph = (self.exe_mode == "ge_graph")
        # FA op dispatch: prefill always torch.ops.npu; decode under ge_graph
        # uses torchair.ops (tng.ops). See _fa_ops_for().
        self._torchair_ops = None
        if self.enable_gegraph:
            import torchair as tng
            self._torchair_ops = tng.ops

    def _fa_ops_for(self, is_prefill: bool):
        if (not is_prefill) and self.enable_gegraph and self._torchair_ops is not None:
            return self._torchair_ops
        return torch.ops.npu

    def forward(
        self,
        hidden_states,
        cos_sin=None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping=None,
        block_table=None,
        **kwargs,
    ):
        """Hybrid: sliding -> TND-PA, full -> BNSD-PA. Packed [T, H] in, [T, H] out."""
        is_prefill = forward_metadata.is_prefill
        q_len = hidden_states.shape[0]

        if is_prefill:
            # 变长 prefill 暂不支持：sliding 层 RoPE 的 4D reshape 与 full 层 BNSD reshape
            # 均按 (batch_size, seq_len) 切分，要求同批次各请求等长，变长会在 reshape 处
            # 直接崩。此处提前给出可读错误。decode 每请求 1 token 天然等长、且图模式禁
            # Tensor 驱动断言，故仅在 prefill 检查。
            seq_q = forward_metadata.actual_seq_lengths_q
            if seq_q is not None and seq_q.numel() > 1 and not bool((seq_q == seq_q[0]).all()):
                raise ValueError(
                    "variable-length prefill is not supported: sliding RoPE 4D view and "
                    "full BNSD reshape require equal per-request lengths within a batch; "
                    f"got actual_seq_lengths_q={seq_q.tolist()}"
                )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        # k_eq_v variant: value shares the key projection.
        value_states = self.v_proj(hidden_states) if self.v_proj is not None else key_states.clone()

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_kv_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        # Sliding: full head_dim fused rotate-half via npu_rotary_mul.
        # 3D TND → 4D BSND view (zero-copy) — npu_rotary_mul requires 4D in graph mode.
        # Full: partial rotary_dim via npu_apply_rotary_pos_emb on 3D TND.
        cos, sin = cos_sin
        if self.is_sliding:
            # npu_rotary_mul 需 4D 输入，按 (batch_size, seq_len) reshape，假设同批次内
            # 各请求等长（q_len = batch_size × seq_len）。decode 每请求 1 token 恒成立；
            # 等长 prefill 成立；变长 prefill 暂不支持。
            batch_size = forward_metadata.actual_seq_lengths_cu_q.shape[0]
            seq_len = q_len // batch_size
            q4 = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k4 = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            cos4 = cos.view(batch_size, seq_len, 1, self.head_dim)
            sin4 = sin.view(batch_size, seq_len, 1, self.head_dim)
            q4 = torch_npu.npu_rotary_mul(q4, cos4, sin4, rotary_mode='half')
            k4 = torch_npu.npu_rotary_mul(k4, cos4, sin4, rotary_mode='half')
            query_states = q4.view(q_len, self.num_heads, self.head_dim)
            key_states = k4.view(q_len, self.num_kv_heads, self.head_dim)
        else:
            rotary_dim = cos.shape[-1]
            q_rot = query_states[..., :rotary_dim].contiguous()
            q_pass = query_states[..., rotary_dim:].contiguous()
            k_rot = key_states[..., :rotary_dim].contiguous()
            k_pass = key_states[..., rotary_dim:].contiguous()
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
                q_rot, k_rot, cos, sin, layout="TND",
            )
            query_states = torch.cat([q_rot, q_pass], dim=-1)
            key_states = torch.cat([k_rot, k_pass], dim=-1)

        attention_mask = forward_metadata.attention_mask

        if not (self.k_cache.numel() and self.v_cache.numel()):
            raise RuntimeError(
                "k_cache/v_cache not bound. Ensure Gemma4ForCausalLM.get_cache_info() "
                "is implemented and ModelWorker._init_kvcache has run.",
            )

        attn_inputs = _AttnInputs(
            query=query_states, key=key_states, value=value_states,
            is_prefill=is_prefill, attention_mask=attention_mask,
            slot_mapping=slot_mapping, block_table=block_table,
            forward_metadata=forward_metadata, q_len=q_len,
        )
        if self.is_sliding:
            attn_output = self._attn_tnd_sliding(attn_inputs)
        else:
            attn_output = self._attn_bnsd_full(attn_inputs)

        # attn_output is [T, N*D].
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _attn_tnd_sliding(self, inputs: _AttnInputs):
        """TND + FA v2 + PA for sliding layers (head_dim=256).

        TND cache uses flat-H view (blocknum, blocksize, H=N*D) — CANN 9.0.0
        TND non-MLA whitelist includes D=256.
        """
        is_prefill = inputs.is_prefill
        forward_metadata = inputs.forward_metadata
        sparse_mode = 4
        pre_tokens = self.sliding_window
        if self.exe_mode == "npugraph_ex" and not is_prefill:
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
        else:
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv

        if is_prefill:
            attn_output, _ = self._fa_ops_for(is_prefill).npu_fused_infer_attention_score_v2(
                inputs.query, inputs.key, inputs.value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale_fa,
                input_layout="TND",
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=0,
                atten_mask=inputs.attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_qlen,
            )
            self._update_cache(inputs.slot_mapping, inputs.key, inputs.value)
        else:
            self._update_cache(inputs.slot_mapping, inputs.key, inputs.value)
            attn_output, _ = self._fa_ops_for(is_prefill).npu_fused_infer_attention_score_v2(
                inputs.query,
                self.k_cache.view(*self.k_cache.shape[:2], -1),
                self.v_cache.view(*self.v_cache.shape[:2], -1),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale_fa,
                input_layout="TND",
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=0,
                atten_mask=inputs.attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                block_table=inputs.block_table[self.attn_type],
                block_size=self.block_size,
            )
        return attn_output.reshape(inputs.q_len, self.num_heads * self.head_dim)

    def _attn_bnsd_full(self, inputs: _AttnInputs):
        """BNSD + FA v2 + PA for full layers (head_dim=512).

        TND non-MLA rejects D=512; BNSD has no D cap. Q transposes to
        [B,N,S,D]; KV cache uses non-contig BNBD view [bn, N, bs, D].
        """
        is_prefill = inputs.is_prefill
        forward_metadata = inputs.forward_metadata
        sparse_mode_prefill = 3
        pre_tokens = torch.iinfo(torch.int32).max
        if self.exe_mode == "npugraph_ex" and not is_prefill:
            per_batch_q = forward_metadata.actual_seq_lengths_list_q
            per_batch_kv = forward_metadata.actual_seq_lengths_list_kv
        else:
            per_batch_q = forward_metadata.actual_seq_lengths_q
            per_batch_kv = forward_metadata.actual_seq_lengths_kv
        batch_size = forward_metadata.actual_seq_lengths_cu_q.shape[0]

        # BNSD reshape 同样假设同批次内各请求等长（同 sliding 分支）；变长 batch 暂不支持。
        seq_len = inputs.q_len // batch_size
        q_bnsd = inputs.query.view(
            batch_size, seq_len, self.num_heads, self.head_dim,
        ).transpose(1, 2).contiguous()

        if is_prefill:
            k_bnsd = inputs.key.view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim,
            ).transpose(1, 2).contiguous()
            v_bnsd = inputs.value.view(
                batch_size, seq_len, self.num_kv_heads, self.head_dim,
            ).transpose(1, 2).contiguous()
            attn_output, _ = self._fa_ops_for(is_prefill).npu_fused_infer_attention_score_v2(
                q_bnsd, k_bnsd, v_bnsd,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale_fa,
                input_layout="BNSD",
                sparse_mode=sparse_mode_prefill,
                pre_tokens=pre_tokens,
                next_tokens=0,
                atten_mask=inputs.attention_mask,
                actual_seq_qlen=per_batch_q,
                actual_seq_kvlen=per_batch_q,
            )
            self._update_cache(inputs.slot_mapping, inputs.key, inputs.value)
        else:
            self._update_cache(inputs.slot_mapping, inputs.key, inputs.value)
            k_cache_bnbd = self.k_cache.transpose(1, 2)
            v_cache_bnbd = self.v_cache.transpose(1, 2)
            attn_output, _ = self._fa_ops_for(is_prefill).npu_fused_infer_attention_score_v2(
                q_bnsd, k_cache_bnbd, v_cache_bnbd,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale_fa,
                input_layout="BNSD",
                sparse_mode=0,
                pre_tokens=pre_tokens,
                next_tokens=0,
                atten_mask=None,
                actual_seq_qlen=per_batch_q,
                actual_seq_kvlen=per_batch_kv,
                block_table=inputs.block_table[self.attn_type],
                block_size=self.block_size,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(inputs.q_len, self.num_heads * self.head_dim)

    def _update_cache(self, slot_mapping, key_states, value_states):
        tmp_slot = slot_mapping[self.attn_type].view(-1, 1)
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_kv_heads, self.head_dim),
            tmp_slot,
            key_states,
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_kv_heads, self.head_dim),
            tmp_slot,
            value_states,
        )


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Gemma4Attention(config, infer_config, comm_manager, layer_idx, prefix=f"{prefix}.self_attn")
        self.mlp = Gemma4MLP(config, layer_idx)

        # LayerNorms
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Per-layer 真实权重：HF 每层一个 layer_scalar（非 1），由 load_weights 的 buffer
        # 路径按 name 匹配加载（torch.ones 仅为加载前占位），forward 末尾乘到该层输出上。
        self.register_buffer("layer_scalar", torch.ones(1))

        # MoE block (every layer has both dense MLP and MoE)
        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.moe_block = Gemma4SparseMoeBlock(config, infer_config, comm_manager, prefix=f"{prefix}.moe_block")
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        cos_sin=None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping=None,
        block_table=None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            slot_mapping=slot_mapping,
            block_table=block_table,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm.forward_add(residual, hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            # Router runs on un-normed residual (has its own internal norm).
            topk_idx, topk_weight = self.moe_block.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe_block(
                hidden_states_2, is_prefill=is_prefill,
                topk_idx=topk_idx, topk_weight=topk_weight,
            )
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states, _ = self.post_feedforward_layernorm.forward_add(hidden_states_1, hidden_states_2)
        else:
            hidden_states = self.post_feedforward_layernorm(hidden_states)

        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Text Model
# ---------------------------------------------------------------------------

class Gemma4TextModel(nn.Module):

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.rank_id = infer_config.parallel_config.global_rank

        if self.embed_tp_size > 1:
            embed_tp_rank = comm_manager.get_rank("embed_tp_group") if comm_manager else 0
            self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
            self.embed_tokens = Gemma4ScaledVocabParallelEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx,
                embed_scale=config.hidden_size ** 0.5,
                tp_size=self.embed_tp_size,
                tp_rank=embed_tp_rank,
            )
        else:
            self.vocab_size_per_rank = self.vocab_size
            self.embed_tokens = Gemma4ScaledWordEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx,
                embed_scale=config.hidden_size ** 0.5,
            )

        self.layers = nn.ModuleList([
            Gemma4DecoderLayer(config, layer_idx, infer_config, comm_manager, prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        max_pos = (
            infer_config.data_config.input_truncated_len
            + infer_config.scheduler_config.max_new_tokens
        )
        self.rotary_emb = Gemma4RotaryEmbedding(config, max_position_embeddings=max_pos)
        self.unique_layer_types = set(config.layer_types)

    def forward(
        self,
        input_ids,
        position_ids=None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        """Packed 1D input_ids → packed 2D [T, H] through layer stack."""
        if position_ids is None:
            raise RuntimeError("Gemma4TextModel requires packed 1D position_ids.")
        slot_mapping = forward_metadata.slot_mapping
        block_table = forward_metadata.block_table

        if self.embed_tp_size > 1:
            new_input_ids = input_ids - self.rank_id * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Cache cos/sin per layer_type to avoid duplicate index_select across 30 layers.
        cos_sin_dict = {
            layer_type: self.rotary_emb(position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i] if hasattr(self.config, "layer_types") else "sliding_attention"
            hidden_states = decoder_layer(
                hidden_states,
                cos_sin=cos_sin_dict.get(layer_type),
                forward_metadata=forward_metadata,
                slot_mapping=slot_mapping,
                block_table=block_table,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------

class Gemma4ForCausalLM(nn.Module):

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.vocab_size = config.vocab_size
        self.exe_mode = getattr(infer_config.model_config, "exe_mode", "eager")

        pc = infer_config.parallel_config
        self.world_size = pc.world_size
        self.attn_tp_size = pc.attn_tp_size
        self.moe_tp_size = pc.moe_tp_size
        self.moe_ep_size = pc.moe_ep_size
        self.embed_tp_size = pc.embed_tp_size
        self.lmhead_tp_size = pc.lmhead_tp_size
        self.attn_dp_size = pc.attn_dp_size
        self.embed_dp_size = pc.embed_dp_size

        self.max_position_embeddings = (
            infer_config.data_config.input_truncated_len
            + infer_config.scheduler_config.max_new_tokens
        )
        self.batch_size = infer_config.scheduler_config.batch_size
        self.block_size = infer_config.scheduler_config.block_size

        self.init_parallel_comm_group()
        self.model = Gemma4TextModel(config, infer_config, comm_manager, prefix=f"{prefix}model" if prefix else "model")
        self.experts_per_rank = (
            config.num_experts // self.moe_ep_size if getattr(config, "num_experts", None) else 0
        )
        self.num_experts = getattr(config, "num_experts", 0) or 0
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 0) or 0

        if self.lmhead_tp_size > 1:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                tp_size=self.lmhead_tp_size,
                tp_rank=comm_manager.get_rank("lmhead_tp_group"),
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def init_parallel_comm_group(self):
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=self.world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
        )
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=self.world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
        )
        if self.moe_tp_size > 1:
            self.comm_manager.register_group(
                name="moe_tp_group",
                group_num=self.world_size // self.moe_tp_size,
                group_size=self.moe_tp_size,
            )
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
            )
            if self.moe_tp_size == 1 and self.moe_ep_size > 1 and getattr(self.config, "num_experts", 0):
                moe_ep_mc2_buffer_size = calc_moe_hccl_buffer_size(
                    self.infer_config, self.config, is_full_mesh_v2=True
                )
                self.comm_manager.register_group(
                    name="moe_ep_group_mc2",
                    group_num=moe_ep_group_num,
                    group_size=self.moe_ep_size,
                    group_stride=moe_ep_group_num,
                    return_name=True,
                    allow_physical_reuse=False,
                    hccl_buffer_size=moe_ep_mc2_buffer_size,
                    group_type=None,
                )

    def forward(
        self,
        input_ids,
        position_ids=None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        """
        Packed 1D input_ids → logits [bs, 1, V]. Prefill gathers last token
        per sequence via cu_q-1; decode is already [bs, H].
        """
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
        )

        if forward_metadata.is_prefill:
            cu_q = forward_metadata.actual_seq_lengths_cu_q
            if cu_q is None:
                raise RuntimeError(
                    "Prefill path requires forward_metadata.actual_seq_lengths_cu_q.",
                )
            hidden_states = torch.index_select(hidden_states, 0, cu_q - 1)

        logits = self.lm_head(hidden_states)

        # AllGather lm_head TP shards on vocab dim.
        if self.lmhead_tp_size > 1:
            tp = self.lmhead_tp_size
            gathered = torch.empty(
                [logits.shape[0] * tp, logits.shape[-1]],
                dtype=logits.dtype, device=logits.device,
            )
            dist.all_gather_into_tensor(
                gathered, logits,
                group=self.comm_manager.get_group("lmhead_tp_group"),
            )
            bsz = logits.shape[0]
            logits = gathered.view(tp, bsz, logits.shape[-1]).permute(1, 0, 2).reshape(bsz, -1)

        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap

        # Framework slices logits[:, -1:, :] downstream; keep 3D.
        logits = logits.view(logits.shape[0], 1, -1)
        logits = logits.float()
        return logits

    def get_cache_info(self) -> ModelCacheInfo:
        """
        Per-layer cache_entries (k_cache + v_cache) grouped by attn_type
        for the framework PA allocator.
        """
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(layer.self_attn.cache_entries),
                )
            )
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            block_size=self.block_size,
            layer_infos=layer_infos,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """
        Load weights, mapping HF multimodal-prefixed keys to the LM-only
        layout, slicing packed expert tensors by EP rank, and tying lm_head
        to embed_tokens when applicable.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        all_params = {**params_dict, **buffers_dict}
        loaded_params = set()

        skip_prefixes = (
            "model.vision_tower",
            "model.audio_tower",
            "model.embed_vision",
            "model.embed_audio",
            "model.multimodal_projector",
        )

        ep_size = self.moe_ep_size
        experts_per_rank = self.experts_per_rank
        ep_rank = self.comm_manager.get_rank("moe_ep_group") if (ep_size > 1 and self.comm_manager is not None) else 0

        for name, loaded_weight in weights:
            if any(name.startswith(p) for p in skip_prefixes):
                continue

            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            # tie_word_embeddings: shared with lm_head.
            if name == "model.embed_tokens.weight":
                for target_name in ["model.embed_tokens.weight",
                                    "model.embed_tokens.embedding.weight",
                                    "lm_head.weight"]:
                    if target_name in params_dict:
                        param = params_dict[target_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(target_name)
                continue

            # Packed expert tensors → EP-rank slice into w13/w2.
            if name.endswith(".experts.gate_up_proj"):
                target_name = name.replace(".experts.gate_up_proj",
                                           ".moe_block.experts.w13_weight")
                if target_name in params_dict:
                    param = params_dict[target_name]
                    start_expert = ep_rank * experts_per_rank
                    end_expert = start_expert + experts_per_rank
                    sliced = loaded_weight[start_expert:end_expert].contiguous()
                    param.data.copy_(sliced)
                    loaded_params.add(target_name)
                continue

            if name.endswith(".experts.down_proj"):
                target_name = name.replace(".experts.down_proj",
                                           ".moe_block.experts.w2_weight")
                if target_name in params_dict:
                    param = params_dict[target_name]
                    start_expert = ep_rank * experts_per_rank
                    end_expert = start_expert + experts_per_rank
                    sliced = loaded_weight[start_expert:end_expert].contiguous()
                    param.data.copy_(sliced)
                    loaded_params.add(target_name)
                continue

            if ".router." in name and ".moe_block.router." not in name:
                name = name.replace(".router.", ".moe_block.router.")

            if name not in all_params:
                continue

            param = all_params[name]
            if isinstance(param, nn.Parameter):
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                param.copy_(loaded_weight)
            loaded_params.add(name)

        return loaded_params

    def process_weights_after_loading(self):
        """
        Apply per-module quant_method weight transforms (NZ pack, K/N
        transpose) expected by npu_grouped_matmul, and eagerly fill MC2 kwargs
        on MoE blocks so the first decode step doesn't trigger graph recompile.
        """
        is_nz = self.infer_config.model_config.enable_weight_nz
        for module in self.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            if hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(module, is_nz=is_nz)

        for module in self.modules():
            if isinstance(module, Gemma4SparseMoeBlock) and module.moe_ep_size > 1:
                module.set_mc2_kwargs()
