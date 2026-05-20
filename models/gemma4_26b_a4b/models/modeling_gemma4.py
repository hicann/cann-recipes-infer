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

import math
import os
import logging
from typing import List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_gemma4 import Gemma4TextConfig

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

    def forward(self, hidden_states):
        result = torch_npu.npu_rms_norm(
            hidden_states, self.weight if self.with_scale else torch.ones(
                hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
            ), self.eps
        )[0]
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
    """Dual-mode RoPE for sliding_attention and full_attention layers.

    - sliding_attention: head_dim=256, theta=10000, full rotation
    - full_attention: global_head_dim=512, theta=1000000, partial_rotary_factor=0.25
      => rotary_dim = 512 * 0.25 = 128 (only first 128 dims get RoPE)

    cos/sin are precomputed at init time up to max_position_embeddings to avoid
    .item() calls in forward, which would break graph mode.
    """

    def __init__(self, config, device=None, max_position_embeddings=4096):
        super().__init__()
        self.config = config
        self.max_position_embeddings = max_position_embeddings

        # sliding_attention RoPE
        sliding_params = config.rope_parameters["sliding_attention"]
        sliding_theta = sliding_params["rope_theta"]
        sliding_dim = config.head_dim  # 256
        inv_freq_sliding = 1.0 / (
            sliding_theta ** (torch.arange(0, sliding_dim, 2, dtype=torch.float32, device=device) / sliding_dim)
        )

        # full_attention RoPE (partial rotary)
        full_params = config.rope_parameters["full_attention"]
        full_theta = full_params["rope_theta"]
        partial_rotary_factor = full_params.get("partial_rotary_factor", 0.25)
        full_dim = config.global_head_dim  # 512
        rotary_dim = int(full_dim * partial_rotary_factor)  # 128
        self.rotary_dim_full = rotary_dim
        inv_freq_full = 1.0 / (
            full_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )

        # Precompute cos/sin up to max_position_embeddings for graph mode compatibility.
        # This eliminates .item() calls in forward. Values computed on CPU at init;
        # register_buffer ensures they move to NPU with the model.
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

    def forward(self, hidden_states, seq_len, kv_len, max_seq_len=None, layer_type="sliding_attention"):
        """
        Returns (cos, sin) shaped for BSH layout with npu_apply_rotary_pos_emb.

        For Prefill (seq_len > 1):  cos/sin shape = [1, S, 1, D_rope]
        For Decode (seq_len == 1):  cos/sin shape = [B, 1, 1, D_rope]  indexed by kv_len
        """
        batch_size = hidden_states.shape[0]

        if layer_type == "sliding_attention":
            cos, sin = self.cos_sliding, self.sin_sliding
        else:
            cos, sin = self.cos_full, self.sin_full

        if seq_len == 1:
            # Decode: index by kv_len per batch (no .item() needed)
            cos_out = torch.index_select(cos, dim=0, index=kv_len.long()).unsqueeze(1).unsqueeze(1)
            sin_out = torch.index_select(sin, dim=0, index=kv_len.long()).unsqueeze(1).unsqueeze(1)
        else:
            # Prefill: sequential positions
            cos_out = cos[:seq_len].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1, -1)
            sin_out = sin[:seq_len].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1, -1)

        return cos_out, sin_out


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply RoPE to only the first rotary_dim dimensions of q and k.

    Uses npu_rotary_mul on the rotated slice (fused mul+rotate_half+add) and
    leaves the trailing pass-through dimensions untouched.
    """
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rot = torch_npu.npu_rotary_mul(q_rot, cos, sin, rotary_mode='half')
    k_rot = torch_npu.npu_rotary_mul(k_rot, cos, sin, rotary_mode='half')

    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)
    return q, k


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
        # npu_fast_gelu matches gelu(approximate='tanh') and avoids aclgraph
        # CPU fallback on aten::gelu
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
        """
        hidden_states: [T, H] (flattened)
        Returns: topk_idx [T, K], topk_weight [T, K]
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)
        # Fused softmax + topk via npu_moe_gating_top_k_softmax
        top_k_weights, top_k_index, _ = torch_npu.npu_moe_gating_top_k_softmax(
            expert_scores, None, k=self.top_k_experts
        )
        # Post-processing: normalize weights and apply per-expert scaling
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_index, top_k_weights


class _GegluMoEMethod(QuantizeMethodBase):
    """Custom quant method that uses GEGLU (GELU_tanh) instead of SiLU.

    Gemma4 MoE experts use gelu_pytorch_tanh activation, not SiLU.
    The default UnquantizedFusedMoEGMMMethod hardcodes npu_swiglu.

    GEGLU mode is selected dynamically per-call via the is_prefill parameter:
    - Prefill (eager) or eager-only mode: npu_geglu (fused, high precision)
    - Decode in graph mode: manual GEGLU via F.gelu + mul (graph-compatible)
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
            # Prefill (always eager) or eager-only mode: use fused npu_geglu
            mm1_mm3, _ = torch_npu.npu_geglu(mm1_mm3, -1, 1)
        else:
            # Decode in graph mode: manual GEGLU using NPU-native gelu
            # (F.gelu hits CPU fallback under aclgraph capture; torch_npu.npu_geglu
            # meta has a signature mismatch under torch.compile)
            gate, up = mm1_mm3.chunk(2, dim=-1)
            mm1_mm3 = torch_npu.npu_fast_gelu(gate) * up
        out = torch_npu.npu_grouped_matmul(
            [mm1_mm3], [layer.w2_weight],
            group_list=expert_tokens, group_type=0,
            group_list_type=group_list_type, split_item=3,
        )[0]
        return out


class Gemma4GegluMoEGMM(FusedMoEGMM):
    """FusedMoEGMM subclass that uses GEGLU activation for Gemma4.

    In graph mode (ge_graph), npu_geglu is not supported so we fall back to
    manual GEGLU using standard PyTorch ops for Decode. Prefill always uses
    npu_geglu since it runs in eager mode.
    """

    def __init__(self, *args, **kwargs):
        use_manual_geglu = kwargs.pop("use_manual_geglu", False)
        super().__init__(*args, **kwargs)
        self.quant_method = _GegluMoEMethod(self.quant_method, use_manual_geglu=use_manual_geglu)

    def forward(self, x, expert_tokens, group_list_type=0, is_prefill=False, **kwargs):
        """Override to pass is_prefill through to the GEGLU method."""
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

        # Graph-mode dispatch: both ge_graph and npugraph_ex use local-expert MoE
        # (fixed shape, no AllToAll). Both also need the manual-GEGLU fallback:
        # ge_graph because torchair has no converter for npu_geglu, and
        # npugraph_ex because npu_geglu's torch.compile meta function has a
        # broken signature. Manual path uses npu_fast_gelu to stay graph-safe.
        self.exe_mode = infer_config.model_config.exe_mode
        use_manual_geglu = self.exe_mode in ("ge_graph", "npugraph_ex")
        # MoE EP decode dispatch strategy:
        #   "mc2": npu_moe_distribute_dispatch_v2 + npu_moe_distribute_combine_v2;
        #     the standard MC2 path used by qwen3_moe / longcat-flash / deepseek_r1.
        #     Atlas A2 caps experts_per_rank <= 24 on dispatch_v2; A3 has no such cap.
        #   "local_experts": every rank runs all experts under a routing mask, then
        #     AllReduce; required fallback when experts_per_rank > 24 on A2.
        # Default is "mc2" iff within the A2 cap (also fine on A3). Override via
        # model_config.custom_params.moe_ep_decode_mode.
        custom = getattr(infer_config.model_config, "custom_params", {}) or {}
        default_mode = "mc2" if self.experts_per_rank <= 24 else "local_experts"
        self._ep_decode_mode = custom.get("moe_ep_decode_mode", default_mode)
        if self._ep_decode_mode not in ("local_experts", "mc2"):
            raise ValueError(
                f"moe_ep_decode_mode must be 'local_experts' or 'mc2', got "
                f"{self._ep_decode_mode!r}"
            )
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

        # Pre-compute row_idx for Decode local-expert mode (graph-friendly, fixed shape)
        row_idx_decode_len = self.batch_size_decode * self.top_k
        self.row_idx_decode = torch.arange(
            0, row_idx_decode_len, dtype=torch.int32,
        ).view(self.top_k, -1).permute(1, 0).contiguous().npu()

        # Local expert range for this EP rank
        self.local_expert_start = self.ep_rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank

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
        """
        hidden_states: [T, H] (already flattened, normed for expert computation)
        topk_idx, topk_weight: pre-computed routing results (from router on un-normed input)
        Returns: [T, H]
        """
        if topk_idx is None or topk_weight is None:
            topk_idx, topk_weight = self.router(hidden_states)
        topk_idx = topk_idx.to(torch.int32)

        if self.moe_ep_size <= 1:
            # Single-rank MoE (no EP)
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, is_prefill)
        elif is_prefill:
            # EP Prefill: double_routing with AllToAll (always)
            return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, is_prefill=True)
        elif self._ep_decode_mode == "mc2":
            # EP Decode + mc2 dispatch_v2 (A2 cap experts_per_rank<=24; A3 no cap)
            return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight)
        elif self.exe_mode in ("ge_graph", "npugraph_ex"):
            # EP Decode + graph mode: local-expert (graph-friendly, no AllToAll)
            return self.moe_infer_local_experts(hidden_states, topk_idx, topk_weight)
        else:
            # EP Decode + eager: double_routing (exact, same as Prefill)
            return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, is_prefill=False)

    def moe_infer_tp(self, hidden_states, topk_idx, topk_weight, is_prefill=False):
        """Non-EP MoE path (TP only or single rank)."""
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
        """AllToAll dispatch for EP double-routing (Prefill)."""
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
        """EP double-routing path (Prefill or eager Decode)."""
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

        # Re-route gathered tokens to local experts
        hidden_states_ordered, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1))

        # Expert computation
        hidden_states_ordered = self.experts(
            hidden_states_ordered,
            tokens_per_local_expert,
            group_list_type=1,
            is_prefill=is_prefill,
        )

        # Restore order and AllToAll combine
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        new_x = torch.index_select(hidden_states_ordered, 0, gathered_ids_unsort.float().argsort().int())
        gathered_back = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_back, new_x, input_splits, output_splits, group=moe_ep_group)

        # Finalize routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_back, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_back.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )
        return hidden_states

    def moe_infer_dispatch_combine(self, hidden_states, topk_ids, topk_weight):
        """EP Decode: MC2 dispatch_v2 + combine_v2 (A2: experts_per_rank<=24; A3: no cap)."""
        self.set_mc2_kwargs()

        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            **self.dispatch_kwargs,
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        # Expert computation
        hidden_states_ordered = self.experts(expand_x, expert_token_num, group_list_type=1, is_prefill=False)

        # MC2 combine
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

    def moe_infer_local_experts(self, hidden_states, topk_ids, topk_weight):
        """Decode: graph-friendly local-expert mode using npu_moe_init_routing v1.

        Each rank only computes its local experts (experts_per_rank).
        Tokens routed to remote experts get zero contribution from this rank.
        All ranks' contributions are summed via all_reduce.

        Gemma-4 has no zero experts (all 128 are routed), so zero expert
        handling is omitted.
        """
        routing_weights = topk_weight.to(hidden_states.dtype)
        expert_idx = topk_ids.int()

        # Build local routing weights: non-local experts get weight=0
        not_local = (expert_idx < self.local_expert_start) | (expert_idx >= self.local_expert_end)
        routing_weights_local = routing_weights.masked_fill(not_local, 0)

        # Remap global expert IDs to local range [0, experts_per_rank)
        # Non-local experts are mapped to local expert 0 with weight=0
        local_expert_idx = expert_idx - self.local_expert_start
        local_expert_idx = torch.where(not_local, 0, local_expert_idx)

        # Route via init_routing v1 (graph-friendly, fixed-capacity expansion)
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=self.row_idx_decode,
            expert_idx=local_expert_idx,
            active_num=hidden_states.shape[0] * self.top_k,
        )

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, self.experts_per_rank
        )
        expert_tokens = expert_tokens.to(torch.int64)

        # Expert computation (local experts only, Decode path -> is_prefill=False)
        hidden_states_ordered = self.experts(expanded_x, expert_tokens, group_list_type=0, is_prefill=False)

        # Collect back with local routing weights
        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered,
            skip1=None, skip2=None, bias=None,
            scales=routing_weights_local.to(hidden_states_ordered.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=local_expert_idx,
        )

        # All-reduce across EP group to sum local expert contributions
        dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_ep_group"))
        return hidden_states


# ---------------------------------------------------------------------------
# Attention (dual mode: sliding + full)
# ---------------------------------------------------------------------------

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

        # Head dimensions differ between sliding and full attention
        if self.is_sliding:
            self.head_dim = config.head_dim  # 256
            self.num_kv_heads = config.num_key_value_heads  # 8
        else:
            self.head_dim = config.global_head_dim  # 512
            self.num_kv_heads = config.num_global_key_value_heads or config.num_key_value_heads  # 2

        self.num_heads = config.num_attention_heads  # 16
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        else:
            self.v_proj = None
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        # QK norms and V norm
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        # Gemma4 uses scaling=1.0 (not 1/sqrt(head_dim)) because QK norms
        # already normalize the query and key vectors before attention.
        self.scale_fa = 1.0

        # KV cache will be bound by ModelWorker (via self.k_cache / self.v_cache)
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.cache_unit = (self.num_kv_heads * self.head_dim,)

        self.batch_size = infer_config.scheduler_config.batch_size
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_gegraph = (self.exe_mode == "ge_graph")
        self.fa_ops = torch.ops.npu
        if self.enable_gegraph:
            import torchair as tng
            self.fa_ops = tng.ops

    def forward(
        self,
        hidden_states,
        cos_sin=None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        kv_len = forward_metadata.kv_len
        attention_mask = forward_metadata.attention_mask
        bsz, q_len, _ = hidden_states.size()

        # BSND shim: framework keeps kv_len/actual_seq_lengths_kv as packed [B*S]
        # across decode steps (_pad_batch is no-op when shape >= target). Pull
        # [B] views for BSND scatter_update / FA — start-slot per seq for prefill,
        # last-bsz for decode. Decode form correct only for batch_size_per_dp_rank == 1.
        if kv_len.shape[0] != bsz:
            if forward_metadata.is_prefill and kv_len.shape[0] == bsz * q_len:
                kv_len = kv_len[::q_len].contiguous()
            elif not forward_metadata.is_prefill:
                kv_len = kv_len[-bsz:].contiguous()

        # FA op variant: ge_graph uses torchair.ops (Tensor OK); eager and
        # npugraph_ex use torch.ops.npu, where dynamo enforces SymInt[] schema
        # for actual_seq_lengths_kv — must be a host list, not a Tensor.
        if self.exe_mode == "npugraph_ex" and forward_metadata.actual_seq_lengths_list_kv is not None:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_list_kv[-bsz:]
        else:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
            if actual_seq_lengths_kv is not None and actual_seq_lengths_kv.shape[0] != bsz:
                actual_seq_lengths_kv = actual_seq_lengths_kv[-bsz:].contiguous()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.v_proj is not None:
            value_states = self.v_proj(hidden_states)
        else:
            # k_eq_v: value = key (before RoPE and after k_norm + v_norm)
            value_states = key_states.clone()

        # Reshape for norm: [B, S, N, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Apply QK norms
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        # Apply RoPE
        cos, sin = cos_sin
        if self.is_sliding:
            # Fused RoPE for sliding attention (head_dim=256, BSND layout)
            # npu_rotary_mul fuses rotate_half + mul + add into single kernel
            query_states = torch_npu.npu_rotary_mul(query_states, cos, sin, rotary_mode='half')
            key_states = torch_npu.npu_rotary_mul(key_states, cos, sin, rotary_mode='half')
        else:
            # Partial rotation for full attention (rotary_dim=128)
            rotary_dim = cos.shape[-1]  # 128 (64*2)
            query_states, key_states = apply_partial_rotary_pos_emb(
                query_states, key_states, cos, sin, rotary_dim
            )

        # Reshape to BSH for cache and FA
        query_states = query_states.view(bsz, q_len, -1)  # [B, S, N*D]
        key_states = key_states.view(bsz, q_len, -1)  # [B, S, Nkv*D]
        value_states = value_states.view(bsz, q_len, -1)  # [B, S, Nkv*D]

        # KV Cache write (framework pre-allocates self.k_cache/self.v_cache via cache_unit)
        torch_npu.scatter_update_(self.k_cache, kv_len, key_states, -2)
        torch_npu.scatter_update_(self.v_cache, kv_len, value_states, -2)

        # FA attention
        if q_len == 1:
            past_key_states, past_value_states = self.k_cache, self.v_cache
            if self.is_sliding:
                decode_sparse_mode = 4
                decode_pre_tokens = self.sliding_window
            else:
                decode_sparse_mode = 0
                decode_pre_tokens = torch.iinfo(torch.int32).max
            attn_output, _ = self.fa_ops.npu_fused_infer_attention_score(
                query_states, past_key_states, past_value_states,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSH",
                atten_mask=None,
                sparse_mode=decode_sparse_mode,
                pre_tokens=decode_pre_tokens,
                next_tokens=0,
                scale=self.scale_fa,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
            )
        else:
            # Prefill: sliding layers use sparse_mode=4 (band/sliding window causal),
            # full attention layers use sparse_mode=3 (standard causal)
            if self.is_sliding:
                prefill_sparse_mode = 4
                prefill_pre_tokens = self.sliding_window  # 1024
            else:
                prefill_sparse_mode = 3
                prefill_pre_tokens = torch.iinfo(torch.int32).max
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query_states, key_states, value_states,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSH",
                atten_mask=attention_mask,
                sparse_mode=prefill_sparse_mode,
                pre_tokens=prefill_pre_tokens,
                next_tokens=0,
                scale=self.scale_fa,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


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

        # Layer scalar
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
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Fused: (residual + hidden_states) + pre_feedforward_layernorm
        hidden_states, residual = self.pre_feedforward_layernorm.forward_add(residual, hidden_states)

        # Dense MLP
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            # Post-norm the dense MLP output
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            # MoE path: route on raw residual (matching HF), expert input is normed
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            # Router operates on un-normed residual (router has its own internal norm)
            topk_idx, topk_weight = self.moe_block.router(hidden_states_flat)
            # Expert input is normed with pre_feedforward_layernorm_2
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.moe_block(
                hidden_states_2, is_prefill=is_prefill,
                topk_idx=topk_idx, topk_weight=topk_weight,
            )
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Fused: (hidden_states_1 + hidden_states_2) + post_feedforward_layernorm
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

        # Parallel settings
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.rank_id = infer_config.parallel_config.global_rank

        # Embedding
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

        # Precompute RoPE cos/sin up to max_position_embeddings for graph mode compat.
        # Must cover prefill (input_truncated_len) + decode (max_new_tokens) — RoPE
        # has no dynamic resize, out-of-range index_select crashes NPU vector unit.
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
        kv_len = forward_metadata.kv_len
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        attention_mask = forward_metadata.attention_mask
        is_prefill = forward_metadata.is_prefill
        batch_size, seq_length = input_ids.shape

        # Same BSND shim as Gemma4Attention: kv_len is packed [B*S]; reduce to [B]
        # for RoPE index_select. Idempotent; attention re-applies its own shim.
        if kv_len.shape[0] != batch_size:
            if is_prefill and kv_len.shape[0] == batch_size * seq_length:
                kv_len = kv_len[::seq_length].contiguous()
            elif not is_prefill:
                kv_len = kv_len[-batch_size:].contiguous()

        if self.embed_tp_size > 1:
            new_input_ids = input_ids - self.rank_id * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Compute RoPE cos/sin for each layer type
        max_pos = None
        cos_sin_dict = {}
        for layer_type in self.unique_layer_types:
            cos_sin_dict[layer_type] = self.rotary_emb(
                hidden_states, seq_length, kv_len, max_pos, layer_type
            )

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i] if hasattr(self.config, "layer_types") else "sliding_attention"
            hidden_states = decoder_layer(
                hidden_states,
                cos_sin=cos_sin_dict.get(layer_type),
                forward_metadata=forward_metadata,
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

        # Pull sizes from infer_config
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

        self.init_parallel_comm_group()
        self.model = Gemma4TextModel(config, infer_config, comm_manager, prefix=f"{prefix}model" if prefix else "model")
        self.experts_per_rank = (
            config.num_experts // self.moe_ep_size if getattr(config, "num_experts", None) else 0
        )

        # LMHead: ColumnParallelLinear when lmhead_tp > 1 else plain nn.Linear
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
            experts_per_rank = (
                self.config.num_experts // self.moe_ep_size
                if getattr(self.config, "num_experts", None)
                else 0
            )
            custom = getattr(self.infer_config.model_config, "custom_params", {}) or {}
            default_mode = "mc2" if experts_per_rank <= 24 else "local_experts"
            if (
                self.moe_tp_size == 1
                and experts_per_rank > 0
                and custom.get("moe_ep_decode_mode", default_mode) == "mc2"
            ):
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
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        # BSND shim: framework feeds packed 1D input_ids/position_ids; gemma
        # internals are BSND. Reshape at this boundary so all downstream sees 2D.
        # Do NOT mutate forward_metadata.kv_len here — framework's _build_model_inputs
        # reads it back next step expecting packed shape; attention does its own
        # local kv_len shim.
        if input_ids.dim() == 1:
            bs = self.infer_config.scheduler_config.batch_size_per_dp_rank
            if forward_metadata.is_prefill:
                if input_ids.shape[0] % bs != 0:
                    raise NotImplementedError(
                        "gemma-4 BSND-shim requires equal-length packed prefill "
                        f"(got total_tokens={input_ids.shape[0]}, "
                        f"batch_size_per_dp_rank={bs}). Variable-length packed "
                        "input requires migrating attention to TND layout."
                    )
                seq_length = input_ids.shape[0] // bs
                input_ids = input_ids.view(bs, seq_length)
                if position_ids is not None:
                    position_ids = position_ids.view(bs, seq_length)
            else:
                input_ids = input_ids.view(bs, 1)
                if position_ids is not None:
                    position_ids = position_ids.view(bs, 1)

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
        )

        # Only compute logits for last token during prefill
        if hidden_states.size(1) > 1 and position_ids is not None:
            gather_index, _ = torch.max(position_ids, dim=-1)
            gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, hidden_states.shape[-1])
            hidden_states = torch.gather(hidden_states, 1, gather_index)

        logits = self.lm_head(hidden_states)

        # LMHead AllGather for parallel
        if self.lmhead_tp_size > 1:
            new_logits = [torch.empty_like(logits) for _ in range(self.lmhead_tp_size)]
            dist.all_gather(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.cat(new_logits, dim=-1)

        # Final logit softcapping
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap

        logits = logits.float()
        return logits

    # GEGLU mode is now determined dynamically per-call via is_prefill parameter,
    # so set_geglu_mode() is no longer needed.

    def prefill(self, **kwargs):
        return self.forward(**kwargs)

    def decode(self, **kwargs):
        return self.forward(**kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        """Load weights from checkpoint, handling the multimodal prefix mapping.

        Checkpoint keys:  model.language_model.layers.X.xxx
        Our model keys:   model.layers.X.xxx

        Handles:
        - tie_word_embeddings: lm_head.weight = model.embed_tokens.weight
        - Expert weights: packed 3D tensors [num_experts, ...] sliced by EP rank
        - Skipping vision/audio tower weights
        - Name mapping: ckpt 'layers.X.router.*' -> model 'layers.X.moe_block.router.*'
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        all_params = {**params_dict, **buffers_dict}
        loaded_params = set()

        # Prefixes to skip (vision/audio tower)
        skip_prefixes = (
            "model.vision_tower",
            "model.audio_tower",
            "model.embed_vision",
            "model.embed_audio",
            "model.multimodal_projector",
        )

        # EP rank info for expert slicing
        ep_size = self.moe_ep_size
        experts_per_rank = self.experts_per_rank
        ep_rank = self.comm_manager.get_rank("moe_ep_group") if (ep_size > 1 and self.comm_manager is not None) else 0

        for name, loaded_weight in weights:
            # Skip non-text weights
            if any(name.startswith(p) for p in skip_prefixes):
                continue

            # Map checkpoint prefix: model.language_model.X -> model.X
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            # Handle tie_word_embeddings: embed_tokens.weight -> both embed and lm_head
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

            # Handle packed expert weights: experts.gate_up_proj [128, 1408, 2816]
            #                               experts.down_proj [128, 2816, 704]
            # Checkpoint: model.layers.X.experts.gate_up_proj
            # Model:      model.layers.X.moe_block.experts.w13_weight / w2_weight
            if name.endswith(".experts.gate_up_proj"):
                # gate_up_proj -> w13_weight (slice by EP rank)
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
                # down_proj -> w2_weight (slice by EP rank)
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

            # Handle router weights: ckpt 'layers.X.router.*' -> model 'layers.X.moe_block.router.*'
            if ".router." in name and ".moe_block.router." not in name:
                name = name.replace(".router.", ".moe_block.router.")

            if name not in all_params:
                continue

            param = all_params[name]
            if isinstance(param, nn.Parameter):
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                # Buffer (e.g. layer_scalar)
                param.copy_(loaded_weight)
            loaded_params.add(name)

        return loaded_params

    def process_weights_after_loading(self):
        """Post-process MoE weights after loading — transpose + NZ/quant formatting.

        Matches the framework's expected flow: ModelWorker calls this after weight
        load to let each module's quant_method rewrite the stored weight tensor
        into the layout the grouped-matmul op actually consumes (e.g. transpose
        K/N dims, pack NZ format). Without this, `npu_grouped_matmul` sees the
        raw HF layout and fails with K-mismatch.
        """
        is_nz = self.infer_config.model_config.enable_weight_nz
        for module in self.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            if hasattr(quant_method, "process_weights_after_loading"):
                quant_method.process_weights_after_loading(module, is_nz=is_nz)
