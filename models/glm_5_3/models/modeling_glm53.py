# coding=utf-8
# GLM-5.3 (Glm5Next) NPU inference modeling.
# Adapted from
# https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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

import logging
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

import torch_npu

from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils.forward_metadata import ForwardMetaData
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.stream_utils import (
    create_event, create_stream, npu_stream_switch, record_event,
    record_stream, wait_event,
)
from executor.model_loader.weight_utils import default_weight_loader
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization.mxfp8 import MxFp8LinearMethod, reshape_mx_scale

from .configuration_glm53 import Glm53Config
from .modules import Glm53RMSNorm, hc_pre, hc_post, hc_head_mean
from .kda import Glm53KdaAttention, Glm53StepMetaData
from .indexer import (
    Glm53Indexer, build_pool_gather,
    build_pool_commit_slots,
)

logger = logging.getLogger(__name__)


class Glm53FFN(nn.Module):
    """Dense FFN and the MoE shared expert"""

    def __init__(self, config, infer_config, intermediate_size, prefix, **kwargs):
        super().__init__()
        self.exe_mode = infer_config.model_config.exe_mode
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size, output_sizes=[intermediate_size] * 2,
            bias=False, tp_size=1, tp_rank=0,
            quant_config=config.quant_config, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            intermediate_size, config.hidden_size,
            bias=False, tp_size=1, tp_rank=0,
            quant_config=config.quant_config, prefix=f"{prefix}.down_proj")
        if isinstance(getattr(self.down_proj, "scheme", None), MxFp8LinearMethod):
            self.forward = self.forward_mxfp8

    def forward(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged = self.gate_up_proj(x)
        inter = merged.shape[-1] // 2
        merged[..., :inter].clamp_(max=self.swiglu_limit)
        merged[..., inter:].clamp_(min=-self.swiglu_limit, max=self.swiglu_limit)
        intermediate = torch_npu.npu_swiglu(merged)
        wait_event(enable_decode_stream and shared_expert_event is not None,
                   shared_expert_event, 0, exe_mode=self.exe_mode)
        return self.down_proj(intermediate)

    def forward_mxfp8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged = self.gate_up_proj(x)
        intermediate, pergroup_scale, _ = torch.ops.custom.npu_swiglu_group_quant(
            merged,
            dst_type=torch.float8_e4m3fn,
            round_scale=True,
            quant_mode=1,          # 1 = MX (per-32-channel e8m0 scale)
            clamp_limit=self.swiglu_limit,
        )
        wait_event(enable_decode_stream and shared_expert_event is not None,
                   shared_expert_event, 0, exe_mode=self.exe_mode)
        return self.down_proj(intermediate, pergroup_scale)


class Glm53MoE(nn.Module):
    """MoE block: routed experts plus one shared expert."""

    DISPATCH_QUANT_MODE = {
        "w16a16": 0,
        "w8a8float8": 3,
        "w8a8mxfloat8": 4,
        "w4a8mxfloat4": 4,
    }

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx=None, prefix="", **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.comm_manager = comm_manager
        parallel_config = infer_config.parallel_config
        self.moe_tp_size = parallel_config.moe_tp_size
        self.moe_ep_size = parallel_config.moe_ep_size
        self.global_rank = parallel_config.global_rank
        self.platform_version = infer_config.model_config.platform_version.value \
            if hasattr(infer_config.model_config.platform_version, "value") \
            else str(infer_config.model_config.platform_version)
        self.hidden_dim = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=0,
            ep_size=self.moe_ep_size,
            ep_rank=(comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0),
            prefix=f"{prefix}.experts",
        )

        gmm_quant_mode = (config.quant_config.gmm_quant_mode
                          if config.quant_config is not None else "w16a16")
        if gmm_quant_mode not in ("w16a16", "w4a8mxfloat4"):
            raise NotImplementedError(
                f"glm_5_3 supports bf16 and W4A8-MXFP4 experts, got {gmm_quant_mode}")
        self.gmm_act_kwargs = ({} if gmm_quant_mode == "w16a16" else
                               {"swiglu_limit": config.swiglu_limit,
                                "enable_custom_ops": True})

        self.gate = ReplicatedLinear(
            config.hidden_size, self.num_experts, bias=False, quant_config=None,
            params_dtype=torch.float32, prefix=f"{prefix}.gate")
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32), requires_grad=False)

        # Shared-expert overlap: the shared expert does not depend on the routed
        # path, so run it on a side stream and join right before the combine.
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", True)
        self.shared_expert_stream = kwargs.get("shared_expert_stream", None)
        self.npu_events = tuple(
            create_event(self.exe_mode, self.enable_multi_streams) for _ in range(2))
        self.shared_expert_event = (create_event(self.exe_mode, self.enable_multi_streams),)
        self.gmm_quant_mode = gmm_quant_mode
        self.dispatch_quant_mode = self.DISPATCH_QUANT_MODE.get(gmm_quant_mode, 0)

        self.quantized_dispatch = "a8mxfloat" in gmm_quant_mode
        self.shared_experts = Glm53FFN(
            config, infer_config,
            config.moe_intermediate_size * config.n_shared_experts,
            prefix=f"{prefix}.shared_experts", **kwargs)

    def _forward_gate(self, hidden_states):
        logits = self.gate(hidden_states.float())
        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            k=self.top_k,
            bias=self.gate.e_score_correction_bias.float(),
            k_group=self.topk_group,
            group_count=self.n_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1,  # sigmoid
            routed_scaling_factor=self.routed_scaling_factor,
            eps=float(1e-20))
        return topk_idx, topk_weight

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, **kwargs):
        if is_prefill or self.moe_ep_size == 1:
            hidden_states_share = self.shared_experts(hidden_states)
        else:
            record_stream(self.enable_multi_streams, hidden_states,
                          self.shared_expert_stream, exe_mode=self.exe_mode)
            record_event(self.enable_multi_streams, self.npu_events, 0,
                         exe_mode=self.exe_mode)
            hidden_states_share = None
        topk_idx, topk_weight = self._forward_gate(hidden_states)
        if cur_topk_list is not None:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        if self.moe_ep_size == 1:
            return self._moe_infer_local(hidden_states, topk_idx, topk_weight,
                                         hidden_states_share)
        if is_prefill:
            return self._moe_infer_double_routing(hidden_states, topk_idx, topk_weight,
                                                  hidden_states_share)
        return self._moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight)

    def _moe_infer_local(self, x, topk_ids, topk_weight, hidden_states_share):
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            x,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1)
        out = self.experts(expanded_x, tokens_per_expert, group_list_type=1,
                           **self.gmm_act_kwargs)
        return torch_npu.npu_moe_finalize_routing(
            out, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(out.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2)

    def _moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share):
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")

        if self.quantized_dispatch:
            moe_init_routing = torch_npu.npu_moe_init_routing_group_quant
            routing_args = {"quant_mode": 3, "row_idx_type": 0, "drop_pad_mode": 0}
        else:
            moe_init_routing = torch_npu.npu_moe_init_routing_v2
            routing_args = {"quant_mode": -1}

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = moe_init_routing(
            x,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=None,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            **routing_args)

        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits,
                               group=moe_ep_group)

        gathered_pertoken_scale = None
        if self.quantized_dispatch:
            pertoken_scale = pertoken_scale.view(torch.int8)
            gathered_pertoken_scale = pertoken_scale.new_empty(
                gathered_tokens.shape[0], *pertoken_scale.shape[1:])
            dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale,
                                   output_splits, input_splits, group=moe_ep_group)
            # npu_moe_re_routing wants the scales 2-D; the GMM wants them back as
            # (rows, cols // 2, 2). Flatten going in, reshape_mx_scale coming out.
            gathered_pertoken_scale = gathered_pertoken_scale.view(
                torch.float8_e8m0fnu).flatten(1)

        hidden_ordered, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(
                gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=gathered_pertoken_scale)
        gmm_kwargs = {}
        if self.quantized_dispatch:
            gmm_kwargs["pertoken_scale"] = reshape_mx_scale(gathered_pertoken_scale)
        hidden_ordered = self.experts(hidden_ordered, tokens_per_local_expert,
                                      group_list_type=1, **gmm_kwargs,
                                      **self.gmm_act_kwargs)
        new_x = torch.index_select(
            hidden_ordered, 0, gathered_ids_unsort.float().argsort().int())

        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits,
                               group=moe_ep_group)

        return torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2)

    def _moe_infer_dispatch_combine(self, x, topk_ids, topk_weight):
        """EP decode path (npu_moe_distribute dispatch/combine mc2), from glm_5."""
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        common = {
            "x_active_mask": None,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "group_ep": mc2_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": self.global_rank // self.moe_tp_size,
            "group_tp": mc2_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": self.global_rank % self.moe_tp_size,
        }
        quant_mode = self.dispatch_quant_mode
        dispatch_kwargs = dict(common, scales=None, quant_mode=quant_mode)
        if quant_mode in (3, 4):
            dispatch_kwargs["y_dtype"] = torch.float8_e4m3fn
        if self.platform_version != "950":
            dispatch_kwargs["comm_alg"] = "fullmesh_v2"

        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x, expert_ids=topk_ids, **dispatch_kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, \
            tp_recv_counts = output[:6]

        gmm_kwargs = {}
        if "a8" in self.gmm_quant_mode:
            # Tokens arrive already quantized, so the expert method must not
            # re-quantize them; hand it the scale the dispatch produced.
            if "mxfloat" in self.gmm_quant_mode:
                dynamic_scale = reshape_mx_scale(dynamic_scale)
            gmm_kwargs["pertoken_scale"] = dynamic_scale

        expand_x = self.experts(expand_x, expert_token_num, group_list_type=1,
                                **gmm_kwargs, **self.gmm_act_kwargs)

        # Routed GMM is issued: run the shared expert on the side stream so it
        # overlaps the combine below.
        enable = self.enable_multi_streams
        record_event(enable, self.shared_expert_event, 0, exe_mode=self.exe_mode)
        with npu_stream_switch(enable, self.shared_expert_stream,
                               exe_mode=self.exe_mode):
            wait_event(enable, self.npu_events, 0, exe_mode=self.exe_mode)
            hidden_states_share = self.shared_experts(
                x, enable_decode_stream=enable,
                shared_expert_event=self.shared_expert_event)
            record_event(enable, self.npu_events, 1, exe_mode=self.exe_mode)

        wait_event(enable, self.npu_events, 1, exe_mode=self.exe_mode)
        record_stream(enable, hidden_states_share, torch.npu.current_stream(),
                      exe_mode=self.exe_mode)
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expand_x,
            shared_expert_x=hidden_states_share,
            expert_ids=topk_ids,
            assist_info_for_combine=expand_idx,
            expert_scales=topk_weight.to(torch.float32),
            ep_send_counts=ep_recv_counts,
            tp_send_counts=tp_recv_counts,
            **common)


class SharedZeroRope:
    """Zero rope tensors for the NoPE DSA layers, cached and shared."""

    def __init__(self, rope_dim: int = 64):
        self.rope_dim = rope_dim
        self.tensor: Optional[torch.Tensor] = None
        # keyed by is_prefill: separate slots stop an interleaved prefill from
        # reallocating the decode buffer that npugraph_ex has captured
        self.query: Dict[bool, torch.Tensor] = {}

    def get_query(self, like: torch.Tensor, tokens: int, num_heads: int,
                  is_prefill: bool) -> torch.Tensor:
        buf = self.query.get(is_prefill)
        wanted = (tokens, num_heads, like.dtype, like.device)
        if buf is None or (*buf.shape[:2], buf.dtype, buf.device) != wanted:
            buf = torch.zeros(tokens, num_heads, self.rope_dim,
                              dtype=like.dtype, device=like.device)
            self.query[is_prefill] = buf
        return buf

    def get(self, like_cache: torch.Tensor) -> torch.Tensor:
        if self.tensor is None or self.tensor.shape[0] != like_cache.shape[0]:
            self.tensor = torch.zeros(
                like_cache.shape[0], like_cache.shape[1], 1, self.rope_dim,
                dtype=torch.bfloat16, device=like_cache.device)
        return self.tensor


class Glm53SparseAttention(nn.Module):
    """DSA attention: absorbed MLA, NoPE, sparse over the indexer's top-k."""

    # Pooled-key cache manager key; its own group because FullAttentionManager
    # requires every entry in a group to share one compress_ratio.
    POOL_MANAGER_KEY = "DsaPool"

    def __init__(self, config: Glm53Config, infer_config: InferenceConfig,
                 comm_manager: CommManager, layer_idx: int, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim  # == qk_nope_head_dim (NoPE)
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        self.index_topk = config.index_topk

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        if self.attn_tp_size > 1:
            raise NotImplementedError("glm_5_3 supports attn_tp_size == 1 only")
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size

        self.zero_rope: SharedZeroRope = kwargs["zero_rope"]
        self.block_size = infer_config.scheduler_config.block_size

        quant_config = getattr(config, "quant_config", None)
        self.q_a_proj = ReplicatedLinear(self.hidden_size, self.q_lora_rank, bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.q_a_proj")
        self.q_a_layernorm = Glm53RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim,
            bias=False, tp_size=1, tp_rank=0, quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj")
        # NoPE: kv_a projects to kv_lora_rank only (no +qk_rope_head_dim block)
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size, self.kv_lora_rank, bias=False, quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = Glm53RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # kv_b_proj must stay BF16: it is split into the absorbed w_k / w_v below.
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, tp_size=1, tp_rank=0, quant_config=None,
            prefix=f"{prefix}.kv_b_proj")
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False,
            tp_size=1, tp_rank=0, input_is_parallel=True, quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        # absorbed weights, split from kv_b_proj after loading:
        #   kv_b_proj_w_k [N, qk_nope, kv_lora]  (q absorb)
        #   kv_b_proj_w_v [N, kv_lora, v_head]   (output projection)
        self.kv_b_proj_w_k = None
        self.kv_b_proj_w_v = None

        self.indexer = Glm53Indexer(config, layer_idx)

        # framework-managed paged caches
        self.attn_type = "FullAttention"
        self.nope_cache = torch.Tensor([])
        self.indexer_key_cache = torch.Tensor([])
        self.indexer_gate_cache = torch.Tensor([])
        self.indexer_pool_cache = torch.Tensor([])
        if self.block_size % config.index_kpool != 0:
            # _commit_pools reads a pool's kpool tokens as consecutive slots,
            # which only holds while a pool cannot straddle a block.
            raise ValueError(
                f"block_size ({self.block_size}) must be divisible by "
                f"index_kpool ({config.index_kpool})")
        self.cache_entries = [
            CacheEntry(
                cache_name="nope_cache",
                attn_type=self.attn_type,
                dim=self.kv_lora_rank,
                num_head=1,
                dtype=torch.bfloat16,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "nope_cache", tensor),
            ),
            CacheEntry(
                cache_name="indexer_key_cache",
                attn_type=self.attn_type,
                dim=config.index_head_dim,
                num_head=1,
                dtype=torch.bfloat16,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(
                    layer, "indexer_key_cache", tensor),
            ),
            CacheEntry(
                cache_name="indexer_gate_cache",
                attn_type=self.attn_type,
                dim=config.index_head_dim,
                num_head=1,
                dtype=torch.bfloat16,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(
                    layer, "indexer_gate_cache", tensor),
            ),
            # Pooled indexer keys, one entry per complete pool of kpool tokens.
            # Own manager (own block pool and block table) with
            # compress_ratio=kpool, so block_size is the logical token span and
            # storage_block_size = block_size // kpool physical slots.
            CacheEntry(
                cache_name="indexer_pool_cache",
                attn_type=self.attn_type,
                dim=config.index_head_dim,
                num_head=1,
                dtype=torch.bfloat16,
                needs_block=True,
                block_size=self.block_size * config.index_kpool,
                manager_key=self.POOL_MANAGER_KEY,
                compress_ratio=config.index_kpool,
                tensor_setter=lambda tensor, layer=self: setattr(
                    layer, "indexer_pool_cache", tensor),
            ),
        ]

    def prepare_absorbed_weights(self):
        w = self.kv_b_proj.weight  # [N*(nope+v), kv_lora] (pre-transpose layout)
        w = w.view(self.num_heads_per_rank, self.qk_nope_head_dim + self.v_head_dim,
                   self.kv_lora_rank)
        w_k = w[:, :self.qk_nope_head_dim, :]                      # [N, nope, lora]
        w_v = w[:, self.qk_nope_head_dim:, :].transpose(-1, -2)    # [N, lora, v]
        self.kv_b_proj_w_k = nn.Parameter(w_k.contiguous(), requires_grad=False)
        self.kv_b_proj_w_v = nn.Parameter(w_v.contiguous(), requires_grad=False)
        self.kv_b_proj.weight = None

    def forward(
        self,
        hidden_states: torch.Tensor,             # [T, hidden] packed TND
        position_ids: torch.Tensor,               # [T]
        forward_metadata: ForwardMetaData,
        step_metadata: Glm53StepMetaData,
        **kwargs,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        slot_mapping = forward_metadata.slot_mapping[self.attn_type]
        block_table = forward_metadata.block_table[self.attn_type]
        block_table_pool = forward_metadata.block_table[self.POOL_MANAGER_KEY]
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        actual_seq_lengths_cu_q = forward_metadata.actual_seq_lengths_cu_q

        # ---- projections + latent cache write ----
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_nope = self.q_b_proj(q_resid).view(tokens, self.num_heads_per_rank,
                                             self.qk_head_dim)
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(hidden_states))
        torch_npu.npu_scatter_nd_update_(
            self.nope_cache.view(-1, self.kv_lora_rank),
            slot_mapping.view(-1, 1),
            latent.to(self.nope_cache.dtype))

        # ---- top-k selection (k-pool indexer) ----
        topk_indices = self.indexer(
            hidden_states, q_resid, position_ids,
            self.indexer_key_cache, self.indexer_gate_cache, slot_mapping,
            self.indexer_pool_cache, step_metadata.pool_commit_slots,
            step_metadata.n_pools_total, step_metadata.pool_slots,
            step_metadata.pool_valid,
            pool_block_table=block_table_pool,
            n_pools_per_req=step_metadata.n_pools_per_req,
            cu_q=actual_seq_lengths_cu_q,
            is_decode=not forward_metadata.is_prefill,
            query_boundaries=step_metadata.query_boundaries)

        q_latent = torch.einsum("tnd,ndl->tnl", q_nope, self.kv_b_proj_w_k)

        attn_lat = self._sfa_attention(
            q_latent, topk_indices, block_table,
            actual_seq_lengths_cu_q, actual_seq_lengths_kv,
            forward_metadata.is_prefill)

        # ---- epilog: latent -> v heads -> o_proj (glm_5 layout) ----
        # attn_lat: [N, T, kv_lora]
        attn_output = torch.matmul(attn_lat, self.kv_b_proj_w_v).transpose(0, 1)
        attn_output = attn_output.reshape(tokens, self.num_heads_per_rank * self.v_head_dim)
        return self.o_proj(attn_output)

    def _sfa_attention(self, q_latent, topk_indices, block_table,
                       actual_seq_lengths_cu_q, actual_seq_lengths_kv,
                       is_prefill):
        """npu_sparse_flash_attention over the paged latent cache; [N, T, lora]."""
        tokens = q_latent.shape[0]
        sfa_kwargs = {
            "query": q_latent,
            "key": self.nope_cache,
            "value": self.nope_cache,
            "sparse_indices": topk_indices.view(tokens, 1, -1),
            "scale_value": self.softmax_scale,
            "actual_seq_lengths_query": actual_seq_lengths_cu_q.to(torch.int32),
            "actual_seq_lengths_kv": actual_seq_lengths_kv.to(torch.int32),
            "block_table": block_table,
            "sparse_block_size": 1,
            "layout_query": "TND",
            "layout_kv": "PA_BSND",
            "sparse_mode": 3,
            "attention_mode": 2,
            "query_rope": self.zero_rope.get_query(
                q_latent, tokens, self.num_heads_per_rank, is_prefill),
            "key_rope": self.zero_rope.get(self.nope_cache),
        }
        attn_out, _, _ = torch_npu.npu_sparse_flash_attention(**sfa_kwargs)
        return attn_out.transpose(0, 1)  # [T, N, lora] -> [N, T, lora]


class Glm53DecoderLayer(nn.Module):
    """Decoder layer: mHC-wrapped attention plus FFN."""

    def __init__(self, config: Glm53Config, infer_config: InferenceConfig,
                 comm_manager: CommManager, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        if self.is_kda:
            self.self_attn = Glm53KdaAttention(
                config, infer_config, comm_manager, layer_idx,
                prefix=f"{prefix}.self_attn", **kwargs)
        else:
            self.self_attn = Glm53SparseAttention(
                config, infer_config, comm_manager, layer_idx,
                prefix=f"{prefix}.self_attn", **kwargs)

        if self.is_moe:
            self.mlp = Glm53MoE(config, infer_config, comm_manager, layer_idx=layer_idx,
                                prefix=f"{prefix}.mlp", **kwargs)
        else:
            self.mlp = Glm53FFN(config, infer_config, config.intermediate_size,
                                prefix=f"{prefix}.mlp", **kwargs)

        self.input_layernorm = Glm53RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm53RMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            self.register_parameter(name, nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False))
        for name in ("hc_attn_base", "hc_ffn_base"):
            self.register_parameter(name, nn.Parameter(
                torch.empty(mix_hc, dtype=torch.float32), requires_grad=False))
        for name in ("hc_attn_scale", "hc_ffn_scale"):
            self.register_parameter(name, nn.Parameter(
                torch.empty(3, dtype=torch.float32), requires_grad=False))

    def forward(
        self,
        hidden_streams: torch.Tensor,            # [T, hc, D]
        position_ids: torch.Tensor,
        forward_metadata: ForwardMetaData,
        step_metadata: Glm53StepMetaData,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # ---- attention sublayer ----
        residual = hidden_streams
        x, post, comb = hc_pre(hidden_streams, self.hc_attn_fn, self.hc_attn_scale,
                               self.hc_attn_base, self.hc_mult, self.hc_sinkhorn_iters,
                               self.norm_eps, self.hc_eps)
        x = self.input_layernorm(x)
        if self.is_kda:
            x = self.self_attn(hidden_states=x, forward_metadata=forward_metadata,
                               step_metadata=step_metadata)
        else:
            x = self.self_attn(
                hidden_states=x, position_ids=position_ids,
                forward_metadata=forward_metadata, step_metadata=step_metadata)
        hidden_streams = hc_post(x, residual, post, comb)

        # ---- FFN sublayer ----
        residual = hidden_streams
        x, post, comb = hc_pre(hidden_streams, self.hc_ffn_fn, self.hc_ffn_scale,
                               self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
                               self.norm_eps, self.hc_eps)
        x = self.post_attention_layernorm(x)
        if self.is_moe:
            x = self.mlp(x, is_prefill=forward_metadata.is_prefill,
                         cur_topk_list=cur_topk_list)
        else:
            x = self.mlp(x)
        hidden_streams = hc_post(x, residual, post, comb)

        return hidden_streams


class Glm53Model(nn.Module):
    def __init__(self, config: Glm53Config, infer_config: InferenceConfig,
                 comm_manager: CommManager, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.block_size = infer_config.scheduler_config.block_size
        self.hc_mult = config.hc_mult
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.vocab_size_per_rank = config.vocab_size // self.embed_tp_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, None, torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=(comm_manager.get_rank("embed_tp_group")
                     if self.embed_tp_size > 1 else 0))
        self.layers = nn.ModuleList([
            Glm53DecoderLayer(config, infer_config, comm_manager, layer_idx,
                              prefix=f"model.layers.{layer_idx}", **kwargs)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Glm53RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def calc_input_embeddings(self, input_ids):
        if self.embed_tp_size > 1:
            embed_rank = self.comm_manager.get_rank("embed_tp_group")
            new_input_ids = input_ids - embed_rank * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            inputs_embeds = self.embed_tokens(new_input_ids * mask) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
            return inputs_embeds
        return self.embed_tokens(input_ids)

    def build_step_metadata(self, forward_metadata: ForwardMetaData,
                            device, position_ids: torch.Tensor = None) -> Glm53StepMetaData:
        """Per-forward metadata bundle, shared by all layers.

        The prefill branch reads packed request boundaries on the host; that is
        fine because graph mode only ever compiles the decode step. The decode
        branch stays free of host reads so it traces under fullgraph=True.
        """
        kpool = self.config.index_kpool
        pool_bt = forward_metadata.block_table[Glm53SparseAttention.POOL_MANAGER_KEY]
        n_pools = forward_metadata.actual_seq_lengths_kv // kpool

        if forward_metadata.is_prefill:
            pool_slots, pool_valid = build_pool_gather(pool_bt, n_pools,
                                                       self.block_size)
            n_pools_total = pool_slots.shape[1]
            cu = forward_metadata.actual_seq_lengths_cu_q
            boundaries = [0] + cu.cpu().tolist()
            token_idx = torch.arange(boundaries[-1], device=device)
            token_batch_idx = torch.bucketize(token_idx, cu, right=True)
        else:
            pool_slots = pool_valid = None
            n_pools_total = pool_bt.shape[1] * self.block_size
            token_batch_idx = torch.arange(
                forward_metadata.actual_seq_lengths_kv.shape[0], device=device)

        common = dict(
            token_batch_idx=token_batch_idx,
            n_pools_total=n_pools_total,
            pool_slots=pool_slots,
            pool_valid=pool_valid,
            pool_commit_slots=build_pool_commit_slots(
                pool_bt, position_ids, token_batch_idx, self.block_size, kpool),
            n_pools_per_req=n_pools,
        )
        if not forward_metadata.is_prefill:
            return Glm53StepMetaData(**common)
        return Glm53StepMetaData(
            **common,
            query_boundaries=boundaries,
            query_start_loc=F.pad(cu, (1, 0)).to(torch.int32),
            has_initial_state=torch.zeros(len(boundaries) - 1, dtype=torch.int32,
                                          device=device),
        )

    def forward(
        self,
        input_ids: torch.Tensor,           # [T] packed
        position_ids: torch.Tensor,         # [T]
        forward_metadata: ForwardMetaData,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.calc_input_embeddings(input_ids.to(torch.int32))
        step_metadata = self.build_step_metadata(
            forward_metadata, hidden_states.device, position_ids)

        # expand into hc_mult parallel residual streams [T, hc, D]
        hidden_streams = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for decoder_layer in self.layers:
            hidden_streams = decoder_layer(
                hidden_streams,
                position_ids=position_ids,
                forward_metadata=forward_metadata,
                step_metadata=step_metadata,
                cur_topk_list=cur_topk_list,
            )

        hidden_states = hc_head_mean(hidden_streams)
        return self.norm(hidden_states)  # [T, D]


class Glm53ForCausalLM(nn.Module):
    def __init__(self, config: Glm53Config, infer_config: InferenceConfig,
                 comm_manager: CommManager = None, prefix: str = ""):
        super().__init__()
        if comm_manager is None:
            raise RuntimeError("Glm53ForCausalLM requires a CommManager "
                               "(executor framework path)")
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        parallel_config = infer_config.parallel_config
        self.world_size = parallel_config.world_size
        self.attn_tp_size = parallel_config.attn_tp_size
        self.embed_tp_size = parallel_config.embed_tp_size
        self.lmhead_tp_size = parallel_config.lmhead_tp_size
        self.moe_tp_size = parallel_config.moe_tp_size
        self.moe_ep_size = parallel_config.moe_ep_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.check_model_settings()
        self.init_parallel_comm_group()

        # One side stream for the whole model; the per-layer events live on
        # each Glm53MoE.
        shared_expert_stream = create_stream(
            "glm53_shared_expert", infer_config.model_config.exe_mode) \
            if infer_config.model_config.custom_params.get(
                "enable_multi_streams", True) else None
        kwargs = {"zero_rope": SharedZeroRope(),
                  "shared_expert_stream": shared_expert_stream}

        self.model = Glm53Model(config, infer_config, comm_manager, prefix, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=config.vocab_size,
            bias=False, tp_size=self.lmhead_tp_size,
            tp_rank=(comm_manager.get_rank("lmhead_tp_group")
                     if self.lmhead_tp_size > 1 else 0),
            quant_config=None, prefix="lm_head")

    @classmethod
    def update_model_cfg(cls, hf_config, infer_config: InferenceConfig):
        """Validate the checkpoint's quant format; None means plain BF16."""
        quant_config = getattr(hf_config, "quant_config", None)
        if quant_config is None:
            return
        # (Linear, MoEGMM) modes; hif8 is the three-tier Hybrid HiF8-MXFP8-MXFP4
        # split. The shared experts' MXFP8 group is not visible here, it is
        # resolved per module by prefix at build time.
        modes = (quant_config.mm_quant_mode, quant_config.gmm_quant_mode)
        if modes != ("w8a8hifloat8", "w4a8mxfloat4"):
            raise NotImplementedError(
                "glm_5_3 supports BF16 and Hybrid HiF8-MXFP8-MXFP4, got "
                f"mm_quant_mode={modes[0]}, gmm_quant_mode={modes[1]}")

    def check_model_settings(self):
        parallel_config = self.infer_config.parallel_config
        model_config = self.infer_config.model_config
        if model_config.next_n > 0:
            raise NotImplementedError("glm_5_3 does not support MTP yet.")
        if parallel_config.cp_size > 1:
            raise NotImplementedError("glm_5_3 does not support CP.")
        for name in ("attn_tp_size", "moe_tp_size", "dense_tp_size",
                     "o_proj_tp_size", "shared_tp_size"):
            if getattr(parallel_config, name, 1) != 1:
                raise NotImplementedError(f"glm_5_3 requires {name} == 1 "
                                          "(attention/dense run DP; parallelism "
                                          "comes from MoE EP)")

    def init_parallel_comm_group(self):
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size)
        if self.embed_tp_size > 1:
            self.comm_manager.register_group(
                name="embed_tp_group",
                group_num=self.world_size // self.embed_tp_size,
                group_size=self.embed_tp_size, return_name=True)
        if self.lmhead_tp_size > 1:
            self.comm_manager.register_group(
                name="lmhead_tp_group",
                group_num=self.world_size // self.lmhead_tp_size,
                group_size=self.lmhead_tp_size, return_name=True)
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True)
            platform = self.infer_config.model_config.platform_version
            platform = platform.value if hasattr(platform, "value") else str(platform)
            self.comm_manager.register_group(
                name="moe_ep_group_mc2",
                group_num=self.world_size // self.moe_ep_size,
                group_size=self.moe_ep_size,
                group_stride=self.world_size // self.moe_ep_size,
                return_name=True,
                allow_physical_reuse=False,
                hccl_buffer_size=calc_moe_hccl_buffer_size(
                    self.infer_config, self.config,
                    is_full_mesh_v2=platform != "950"),
                group_type=3 if platform == "950" else None)

    def get_cache_info(self) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_infos.append(LayerCacheInfo(
                layer_idx=layer_idx,
                caches=list(layer.self_attn.cache_entries)))
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True)

    def process_weights_after_loading(self):
        from .kda import Glm53KdaAttention as _Kda
        for module in self.modules():
            if isinstance(module, Glm53SparseAttention):
                module.prepare_absorbed_weights()
            elif isinstance(module, _Kda):
                module.build_conv_weight()
        enable_weight_nz = self.infer_config.model_config.enable_weight_nz
        from module.quantization import QuantizeMethodBase
        scales_dtype = {}
        for module_name, module in self.named_modules():
            if "kv_b_proj" in module_name:
                continue  # weight consumed by the absorbed split above
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module, is_nz=enable_weight_nz, scales_dtype=scales_dtype)

    # ---- forward ---------------------------------------------------------
    def forward_lm_head(self, hidden_states, forward_metadata):
        # hidden_states: [T, D]; gather the last token of each request at
        # prefill (packed TND), pass through at decode ([B, D]).
        if forward_metadata.is_prefill:
            seq_index = forward_metadata.actual_seq_lengths_cu_q - 1
            hidden_states = torch.index_select(hidden_states, 0, seq_index)
        logits = self.lm_head(hidden_states)          # [B, V] (lmhead_tp == 1)
        return logits.view(logits.shape[0], 1, -1).float()

    def forward(self, input_ids=None, position_ids=None, forward_metadata=None,
                cur_topk_list=None, **kwargs):
        if forward_metadata is None:
            raise ValueError("glm_5_3 framework path requires forward_metadata")
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            cur_topk_list=cur_topk_list,
        )
        logits = self.forward_lm_head(hidden_states, forward_metadata)
        return logits, hidden_states

    # ---- weight loading --------------------------------------------------
    def _normalize_name(self, name: str) -> Optional[str]:
        """Map checkpoint names onto this module tree; None -> skip."""
        if name.startswith("model.visual."):
            return None  # vision tower unsupported (text-only recipe)
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        # KDA depth-wise conv weights are registered as bare parameters
        # (self_attn.q_conv1d), while the checkpoint names them like nn.Conv1d
        # modules (self_attn.q_conv1d.weight).
        for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
            if name.endswith(f".{conv}.weight"):
                name = name[: -len(".weight")]
                break
        # The checkpoint spells the KDA decay parameter "A_log"; the module
        # registers it snake_case as self_attn.a_log.
        if name.endswith(".A_log"):
            name = name[: -len("A_log")] + "a_log"

        # MTP draft layers are not built (next_n == 0), so skip their weights.
        cfg = self.config
        mtp_prefixes = tuple(
            f"model.layers.{cfg.num_hidden_layers + i}."
            for i in range(cfg.num_nextn_predict_layers))
        return None if name.startswith(mtp_prefixes) else name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        unmatched: list = []

        for raw_name, loaded_weight in weights:
            name = self._normalize_name(raw_name)
            if name is None:
                continue

            target = name
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue  # per-expert weights handled below
                candidate = name.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                param.weight_loader(param, loaded_weight, shard_id)
                target = candidate
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    candidate = name.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    param.weight_loader(param, loaded_weight, candidate,
                                        shard_id=shard_id, expert_id=expert_id)
                    target = candidate
                    break
                else:
                    if name not in params_dict:
                        unmatched.append(raw_name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(target)

        missing = sorted(n for n in set(params_dict) - loaded_params
                         if "smooth_scale" not in n)
        if missing:
            logger.warning("glm_5_3: %d parameters were NOT initialized from the "
                           "checkpoint, e.g. %s", len(missing), missing[:12])
        if unmatched:
            logger.info("glm_5_3: %d checkpoint tensors had no target parameter, "
                        "e.g. %s", len(unmatched), unmatched[:12])
        return loaded_params
