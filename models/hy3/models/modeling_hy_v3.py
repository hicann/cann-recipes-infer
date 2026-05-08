# coding=utf-8
# Adapted from transformers/models/hy_v3/modeling_hy_v3.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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

"""PyTorch HYV3 model implementation adapted for CANN NPU inference framework."""

import logging
import math
import os
import re
from typing import Generator, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch_npu
import torch.distributed as dist

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import superkernel_scope, npu_stream_switch
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.core.config import InferenceConfig, CommManager
from module.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    VocabParallelEmbedding,
)

from .configuration_hy_v3 import HYV3Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class HYV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, *args):
        """RMSNorm using NPU fused kernel, with optional residual fusion.

        forward(hidden_states) -> rms_norm(hidden_states)
        forward(hidden_states, None) -> (rms_norm(hidden_states), hidden_states) for first layer
        forward(hidden_states, residual) -> (residual + rms_norm, rms_norm) fused via npu_add_rms_norm
        """
        if len(args) == 0:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        elif len(args) == 1 and args[0] is None:
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1:
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(
                residual, hidden_states, self.weight, self.variance_epsilon
            )
            return (y, x)
        else:
            raise NotImplementedError(
                f"insupportable HYV3RMSNorm for input_args len as (include hid): {len(args) + 1}"
            )


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------

class HYV3RotaryEmbedding(nn.Module):
    def __init__(self, config: HYV3Config, max_position_embeddings=2048, device=None):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = config.rope_parameters.get("rope_theta", config.default_theta)

        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def forward(self, x, position_ids, max_seq_len=None):
        # max_seq_len is always provided by HYV3Model caller
        if max_seq_len is not None and max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        # position_ids is 1D packed [total_tokens] or [batch_size]
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        if x.dim() == 3:
            # 2D hidden states: (batch, seq_len, hidden)
            batch_size, seq_len, _ = x.shape
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            # 1D hidden states: (batch, hidden) — decode
            batch_size = x.shape[0]
            cos = cos.view(batch_size, 1, 1, self.dim)
            sin = sin.view(batch_size, 1, 1, self.dim)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HYV3Attention(nn.Module):
    """Multi-headed attention with GQA, QK RMSNorm, and RoPE. Supports TP via QKVParallelLinear."""

    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.comm_manager = comm_manager

        self.num_heads_per_rank = max(self.num_heads // self.attn_tp_size, 1)
        self.num_kv_heads_per_rank = max(self.num_kv_heads // self.attn_tp_size, 1)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scale_fa = 1.0 / math.sqrt(self.head_dim)
        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"
        self._tng_ops = None
        if self.enable_gegraph:
            import torchair as tng
            self._tng_ops = tng.ops

        # QKV merged projection with TP
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0,
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False,
        )

        self.q_norm = HYV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HYV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # O projection with TP
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0,
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.o_proj",
        )

        # KV cache placeholders — allocated by framework via init_cache
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.cache_unit = (self.num_kv_heads_per_rank * self.head_dim,)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        kv_len_for_scatter: torch.Tensor = None,
        seq_lengths_kv=None,
    ):
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len
        attention_mask = forward_metadata.attention_mask

        bsz, q_len, h = hidden_states.size()

        # All-gather hidden states within TP group when DP > 1
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            h_dtype = hidden_states.dtype
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            new_hidden_states = torch.empty(
                [bsz * self.attn_tp_size, q_len, h], dtype=h_dtype, device="npu"
            )
            dist.all_gather_into_tensor(new_hidden_states, hidden_states, group=attn_tp_group)
            hidden_states = new_hidden_states
            bsz = bsz * self.attn_tp_size

        # Merged QKV projection
        qkv = self.merged_qkv_proj(hidden_states)

        # Split Q, K, V
        query_states, key_states, value_states = qkv.split(
            (self.num_heads_per_rank * self.head_dim,
             self.num_kv_heads_per_rank * self.head_dim,
             self.num_kv_heads_per_rank * self.head_dim),
            dim=2
        )

        # Reshape to per-head 4D for QK norm
        query_shape = (bsz, q_len, self.num_heads_per_rank, self.head_dim)
        key_value_shape = (bsz, q_len, self.num_kv_heads_per_rank, self.head_dim)

        query_states = self.q_norm(query_states.contiguous().view(query_shape))
        key_states = self.k_norm(key_states.contiguous().view(key_value_shape))
        value_states = value_states.view(key_value_shape)

        # RoPE via NPU fusion operator (supports 4D [B,S,N,D] with layout='BSH')
        cos, sin = cos_sin
        query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, layout='BSH'
        )

        # Flatten back to BSH
        query_states = query_states.view(bsz, q_len, -1)
        key_states = key_states.view(bsz, q_len, -1)
        value_states = value_states.view(bsz, q_len, -1)

        # KV Cache write via scatter_update.
        # kv_len_for_scatter is pre-computed once in HYV3Model.forward and shared
        # across all layers to avoid per-layer tensor creation overhead.
        if kv_len_for_scatter is None:
            if is_prefill:
                kv_len_for_scatter = torch.zeros(bsz, dtype=torch.int32, device=hidden_states.device)
            else:
                kv_len_for_scatter = kv_len.to(torch.int32)
                if self.attn_tp_size > 1 and self.attn_dp_size > 1:
                    kv_len_for_scatter = kv_len_for_scatter.repeat(self.attn_tp_size)

        torch_npu.scatter_update_(self.k_cache, kv_len_for_scatter, key_states, -2)
        torch_npu.scatter_update_(self.v_cache, kv_len_for_scatter, value_states, -2)

        # Flash Attention
        # Prefill always uses torch.ops.npu (eager). Decode in GE graph mode uses
        # torchair.ops (accepts tensor actual_seq_lengths_kv, no graph break);
        # decode in other modes uses torch.ops.npu with List[int] conversion.
        fa_ops = torch.ops.npu
        if not is_prefill and self._tng_ops is not None:
            fa_ops = self._tng_ops

        if not is_prefill:
            # Decode — seq_lengths_kv is pre-computed once in HYV3Model.forward
            # (already expanded for TP+DP) and shared across all layers to avoid
            # per-layer .repeat() tensor allocation overhead.
            atten_mask = None
            # In the decode path seq_lengths_kv is always pre-computed in
            # HYV3Model.forward (from forward_metadata.actual_seq_lengths_kv).
            # The fallback to kv_len only exists for edge-case callers that
            # invoke the attention layer directly with seq_lengths_kv=None.
            actual_seq_lengths_kv = seq_lengths_kv
            attn_output, _ = fa_ops.npu_fused_infer_attention_score(
                query_states, self.k_cache, self.v_cache,
                num_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="BSH",
                sparse_mode=0,
                atten_mask=atten_mask,
                scale=self.scale_fa,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                antiquant_mode=0, antiquant_scale=None,
            )
        else:
            # Prefill — sparse_mode=3 (causal)
            attn_output, _ = fa_ops.npu_fused_infer_attention_score(
                query_states, key_states, value_states,
                num_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="BSH",
                atten_mask=attention_mask,
                sparse_mode=3,
                scale=self.scale_fa,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads_per_rank * self.head_dim)

        # O projection with TP: RowParallelLinear handles all_reduce internally
        attn_output = self.o_proj(attn_output)

        # Reduce-scatter when both TP and DP are active
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            new_output = torch.empty(
                [bsz // self.attn_tp_size, q_len, h],
                dtype=attn_output.dtype, device="npu"
            )
            dist.reduce_scatter_tensor(
                new_output, attn_output,
                group=self.comm_manager.get_group("attn_tp_group")
            )
            attn_output = new_output

        return attn_output


# ---------------------------------------------------------------------------
# Dense MLP (for layer 0 and shared expert) — parallelized
# ---------------------------------------------------------------------------

class HYV3MLP(nn.Module):
    """Dense feed-forward network with SiLU activation, parallelized via TP."""

    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig = None,
        comm_manager: CommManager = None,
        intermediate_size: Optional[int] = None,
        dense_tp_size: int = 1,
        dense_tp_group=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.dense_tp_size = dense_tp_size
        self.dense_tp_group = dense_tp_group
        self.comm_manager = comm_manager

        if dense_tp_size > 1:
            if comm_manager is not None:
                tp_rank = comm_manager.get_rank("dense_tp_group")
            else:
                tp_rank = dist.get_rank(dense_tp_group)
        else:
            tp_rank = 0

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size,
            bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        merged = torch.cat([gate, up], dim=-1)
        act = torch_npu.npu_swiglu(merged)
        down = self.down_proj(act)
        if self.dense_tp_size > 1:
            dist.all_reduce(down, group=self.dense_tp_group)
        return down


# ---------------------------------------------------------------------------
# MoE Router
# ---------------------------------------------------------------------------

class HYV3TopKRouter(nn.Module):
    """Sigmoid-based Top-K router for MoE.

    Uses npu_moe_gating_top_k (norm_type=1 sigmoid) for the decode path
    to fuse sigmoid + topk + weight normalization into a single NPU kernel.
    """

    def __init__(self, config: HYV3Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.router_scaling_factor = config.router_scaling_factor
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.gate(hidden_states.float())

        top_k_weights, top_k_index, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            k=self.top_k,
            bias=e_score_correction_bias.float(),
            norm_type=1,  # sigmoid
            routed_scaling_factor=self.router_scaling_factor,
            eps=float(1e-20),
        )
        top_k_weights = top_k_weights.to(hidden_states.dtype)

        row_idx = torch.arange(
            0, top_k_index.shape[0] * self.top_k,
            dtype=torch.int32, device=top_k_index.device
        ).view(top_k_index.shape[0], self.top_k)

        return router_logits, top_k_weights, top_k_index, row_idx


# ---------------------------------------------------------------------------
# MoE Experts — EP-aware packed tensors
# ---------------------------------------------------------------------------

class HYV3Experts(nn.Module):
    """Collection of expert weights stored as packed 3D tensors, EP-aware.

    With EP: each rank stores experts_per_rank experts instead of all experts.
    """

    def __init__(self, config: HYV3Config, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.num_experts = config.num_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.experts_per_rank = max(self.num_experts // self.ep_size, 1)
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # Packed gate and up projections: [experts_per_rank, 2*intermediate, hidden]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.experts_per_rank, 2 * self.intermediate_dim, self.hidden_dim)
        )
        # Packed down projections: [experts_per_rank, hidden, intermediate]
        self.down_proj = nn.Parameter(
            torch.empty(self.experts_per_rank, self.hidden_dim, self.intermediate_dim)
        )

    def forward_gmm(self, x: torch.Tensor, expert_tokens: torch.Tensor, group_list_type: int = 1):
        """Compute experts via npu_grouped_matmul (graph-compatible, no per-expert loop)."""
        w1 = self.gate_up_proj.transpose(1, 2)  # [E, 2*I, H] -> [E, H, 2*I]
        mm1 = torch_npu.npu_grouped_matmul(
            [x], [w1],
            group_list=expert_tokens, group_type=0,
            group_list_type=group_list_type, split_item=2,
        )[0]
        mm1 = torch_npu.npu_swiglu(mm1)
        w2 = self.down_proj.transpose(1, 2)  # [E, H, I] -> [E, I, H]
        out = torch_npu.npu_grouped_matmul(
            [mm1], [w2],
            group_list=expert_tokens, group_type=0,
            group_list_type=group_list_type, split_item=2,
        )[0]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_hit in expert_hit:
            global_expert_id = expert_idx_hit[0].item()
            if global_expert_id >= self.num_experts:
                continue
            local_expert_id = global_expert_id - self.ep_rank * self.experts_per_rank
            if local_expert_id < 0 or local_expert_id >= self.experts_per_rank:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[global_expert_id])
            current_state = hidden_states[token_idx]
            merged = F.linear(current_state, self.gate_up_proj[local_expert_id])
            current_hidden_states = torch_npu.npu_swiglu(merged)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[local_expert_id])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward_ordered(
        self,
        hidden_states_ordered: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Compute experts on already-routed and ordered tokens (for EP)."""
        outputs = []
        start = 0
        for num_tokens in tokens_per_expert:
            num_tokens = num_tokens.item()
            if num_tokens == 0:
                outputs.append(torch.empty(0, self.hidden_dim,
                                           dtype=hidden_states_ordered.dtype,
                                           device=hidden_states_ordered.device))
                continue
            end = start + num_tokens
            expert_hidden = hidden_states_ordered[start:end]
            expert_idx = len(outputs)
            if expert_idx < self.experts_per_rank:
                merged = F.linear(expert_hidden, self.gate_up_proj[expert_idx])
                expert_out = F.linear(torch_npu.npu_swiglu(merged), self.down_proj[expert_idx])
                outputs.append(expert_out)
            else:
                outputs.append(torch.zeros(num_tokens, self.hidden_dim,
                                           dtype=hidden_states_ordered.dtype,
                                           device=hidden_states_ordered.device))
            start = end
        return torch.cat(outputs, dim=0) if outputs else torch.empty(
            0, self.hidden_dim, dtype=hidden_states_ordered.dtype, device=hidden_states_ordered.device
        )


# ---------------------------------------------------------------------------
# MoE Block (Router + Experts + Shared Expert) — with EP routing
# ---------------------------------------------------------------------------

class HYV3MoE(nn.Module):
    """MoE block with EP routing: replicated router, EP-split experts, parallelized shared expert."""

    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = "",
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.router_scaling_factor = config.router_scaling_factor
        self.enable_moe_fp32_combine = config.enable_moe_fp32_combine

        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.comm_manager = comm_manager
        self.experts_per_rank = max(self.num_experts // self.moe_ep_size, 1)

        ep_rank = comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", False
        )

        # Router: replicated (all ranks have full router)
        self.router = HYV3TopKRouter(config)

        # Expert bias: replicated
        self.register_buffer("expert_bias", torch.zeros(config.num_local_experts))

        # EP-split experts
        self.experts = HYV3Experts(config, ep_size=self.moe_ep_size, ep_rank=ep_rank)

        # Shared expert: parallelized via dense_tp
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        dense_tp_group = comm_manager.get_group("dense_tp_group") if self.dense_tp_size > 1 else None
        self.shared_mlp = HYV3MLP(
            config, intermediate_size=shared_intermediate,
            dense_tp_size=self.dense_tp_size, dense_tp_group=dense_tp_group,
            prefix=f"{prefix}.shared_mlp",
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool = False) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router: produces global topk indices
        _, top_k_weights, top_k_index, row_idx = self.router(hidden_states_flat, self.expert_bias)
        top_k_index = top_k_index.to(torch.int32)

        if self.moe_ep_size > 1:
            if self.exe_mode == "ge_graph" and not is_prefill:
                routed_output = self._moe_ep_mc2_decode(
                    hidden_states_flat, top_k_index, top_k_weights
                )
            else:
                routed_output = self._moe_ep_manual(
                    hidden_states, hidden_states_flat, top_k_index, top_k_weights, row_idx
                )
        else:
            routed_output = self.experts(hidden_states_flat, top_k_index, top_k_weights)

        # Shared expert — run on secondary stream to overlap with routed expert path
        enable_multi_streams = self.enable_multi_streams and not is_prefill
        with npu_stream_switch(enable_multi_streams, "11"):
            shared_output = self.shared_mlp(hidden_states_flat)

        if self.enable_moe_fp32_combine:
            hidden_states_out = (
                routed_output.float() + shared_output.float()
            ).to(hidden_states_flat.dtype)
        else:
            hidden_states_out = routed_output + shared_output

        return hidden_states_out.reshape(batch_size, seq_len, hidden_dim)

    def _moe_ep_manual(self, hidden_states, hidden_states_flat, topk_ids, topk_weight, row_idx):
        """Manual EP routing via npu_moe_init_routing_v2 + all_to_all + re_routing + finalize."""
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")

        routing_args = {
            "expert_idx": topk_ids,
            "active_num": topk_ids.shape[0] * topk_ids.shape[1],
            "expert_num": self.num_experts,
            "expert_tokens_num_type": 1,
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": -1,
        }
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states_flat, **routing_args
        )

        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)

        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        all_tokens_sum = sum(output_splits)

        gathered_tokens = expanded_x.new_empty(all_tokens_sum, expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)

        hidden_states_ordered, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(
                gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1)
            )

        hidden_states_ordered = self.experts.forward_ordered(
            hidden_states_ordered, tokens_per_local_expert
        )

        new_x = torch.index_select(
            hidden_states_ordered, 0,
            gathered_ids_unsort.float().argsort().int()
        )

        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )

        return hidden_states

    def _moe_ep_mc2_decode(self, hidden_states_flat, topk_ids, topk_weight):
        """MC2 dispatch/combine for decode — graph-compatible, no data-dependent splits."""
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group")
        ep_rank = self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0

        dispatch_kwargs = {
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "x_active_mask": None,
            "group_ep": moe_ep_group_name,
            "group_tp": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": ep_rank,
            "tp_world_size": 1,
            "tp_rank_id": 0,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "quant_mode": 0,
        }
        combine_kwargs = {
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "x_active_mask": None,
            "expand_scales": None,
            "group_ep": moe_ep_group_name,
            "group_tp": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": ep_rank,
            "tp_world_size": 1,
            "tp_rank_id": 0,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "comm_quant_mode": 0,
        }

        dispatch_result = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states_flat,
            expert_ids=topk_ids,
            **dispatch_kwargs,
        )
        expand_x = dispatch_result[0]
        expand_idx = dispatch_result[2]
        expert_token_num = dispatch_result[3]
        ep_recv_counts = dispatch_result[4]
        tp_recv_counts = dispatch_result[5] if len(dispatch_result) > 5 else None

        expert_output = self.experts.forward_gmm(expand_x, expert_token_num)

        output = torch_npu.npu_moe_distribute_combine_v2(
            expert_output, topk_ids, expand_idx,
            ep_recv_counts, topk_weight,
            tp_send_counts=tp_recv_counts,
            **combine_kwargs,
        )
        return output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class HYV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = HYV3Attention(
            config=config, infer_config=infer_config, comm_manager=comm_manager,
            layer_idx=layer_idx, prefix=f"{prefix}.self_attn",
        )

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = HYV3MoE(
                config, infer_config=infer_config, comm_manager=comm_manager,
                prefix=f"{prefix}.mlp",
            )
        else:
            dense_tp_size = infer_config.parallel_config.dense_tp_size
            dense_tp_group = comm_manager.get_group("dense_tp_group") if dense_tp_size > 1 else None
            self.mlp = HYV3MLP(
                config, infer_config=infer_config, comm_manager=comm_manager,
                intermediate_size=config.intermediate_size,
                dense_tp_size=dense_tp_size, dense_tp_group=dense_tp_group,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        past_residual: Optional[torch.Tensor] = None,
        kv_len_for_scatter: torch.Tensor = None,
        seq_lengths_kv=None,
        **kwargs,
    ):
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            kv_len_for_scatter=kv_len_for_scatter,
            seq_lengths_kv=seq_lengths_kv,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if isinstance(self.mlp, HYV3MoE):
            is_prefill = forward_metadata.is_prefill if forward_metadata else False
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
        else:
            hidden_states = self.mlp(hidden_states)

        outputs = (residual, hidden_states)
        return outputs


# ---------------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------------

class HYV3Model(nn.Module):
    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.max_position_embeddings = 131072
        if infer_config is not None:
            self.max_position_embeddings = infer_config.data_config.input_truncated_len + \
                infer_config.scheduler_config.max_new_tokens

        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.comm_manager = comm_manager
        self.enable_superkernel = infer_config.model_config.custom_params.get(
            "enable_superkernel", False
        )
        self.enable_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", False
        )

        # VocabParallelEmbedding: adjust padding_idx for per-rank vocab partition
        embed_tp_rank = comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0
        vocab_per_rank = config.vocab_size // self.embed_tp_size
        pad_start = embed_tp_rank * vocab_per_rank
        pad_end = pad_start + vocab_per_rank
        if pad_start <= self.padding_idx < pad_end:
            per_rank_padding_idx = self.padding_idx - pad_start
        else:
            per_rank_padding_idx = None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            per_rank_padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=embed_tp_rank,
        )

        self.layers = nn.ModuleList(
            [HYV3DecoderLayer(config, infer_config, comm_manager, layer_idx,
                              prefix=f"{prefix}.layers.{layer_idx}")
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV3RotaryEmbedding(
            config=config, max_position_embeddings=self.max_position_embeddings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len

        # Reshape 1D packed input to 2D BSH format
        if is_prefill:
            # Prefill: 1D packed (total_tokens,) → 2D (batch_size, seq_len)
            total_tokens = input_ids.shape[0]
            actual_seq_lengths_q = forward_metadata.actual_seq_lengths_q
            batch_size = actual_seq_lengths_q.shape[0]
            seq_len = total_tokens // batch_size
            input_ids = input_ids.view(batch_size, seq_len)
            position_ids = position_ids.view(batch_size, seq_len)
        else:
            # Decode: 1D (batch_size,) → 2D (batch_size, 1)
            batch_size = input_ids.shape[0]
            input_ids = input_ids.view(batch_size, 1)
            position_ids = position_ids.view(batch_size, 1)
            seq_len = 1

        batch_size, seq_len = input_ids.shape

        # Pre-compute kv_len scatter index once, shared across all layers.
        # This avoids per-layer tensor creation (torch.zeros / .to(int32) / .repeat)
        # which adds up to significant overhead across ~80 layers.
        # All shapes are compile-time constants, safe for ge_graph.
        tp_dp_active = self.attn_tp_size > 1 and self.attn_dp_size > 1
        if is_prefill:
            scatter_bsz = batch_size * self.attn_tp_size if tp_dp_active else batch_size
            kv_len_for_scatter = torch.zeros(scatter_bsz, dtype=torch.int32, device=input_ids.device)
            seq_lengths_kv = None
        else:
            kv_len_for_scatter = kv_len.to(torch.int32)
            # Pre-compute expanded actual_seq_lengths_kv once per forward (shared
            # across all layers). Within a TP group all ranks share the same
            # prompts, so .repeat() is equivalent to all_gather (no comm needed).
            # BSH FA requires actual_seq_lengths_kv to match query/cache batch dim.
            seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
            if seq_lengths_kv is not None and tp_dp_active:
                if isinstance(seq_lengths_kv, torch.Tensor):
                    seq_lengths_kv = seq_lengths_kv.unsqueeze(0).expand(self.attn_tp_size, -1).reshape(-1)
                elif isinstance(seq_lengths_kv, list):
                    seq_lengths_kv = seq_lengths_kv * self.attn_tp_size
            if tp_dp_active:
                kv_len_for_scatter = kv_len_for_scatter.repeat(self.attn_tp_size)

        # Embedding lookup with TP
        if self.embed_tp_size > 1:
            embed_tp_rank = self.comm_manager.get_rank("embed_tp_group")
            new_input_ids = input_ids - embed_tp_rank * (self.vocab_size // self.embed_tp_size)
            mask = (new_input_ids >= 0) & (new_input_ids < (self.vocab_size // self.embed_tp_size))
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # RoPE: position_ids is 2D (batch, seq_len) after reshaping
        flat_position_ids = position_ids.view(-1)
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            repeated_hidden = hidden_states.repeat(self.attn_tp_size, 1, 1)
            repeated_pos = flat_position_ids.repeat(self.attn_tp_size)
            cos_sin = self.rotary_emb(repeated_hidden, repeated_pos, self.max_position_embeddings)
        else:
            cos_sin = self.rotary_emb(hidden_states, flat_position_ids, self.max_position_embeddings)

        residual = None
        sk_option = ("stream-fusion=1" if (self.enable_superkernel
                     and self.enable_multi_streams and not is_prefill) else "")
        with superkernel_scope(
            self.enable_superkernel and not is_prefill,
            "decode_layer", sk_option
        ):
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    cos_sin=cos_sin,
                    forward_metadata=forward_metadata,
                    past_residual=residual,
                    kv_len_for_scatter=kv_len_for_scatter,
                    seq_lengths_kv=seq_lengths_kv,
                    **kwargs,
                )
                residual, hidden_states = layer_outputs

        # Final Norm with fused residual add (npu_add_rms_norm)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# Causal LM (Model + LM Head)
# ---------------------------------------------------------------------------

class HYV3ForCausalLM(nn.Module):
    """HYV3 model with a language modeling head for causal generation.

    Supports TP on attention (attn_tp), dense FFN (dense_tp), embedding (embed_tp),
    LM head (lmhead_tp), and EP on MoE experts (moe_ep).
    """

    _ignore_weights_patterns = ["model.layers.80."]

    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert torch_dtype from string to actual dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
        }
        if isinstance(config.torch_dtype, str):
            config.torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.enable_lm_head_fp32 = getattr(config, "enable_lm_head_fp32", True)
        self.vocab_size = config.vocab_size

        # Parse parallel settings
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size

        self.model = HYV3Model(config, infer_config, comm_manager, prefix="model")

        # LM Head with ColumnParallelLinear
        self.vocab_size_per_rank = self.vocab_size // self.lmhead_tp_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
            prefix="lm_head",
        )

    def init_cache(self, device):
        """Allocate BSH-format KV cache tensors on each attention layer.

        Handles TP+DP batch expansion: when both attn_tp and attn_dp are active,
        the cache batch dimension is enlarged by attn_tp_size to accommodate
        all_gather'd hidden states.
        """
        cache_seq_len = self.infer_config.data_config.input_truncated_len + \
            self.infer_config.scheduler_config.max_new_tokens
        batch_size_per_dp_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        dtype = self.config.torch_dtype

        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            cache_batch_size = batch_size_per_dp_rank * self.attn_tp_size
        else:
            cache_batch_size = batch_size_per_dp_rank

        for layer in self.model.layers:
            attn = layer.self_attn
            cache_shape = (cache_batch_size, cache_seq_len, *attn.cache_unit)
            attn.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
            attn.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

    def process_weights_after_loading(self):
        for _, module in self.named_modules():
            qm = getattr(module, "quant_method", None)
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                qm.process_weights_after_loading(module)

    def load_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None]) -> Set[str]:
        """Load weights from the checkpoint iterator.

        Handles:
        - EP filtering: only load experts belonging to current ep_rank
        - Packed expert tensors: checkpoint gate_proj + up_proj -> gate_up_proj
        - TP-split weights via ParallelLinear weight_loader
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params: Set[str] = set()

        experts_per_rank = self.config.num_experts // self.moe_ep_size
        ep_rank = self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        ep_start = ep_rank * experts_per_rank
        ep_end = ep_start + experts_per_rank

        expert_gate_up_pending: dict = {}

        for name, loaded_weight in weights:
            if any(pattern in name for pattern in self._ignore_weights_patterns):
                continue

            if name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
                loaded_params.add(name)
                continue

            # Handle per-expert weights with EP filtering
            expert_match = re.match(
                r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight)",
                name
            )

            if expert_match:
                prefix = expert_match.group(1)
                expert_id = int(expert_match.group(2))
                proj_type = expert_match.group(3)

                if expert_id < ep_start or expert_id >= ep_end:
                    continue

                local_expert_id = expert_id - ep_start

                if proj_type == "down_proj":
                    param_name = f"{prefix}.down_proj"
                    if param_name in params_dict:
                        params_dict[param_name].data[local_expert_id].copy_(loaded_weight)
                        loaded_params.add(param_name)
                else:
                    param_name = f"{prefix}.gate_up_proj"
                    if param_name not in params_dict:
                        continue

                    if param_name not in expert_gate_up_pending:
                        expert_gate_up_pending[param_name] = {}
                    if local_expert_id not in expert_gate_up_pending[param_name]:
                        expert_gate_up_pending[param_name][local_expert_id] = {}

                    expert_gate_up_pending[param_name][local_expert_id][proj_type] = loaded_weight
                    pending = expert_gate_up_pending[param_name][local_expert_id]

                    if "gate_proj" in pending and "up_proj" in pending:
                        gate = pending.pop("gate_proj")
                        up = pending.pop("up_proj")
                        intermediate = up.shape[0]
                        params_dict[param_name].data[local_expert_id, :intermediate, :].copy_(gate)
                        params_dict[param_name].data[local_expert_id, intermediate:, :].copy_(up)
                        if not pending:
                            del expert_gate_up_pending[param_name][local_expert_id]

                    loaded_params.add(param_name)
                continue

            # Handle QKV projection weights
            qkv_match = re.match(
                r"(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)\.(weight)",
                name
            )
            if qkv_match:
                prefix = qkv_match.group(1)
                shard_id = qkv_match.group(2)[0]
                param_name = f"{prefix}.merged_qkv_proj.weight"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, loaded_shard_id=shard_id)
                    loaded_params.add(param_name)
                continue

            # Standard parameter matching
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            **kwargs,
        )

        is_prefill = forward_metadata.is_prefill
        batch_size, seq_len_h, h = hidden_states.shape

        # For Prefill: gather the last token per sequence
        if seq_len_h > 1:
            position_ids_2d = position_ids.view(batch_size, -1)
            gather_index = torch.max(position_ids_2d, dim=-1)[1]
            gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(
                1, 1, hidden_states.shape[-1]
            )
            hidden_states = torch.gather(hidden_states, 1, gather_index)
            hidden_states = hidden_states.view(batch_size, 1, h)

        # LM Head: ColumnParallelLinear produces partial logits
        logits = self.lm_head(hidden_states)

        # Gather logits across lmhead_tp_group to get full vocab
        if self.lmhead_tp_size > 1:
            logits = logits.float()
            lmhead_tp_group = self.comm_manager.get_group("lmhead_tp_group")
            bs, q_len, _ = logits.shape
            gathered_list = [
                torch.empty_like(logits) for _ in range(self.lmhead_tp_size)
            ]
            dist.all_gather(gathered_list, logits, group=lmhead_tp_group)
            logits = torch.cat(gathered_list, dim=-1)

        if self.enable_lm_head_fp32 and logits.dtype != torch.float32:
            logits = logits.float()
        return logits
