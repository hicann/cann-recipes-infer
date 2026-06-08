# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Step-3.7-Flash text MoE model for NPU unified-flow inference."""

import os
import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import torch_npu
import torchair

from module.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
    VocabParallelEmbedding,
)
from module.fuse_moe_gmm import FusedMoEGMM
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_step3p7 import Step3p7TextConfig

torchair.patch_for_hcom()


def _resolve_dtype(dtype, default=torch.bfloat16):
    """config.torch_dtype may be a str ('bfloat16') or a torch.dtype."""
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype).replace("torch.", ""), default)


FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
# Cache-info attn_type keys (must be distinct so the framework allocates a
# separate KV cache pool / block_table / slot_mapping per attention family).
ATTN_TYPE_FULL = "FullAttention"
ATTN_TYPE_SLIDING = "SlidingWindow"

# `<im_patch>` placeholder token id (config.image_token_id). The text-only
# config (Step3p7TextConfig) intentionally drops the multimodal-only
# image_token_id field, so the vision-injection entry uses this default and lets
# the caller override it via the `image_token_id` forward kwarg. This constant is
# ONLY read on the opt-in vision path; the pure-text path never references it.
DEFAULT_IMAGE_TOKEN_ID = 128001


class Step3p7RMSNorm(nn.Module):
    """RMSNorm with the Step convention: normed * (weight + 1).

    The Step checkpoints store the RMSNorm gain centered at 0 (so the effective
    scale is ``weight + 1``). We pre-add 1.0 to the loaded weight in
    ``load_weights`` so the fused ``npu_rms_norm`` (which multiplies by the raw
    gain) reproduces the original ``weight + 1`` behaviour. ``self.plus_one``
    is True when this normalization expects the +1 offset.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, *args):
        # Mirrors qwen3_moe RMSNorm calling convention:
        #   norm(x)            -> normed tensor (e.g. final norm)
        #   norm(x, None)      -> (normed, x)            (first decoder layer)
        #   norm(x, residual)  -> fused add+norm -> (normed, new_residual)
        if len(args) == 0:
            return torch_npu.npu_rms_norm(hidden_states, self.weight,
                                          self.variance_epsilon)[0]
        residual = args[0]
        if residual is None:
            normed = torch_npu.npu_rms_norm(hidden_states, self.weight,
                                            self.variance_epsilon)[0]
            return normed, hidden_states
        y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states,
                                             self.weight, self.variance_epsilon)
        return y, x


class Step3p7PerHeadRMSNorm(nn.Module):
    """Per-head RMSNorm over head_dim (q_norm / k_norm), Step weight+1 offset.

    Operates on tensors shaped ``[..., head_dim]``. The +1 offset is folded into
    the stored weight at load time (see ``load_weights``).
    """

    def __init__(self, head_dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]


def _compute_inv_freq(head_dim, partial_rotary_factor, theta, rope_scaling):
    """Compute RoPE inverse frequencies for one attention family.

    Supports the llama3 rope scaling used by Step-3.7-Flash (only applied to
    full-attention layers via ``yarn_only_types``). When ``rope_scaling`` is
    None the default RoPE frequencies are returned.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
    )
    if rope_scaling is None:
        return inv_freq, rotary_dim

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
    if rope_type != "llama3":
        # Only llama3 scaling is present in Step-3.7-Flash configs.
        return inv_freq, rotary_dim

    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama, rotary_dim


class Step3p7RotaryEmbedding(nn.Module):
    """Precomputed cos/sin cache for a single (theta, partial_rotary, scaling)
    rope family. Step-3.7-Flash has two families (full vs sliding), so the model
    instantiates one of these per family and selects per layer.
    """

    def __init__(self, head_dim, partial_rotary_factor, theta, rope_scaling,
                 max_position_embeddings, device=None):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        inv_freq, self.rotary_dim = _compute_inv_freq(
            head_dim, partial_rotary_factor, theta, rope_scaling)
        self.register_buffer("inv_freq", inv_freq.to(device), persistent=False)
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, position_ids, is_prefill=True):
        # Keep the cos/sin cache resident on the same device as the query so
        # indexing by an NPU position_ids tensor stays on-device.
        #
        # Graph-mode (Decode) constraint: the cache was already built to
        # ``max_position_embeddings`` in ``__init__`` and Decode positions never
        # exceed it, so Decode is a pure index_select with NO host sync. The
        # ``.item()`` position-bound rebuild (a hard graph break) is only taken on
        # the Prefill/eager path, mirroring qwen3_moe's RotaryEmbedding (which
        # passes a static max_seq_len to skip the .item() under graph mode).
        # The device/dtype rebuild is kept eager-only too; under npugraph_ex the
        # cache device/dtype are compile-time-static, so Decode never rebuilds.
        if is_prefill:
            need_rebuild = (
                self.cos_cached is None
                or self.cos_cached.device != x.device
                or self.cos_cached.dtype != x.dtype
            )
            max_pos = (int(position_ids.max().item()) + 1) if position_ids.numel() > 0 else 1
            if need_rebuild or max_pos > self.max_seq_len_cached:
                self._set_cos_sin_cache(max(max_pos, self.max_seq_len_cached), x.device, x.dtype)
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        # npu_rotary_mul (TND layout) broadcasts cos/sin [T, 1, rotary_dim] over
        # the head axis. cos/sin span the full rotary_dim (cat((freqs,freqs))),
        # consumed directly by rotary_mode='half' (no extra concat needed).
        return cos.unsqueeze(1), sin.unsqueeze(1)


def _npu_rope_tnd(q_rot, k_rot, cos, sin):
    """Apply RoPE to q/k (TND layout) via npu_rotary_mul, one call each.

    ``npu_rotary_mul(x, r1=cos, r2=sin, rotary_mode='half')`` is the rotate_half
    (GPT-NeoX) RoPE used by Step-3.7-Flash. It takes a single tensor, is NOT
    in-place (returns a new tensor), and has no UB head-count tiling limit, so
    BOTH families work in a single call per q/k regardless of head count:
      - full   : q[T,64,64],  k[T,8,64],  cos/sin[T,1,64]
      - sliding : q[T,96,128], k[T,8,128], cos/sin[T,1,128]
    This replaces the previous head-blocking npu_apply_rotary_pos_emb path,
    which (a) hit error 561002 for nq=96/D=128 and (b) rotated k in place,
    requiring clone-throwaway gymnastics. cos/sin already span the full
    rotary_dim (built as cat((freqs,freqs))), so rotary_mode='half' consumes
    them directly with no extra concat. Verified on-device vs both the prior
    head-blocking impl (cos=1.0) and a pure-torch rotate_half reference
    (cos=0.999999).
    """
    cos = cos.contiguous()
    sin = sin.contiguous()
    q_out = torch_npu.npu_rotary_mul(q_rot.contiguous(), cos, sin, rotary_mode='half')
    k_out = torch_npu.npu_rotary_mul(k_rot.contiguous(), cos, sin, rotary_mode='half')
    return q_out, k_out


def apply_partial_rotary(query, key, cos, sin):
    """Apply RoPE only to the first ``rotary_dim`` channels (partial rotary).

    query/key are [T, num_heads, head_dim]; cos/sin are [T, 1, rotary_dim], so the
    rotary span is taken as ``cos.shape[-1]``.
    """
    rotary_dim = cos.shape[-1]
    if rotary_dim == query.shape[-1]:
        q_rot, k_rot = query, key
        q_pass = k_pass = None
    else:
        q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_rot, k_rot = _npu_rope_tnd(q_rot, k_rot, cos, sin)
    if q_pass is None:
        return q_rot, k_rot
    return (torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1))


class Step3p7MLP(nn.Module):
    """Dense SwiGLU MLP (dense layers 0-2 and shared expert), TP-parallel."""

    def __init__(self, config, infer_config, comm_manager, intermediate_size,
                 swiglu_limit=None, prefix=""):
        super().__init__()
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        tp_rank = comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0
        self.comm_manager = comm_manager
        self.gate_proj = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=intermediate_size,
            bias=False, tp_size=self.dense_tp_size, tp_rank=tp_rank,
            prefix=f"{prefix}.gate_proj")
        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=intermediate_size,
            bias=False, tp_size=self.dense_tp_size, tp_rank=tp_rank,
            prefix=f"{prefix}.up_proj")
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size, output_size=config.hidden_size,
            bias=False, tp_size=self.dense_tp_size, tp_rank=tp_rank,
            input_is_parallel=True, prefix=f"{prefix}.down_proj")
        self.limit = swiglu_limit

    def forward(self, x):
        # HF reference (Step3p7MLP.forward / .original_ref:447-454): the clamp is
        # applied AFTER silu on the gate (gate = silu(gate_proj(x)); clamp(max)),
        # and to the raw up projection (clamp(-limit, limit)). silu(clamp(x)) !=
        # clamp(silu(x)), so the clamp ordering is load-bearing.
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        out = self.down_proj(gate * up)
        if self.dense_tp_size > 1:
            dist.all_reduce(out, group=self.comm_manager.get_group("dense_tp_group"))
        return out


class Step3p7MoE(nn.Module):
    """Sparse MoE block: sigmoid(+bias) routing, renorm, scaling, shared expert.

    Routed experts use FusedMoEGMM with EP (moe_tp_size==1 -> pure EP) following
    the qwen3_moe Prefill(double_routing)/Decode(MC2 dispatch/combine) split.
    """

    def __init__(self, config, infer_config, comm_manager, swiglu_limit=None,
                 share_swiglu_limit=None, prefix=""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.moe_intermediate_size = config.moe_intermediate_size
        self.norm_expert_weight = config.norm_expert_weight
        self.routed_scaling_factor = config.moe_router_scaling_factor
        self.use_moe_router_bias = config.use_moe_router_bias
        self.need_fp32_gate = config.need_fp32_gate
        self.limit = swiglu_limit
        # Fused routing (npu_moe_gating_top_k) is on by default; verified
        # bit-exact vs native _route. Set custom_params route_use_fused=False to
        # fall back to the native eager routing path.
        custom_gate = getattr(infer_config.model_config, "custom_params", None) or {}
        self._use_fused_gating = bool(custom_gate.get("route_use_fused", True))

        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.force_eplb = infer_config.model_config.force_eplb
        custom = getattr(infer_config.model_config, "custom_params", None) or {}
        # MC2 dispatch comm_alg. Decode MoE EP always runs MC2
        # (npu_moe_distribute_*_v2). The default uses the op's default comm_alg
        # (empty string), which is the non-fullmesh window layout and is verified
        # production-usable across npugraph_ex / eager (byte-exact vs the
        # prefill double_routing path). mc2_comm_alg stays an optional
        # custom_param so a build that supports the "fullmesh_v2" window layout
        # can opt in by setting it explicitly; "" / "default" / None all use the
        # op default. The buffer-size formula (init_parallel_comm_group) reads the
        # same param via is_full_mesh_v2 so buffer and dispatch stay consistent.
        self.mc2_comm_alg = custom.get("mc2_comm_alg", "")

        # fp32 router gate (Step uses need_fp32_gate=True).
        self.gate = ReplicatedLinear(
            self.hidden_dim, self.num_experts, bias=False,
            params_dtype=torch.float32, prefix=f"{prefix}.gate")
        if self.use_moe_router_bias:
            self.router_bias = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32),
                requires_grad=False)
        else:
            self.register_parameter("router_bias", None)

        self.experts = FusedMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            bias=False, quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts")

        # Shared expert (dense SwiGLU, dim = share_expert_dim), TP-parallel.
        self.share_expert = Step3p7MLP(
            config, infer_config, comm_manager,
            intermediate_size=config.share_expert_dim,
            swiglu_limit=share_swiglu_limit, prefix=f"{prefix}.share_expert")

        # MC2 dispatch/combine kwargs are static config (group name, rank ids,
        # expert/world sizes, comm_alg) — build once here instead of rebuilding
        # them on every decode MoE forward. Only the decode MC2 path
        # (moe_ep_size > 1 and moe_tp_size == 1) registers moe_ep_group_mc2;
        # other layouts use _moe_tp / prefill double_routing and never read
        # these, so leave them unset there.
        self.dispatch_kwargs = None
        self.combine_kwargs = None
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            self._set_mc2_kwargs()

    def _route(self, hidden_states):
        logits = self.gate(hidden_states)  # ReplicatedLinear in fp32
        # Fused routing path: npu_moe_gating_top_k fuses sigmoid + bias-corrected
        # top-k selection + raw-sigmoid weight gather + renorm + scaling into one
        # kernel. Verified on-device to be
        # BIT-EXACT vs the native _route below (row_exact_match=1.0, weight
        # cos=1.0, max_abs=0). Step3p7 has no expert grouping -> group_count=1,
        # k_group=1. norm_type=1 (sigmoid), renorm=0 (score->topk), with the op's
        # yOut using RAW sigmoid scores (not bias-corrected) which matches
        # _route's torch.gather(gate_prob, idx). Mirrors DeepSeek-R1 noaux_tc.
        if self.use_moe_router_bias and self._use_fused_gating:
            # npu_moe_gating_top_k requires logits and bias to share a dtype.
            # The router bias is fp32; the gate logits dtype follows the matmul
            # output (bf16 when params are bf16), so cast logits to fp32 to
            # match. The native _route also operates in fp32 (sigmoid(.float())).
            topk_weight, indices, _ = torch_npu.npu_moe_gating_top_k(
                logits.float(),
                k=self.top_k,
                bias=self.router_bias.float(),
                k_group=1,
                group_count=1,
                group_select_mode=1,
                renorm=0,
                norm_type=1,
                routed_scaling_factor=(self.routed_scaling_factor
                                       if self.norm_expert_weight else 1.0),
                eps=float(1e-20),
            )
            if not self.norm_expert_weight:
                # op renorms internally; if the model does not renorm we cannot
                # use the fused renorm. (Step3p7 always renorms -> not reached.)
                topk_weight = topk_weight * self.routed_scaling_factor
            return indices.to(torch.int32), topk_weight
        return self._route_native(logits)

    def _route_native(self, logits):
        """Native (eager) routing — kept as a verified-correct fallback."""
        gate_prob = torch.sigmoid(logits.float())
        if self.use_moe_router_bias:
            gate_prob_with_bias = gate_prob + self.router_bias.unsqueeze(0)
            _, indices = torch.topk(gate_prob_with_bias, k=self.top_k, dim=-1)
            topk_weight = torch.gather(gate_prob, 1, indices)
        else:
            # sigmoid routing: normalize then top-k
            gate_prob = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
            topk_weight, indices = torch.topk(gate_prob, k=self.top_k, dim=-1)
        if self.norm_expert_weight:
            topk_weight = topk_weight / (
                torch.sum(topk_weight, dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return indices.to(torch.int32), topk_weight

    def forward(self, hidden_states):
        is_prefill = get_forward_metadata().is_prefill
        token_count, h = hidden_states.shape
        topk_idx, topk_weight = self._route(hidden_states)
        share_output = self.share_expert(hidden_states)
        if self.moe_tp_size > 1:
            moe_output = self._moe_tp(hidden_states, topk_idx, topk_weight)
        elif is_prefill:
            # Prefill EP: init_routing + all_to_all + re_routing (double_routing).
            moe_output = self._moe_ep_prefill(hidden_states, topk_idx, topk_weight)
        else:
            # Decode EP: MC2 (npu_moe_distribute_dispatch_v2 / combine_v2).
            moe_output = self._moe_ep_decode(hidden_states, topk_idx, topk_weight)
        return moe_output.view(token_count, h) + share_output

    def _moe_tp(self, hidden_states, topk_idx, topk_weight):
        token_count, h = hidden_states.shape
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states, expert_idx=topk_idx,
            active_num=token_count * self.top_k, scale=None,
            expert_num=self.num_experts, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=-1)
        out = self.experts(expanded_x, tokens_per_expert, group_list_type=1,
                           swiglu_limit=self.limit)
        out = torch_npu.npu_moe_finalize_routing(
            out, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(out.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2)
        if self.moe_tp_size > 1:
            dist.all_reduce(out, group=self.comm_manager.get_group("moe_tp_group"))
        return out

    def _dispatch_double_routing(self, tokens_per_expert, expanded_x):
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

    def _moe_ep_prefill(self, hidden_states, topk_idx, topk_weight):
        token_count, h = hidden_states.shape
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states, expert_idx=topk_idx,
            active_num=topk_idx.shape[0] * topk_idx.shape[1], scale=None,
            expert_num=self.num_experts, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=-1)
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group, gathered_tokens, input_splits, output_splits = \
            self._dispatch_double_routing(tokens_per_expert, expanded_x)
        ordered, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(
                gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1))
        ordered = self.experts(ordered, tokens_per_local_expert, group_list_type=1,
                               swiglu_limit=self.limit)
        new_x = torch.index_select(ordered, 0, gathered_ids_unsort.float().argsort().int())
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)
        out = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2)
        return out

    def _set_mc2_kwargs(self):
        global_rank = self.comm_manager.config.global_rank
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        common = dict(
            moe_expert_num=self.num_experts, global_bs=0,
            group_ep=mc2_group_name, ep_world_size=self.moe_ep_size,
            ep_rank_id=global_rank // self.moe_tp_size,
            group_tp=mc2_group_name, tp_world_size=self.moe_tp_size,
            tp_rank_id=global_rank % self.moe_tp_size,
            expert_shard_type=0, shared_expert_num=0, shared_expert_rank_num=0)
        # comm_alg only set in dispatch_kwargs (combine infers the algorithm from
        # dispatch outputs). Empty / "default" / None -> drop comm_alg so the op
        # uses its default (non-fullmesh) window layout. A non-empty value opts
        # into that explicit comm_alg (e.g. "fullmesh_v2").
        use_fullmesh = self.mc2_comm_alg not in ("", "default", None)
        if use_fullmesh:
            self.dispatch_kwargs = dict(x_active_mask=None, scales=None,
                                        quant_mode=0, comm_alg=self.mc2_comm_alg,
                                        **common)
        else:
            self.dispatch_kwargs = dict(x_active_mask=None, scales=None,
                                        quant_mode=0, **common)
        self.combine_kwargs = dict(x_active_mask=None, comm_quant_mode=0, **common)

    def _moe_ep_decode(self, hidden_states, topk_idx, topk_weight):
        # dispatch_kwargs / combine_kwargs are prebuilt once in __init__.
        out = torch_npu.npu_moe_distribute_dispatch_v2(
            x=hidden_states, expert_ids=topk_idx, **self.dispatch_kwargs)
        expand_x, _, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = out[:6]
        ordered = self.experts(expand_x, expert_token_num, group_list_type=1,
                               swiglu_limit=self.limit)
        out = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=ordered, expert_ids=topk_idx,
            assist_info_for_combine=expand_idx,
            expert_scales=topk_weight.to(torch.float32),
            ep_send_counts=ep_recv_counts, tp_send_counts=tp_recv_counts,
            **self.combine_kwargs)
        return out


class Step3p7Attention(nn.Module):
    """GQA attention with mixed full/sliding window, partial RoPE, per-head
    q/k RMSNorm and head-wise sigmoid output gate. attn_tp parallel.
    """

    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.comm_manager = comm_manager
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.block_size = infer_config.scheduler_config.block_size
        # Graph-mode backend flag (mirror qwen3_moe exec_qkv). Under npugraph_ex
        # the FA decode call must take actual_seq_lengths as list[int]; the engine
        # prebuilds these in forward_metadata.actual_seq_lengths_list_kv /
        # actual_seq_lengths_cu_list_q. Prefill stays eager (Tensor args).
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        tp_rank = comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0

        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == SLIDING_ATTENTION
        if self.is_sliding:
            other = config.attention_other_setting
            self.num_heads = other["num_attention_heads"]
            self.num_kv_heads = other["num_attention_groups"]
            self.sliding_window = config.sliding_window
            self.attn_type = ATTN_TYPE_SLIDING
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups
            self.sliding_window = None
            self.attn_type = ATTN_TYPE_FULL

        self.head_dim = config.head_dim
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_kv_heads_per_rank = max(self.num_kv_heads // self.attn_tp_size, 1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_head_wise_attn_gate = config.use_head_wise_attn_gate

        self.q_proj = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=self.q_size, bias=False,
            tp_size=self.attn_tp_size, tp_rank=tp_rank, prefix=f"{prefix}.q_proj")
        self.k_proj = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=self.kv_size, bias=False,
            tp_size=self.attn_tp_size, tp_rank=tp_rank, prefix=f"{prefix}.k_proj")
        self.v_proj = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=self.kv_size, bias=False,
            tp_size=self.attn_tp_size, tp_rank=tp_rank, prefix=f"{prefix}.v_proj")
        self.o_proj = RowParallelLinear(
            input_size=self.q_size, output_size=config.hidden_size, bias=False,
            tp_size=self.attn_tp_size, tp_rank=tp_rank, input_is_parallel=True,
            prefix=f"{prefix}.o_proj")
        self.q_norm = Step3p7PerHeadRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7PerHeadRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.use_head_wise_attn_gate:
            # one gate scalar per Q head; shard by attn_tp on output dim.
            self.g_proj = ColumnParallelLinear(
                input_size=config.hidden_size, output_size=self.num_heads,
                bias=False, tp_size=self.attn_tp_size, tp_rank=tp_rank,
                prefix=f"{prefix}.g_proj")

        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        cache_dtype = _resolve_dtype(getattr(config, "torch_dtype", None))
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache", attn_type=self.attn_type, dim=self.head_dim,
                num_head=self.num_kv_heads_per_rank, dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
                sliding_window=self.sliding_window),
            CacheEntry(
                cache_name="v_cache", attn_type=self.attn_type, dim=self.head_dim,
                num_head=self.num_kv_heads_per_rank, dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
                sliding_window=self.sliding_window),
        ]

    def _update_cache(self, slot_mapping, key_states, value_states):
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping[self.attn_type].view(-1, 1),
            key_states.view(-1, self.num_kv_heads_per_rank, self.head_dim))
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping[self.attn_type].view(-1, 1),
            value_states.view(-1, self.num_kv_heads_per_rank, self.head_dim))

    def forward(self, hidden_states, position_embeddings, forward_metadata,
                slot_mapping=None, block_table=None, **kwargs):
        is_prefill = forward_metadata.is_prefill if forward_metadata else False
        q_len = hidden_states.size(0)

        # attn DP: gather tokens across the tp group when attn is replicated (DP)
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            h_dim = hidden_states.size(-1)
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            gathered = torch.empty([q_len * self.attn_tp_size, h_dim],
                                   dtype=hidden_states.dtype, device="npu")
            dist.all_gather_into_tensor(gathered, hidden_states, group=attn_tp_group)
            hidden_states = gathered
            q_len = hidden_states.size(0)

        padded_q_len = q_len
        prompt_tokens = forward_metadata.prompt_tokens if forward_metadata else q_len
        if is_prefill and prompt_tokens < padded_q_len:
            hidden_states = hidden_states[:prompt_tokens]
            q_len = prompt_tokens

        query = self.q_proj(hidden_states).view(q_len, self.num_heads_per_rank, self.head_dim)
        key = self.k_proj(hidden_states).view(q_len, self.num_kv_heads_per_rank, self.head_dim)
        value = self.v_proj(hidden_states).view(q_len, self.num_kv_heads_per_rank, self.head_dim)
        query = self.q_norm(query.contiguous())
        key = self.k_norm(key.contiguous())
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)  # [q_len, num_heads_per_rank]

        cos, sin = position_embeddings[self.attn_type]
        query, key = apply_partial_rotary(query, key, cos, sin)

        sparse_mode = 4 if self.sliding_window else 3
        pre_tokens = self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max
        attention_mask = forward_metadata.attention_mask

        # FA decode needs backend-specific actual_seq_lengths:
        #   - npugraph_ex : list[int] (engine prebuilds *_list_kv / *_cu_list_q)
        #   - eager       : Tensor via torch.ops.npu
        # (mirrors qwen3_moe exec_qkv). Prefill always stays eager (Tensor).
        if is_prefill:
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                query, key, value,
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="TND", softmax_scale=self.scaling,
                sparse_mode=sparse_mode, pre_tokens=pre_tokens, next_tokens=0,
                actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_qlen,
                atten_mask=attention_mask)
            self._update_cache(slot_mapping, key, value)
        else:
            self._update_cache(slot_mapping, key, value)
            if self.enable_npugraph_ex:
                actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
                actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
            else:
                actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv
                actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                query,
                self.k_cache.view(*self.k_cache.shape[:2], -1),
                self.v_cache.view(*self.v_cache.shape[:2], -1),
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="TND", softmax_scale=self.scaling,
                sparse_mode=sparse_mode, pre_tokens=pre_tokens, next_tokens=0,
                actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen,
                atten_mask=attention_mask,
                block_table=block_table[self.attn_type], block_size=self.block_size)

        attn_output = attn_output.reshape(q_len, self.num_heads_per_rank, self.head_dim)
        if self.use_head_wise_attn_gate:
            attn_output = attn_output * gate_states.unsqueeze(-1).sigmoid()
        attn_output = attn_output.reshape(q_len, self.num_heads_per_rank * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if is_prefill and q_len < padded_q_len:
            pad = torch.zeros((padded_q_len - q_len, attn_output.size(-1)),
                              dtype=attn_output.dtype, device=attn_output.device)
            attn_output = torch.cat([attn_output, pad], dim=0)
            q_len = padded_q_len

        # attn DP: reduce-scatter o_proj output back to per-rank tokens
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            token_count, h = attn_output.size()
            reduced = torch.empty([token_count // self.attn_tp_size, h],
                                  dtype=attn_output.dtype, device="npu")
            dist.reduce_scatter_tensor(
                reduced, attn_output.view(token_count, h),
                group=self.comm_manager.get_group("attn_tp_group"))
            attn_output = reduced
        elif self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))
        return attn_output


class Step3p7DecoderLayer(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(
            config, infer_config, comm_manager, layer_idx, prefix=f"{prefix}.self_attn")
        self.attention_type = config.layer_types[layer_idx]

        self.is_moe_layer = layer_idx in config.moe_layers_idx
        swiglu_limit = config.swiglu_limits[layer_idx] if config.swiglu_limits else None
        swiglu_limit = swiglu_limit if swiglu_limit else None
        share_limit = config.swiglu_limits_shared[layer_idx] if config.swiglu_limits_shared else None
        share_limit = share_limit if share_limit else None

        if self.is_moe_layer:
            self.mlp = Step3p7MoE(
                config, infer_config, comm_manager, swiglu_limit=swiglu_limit,
                share_swiglu_limit=share_limit, prefix=f"{prefix}.moe")
        else:
            self.mlp = Step3p7MLP(
                config, infer_config, comm_manager,
                intermediate_size=config.intermediate_size,
                swiglu_limit=share_limit, prefix=f"{prefix}.mlp")

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_residual=None,
                forward_metadata=None, **kwargs):
        # slot_mapping / block_table flow through **kwargs to self_attn (the
        # caller passes them as keywords); they are not used in this layer body.
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
        hidden_states = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings,
            forward_metadata=forward_metadata, **kwargs)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return residual, hidden_states


class Step3p7Model(nn.Module):
    def __init__(self, config, infer_config, comm_manager, prefix=""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.rank_id = int(os.getenv("LOCAL_RANK", "0"))
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.max_position_embeddings = config.max_position_embeddings

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size, self.padding_idx, torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList([
            Step3p7DecoderLayer(config, infer_config, comm_manager, layer_idx,
                                prefix=f"model.layers.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)])
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Two RoPE families: full (theta_full, partial_full, +yarn) and sliding.
        self._build_rotary(config)

    def _build_rotary(self, config):
        rope_theta = config.rope_theta
        partials = config.partial_rotary_factors
        yarn_only = config.yarn_only_types or []

        def family_params(layer_type):
            idx = config.layer_types.index(layer_type)
            theta = rope_theta[idx] if isinstance(rope_theta, list) else rope_theta
            partial = partials[idx] if partials else 1.0
            scaling = config.rope_scaling if layer_type in yarn_only else None
            return theta, partial, scaling

        theta_f, partial_f, scaling_f = family_params(FULL_ATTENTION)
        self.rotary_full = Step3p7RotaryEmbedding(
            config.head_dim, partial_f, theta_f, scaling_f, self.max_position_embeddings)
        if SLIDING_ATTENTION in config.layer_types:
            theta_s, partial_s, scaling_s = family_params(SLIDING_ATTENTION)
            self.rotary_sliding = Step3p7RotaryEmbedding(
                config.head_dim, partial_s, theta_s, scaling_s, self.max_position_embeddings)
        else:
            self.rotary_sliding = None

    def _inject_image_embeds(self, inputs_embeds, local_input_ids, shard,
                             image_embeds, img_tok_id):
        """Overwrite `<im_patch>` placeholder rows with vision embeds.

        Runs AFTER embed_tp AllReduce (so we scatter onto the final per-token
        hidden, never on the vocab-masked partial embed) and AFTER attn-DP token
        sharding (so each rank only touches its local token window).

        Args:
            inputs_embeds: [local_T, hidden] hidden for this rank's token slice.
            local_input_ids: [local_T] the (possibly sharded) token ids that
                produced `inputs_embeds`.
            shard: (full_input_ids, local_start) tuple describing this rank's
                window within the full sequence:
                  full_input_ids [T] -- the full-sequence ids (== local_input_ids
                    when no attn-DP sharding), used to map this rank's local
                    placeholders to the matching contiguous range of image_embeds;
                  local_start -- index of this rank's first token in the full
                    sequence.
            image_embeds: [num_placeholders_total, hidden] flattened vision
                embeds, in placeholder order over the FULL sequence.
            img_tok_id: placeholder token id (config.image_token_id, 128001).

        Returns:
            inputs_embeds with placeholder rows replaced (out-of-place, so the
            text path's tensor is never mutated when this is not called).
        """
        full_input_ids, local_start = shard
        image_embeds = image_embeds.to(inputs_embeds.dtype).to(inputs_embeds.device)
        local_mask = (local_input_ids == img_tok_id)
        # Offset into the flattened image_embeds: how many placeholders appear in
        # the full sequence strictly before this rank's local window. With
        # attn_tp_size == 1 (candidate A) local_start == 0 -> offset == 0 and the
        # whole image_embeds tensor applies to the (full) local sequence.
        if local_start > 0:
            prior = int((full_input_ids[:local_start] == img_tok_id).sum().item())
        else:
            prior = 0
        n_local = int(local_mask.sum().item())
        chunk = image_embeds[prior:prior + n_local]
        # Out-of-place scatter (torch boolean-mask assignment) keeps graph-mode
        # off the hook (vision runs Prefill-only / eager) and avoids aliasing the
        # text path. One-shot per Prefill over ~169 of 4K positions -> no fusion
        # op is warranted.
        merged = inputs_embeds.clone()
        merged[local_mask] = chunk
        return merged

    def forward(self, input_ids, position_ids=None, forward_metadata=None,
                image_embeds=None, image_token_id=None, **kwargs):
        is_prefill = forward_metadata.is_prefill if forward_metadata else False
        token_count = input_ids.shape[0]
        position_ids = position_ids.view(-1).long()

        # --- Vision injection (opt-in) ----------------
        # `image_embeds` is the flattened [num_placeholder_tokens, hidden] vision
        # embedding aligned, in order, to every `<im_patch>` (id=image_token_id)
        # position in the FULL input_ids. When provided we must remember the
        # full-sequence placeholder layout BEFORE the attn-DP token sharding
        # below slices input_ids to the rank-local window, so that each rank can
        # pick the image-embed sub-range matching its local placeholders. When
        # image_embeds is None this whole block is skipped and the text path is
        # byte-for-byte unchanged.
        inject_vision = image_embeds is not None
        full_input_ids = input_ids if inject_vision else None
        img_tok_id = (image_token_id if image_token_id is not None
                      else DEFAULT_IMAGE_TOKEN_ID)

        # attn DP token sharding (matches qwen3_moe): split input across tp group
        local_start = 0
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            rank_in_tp = self.comm_manager.get_rank("attn_tp_group")
            if is_prefill:
                prompt_tokens = forward_metadata.prompt_tokens
                padded = ((prompt_tokens + self.attn_tp_size - 1) // self.attn_tp_size) * self.attn_tp_size
                if padded > token_count:
                    pad = torch.zeros((padded - token_count,), dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, pad], dim=0)
                    token_count = padded
                    if inject_vision:
                        full_input_ids = input_ids
            tokens_per_rank = token_count // self.attn_tp_size
            local_start = rank_in_tp * tokens_per_rank
            input_ids = input_ids[local_start:local_start + tokens_per_rank]

        if self.embed_tp_size > 1:
            new_ids = input_ids - self.rank_id * self.vocab_size_per_rank
            mask = (new_ids >= 0) & (new_ids < self.vocab_size_per_rank)
            inputs_embeds = self.embed_tokens(new_ids * mask) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        if inject_vision:
            inputs_embeds = self._inject_image_embeds(
                inputs_embeds, input_ids, (full_input_ids, local_start),
                image_embeds, img_tok_id)

        hidden_states = inputs_embeds

        position_embeddings = {
            ATTN_TYPE_FULL: self.rotary_full(hidden_states, position_ids, is_prefill),
        }
        if self.rotary_sliding is not None:
            position_embeddings[ATTN_TYPE_SLIDING] = self.rotary_sliding(
                hidden_states, position_ids, is_prefill)

        block_table = forward_metadata.block_table
        slot_mapping = forward_metadata.slot_mapping
        residual = None
        for layer in self.layers:
            residual, hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings,
                past_residual=residual, forward_metadata=forward_metadata,
                slot_mapping=slot_mapping, block_table=block_table, **kwargs)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Step3p5ForCausalLM(nn.Module):
    """Step-3.7-Flash text MoE backbone, unified-flow entry class."""

    def __init__(self, config, infer_config: InferenceConfig,
                 comm_manager: CommManager = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.num_experts = config.moe_num_experts
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size

        self.init_parallel_comm_group()
        self.model = Step3p7Model(config, infer_config, comm_manager, prefix)
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size, output_size=config.vocab_size, bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0)

    def init_parallel_comm_group(self):
        cm = self.comm_manager
        cm.register_group(name="attn_tp_group",
                          group_num=self.world_size // self.attn_tp_size,
                          group_size=self.attn_tp_size)
        cm.register_group(name="embed_tp_group",
                          group_num=self.world_size // self.embed_tp_size,
                          group_size=self.embed_tp_size)
        cm.register_group(name="lmhead_tp_group",
                          group_num=self.world_size // self.lmhead_tp_size,
                          group_size=self.lmhead_tp_size)
        if self.dense_tp_size > 1:
            cm.register_group(name="dense_tp_group",
                              group_num=self.world_size // self.dense_tp_size,
                              group_size=self.dense_tp_size)
        if self.moe_tp_size > 1:
            cm.register_group(name="moe_tp_group",
                              group_num=self.world_size // self.moe_tp_size,
                              group_size=self.moe_tp_size)
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            cm.register_group(name="moe_ep_group", group_num=moe_ep_group_num,
                              group_size=self.moe_ep_size, group_stride=moe_ep_group_num,
                              return_name=True)
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            # MC2 decode HCCL buffer. Its size formula must match the dispatch
            # comm_alg window layout, so read the same mc2_comm_alg custom_param
            # Step3p7MoE uses (default "" -> op default / non-fullmesh layout).
            # is_full_mesh_v2 follows mc2_comm_alg so buffer and dispatch stay
            # consistent (a mismatch can corrupt the MC2 AIV window).
            mc2_custom = getattr(
                self.infer_config.model_config, "custom_params", None) or {}
            mc2_comm_alg = mc2_custom.get("mc2_comm_alg", "")
            mc2_full_mesh = mc2_comm_alg not in ("", "default", None)
            mc2_buffer_size = calc_moe_hccl_buffer_size(
                self.infer_config, self.config, is_full_mesh_v2=mc2_full_mesh)
            cm.register_group(name="moe_ep_group_mc2",
                              group_num=self.world_size // self.moe_ep_size,
                              group_size=self.moe_ep_size,
                              group_stride=self.world_size // self.moe_ep_size,
                              return_name=True, allow_physical_reuse=False,
                              hccl_buffer_size=mc2_buffer_size, group_type=None)

    def forward(self, input_ids=None, position_ids=None, forward_metadata=None, **kwargs):
        hidden_states = self.model(
            input_ids=input_ids, position_ids=position_ids,
            forward_metadata=forward_metadata, **kwargs)
        hidden_size = hidden_states.shape[-1]

        # Select the last token of each sequence (prefill) for next-token logits.
        if forward_metadata.is_prefill:
            cu_seq_lens_q = forward_metadata.actual_seq_lengths_cu_q
            seq_index = (cu_seq_lens_q.to(torch.int32) - 1).npu()
            if self.attn_tp_size > 1 and self.attn_dp_size > 1:
                q_len = hidden_states.size(0)
                gathered = torch.empty([self.attn_tp_size * q_len, hidden_size],
                                       dtype=hidden_states.dtype, device=hidden_states.device)
                dist.all_gather_into_tensor(
                    gathered, hidden_states, group=self.comm_manager.get_group("attn_tp_group"))
                hidden_states = gathered
            hidden_states = torch.index_select(hidden_states.reshape(-1, hidden_size), 0, seq_index.view(-1))
            hidden_states = hidden_states.view(seq_index.numel(), 1, hidden_size)
        else:
            hidden_states = hidden_states.view(hidden_states.shape[0], 1, hidden_size)

        logits = self.lm_head(hidden_states)

        if not forward_metadata.is_prefill and self.attn_tp_size > 1 and self.attn_dp_size > 1:
            bsz, q_len, v = logits.size()
            new_logits = torch.empty([bsz * self.attn_tp_size, q_len, v],
                                     dtype=logits.dtype, device="npu")
            dist.all_gather_into_tensor(new_logits, logits,
                                        group=self.comm_manager.get_group("attn_tp_group"))
            logits = new_logits
        if self.lmhead_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lmhead_tp_size)]
            dist.all_gather(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.concat(new_logits, dim=-1)
        return logits.float()

    def get_cache_info(self) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_infos.append(LayerCacheInfo(
                layer_idx=layer_idx, caches=list(layer.self_attn.cache_entries)))
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            block_size=self.infer_config.scheduler_config.block_size,
            layer_infos=layer_infos)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Step-3.7-Flash text-backbone weights.

        Notable checkpoint specifics handled here:
        - RMSNorm gains use the +1 convention -> add 1.0 to all *norm weights.
        - MoE expert weights are *stacked* per layer: ``moe.{gate,up,down}_proj``
          have shape ``[num_experts, ...]``. We slice each expert out and route
          through ``FusedMoEGMM.weight_loader`` (w1=gate, w3=up, w2=down) so EP
          rank filtering / TP sharding happen consistently.
        - MTP / vision / lm-head-tie weights are skipped.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        num_layers = self.config.num_hidden_layers

        # Map of (ckpt suffix) -> (FusedMoEGMM shard_id)
        expert_shard = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

        def is_skipped(name):
            # MTP layers (>= num_hidden_layers) and vision tower are out of scope.
            if name.startswith("vision_model.") or name.startswith("model.vision_model."):
                return True
            for li in range(num_layers, num_layers + 8):
                if name.startswith(f"model.layers.{li}."):
                    return True
            return False

        for name, loaded_weight in weights:
            if is_skipped(name):
                continue

            # ---- MoE stacked expert weights -------------------------------
            moe_hit = False
            for ck_suffix, shard_id in expert_shard.items():
                token = f".moe.{ck_suffix}.weight"
                if name.endswith(token) and ".moe.gate." not in name:
                    layer_prefix = name[: -len(token)]  # model.layers.{L}
                    param_name = f"{layer_prefix}.mlp.experts." + (
                        "w13_weight" if shard_id in ("w1", "w3") else "w2_weight")
                    if param_name not in params_dict:
                        moe_hit = True
                        break
                    param = params_dict[param_name]
                    wl = param.weight_loader
                    # loaded_weight: [num_experts, out, in]; slice per expert.
                    for expert_id in range(self.num_experts):
                        wl(param, loaded_weight[expert_id], param_name,
                           shard_id=shard_id, expert_id=expert_id)
                    loaded_params.add(param_name)
                    moe_hit = True
                    break
            if moe_hit:
                continue

            # ---- router gate / bias (rename moe.gate -> mlp.gate) ----------
            if ".moe.gate.weight" in name:
                pname = name.replace(".moe.gate.weight", ".mlp.gate.weight")
                if pname in params_dict:
                    params_dict[pname].weight_loader(params_dict[pname], loaded_weight.float())
                    loaded_params.add(pname)
                continue
            if name.endswith(".moe.router_bias"):
                pname = name.replace(".moe.router_bias", ".mlp.router_bias")
                if pname in params_dict:
                    params_dict[pname].data.copy_(loaded_weight.float())
                    loaded_params.add(pname)
                continue

            # ---- shared expert / dense mlp rename --------------------------
            # ckpt: model.layers.{L}.share_expert.* -> model module mlp.share_expert.*
            mapped = name
            if ".share_expert." in name:
                mapped = name.replace(".share_expert.", ".mlp.share_expert.")
            # dense mlp (layers 0-2) ckpt key is model.layers.{L}.mlp.* already.

            # ---- RMSNorm +1 offset -----------------------------------------
            if mapped.endswith("norm.weight") or mapped.endswith("_norm.weight"):
                if mapped in params_dict:
                    params_dict[mapped].data.copy_(loaded_weight + 1.0)
                    loaded_params.add(mapped)
                continue

            if mapped not in params_dict:
                # tied lm_head (ckpt has separate lm_head.weight, keep), final norm
                # `model.norm.weight` handled above; anything else unmatched skip.
                continue
            param = params_dict[mapped]
            wl = getattr(param, "weight_loader", default_weight_loader)
            wl(param, loaded_weight)
            loaded_params.add(mapped)
        return loaded_params

    def process_weights_after_loading(self):
        for module_name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(
                    module, is_nz=self.infer_config.model_config.enable_weight_nz)
                if "gate" in module_name:
                    module.weight.data = module.weight.data.contiguous()


__all__ = ["Step3p5ForCausalLM", "Step3p7Model"]
