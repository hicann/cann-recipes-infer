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

import glob
import logging
import math
import os
import re
import sysconfig
from dataclasses import replace
from functools import lru_cache
from typing import Generator, Optional, Set, Tuple

import torch
from torch import nn
import torch_npu
import torch.distributed as dist

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.stream_utils import (
    create_event,
    create_stream,
    npu_stream_switch,
    record_event,
    record_stream,
    wait_event,
)
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
    VocabParallelEmbedding,
    ReplicatedLinear,
)
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization.mxfp4 import W4A8MxFp4MoEGMMMethod

from .configuration_hy_v3 import HYV3Config

logger = logging.getLogger(__name__)


def _custom_ops_dir():
    return os.path.join(sysconfig.get_paths()["purelib"], "custom_ops")


def _load_swiglu_group_quant_op():
    """Load the installed swiglu_group_quant op without importing custom_ops."""
    if hasattr(torch.ops.custom, "npu_swiglu_group_quant"):
        return
    so = glob.glob(os.path.join(_custom_ops_dir(), "custom_ops_lib*.so"))
    if so:
        torch.ops.load_library(so[0])


_load_swiglu_group_quant_op()


def _sp_enabled(infer_config):
    """Explicit sequence-parallel switch.

    SP (token-shard across the attn_tp group + attention AllGather-in /
    ReduceScatter-out, prefill-only) is driven by an explicit config flag
    (custom_params.enable_sp), deployment-controlled rather than inferred from
    topology.

    The ONLY physical precondition for attention SP is attn_tp>1 (checked at the
    use site). attn_dp / ep do NOT gate SP: attn_dp is a batch-DP encoding, and
    attn_tp==ep only selects the MoE dispatch (moe_infer_ag vs manual AllToAll).
    """
    try:
        return bool(infer_config.model_config.custom_params.get("enable_sp", False))
    except Exception:
        return False


def _sp_transport_quant(x, gmm_quant_mode, target_linear=None):
    """Quantize an activation for the SP / DP-TP-DP transport, dispatched by quant tier.

    mxfp4 (w4a8mx): dynamic MXFP8, returns (values, per-group scale).
    fp8 (w8a8float8): static per-tensor quant by `target_linear`.input_scale,
    returns (values, None); the scalar scale is not transported.
    """
    if gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx"):
        return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
    if gmm_quant_mode == "w8a8float8":
        q = torch_npu.npu_quantize(
            x, target_linear.input_scale, None,
            torch.float8_e4m3fn, -1, True)
        return q, None
    raise NotImplementedError(
        f"DP-TP-DP / SP transport quant not implemented for gmm_quant_mode="
        f"{gmm_quant_mode!r}; wired tiers: mxfp4 (dynamic MXFP8), fp8 "
        f"(static per-tensor)."
    )


def _equal_all_to_all(x, group, group_size):
    """Equal-split all_to_all over dim0, flattened to 1D with explicit split
    sizes so it captures correctly under npugraph_ex. dim0 must be divisible by
    group_size.
    """
    flat = x.reshape(-1)
    n = flat.shape[0]
    chunk = n // group_size
    splits = [chunk] * group_size
    out = torch.empty_like(flat)
    dist.all_to_all_single(out, flat, output_split_sizes=splits,
                           input_split_sizes=splits, group=group)
    return out.view_as(x)


def _build_pad_aware_prefill_metadata(forward_metadata, slot_mapping, block_table,
                                      pad_len, prompt_tokens):
    """Append the SP alignment pad as an independent dummy segment so FA runs on the
    full padded length (model-side, no framework changes).

    The pad (pad_len <= attn_tp-1, from right-padding the prompt to a multiple of
    attn_tp) becomes one extra request-segment in the packed metadata:
      * actual_seq_lengths_cu_q: append prompt_tokens + pad_len
      * actual_seq_lengths_kv:   append pad_len (the dummy's own kv length)
      * slot_mapping / block_table: point the dummy at null_block (block 0)

    The dummy writes its pad K/V to null_block and reads back the same offsets, so its
    attention is self-consistent (no stale read). null_block is block 0 by BlockPool
    invariant (free_queue[0] = deque(range(N)).popleft(), never given to real requests,
    so real slots >= block_size never collide). Real segments are byte-for-byte
    unchanged; the dummy output is dropped by the real-cu_q index_select at the model tail.

    Returns fresh (forward_metadata, slot_mapping, block_table); inputs untouched.
    """
    cu_q = forward_metadata.actual_seq_lengths_cu_q
    kv = forward_metadata.actual_seq_lengths_kv
    padded_cu_q = torch.cat([cu_q, cu_q.new_tensor([prompt_tokens + pad_len])])
    padded_kv = torch.cat([kv, kv.new_tensor([pad_len])])
    fmeta = replace(
        forward_metadata,
        actual_seq_lengths_cu_q=padded_cu_q,
        actual_seq_lengths_kv=padded_kv,
    )

    new_slot_mapping = dict(slot_mapping) if slot_mapping else slot_mapping
    if slot_mapping:
        for key, sm in slot_mapping.items():
            sm_flat = sm.view(-1)
            # DISCARD_SLOT = null_block(=0) * block_size + [0..pad_len-1] = arange(pad_len)
            dummy_slots = torch.arange(pad_len, device=sm_flat.device, dtype=sm_flat.dtype)
            new_slot_mapping[key] = torch.cat([sm_flat, dummy_slots])

    new_block_table = dict(block_table) if block_table else block_table
    if block_table:
        for key, bt in block_table.items():
            dummy_row = bt.new_zeros((1, bt.shape[1]))  # all null_block (0)
            new_block_table[key] = torch.cat([bt, dummy_row], dim=0)

    return fmeta, new_slot_mapping, new_block_table


@lru_cache(maxsize=1)
def _ensure_qkv_fused_kscale_registered():
    from cann_ops_transformer.ops import (
        qkv_rms_norm_rope_cache_with_k_scale as _register_qkv_fused_kscale,  # noqa: F401
    )


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

        if position_ids.dim() != 1:
            raise RuntimeError("HYV3 expects packed 1D position_ids.")

        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        if x.dim() == 2:
            # TND packed hidden states: (total_tokens, hidden)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif x.dim() == 3:
            # Legacy BSH hidden states: (batch, seq_len, hidden)
            batch_size, seq_len, _ = x.shape
            cos = cos.view(batch_size, seq_len, 1, self.dim)
            sin = sin.view(batch_size, seq_len, 1, self.dim)
        else:
            raise RuntimeError(f"Unsupported HYV3 RoPE input dim: {x.dim()}")

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def get_cos_sin_table(self, max_seq_len=None):
        if max_seq_len is not None and max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return self.cos_sin_cached

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.register_buffer(
            "cos_sin_cached",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.float32),
            persistent=False,
        )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

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
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.comm_manager = comm_manager
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size

        # Sequence parallel: the ONLY physical precondition is attn_tp>1 (+enable_sp,
        # +prefill). NOT gated on attn_dp (a batch-DP encoding) nor on ep (which only
        # picks the MoE dispatch). Decode stays unsharded (single token would 0-row
        # crash if split across attn_tp).
        self.enable_sp = _sp_enabled(infer_config)
        sp_on = self.enable_sp and self.attn_tp_size > 1

        self.quant_config = getattr(config, "quant_config", None)
        self.mm_quant_mode = (
            self.quant_config.mm_quant_mode if self.quant_config is not None else "w16a16"
        )
        self.gmm_quant_mode = (
            self.quant_config.gmm_quant_mode if self.quant_config is not None else "w16a16"
        )
        # SP attention transport, selected by quant tier (trigger unified = sp_on):
        #   4bit-gmm  -> sp_quant: MXFP8 AG-in(+scale) / AlltoAll-out +
        #                ReplicatedLinear o_proj.
        #   otherwise -> sp_bf16: bf16 AG-in / RowParallel o_proj + RS-out. fp8/mxfp8
        #                attention linears quantize internally, so transport is bf16.
        # MXFP8 SP transport applies to MoE (sparse) layers only; dense layers use
        # the bf16 transport (their DecoderLayer feeds bf16, not an MXFP8 scale).
        _gmm_4bit = self.gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx")
        _gmm_fp8 = self.gmm_quant_mode == "w8a8float8"
        # SP / DP-TP-DP transport tier (mxfp4 and fp8): shared AllGather-in /
        # AllToAll-out + Replicated o_proj control flow; transport quant differs.
        _sp_unified = _gmm_4bit or _gmm_fp8
        _is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"
        self.sp_quant = sp_on and _sp_unified and _is_moe_layer
        self.sp_bf16 = sp_on and not self.sp_quant
        # Only mxfp4 transports a per-group scale alongside the fp8 values; fp8's
        # per-tensor scale is a global scalar (unchanged by AllGather/AllToAll),
        # so fp8 moves only the fp8 values and leaves the scale untransported.
        self.sp_scale_transport = _gmm_4bit
        self.dynamic_mx_block = 64

        self.num_heads_per_rank = max(self.num_heads // self.attn_tp_size, 1)
        self.num_kv_heads_per_rank = max(self.num_kv_heads // self.attn_tp_size, 1)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scale_fa = 1.0 / math.sqrt(self.head_dim)
        self.block_size = infer_config.scheduler_config.block_size
        self.attn_type = "FullAttention"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"

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
            quant_config=self.quant_config,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False,
        )

        self.q_norm = HYV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = HYV3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # O projection: row-parallel by default; under SP+TP sp_quant each rank
        # holds the full o_proj weight (ReplicatedLinear) and recombines via AlltoAll.
        if self.sp_quant:
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.head_dim,
                config.hidden_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.o_proj = RowParallelLinear(
                self.num_heads * self.head_dim,
                config.hidden_size,
                tp_size=self.attn_tp_size,
                tp_rank=comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0,
                bias=False,
                input_is_parallel=True,
                quant_config=self.quant_config,
                prefix=f"{prefix}.o_proj",
            )

        # KV cache placeholders bound by the framework PA manager via cache_entries.
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.k_scale_cache = torch.Tensor([])
        self.cache_unit = (self.num_kv_heads_per_rank * self.head_dim,)

        # FIA FP8 paged path is derived from config.json's kv_cache_scheme after
        # the framework parses it into config.quant_config.kv_cache_quant_mode:
        # C8/float8 KV cache requires the fused qkv_rms_norm_rope_cache_with_k_scale
        # producer so K/V cache dtype, layout, and scales stay consistent.
        self.kv_cache_quant_mode = (
            config.quant_config.kv_cache_quant_mode
            if config.quant_config is not None
            else "unquant"
        )
        kv_cache_quant_mode = self.kv_cache_quant_mode
        self.use_fia_fp8 = kv_cache_quant_mode == "float8"
        if self.use_fia_fp8:
            if self.head_dim != 128:
                raise ValueError("FIA GQA full-quant paged path requires head_dim == 128")
            if self.block_size != 128:
                raise ValueError("FIA GQA full-quant paged path requires block_size == 128")
            _ensure_qkv_fused_kscale_registered()

        if self.use_fia_fp8:
            cache_dtype = torch.float8_e4m3fn
        else:
            cache_dtype = config.torch_dtype if config.torch_dtype is not None else torch.bfloat16
        # Source branch passes cache_layout ("BnNBsD" for fia / "BnBsND" otherwise);
        # this repo's CacheEntry has no cache_layout field, so pass it only on the
        # fia path (keeps the default path byte-identical to before). Enabling
        # fia_fp8 additionally requires the framework CacheEntry to gain cache_layout.
        _kv_layout_kw = {"cache_layout": "BnNBsD"} if self.use_fia_fp8 else {}
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=self._set_k_cache,
                **_kv_layout_kw,
            ),
            CacheEntry(
                cache_name="v_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=self._set_v_cache,
                **_kv_layout_kw,
            ),
        ]

        if self.use_fia_fp8:
            self.cache_entries.append(
                CacheEntry(
                    cache_name="k_scale_cache",
                    attn_type=self.attn_type,
                    dim=1,
                    num_head=self.num_kv_heads_per_rank,
                    dtype=torch.float32,
                    block_size=self.block_size,
                    needs_block=True,
                    tensor_setter=self._set_k_scale_cache,
                    cache_layout="BnNBsD",
                )
            )
            self.register_buffer(
                "q_rotation", torch.eye(self.head_dim, dtype=torch.bfloat16), persistent=False
            )
            self.register_buffer(
                "k_rotation", torch.eye(self.head_dim, dtype=torch.bfloat16), persistent=False
            )
            self.register_buffer(
                "v_scale", torch.ones(self.num_kv_heads_per_rank, dtype=torch.float32), persistent=False
            )
            self.register_buffer(
                "fia_v_dequant_scale",
                torch.ones(self.num_kv_heads_per_rank, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "fia_p_scale", torch.ones(1, dtype=torch.float32), persistent=False
            )

    def _set_k_cache(self, tensor: torch.Tensor) -> None:
        if not self.use_fia_fp8:
            self.k_cache = tensor
            return
        # Keep unwritten FP8 cache slots deterministic for paged attention.
        tensor.zero_()
        self.k_cache = tensor

    def _set_v_cache(self, tensor: torch.Tensor) -> None:
        if not self.use_fia_fp8:
            self.v_cache = tensor
            return
        # Keep unwritten FP8 cache slots deterministic for paged attention.
        tensor.zero_()
        self.v_cache = tensor

    def _set_k_scale_cache(self, tensor: torch.Tensor) -> None:
        tensor.zero_()
        self.k_scale_cache = tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping=None,
        block_table=None,
        dp_decode: bool = False,
        cos_sin_table: Optional[torch.Tensor] = None,
        qkv_fused_cu_seq_len: Optional[torch.Tensor] = None,
        qkv_fused_actual_seq_lens: Optional[torch.Tensor] = None,
        qkv_fused_slot_mapping: Optional[torch.Tensor] = None,
        prefill_fa_actual_seq_qlen: Optional[list] = None,
        prefill_fa_actual_seq_kvlen: Optional[list] = None,
        **kwargs,
    ):
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()

        is_prefill = forward_metadata.is_prefill
        attention_mask = forward_metadata.attention_mask
        # sp_transport gates the AllGather-in / AllToAll-out + Replicated o_proj
        # path for prefill (token-shard) and decode-DP (request-shard).
        sp_transport = self.sp_quant and (is_prefill or dp_decode)

        # sp_quant feeds (hidden_states, scale) tuples for MXFP8 SP path.
        hidden_states_scale = None
        if isinstance(hidden_states, (tuple, list)):
            if len(hidden_states) == 2:
                hidden_states, hidden_states_scale = hidden_states
            else:
                hidden_states = hidden_states[0]

        if hidden_states.dim() != 2:
            raise RuntimeError("HYV3Attention expects TND packed hidden_states [TotalTokens, hidden_size].")

        q_len, h = hidden_states.size()

        if sp_transport:
            # SP+TP: AllGather the fp8 sub-shard into the full sequence (prefill:
            # token shard -> full seq; decode-DP: request shard -> full B). The
            # value AllGather is common to both tiers; mxfp4 additionally gathers
            # the per-group scale, while fp8 (per-tensor scalar) moves only the
            # values and keeps hidden_states_scale None.
            h_dtype = hidden_states.dtype
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            new_hidden_states = torch.empty([q_len * self.attn_tp_size, h], dtype=h_dtype, device="npu")
            dist.all_gather_into_tensor(new_hidden_states, hidden_states, group=attn_tp_group)
            if self.sp_scale_transport:
                s_dtype = hidden_states_scale.dtype
                new_scale = torch.empty(
                    [self.attn_tp_size * hidden_states_scale.shape[0], h // self.dynamic_mx_block, 2],
                    dtype=s_dtype, device="npu")
                dist.all_gather_into_tensor(new_scale, hidden_states_scale, group=attn_tp_group)
                hidden_states_scale = new_scale
            hidden_states = new_hidden_states
            q_len = q_len * self.attn_tp_size
        # Sequence-parallel attention: BF16 path covers attn_tp==ep even with
        # dense-TP shared MLP (shared MLP stays TP, MoE sub-seq via AG/RS).
        # Decode-DP: the dense layer (sp_quant=False -> sp_bf16=True, RowParallel
        # o_proj) reconstructs the full request batch here via BF16 AllGather, then
        # ReduceScatters back below -- mirrors the prefill dense-layer transport.
        elif self.sp_bf16 and (is_prefill or dp_decode):
            h_dtype = hidden_states.dtype
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            new_hidden_states = torch.empty(
                [q_len * self.attn_tp_size, h], dtype=h_dtype, device="npu"
            )
            dist.all_gather_into_tensor(new_hidden_states, hidden_states, group=attn_tp_group)
            hidden_states = new_hidden_states
            q_len = q_len * self.attn_tp_size

        # Merged QKV projection
        qkv = self.merged_qkv_proj(hidden_states, dynamic_scale=hidden_states_scale)

        # pad-aware FA: q_len stays the full padded length (no trim/re-pad); the SP pad
        # tail rides as a dummy segment in the padded metadata (see
        # _build_pad_aware_prefill_metadata). pad_len == 0 -> original packed path.

        if slot_mapping is None or block_table is None:
            raise RuntimeError("PagedAttention requires slot_mapping and block_table.")
        if not self.k_cache.numel() or not self.v_cache.numel():
            raise RuntimeError("PagedAttention k_cache/v_cache are not initialized.")

        if not self.use_fia_fp8:
            # Split Q, K, V
            query_states, key_states, value_states = qkv.split(
                (self.num_heads_per_rank * self.head_dim,
                 self.num_kv_heads_per_rank * self.head_dim,
                 self.num_kv_heads_per_rank * self.head_dim),
                dim=1
            )

            # Reshape to per-head TND for QK norm
            query_shape = (q_len, self.num_heads_per_rank, self.head_dim)
            key_value_shape = (q_len, self.num_kv_heads_per_rank, self.head_dim)

            query_states = self.q_norm(query_states.contiguous().view(query_shape))
            key_states = self.k_norm(key_states.contiguous().view(key_value_shape))
            value_states = value_states.view(key_value_shape)

            # RoPE via NPU fusion operator on packed TND layout.
            cos, sin = cos_sin
            query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(
                query_states, key_states, cos, sin, layout='TND'
            )

            slot_mapping_for_attn = slot_mapping[self.attn_type].view(-1)
            torch_npu.npu_scatter_nd_update_(
                self.k_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
                slot_mapping_for_attn.view(-1, 1),
                key_states,
            )
            torch_npu.npu_scatter_nd_update_(
                self.v_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
                slot_mapping_for_attn.view(-1, 1),
                value_states,
            )
        else:
            # FIA FP8 fused path: qkv_rms_norm_rope_cache_with_k_scale writes the
            # fp8 K/V cache (+ per-token K scale) and returns the fp8 Q + its scale.
            qkv_ntd = qkv.view(
                q_len, self.num_heads_per_rank + 2 * self.num_kv_heads_per_rank, self.head_dim
            )
            qkv_ntd = qkv_ntd.transpose(0, 1).contiguous()
            if (
                qkv_fused_cu_seq_len is None
                or qkv_fused_actual_seq_lens is None
                or qkv_fused_slot_mapping is None
            ):
                raise RuntimeError("fused qkv operator requires precomputed qkv fused metadata")
            cu_seq_len = qkv_fused_cu_seq_len
            actual_seq_lens = qkv_fused_actual_seq_lens
            slot_mapping_flat = qkv_fused_slot_mapping[:q_len]
            head_nums = [self.num_heads_per_rank, self.num_kv_heads_per_rank, self.num_kv_heads_per_rank]
            query_states, q_scale = torch.ops.cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
                qkv_ntd, self.q_norm.weight, self.k_norm.weight, cos_sin_table, slot_mapping_flat,
                self.k_cache, self.v_cache, self.k_scale_cache, cu_seq_len, actual_seq_lens, head_nums, "NTD",
                q_rotation=self.q_rotation,
                k_rotation=self.k_rotation,
                v_scale=self.v_scale,
                epsilon=self.q_norm.variance_epsilon,
            )
            query_states = query_states.contiguous()
            q_scale = q_scale.contiguous()
            fia_k_scale = self.k_scale_cache.squeeze(-1)
            fia_v_dequant_scale = self.fia_v_dequant_scale

        # Flash Attention. npugraph_ex decode uses torch_npu with host list lengths.
        fa_ops = torch.ops.npu

        if is_prefill and prefill_fa_actual_seq_qlen is not None and prefill_fa_actual_seq_kvlen is not None:
            actual_seq_qlen = prefill_fa_actual_seq_qlen
            actual_seq_kvlen = prefill_fa_actual_seq_kvlen
        elif not is_prefill and self.enable_npugraph_ex:
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
        else:
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv

        if self.use_fia_fp8:
            attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
                query_states, self.k_cache, self.v_cache,
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="NTD_TND",
                sparse_mode=3,
                atten_mask=attention_mask,
                softmax_scale=self.scale_fa,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                block_table=block_table[self.attn_type],
                block_size=self.block_size,
                dequant_scale_query=q_scale,
                dequant_scale_key=fia_k_scale,
                dequant_scale_value=fia_v_dequant_scale,
                quant_scale_p=self.fia_p_scale,
                query_quant_mode=3, key_quant_mode=3, value_quant_mode=2,
                out_dtype=torch.bfloat16,
            )
        else:
            attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
                query_states,
                self.k_cache.view(*self.k_cache.shape[:2], -1),
                self.v_cache.view(*self.v_cache.shape[:2], -1),
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_kv_heads_per_rank,
                input_layout="TND",
                sparse_mode=3,
                atten_mask=attention_mask,
                softmax_scale=self.scale_fa,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                block_table=block_table[self.attn_type],
                block_size=self.block_size,
            )

        attn_output = attn_output.reshape(q_len, self.num_heads_per_rank * self.head_dim)

        if self.sp_quant:
            # SP+TP: ReplicatedLinear o_proj recombined via AlltoAll. prefill
            # (token shard) and decode-DP (request shard) both take this AlltoAll
            # branch (transport quant dispatched by tier). Config validation
            # guarantees sp_quant decode always shards (bs % attn_tp == 0 and
            # >= attn_tp) for BOTH the main verify and the MTP draft head, so
            # pure-TP decode is unsupported.
            if not (is_prefill or dp_decode):
                raise RuntimeError(
                    "sp_quant decode must run dp-tp-dp (batch_size_per_dp_rank "
                    "must be a multiple of attn_tp and >= attn_tp); the pure-TP "
                    "decode fallback was removed."
                )
            # pad-aware FA: attn_output is the full padded length (divisible by attn_tp),
            # so AlltoAll needs no re-pad; dummy pad rows carry finite (dropped) values.
            attn_output, attn_output_scale = _sp_transport_quant(
                attn_output, self.gmm_quant_mode, self.o_proj)
            out_dim = attn_output.shape[-1]
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            tp = self.attn_tp_size
            # Value AllToAll is common to both tiers; only mxfp4 also reorders the
            # per-group scale. fp8 transports only the values (o_proj applies its
            # static input_scale internally).
            ao_dp = _equal_all_to_all(attn_output, attn_tp_group, tp)
            attn_output = ao_dp.view(tp, -1, out_dim).transpose(0, 1).contiguous().view(-1, tp * out_dim)
            if self.sp_scale_transport:
                if out_dim % self.dynamic_mx_block != 0:
                    raise ValueError(f"o_proj in-dim {out_dim} not divisible by {self.dynamic_mx_block}")
                as_dp = _equal_all_to_all(attn_output_scale, attn_tp_group, tp)
                attn_output_scale = as_dp.view(tp, -1, out_dim // self.dynamic_mx_block, 2) \
                    .transpose(0, 1).contiguous().view(-1, tp * (out_dim // self.dynamic_mx_block), 2)
                attn_output = self.o_proj(attn_output, dynamic_scale=attn_output_scale)
            else:
                attn_output = self.o_proj(attn_output)
            return attn_output

        # O projection with TP.
        attn_output = self.o_proj(attn_output)
        # pad-aware FA: attn_output is already the full padded length; the reduce_scatter
        # below shards padded_q_len // attn_tp == this rank's token shard, no re-pad.

        # Reduce-scatter the RowParallel o_proj sum back to this rank's
        # token/request shard (token-SP prefill or dense-layer decode-DP).
        if self.sp_bf16 and (is_prefill or dp_decode):
            new_output = torch.empty(
                [q_len // self.attn_tp_size, h],
                dtype=attn_output.dtype, device="npu"
            )
            dist.reduce_scatter_tensor(
                new_output, attn_output,
                group=self.comm_manager.get_group("attn_tp_group")
            )
            attn_output = new_output
        elif self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))

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
        fuse_gate_up: bool = False,
        quantize_gate_up: bool = True,
        enable_swiglu_group_quant: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.dense_tp_size = dense_tp_size
        self.dense_tp_group = dense_tp_group
        self.comm_manager = comm_manager
        # HYV3 checkpoints may mix quantized MLPs with explicitly ignored bf16
        # dense FFNs. Keep quant_config on normal linears, while allowing layer0
        # dense gate/up to be fused as an unquantized MatMulV3.
        quant_config = getattr(config, "quant_config", None)
        gate_up_quant_config = quant_config if quantize_gate_up else None
        self.fuse_gate_up = fuse_gate_up and (
            gate_up_quant_config is not None or not quantize_gate_up
        )
        self.enable_swiglu_group_quant = (
            enable_swiglu_group_quant
            and self.fuse_gate_up
            and gate_up_quant_config is not None
        )

        if dense_tp_size > 1:
            if comm_manager is not None:
                tp_rank = comm_manager.get_rank("dense_tp_group")
            else:
                tp_rank = dist.get_rank(dense_tp_group)
        else:
            tp_rank = 0

        if self.fuse_gate_up:
            if gate_up_quant_config is not None:
                quant_config.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
            self.gate_up_proj = MergedColumnParallelLinear(
                self.hidden_size, [self.intermediate_size, self.intermediate_size],
                bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
                quant_config=gate_up_quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size,
                bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size, self.intermediate_size,
                bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size,
            bias=False, tp_size=dense_tp_size, tp_rank=tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # npu_swiglu_group_quant emits an e8m0 (MX) per-token scale, which the
        # per-tensor static fp8 down_proj cannot consume (its aclnnQuantMatmul
        # requires a float32 pertoken scale). Disable the fused swiglu-quant for
        # per-tensor fp8 and fall back to native npu_swiglu + the static
        # input_scale path (the scheme the checkpoint is calibrated for). Block
        # tiers (mxfp8/mxfp4) consume the e8m0 dynamic scale correctly and keep
        # the fusion.
        if getattr(self.down_proj.quant_method, "is_per_tensor", False):
            self.enable_swiglu_group_quant = False

    def forward(
        self,
        x,
        sequence_parallel: bool = False,
        use_swiglu_group_quant: bool = False,
    ):
        local_tokens = x.shape[0]
        gather_tokens = sequence_parallel and self.dense_tp_size > 1
        if gather_tokens:
            if x.dim() != 2:
                raise RuntimeError("HYV3MLP sequence-parallel path expects [tokens, hidden].")
            # Token-SP ranks own different sequence slices. Gather before dense
            # TP MLP so RowParallel partials correspond to the same tokens.
            gathered_x = torch.empty(
                [local_tokens * self.dense_tp_size, x.shape[-1]],
                dtype=x.dtype,
                device=x.device,
            )
            dist.all_gather_into_tensor(gathered_x, x, group=self.dense_tp_group)
            x = gathered_x

        if self.fuse_gate_up:
            merged = self.gate_up_proj(x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            merged = torch.cat([gate, up], dim=-1)

        if self.enable_swiglu_group_quant and use_swiglu_group_quant:
            act, act_scale, _ = torch.ops.custom.npu_swiglu_group_quant(
                merged,
                dst_type=torch.float8_e4m3fn,
                round_scale=True,
                quant_mode=1,
            )
            down = self.down_proj(act, dynamic_scale=act_scale)
        else:
            act = torch_npu.npu_swiglu(merged)
            down = self.down_proj(act)
        if self.dense_tp_size > 1:
            if gather_tokens:
                # Sum dense-TP partials and scatter the token dimension back.
                reduced = torch.empty(
                    [local_tokens, down.shape[-1]],
                    dtype=down.dtype,
                    device=down.device,
                )
                dist.reduce_scatter_tensor(reduced, down, group=self.dense_tp_group)
                down = reduced
            else:
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
# MoE Block (Router + Experts + Shared Expert) — with EP routing
# ---------------------------------------------------------------------------

class HYV3MoE(nn.Module):
    """MoE block with EP routing: replicated router, EP-split experts, parallelized shared expert."""

    def __init__(
        self,
        config: HYV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        shared_mlp_stream=None,
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
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.comm_manager = comm_manager
        self.quant_config = getattr(config, "quant_config", None)
        self.gmm_quant_mode = (
            self.quant_config.gmm_quant_mode if self.quant_config is not None else "w16a16"
        )
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.enable_sp = _sp_enabled(infer_config)
        # sp_on / sp_ep_aligned gate the SP *transport layout* only (attn_tp>1
        # token-shard, fp32/bf16 gate split, shared-expert sequence parallel).
        # attn_tp>1 SP is the *mechanism* that makes the ep group hold a 1/ep token
        # shard, not the root condition for AG dispatch; kept untouched below.
        sp_on = self.enable_sp and self.attn_tp_size > 1
        self.sp_ep_aligned = sp_on and (self.attn_tp_size == self.moe_ep_size)

        # AG (AllGather) prefill dispatch precondition -- a true predicate, not the
        # sp_ep_aligned proxy. AG broadcasts every token to all ep cards, routes the
        # local experts, then ReduceScatter back. It is correct AND cheaper than a
        # targeted AllToAll only when all three hold:
        #   1. partitioned_over_ep: each ep card holds a non-overlapping 1/ep token
        #      shard (not a TP replica). attn_tp>1 -> via SP token-shard when
        #      attn_tp==ep; attn_tp==1 -> via attn_dp batch-shard when attn_dp==ep
        #      (no such config today, encoded for correctness).
        #   2. attn_tp==ep: the ep gather group coincides with the token-shard group.
        #   3. top_k>ep: each token routes to > ep experts, so it lands on ~all ep
        #      cards anyway -> full-token broadcast is not wasteful vs a device-host
        #      synced AllToAll (wiki moe-comm-overlap: allgather = full-token-to-every
        #      -card, simple; alltoall = per-expert split, smaller comm but needs
        #      device->host sync). top_k<ep favours AllToAll.
        if self.attn_tp_size > 1:
            partitioned_over_ep = sp_on and (self.attn_tp_size == self.moe_ep_size)
        else:
            partitioned_over_ep = self.attn_dp_size == self.moe_ep_size
        self.moe_ag_dispatch = (
            self.moe_ep_size > 1
            and partitioned_over_ep
            and (self.attn_tp_size == self.moe_ep_size)
            and (self.top_k > self.moe_ep_size)
        )

        ep_rank = comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        tp_rank = comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0
        self.enable_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", False
        )
        self.exe_mode = infer_config.model_config.exe_mode
        self.shared_mlp_stream = shared_mlp_stream
        if self.enable_multi_streams and self.shared_mlp_stream is None:
            self.shared_mlp_stream = create_stream("11", self.exe_mode)
        self.npu_events = tuple(
            create_event(self.exe_mode, self.enable_multi_streams) for _ in range(2)
        )
        # Maximum token count per MoE chunk during prefill.
        # Splits the expert dispatch/combine along the sequence dimension to
        # bound peak memory of the expanded_x / gathered_tokens intermediates
        # (each ~ hidden * top_k per token).  Default 65536: effectively no
        # chunking for typical prefill lengths; set to e.g. 4096 to reduce peak.
        self.moe_chunk_max_len = infer_config.model_config.custom_params.get(
            "moe_chunk_max_len", 65536
        )

        # Router: replicated (all ranks have full router)
        self.router = HYV3TopKRouter(config)

        # Expert bias: replicated
        self.register_buffer("expert_bias", torch.zeros(config.num_local_experts))

        # Routed experts use the common parameter layout and weight loader.
        self.experts = FusedMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=config.moe_intermediate_size,
            bias=False,
            quant_config=self.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=tp_rank,
            ep_size=self.moe_ep_size,
            ep_rank=ep_rank,
            prefix=f"{prefix}.experts",
        )
        # Hy3 keeps the checkpoint expert layout and validated split_item=2 GMM path.
        self.experts.split_item = 2

        # Shared expert: parallelized via dense_tp
        shared_intermediate = config.moe_intermediate_size * config.num_shared_experts
        dense_tp_group = comm_manager.get_group("dense_tp_group") if self.dense_tp_size > 1 else None
        self.shared_mlp = HYV3MLP(
            config, intermediate_size=shared_intermediate,
            dense_tp_size=self.dense_tp_size, dense_tp_group=dense_tp_group,
            prefix=f"{prefix}.shared_mlp",
            fuse_gate_up=True,
            enable_swiglu_group_quant=True,
        )

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool = False) -> torch.Tensor:
        # sp_ep_aligned feeds (fp32-for-gate, bf16-for-experts); gate keeps fp32 precision.
        if self.sp_ep_aligned and isinstance(hidden_states, (tuple, list)):
            hidden_states_fp32, hidden_states_bf16 = hidden_states
            hidden_states = hidden_states_bf16
            hidden_states_gate = hidden_states_fp32
        else:
            hidden_states_gate = hidden_states
        origin_shape = hidden_states.shape
        hidden_dim = origin_shape[-1]
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        hidden_states_gate_flat = hidden_states_gate.reshape(-1, hidden_dim)

        shared_sequence_parallel = (
            self.sp_ep_aligned
            and is_prefill
            and self.dense_tp_size == self.attn_tp_size
        )
        enable_shared_mlp_multi_streams = (
            self.enable_multi_streams
            and self.dense_tp_size == 1
            and (not is_prefill or not shared_sequence_parallel)
        )
        shared_output = None

        # Router must see all tokens for correct top-k selection (not chunked).
        _, top_k_weights, top_k_index, _ = self.router(hidden_states_gate_flat, self.expert_bias)
        top_k_index = top_k_index.to(torch.int32)

        if enable_shared_mlp_multi_streams:
            shared_output = self._forward_shared_mlp(
                hidden_states_flat, shared_sequence_parallel, enable_shared_mlp_multi_streams,
                use_swiglu_group_quant=is_prefill,
            )

        num_tokens = hidden_states_flat.shape[0]
        if self.moe_ag_dispatch and is_prefill:
            if self.gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx"):
                routed_output = self.moe_infer_ag_w4a8mx(
                    hidden_states_flat, top_k_index, top_k_weights
                )
            else:
                routed_output = self.moe_infer_ag(
                    hidden_states_flat, top_k_index, top_k_weights
                )
        elif is_prefill:
            # Split expert dispatch/combine along the token dim to bound
            # peak memory of expanded_x / gathered_tokens intermediates.
            routed_output_list = []
            for chunk_x, chunk_ids, chunk_weights in zip(
                *self._split_tensors(num_tokens, hidden_states_flat,
                                     top_k_index, top_k_weights)
            ):
                if self.moe_ep_size > 1:
                    chunk_out = self._moe_ep_manual(chunk_x, chunk_ids, chunk_weights)
                else:
                    chunk_out = self._moe_local_or_tp(chunk_x, chunk_ids, chunk_weights)
                routed_output_list.append(chunk_out)
            routed_output = (torch.cat(routed_output_list, dim=0)
                             if len(routed_output_list) > 1 else routed_output_list[0])
        elif self.moe_ep_size > 1 and self.moe_tp_size == 1:
            routed_output = self._moe_ep_mc2_decode(
                hidden_states_flat, top_k_index, top_k_weights
            )
        elif self.moe_ep_size > 1:
            routed_output = self._moe_ep_manual(
                hidden_states_flat, top_k_index, top_k_weights
            )
        else:
            routed_output = self._moe_local_or_tp(
                hidden_states_flat, top_k_index, top_k_weights
            )

        if shared_output is not None:
            wait_event(enable_shared_mlp_multi_streams, self.npu_events, 1, exe_mode=self.exe_mode)
        else:
            shared_output = self.shared_mlp(
                hidden_states_flat, sequence_parallel=shared_sequence_parallel,
                use_swiglu_group_quant=is_prefill,
            )

        if self.enable_moe_fp32_combine:
            hidden_states_out = (
                routed_output.float() + shared_output.float()
            ).to(hidden_states_flat.dtype)
        else:
            hidden_states_out = routed_output + shared_output

        return hidden_states_out.reshape(origin_shape)

    def _forward_shared_mlp(
        self,
        hidden_states_flat,
        sequence_parallel,
        enable_multi_streams,
        use_swiglu_group_quant,
    ):
        record_stream(
            enable_multi_streams,
            hidden_states_flat,
            self.shared_mlp_stream,
            exe_mode=self.exe_mode,
        )
        record_event(enable_multi_streams, self.npu_events, 0, exe_mode=self.exe_mode)
        with npu_stream_switch(enable_multi_streams, self.shared_mlp_stream, exe_mode=self.exe_mode):
            wait_event(enable_multi_streams, self.npu_events, 0, exe_mode=self.exe_mode)
            shared_output = self.shared_mlp(
                hidden_states_flat,
                sequence_parallel=sequence_parallel,
                use_swiglu_group_quant=use_swiglu_group_quant,
            )
            record_event(enable_multi_streams, self.npu_events, 1, exe_mode=self.exe_mode)
        return shared_output

    def _split_tensors(self, num_tokens, hidden_states_flat, topk_ids, topk_weight):
        """Chunk MoE inputs along the token dimension to bound peak memory.

        When num_tokens > self.moe_chunk_max_len, splits each tensor into
        ceil(num_tokens / moe_chunk_max_len) chunks so that each chunk's
        expanded_x / gathered_tokens intermediate stays within a predictable
        size (~ moe_chunk_max_len * top_k * hidden_dim).
        """
        if num_tokens > self.moe_chunk_max_len:
            num_chunks = (num_tokens + self.moe_chunk_max_len - 1) // self.moe_chunk_max_len
            x_list = hidden_states_flat.chunk(num_chunks, dim=0)
            topk_ids_list = topk_ids.chunk(num_chunks, dim=0)
            topk_weight_list = topk_weight.chunk(num_chunks, dim=0)
        else:
            x_list = [hidden_states_flat]
            topk_ids_list = [topk_ids]
            topk_weight_list = [topk_weight]
        return x_list, topk_ids_list, topk_weight_list

    def _moe_local_or_tp(self, hidden_states_flat, topk_ids, topk_weight):
        """Route tokens to local experts and run the common GMM expert kernel."""
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states_flat,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )
        expert_output = self.experts(
            expanded_x, tokens_per_expert, group_list_type=1)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            expert_output, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(expert_output.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_tp_group"))
        return hidden_states

    def _moe_ep_manual(self, hidden_states_flat, topk_ids, topk_weight):
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

        hidden_states_ordered = self.experts(hidden_states_ordered, tokens_per_local_expert, group_list_type=1)

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

    def moe_infer_ag_w4a8mx(self, hidden_states, topk_ids, topk_weight):
        """Prefill EP via AllGather + ReduceScatter (W4A8 MXFP4 experts).

        Each rank quantizes its sub-sequence to MXFP8, AllGathers x/scale/topk,
        routes only local experts, runs GMM, finalizes, then ReduceScatters back
        to its sub-sequence. Cheaper than double-routing AlltoAll when top_k > ep.
        """
        bs_qlen, h = hidden_states.shape
        h_dtype = hidden_states.dtype
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        ep_rank = dist.get_rank(moe_ep_group)
        total_tokens = bs_qlen * self.attn_tp_size

        hidden_states_mx, mx_scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn)
        x = torch.empty([total_tokens, h], dtype=hidden_states_mx.dtype, device="npu")
        dist.all_gather_into_tensor(x, hidden_states_mx, group=moe_ep_group)
        x_scale = torch.empty(
            [total_tokens, mx_scale.shape[-2], mx_scale.shape[-1]],
            dtype=mx_scale.dtype, device="npu")
        dist.all_gather_into_tensor(x_scale, mx_scale, group=moe_ep_group)
        ids_ag = torch.empty([total_tokens, topk_ids.shape[1]], dtype=topk_ids.dtype, device="npu")
        dist.all_gather_into_tensor(ids_ag, topk_ids, group=moe_ep_group)
        topk_ids = ids_ag
        w_ag = torch.empty([total_tokens, topk_weight.shape[1]], dtype=topk_weight.dtype, device="npu")
        dist.all_gather_into_tensor(w_ag, topk_weight, group=moe_ep_group)
        topk_weight = w_ag

        active_num = topk_ids.shape[0] * topk_ids.shape[1]
        routing_kwargs = dict(
            expert_idx=topk_ids, active_num=active_num, expert_num=self.num_experts,
            expert_tokens_num_type=1, expert_tokens_num_flag=True,
            active_expert_range=[ep_rank * self.experts_per_rank,
                                 (ep_rank + 1) * self.experts_per_rank],
            quant_mode=-1,
        )
        expanded_x, expanded_row_idx, tokens_per_expert, _ = \
            torch_npu.npu_moe_init_routing_v2(x.view(torch.bfloat16), **routing_kwargs)
        expanded_x = expanded_x.view(x.dtype)
        x_scale_bf16 = x_scale.reshape(total_tokens, -1).to(torch.bfloat16)
        exp_scale_bf16, _, _, _ = torch_npu.npu_moe_init_routing_v2(x_scale_bf16, **routing_kwargs)
        pertoken_scale = exp_scale_bf16.to(x_scale.dtype).view(-1, *x_scale.shape[1:])

        ordered = self.experts(expanded_x, tokens_per_expert, group_list_type=1, pertoken_scale=pertoken_scale)
        # subset expert range emits -1 for non-local (token,k); same -1 fix as moe_infer_ag.
        lo = ep_rank * self.experts_per_rank
        hi = (ep_rank + 1) * self.experts_per_rank
        local_mask = ((topk_ids >= lo) & (topk_ids < hi)).to(ordered.dtype)
        scales_local = topk_weight.to(ordered.dtype) * local_mask
        row_idx_safe = torch.where(
            expanded_row_idx < 0, torch.zeros_like(expanded_row_idx), expanded_row_idx)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            ordered, skip1=None, skip2=None, bias=None, scales=scales_local,
            expanded_src_to_dst_row=row_idx_safe, export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )
        rs = torch.empty([bs_qlen, h], dtype=h_dtype, device="npu")
        dist.reduce_scatter_tensor(rs, hidden_states, group=moe_ep_group)
        return rs.view(bs_qlen, self.hidden_dim)

    def moe_infer_ag(self, hidden_states, topk_ids, topk_weight):
        """Prefill EP via AllGather + ReduceScatter (BF16 experts, no quant).

        Each rank AllGathers its sub-sequence x/topk into the full sequence,
        routes only its local experts (quant_mode=-1), runs GMM, finalizes, then
        ReduceScatters routed output back to its sub-sequence. Shared expert is
        added by forward(); this returns the routed contribution only.
        """
        bs_qlen, h = hidden_states.shape
        h_dtype = hidden_states.dtype
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        ep_rank = dist.get_rank(moe_ep_group)
        total_tokens = bs_qlen * self.attn_tp_size

        x = torch.empty([total_tokens, h], dtype=h_dtype, device="npu")
        dist.all_gather_into_tensor(x, hidden_states, group=moe_ep_group)
        ids_ag = torch.empty([total_tokens, topk_ids.shape[1]], dtype=topk_ids.dtype, device="npu")
        dist.all_gather_into_tensor(ids_ag, topk_ids, group=moe_ep_group)
        topk_ids = ids_ag
        w_ag = torch.empty([total_tokens, topk_weight.shape[1]], dtype=topk_weight.dtype, device="npu")
        dist.all_gather_into_tensor(w_ag, topk_weight, group=moe_ep_group)
        topk_weight = w_ag

        expanded_x, expanded_row_idx, tokens_per_expert, _ = \
            torch_npu.npu_moe_init_routing_v2(
                x,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=None,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[ep_rank * self.experts_per_rank,
                                     (ep_rank + 1) * self.experts_per_rank],
                quant_mode=-1,
            )
        ordered = self.experts(expanded_x, tokens_per_expert, group_list_type=1)
        # subset expert range: init_routing emits -1 for non-local (token,k), and this
        # torch_npu OOB-gathers on -1 in finalize (~10x blowup). Zero non-local scales
        # + clamp -1->0 so they add 0; ReduceScatter sums local parts = full output.
        lo = ep_rank * self.experts_per_rank
        hi = (ep_rank + 1) * self.experts_per_rank
        local_mask = ((topk_ids >= lo) & (topk_ids < hi)).to(ordered.dtype)
        scales_local = topk_weight.to(ordered.dtype) * local_mask
        row_idx_safe = torch.where(
            expanded_row_idx < 0, torch.zeros_like(expanded_row_idx), expanded_row_idx)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            ordered, skip1=None, skip2=None, bias=None,
            scales=scales_local,
            expanded_src_to_dst_row=row_idx_safe, export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )
        rs = torch.empty([bs_qlen, h], dtype=h_dtype, device="npu")
        dist.reduce_scatter_tensor(rs, hidden_states, group=moe_ep_group)
        return rs.view(bs_qlen, self.hidden_dim)

    def _moe_ep_mc2_decode(self, hidden_states_flat, topk_ids, topk_weight):
        """MC2 dispatch/combine for decode — graph-compatible, no data-dependent splits."""
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
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

        expert_output = self.experts(
            expand_x, expert_token_num, group_list_type=1,
        )

        output = torch_npu.npu_moe_distribute_combine_v2(
            expert_output, topk_ids, expand_idx,
            ep_recv_counts, topk_weight.to(torch.float32),
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
        shared_mlp_stream=None,
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
                shared_mlp_stream=shared_mlp_stream,
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
                fuse_gate_up=True,
                quantize_gate_up=True,
            )

        self.input_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        quant_config = getattr(config, "quant_config", None)
        gmm_quant_mode = quant_config.gmm_quant_mode if quant_config is not None else "w16a16"
        self.gmm_quant_mode = gmm_quant_mode
        attn_tp = infer_config.parallel_config.attn_tp_size
        # sp_quant (MXFP8 fused norm feeding the sharded attention) is the 4bit SP
        # transport variant: SP-on (enable_sp AND attn_tp>1) AND 4bit-gmm. No attn_dp.
        sp_on = _sp_enabled(infer_config) and attn_tp > 1
        self.sp_quant = (sp_on
                            and gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx", "w8a8float8")
                            and isinstance(self.mlp, HYV3MoE))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        past_residual: Optional[torch.Tensor] = None,
        slot_mapping=None,
        block_table=None,
        dp_decode: bool = False,
        cos_sin_table: Optional[torch.Tensor] = None,
        qkv_fused_cu_seq_len: Optional[torch.Tensor] = None,
        qkv_fused_actual_seq_lens: Optional[torch.Tensor] = None,
        qkv_fused_slot_mapping: Optional[torch.Tensor] = None,
        prefill_fa_actual_seq_qlen: Optional[list] = None,
        prefill_fa_actual_seq_kvlen: Optional[list] = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill if forward_metadata else False
        # attn_quant: quantize the input-norm output for the sharded AllGather-in
        # transport (prefill SP and decode-DP).
        attn_quant = self.sp_quant and (is_prefill or dp_decode)
        # moe_fp32_gate: the fp32 router gate is prefill-only; decode-DP keeps the
        # bf16 gate (a fp32 gate can select different experts).
        moe_fp32_gate = self.sp_quant and is_prefill
        if attn_quant:
            # (add-)RMSNorm + transport quant for the attention input.
            # mxfp4: fused norm+MXFP8 op. fp8: two-step (npu_quantize has no
            # mx-fused op).
            if self.gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx"):
                if past_residual is not None:
                    hidden_mx, residual, hidden_scale, _ = torch_npu.npu_add_rms_norm_dynamic_mx_quant(
                        hidden_states, past_residual, self.input_layernorm.weight,
                        beta=None, epsilon=self.input_layernorm.variance_epsilon,
                        round_mode='rint', dst_type=torch.float8_e4m3fn)
                else:
                    residual = hidden_states
                    hidden_mx, hidden_scale, _ = torch_npu.npu_rms_norm_dynamic_mx_quant(
                        hidden_states, self.input_layernorm.weight,
                        beta=None, epsilon=self.input_layernorm.variance_epsilon,
                        round_mode='rint', dst_type=torch.float8_e4m3fn)
            else:
                if past_residual is not None:
                    hidden_norm, _, residual = torch_npu.npu_add_rms_norm(
                        hidden_states, past_residual, self.input_layernorm.weight,
                        self.input_layernorm.variance_epsilon)
                else:
                    residual = hidden_states
                    hidden_norm, _ = torch_npu.npu_rms_norm(
                        hidden_states, self.input_layernorm.weight,
                        self.input_layernorm.variance_epsilon)
                hidden_mx, hidden_scale = _sp_transport_quant(
                    hidden_norm, self.gmm_quant_mode, self.self_attn.merged_qkv_proj)
            attn_in = (hidden_mx, hidden_scale)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
            attn_in = hidden_states

        hidden_states = self.self_attn(
            hidden_states=attn_in,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            slot_mapping=slot_mapping,
            block_table=block_table,
            dp_decode=dp_decode,
            cos_sin_table=cos_sin_table,
            qkv_fused_cu_seq_len=qkv_fused_cu_seq_len,
            qkv_fused_actual_seq_lens=qkv_fused_actual_seq_lens,
            qkv_fused_slot_mapping=qkv_fused_slot_mapping,
            prefill_fa_actual_seq_qlen=prefill_fa_actual_seq_qlen,
            prefill_fa_actual_seq_kvlen=prefill_fa_actual_seq_kvlen,
        )

        if moe_fp32_gate:
            # MoE needs fp32 for gate + bf16 for experts; fuse residual-add+norm+cast.
            hidden_fp32, hidden_bf16, _, residual = torch_npu.npu_add_rms_norm_cast(
                hidden_states, residual, self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon)
            hidden_states = self.mlp((hidden_fp32, hidden_bf16), is_prefill=is_prefill)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            if isinstance(self.mlp, HYV3MoE):
                hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)
            else:
                hidden_states = self.mlp(
                    hidden_states,
                    sequence_parallel=(
                        is_prefill
                        and self.self_attn.sp_bf16
                        and self.self_attn.dense_tp_size == self.self_attn.attn_tp_size
                    ),
                )

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
        # MTP needs the full-prompt hidden [T,H]; baseline (next_n=0) keeps last-token only.
        self.next_n = infer_config.model_config.next_n if infer_config is not None else 0

        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.enable_sp = _sp_enabled(infer_config)
        sp_on = self.enable_sp and self.attn_tp_size > 1
        _qc = getattr(config, "quant_config", None)
        _gmm = _qc.gmm_quant_mode if _qc is not None else "w16a16"
        # Token-shard SP (sp_shard in forward): driven by sp_on, transport by tier.
        # No attn_dp -- the attention AllGathers the sharded hidden back to the full
        # sequence before touching the framework KV ledger, so full-sequence metadata
        # (which holds at attn_dp=1 too) is exactly what SP needs.
        _gmm_4bit = _gmm in ("w4a8mxfloat4", "w4a8mx")
        # Unified SP / DP-TP-DP tier: mxfp4 (dynamic MXFP8) + fp8 (static
        # per-tensor) share the model-level sp_quant gate that drives
        # _dp_decode_active; the per-tier transport differs downstream.
        _sp_unified = _gmm_4bit or _gmm == "w8a8float8"
        self.sp_quant = sp_on and _sp_unified
        self.sp_bf16 = sp_on and not _sp_unified
        # DP-TP-DP decode (sp_quant tiers) shards the decode batch along the
        # request axis and only supports multi-batch. Validate the PER-ATTN-DP-
        # GROUP decode batch = batch_size_per_dp_rank (== batch_size // attn_dp),
        # NOT the raw batch_size -- decode's static graph is compiled at this per
        # -dp-rank shape, and dp-tp-dp shards exactly this per-group batch along
        # attn_tp. At attn_dp==1 the two coincide. Single-request / bs<attn_tp
        # are rejected (pure-TP decode is unsupported). MTP shares this
        # gate: next_n scales rows, not request count.
        if self.enable_sp:
            # hy3 dp-tp-dp shards the decode batch within a single attn_tp group
            # (request axis). It is only validated/adapted for attn_dp==1 (one TP
            # group spanning all ranks); attn_dp>1 (multi-group) is not supported.
            if self.attn_dp_size != 1:
                raise ValueError(
                    "hy3 dp-tp-dp 只支持 attn_dp==1(单TP组);attn_dp>1 未适配 "
                    f"(attn_dp_size={self.attn_dp_size})"
                )
        if self.sp_quant and infer_config is not None:
            _bs = infer_config.scheduler_config.batch_size_per_dp_rank
            if _bs < self.attn_tp_size or _bs % self.attn_tp_size != 0:
                raise ValueError(
                    "A5 dp-tp-dp decode requires batch_size_per_dp_rank"
                    f"(=batch_size//attn_dp={_bs}) to be a positive integer "
                    f"multiple of attn_tp(={self.attn_tp_size}); current "
                    f"batch_size_per_dp_rank={_bs}, attn_tp={self.attn_tp_size}, "
                    f"attn_dp={self.attn_dp_size} does not satisfy it."
                )
        self.comm_manager = comm_manager
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", False
        )
        self.shared_mlp_stream = (
            create_stream("11", self.exe_mode) if self.enable_multi_streams else None
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
                              shared_mlp_stream=self.shared_mlp_stream,
                              prefix=f"{prefix}.layers.{layer_idx}")
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV3RotaryEmbedding(
            config=config, max_position_embeddings=self.max_position_embeddings
        )
        # FIA fused qkv path needs the packed cos_sin_table; derive from the layers.
        self.use_fia_fp8 = any(layer.self_attn.use_fia_fp8 for layer in self.layers)
        self.qkv_fused_attn_type = next(
            (layer.self_attn.attn_type for layer in self.layers if layer.self_attn.use_fia_fp8),
            None,
        )

    def _dp_decode_active(self, is_prefill, token_count):
        """DP-TP-DP decode gate: shard the decode batch along the REQUEST axis
        across the attn_tp group. Active for the sp_quant tiers (mxfp4, fp8).

        Decode packs (next_n+1) rows per request (next_n=0 -> 1 row). The shard
        is per request block: only bs (request count) must be a multiple of
        attn_tp -- then the row-major request layout keeps every (next_n+1)-row
        block on one card (equal all_to_all_single splits on request-block
        boundaries), so MTP (next_n>0) uses the same path, not a pure-TP fallback.
        """
        if is_prefill or self.attn_tp_size <= 1:
            return False
        if not self.sp_quant:
            return False
        q_len = self.next_n + 1
        batch = token_count // q_len  # request count (rows are next_n+1 per req)
        return batch >= self.attn_tp_size and batch % self.attn_tp_size == 0

    @staticmethod
    def _build_prefill_fa_actual_seq_lengths(forward_metadata: ForwardMetaData):
        if forward_metadata.actual_seq_lengths_cu_q is None or forward_metadata.actual_seq_lengths_kv is None:
            raise RuntimeError("FA prefill requires actual_seq_lengths_cu_q and actual_seq_lengths_kv")

        if forward_metadata.actual_seq_lengths_cu_q.numel() == 1:
            prompt_tokens = int(forward_metadata.prompt_tokens)
            return [prompt_tokens], [prompt_tokens]

        actual_seq_qlen = [
            int(length) for length in forward_metadata.actual_seq_lengths_cu_q.detach().cpu().tolist()
        ]
        actual_seq_kvlen = [
            int(length) for length in forward_metadata.actual_seq_lengths_kv.detach().cpu().tolist()
        ]
        if len(actual_seq_qlen) != len(actual_seq_kvlen):
            raise RuntimeError("FA prefill q and kv sequence length metadata must have the same batch size")
        return actual_seq_qlen, actual_seq_kvlen

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
        token_count = input_ids.shape[0]
        position_ids = position_ids.view(-1).long()
        prefill_sp = (self.sp_quant or self.sp_bf16) and is_prefill
        # decode_dp: only hidden_states is sharded to [B/tp, H]; position_ids,
        # cos_sin, slot_mapping, block_table, actual_seq_lengths_* stay full-B
        # (attention AllGathers hidden back to full B before consuming them).
        decode_dp = self._dp_decode_active(is_prefill, token_count)
        tp_dp_active = prefill_sp or decode_dp

        # bug#2: sharding input_ids BEFORE the vocab-parallel embedding all_reduce
        # (over embed_tp_group == attn_tp ranks) mixes different tokens -> corrupt
        # embeddings from layer 0. Fix: embed on the full seq FIRST, then shard.
        sp_shard = None
        sp_pad_len = 0
        if tp_dp_active:
            rank_in_tp_group = self.comm_manager.get_rank("attn_tp_group")
            if is_prefill:
                prompt_tokens = forward_metadata.prompt_tokens
                padded_tokens = ((prompt_tokens + self.attn_tp_size - 1) // self.attn_tp_size) * self.attn_tp_size
                sp_pad_len = padded_tokens - prompt_tokens
                if padded_tokens > token_count:
                    pad = torch.zeros(
                        (padded_tokens - token_count,),
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    input_ids = torch.cat([input_ids, pad], dim=0)
                    token_count = padded_tokens
            tokens_per_rank = token_count // self.attn_tp_size
            start_idx = rank_in_tp_group * tokens_per_rank
            sp_shard = (start_idx, start_idx + tokens_per_rank)

        # Vocab-TP embedding on the full padded seq (all attn_tp ranks share it here).
        if self.embed_tp_size > 1:
            embed_tp_rank = self.comm_manager.get_rank("embed_tp_group")
            new_input_ids = input_ids - embed_tp_rank * (self.vocab_size // self.embed_tp_size)
            mask = (new_input_ids >= 0) & (new_input_ids < (self.vocab_size // self.embed_tp_size))
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        # SP: slice this rank's token chunk from the embedded full sequence.
        if sp_shard is not None:
            inputs_embeds = inputs_embeds[sp_shard[0]:sp_shard[1]]

        hidden_states = inputs_embeds

        block_table = forward_metadata.block_table
        slot_mapping = forward_metadata.slot_mapping

        # pad-aware FA: carry the SP pad as a dummy segment (see
        # _build_pad_aware_prefill_metadata); keep the ORIGINAL forward_metadata (real
        # cu_q) for the output-tail selection that drops the dummy. pad_len == 0 -> no-op.
        padded_forward_metadata = forward_metadata
        if sp_pad_len > 0:
            position_ids = torch.cat([position_ids, position_ids.new_zeros(sp_pad_len)])
            padded_forward_metadata, slot_mapping, block_table = _build_pad_aware_prefill_metadata(
                forward_metadata, slot_mapping, block_table, sp_pad_len, prompt_tokens
            )

        cos_sin = self.rotary_emb(hidden_states, position_ids, self.max_position_embeddings)
        cos_sin_table = (
            self.rotary_emb.get_cos_sin_table(self.max_position_embeddings)
            if self.use_fia_fp8 else None
        )
        qkv_fused_cu_seq_len = None
        qkv_fused_actual_seq_lens = None
        qkv_fused_slot_mapping = None
        prefill_fa_actual_seq_qlen = None
        prefill_fa_actual_seq_kvlen = None
        if is_prefill:
            prefill_fa_actual_seq_qlen, prefill_fa_actual_seq_kvlen = self._build_prefill_fa_actual_seq_lengths(
                padded_forward_metadata
            )
        if self.use_fia_fp8:
            if (
                padded_forward_metadata.actual_seq_lengths_cu_q is None
                or padded_forward_metadata.actual_seq_lengths_kv is None
            ):
                raise RuntimeError("fused qkv operator requires actual_seq_lengths_cu_q and actual_seq_lengths_kv")
            if slot_mapping is None or self.qkv_fused_attn_type not in slot_mapping:
                raise RuntimeError("fused qkv operator requires slot_mapping for the attention type")
            qkv_fused_cu_q = padded_forward_metadata.actual_seq_lengths_cu_q.to(torch.int32).contiguous()
            qkv_fused_actual_seq_lens = padded_forward_metadata.actual_seq_lengths_kv.to(torch.int32).contiguous()
            qkv_fused_cu_seq_len = torch.cat((qkv_fused_cu_q.new_zeros(1), qkv_fused_cu_q), dim=0)
            # padded slot_mapping already carries the dummy tail; token_count is the
            # full padded length (prompt_tokens + pad_len when SP-padded, else prompt
            # length for prefill / decode token count).
            qkv_token_count = token_count
            qkv_fused_slot_mapping = (
                slot_mapping[self.qkv_fused_attn_type]
                .view(-1)[:qkv_token_count]
                .to(torch.int32)
                .contiguous()
            )

        residual = None
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                cos_sin=cos_sin,
                forward_metadata=padded_forward_metadata,
                past_residual=residual,
                slot_mapping=slot_mapping,
                block_table=block_table,
                dp_decode=decode_dp,
                cos_sin_table=cos_sin_table,
                qkv_fused_cu_seq_len=qkv_fused_cu_seq_len,
                qkv_fused_actual_seq_lens=qkv_fused_actual_seq_lens,
                qkv_fused_slot_mapping=qkv_fused_slot_mapping,
                prefill_fa_actual_seq_qlen=prefill_fa_actual_seq_qlen,
                prefill_fa_actual_seq_kvlen=prefill_fa_actual_seq_kvlen,
                **kwargs,
            )
            residual, hidden_states = layer_outputs

        # Final Norm with fused residual add (npu_add_rms_norm)
        hidden_states, _ = self.norm(hidden_states, residual)
        cu_seq_lens_q = forward_metadata.actual_seq_lengths_cu_q
        if cu_seq_lens_q is None:
            raise RuntimeError("actual_seq_lengths_cu_q is required for HYV3 packed output.")

        if is_prefill:
            seq_index = cu_seq_lens_q - 1
            if tp_dp_active:
                q_len, hidden_size = hidden_states.size()
                gathered_hidden_states = torch.empty(
                    [self.attn_tp_size * q_len, hidden_size],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                dist.all_gather_into_tensor(
                    gathered_hidden_states,
                    hidden_states,
                    group=self.comm_manager.get_group("attn_tp_group"),
                )
                hidden_states = gathered_hidden_states
            if self.next_n > 0:
                # MTP: return full-prompt hidden [T,H]; CausalLM picks last token for logits.
                return hidden_states
            hidden_states = torch.index_select(hidden_states, 0, seq_index)
            hidden_states = hidden_states.view(seq_index.numel(), 1, hidden_states.size(-1))
        else:
            # MTP verify decode packs next_n+1 tokens per request; keep them on the
            # q_len axis as [B, next_n+1, H] so lm_head yields [B, next_n+1, vocab]
            # for token-level verify. next_n=0 stays single-token [B, 1, H].
            q_len = self.next_n + 1
            bs = hidden_states.shape[0] // q_len
            hidden_states = hidden_states.view(bs, q_len, hidden_states.shape[-1])

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
        self.next_n = infer_config.model_config.next_n
        self.enable_sp = _sp_enabled(infer_config)

        self.init_parallel_comm_group()
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
        if self.dense_tp_size > 1:
            self.comm_manager.register_group(
                name="dense_tp_group",
                group_num=self.world_size // self.dense_tp_size,
                group_size=self.dense_tp_size,
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
            if self.moe_tp_size == 1:
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
            if isinstance(module, HYV3Attention) and module.use_fia_fp8:
                module.q_norm.weight.data = module.q_norm.weight.data.to(torch.float32)
                module.k_norm.weight.data = module.k_norm.weight.data.to(torch.float32)
                module.fia_v_dequant_scale.copy_(torch.reciprocal(module.v_scale))
            qm = getattr(module, "quant_method", None)
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                is_moe_gmm = isinstance(module, FusedMoEGMM)
                # mxfp4 W4A8 (x=fp8 / w=fp4) GMM is only supported by the NZ
                # grouped-matmul kernel (AclnnGroupedMatmulNz); the ND kernel
                # (GroupedMatmulV4) rejects any fp4 weight -> 161002. So mxfp4
                # experts MUST keep NZ weight, independent of the 950 global
                # weight-NZ gate. fp8/mxfp8 experts use the ND grouped matmul,
                # which on 950 rejects NZ fp8, so they stay is_nz=False.
                moe_needs_nz = is_moe_gmm and isinstance(qm, W4A8MxFp4MoEGMMMethod)
                # Experts transpose weights at load (framework default); GE folds
                # the constant transpose so decode does not re-transpose per step.
                qm.process_weights_after_loading(
                    module,
                    is_transpose=True,
                    is_nz=(
                        True if moe_needs_nz
                        else (False if is_moe_gmm
                              else self.infer_config.model_config.enable_weight_nz)
                    ),
                )
                if is_moe_gmm:
                    # Keep the original Hy3 expert parameter state; freezing these
                    # large tensors changes GE compile behavior for this model.
                    # mxfp4 ND keeps uint8-packed float4 weights (integer dtype
                    # cannot require grad); only float experts (bf16/fp8/mxfp8)
                    # need the grad flag, so guard on floating-point dtype.
                    if module.w13_weight.is_floating_point():
                        module.w13_weight.requires_grad_(True)
                    if module.w2_weight.is_floating_point():
                        module.w2_weight.requires_grad_(True)

    def get_cache_info(self) -> ModelCacheInfo:
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
            layer_infos=layer_infos,
            is_mla_backend=False,
        )

    def load_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None]) -> Set[str]:
        """Load weights from the checkpoint iterator.

        Handles:
        - EP filtering and packed expert loading via FusedMoEGMM
        - TP-split weights via ParallelLinear weight_loader
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params: Set[str] = set()
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        for name, loaded_weight in weights:
            if any(pattern in name for pattern in self._ignore_weights_patterns):
                continue

            if name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
                loaded_params.add(name)
                continue

            is_expert_weight = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    break
                param = params_dict[name_mapped]
                param.weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_mapped)
                break
            if is_expert_weight:
                continue

            # Handle QKV projection weights
            qkv_match = re.match(
                r"(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj)\.(weight_scale|input_scale|scale|weight)$",
                name
            )
            if qkv_match:
                prefix = qkv_match.group(1)
                shard_id = qkv_match.group(2)[0]
                suffix = qkv_match.group(3)
                param_name = f"{prefix}.merged_qkv_proj.{suffix}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if getattr(param, "needs_scalar_to_array", False):
                        # Per-tensor fp8 scalar scale: scatter into the fused
                        # [q, k, v] slot by shard id.
                        param.data[{"q": 0, "k": 1, "v": 2}[shard_id]] = loaded_weight.reshape(-1)[0]
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, loaded_shard_id=shard_id)
                    loaded_params.add(param_name)
                continue

            # Dense MLP / MoE shared_mlp gate-up shards are stored unfused in
            # the checkpoint.
            gate_up_match = re.match(
                r"(model\.layers\.\d+\.mlp(?:\.shared_mlp)?)\."
                r"(gate_proj|up_proj)\.(weight_scale|input_scale|scale|weight)$",
                name
            )
            if gate_up_match:
                prefix = gate_up_match.group(1)
                shard_id = 0 if gate_up_match.group(2) == "gate_proj" else 1
                suffix = gate_up_match.group(3)
                param_name = f"{prefix}.gate_up_proj.{suffix}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    if getattr(param, "needs_scalar_to_array", False):
                        # Per-tensor fp8 scalar scale: scatter into the fused
                        # [gate, up] slot by shard id (MergedColumnParallelLinear's
                        # weight_loader lacks the qkv-style needs_scalar_to_array path).
                        param.data[shard_id] = loaded_weight.reshape(-1)[0]
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, loaded_shard_id=shard_id)
                    loaded_params.add(param_name)
                    continue
                # unfused gate_proj/up_proj: fall through to standard matching

            # Standard parameter matching
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    @property
    def main_decode(self):
        # Graph-compile entry: compile forward directly (required by the FIA FP8 paged path).
        return self.forward

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

        # MTP feedback: keep the full-prompt hidden [T,H], but lm_head only needs the
        # last token per request for logits (decode is already [N,1,H]).
        prev_hidden_states = None
        if self.next_n > 0 and is_prefill:
            # Prefill: prev_hidden is the full-prompt [T,H] (already gathered in
            # Model.forward under SP); lm_head only needs the last token/request.
            prev_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            seq_index = forward_metadata.actual_seq_lengths_cu_q - 1
            hidden_states = torch.index_select(prev_hidden_states, 0, seq_index)
            hidden_states = hidden_states.view(seq_index.numel(), 1, hidden_states.size(-1))

        # DP-TP-DP decode holds a [B/tp,q_len,H] hidden shard. Gather it to full
        # [B,q_len,H] across the attn_tp group before lm_head (lm_head is
        # lmhead_tp column-parallel over the same ranks). q_len is next_n+1 under
        # MTP verify, 1 otherwise.
        decode_dp = (not is_prefill) and self.model._dp_decode_active(
            is_prefill, input_ids.shape[0])
        if decode_dp:
            bsz, q_len, hdim = hidden_states.size()
            gathered_hidden = torch.empty(
                [bsz * self.attn_tp_size, q_len, hdim],
                dtype=hidden_states.dtype, device="npu",
            )
            dist.all_gather_into_tensor(
                gathered_hidden, hidden_states.contiguous(),
                group=self.comm_manager.get_group("attn_tp_group"),
            )
            hidden_states = gathered_hidden

        # MTP decode feedback: derive prev_hidden from the FULL-B hidden (post
        # decode-DP gather) so the pure-TP MTP head gets full-B prev_hidden that
        # matches its full-B input_ids [bs*(next_n+1)]. Non-DP decode keeps the
        # already-full [bs,next_n+1,H] hidden, so this reshape yields the same value.
        if self.next_n > 0 and not is_prefill:
            prev_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

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
        if self.next_n > 0:
            return logits, prev_hidden_states
        return logits


# ---------------------------------------------------------------------------
# MTP (Multi-Token Prediction) speculative head — PA+TND
# ---------------------------------------------------------------------------

class HYV3ModelMTPLayer(HYV3Model):
    """MTP layer container: HYV3 backbone trimmed to the single extra nextn layer
    at index num_hidden_layers (=80). embed_tokens shared from main model. Built
    directly to avoid allocating the 80 unused main layers."""

    def __init__(self, config, infer_config, comm_manager, prefix=""):
        nn.Module.__init__(self)
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
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.next_n = infer_config.model_config.next_n if infer_config is not None else 0
        self.enable_sp = _sp_enabled(infer_config)
        sp_on = self.enable_sp and self.attn_tp_size > 1
        # DP-TP-DP decode tier: mirror HYV3Model so _dp_decode_active works on the
        # MTP draft head too (its layers already carry sp_quant per-layer).
        _qc = getattr(config, "quant_config", None)
        _gmm = _qc.gmm_quant_mode if _qc is not None else "w16a16"
        _sp_unified = _gmm in ("w4a8mxfloat4", "w4a8mx", "w8a8float8")
        self.sp_quant = sp_on and _sp_unified
        self.sp_bf16 = sp_on and not _sp_unified
        self.comm_manager = comm_manager
        self.exe_mode = infer_config.model_config.exe_mode
        self.enable_multi_streams = infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.embed_tokens = None
        self.mtp_start_layer_idx = config.num_hidden_layers
        # MTP layer(s) are sparse-MoE; extend mlp_layer_types (built for 0..79) to cover idx 80+
        need = self.mtp_start_layer_idx + config.num_nextn_predict_layers
        if len(config.mlp_layer_types) < need:
            config.mlp_layer_types = list(config.mlp_layer_types) + \
                ["sparse"] * (need - len(config.mlp_layer_types))
        self.layers = nn.ModuleDict({
            str(self.mtp_start_layer_idx + i):
                HYV3DecoderLayer(config, infer_config, comm_manager, self.mtp_start_layer_idx + i,
                                 prefix=f"model.layers.{self.mtp_start_layer_idx + i}")
            for i in range(config.num_nextn_predict_layers)
        })
        self.norm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV3RotaryEmbedding(config=config, max_position_embeddings=self.max_position_embeddings)

        # Plain list of the nextn layers for the graph-captured forward;
        # iterating ModuleDict.values() inside the graph trips dynamo.
        self.mtp_layers = list(self.layers.values())
        # FIA fused qkv path needs the packed cos_sin_table; derive from the layers.
        self.use_fia_fp8 = any(layer.self_attn.use_fia_fp8 for layer in self.mtp_layers)
        self.qkv_fused_attn_type = next(
            (layer.self_attn.attn_type for layer in self.mtp_layers if layer.self_attn.use_fia_fp8),
            None,
        )

    def get_layer(self, i):
        return self.mtp_layers[i]


class HYV3ModelMTP(HYV3ForCausalLM):
    """MTP head: one trained decoder layer (idx 80) + enorm/hnorm/eh_proj fusion.
    lm_head/embed_tokens shared from main model. Weights live under
    model.layers.80.* in the checkpoint; closing norm = layers.80.final_layernorm."""

    # MTP loads only its own layer; never inherit the main "skip layer 80" rule.
    _ignore_weights_patterns = []

    def __init__(self, config, infer_config, comm_manager=None, prefix=""):
        super().__init__(config, infer_config, comm_manager, prefix=prefix)
        self.is_mtp = True
        self.hidden_size = config.hidden_size
        self.model = HYV3ModelMTPLayer(config, infer_config, comm_manager, prefix="model")
        self.lm_head = None  # shared from main model
        self.enorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.final_layernorm = HYV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init_cache(self, device):
        cache_seq_len = self.infer_config.data_config.input_truncated_len + \
            self.infer_config.scheduler_config.max_new_tokens
        bsz = self.infer_config.scheduler_config.batch_size_per_dp_rank
        dtype = self.config.torch_dtype
        cache_batch = bsz * self.attn_tp_size if (self.attn_tp_size > 1 and self.attn_dp_size > 1) else bsz
        for layer in self.model.layers.values():
            attn = layer.self_attn
            cache_shape = (cache_batch, cache_seq_len, *attn.cache_unit)
            attn.k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
            attn.v_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

    def get_cache_info(self) -> ModelCacheInfo:
        # relative idx; merge() offsets by main layer count (deepseek pattern)
        layer_infos = [LayerCacheInfo(layer_idx=i, caches=list(layer.self_attn.cache_entries))
                       for i, layer in enumerate(self.model.layers.values())]
        return ModelCacheInfo(num_layers=len(layer_infos), layer_infos=layer_infos, is_mla_backend=False)

    def forward(self, input_ids=None, position_ids=None, forward_metadata=None,
                prev_hidden_states=None, **kwargs):
        if forward_metadata is None:
            forward_metadata = get_forward_metadata()
        m = self.model
        is_prefill = forward_metadata.is_prefill
        token_count = input_ids.shape[0]
        position_ids = position_ids.view(-1).long()

        # SP token-shard, unified for prefill and decode-DP (mirror HYV3Model.forward).
        # prefill_sp: token-shard the packed prompt across the attn_tp group. The MTP
        #   layers are sparse-MoE (sp_quant), so with SP on the draft head shards its
        #   prefill just like the main model: a real token shard where the
        #   attention AllGather-in rebuilds the full padded seq and trim drops
        #   the <= attn_tp-1 SP pad.
        # decode_dp: request-block shard along the attn_tp group (unchanged mechanism).
        prefill_sp = (m.sp_quant or m.sp_bf16) and is_prefill
        decode_dp = (not is_prefill) and m._dp_decode_active(is_prefill, token_count)
        tp_dp_active = prefill_sp or decode_dp

        # bug#2 (mirror HYV3Model): shard AFTER the vocab-parallel embed all_reduce +
        # eh_proj, never before -- sharding input_ids first would mix tokens across the
        # embed_tp all_reduce. prefill pads input_ids AND prev_hidden_states to a
        # multiple of attn_tp (equal per-rank chunks); pad rows are zero and get
        # dropped by the real-cu_q output selection below. decode-DP keeps the full-B
        # row-major request blocks (token_count % attn_tp == 0 from the gate).
        sp_shard = None
        sp_pad_len = 0
        if tp_dp_active:
            rank_in_tp_group = self.comm_manager.get_rank("attn_tp_group")
            if is_prefill:
                prompt_tokens = forward_metadata.prompt_tokens
                padded_tokens = ((prompt_tokens + m.attn_tp_size - 1) // m.attn_tp_size) * m.attn_tp_size
                sp_pad_len = padded_tokens - prompt_tokens
                if padded_tokens > token_count:
                    pad_ids = torch.zeros(
                        (padded_tokens - token_count,), dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, pad_ids], dim=0)
                    pad_h = torch.zeros(
                        (padded_tokens - token_count, prev_hidden_states.shape[-1]),
                        dtype=prev_hidden_states.dtype, device=prev_hidden_states.device)
                    prev_hidden_states = torch.cat([prev_hidden_states, pad_h], dim=0)
                    token_count = padded_tokens
            per = token_count // m.attn_tp_size
            sp_shard = (rank_in_tp_group * per, (rank_in_tp_group + 1) * per)

        # Vocab-TP embedding on the full (padded) seq -- every attn_tp rank shares it.
        if m.embed_tp_size > 1:
            etr = self.comm_manager.get_rank("embed_tp_group")
            nid = input_ids - etr * (m.vocab_size // m.embed_tp_size)
            mask = (nid >= 0) & (nid < (m.vocab_size // m.embed_tp_size))
            emb = m.embed_tokens(nid * mask) * mask.unsqueeze(-1)
            dist.all_reduce(emb, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            emb = m.embed_tokens(input_ids)

        # e=embedding, h=prev_hidden : eh_proj([enorm(e), hnorm(h)]) -> [T,H]
        hidden_states = self.eh_proj(torch.cat([self.enorm(emb), self.hnorm(prev_hidden_states)], dim=-1))
        # SP: slice this rank's token chunk (prefill) / request block (decode-DP) from
        # the full eh_proj output. position_ids/cos_sin/slot_mapping/block_table stay
        # full-B (the attention AllGathers hidden back to full before consuming them).
        if sp_shard is not None:
            hidden_states = hidden_states[sp_shard[0]:sp_shard[1]]

        block_table = forward_metadata.block_table
        slot_mapping = forward_metadata.slot_mapping

        # pad-aware FA: same dummy-segment handling as HYV3Model.forward; keep the
        # ORIGINAL forward_metadata (real cu_q) for the output-tail selection.
        # pad_len == 0 -> no-op.
        padded_forward_metadata = forward_metadata
        if sp_pad_len > 0:
            position_ids = torch.cat([position_ids, position_ids.new_zeros(sp_pad_len)])
            padded_forward_metadata, slot_mapping, block_table = _build_pad_aware_prefill_metadata(
                forward_metadata, slot_mapping, block_table, sp_pad_len, prompt_tokens
            )

        cos_sin = m.rotary_emb(hidden_states, position_ids, m.max_position_embeddings)
        cos_sin_table = (
            m.rotary_emb.get_cos_sin_table(m.max_position_embeddings)
            if m.use_fia_fp8 else None
        )

        qkv_fused_cu_seq_len = None
        qkv_fused_actual_seq_lens = None
        qkv_fused_slot_mapping = None
        prefill_fa_actual_seq_qlen = None
        prefill_fa_actual_seq_kvlen = None
        if is_prefill:
            prefill_fa_actual_seq_qlen, prefill_fa_actual_seq_kvlen = m._build_prefill_fa_actual_seq_lengths(
                padded_forward_metadata
            )
        if m.use_fia_fp8:
            if (
                padded_forward_metadata.actual_seq_lengths_cu_q is None
                or padded_forward_metadata.actual_seq_lengths_kv is None
            ):
                raise RuntimeError("fused qkv operator requires actual_seq_lengths_cu_q and actual_seq_lengths_kv")
            if slot_mapping is None or m.qkv_fused_attn_type not in slot_mapping:
                raise RuntimeError("fused qkv operator requires slot_mapping for the attention type")
            qkv_fused_cu_q = padded_forward_metadata.actual_seq_lengths_cu_q.to(torch.int32).contiguous()
            qkv_fused_actual_seq_lens = padded_forward_metadata.actual_seq_lengths_kv.to(torch.int32).contiguous()
            qkv_fused_cu_seq_len = torch.cat((qkv_fused_cu_q.new_zeros(1), qkv_fused_cu_q), dim=0)
            qkv_token_count = token_count
            qkv_fused_slot_mapping = (
                slot_mapping[m.qkv_fused_attn_type]
                .view(-1)[:qkv_token_count]
                .to(torch.int32)
                .contiguous()
            )

        residual = None
        # Iterate the materialized nextn-layer list; iterating ModuleDict.values()
        # inside the graph-captured forward trips dynamo (deepseek MTP style).
        for layer in m.mtp_layers:
            residual, hidden_states = layer(hidden_states, cos_sin=cos_sin, forward_metadata=padded_forward_metadata,
                                            past_residual=residual, slot_mapping=slot_mapping, block_table=block_table,
                                            dp_decode=decode_dp,
                                            cos_sin_table=cos_sin_table,
                                            qkv_fused_cu_seq_len=qkv_fused_cu_seq_len,
                                            qkv_fused_actual_seq_lens=qkv_fused_actual_seq_lens,
                                            qkv_fused_slot_mapping=qkv_fused_slot_mapping,
                                            prefill_fa_actual_seq_qlen=prefill_fa_actual_seq_qlen,
                                            prefill_fa_actual_seq_kvlen=prefill_fa_actual_seq_kvlen)
        hidden_states, _ = self.final_layernorm(hidden_states, residual)

        cu_q = forward_metadata.actual_seq_lengths_cu_q
        if is_prefill:
            if prefill_sp:
                # AllGather the [per_rank, H] token shard back to the full padded
                # sequence before selecting each request's last real token -- mirrors
                # the main model's prefill tp_dp_active gather tail.
                per_rank, hsz = hidden_states.size()
                gathered = torch.empty(
                    [m.attn_tp_size * per_rank, hsz], dtype=hidden_states.dtype, device="npu")
                dist.all_gather_into_tensor(
                    gathered, hidden_states.contiguous(),
                    group=self.comm_manager.get_group("attn_tp_group"))
                hidden_states = gathered
            seq_index = cu_q - 1
            # prev_out is discarded in prefill (loop_mtp=1); trim the SP pad to keep
            # the full real [prompt_tokens, H] shape identical to the pre-shard path
            # (no-op when not sharded, since token_count == prompt_tokens there).
            prev_out = hidden_states[:forward_metadata.prompt_tokens]
            logits_h = torch.index_select(hidden_states, 0, seq_index).view(seq_index.numel(), 1, -1)
        else:
            # MTP decode runs next_n+1 tokens/request; keep q_len axis so postprocess
            # argmax([B, next_n+1, vocab]) aligns with spec-token gather.
            q_len = self.next_n + 1
            bs = hidden_states.shape[0] // q_len
            logits_h = hidden_states.view(bs, q_len, hidden_states.shape[-1])
            if decode_dp:
                # Gather the [bs/tp, q_len, H] shard back to full [bs, q_len, H]
                # across attn_tp before lm_head (lm_head is lmhead_tp column-parallel
                # over the same ranks) -- mirrors the main model's decode-DP tail.
                gathered = torch.empty(
                    [bs * self.attn_tp_size, q_len, hidden_states.shape[-1]],
                    dtype=hidden_states.dtype, device="npu",
                )
                dist.all_gather_into_tensor(
                    gathered, logits_h.contiguous(),
                    group=self.comm_manager.get_group("attn_tp_group"),
                )
                logits_h = gathered
            # prev_out is fed back to the next MTP step and to postprocess as full-B
            # [bs*(next_n+1), H]; derive it from the full (gathered) hidden.
            prev_out = logits_h.reshape(-1, logits_h.shape[-1])

        logits = self.lm_head(logits_h)
        if self.lmhead_tp_size > 1:
            logits = logits.float()
            gl = [torch.empty_like(logits) for _ in range(self.lmhead_tp_size)]
            dist.all_gather(gl, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.cat(gl, dim=-1)
        if self.enable_lm_head_fp32 and logits.dtype != torch.float32:
            logits = logits.float()
        return logits, prev_out

    def decode(self, **kwargs):
        return self.forward(is_prefill=False, **kwargs)

    @property
    def mtp_decode(self):
        # Graph-compile entry for the MTP head: compile forward directly.
        return self.forward

    def load_weights(self, weights: Generator[Tuple[str, torch.Tensor], None, None]) -> Set[str]:
        mtp_prefix = f"model.layers.{self.config.num_hidden_layers}."
        unique = ("enorm", "hnorm", "eh_proj", "final_layernorm")

        def remapped():
            for name, w in weights:
                if "embed_tokens" in name or name.startswith("lm_head"):
                    continue
                if name.startswith(mtp_prefix):
                    tail = name[len(mtp_prefix):]
                    if any(tail.startswith(u + ".") for u in unique):
                        yield tail, w
                    else:
                        yield name, w
                    continue
                if name.startswith("model.layers."):
                    continue  # drop main backbone 0..79
                yield name, w

        return super().load_weights(remapped())
