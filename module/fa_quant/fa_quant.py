# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


def npu_mxfp8_quant(x, dst_type=torch.float8_e4m3fn, output_layout="TND", axis=-1, *, scale_alg=0):
    """
    Perform MXFP8 dynamic quantization for NPU fused attention.

    Args:
        x: Input tensor with shape (batch, seq_len, num_heads, head_dim) in BSND format.
        dst_type: Target data type for quantized output, default is FP8 E4M3.
        output_layout: Output layout for FA v2. Supports "TND" now.
        axis: Quantization axis. -1 groups along head_dim (for Q/K);
            0 groups 64 consecutive tokens (for V).
        scale_alg: Block scale exponent rounding. 0 floors the exponent
            (default, historical behavior); 1 rounds it up so no block value
            exceeds the FP8 representable range.

    Returns:
        tuple: (quantized_tensor, scale_tensor)
    """
    b, s, n, d = x.shape
    if scale_alg not in (0, 1):
        raise ValueError("MXFP8 quantization scale_alg must be 0 or 1")

    if output_layout == "TND":
        quant_kwargs = {"dst_type": dst_type, "axis": axis}
        if scale_alg:
            quant_kwargs["scale_alg"] = scale_alg
        x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(
            x.view(b * s, n, d),
            **quant_kwargs
        )
    else:
        raise ValueError("Flash Attention MXFP8 quantization currently only supports the TND layout")

    return x_quant, x_scale


def _actual_seq_lens(batch_size, seq_len, device):
    return torch.arange(
        seq_len,
        seq_len * (batch_size + 1),
        seq_len,
        dtype=torch.int64,
        device=device,
    )


def npu_mxfp8_attn(q, k, v, dst_type=torch.float8_e4m3fn, softmax_scale=None, *, scale_alg=0):
    """
    Perform MXFP8 quantized attention computation on NPU.

    Q/K/V are dynamically quantized with npu_dynamic_mx_quant, then passed to
    npu_fused_infer_attention_score_v2 with MXFP8 quant modes.
    """
    b, s, n, d = q.shape
    kv_s = k.shape[1]
    out_dtype = q.dtype

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    q_quant, q_scale = npu_mxfp8_quant(q, dst_type=dst_type, output_layout="TND", scale_alg=scale_alg)
    k_quant, k_scale = npu_mxfp8_quant(k, dst_type=dst_type, output_layout="TND", scale_alg=scale_alg)
    v_quant, v_scale = npu_mxfp8_quant(v, dst_type=dst_type, output_layout="TND", axis=0, scale_alg=scale_alg)

    attn_out = torch_npu.npu_fused_infer_attention_score_v2(
        q_quant,
        k_quant,
        v_quant,
        actual_seq_qlen=_actual_seq_lens(b, s, q.device),
        actual_seq_kvlen=_actual_seq_lens(b, kv_s, k.device),
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        num_query_heads=n,
        num_key_value_heads=k.shape[2],
        softmax_scale=softmax_scale,
        input_layout="TND",
        sparse_mode=0,       # non-sparse
        query_quant_mode=6,  # per-channel group quant
        key_quant_mode=6,    # per-channel group quant
        value_quant_mode=8,  # per-token group quant
        query_dtype=torch.float8_e4m3fn,
        key_dtype=torch.float8_e4m3fn,
        value_dtype=torch.float8_e4m3fn,
        dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
        out_dtype=out_dtype
    )[0]

    return attn_out.view(b, s, n, d)


class FaMxfp8Unsupported(RuntimeError):
    """Raised by the MXFP8 attention path for shapes/environments it cannot run.

    Carries the human-readable reason used by the caller's fallback audit
    log / fail-fast error.
    """


def _dense_unsupported_reason(query, cp_world_size):
    """Return a reason string when the dense MXFP8 path is unsupported, else None."""
    if not query.is_npu:
        return "input tensors are not on NPU"
    if cp_world_size > 1:
        return "context-parallel (cp_size>1) is not validated for fa-mxfp8"
    d = query.shape[-1]
    if d % 32 != 0:
        return f"head_dim={d} is not a multiple of 32 (MX group size)"
    return None


def npu_mxfp8_dense_attn(query, key, value, softmax_scale=None, *, scale_alg=0, cp_world_size=1):
    """Guarded dense MXFP8 attention.

    Checks the dense-path preconditions, casts Q/K/V to bfloat16 and runs
    npu_mxfp8_attn. Every unsupported shape / environment / kernel rejection
    is normalized into a single FaMxfp8Unsupported carrying the reason; the
    fail-fast vs BF16-fallback policy and the audit log belong to the caller.
    cp_world_size is passed in by the caller because the distributed context
    is model-side state.
    """
    reason = _dense_unsupported_reason(query, cp_world_size)
    if reason is not None:
        raise FaMxfp8Unsupported(reason)
    try:
        return npu_mxfp8_attn(
            query.to(torch.bfloat16).contiguous(),
            key.to(torch.bfloat16).contiguous(),
            value.to(torch.bfloat16).contiguous(),
            softmax_scale=softmax_scale,
            scale_alg=scale_alg,
        )
    except (RuntimeError, TypeError) as exc:
        raise FaMxfp8Unsupported(f"MXFP8 attention kernel rejected the input ({exc})") from exc
