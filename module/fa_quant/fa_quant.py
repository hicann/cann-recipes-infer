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

import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


def npu_group_quant(x, group_size, dst_type=torch.float8_e4m3fn, output_bnsd=True):
    """
    Perform per-block quantization on input tensor for NPU acceleration.

    This function quantizes the input tensor using dynamic block quantization,
    which is optimized for NPU operations. The quantization is performed per-block
    with specified group size, producing both quantized values and scale factors.

    Args:
        x: Input tensor with shape (batch, seq_len, num_heads, head_dim) in BNSD format
        group_size: Block size for quantization granularity (128 for Q, 256 for K/V)
        dst_type: Target data type for quantized output, default is FP8 E4M3
        output_bnsd: If True, output tensors are transposed to (batch, num_heads, seq_len, head_dim)

    Returns:
        tuple: (quantized_tensor, scale_tensor)
               - quantized_tensor: FP8 quantized values
               - scale_tensor: Per-block scale factors for dequantization

    Raises:
        ValueError: If group_size is not 128 or 256 (current supported granularities)
    """
    # Validate group size - only 128 (for Q) and 256 (for K/V) are supported
    if group_size not in [128, 256]:
        raise ValueError("Current per-block quantization granularity only supports q=128, kv=256")

    # Reshape input from (b, s, n, d) to (b*s, n*d) for block quantization
    # This flattening is required by npu_dynamic_block_quant API
    b, s, n, d = x.shape
    x = x.contiguous().view(b * s, n * d)

    # Perform dynamic block quantization using NPU-specific kernel
    # row_block_size: number of elements per row block
    # col_block_size: dimension size (d), each column block spans one head dimension
    x_quant, x_scale = torch_npu.npu_dynamic_block_quant(
        x,
        row_block_size=group_size,
        col_block_size=d,
        dst_type=dst_type
    )

    # Reshape quantized output back to original 4D shape
    x_quant = x_quant.contiguous().view(b, s, n, d)

    # Reshape scale tensor: scales are computed per block, so dimension is reduced
    # Scale shape: (b, ceil(s/group_size), n, 1) - one scale per block of group_size elements
    x_scale = x_scale.contiguous().view(b, -1, n, 1)

    # Optionally transpose to BNSD format (batch, num_heads, seq_len, head_dim)
    # This format is required by npu_fused_infer_attention_score_v2
    if output_bnsd:
        return x_quant.transpose(1, 2), x_scale.transpose(1, 2)
    else:
        return x_quant, x_scale


def npu_fp8_attn(q, k, v, dst_type=torch.float8_e4m3fn, softmax_scale=None):
    """
    Perform FP8 quantized attention computation on NPU.

    This function implements fused attention with FP8 quantization, which reduces
    memory bandwidth and improves performance on NPU hardware. The quantization
    uses different group sizes for Q (128) and K/V (256) based on their different
    precision requirements.

    Args:
        q: Query tensor with shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor with shape (batch, seq_len, num_heads, head_dim)
        v: Value tensor with shape (batch, seq_len, num_heads, head_dim)
        dst_type: Target FP8 data type for quantization, default is E4M3 format
        softmax_scale: Optional softmax scale factor. If None, uses 1/sqrt(head_dim)

    Returns:
        torch.Tensor: Attention output with shape (batch, seq_len, num_heads, head_dim)
                      in BF16 format
    """
    b, s, n, d = q.shape

    # Compute softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    # Per-block quantization: Q with group_size=128, K/V with group_size=256
    # Q uses smaller group size for higher precision since it directly affects attention scores
    # K/V use larger group size as they have more tolerance for quantization noise
    q_quant, q_scale = npu_group_quant(q, 128, dst_type, output_bnsd=True)
    k_quant, k_scale = npu_group_quant(k, 256, dst_type, output_bnsd=True)
    v_quant, v_scale = npu_group_quant(v, 256, dst_type, output_bnsd=True)

    # Tensor shapes after quantization (in BNSD format):
    # q_quant: (b, n, q_s, d) - quantized query
    # k_quant: (b, n, kv_s, d) - quantized key
    # v_quant: (b, n, kv_s, d) - quantized value
    # q_scale: (b, n, ceil(q_s/128), 1) - one scale per 128-element block
    # k_scale: (b, n, ceil(kv_s/256), 1) - one scale per 256-element block
    # v_scale: (b, n, ceil(kv_s/256), 1) - one scale per 256-element block

    # Fused attention computation with built-in dequantization
    # quant_mode=7 indicates per-block quantization with dynamic scales
    attn_out = torch_npu.npu_fused_infer_attention_score_v2(
                q_quant,
                k_quant,
                v_quant,
                num_query_heads=n,
                num_key_value_heads=n,
                input_layout="BNSD",
                softmax_scale=softmax_scale,
                query_quant_mode=7,
                key_quant_mode=7,
                value_quant_mode=7,
                dequant_scale_query=q_scale,
                dequant_scale_key=k_scale,
                dequant_scale_value=v_scale,
    )[0]

    # Transpose output from BNSD back to BSND format and convert to BF16
    # BSND format: (batch, seq_len, num_heads, head_dim)
    return attn_out.transpose(1, 2).to(torch.bfloat16)
