# coding=utf-8
# Adapted from
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/attention.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from einops import rearrange

TORCH_NPU_AVAILABLE = False
try:
    import torch_npu
    if hasattr(torch_npu, "npu_fused_infer_attention_score"):
        TORCH_NPU_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AttentionInputParams:
    """Encapsulates attention input parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_lens: Optional[Union[torch.Tensor, List]] = None
    k_lens: Optional[Union[torch.Tensor, List]] = None
    dropout_p: float = 0.
    softmax_scale: Optional[float] = None
    q_scale: Optional[float] = None
    causal: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    deterministic: bool = False
    dtype: torch.dtype = torch.bfloat16
    is_cross_attention: bool = False


@dataclass
class AttentionLSEParams:
    """Encapsulates attention with LSE parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_lens: Optional[Union[torch.Tensor, List]] = None
    k_lens: Optional[Union[torch.Tensor, List]] = None
    dropout_p: float = 0.
    softmax_scale: Optional[float] = None
    q_scale: Optional[float] = None
    causal: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    deterministic: bool = False
    dtype: torch.dtype = torch.bfloat16


@dataclass
class NPUAttentionParams:
    """Encapsulates NPU attention parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_lens: Optional[Union[torch.Tensor, List]] = None
    k_lens: Optional[Union[torch.Tensor, List]] = None
    softmax_scale: Optional[float] = None
    q_scale: Optional[float] = None
    causal: bool = False
    is_cross_attention: bool = False


def attention(q, k, v, **kwargs):
    """Attention wrapper with original interface."""
    params = AttentionInputParams(q=q, k=k, v=v, **kwargs)
    return _attention_impl(params)


def _attention_impl(params: AttentionInputParams):
    """Internal attention implementation using AttentionInputParams."""
    if TORCH_NPU_AVAILABLE and torch.npu.is_available() and params.q.device.type == 'npu':
        npu_params = NPUAttentionParams(
            q=params.q, k=params.k, v=params.v,
            q_lens=params.q_lens,
            k_lens=params.k_lens,
            softmax_scale=params.softmax_scale,
            q_scale=params.q_scale,
            causal=params.causal,
            is_cross_attention=params.is_cross_attention,
        )
        return _npu_fused_attention_impl(npu_params)
    else:
        return _flash_attention_impl(params)


def npu_fused_attention(q, k, v, **kwargs):
    """NPU fused attention with original interface."""
    params = NPUAttentionParams(q=q, k=k, v=v, **kwargs)
    return _npu_fused_attention_impl(params)


def _npu_fused_attention_impl(params: NPUAttentionParams) -> torch.Tensor:
    """Internal NPU fused attention implementation."""
    q, k, v = params.q, params.k, params.v
    out_dtype = q.dtype
    
    if params.q_scale is not None:
        q = q * params.q_scale
    
    target_dtype = out_dtype
    if q.dtype != target_dtype:
        q = q.to(target_dtype)
    if k.dtype != target_dtype:
        k = k.to(target_dtype)
    if v.dtype != target_dtype:
        v = v.to(target_dtype)
    
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    b, n, s, d = q.shape
    num_key_value_heads = k.shape[1]
    
    if n % num_key_value_heads != 0:
        raise ValueError(
            f"GQA requires num_heads divisible by num_key_value_heads, "
            f"got num_heads={n}, num_key_value_heads={num_key_value_heads}"
        )
    
    if n == num_key_value_heads:
        num_key_value_heads = 0
    
    softmax_scale = params.softmax_scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    
    actual_seq_lengths = None
    actual_seq_lengths_kv = None

    if params.q_lens is not None:
        if isinstance(params.q_lens, torch.Tensor):
            actual_seq_lengths = params.q_lens.cpu().tolist()
        elif isinstance(params.q_lens, list):
            actual_seq_lengths = params.q_lens
        else:
            actual_seq_lengths = list(params.q_lens)

    if params.k_lens is not None:
        if isinstance(params.k_lens, torch.Tensor):
            actual_seq_lengths_kv = params.k_lens.cpu().tolist()
        elif isinstance(params.k_lens, list):
            actual_seq_lengths_kv = params.k_lens
        else:
            actual_seq_lengths_kv = list(params.k_lens)

    attention_out, _ = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_heads=n,
        scale=float(softmax_scale),
        input_layout="BNSD",
        num_key_value_heads=num_key_value_heads,
        pre_tokens=65535,
        next_tokens=65535 if not params.causal else 0,
        sparse_mode=0,
        inner_precise=0,
    )
    
    attention_out = attention_out.transpose(1, 2).contiguous()
    
    return attention_out.to(out_dtype)


def npu_fused_attention_with_lse(q, k, v, **kwargs):
    """NPU fused attention with LSE with original interface."""
    params = AttentionLSEParams(q=q, k=k, v=v, **kwargs)
    return _npu_fused_attention_with_lse_impl(params)


def _npu_fused_attention_with_lse_impl(params: AttentionLSEParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal NPU fused attention with LSE implementation."""
    q, k, v = params.q, params.k, params.v
    out_dtype = q.dtype
    
    if params.q_scale is not None:
        q = q * params.q_scale
    
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    num_heads = q.shape[1]
    num_key_value_heads = k.shape[1]
    head_dim = q.shape[-1]
    
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            f"GQA requires num_heads divisible by num_key_value_heads, "
            f"got num_heads={num_heads}, num_key_value_heads={num_key_value_heads}"
        )
    
    if num_heads == num_key_value_heads:
        num_key_value_heads = 0
    
    softmax_scale = params.softmax_scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    actual_seq_lengths = None
    actual_seq_lengths_kv = None
    
    if params.q_lens is not None:
        if isinstance(params.q_lens, torch.Tensor):
            actual_seq_lengths = params.q_lens.cpu().tolist()
        elif isinstance(params.q_lens, list):
            actual_seq_lengths = params.q_lens
        else:
            actual_seq_lengths = list(params.q_lens)
    
    if params.k_lens is not None:
        if isinstance(params.k_lens, torch.Tensor):
            actual_seq_lengths_kv = params.k_lens.cpu().tolist()
        elif isinstance(params.k_lens, list):
            actual_seq_lengths_kv = params.k_lens
        else:
            actual_seq_lengths_kv = list(params.k_lens)
    
    attention_out, softmax_lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_heads=num_heads,
        scale=float(softmax_scale),
        input_layout="BNSD",
        num_key_value_heads=num_key_value_heads,
        pre_tokens=65535,
        next_tokens=65535 if not params.causal else 0,
        sparse_mode=0,
        inner_precise=0,
        softmax_lse_flag=True,
    )
    
    attention_out = attention_out.transpose(1, 2).contiguous()
    softmax_lse = softmax_lse.squeeze(-1)
    
    return attention_out.to(out_dtype), softmax_lse


def attention_with_lse(q, k, v, **kwargs):
    """Attention with LSE wrapper with original interface."""
    params = AttentionLSEParams(q=q, k=k, v=v, **kwargs)
    return _attention_with_lse_impl(params)


def _attention_with_lse_impl(params: AttentionLSEParams):
    """Internal attention with LSE implementation."""
    if TORCH_NPU_AVAILABLE and torch.npu.is_available() and params.q.device.type == 'npu':
        return _npu_fused_attention_with_lse_impl(params)
    else:
        raise NotImplementedError("attention_with_lse only implemented for NPU")


def flash_attention(q, k, v, **kwargs):
    """Flash attention with original interface."""
    params = AttentionInputParams(q=q, k=k, v=v, **kwargs)
    return _flash_attention_impl(params)


def _flash_attention_impl(params: AttentionInputParams) -> torch.Tensor:
    """Internal flash attention implementation."""
    from flash_attn import flash_attn_varlen_func, flash_attn_func

    q, k, v = params.q, params.k, params.v
    
    half_dtypes = (torch.float16, torch.bfloat16)
    assert q.dtype in half_dtypes and k.dtype in half_dtypes and v.dtype in half_dtypes

    if params.q_scale is not None:
        q = q * params.q_scale

    if params.q_lens is not None or params.k_lens is not None:
        assert params.q_lens is not None and params.k_lens is not None
        q = rearrange(q, 'b s h d -> (b s) h d')
        k = rearrange(k, 'b s h d -> (b s) h d')
        v = rearrange(v, 'b s h d -> (b s) h d')
        
        q_lens = params.q_lens.to(torch.int32).flatten()
        k_lens = params.k_lens.to(torch.int32).flatten()
        
        assert q_lens.shape[0] == k_lens.shape[0]
        
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device)
        max_seqlen_q = q_lens.max().item()
        max_seqlen_k = k_lens.max().item()
        batch_size = q_lens.shape[0]
        
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=params.dropout_p,
            softmax_scale=params.softmax_scale,
            causal=params.causal,
            window_size=params.window_size,
            deterministic=params.deterministic,
        )
        
        output = rearrange(output, '(b s) h d -> b s h d', b=batch_size)
    else:
        output = flash_attn_func(
            q, k, v,
            dropout_p=params.dropout_p,
            softmax_scale=params.softmax_scale,
            causal=params.causal,
            window_size=params.window_size,
            deterministic=params.deterministic,
        )
    
    return output