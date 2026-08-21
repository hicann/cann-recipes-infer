# coding=utf-8
# Adapted from
# https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/simple_gla
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def _simple_gla_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    inplace: bool = False,
):
    gk = gk.to(q.dtype)
    q = q * scale
    g = torch.exp(gk)
    if inplace and state is not None:
        state.mul_(g[..., None, None])
        state.add_(k.unsqueeze(-1) * v.unsqueeze(-2))
        out = torch.einsum("bhd,bhdv->bhv", q, state)
        return out, state
    state = state * g[..., None, None]
    state = state + k.unsqueeze(-1) * v.unsqueeze(-2)
    out = torch.einsum("bhd,bhdv->bhv", q, state)
    return out, state


def fused_recurrent_simple_gla_torch(
    q, k, v, gk,
    initial_state,
    output_final_state: bool = False,
    head_first: bool = False,
    scale: Optional[float] = None,
    inplace_state: bool = False,
):
    if head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        gk = gk.transpose(1, 2)

    if q.shape[1] != 1:
        raise NotImplementedError(f"fused_recurrent only supports seq_len==1, got {q.shape[1]}")

    # select, not [:, 0], so backends don't treat it as a dynamic slice
    q = q.select(1, 0)
    k = k.select(1, 0)
    v = v.select(1, 0)
    gk = gk.select(1, 0)

    k_dim = q.shape[-1]
    if scale is None:
        scale = k_dim ** -0.5

    if initial_state is None:
        bsz, num_heads, v_dim = v.shape
        initial_state = q.new_zeros(bsz, num_heads, k_dim, v_dim)

    out, new_state = _simple_gla_step(q, k, v, gk, initial_state, scale, inplace=inplace_state)

    out = out.unsqueeze(1)         # [bsz, 1, num_heads, v_dim]
    if head_first:
        out = out.transpose(1, 2)

    out = out.to(q.dtype)
    if not inplace_state:
        new_state = new_state.to(q.dtype)
    if not output_final_state:
        return out, None
    return out, new_state


def _prepare_inputs(q, k, v, gk, head_first):
    squeeze_batch = False
    if not head_first:
        if q.dim() == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            gk = gk.unsqueeze(0)
            squeeze_batch = True
        elif q.dim() != 4:
            raise ValueError(f"Unsupported q dim: {q.dim()}")
    else:
        if q.dim() != 4:
            raise ValueError("head_first=True only supports 4D input")
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        gk = gk.transpose(1, 2)
    return q, k, v, gk, squeeze_batch


def _restore_layout(out, final_state, head_first, squeeze_batch):
    if head_first:
        out = out.transpose(1, 2)
    if squeeze_batch:
        out = out.squeeze(0)
        if final_state is not None and final_state.dim() == 4 and final_state.shape[0] == 1:
            final_state = final_state.squeeze(0)
    return out, final_state


def _chunk_simple_gla_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    chunk_size: int,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, seq_len, num_heads, k_dim = q.shape
    v_dim = v.shape[-1]
    chunk_len = chunk_size
    dtype = q.dtype
    device = q.device

    # pad seq_len up to a multiple of chunk_size
    pad = (-seq_len) % chunk_len
    if pad > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        gk = F.pad(gk, (0, 0, 0, pad))
    padded_len = seq_len + pad
    num_chunks = padded_len // chunk_len

    q = q * scale

    # reshape (not view) to tolerate the non-contiguous layout after a head_first transpose.
    q = q.reshape(bsz, num_chunks, chunk_len, num_heads, k_dim)
    k = k.reshape(bsz, num_chunks, chunk_len, num_heads, k_dim)
    v = v.reshape(bsz, num_chunks, chunk_len, num_heads, v_dim)
    gk = gk.reshape(bsz, num_chunks, chunk_len, num_heads)

    # intra-chunk cumulative log-gate
    g_cumsum_f = torch.cumsum(gk.float(), dim=2)
    g_chunk_f = g_cumsum_f[:, :, -1, :]

    exp_g_cumsum = torch.exp(g_cumsum_f).to(dtype)
    exp_g_chunk = torch.exp(g_chunk_f).to(dtype)
    w_to_end = torch.exp(g_chunk_f.unsqueeze(2) - g_cumsum_f).to(dtype)

    # ΔS_i: one einsum, all chunks in parallel
    k_weighted = k * w_to_end.unsqueeze(-1)
    delta_state = torch.einsum("bnchd,bnchv->bnhdv", k_weighted, v)

    if initial_state is None:
        state_prev = q.new_zeros(bsz, num_heads, k_dim, v_dim)
    else:
        state_prev = initial_state.to(device=device, dtype=dtype)

    state_in_list = []  # state entering each chunk
    for i in range(num_chunks):
        state_in_list.append(state_prev)
        state_prev = exp_g_chunk[:, i, :, None, None] * state_prev + delta_state[:, i]
    state_in = torch.stack(state_in_list, dim=1)

    # inter-chunk
    out_inter = torch.einsum("bnchd,bnhdv->bnchv", q, state_in)
    out_inter = out_inter * exp_g_cumsum.unsqueeze(-1)

    # intra-chunk: causal-masked attention within a chunk + gate decay
    g_diff = g_cumsum_f.unsqueeze(3) - g_cumsum_f.unsqueeze(2)
    causal = torch.tril(torch.ones(chunk_len, chunk_len, device=device, dtype=torch.bool))
    decay = torch.exp(g_diff).masked_fill(
        ~causal[None, None, :, :, None], 0.0
    ).to(dtype)

    qk = torch.einsum("bnjhd,bnmhd->bnjmh", q, k)
    out_intra = torch.einsum("bnjmh,bnmhv->bnjhv", qk * decay, v)

    out = (out_inter + out_intra).reshape(bsz, padded_len, num_heads, v_dim)
    if pad > 0:
        out = out[:, :seq_len]

    final_state = state_prev if output_final_state else None
    return out, final_state


def chunk_simple_gla_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    head_first: bool = False,
    scale: Optional[float] = None,
    chunk_size: int = 64,
    **kwargs,
):
    """Chunk-parallel pure-PyTorch equivalent of FLA chunk_simple_gla.

    Layout:
      head_first=False:  q/k/v [bsz, seq_len, num_heads, dim] or [seq_len, num_heads, dim]
      head_first=True:   q/k/v [bsz, num_heads, seq_len, dim]
    """
    q, k, v, gk, squeeze_batch = _prepare_inputs(q, k, v, gk, head_first=head_first)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    out, final_state = _chunk_simple_gla_inner(
        q, k, v, gk,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=scale,
    )

    out, final_state = _restore_layout(
        out=out,
        final_state=final_state,
        head_first=head_first,
        squeeze_batch=squeeze_batch,
    )
    return out, final_state
