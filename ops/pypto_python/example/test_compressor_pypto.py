#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test module for compressor."""

import os
import sys
from numpy.testing import assert_allclose
import torch
import torch_npu
import numpy as np


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, "impl")
sys.path.insert(0, impl_path)


from compressor_pypto import compressor_pypto

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={"float": "{:.6f}".format})


def overlap_transform(tensor: torch.Tensor, value: float) -> torch.Tensor:
    # tensor shape: [batch_size, seq_len, ratio, 2*dim]
    b, s, ratio, d = tensor.size()
    d = d // 2
    new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
    return new_tensor


def rms_norm_golden(x: torch.Tensor, eps: float, weight: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (weight * x).to(dtype)


def apply_rotary_pos_emb_v2(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    mode: str = "half",
) -> torch.Tensor:
    input_dtype = x.dtype
    if input_dtype != torch.float32:
        x = x.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
    if mode == "half":
        b, s, d = x.shape
        x = x.reshape(b, s, d // 2, 2).permute(0, 1, 3, 2).reshape(b, s, d)

        x1, x2 = x.chunk(2, dim=-1)
        p = torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        p = torch.stack((-x2, x1), dim=-1).flatten(-2)
    x_embed = (x * cos) + (p * sin)
    x_embed = x_embed.to(input_dtype)

    return x_embed


def golden_compress(
    x,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    hadamard,
    ratio,
    start_pos_dy,
    rope_head_dim,
    rotate,
    eps=1e-6,
):
    bsz, s1, _ = x.size()
    overlap = ratio == 4
    dtype = x.dtype
    x = x.float()  # b,s,h

    wkv = wkv.transpose(-2, -1).to(torch.float32)
    wgate = wgate.transpose(-2, -1).to(torch.float32)
    d = wkv.size(1) // (1 + overlap)

    kv_total = torch.matmul(x, wkv)  # b,s,2d
    score_total = torch.matmul(x, wgate)  # b,s,2d

    block_size = kv_state.shape[1]
    kv_output = torch.zeros(
        (min(bsz * s1, bsz * s1 // ratio + bsz), d),
        dtype=torch.bfloat16,
        device=x.device,
    )
    for b_idx in range(bsz):
        for i in range(s1):
            start_pos = start_pos_dy[b_idx]
            should_compress = (start_pos + i + 1) % ratio == 0
            pos = (start_pos + i) % ratio
            kv = kv_total[b_idx, i : i + 1, :].clone()
            score = score_total[b_idx, i : i + 1, :].clone()
            score += ape[pos]
            if overlap:
                kv_block_idx = kv_block_table[b_idx, (start_pos + i) // block_size]
                score_block_idx = score_block_table[
                    b_idx, (start_pos + i) // block_size
                ]
                cur_pos = (start_pos + i) % block_size
                kv_state[kv_block_idx, cur_pos, :] = kv.squeeze(0)
                score_state[score_block_idx, cur_pos, :] = score.squeeze(0)
                if should_compress:
                    pre_kv_block_idx = kv_block_table[
                        b_idx, (start_pos + i - 2 * ratio + 1) // block_size
                    ]
                    pre_score_block_idx = score_block_table[
                        b_idx, (start_pos + i - 2 * ratio + 1) // block_size
                    ]
                    pre_start = (start_pos + i - 2 * ratio + 1) % block_size
                    pre_end = pre_start + ratio
                    cur_start = (start_pos + i - ratio + 1) % block_size
                    cur_end = cur_start + ratio
                    if start_pos < ratio:
                        kv_state_tmp = torch.cat(
                            [
                                kv_state[pre_kv_block_idx, pre_start:pre_end, :d] * 0,
                                kv_state[kv_block_idx, cur_start:cur_end, d:],
                            ],
                            dim=0,
                        )
                        score_state_tmp = torch.cat(
                            [
                                score_state[pre_score_block_idx, pre_start:pre_end, :d]
                                - float("inf"),
                                score_state[score_block_idx, cur_start:cur_end, d:],
                            ],
                            dim=0,
                        )
                    else:
                        kv_state_tmp = torch.cat(
                            [
                                kv_state[pre_kv_block_idx, pre_start:pre_end, :d],
                                kv_state[kv_block_idx, cur_start:cur_end, d:],
                            ],
                            dim=0,
                        )  # 8,d
                        score_state_tmp = torch.cat(
                            [
                                score_state[pre_score_block_idx, pre_start:pre_end, :d],
                                score_state[score_block_idx, cur_start:cur_end, d:],
                            ],
                            dim=0,
                        )  # 8,d
                    kv = (kv_state_tmp * score_state_tmp.softmax(dim=0)).sum(
                        dim=0, keepdim=False
                    )  # 1,d
            else:
                kv_block_idx = kv_block_table[b_idx, (start_pos + i) // block_size]
                score_block_idx = score_block_table[
                    b_idx, (start_pos + i) // block_size
                ]
                cur_pos = (start_pos + i) % block_size
                kv_state[kv_block_idx, cur_pos, :] = kv.squeeze(0)
                score_state[score_block_idx, cur_pos, :] = score.squeeze(0)
                if should_compress:
                    kv_tmp = torch.cat((kv_state[kv_block_idx, :-1, :], kv), dim=0)
                    score_tmp = torch.cat(
                        (score_state[score_block_idx, :-1, :], score), dim=0
                    )
                    kv = (kv_tmp * score_tmp.softmax(dim=0)).sum(dim=0, keepdim=False)

            if should_compress:
                kv = rms_norm_golden(kv.to(dtype), eps, weight)  # 1,d
                kv_rope = kv[..., -rope_head_dim:].clone()
                kv_new = kv.clone()
                kv_new[..., -rope_head_dim:] = apply_rotary_pos_emb_v2(
                    kv_rope, sin[b_idx, ...], cos[b_idx, ...], "interleave"
                )
                if rotate:
                    kv_output[b_idx, :] = torch.matmul(kv_new, hadamard)  # 1,d
                else:
                    kv_output[b_idx, :] = kv_new  # b,1,d
    return kv_output


def gen_inputs(
    bsz: int,
    seq: int,
    h: int,
    d: int,
    rope_head_dim: int,
    ratio: int,
    device: str,
):
    torch.manual_seed(42)
    overlap = ratio == 4
    coff = 1 + overlap
    x = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device)
    rope_axis0 = min(bsz * seq, bsz * seq // ratio + bsz)
    sin = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16, device=device)
    cos = torch.rand((rope_axis0, rope_head_dim), dtype=torch.bfloat16, device=device)
    wkv = torch.rand((coff * d, h), dtype=torch.bfloat16, device=device)
    wgate = torch.rand((coff * d, h), dtype=torch.bfloat16, device=device)
    ape = torch.rand((ratio, coff * d), dtype=torch.float32, device=device)
    weight = torch.ones(d, dtype=torch.bfloat16, device=device)
    if overlap:
        block_table = (
            torch.ones(bsz, 100, dtype=torch.int32, device=device)
            + torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2
        )
    else:
        block_table = (
            torch.arange(100, dtype=torch.int32, device=device) % 2
            + 1
            + torch.arange(bsz, dtype=torch.int32, device=device).view(-1, 1) * 2
        )
    kv_state = torch.zeros(
        (block_table.max() + 1, 128, coff * d), dtype=torch.float32, device=device
    )
    score_state = torch.zeros(
        (block_table.max() + 1, 128, coff * d), dtype=torch.float32, device=device
    )
    hadamard = torch.rand((d, d), dtype=torch.bfloat16, device=device) * (d**-0.5)
    return (
        x,
        sin,
        cos,
        wkv,
        wgate,
        ape,
        weight,
        kv_state,
        score_state,
        block_table,
        hadamard,
    )


def test_comp_128():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True

    ra = 128
    ro = False

    bsz = 64
    st = torch.tensor([8192] * bsz, dtype=torch.int32, device=device)
    print(f"test_compressor_decode (ratio: {ra}, rotate: {ro}) begin!")
    seq = 2
    h = 4096
    d = 512
    rope_head_dim = 64
    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, \
        block_table, hadamard = gen_inputs(bsz, seq, h, d, rope_head_dim, ra, device)

    out = compressor_pypto(x, kv_state, score_state, block_table, block_table, \
                           sin, cos, wkv, wgate, ape, weight, hadamard, st, ra, rope_head_dim, ro)
    kv_state_out = kv_state.clone()
    score_state_out = score_state.clone()

    kv = golden_compress(x, sin, cos, wkv, wgate, ape, weight, \
                         kv_state, score_state, block_table, block_table, hadamard, ra, st, rope_head_dim, ro)
    assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    if kv is not None:
        assert_allclose(
            out.cpu().float().numpy(),
            kv.cpu().float().numpy(),
            rtol=0.0078125,
            atol=1e-4,
        )

    print("test_compressor_decode passed!")


def test_comp_4():
    """Test Compressor"""
    print("Test: Compressor")
    print("=" * 60)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True

    ra = 4
    ro = False

    bsz = 64
    st = torch.tensor([2] * bsz, dtype=torch.int32, device=device)
    print(f"test_compressor_decode (ratio: {ra}, rotate: {ro}) begin!")
    seq = 2
    h = 4096
    d = 512
    rope_head_dim = 64
    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, block_table, hadamard = gen_inputs(
        bsz, seq, h, d, rope_head_dim, ra, device
    )

    out = compressor_pypto(
        x, kv_state, score_state, block_table, block_table, sin, cos, wkv, wgate, ape, weight, hadamard,
        st, ra, rope_head_dim, ro
    )
    kv_state_out = kv_state.clone()
    score_state_out = score_state.clone()

    kv = golden_compress(
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, block_table, block_table,
        hadamard, ra, st, rope_head_dim, ro
    )

    assert_allclose(kv_state_out.cpu().float().numpy(), kv_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    assert_allclose(score_state_out.cpu().float().numpy(), score_state.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    if kv is not None:
        assert_allclose(
            out.cpu().float().numpy(),
            kv.cpu().float().numpy(),
            rtol=0.0078125,
            atol=1e-4,
        )

    print("test_compressor_decode passed!")


def test_comp_indexer():
    """Test Compressor"""
    print("=" * 60)
    print("Test: Compressor")
    print("=" * 60)

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    
    ra = 4
    ro = True

    bsz = 64
    st = torch.tensor([1] * bsz, dtype=torch.int32, device=device)
    print(f"test_compressor_decode (ratio: {ra}, rotate: {ro}) begin!")
    seq = 2
    h = 4096
    d = 128
    rope_head_dim = 64
    x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, block_table, hadamard = gen_inputs(
        bsz, seq, h, d, rope_head_dim, ra, device
    )

    out = compressor_pypto(
        x, kv_state, score_state, block_table, block_table, sin, cos, wkv, wgate, ape, weight, hadamard,
        st, ra, rope_head_dim, ro
    )
    kv_state_out = kv_state.clone()
    score_state_out = score_state.clone()

    kv = golden_compress(
        x, sin, cos, wkv, wgate, ape, weight, kv_state, score_state, block_table, block_table,
        hadamard, ra, st, rope_head_dim, ro
    )
    assert_allclose(
        kv_state_out.cpu().float().numpy(),
        kv_state.cpu().float().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        score_state_out.cpu().float().numpy(),
        score_state.cpu().float().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    if kv is not None:
        assert_allclose(
            out.cpu().float().numpy(),
            kv.cpu().float().numpy(),
            rtol=0.0078125,
            atol=1e-4,
        )

    print("test_compressor_decode passed!")


if __name__ == "__main__":
    test_comp_128()
    test_comp_4()
    test_comp_indexer()
