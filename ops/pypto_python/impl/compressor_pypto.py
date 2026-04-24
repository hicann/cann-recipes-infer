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
"""
Compressor implementation for NPU acceleration.

This module implements compressor mechanism for deepseek_v4 model,
designed to reduce memory usage by compressing key-value cache with attention scoring.

Main Functions:
    - compressor: Main compressor function
    - check_args: Argument validation for compressor function
"""

import pypto
import torch
from torch._dynamo import allow_in_graph
from dataclasses import dataclass
from typing import List


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "compressor(Tensor x, Tensor kv_state, Tensor score_state, Tensor kv_block_table, \
    Tensor score_block_table, Tensor sin, Tensor cos, Tensor wkv, Tensor wgate, Tensor ape, Tensor weight, \
    Tensor hadamard, Tensor start_pos, int ratio, int rope_head_dim, bool rotate) -> (Tensor)"
)


@torch.library.impl(pyptolib, "compressor", "Meta")
def compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate,
):
    out = torch.empty((min(x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] // ratio + x.shape[0]), weight.shape[0]),
        dtype=x.dtype, device=x.device)
    return out


@torch.library.impl(pyptolib, "compressor", "NPU")
def compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate,
):
    return npu_compressor(
        x,
        kv_state,
        score_state,
        kv_block_table,
        score_block_table,
        sin,
        cos,
        wkv,
        wgate,
        ape,
        weight,
        hadamard,
        start_pos,
        ratio,
        rope_head_dim,
        rotate,
    )


def compressor_pypto(
    x,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate,
):
    return torch.ops.pypto.compressor(
        x,
        kv_state,
        score_state,
        kv_block_table,
        score_block_table,
        sin,
        cos,
        wkv,
        wgate,
        ape,
        weight,
        hadamard,
        start_pos,
        ratio,
        rope_head_dim,
        rotate,
    )


@allow_in_graph
def npu_compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate,
):
    check_args(
        x, kv_state, score_state, kv_block_table, score_block_table, sin, cos, wkv, wgate, ape, weight, hadamard,
        start_pos, ratio, rope_head_dim, rotate
    )

    out = torch.zeros((min(x.shape[0] * x.shape[1], x.shape[0] * x.shape[1] // ratio + x.shape[0]), weight.shape[0]),
        dtype=x.dtype, device=x.device)

    x = pypto.from_torch(x, dynamic_axis=[0])
    kv_state_in = pypto.from_torch(kv_state, dynamic_axis=[])
    score_state_in = pypto.from_torch(score_state, dynamic_axis=[])
    kv_block_table = pypto.from_torch(kv_block_table, dynamic_axis=[0])
    score_block_table = pypto.from_torch(score_block_table, dynamic_axis=[0])
    sin = pypto.from_torch(sin, dynamic_axis=[0])
    cos = pypto.from_torch(cos, dynamic_axis=[0])
    wkv = pypto.from_torch(wkv, dynamic_axis=[])
    wgate = pypto.from_torch(wgate, dynamic_axis=[])
    ape = pypto.from_torch(ape, dynamic_axis=[])
    weight = pypto.from_torch(weight, dynamic_axis=[])

    hadamard = pypto.from_torch(hadamard, dynamic_axis=[])
    start_pos = pypto.from_torch(start_pos, dynamic_axis=[])
    kv_state_out = pypto.from_torch(kv_state, dynamic_axis=[])
    score_state_out = pypto.from_torch(score_state, dynamic_axis=[])

    out_res = pypto.from_torch(out, dynamic_axis=[0])

    pto_inputs = [
        x,
        kv_state_in,
        score_state_in,
        kv_block_table,
        score_block_table,
        sin,
        cos,
        wkv,
        wgate,
        ape,
        weight,
    ]
    pto_outputs = [out_res]

    if rotate and ratio == 4:
        compressor_ratio_4_rotate_kernel(
            *pto_inputs,
            hadamard,
            *pto_outputs,
            kv_state_out,
            score_state_out,
            ratio,
            start_pos,
            rope_head_dim,
        )
    elif not rotate and ratio == 4:
        compressor_ratio_4_kernel(
            *pto_inputs,
            *pto_outputs,
            kv_state_out,
            score_state_out,
            ratio,
            start_pos,
            rope_head_dim,
        )
    elif not rotate and ratio == 128:
        compressor_ratio_128_kernel(
            *pto_inputs,
            *pto_outputs,
            kv_state_out,
            score_state_out,
            ratio,
            start_pos,
            rope_head_dim,
        )

    return out


def check_args(
    x,
    kv_state,
    score_state,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate,
):
    overlap = ratio == 4
    coff = 1 + overlap
    d = weight.shape[0]
    bsz = x.size(0)
    assert ratio == 4 or ratio == 128, f"ratio is {ratio}, expected 4 or 128"
    assert rope_head_dim == 64, f"rope_head_dim is {rope_head_dim}, expected 64"
    assert isinstance(rotate, bool), f"rotate dtype is {type(rotate)}, expected bool"

    assert weight.dim() == 1 and ((d == 128 and rotate) or (d == 512 and not rotate)), (
        f"weight dim num is {weight.dim()}, weight axis1 is {d}, \
        expected 1, (d = 512 and rotate = False) or (d = 128 and rotate = True)"
    )

    assert x.dim() == 3 and x.size(1) in [1, 2, 3, 4] and x.size(2) == 4096, (
        f"x dim num is {x.dim()}, x axis1 is {x.size(1)}, x axis2 is {x.size(2)}, "
        f"expected 3 dimensions, axis1 in [1, 2, 3, 4], axis2 == 4096"
    )

    assert (
        kv_state.dim() == 3
        and kv_state.size(1) == 128
        and kv_state.size(2) == coff * d
    ), (
        f"kv_state dim num is {kv_state.dim()}, kv_state axis1 is {kv_state.size(1)}, \
        kv_state axis2 is {kv_state.size(2)}, expected 3, 128, {coff * d}"
    )

    assert (
        score_state.dim() == 3
        and score_state.size(1) == 128
        and score_state.size(2) == coff * d
    ), (
        f"score_state dim num is {score_state.dim()}, score_state axis1 is {score_state.size(1)}, \
        score_state axis2 is {score_state.size(2)}, expected 3, 128, {coff * d}"
    )

    assert (
        kv_block_table.dim() == 2
        and kv_block_table.size(0) == bsz
    ), (
        f"kv_block_table dim num is {kv_block_table.dim()}, kv_block_table axis0 is {kv_block_table.size(0)}, \
        expected 2, {bsz}"
    )

    assert (
        score_block_table.dim() == 2
        and score_block_table.size(0) == bsz
    ), (
        f"score_block_table dim num is {score_block_table.dim()}, \
        score_block_table axis0 is {score_block_table.size(0)}, expected 2, {bsz}"
    )

    expected_rope_axis0 = min(bsz * x.size(1), bsz * x.size(1) // ratio + bsz)
    assert sin.dim() == 2 and sin.size(0) == expected_rope_axis0 and sin.size(1) == rope_head_dim, (
        f"sin dim num is {sin.dim()}, sin axis0 is {sin.size(0)}, sin axis1 is {sin.size(1)}, "
        f"expected 2, {expected_rope_axis0}, {rope_head_dim}"
    )

    assert cos.dim() == 2 and cos.size(0) == expected_rope_axis0 and cos.size(1) == rope_head_dim, (
        f"cos dim num is {cos.dim()}, cos axis0 is {cos.size(0)}, cos axis1 is {cos.size(1)}, "
        f"expected 2, {expected_rope_axis0}, {rope_head_dim}"
    )

    assert wkv.dim() == 2 and wkv.size(1) == 4096 and wkv.size(0) == coff * d, (
        f"wkv dim num is {wkv.dim()}, wkv axis0 is {wkv.size(0)}, wkv axis1 is {wkv.size(1)}, \
        expected 2, {coff * d}, 4096"
    )

    assert wgate.dim() == 2 and wgate.size(1) == 4096 and wgate.size(0) == coff * d, (
        f"wgate dim num is {wgate.dim()}, wgate axis0 is {wgate.size(0)}, wgate axis1 is {wgate.size(1)}, \
        expected 2, {coff * d}, 4096"
    )

    assert ape.dim() == 2 and ape.size(0) == ratio and ape.size(1) == coff * d, (
        f"ape dim num is {ape.dim()}, ape axis0 is {ape.size(0)}, ape axis1 is {ape.size(1)}, \
        expected 2, {ratio}, {coff * d}"
    )

    assert hadamard.dim() == 2 and hadamard.size(0) == d and hadamard.size(1) == d, (
        f"hadamard dim num is {hadamard.dim()}, hadamard axis0 is {hadamard.size(0)}, \
        hadamard axis1 is {hadamard.size(1)}, expected 2, {d}, {d}"
    )

    assert start_pos.dim() == 1 and start_pos.size(0) == bsz, (
        f"start_pos dim num is {start_pos.dim()}, start_pos axis0 is {start_pos.size(0)}, expected 1, {bsz}"
    )

    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert cos.dtype == torch.bfloat16, (
        f"cos.dtype is {cos.dtype}, expected torch.bfloat16"
    )
    assert sin.dtype == torch.bfloat16, (
        f"sin.dtype is {sin.dtype}, expected torch.bfloat16"
    )
    assert hadamard.dtype == torch.bfloat16, (
        f"hadamard.dtype is {hadamard.dtype}, expected torch.bfloat16"
    )
    assert weight.dtype == torch.bfloat16, (
        f"hadamard.dtype is {weight.dtype}, expected torch.bfloat16"
    )
    assert kv_state.dtype == torch.float32, (
        f"kv_state.dtype is {kv_state.dtype}, expected torch.float32"
    )
    assert score_state.dtype == torch.float32, (
        f"score_state.dtype is {score_state.dtype}, expected torch.float32"
    )
    assert kv_block_table.dtype == torch.int32, (
        f"kv_block_table.dtype is {kv_block_table.dtype}, expected torch.int32"
    )
    assert score_block_table.dtype == torch.int32, (
        f"score_block_table.dtype is {score_block_table.dtype}, expected torch.int32"
    )
    assert start_pos.dtype == torch.int32, (
        f"start_pos.dtype is {start_pos.dtype}, expected torch.int32"
    )
    assert wkv.dtype == torch.bfloat16, (
        f"wkv.dtype is {wkv.dtype}, expected torch.bfloat16"
    )
    assert wgate.dtype == torch.bfloat16, (
        f"wgate.dtype is {wgate.dtype}, expected torch.bfloat16"
    )
    assert ape.dtype == torch.float32, (
        f"ape.dtype is {ape.dtype}, expected torch.float32"
    )


@dataclass
class Rope2dTileConfig:
    two_dim_tile: List[int]
    three_dim_tile: List[int]


def softmax(x: pypto.Tensor, dim) -> pypto.Tensor:
    xmax = pypto.amax(x, dim, keepdim=True)
    xsub = pypto.sub(x, xmax)
    xexp = pypto.exp(xsub)
    xsum = pypto.sum(xexp, dim, keepdim=True)
    xdiv = pypto.div(xexp, xsum)
    return xdiv


def rms_norm(
    input_tensor: pypto.Tensor, gamma: pypto.Tensor, epsilon=1e-6
) -> pypto.Tensor:
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    shape = [1] * dim
    shape[dim - 1] = gamma.shape[0]
    gamma_cast = pypto.reshape(gamma, shape)
    gamma_fp32 = pypto.cast(gamma_cast, pypto.DT_FP32)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    y = pypto.mul(gamma_fp32, y)
    y = pypto.cast(y, input_tensor.dtype)
    return y


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    chunk_size = 2
    shape = input_tensor.shape
    shape_size = len(shape)
    shape[shape_size - 1] //= chunk_size
    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]
    x1 = pypto.view(input_tensor, shape, offset1)
    x2 = pypto.view(input_tensor, shape, offset2)
    return pypto.concat([x2 * (-1.0), x1], -1)


def interleaved_rope_2d(
    x: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    rope_2d_config: Rope2dTileConfig,
):
    pypto.set_vec_tile_shapes(*rope_2d_config.two_dim_tile)  # (1, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(*rope_2d_config.three_dim_tile)  # (1, 128, 128)
    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1] // 2, 2])
    x_trans = pypto.transpose(x_view, 1, 2)
    x_trans = pypto.reshape(x_trans, x.shape)
    x_trans = rotate_half(x_trans)
    x_trans_reshape = pypto.reshape(x_trans, [x.shape[0], 2, x.shape[1] // 2])
    x_trans_embed = pypto.transpose(x_trans_reshape, 1, 2)
    x_second = pypto.reshape(x_trans_embed, x.shape)

    x_embed = cast_x * cast_cos + x_second * cast_sin

    return pypto.cast(x_embed, x.dtype)


def scatter_update_3d(input, index, src):
    input_shape = input.shape
    d = src.shape[2]
    pypto.set_vec_tile_shapes(1, 64, d)
    src = pypto.reshape(src, [src.shape[0] * src.shape[1], src.shape[2]])
    input = pypto.reshape(input, [input.shape[0] * input.shape[1], input.shape[2]])
    pypto.set_vec_tile_shapes(64, d)
    output = pypto.scatter_update(input, -2, index, src)
    return pypto.reshape(output, input_shape)


@pypto.jit(
    pass_options={},
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "device_sched_mode": 3,
        "triple_stream_sched": True
    },
    debug_options=dict(compile_debug_mode=0, runtime_debug_mode=0),
)
def compressor_ratio_4_kernel(
    x,
    kv_state_total,
    score_state_total,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    start_pos_dy,
    rope_head_dim,
):
    dtype = x.dtype
    bsz, s1, h = x.shape
    x = pypto.reshape(x, [bsz * s1, h], inplace=True)
    ratio = 4
    coff = 2
    d = 512
    block_size = kv_state_total.shape[1]
    pypto.set_vec_tile_shapes(block_size)
    cache_index = pypto.arange(block_size)

    b = 64
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz - b_idx * b).min(b)
        x_view = pypto.view(x, [b * s1, h], [b_idx * b * s1, 0])

        # Matmul
        pypto.set_cube_tile_shapes([128, 128], [256, 512], [128, 128], True)
        pypto.set_vec_tile_shapes(16, 2, 1024)
        kv_t = pypto.matmul(x_view, wkv, pypto.DT_FP32, b_trans=True)  # b*s,2d
        score_t = pypto.matmul(x_view, wgate, pypto.DT_FP32, b_trans=True)

        for _ in pypto.loop(1):
            pypto.set_pass_options(pg_skip_partition=True)
            kv_t = pypto.reshape(kv_t, [b, s1, coff*d], inplace=True)
            score_t = pypto.reshape(score_t, [b, s1, coff*d], inplace=True)
            cache_index = pypto.reshape(cache_index, [1, block_size], inplace=True)

        for c_idx in pypto.loop(b_valid, name="LOOP_COMP_2", idx_name="c_idx"):
            pypto.set_pass_options(pg_skip_partition=True)
            start_pos = start_pos_dy[b_idx * b + c_idx]
            # No compression
            if pypto.cond(start_pos % ratio + s1 < ratio):
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, coff * d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, coff * d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 1024)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 1024)
                    score = pypto.add(score, ape_view)  # b,1,2d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.set_vec_tile_shapes(1, 1, 1024)
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
            # Compression exists
            else:
                pypto.set_vec_tile_shapes(1, 16, 1024)
                kv_block_idx = kv_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                score_block_idx = score_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                start = ((start_pos // ratio) * ratio) % block_size
                kv_state = pypto.view(
                    kv_state_total, [1, ratio, coff * d], [kv_block_idx, start, 0]
                )
                score_state = pypto.view(
                    score_state_total, [1, ratio, coff * d], [score_block_idx, start, 0]
                )

                if pypto.cond(start_pos < ratio):
                    pre_kv_state = pypto.full([1, ratio, d], 0.0, pypto.DT_FP32)
                    pre_score_state = pypto.full(
                        [1, ratio, d], float("-inf"), pypto.DT_FP32
                    )
                else:
                    pre_start = ((start_pos // ratio) * ratio - ratio) % block_size
                    pre_kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos - ratio) // block_size
                    ]
                    pre_score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos - ratio) // block_size
                    ]
                    pre_kv_state = pypto.view(
                        kv_state_total, [1, ratio, d], [pre_kv_block_idx, pre_start, 0]
                    )
                    pre_score_state = pypto.view(
                        score_state_total,
                        [1, ratio, d],
                        [pre_score_block_idx, pre_start, 0],
                    )

                # Only adapt to s1 <= ratio
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, coff * d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, coff * d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 1024)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 1024)
                    score = pypto.add(score, ape_view)  # b,1,2d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.set_vec_tile_shapes(1, 1, 1024)
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
                    if pypto.cond(pos <= ratio - 1):
                        index = cache_index[:, pos : pos + 1]
                        kv_state = scatter_update_3d(kv_state, index, kv)  # b,4,2d
                        score_state = scatter_update_3d(score_state, index, score)
                        if pypto.cond(pos == ratio - 1):
                            pypto.set_vec_tile_shapes(1, 8, 1024)
                            kv_state_tmp = pypto.concat(
                                [pre_kv_state, kv_state[:, :, d:]], 1
                            )  # b,8,d
                            score_state_tmp = pypto.concat(
                                [pre_score_state, score_state[:, :, d:]], 1
                            )  # b,8,d
                            kv = kv_state_tmp * softmax(score_state_tmp, 1)
                            kv = pypto.sum(kv, 1)  # b,d

                            # RMSNorm/RoPE
                            pypto.set_vec_tile_shapes(1, 512)
                            kv = rms_norm(pypto.cast(kv, dtype), weight)  # b,d

                            kv_nope = kv[:, : d - rope_head_dim]
                            kv_rope = kv[:, d - rope_head_dim :]
                            sin_tile = pypto.view(
                                sin, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            cos_tile = pypto.view(
                                cos, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            rope2d_tile_config = Rope2dTileConfig(
                                [1, 64], [1, 128, 128]
                            )
                            kv_rope = interleaved_rope_2d(
                                kv_rope, cos_tile, sin_tile, rope2d_tile_config
                            )
                            pypto.set_vec_tile_shapes(1, 512)
                            kv = pypto.concat([kv_nope, kv_rope], dim=-1)  # b,d
                            pypto.assemble(kv, [b_idx * b + c_idx, 0], out)


@pypto.jit(
    pass_options={},
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "device_sched_mode": 3,
        "triple_stream_sched": True
    },
    debug_options=dict(compile_debug_mode=0, runtime_debug_mode=0),
)
def compressor_ratio_4_rotate_kernel(
    x,
    kv_state_total,
    score_state_total,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    start_pos_dy,
    rope_head_dim,
):
    dtype = x.dtype
    bsz, s1, h = x.shape
    x = pypto.reshape(x, [bsz * s1, h], inplace=True)
    ratio = 4
    coff = 2
    d = 128
    block_size = kv_state_total.shape[1]
    pypto.set_vec_tile_shapes(block_size)
    cache_index = pypto.arange(block_size)
    out_t = pypto.Tensor([bsz, d], pypto.DT_BF16)
    pypto.set_vec_tile_shapes(1, 1)
    is_compress = pypto.SymbolicScalar(0)
    pypto.set_vec_tile_shapes(1, 256)
    zero = pypto.full([1, d], 0.0, pypto.DT_FP32)
    zero = pypto.cast(zero, pypto.DT_BF16)

    b = 64
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz - b_idx * b).min(b)
        x_view = pypto.view(x, [b * s1, h], [b_idx * b * s1, 0])

        # Matmul
        pypto.set_cube_tile_shapes([128, 128], [256, 512], [64, 64], True)
        pypto.set_vec_tile_shapes(64, 2, 256)
        kv_t = pypto.matmul(x_view, wkv, pypto.DT_FP32, b_trans=True)  # b*s,2d
        score_t = pypto.matmul(x_view, wgate, pypto.DT_FP32, b_trans=True)

        for _ in pypto.loop(1):
            pypto.set_pass_options(pg_skip_partition=True)
            kv_t = pypto.reshape(kv_t, [b, s1, coff*d], inplace=True)
            score_t = pypto.reshape(score_t, [b, s1, coff*d], inplace=True)
            cache_index = pypto.reshape(cache_index, [1, block_size], inplace=True)

        for c_idx in pypto.loop(b_valid, name="LOOP_COMP_2", idx_name="c_idx"):
            pypto.set_pass_options(pg_skip_partition=True)
            start_pos = start_pos_dy[b_idx * b + c_idx]
            # No compression
            if pypto.cond(start_pos % ratio + s1 < ratio):
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, coff * d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, coff * d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 256)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 256)
                    score = pypto.add(score, ape_view)  # b,1,2d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.set_vec_tile_shapes(1, 1, 256)
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
                pypto.set_vec_tile_shapes(1, 256)
                pypto.assemble(zero, [b_idx * b + c_idx, 0], out_t)

            # Compression exists
            else:
                is_compress = pypto.SymbolicScalar("is_compress") + 1
                pypto.set_vec_tile_shapes(1, 16, 256)
                kv_block_idx = kv_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                score_block_idx = score_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                start = ((start_pos // ratio) * ratio) % block_size
                kv_state = pypto.view(
                    kv_state_total, [1, ratio, coff * d], [kv_block_idx, start, 0]
                )
                score_state = pypto.view(
                    score_state_total, [1, ratio, coff * d], [score_block_idx, start, 0]
                )

                if pypto.cond(start_pos < ratio):
                    pre_kv_state = pypto.full([1, ratio, d], 0.0, pypto.DT_FP32)
                    pre_score_state = pypto.full(
                        [1, ratio, d], float("-inf"), pypto.DT_FP32
                    )
                else:
                    pre_start = ((start_pos // ratio) * ratio - ratio) % block_size
                    pre_kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos - ratio) // block_size
                    ]
                    pre_score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos - ratio) // block_size
                    ]
                    pre_kv_state = pypto.view(
                        kv_state_total, [1, ratio, d], [pre_kv_block_idx, pre_start, 0]
                    )
                    pre_score_state = pypto.view(
                        score_state_total,
                        [1, ratio, d],
                        [pre_score_block_idx, pre_start, 0],
                    )
                # Only adapt to s1 <= ratio
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, coff * d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, coff * d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 256)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 256)
                    score = pypto.add(score, ape_view)  # b,1,2d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.set_vec_tile_shapes(1, 1, 256)
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
                    if pypto.cond(pos <= ratio - 1):
                        index = cache_index[:, pos : pos + 1]
                        kv_state = scatter_update_3d(kv_state, index, kv)  # b,4,2d
                        score_state = scatter_update_3d(score_state, index, score)
                        if pypto.cond(pos == ratio - 1):
                            pypto.set_vec_tile_shapes(1, 8, 256)
                            kv_state_tmp = pypto.concat(
                                [pre_kv_state, kv_state[:, :, d:]], 1
                            )  # b,8,d
                            score_state_tmp = pypto.concat(
                                [pre_score_state, score_state[:, :, d:]], 1
                            )  # b,8,d
                            kv = kv_state_tmp * softmax(score_state_tmp, 1)
                            kv = pypto.sum(kv, 1)  # b,d

                            # RMSNorm/RoPE
                            pypto.set_vec_tile_shapes(1, 256)
                            kv = rms_norm(pypto.cast(kv, dtype), weight)  # b,d

                            kv_nope = kv[:, : d - rope_head_dim]
                            kv_rope = kv[:, d - rope_head_dim :]
                            sin_tile = pypto.view(
                                sin, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            cos_tile = pypto.view(
                                cos, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            rope2d_tile_config = Rope2dTileConfig(
                                [1, 64], [1, 128, 128]
                            )
                            kv_rope = interleaved_rope_2d(
                                kv_rope, cos_tile, sin_tile, rope2d_tile_config
                            )
                            pypto.set_vec_tile_shapes(1, 256)
                            kv = pypto.concat([kv_nope, kv_rope], dim=-1)  # b,d
                            pypto.assemble(kv, [b_idx * b + c_idx, 0], out_t)
    if pypto.cond(is_compress > 0):
        for _ in pypto.loop(1, submit_before_loop=True):
            continue
        for b_idx in pypto.loop(b_loop, name="LOOP_HADAMARD", idx_name="b_idx"):
            b_valid = (bsz - b_idx * b).min(b)
            pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128], True)
            pypto.set_vec_tile_shapes(64, 128)
            out_view = pypto.view(
                out_t, [b, d], [b_idx * b, 0], valid_shape=[b_valid, d]
            )
            out_view = pypto.matmul(out_view, hadamard, pypto.DT_BF16)  # b,d
            pypto.assemble(out_view, [b_idx * b, 0], out)


@pypto.jit(
    pass_options={},
    runtime_options={
        "stitch_function_num_initial": 128,
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "device_sched_mode": 3,
        "triple_stream_sched": True
    },
    debug_options=dict(compile_debug_mode=0, runtime_debug_mode=0),
)
def compressor_ratio_128_kernel(
    x,
    kv_state_total,
    score_state_total,
    kv_block_table,
    score_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    out,
    kv_state_out,
    score_state_out,
    ratio,
    start_pos_dy,
    rope_head_dim,
):
    dtype = x.dtype
    bsz, s1, h = x.shape
    x = pypto.reshape(x, [bsz * s1, h], inplace=True)

    ratio = 128
    d = 512
    block_size = kv_state_total.shape[1]
    pypto.set_vec_tile_shapes(block_size)
    cache_index = pypto.arange(block_size)

    b = 64
    b_loop = (bsz + b - 1) // b
    for b_idx in pypto.loop(b_loop, name="LOOP_COMP_1", idx_name="b_idx"):
        b_valid = (bsz - b_idx * b).min(b)
        x_view = pypto.view(x, [b * s1, h], [b_idx * b * s1, 0])

        # Matmul
        pypto.set_cube_tile_shapes([128, 128], [256, 512], [64, 64], True)
        pypto.set_vec_tile_shapes(32, 2, 512)
        kv_t = pypto.matmul(x_view, wkv, pypto.DT_FP32, b_trans=True)  # b*s,d
        score_t = pypto.matmul(x_view, wgate, pypto.DT_FP32, b_trans=True)

        for _ in pypto.loop(1):
            pypto.set_pass_options(pg_skip_partition=True)
            kv_t = pypto.reshape(kv_t, [b, s1, d], inplace=True)
            score_t = pypto.reshape(score_t, [b, s1, d], inplace=True)
            cache_index = pypto.reshape(cache_index, [1, block_size], inplace=True)

        for c_idx in pypto.loop(b_valid, name="LOOP_COMP_2", idx_name="c_idx"):
            pypto.set_pass_options(pg_skip_partition=True)
            start_pos = start_pos_dy[b_idx * b + c_idx]
            # No compression
            if pypto.cond(start_pos % ratio + s1 < ratio):
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 512)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 512)
                    score = pypto.add(score, ape_view)  # b,1,d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.set_vec_tile_shapes(1, 1, 512)
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
            # Compression exists
            else:
                pypto.set_vec_tile_shapes(1, 32, 512)
                kv_block_idx = kv_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                score_block_idx = score_block_table[
                    b_idx * b + c_idx, start_pos // block_size
                ]
                kv_state = pypto.view(
                    kv_state_total, [1, block_size, d], [kv_block_idx, 0, 0]
                )
                score_state = pypto.view(
                    score_state_total, [1, block_size, d], [score_block_idx, 0, 0]
                )
                for i in range(s1):
                    pos = (start_pos + i) % ratio
                    kv = pypto.view(kv_t, [1, 1, d], [c_idx, i, 0])
                    score = pypto.view(score_t, [1, 1, d], [c_idx, i, 0])
                    pypto.set_vec_tile_shapes(1, 512)
                    ape_view = ape[pos, :]
                    pypto.set_vec_tile_shapes(1, 1, 512)
                    score = pypto.add(score, ape_view)  # b,1,d

                    kv_block_idx = kv_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    score_block_idx = score_block_table[
                        b_idx * b + c_idx, (start_pos + i) // block_size
                    ]
                    cur_pos = (start_pos + i) % block_size
                    pypto.assemble(kv, [kv_block_idx, cur_pos, 0], kv_state_out)
                    pypto.assemble(
                        score, [score_block_idx, cur_pos, 0], score_state_out
                    )
                    if pypto.cond(pos <= ratio - 1):
                        index = cache_index[:, cur_pos : cur_pos + 1]
                        kv_state = scatter_update_3d(kv_state, index, kv)  # b,128,d
                        score_state = scatter_update_3d(score_state, index, score)
                        if pypto.cond(pos == ratio - 1):
                            pypto.set_vec_tile_shapes(1, 128, 128)
                            kv = kv_state * softmax(score_state, 1)
                            kv = pypto.sum(kv, 1)  # b,d

                            # RMSNorm/RoPE
                            pypto.set_vec_tile_shapes(1, 512)
                            kv = rms_norm(pypto.cast(kv, dtype), weight)  # b,d

                            kv_nope = kv[:, : d - rope_head_dim]
                            kv_rope = kv[:, d - rope_head_dim :]
                            sin_tile = pypto.view(
                                sin, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            cos_tile = pypto.view(
                                cos, kv_rope.shape, [b_idx * b + c_idx, 0]
                            )  # b, 1, 64
                            rope2d_tile_config = Rope2dTileConfig(
                                [1, 64], [1, 128, 128]
                            )
                            kv_rope = interleaved_rope_2d(
                                kv_rope, cos_tile, sin_tile, rope2d_tile_config
                            )
                            pypto.set_vec_tile_shapes(1, 512)
                            kv = pypto.concat([kv_nope, kv_rope], dim=-1)  # b,d
                            pypto.assemble(kv, [b_idx * b + c_idx, 0], out)
