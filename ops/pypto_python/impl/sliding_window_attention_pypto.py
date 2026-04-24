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
"""Sliding Window Attention implementation for NPU acceleration.

This module implements sliding window attention mechanism for deepseek_v4 model,
designed to reduce computational cost by focusing attention within a local window.

Main Functions:
    - npu_sliding_window_attention: Main sliding window attention function
    - get_mask: Generate attention mask for sliding window
"""

import torch
import pypto
from torch._dynamo import allow_in_graph


def check_args_tnd(
    q_tnd: torch.Tensor,
    block_table: torch.Tensor,
    kv_cache: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    win_size: int,
    cu_seqlens_q: torch.Tensor,
):
    assert (
        q_tnd is not None
        and block_table is not None
        and kv_cache is not None
        and seqused_kv is not None
        and sinks is not None
        and cu_seqlens_q is not None
    )

    assert (
        q_tnd.dtype == torch.bfloat16
        and q_tnd.ndim == 3
        and q_tnd.shape[1] == 64
        and q_tnd.shape[2] == 512
    ), (
        f"q dtype is {q_tnd.dtype}, ndim is {q_tnd.ndim}, "
        f"axis2 is {q_tnd.shape[1]}, axis3 is {q_tnd.shape[2]}"
    )

    assert block_table.ndim == 2, f"block_table ndim is {block_table.ndim}"

    assert (
        kv_cache.dtype == torch.bfloat16
        and kv_cache.ndim == 4
        and kv_cache.shape[1] == 128
        and kv_cache.shape[2] == 1
        and kv_cache.shape[3] == 512
    ), (
        f"kv_cache dtype is {kv_cache.dtype}, ndim is {kv_cache.ndim}, "
        f"axis2 is {kv_cache.shape[1]}, axis3 is {kv_cache.shape[2]}, "
        f"axis4 is {kv_cache.shape[3]}"
    )

    assert sinks.dtype == torch.float32 and sinks.ndim == 1 and sinks.shape[0] == 64, (
        f"sinks dtype is {sinks.dtype}, ndim is {sinks.ndim}, axis1 is {sinks.shape[0]}"
    )

    assert seqused_kv.dtype == torch.int and seqused_kv.ndim == 1, (
        f"seqused_kv dtype is {seqused_kv.dtype}, ndim is {seqused_kv.ndim}"
    )

    assert win_size == 128, f"win_size is {win_size}"

    assert (
        cu_seqlens_q.dtype == torch.int
        and cu_seqlens_q.ndim == 1
        and cu_seqlens_q.shape[0] == seqused_kv.shape[0] + 1
    ), (
        f"cu_seqlens_q dtype is {cu_seqlens_q.dtype}, ndim is {cu_seqlens_q.ndim}, "
        f"axis1 is {cu_seqlens_q.shape[0]}, seqused_kv axis1 is {seqused_kv.shape[0]}"
    )


@pypto.jit(
    runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_function_num_initial": 128,
        "triple_stream_sched": True
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 4},
        "cube_nbuffer_setting": {1: 2},
        "vec_nbuffer_mode": 2,
        "vec_nbuffer_setting": {-1: 4},
    },
    debug_options={"runtime_debug_mode": 2},
)
def sliding_window_attention_kernel(
    q_tnd,
    block_table,
    kv_cache,
    seqused_kv_list,
    sinks,
    cu_seqlens_q,
    mask2,
    atten_out,
    win,
):
    pypto.experimental.set_operation_config(combine_axis=True)
    t = q_tnd.shape[0]
    n_q = q_tnd.shape[1]
    d_q = q_tnd.shape[2]
    scalar = d_q**-0.5
    block_size = kv_cache.shape[1]
    d_kv = kv_cache.shape[3]
    b = seqused_kv_list.shape[0]
    dtype = q_tnd.dtype

    q_2d = pypto.reshape(q_tnd, [t * n_q, d_q], inplace=True)
    kv_2d = pypto.reshape(
        kv_cache,
        [kv_cache.shape[0] * block_size * kv_cache.shape[2], d_kv],
        inplace=True,
    )

    for b_idx in pypto.loop(b, name="LOOP_B", idx_name="B_idx"):
        cur_s_q = cu_seqlens_q[b_idx + 1] - cu_seqlens_q[b_idx]
        t_start_idx = cu_seqlens_q[b_idx]
        group_size = 2
        groups_num = pypto.ceil(cur_s_q, group_size)

        for g_idx in pypto.loop(groups_num, name="LOOP_G", idx_name="G_idx"):
            group_start_t_idx = t_start_idx + g_idx * group_size
            group_end_t_idx = pypto.min(
                t_start_idx + g_idx * group_size + 3, t_start_idx + cur_s_q - 1
            )
            valid_group_len = group_end_t_idx - group_start_t_idx + 1

            cur_offset = group_start_t_idx * n_q
            actual_seq = seqused_kv_list[b_idx]
            pypto.set_vec_tile_shapes(128, 512)
            q_tensor_cur = pypto.view(
                q_2d,
                [group_size * n_q, d_q],
                [cur_offset, 0],
                valid_shape=[valid_group_len * n_q, d_q],
            )

            group_end_s1_idx = g_idx * group_size + valid_group_len - 1

            cur_end_s2_pos = actual_seq - (cur_s_q - 1) + group_end_s1_idx - 1
            cur_start_s2_pos = pypto.max(
                0, cur_end_s2_pos - win - (valid_group_len - 1) + 1
            )
            start_block = cur_start_s2_pos // block_size
            start_block_offset = cur_start_s2_pos % block_size
            end_block = cur_end_s2_pos // block_size
            end_block_offset = cur_end_s2_pos % block_size

            pypto.set_vec_tile_shapes(128, 128)
            atten_sink_2d = pypto.reshape(sinks, [sinks.shape[0], 1], inplace=True)
            atten_sink_2d = pypto.concat([atten_sink_2d] * group_size, dim=0)
            atten_sink_2d_temp = atten_sink_2d

            if pypto.cond(start_block + 2 == end_block):
                acc_s = pypto.tensor(
                    [group_size * n_q, block_size * 3], pypto.DT_FP32, "acc_s"
                )
                kv_block_temp = pypto.tensor(
                    [block_size * 3, d_kv], dtype, "kv_block_temp"
                )

                start_block_id = block_table[b_idx, start_block]
                kv_block_0 = pypto.view(
                    kv_2d, [block_size, d_kv], [start_block_id * block_size, 0]
                )
                mid_block_id = block_table[b_idx, start_block + 1]
                kv_block_1 = pypto.view(
                    kv_2d, [block_size, d_kv], [mid_block_id * block_size, 0]
                )
                end_block_id = block_table[b_idx, end_block]
                kv_block_2 = pypto.view(
                    kv_2d, [block_size, d_kv], [end_block_id * block_size, 0]
                )

                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                acc_s_0 = pypto.matmul(
                    q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True
                )
                acc_s_1 = pypto.matmul(
                    q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True
                )
                acc_s_2 = pypto.matmul(
                    q_tensor_cur, kv_block_2, pypto.DT_FP32, b_trans=True
                )
                pypto.assemble(acc_s_0, [0, 0], acc_s)
                pypto.assemble(acc_s_1, [0, block_size], acc_s)
                pypto.assemble(acc_s_2, [0, block_size * 2], acc_s)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(
                    mask2,
                    [group_size * n_q, block_size * 3],
                    [0, 128 - start_block_offset],
                    valid_shape=[valid_group_len * n_q, block_size * 3],
                )
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

                acc_s = pypto.mul(acc_s, scalar)
                scores_max = pypto.amax(acc_s, -1, True)
                sub_res = pypto.sub(acc_s, scores_max)
                acc_s = pypto.exp(sub_res)

                sum_exp = pypto.sum(acc_s, -1, True)
                sub_res = pypto.sub(atten_sink_2d_temp, scores_max)
                atten_sink_exp = pypto.exp(sub_res)
                sum_exp = pypto.add(sum_exp, atten_sink_exp)

                div_res = pypto.div(acc_s, sum_exp)
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
                pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
                pypto.assemble(kv_block_2, [block_size * 2, 0], kv_block_temp)
                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)

            elif pypto.cond(start_block + 1 == end_block):
                acc_s = pypto.tensor(
                    [group_size * n_q, block_size * 2], pypto.DT_FP32, "acc_s"
                )
                kv_block_temp = pypto.tensor(
                    [block_size * 2, d_kv], dtype, "kv_block_temp"
                )

                start_block_id = block_table[b_idx, start_block]
                kv_block_0 = pypto.view(
                    kv_2d, [block_size, d_kv], [start_block_id * block_size, 0]
                )
                end_block_id = block_table[b_idx, end_block]
                kv_block_1 = pypto.view(
                    kv_2d, [block_size, d_kv], [end_block_id * block_size, 0]
                )

                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                acc_s_0 = pypto.matmul(
                    q_tensor_cur, kv_block_0, pypto.DT_FP32, b_trans=True
                )
                acc_s_1 = pypto.matmul(
                    q_tensor_cur, kv_block_1, pypto.DT_FP32, b_trans=True
                )
                pypto.assemble(acc_s_0, [0, 0], acc_s)
                pypto.assemble(acc_s_1, [0, block_size], acc_s)

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(
                    mask2,
                    [group_size * n_q, block_size * 2],
                    [0, 128 - start_block_offset],
                    valid_shape=[valid_group_len * n_q, block_size * 2],
                )
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

                acc_s = pypto.mul(acc_s, scalar)
                scores_max = pypto.amax(acc_s, -1, True)
                sub_res = pypto.sub(acc_s, scores_max)
                acc_s = pypto.exp(sub_res)

                sum_exp = pypto.sum(acc_s, -1, True)
                sub_res = pypto.sub(atten_sink_2d_temp, scores_max)
                atten_sink_exp = pypto.exp(sub_res)
                sum_exp = pypto.add(sum_exp, atten_sink_exp)

                div_res = pypto.div(acc_s, sum_exp)
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.assemble(kv_block_0, [0, 0], kv_block_temp)
                pypto.assemble(kv_block_1, [block_size, 0], kv_block_temp)
                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                mm2_res = pypto.matmul(div_res_b16, kv_block_temp, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)

            elif pypto.cond(start_block == end_block):
                start_block_id = block_table[b_idx, start_block]
                kv_block = pypto.view(
                    kv_2d, [block_size, d_kv], [start_block_id * block_size, 0]
                )

                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                acc_s = pypto.matmul(
                    q_tensor_cur, kv_block, pypto.DT_FP32, b_trans=True
                )

                pypto.set_vec_tile_shapes(64, 256)
                mask_block = pypto.view(
                    mask2,
                    [group_size * n_q, block_size],
                    [0, 255 + valid_group_len - 1 - end_block_offset],
                    valid_shape=[valid_group_len * n_q, block_size],
                )
                acc_s = pypto.where(mask_block, acc_s, float("-inf"))

                acc_s = pypto.mul(acc_s, scalar)
                scores_max = pypto.amax(acc_s, -1, True)
                sub_res = pypto.sub(acc_s, scores_max)
                acc_s = pypto.exp(sub_res)

                sum_exp = pypto.sum(acc_s, -1, True)
                sub_res = pypto.sub(atten_sink_2d_temp, scores_max)
                atten_sink_exp = pypto.exp(sub_res)
                sum_exp = pypto.add(sum_exp, atten_sink_exp)

                div_res = pypto.div(acc_s, sum_exp)
                div_res_b16 = pypto.cast(div_res, dtype)

                pypto.set_cube_tile_shapes(
                    [128, 128], [128, 128], [256, 256], False, False
                )
                mm2_res = pypto.matmul(div_res_b16, kv_block, dtype)

                pypto.assemble(mm2_res, [cur_offset, 0], atten_out)


@allow_in_graph
def npu_sliding_window_attention(
    q: torch.Tensor,
    ori_block_table: torch.Tensor,
    ori_kv: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    win_size: int,
    mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    check_args_tnd(
        q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, cu_seqlens_q
    )
    atten_out = torch.empty(
        [q.shape[0] * q.shape[1], q.shape[2]], dtype=q.dtype, device=q.device
    )

    inputs = {
        q: [0],
        ori_block_table: [0],
        ori_kv: [0],
        seqused_kv: [0],
        sinks: [],
        cu_seqlens_q: [0],
        mask: [],
    }
    outputs = {
        atten_out: [0],
    }

    pto_inputs = [
        pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()
    ]
    pto_outputs = [
        pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()
    ]

    sliding_window_attention_kernel(*pto_inputs, *pto_outputs, win_size)

    atten_out = atten_out.reshape(q.shape)
    return atten_out


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "sliding_window_attention(Tensor q, Tensor ori_block_table, Tensor ori_kv, "
    "Tensor seqused_kv, Tensor sinks, int win_size, Tensor mask, "
    "Tensor cu_seqlens_q) -> (Tensor)"
)


@torch.library.impl(pyptolib, "sliding_window_attention", "Meta")
def sliding_window_attention(
    q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, mask, cu_seqlens_q
):
    y = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    return y


@torch.library.impl(pyptolib, "sliding_window_attention", "NPU")
def sliding_window_attention(
    q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, mask, cu_seqlens_q
):
    return npu_sliding_window_attention(
        q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, mask, cu_seqlens_q
    )


def sliding_window_attention_pypto(
    q: torch.Tensor,
    ori_block_table: torch.Tensor,
    ori_kv: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    win_size: int,
    mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    atten_out = torch.ops.pypto.sliding_window_attention(
        q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, mask, cu_seqlens_q
    )
    return atten_out


def get_mask(s_q, n_q, device, block_size):
    mask_left = torch.zeros((s_q * n_q, 128), dtype=torch.uint8, device=device)
    mask_tail = torch.zeros((s_q * n_q, 128), dtype=torch.uint8, device=device)
    row_indices = torch.arange(s_q, device=device).unsqueeze(1)
    col_indices = torch.arange(block_size * 2, device=device).unsqueeze(0)
    mask_right = (col_indices >= row_indices) & (col_indices < row_indices + 128).to(
        torch.uint8
    )
    mask_right = mask_right.unsqueeze(1).expand(-1, n_q, -1).reshape(s_q * n_q, 256)
    mask = torch.cat([mask_left, mask_right, mask_tail], -1).to(torch.bool)
    return mask
