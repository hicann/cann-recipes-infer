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
Hello World Example for PyPTO

This example demonstrates the simplest tensor addition.
"""

import pypto
import torch
from torch._dynamo import allow_in_graph


def rms_norm_denom(x: pypto.Tensor, norm_eps=1e-6) -> pypto.Tensor:
    # Compute RMS: sqrt(mean(x^2) + eps)
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / x.shape[-1]
    rms = pypto.sqrt((mean_sq + norm_eps))
    return rms


def sigmoid(x: pypto.Tensor) -> pypto.Tensor:
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul(x, -1.0)
    exp_neg = pypto.exp(x_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    res = pypto.div(ones, exp_neg + 1.0)
    return res


def hc_split_sinkhorn(comb_flag: pypto.Tensor, hc_split_sinkhorn_iters, hc_eps) \
    -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    tile_t, _, _ = comb_flag.shape   # (tile_t, 4, 4)

    if tile_t <= 32:
        pypto.set_vec_tile_shapes(1, 16, 32)
    elif tile_t <= 64:
        pypto.set_vec_tile_shapes(2, 16, 32)
    else:
        pypto.set_vec_tile_shapes(4, 16, 32)

    row_max = pypto.amax(comb_flag, -1, True)   # (tile_t, 4, 1)
    comb_flag = pypto.exp(comb_flag - row_max)  # (tile_t, 4, 4)

    row_sum = pypto.sum(comb_flag, -1, True)    # (tile_t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps    # (tile_t, 4, 4)
    col_sum = pypto.sum(comb_flag, -2, True)    # (tile_t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps)  # (tile_t, 4, 4)

    for _ in range(hc_split_sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True)   # (tile_t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (tile_t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True)   # (tile_t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (tile_t, 4, 4)
    return comb_flag


def hc_split_sinkhorn_trans(comb_flag: pypto.Tensor, hc_split_sinkhorn_iters, hc_eps) \
    -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    sinkhorn_iters = hc_split_sinkhorn_iters
    _, _, tile_t = comb_flag.shape   # (4, 4, tile_t)

    pypto.set_vec_tile_shapes(8, 16, 128)

    row_max = pypto.amax(comb_flag, -2, True)   # (4, 1, tile_t)
    comb_flag = pypto.exp(comb_flag - row_max)  # (4, 4, tile_t)
    row_sum = pypto.sum(comb_flag, -2, True)    # (4, 1, tile_t)
    comb_flag = comb_flag / row_sum + hc_eps    # (4, 4, tile_t)

    col_sum = pypto.sum(comb_flag, -3, True)    # (1, 4, tile_t)
    comb_flag = comb_flag / (col_sum + hc_eps)  # (4, 4, tile_t)

    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-2, keepdim=True)   # (4, 1, tile_t)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (4, 4, tile_t)
        col_sum = comb_flag.sum(-3, keepdim=True)   # (1, 4, tile_t)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (4, 4, tile_t)
    return comb_flag


class HCPreKernelManager:
    def __init__(self):
        self.vec_all_shape = {}
        self.t_vec = [4096, 256, 128, 64, 16, 4, 1]
        self.hc_fn_shape = [24, 4*4096]
        self.hc_scale_shape = [3, ]
        self.hc_base_shape = [24, ]

        for t in self.t_vec:
            x_shape = [t, 4, 4096]
            y_shape = [t, 4096]
            post_shape = [t, 4]
            comb_shape = [t, 4, 4]
            self.vec_all_shape[t] = [x_shape, self.hc_fn_shape, self.hc_scale_shape, self.hc_base_shape, y_shape, post_shape, comb_shape]

    def infer_controlflow_shape(self, *args):
        if not args:
            return [v for v in self.vec_all_shape.values()]

        x_shape = args[0]
        for t in self.t_vec:
            if x_shape[0] >= t:
                return self.vec_all_shape[t]


manager = HCPreKernelManager()


@pypto.jit(
    pass_options={
        "vec_nbuffer_mode": 1,
    },
    runtime_options={
        "stitch_function_inner_memory": 512,
        "stitch_function_outcast_memory": 512,
        "device_sched_mode": 0,
        # for acl graph
        "stitch_cfgcache_size": 2500000,
        "triple_stream_sched": True
    },
    infer_controlflow_shape = manager.infer_controlflow_shape,
)
def hc_pre_kernel(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale_: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor,
                hc_mult: int=4, hc_split_sinkhorn_iters: int=20, hc_eps: float=1e-6
):
    pypto.experimental.set_operation_config(combine_axis=True)

    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]
    mix_hc = (2 + hc) * hc
    split_k = False

    ### check shape
    assert hc == hc_mult, f"hc is {hc}, expected {hc_mult}"
    assert d == 4096, f"d is {d}, expected 4096"
    assert hc_scale_.shape[0] == 3, f"hc_scale.shape[0] is {hc_scale_.shape[0]}, expected 3"

    unroll_list = [256, 64, 16, 4, 1]

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [t, hc*d], inplace=True)
        hc_scale = pypto.reshape(hc_scale_, [3, 1], inplace=True)

    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        tile_t = unrollLength
        pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])
        tile_shapes_1 = [16, 512]
        tile_shape_2 = 64
        if tile_t <= 32:
            split_k = True
            tile_shapes_1 = [1, 16*1024]
            tile_shape_2 = 1
            pypto.set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], \
                                        enable_multi_data_load=True, enable_split_k=False)
        elif tile_t <= 64:
            split_k = True
            tile_shapes_1 = [2, 8*1024]
            tile_shape_2 = 2
            pypto.set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], enable_multi_data_load=True)
        else:
            split_k = False
            tile_shapes_1 = [4, 4*1024]
            tile_shape_2 = 4
            pypto.set_cube_tile_shapes([16, 16], [512, 2*1024], [128, 128], \
                                        enable_multi_data_load=True, enable_split_k=True)

        pypto.set_vec_tile_shapes(tile_shapes_1[0] * 2, tile_shapes_1[1])
        hc_base = pypto.reshape(hc_base_, [1, mix_hc])

        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx, 0])
        pypto.set_pass_options(sg_set_scope = 1)
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        pypto.set_pass_options(sg_set_scope = -1)

        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1])
        pypto.set_pass_options(sg_set_scope = 2)
        rms_res = rms_norm_denom(x_fp32, hc_eps)    # (t, hc*d) -> (t, 1)
        pypto.set_pass_options(sg_set_scope = -1)

        pypto.set_vec_tile_shapes(tile_shape_2, 32)
        if (not split_k):
            # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
            mm_res = pypto.matmul(x_fp32, hc_fn, pypto.DT_FP32, b_trans=True)
        else:
            tile_k = 4*1024
            for k_idx in range(hc*d // tile_k):
                x_view_k = pypto.view(x_fp32, [tile_t, tile_k], [0, k_idx * tile_k])
                hc_fn_k = pypto.view(hc_fn, [mix_hc, tile_k], [0, k_idx * tile_k])
                mm_res_k = pypto.matmul(x_view_k, hc_fn_k, pypto.DT_FP32, b_trans=True)
                if k_idx == 0:
                    mm_res = mm_res_k
                else:
                    mm_res = mm_res + mm_res_k

        pypto.set_vec_tile_shapes(tile_shape_2, 32)
        rms_res = mm_res / rms_res  # t, mix_hc
        hc_scale_hc = hc_scale.expand_clone([3, hc])

        pre = rms_res[:, :hc] * (hc_scale_hc[0:1, :]) + hc_base[:, :hc] # (tile_t, 4)
        pre = sigmoid(pre) + hc_eps # (tile_t, 4)

        pre_3d = pre.reshape([tile_t, hc, 1], inplace=True)
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d]) # (tile_t, hc, d)
        pypto.set_vec_tile_shapes(tile_shapes_1[0], 4, tile_shapes_1[1] // 4)
        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)   # [16,4,8] -> [16,8]
        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1] // 4)
        res_bf16 = pypto.cast(res_fp32, pypto.DT_BF16)
        pypto.assemble(res_bf16, [t_idx, 0], y)

        pypto.set_vec_tile_shapes(tile_shape_2, 32)
        post_ = rms_res[:, hc: 2*hc] * (hc_scale_hc[1:2, :]) + hc_base[:, hc: 2*hc] # (tile_t, 4)
        post_ = sigmoid(post_) * 2.0    # (tile_t, 4)
        pypto.assemble(post_, [t_idx, 0], post)

        hc_scale_hc = hc_scale.expand_clone([3, 4*hc])
        comb_flag = (rms_res[:, 2*hc: ] * (hc_scale_hc[2:3, :]) + hc_base[:, 2*hc: ])
        comb_flag = comb_flag.reshape([tile_t, hc, hc]) # (tile_t, 4, 4)

        # (tile_t, hc), (tile_t, hc), (tile_t, hc, hc)
        comb_ = hc_split_sinkhorn(comb_flag, hc_split_sinkhorn_iters, hc_eps)
        pypto.assemble(comb_, [t_idx, 0, 0], comb)


@pypto.jit(
    pass_options={
        "vec_nbuffer_mode": 1,
    },
    runtime_options={
        "stitch_function_inner_memory": 2048,
        "stitch_function_outcast_memory": 2048,
        "device_sched_mode": 0,
        # for acl graph
        "stitch_cfgcache_size": 2500000
    },
    debug_options=dict(compile_debug_mode=0, runtime_debug_mode=0)
)
def hc_pre_kernel_prefill(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale_: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor,
                hc_mult: int=4, hc_split_sinkhorn_iters: int=20, hc_eps: float=1e-6
):
    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]
    mix_hc = (2 + hc) * hc
    hc_eps = 1e-6
    split_k = False

    ### check shape
    assert hc == 4, f"hc is {hc}, expected 4"
    assert d == 4096, f"d is {d}, expected 4096"
    assert mix_hc == hc_fn.shape[0], f"mix_hc is {hc_fn.shape[0]}, expected 24"
    assert hc_scale_.shape[0] == 3, f"hc_scale.shape[0] is {hc_scale_.shape[0]}, expected 3"

    unroll_list = [128, 1]

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [t, hc*d], inplace=True)
        hc_scale = pypto.reshape(hc_scale_, [1, 3], inplace=True)
    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        tile_t = unrollLength

        tile_shapes_1 = [8, 1024]
        tile_shape_2 = 8
        pypto.set_cube_tile_shapes([16, 16], [256, 1024], [32, 32], enable_multi_data_load=True)

        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1])

        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx, 0])
        hc_base = pypto.reshape(hc_base_, [mix_hc, 1])

        pypto.set_pass_options(sg_set_scope = 1)
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        rms_res = rms_norm_denom(x_fp32)    # (t, hc*d) -> (t, 1)
        pypto.set_pass_options(sg_set_scope = -1)

        pypto.set_vec_tile_shapes(24, 128)
        if (not split_k):
            # (mix_hc, hc*d) @ (t, hc*d)^t = (mix_hc, t)
            mm_res = pypto.matmul(hc_fn, x_fp32, pypto.DT_FP32, b_trans=True)
        else:
            tile_k = 4*1024
            for k_idx in range(hc*d // tile_k):
                x_view_k = pypto.view(x_fp32, [tile_t, tile_k], [0, k_idx * tile_k])
                hc_fn_k = pypto.view(hc_fn, [mix_hc, tile_k], [0, k_idx * tile_k])
                mm_res_k = pypto.matmul(hc_fn_k, x_view_k, pypto.DT_FP32, b_trans=True)
                if k_idx == 0:
                    mm_res = mm_res_k
                else:
                    mm_res = mm_res + mm_res_k

        rms_res = rms_res.reshape([1, tile_t], inplace=True)
        rms_res = mm_res / rms_res
        hc_scale_hc = hc_scale.expand_clone([hc, 3])
        pre = rms_res[:hc, :] * (hc_scale_hc[:, 0:1])
        pre = pre + hc_base[:hc, :] # (4, tile_t)
        print("pre ", pre.shape)
        pre = sigmoid(pre) + hc_eps # (4, tile_t)
        pre = pre.transpose(0, 1)   # (tile_t, 4)

        pre_3d = pre.reshape([tile_t, hc, 1], inplace=True)
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d]) # (tile_t, hc, d)
        pypto.set_vec_tile_shapes(tile_shapes_1[0], 4, tile_shapes_1[1] // 4)
        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)   # [16,4,8] -> [16,8]
        pypto.set_vec_tile_shapes(tile_shapes_1[0], tile_shapes_1[1] // 4)
        res_bf16 = pypto.cast(res_fp32, pypto.DT_BF16)
        pypto.assemble(res_bf16, [t_idx, 0], y)

        pypto.set_vec_tile_shapes(tile_shape_2, 32)
        post_ = rms_res[hc: 2*hc, :] * (hc_scale_hc[:, 1:2]) + hc_base[hc: 2*hc, :] # (4, tile_t)
        post_ = sigmoid(post_) * 2.0    # (4, tile_t)
        post_ = post_.transpose(0, 1)
        pypto.assemble(post_, [t_idx, 0], post)

        hc_scale_hc = hc_scale.expand_clone([4*hc, 3])
        comb_flag = (rms_res[2*hc:, :] * (hc_scale_hc[:, 2:3]) + hc_base[2*hc:, :]) # (16, tile_t)
        comb_flag = comb_flag.reshape([hc, hc, tile_t]) # (4, 4, tile_t)

        comb_ = hc_split_sinkhorn_trans(comb_flag, hc_split_sinkhorn_iters, hc_eps) # (4, 4, tile_t)
        comb_ = comb_.transpose(1, 2)
        pypto.set_vec_tile_shapes(4, 128, 4)
        comb_ = comb_.transpose(0, 1)
        pypto.set_vec_tile_shapes(128, 4, 4)
        pypto.assemble(comb_, [t_idx, 0, 0], comb)



def check_input_output_shape_dtype(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, \
                                    hc_base: torch.Tensor, hc_mult: int=4):
    mix_hc = (2 + hc_mult) * hc_mult

    assert x.dim() == 3 and x.size(1) == hc_mult and x.size(2) == 4096, \
        f"x dim num is {x.dim()}, x axis1 is {x.size(1)}, x axis2 is {x.size(2)}, expected 3, {hc_mult},  4096"
    assert hc_fn.dim() == 2 and hc_fn.size(0) == mix_hc and hc_fn.size(1) == 4 * 4096, \
        f"hc_fn dim num is {hc_fn.dim()}, hc_fn axis0 {hc_fn.size(0)}, hc_fn axis1 {hc_fn.size(1)}, \
            expected 2,  {mix_hc}, 12384"
    assert hc_scale.dim() == 1 and hc_scale.size(0) == 3, f"hc_scale dim num {hc_scale.dim()}, \
            hc_scale axis0 is {hc_scale.size(0)}, expected 1, 3"
    assert hc_base.dim() == 1 and hc_base.size(0) == mix_hc, f"hc_scale dim num {hc_base.dim()}, \
            hc_scale axis0 {hc_base.size(0)}, expected  1, {mix_hc}"

    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert hc_fn.dtype == torch.float32, f"hc_fn.dtype is {hc_fn.dtype}, expected torch.float32"
    assert hc_scale.dtype == torch.float32, f"hc_scale.dtype is {hc_scale.dtype}, expected torch.float32"
    assert hc_base.dtype == torch.float32, f"hc_base.dtype is {hc_base.dtype}, expected torch.float32"


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, int hc_split_sinkhorn_iters, float hc_eps) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(pyptolib, "hc_pre", "Meta")
def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    y = torch.empty([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.empty([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    comb = torch.empty([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{hc_scale.device}')
    return y, post, comb


@torch.library.impl(pyptolib, "hc_pre", "NPU")
def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    return npu_hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)


def hc_pre_pypto(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    return torch.ops.pypto.hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)


@allow_in_graph
def npu_hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, \
                hc_mult: int=4, hc_split_sinkhorn_iters: int=20, hc_eps: float=1e-6)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ### check dtype
    check_input_output_shape_dtype(x, hc_fn, hc_scale, hc_base, hc_mult)

    y = torch.empty([x.size(0), x.size(2)], dtype=x.dtype, device=f'{x.device}')
    post = torch.empty([x.size(0), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')
    comb = torch.empty([x.size(0), x.size(1), x.size(1)], dtype=hc_scale.dtype, device=f'{x.device}')

    in_outs = {
        x: [0],
        hc_fn: None,
        hc_scale: None,
        hc_base: None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    hc_pre_kernel(*pto_in_outs, hc_mult, hc_split_sinkhorn_iters, hc_eps)

    return y, post, comb
