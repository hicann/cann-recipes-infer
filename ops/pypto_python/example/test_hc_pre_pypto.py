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


import os
import sys
import torch
import torch_npu
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, 'impl')
sys.path.insert(0, impl_path)
from hc_pre_pypto import *

from compare import compare

hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 4096, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc


def gen_rms_norm_denom(x):
    _, len = x.shape
    x = x.square()
    x = x.sum(-1, True) / len
    x = x + norm_eps
    x = x.sqrt()
    return x


def gen_sigmoid(x):
    x = -x
    x = x.exp()
    x = 1 / (1 + x)
    return x


def gen_hc_split_sinkhorn(x, hc_scale, hc_base):
    t, _ = x.shape  # (t, 24)

    pre = x[:, :hc] * hc_scale[0] + hc_base[:, :hc] # (t, 4)
    pre = gen_sigmoid(pre) + hc_eps # (t, 4)

    post = x[:, hc: 2*hc] * hc_scale[1] + hc_base[:, hc: 2*hc]  # (t, 4)
    post = 2.0 * gen_sigmoid(post)  # (t, 4)

    comb_flag = (x[:, 2*hc: ] * hc_scale[2] + hc_base[:, 2*hc: ]).reshape(t, hc, hc)    # (t, 4, 4)
    row_max = comb_flag.amax(-1, keepdim=True)  # (t, 4, 1)
    comb_flag = (comb_flag - row_max).exp() # (t, 4, 4)

    row_sum = comb_flag.sum(-1, keepdim=True)   # (t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps    # (t, 4, 4)
    col_sum = comb_flag.sum(-2, keepdim=True)   # (t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps)  # (t, 4, 4)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True)   # (t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True)   # (t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (t, 4, 4)
    return pre, post, comb_flag


def gen_hc_split_sinkhorn_trans(x, hc_scale, hc_base):
    _, t = x.shape  # (24, t)
    hc_base = hc_base.reshape(mix_hc, 1)

    pre = x[:hc, :] * 1.0
    pre = pre * hc_scale[0]
    pre = pre + hc_base[:hc, :] # (4, t)
    pre_ = pre

    pre = gen_sigmoid(pre) + hc_eps # (4, t)

    post = x[hc: 2*hc, :] * hc_scale[1] + hc_base[hc: 2*hc, :]  # (4, t)
    post = 2.0 * gen_sigmoid(post)  # (4, t)

    comb_flag = (x[2*hc:, :] * hc_scale[2] + hc_base[2*hc:, :]).reshape(hc, hc, t)  # (4, 4, t)
    row_max = comb_flag.amax(-2, keepdim=True)  # (4, 1, t)
    comb_flag = (comb_flag - row_max).exp() # (4, 4, t)

    row_sum = comb_flag.sum(-2, keepdim=True)   # (4, 1, t)
    comb_flag = comb_flag / row_sum + hc_eps    # (4, 4, t)
    col_sum = comb_flag.sum(-3, keepdim=True)   # (1, 4, t)
    comb_flag = comb_flag / (col_sum + hc_eps)  # (4, 4, t)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-2, keepdim=True)   # (4, 1, t)
        comb_flag = comb_flag / (row_sum + hc_eps)  # (4, 4, t)
        col_sum = comb_flag.sum(-3, keepdim=True)   # (1, 4, t)
        comb_flag = comb_flag / (col_sum + hc_eps)  # (4, 4, t)
    pre = pre.transpose(0, 1)
    post = post.transpose(0, 1)
    comb_flag = comb_flag.transpose(1, 2).transpose(0, 1)
    return pre, post, comb_flag, pre_


def gen_hc_pre(x, hc_fn, hc_scale, hc_base):
    t = x.shape[0]
    x_16 = x.reshape((t, hc * d))
    hc_base = hc_base.reshape(1, mix_hc)
    x = x_16.to(torch.float32)

    hc_fn = hc_fn.to(torch.float32)
    res = torch.matmul(x, hc_fn.transpose(0, 1))    # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)

    res = res / gen_rms_norm_denom(x)   # (t, mix_hc) / (t, 1) = (t, mix_hc)
    mm_res = res

    pre, post, comb = gen_hc_split_sinkhorn(res, hc_scale, hc_base) # (t, hc), (t, hc), (t, hc, hc)
    mul_res = pre.reshape(t, hc, 1) * x.reshape(t, hc, d)
    res = mul_res.sum(-2)   # (t,mul_res d)
    res = res.to(torch.bfloat16)
    return res, post, comb, mm_res


def gen_hc_pre_trans(x, hc_fn, hc_scale, hc_base):
    t = x.shape[0]
    x_16 = x.reshape((t, hc * d))
    hc_base = hc_base.reshape(1, mix_hc)
    x = x_16.to(torch.float32)

    hc_fn = hc_fn.to(torch.float32)
    res = torch.matmul(hc_fn, x.transpose(0, 1))    # (mix_hc, hc*d)@(t, hc*d)^t = (mix_hc, t)

    rms_res = gen_rms_norm_denom(x)
    res = res / (rms_res.reshape(1, t)) # (mix_hc, t) / (1, t) = (mix_hc, t)

    pre, post, comb, pre_ = gen_hc_split_sinkhorn_trans(res, hc_scale, hc_base) # (t, hc), (t, hc), (t, hc, hc)
    mm_res = pre_

    mul_res = pre.reshape(t, hc, 1) * x.reshape(t, hc, d)
    res = mul_res.sum(-2)   # (t,mul_res d)
    res = res.to(torch.bfloat16)
    return res, post, comb, mm_res


def gen_hc_pre_data(t=16, is_trans=False):
    torch.manual_seed(42)
    x = torch.empty((t, hc, d), dtype=torch.bfloat16).uniform_(-1, 1)
    hc_fn = torch.empty((mix_hc, hc*d), dtype=torch.float32).uniform_(-1, 1)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1)
    hc_base = torch.empty((mix_hc, ), dtype=torch.float32).uniform_(-1, 1)
    if is_trans:
        res, post, comb, mm_res = gen_hc_pre_trans(x, hc_fn, hc_scale, hc_base)
    else:
        res, post, comb, mm_res = gen_hc_pre(x, hc_fn, hc_scale, hc_base)
    return x, hc_fn, hc_scale, hc_base, res, post, comb, mm_res


class HC_PRE(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
        #### add some op here  x = torch.add(x, 0)
        return hc_pre_pypto(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)

def test_hc_pre_inmodel(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)
    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    ### to device
    x = x.npu()
    hc_fn = hc_fn.npu()
    hc_scale = hc_scale.npu()
    hc_base = hc_base.npu()

    hc_mult = 4
    hc_split_sinkhorn_iters = 20
    hc_eps = 1e-6

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(HC_PRE(), dynamic=False, fullgraph=True, backend=npu_backend)
    for _ in range(1):
        y, post, comb = model(x, hc_fn, hc_scale, hc_base, hc_mult, hc_split_sinkhorn_iters, hc_eps)
        pypto.runtime._device_synchronize()

    ### compare
    compare(y.cpu(), y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post.cpu(), post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb.cpu(), comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")

def test_hc_pre(t = 16, is_trans = False):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    check_input_output_shape_dtype(x, hc_fn, hc_scale, hc_base)

    y = torch.zeros_like(y_gd).npu()
    post = torch.zeros_like(post_gd).npu()
    comb = torch.zeros_like(comb_gd).npu()

    in_outs = {
        x.npu(): [0],
        hc_fn.npu(): None,
        hc_scale.npu(): None,
        hc_base.npu(): None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    if not is_trans:
        hc_pre_kernel(*pto_in_outs)
    else:
        hc_pre_kernel_prefill(*pto_in_outs)
    pypto.runtime._device_synchronize()

    y = y.cpu()
    post = post.cpu()
    comb = comb.cpu()

    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")


if __name__ == "__main__":
    print("start test !!!")
    test_hc_pre_inmodel(16)
    test_hc_pre(127)
