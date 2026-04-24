# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests


np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def softmax_func(x, axis=None, eps=None):
    if "float16" in x.dtype.name:
        x = x.astype(np.float32)
    
    x_max = np.max(x, axis=axis, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = np.sum(y, axis=axis, keepdims=True)
    ans = y / x_sum
    ans = ans + eps
    return ans


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def sink_horn_numpy(mixes: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float) -> tuple:
    mixes0 = mixes[..., 0: hc_mult]
    hc_base0 = hc_base[..., 0: hc_mult]
    mixes0 = mixes0 * hc_scale[0]
    mixes0 = mixes0 + hc_base0
    pre = sigmoid_func(mixes0) + hc_eps

    mixes1 = mixes[..., hc_mult: hc_mult * 2]
    hc_base1 = hc_base[..., hc_mult: hc_mult * 2]
    mixes1 = mixes1 * hc_scale[1]
    mixes1 = mixes1 + hc_base1
    post = sigmoid_func(mixes1) * 2

    mixes2 = mixes[..., hc_mult * 2:]
    other_dims = mixes2.shape[:-1]
    new_dims = other_dims + (hc_mult, hc_mult)
    mixes2 = mixes2.reshape(new_dims)
    hc_base2 = hc_base[..., hc_mult * 2:]
    hc_base2 = hc_base2.reshape(hc_mult, hc_mult)
    mixes2 = mixes2 * hc_scale[2]
    mixes2 = mixes2 + hc_base2
    comb_frag = softmax_func(mixes2, -1, hc_eps)
    comb_frag_sum = np.sum(comb_frag, axis=-2, keepdims=True) + hc_eps
    comb_frag = comb_frag / comb_frag_sum
    for _ in range(hc_sinkhorn_iters - 1):
        comb_frag_sum = np.sum(comb_frag, axis=-1, keepdims=True) + hc_eps
        comb_frag = comb_frag / comb_frag_sum 
        comb_frag_sum = np.sum(comb_frag, axis=-2, keepdims=True) + hc_eps
        comb_frag = comb_frag / comb_frag_sum 
    return pre, post, comb_frag


def hc_pre_sinkhorn_numpy(mixes: np.ndarray, rsqrt: np.ndarray, hc_scale: np.ndarray, hc_base: np.ndarray, x: np.ndarray, hc_mult: int, hc_sinkhorn_iters: int, hc_eps: float) -> tuple:
    x = x.astype("float32")
    mixes = mixes * rsqrt.reshape(rsqrt.shape + (1,))
    pre, post, comb_frag = sink_horn_numpy(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    pre = pre.reshape(pre.shape + (1, )) * x
    y = np.sum(pre, axis=2, keepdims=True)
    return torch.from_numpy(y).to(torch.bfloat16).to(torch.float32), torch.from_numpy(post).to(torch.float32), torch.from_numpy(comb_frag).to(torch.float32)


class TestCustomHcPreSinkhorn(TestCase):
    def test_hc_pre_sinkhorn_graph(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps):
                npu_yOut, npu_postOut, npu_comb_fragOut = torch.ops.custom.npu_hc_pre_sinkhorn(
                    mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps
                )
                return npu_yOut, npu_postOut, npu_comb_fragOut

        b = 4
        s = 4
        hc_mix = 24
        hc_mult = 4
        d_list = [4096, 7168]
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6

        for d in d_list:
            np.random.seed(42)

            mixes = torch.tensor(np.random.uniform(-2, 2, (b, s, hc_mix))).to(torch.float32)
            rsqrt = torch.tensor(np.random.uniform(-2, 2, (b, s))).to(torch.float32)
            hc_scale = torch.tensor(np.random.uniform(-2, 2, (3))).to(torch.float32)
            hc_base = torch.tensor(np.random.uniform(-2, 2, (hc_mix))).to(torch.float32)
            x = torch.tensor(np.random.uniform(-2, 2, (b, s, hc_mult, d))).to(torch.bfloat16)

            cpu_yOut, cpu_postOut, cpu_comb_fragOut = hc_pre_sinkhorn_numpy(
                mixes.numpy().astype(np.float64), rsqrt.numpy().astype(np.float64), hc_scale.numpy().astype(np.float64), hc_base.numpy().astype(np.float64), x.float().numpy().astype(np.float64), hc_mult, hc_sinkhorn_iters, 
                hc_eps)
            
            mixes_npu = mixes.to("npu:%s" % DEVICE_ID)
            rsqrt_npu = rsqrt.to("npu:%s" % DEVICE_ID)
            hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
            hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
            x_npu = x.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_postOut, npu_comb_fragOut = torch.ops.custom.npu_hc_pre_sinkhorn(
                mixes_npu, rsqrt_npu, hc_scale_npu, hc_base_npu, x_npu, hc_mult, hc_sinkhorn_iters, hc_eps
            )

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_postOut, npu_comb_fragOut = npu_mode(mixes_npu, rsqrt_npu, hc_scale_npu, hc_base_npu, x_npu, hc_mult, hc_sinkhorn_iters, hc_eps)

            npu_yOut_cpu = npu_yOut.cpu().float()
            npu_postOut_cpu = npu_postOut.cpu().float().numpy()
            npu_comb_fragOut_cpu = npu_comb_fragOut.cpu().float().numpy()

            yOut_close = np.allclose(npu_yOut_cpu.reshape(-1), cpu_yOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)
            postOut_close = np.allclose(npu_postOut_cpu.reshape(-1), cpu_postOut.numpy().reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)
            comb_fragOut_close = np.allclose(npu_comb_fragOut_cpu.reshape(-1), cpu_comb_fragOut.numpy().reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail for d={d}")
            self.assertTrue(postOut_close, f"postOut precision compare fail for d={d}")
            self.assertTrue(comb_fragOut_close, f"comb_fragOut precision compare fail for d={d}")

if __name__ == "__main__":
    run_tests()
