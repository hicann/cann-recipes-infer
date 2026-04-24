# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import torch.nn as nn
import numpy as np
import custom_ops
import math

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

def _run_test_graph():

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, x_npu, cons_npu, sin_npu, partial_slice):
            torch.ops.custom.inplace_partial_rotary_mul(
                x_npu,
                cons_npu,
                sin_npu,
                rotary_mode="interleave",
                partial_slice=partial_slice
            )

    b = 128
    s = 64
    d = 512
    d1 = 64
    rotary_mode = 1
    partial_slice = [448, 512]

    np.random.seed(0)
            # 测试不同数据类型
    dtypes = [torch.bfloat16]

    for dtype in dtypes:
        x = torch.tensor(np.random.uniform(-10, 10, (b, s, 1, d)).astype(np.int32)).to(dtype)
        cons_tensor = torch.tensor(np.random.uniform(1, 17, (b, 1, 1, d1)).astype(np.int32)).to(dtype)
        sin_tensor = torch.tensor(np.random.uniform(-5, 10, (b, 1, 1, d1)).astype(np.int32)).to(dtype)
        chunks2 = torch.split(x, [448, 64], dim=-1)
        a1 = chunks2[0]
        x_tensor = chunks2[1]#128,64,64
        # 1. cpu结果
        aa = cons_tensor * x_tensor + rotate_every_two(x_tensor) * sin_tensor
        gpu_result = torch.cat([a1, aa], -1)
        # 2. 运行NPU操作
        print(f'======================== NPU Eager BEGIN ========================')
        x_npu = x.to("npu:%s" % DEVICE_ID)
        cons_npu = cons_tensor.to("npu:%s" % DEVICE_ID)
        sin_npu = sin_tensor.to("npu:%s" % DEVICE_ID)
        print(f'======================== PTA graph BEGIN ========================')

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_mode(x_npu, cons_npu, sin_npu, partial_slice)
        x_npu_cpu = x_npu.cpu().float().numpy()
        gpu_result_cpu = gpu_result.cpu().float().numpy()
        yOut_close = np.allclose(x_npu_cpu, gpu_result_cpu, rtol=0.001, atol=0.001, equal_nan=True)
        print(f"yOut close: {yOut_close}")
        print(f'======================== PTA graph FINISH ========================')

class TestNpuInplacePartialRotaryMul(TestCase):
    def test_inplace_partial_rotary_mul_with_cpu_benchmark(self):
        """测试函数，包含CPU标杆验证"""
        # 设置测试参数（与原始测试相同）
        b = 128
        s = 64 #网络另一个场景次值是1
        d = 512
        d1 = 64
        rotary_mode = 1
        partial_slice = [448, 512]

        np.random.seed(0)
                # 测试不同数据类型
        dtypes = [torch.bfloat16]

        for dtype in dtypes:
            x = torch.tensor(np.random.uniform(-10, 10, (b, s, 1, d)).astype(np.int32)).to(dtype)
            cons_tensor = torch.tensor(np.random.uniform(1, 17, (b, 1, 1, d1)).astype(np.int32)).to(dtype)
            sin_tensor = torch.tensor(np.random.uniform(-5, 10, (b, 1, 1, d1)).astype(np.int32)).to(dtype)
            chunks2 = torch.split(x, [448, 64], dim=-1)
            a1 = chunks2[0]
            x_tensor = chunks2[1]#128,64,64
            # 1. cpu结果
            aa = cons_tensor * x_tensor + rotate_every_two(x_tensor) * sin_tensor
            gpu_result = torch.cat([a1, aa], -1)
            # 2. 运行NPU操作
            print(f'======================== NPU Eager BEGIN ========================')
            x_npu = x.to("npu:%s" % DEVICE_ID)
            cons_npu = cons_tensor.to("npu:%s" % DEVICE_ID)
            sin_npu = sin_tensor.to("npu:%s" % DEVICE_ID)
            
            torch.ops.custom.inplace_partial_rotary_mul(
                x_npu,
                cons_npu,
                sin_npu,
                rotary_mode="interleave",
                partial_slice=partial_slice
            )
            print(f'======================== NPU Eager FINISH ========================')

            x_npu_cpu = x_npu.cpu().float().numpy()
            gpu_result_cpu = gpu_result.cpu().float().numpy()
            yOut_close = np.allclose(x_npu_cpu, gpu_result_cpu, rtol=0.001, atol=0.001, equal_nan=True)
            print(f"yOut close: {yOut_close}")
            self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype}")
    def test_inplace_partial_rotary_mul_with_garph(self):
        #测试图模式
        _run_test_graph()
def rotate_every_two(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack(((-x_odd, x_even)), dim=-1)
    return stacked.reshape(x.shape)

if __name__ == "__main__":
    run_tests()


