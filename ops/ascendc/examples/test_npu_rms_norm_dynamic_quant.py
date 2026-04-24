# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


import numpy as np
import argparse
import random
import torch.nn.functional as F

def requantize_compare(golden, output):
    output_0 = output.view(torch.int8)
    golden_0 = golden.view(torch.int8)

    diff_results = torch.abs(torch.subtract(output_0.view(-1), golden_0.view(-1)))
    diff_indices = torch.where(diff_results > 1)[0]

    npu_nan, golden_nan = torch.isnan(output.view([-1])), torch.isnan(golden.view([-1]))
    diff_nan = torch.logical_and(npu_nan, golden_nan)
    both_nan_idx = torch.where(diff_nan)

    diff_indices = torch.where(torch.logical_not(torch.isin(diff_indices, both_nan_idx[0])))[0]
    del diff_results, npu_nan, golden_nan, diff_nan

    golden_size, diff_size = golden.numel(), diff_indices.numel()
    precision = (golden_size - diff_size) / golden_size
    is_pass = (1 - precision) <= 0.001
    return is_pass

def rms_norm_and_dynamic_quant(x, gamma, smooth_scale, beta, esp):
        x_npu = x.to("npu:%s" % DEVICE_ID)
        gamma_npu = gamma.to("npu:%s" % DEVICE_ID)
        smooth_scale_npu = smooth_scale.to("npu:%s" % DEVICE_ID)

        npu_y_out, npu_rstd_out = torch_npu.npu_rms_norm(
                x_npu, gamma_npu, epsilon=esp
            )
        print(f"run npu_rms_norm completely")
        npu_out, npu_scale_out = torch_npu.npu_dynamic_quant(npu_y_out, smooth_scales=smooth_scale_npu)
        print(f"run npu_dynamic_quant completely")
        out_cpu = npu_out.cpu()
        scale_out_cpu = npu_scale_out.cpu()
        return  out_cpu, scale_out_cpu

class TestCustomRmsNormDynamicQuant(TestCase):
    def test_rms_norm_dynamic_quant_different_dtypes(self):
        """测试不同数据类型的RmsNormDynamicQuant算子"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数
        bs = 128
        n = 1
        d_list = [4096, 7168]
        esp = 1e-6
        

        # 测试不同数据类型
        dtypes = [torch.bfloat16]

        for dtype in dtypes:
            for d in d_list:
                print(f"Testing dtype: {dtype}, d: {d}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(1, 10, (bs, n, d))).to(dtype)
                gamma = torch.tensor(np.random.uniform(0, 1, (d))).to(dtype)
                smooth_scale = torch.tensor(np.random.uniform(0, 1, (d))).to(dtype)
                
                 # CPU参考结果
                cpu_yOut, cpu_scale_out = rms_norm_and_dynamic_quant(
                    x, gamma, smooth_scale, None, esp)

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)
                gamma_npu = gamma.to("npu:%s" % DEVICE_ID)
                smooth_scale_npu = smooth_scale.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_y_out, npu_scale_out = torch.ops.custom.npu_rms_norm_dynamic_quant(
                    x_npu, gamma_npu, smooth_scale=smooth_scale_npu, beta=None, epsilon=esp
                )

                # # 转换回CPU进行比较
                npu_y_out_cpu = npu_y_out.cpu()
                npu_scale_out_cpu = npu_scale_out.cpu().float().numpy()
                cpu_scale_out = cpu_scale_out.float().numpy()
                # # 验证结果
                yOut_close = requantize_compare(cpu_yOut, npu_y_out_cpu)
                scale_out_close = np.allclose(npu_scale_out_cpu, cpu_scale_out, rtol=0.01, atol=0.001, equal_nan=True)
                print(f"yOut close: {yOut_close}, scale_out_close equal: {scale_out_close}")
                self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype} d {d}")
                self.assertTrue(scale_out_close, f"scale_out compare fail for dtype {dtype} d {d}")

    def test_rms_norm_dynamic_quant_different_dtypes_graph(self):
        """测试不同数据类型的RmsNormDynamicQuant算子"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, gamma, smooth_scale, beta, esp):
                npu_y_out, npu_scale_out = torch.ops.custom.npu_rms_norm_dynamic_quant(
                x, gamma, smooth_scale=smooth_scale, beta=None, epsilon=esp
            )
                return npu_y_out, npu_scale_out
            
        # 设置参数
        bs = 128
        n = 1
        d = 4096
        esp = 1e-6

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            print(f"Testing dtype: {dtype}")

            # 创建输入数据
            np.random.seed(42)
            x = torch.tensor(np.random.uniform(1, 10, (bs, n, d))).to(dtype)
            gamma = torch.tensor(np.random.uniform(0, 1, (d))).to(dtype)
            smooth_scale = torch.tensor(np.random.uniform(0, 1, (d))).to(dtype)
            
             # CPU参考结果
            cpu_yOut, cpu_scale_out = rms_norm_and_dynamic_quant(
                x, gamma, smooth_scale, None, esp)

            # 转换到NPU
            x_npu = x.to("npu:%s" % DEVICE_ID)
            gamma_npu = gamma.to("npu:%s" % DEVICE_ID)
            smooth_scale_npu = smooth_scale.to("npu:%s" % DEVICE_ID)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)

            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out= npu_mode(x_npu, gamma_npu, smooth_scale_npu, None, esp)
            
            # 转换回CPU进行比较
            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_out_cpu = npu_scale_out.cpu().float().numpy()
            cpu_scale_out = cpu_scale_out.float().numpy()
            # # 验证结果
            yOut_close = requantize_compare(cpu_yOut, npu_y_out_cpu)
            scale_out_close = np.allclose(npu_scale_out_cpu, cpu_scale_out, rtol=0.01, atol=0.001, equal_nan=True)
            print(f"yOut close: {yOut_close}, scale_out_close equal: {scale_out_close}")
            self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype}")
            self.assertTrue(scale_out_close, f"scale_out compare fail for dtype {dtype}")
            
if __name__ == "__main__":
    run_tests()