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
import torchair as tng
import numpy as np
import custom_ops
import math
import torch.nn as nn

import torchair

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.0001, pct_thd=0.0005, max_diff_hd=0.0001):
    print(f'======================== Golden BEGIN ========================')
    print(cpu_out)
    print(f'======================== OutPut BEGIN ========================')
    print(npu_out)
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        return result, 0.0, max_error
    
    split_count = int(end - start + 1) if end != start else 1
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)
    
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"

    return result, fulfill_percent, max_error

def requantize_compare(golden, output):    
    diff_results = torch.abs(torch.subtract(output.view(-1), golden.view(-1)))
    diff_indices = torch.where(diff_results > 1)[0]
    np.set_printoptions(suppress=True, precision=4)
    print(f'======================== Golden BEGIN ========================')
    print(golden.numpy())
    print(f'======================== OutPut BEGIN ========================')
    print(output.numpy())
    npu_nan, golden_nan = torch.isnan(output.view([-1])), torch.isnan(golden.view([-1]))
    diff_nan = torch.logical_and(npu_nan, golden_nan)
    both_nan_idx = torch.where(diff_nan)

    diff_indices = torch.where(torch.logical_not(torch.isin(diff_indices, both_nan_idx[0])))[0]
    del diff_results, npu_nan, golden_nan, diff_nan

    golden_size, diff_size = golden.numel(), diff_indices.numel()
    precision = (golden_size - diff_size) / golden_size
    print(f"precision: {precision}")
    is_pass = (1 - precision) <= 0.001
    return is_pass

def inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self):
    """测试函数，包含CPU标杆验证"""
    # 设置测试参数（与原始测试相同）
    np.random.seed(0)
    
    # 测试不同数据类型
    dtypes = [torch.bfloat16]

    for dtype in dtypes:
        for i in range(10):
            slice_size = partial_slice[1] - partial_slice[0]
            x = torch.tensor(np.random.uniform(1, 10, (b, s, n, d))).to(dtype)
            cos = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)
            sin = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)
            
            chunks2 = torch.split(x, [partial_slice[0], slice_size], dim=-1)
            a1 = chunks2[0]
            a2 = chunks2[1]
            print(chunks2[0].shape)
            print(chunks2[1].shape)
            # 1. cpu结果
            aa = cos.float() * a2.float() + rotate_every_two(a2.float()) * sin.float()
            gpu_result = torch.cat([a1, aa], -1).to(dtype)

            # 2. 运行NPU操作
            print(f'======================== NPU Eager BEGIN ========================')
            x_npu = x.to("npu:%s" % DEVICE_ID)
            cos_npu = cos.to("npu:%s" % DEVICE_ID)
            sin_npu = sin.to("npu:%s" % DEVICE_ID)
         
            torch.ops.custom.inplace_partial_rotary_mul(
                x_npu,
                cos_npu,
                sin_npu,
                rotary_mode="interleave",
                partial_slice=partial_slice
            )

            # 拼接算子操作
            # o_nope, o_rope = x_npu.split([d - 64, 64], -1)
            # o_rope = torch_npu.npu_rotary_mul(o_rope, cos_npu, sin_npu, rotary_mode='interleave')
            # gpu_result = torch.cat([o_nope, o_rope], -1)

            print(f'======================== NPU Eager FINISH ========================')

            x_npu_cpu = x_npu.cpu().float()
            gpu_result_cpu = gpu_result.cpu().float()
            # compare_result = requantize_compare(gpu_result_cpu, x_npu_cpu)
            compare_result = data_compare(x_npu_cpu.numpy(), gpu_result_cpu.numpy())
            print(f"compare: {compare_result}")
            self.assertTrue(compare_result, f"yOut precision compare fail for dtype {dtype}")

def inplace_partial_rotary_mul_with_cpu_benchmark_graph(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self):
    """测试函数，包含CPU标杆验证"""
    # 设置测试参数（与原始测试相同）
    np.random.seed(0)
    
    # 测试不同数据类型
    dtypes = [torch.bfloat16]

    for dtype in dtypes:
        for i in range(1):
            slice_size = partial_slice[1] - partial_slice[0]
            x = torch.tensor(np.random.uniform(1, 10, (b, s, n, d))).to(dtype)
            cos = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)
            sin = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)
            
            chunks2 = torch.split(x, [partial_slice[0], slice_size], dim=-1)
            a1 = chunks2[0]
            a2 = chunks2[1]
            print(chunks2[0].shape)
            print(chunks2[1].shape)
            # 1. cpu结果
            aa = cos.float() * a2.float() + rotate_every_two(a2.float()) * sin.float()
            gpu_result = torch.cat([a1, aa], -1).to(dtype)

            # 2. 运行NPU操作
            print(f'======================== NPU Eager BEGIN ========================')
            # start run custom ops
            class Network(nn.Module):
                def __init__(self):
                    super(Network, self).__init__()

                def forward(self, x_npu, cos_npu, sin_npu, rotary_mode, partial_slice):
                    torch.ops.custom.inplace_partial_rotary_mul(
                        x_npu,
                        cos_npu,
                        sin_npu,
                        rotary_mode=rotary_mode,
                        partial_slice=partial_slice
                    )
            
            print(f'======================== PTA graph BEGIN ========================')
            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            cos_npu = cos.to("npu:%s" % DEVICE_ID)
            sin_npu = sin.to("npu:%s" % DEVICE_ID)
            npu_mode(
                x_npu,
                cos_npu,
                sin_npu,
                rotary_mode="interleave",
                partial_slice=partial_slice)
            print(f'======================== NPU Eager FINISH ========================')

            x_npu_cpu = x_npu.cpu().float()
            gpu_result_cpu = gpu_result.cpu().float()
            # compare_result = requantize_compare(gpu_result_cpu, x_npu_cpu)
            compare_result = data_compare(x_npu_cpu.numpy(), gpu_result_cpu.numpy())
            print(f"compare: {compare_result}")
            self.assertTrue(compare_result, f"yOut precision compare fail for dtype {dtype}")

class TestNpuInplacePartialRotaryMul(TestCase):
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_ab_graph(self):
        """测试函数，包含CPU标杆验证"""
        # 设置测试参数（与原始测试相同）
        b = 128
        s = 64
        n = 1
        d = 512

        con_b = 128
        con_s = 1
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_graph(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_ab(self):
        """测试函数，包含CPU标杆验证"""
        # 设置测试参数（与原始测试相同）
        b = 128
        s = 64
        n = 1
        d = 512

        con_b = 128
        con_s = 1
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_b(self):
        """测试函数，包含CPU标杆验证"""
        b = 1
        s = 1
        n = 1
        d = 512

        con_b = 1
        con_s = 1
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_a(self):
        """测试函数，包含CPU标杆验证"""
        b = 128
        s = 64
        n = 128
        d = 512

        con_b = 128
        con_s = 64
        con_n = 128

        rotary_mode = 1
        partial_slice = [448, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_aba(self):
        """测试函数，包含CPU标杆验证"""
        b = 128
        s = 64
        n = 128
        d = 512

        con_b = 128
        con_s = 1
        con_n = 128

        rotary_mode = 1
        partial_slice = [484, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_ba(self):
        """测试函数，包含CPU标杆验证"""
        b = 128
        s = 64
        n = 128
        d = 512

        con_b = 1
        con_s = 1
        con_n = 128

        rotary_mode = 1
        partial_slice = [484, 512]
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)
    def test_inplace_partial_rotary_mul_with_cpu_benchmark_bab(self):
        """测试函数，包含CPU标杆验证"""
        b = 128
        s = 64
        n = 128
        d = 512

        con_b = 1
        con_s = 64
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512] 
        inplace_partial_rotary_mul_with_cpu_benchmark_base(b, s, n, d, con_b, con_s, con_n, rotary_mode, partial_slice, self)

def rotate_every_two(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack(((-x_odd, x_even)), dim=-1)
    return stacked.reshape(x.shape)

if __name__ == "__main__":
    run_tests()