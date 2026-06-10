# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import subprocess
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch_npu
import torchair
import torchair as tng

import custom_ops

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


def get_npu_chip_name():
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'Ascend950' in line or 'Ascend95' in line:
                return 'Ascend950'
            if 'Ascend910_93' in line:
                return 'Ascend910_93'
            if '910B' in line:
                return 'Ascend910B'
            if 'Ascend310' in line:
                return 'Ascend310'
        return 'Unknown'
    except Exception:
        return 'Unknown'

CHIP_NAME = get_npu_chip_name()
IS_ASCEND950 = CHIP_NAME == 'Ascend950'
IS_ASCEND910B = CHIP_NAME == 'Ascend910B'
IS_ASCEND910_93 = CHIP_NAME == 'Ascend910_93'
IS_MEMBASE_MIXED_SUPPORTED = IS_ASCEND910B or IS_ASCEND910_93
IS_MIXED_SUPPORTED = IS_ASCEND950 or IS_MEMBASE_MIXED_SUPPORTED


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.0001, pct_thd=0.0005, max_diff_hd=0.0001):
    print(f'======================== Golden BEGIN ========================')
    print(f'======================== OutPut BEGIN ========================')
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
    
    @unittest.skipIf(not IS_ASCEND950, "This test only runs on Ascend950, because of shape")
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
    
    @unittest.skipIf(not IS_ASCEND950, "This test only runs on Ascend950, because of shape")
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
    
    @unittest.skipIf(not IS_ASCEND950, "This test only runs on Ascend950, because of shape")
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
    
    @unittest.skipIf(not IS_ASCEND950, "This test only runs on Ascend950, because of shape")
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

    def run_mixed_precision_test(self, x_shape, cos_shape, x_dtype=torch.bfloat16, 
                                  cos_sin_dtype=torch.float32, partial_slice_start=64, 
                                  partial_slice_end=128):
        """通用混合精度测试函数：x是bfloat16/float16，cos/sin是float32"""
        np.random.seed(0)
        slice_size = partial_slice_end - partial_slice_start
        partial_slice = [partial_slice_start, partial_slice_end]
        
        b, s, n, d = x_shape
        
        x = torch.tensor(np.random.uniform(1, 10, x_shape)).to(x_dtype)
        cos = torch.tensor(np.random.uniform(-10, 10, cos_shape)).to(cos_sin_dtype)
        sin = torch.tensor(np.random.uniform(-10, 10, cos_shape)).to(cos_sin_dtype)
        
        if partial_slice_end == d:
            chunks = torch.split(x, [partial_slice_start, slice_size], dim=-1)
            a1 = chunks[0]
            a2 = chunks[1]
            aa = cos * a2.float() + rotate_every_two(a2.float()) * sin
            cpu_result = torch.cat([a1, aa.to(x_dtype)], -1)
        else:
            chunks = torch.split(x, [partial_slice_start, slice_size, d - partial_slice_end], dim=-1)
            a1 = chunks[0]
            a2 = chunks[1]
            a3 = chunks[2]
            aa = cos * a2.float() + rotate_every_two(a2.float()) * sin
            cpu_result = torch.cat([a1, aa.to(x_dtype), a3], -1)
        
        # NPU操作
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
        
        # 比对结果
        x_npu_cpu = x_npu.cpu().float()
        cpu_result_float = cpu_result.cpu().float()
        compare_result = data_compare(x_npu_cpu.numpy(), cpu_result_float.numpy())
        print(f"Mixed precision test x_shape={x_shape}, cos_shape={cos_shape}, x_dtype={x_dtype}: {compare_result}")
        self.assertTrue(compare_result, 
            f"Mixed precision compare fail for x_shape={x_shape}, cos_shape={cos_shape}")

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape1(self):
        """Case 1: x=[8192, 128, 1, 512], cos=[8192, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[8192, 128, 1, 512],
            cos_shape=[8192, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape2(self):
        """Case 2: x=[8192, 1, 1, 512], cos=[8192, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[8192, 1, 1, 512],
            cos_shape=[8192, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape3(self):
        """Case 3: x=[8192, 64, 1, 128], cos=[8192, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[8192, 64, 1, 128],
            cos_shape=[8192, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape4(self):
        """Case 4: x=[4, 128, 1, 512], cos=[4, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[4, 128, 1, 512],
            cos_shape=[4, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape5(self):
        """Case 5: x=[4, 1, 1, 512], cos=[4, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[4, 1, 1, 512],
            cos_shape=[4, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MIXED_SUPPORTED, "Mixed precision tests only run on Ascend950, Ascend910B or Ascend910_93")
    def test_mixed_precision_bf16_fp32_shape6(self):
        """Case 6: x=[4, 64, 1, 128], cos=[4, 1, 1, 64]"""
        self.run_mixed_precision_test(
            x_shape=[4, 64, 1, 128],
            cos_shape=[4, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=64,
            partial_slice_end=128
        )

    @unittest.skipIf(not IS_MEMBASE_MIXED_SUPPORTED,
        "Membase mixed precision tests only run on Ascend910B or Ascend910_93")
    def test_membase_mixed_precision_fp16_fp32_aligned(self):
        self.run_mixed_precision_test(
            x_shape=[128, 64, 1, 512],
            cos_shape=[128, 1, 1, 64],
            x_dtype=torch.float16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=448,
            partial_slice_end=512
        )

    @unittest.skipIf(not IS_MEMBASE_MIXED_SUPPORTED,
        "Membase mixed precision tests only run on Ascend910B or Ascend910_93")
    def test_membase_mixed_precision_bf16_fp32_pad(self):
        self.run_mixed_precision_test(
            x_shape=[128, 64, 1, 512],
            cos_shape=[128, 1, 1, 28],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=484,
            partial_slice_end=512
        )

    @unittest.skipIf(not IS_MEMBASE_MIXED_SUPPORTED,
        "Membase mixed precision tests only run on Ascend910B or Ascend910_93")
    def test_membase_mixed_precision_bf16_fp32_middle_slice(self):
        self.run_mixed_precision_test(
            x_shape=[4, 64, 1, 128],
            cos_shape=[4, 1, 1, 64],
            x_dtype=torch.bfloat16,
            cos_sin_dtype=torch.float32,
            partial_slice_start=32,
            partial_slice_end=96
        )

    @unittest.skipIf(not IS_ASCEND950, "Mixed precision tests only run on Ascend950")
    def test_inplace_partial_rotary_mul_mixed_precision_bf16_fp32(self):
        """Mixed precision test: x is bfloat16, cos/sin are float32"""
        b = 128
        s = 64
        n = 1
        d = 512

        con_b = 1
        con_s = 64
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512]
        
        np.random.seed(0)
        slice_size = partial_slice[1] - partial_slice[0]
        
        # x is bfloat16, cos/sin are float32
        x_dtype = torch.bfloat16
        cos_sin_dtype = torch.float32
        
        x = torch.tensor(np.random.uniform(1, 10, (b, s, n, d))).to(x_dtype)
        cos = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(cos_sin_dtype)
        sin = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(cos_sin_dtype)
        
        chunks2 = torch.split(x, [partial_slice[0], slice_size], dim=-1)
        a1 = chunks2[0]
        a2 = chunks2[1]
        
        # CPU result: cast x to float32, compute, then cast back to bfloat16
        aa = cos * a2.float() + rotate_every_two(a2.float()) * sin
        gpu_result = torch.cat([a1, aa.to(x_dtype)], -1)
        
        # NPU operation
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
        
        x_npu_cpu = x_npu.cpu().float()
        gpu_result_cpu = gpu_result.cpu().float()
        compare_result = data_compare(x_npu_cpu.numpy(), gpu_result_cpu.numpy())
        print(f"Mixed precision BF16-FP32 compare: {compare_result}")
        self.assertTrue(compare_result, f"yOut precision compare fail for mixed precision bf16-fp32")

    @unittest.skipIf(not IS_ASCEND950, "Mixed precision tests only run on Ascend950")
    def test_inplace_partial_rotary_mul_mixed_precision_fp16_fp32(self):
        """Mixed precision test: x is float16, cos/sin are float32"""
        b = 128
        s = 64
        n = 1
        d = 512

        con_b = 1
        con_s = 64
        con_n = 1

        rotary_mode = 1
        partial_slice = [448, 512]
        
        np.random.seed(0)
        slice_size = partial_slice[1] - partial_slice[0]
        
        # x is float16, cos/sin are float32
        x_dtype = torch.float16
        cos_sin_dtype = torch.float32
        
        x = torch.tensor(np.random.uniform(1, 10, (b, s, n, d))).to(x_dtype)
        cos = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(cos_sin_dtype)
        sin = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(cos_sin_dtype)
        
        chunks2 = torch.split(x, [partial_slice[0], slice_size], dim=-1)
        a1 = chunks2[0]
        a2 = chunks2[1]
        
        # CPU result: cast x to float32, compute, then cast back to float16
        aa = cos * a2.float() + rotate_every_two(a2.float()) * sin
        gpu_result = torch.cat([a1, aa.to(x_dtype)], -1)
        
        # NPU operation
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
        
        x_npu_cpu = x_npu.cpu().float()
        gpu_result_cpu = gpu_result.cpu().float()
        compare_result = data_compare(x_npu_cpu.numpy(), gpu_result_cpu.numpy())
        print(f"Mixed precision FP16-FP32 compare: {compare_result}")
        self.assertTrue(compare_result, f"yOut precision compare fail for mixed precision fp16-fp32")

def rotate_every_two(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack(((-x_odd, x_even)), dim=-1)
    return stacked.reshape(x.shape)

if __name__ == "__main__":
    run_tests()
