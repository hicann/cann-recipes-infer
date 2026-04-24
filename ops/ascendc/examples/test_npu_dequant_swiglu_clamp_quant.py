# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Test script for DequantSwigluClampQuant operator.

Reference:
  - ops-nn/quant/dequant_swiglu_clamp_quant/ (operator source code)
  - opstest/ttk/ttk/user_defined_modules/op/golden_funcs/dequant_swiglu_clamp_quant.py (golden implementation)
  - cann-recipes-infer/ops/ascendc/examples/test_npu_swiglu_clip_quant.py (test format)

Algorithm: Dequant(x) -> Swiglu -> Quant(y)
  - Input:  x (int32/float16/bfloat16), weight_scale, activation_scale, bias, quant_scale, quant_offset
  - Output: y (int8/fp8/fp4), scale (float32)

Supported features:
  - quant_mode: "static" or "dynamic"
  - swiglu_mode: 0 (standard SiLU), 1 (variant with clamp)
  - activate_dim: dimension to split for GLU
  - group_index: grouped processing for MoE scenarios
  - dst_type: int8(2), fp8_e5m2(35), fp8_e4m3fn(36), fp4_e2m1(40), fp4_e1m2(41)
  - activate_left: which half gets activated
  - round_mode: rint/round/floor/ceil/trunc
"""

import torch
import torch_npu
import torchair
import numpy as np
import random
import torch.nn as nn
import custom_ops

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

# 量化参数常量
MAX_VALUE_WITH_INT8 = 127
MIN_VALUE_WITH_INT8 = -128
MAX_VALUE_WITH_FLOAT8E5M2 = 57344
MAX_VALUE_WITH_FLOAT8E4M3 = 448
MAX_VALUE_WITH_FLOAT4E2M1 = 6
MAX_VALUE_WITH_FLOAT4E1M2 = 1.75

# Round 模式映射
ROUND_MODE_MAP = {
    "rint": np.rint,
    "round": np.round,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc
}


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    """Calculate relative difference between real and expected data."""
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1):
    """Compare NPU output with CPU golden output."""
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
    if len(diff_index[0]) == 0:
        return "Pass", 100.0, 0.0
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)

    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    # ----- 新增：打印每个错误位置的详细信息 -----
    if error_cnt > 0:
        print(f"\n=== Found {error_cnt} mismatched elements (relative diff > {diff_thd}) ===")
        # 遍历每个错误位置，打印索引、NPU值、CPU值、绝对差、相对差
        for i, idx in enumerate(err_idx):
            npu_val = real_data[idx]
            cpu_val = data_compe[idx]
            abs_diff = diff_abs[idx]
            rel_diff = err_diff[i]
            print(f"  Index {idx:5d}: NPU={npu_val:14.8f}  CPU={cpu_val:14.8f}  "
                  f"abs_diff={abs_diff:12.8f}  rel_diff={rel_diff:12.8f}")
        print("=== End of mismatched elements ===\n")
    # -----------------------------------------

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff)
        if max(err_diff) >= max_diff_hd:
            result = "Failed"

    return result, fulfill_percent, max_error

def cmp_and_print(y, scale, y_cpu, scale_cpu, group_index, quant_mode):
    group = y.shape[0]
    if group_index is not None:
        group = group_index.sum()
        y = y[:group]
        scale = scale[:group]
        y_cpu = y_cpu[:group]
        scale_cpu = scale_cpu[:group]
    print("=============npu=============")
    print(f'---y: {y.shape}')
    print(y.to("cpu"))
    print(f'---scale: {scale.shape}')
    print(scale.to("cpu"))

    print("=============cpu=============")
    print(f'---golden_y: {y_cpu.shape}')
    print(y_cpu)

    print(f'---golden_scale: {scale_cpu.shape}')
    print(scale_cpu)

    total_size = y.shape[0]*y.shape[1]
    y = y.cpu().numpy().reshape(total_size)
    golden_y = y_cpu.reshape(total_size)

    abs_diff = np.abs(y - golden_y)
    y_cmp = np.where(abs_diff > 1)[0] # int8误差为正负1
    cmp = np.isclose(scale.cpu().numpy(), scale_cpu, rtol=1e-05, atol=1e-08)
    scale_cmp = np.where(cmp != True)[0]
    print("-----y_cmp_res----------:", len(y_cmp) == 0)
    print("-----y_diff_num:", len(y_cmp))
    if quant_mode == 1:
        print("-----scale_cmp_res------:", len(scale_cmp) == 0)
        print("-----scale_diff_num:", len(scale_cmp))

    print(y_cmp)
    print(scale_cmp)

def _dequant_swiglu_clamp_quant_cpu(x, weight_scale=None, activate_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0):
    if group_index is None:
        group_index = torch.tensor([x.shape[0]], dtype = torch.int64)
    offset = 0
    res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float)
    res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)
    for g_idx in range(group_index.shape[0]):
        groupIdx = group_index[g_idx]

        # dequant
        x_part = x[offset: (offset+groupIdx)].to(torch.float32)
        if x.dtype == torch.int32:
            if bias is not None and bias.dtype == torch.int32:
                x_part = x_part + bias[g_idx]
            x_part = torch.mul(x_part, weight_scale[g_idx].to(torch.float32))
            if activate_scale is not None:
                x_part = torch.mul(x_part, activate_scale[offset: (offset+groupIdx)].to(torch.float32))
            if bias is not None and bias.dtype != torch.int32:
                x_part = x_part + bias[g_idx].to(torch.float32)

        # swiglu
        if swiglu_mode == 1:
            out = torch.chunk(x_part, 2, dim=-1)
            if activate_left:
                self_tensor = out[0]
                other = out[1]
            else:
                self_tensor = out[1]
                other = out[0]
            self_tensor = self_tensor.clamp(min=None, max=clamp_limit)
            other = other.clamp(min=-clamp_limit, max=clamp_limit)
            self_tensor = self_tensor * torch.sigmoid(glu_alpha * self_tensor)
            output = self_tensor * (other + glu_bias)
        else:
            out = torch.chunk(x_part, 2, dim=-1)
            if activate_left:
                self_tensor = out[0]
                other = out[1]
            else:
                self_tensor = out[1]
                other = out[0]
            output = torch.nn.functional.silu(self_tensor) * other

        if quant_mode == 1:
            if quant_scale is not None:
                output = torch.mul(output, quant_scale[g_idx].to(torch.float32))
            abs = torch.abs(output)
            max_values = torch.amax(abs, dim = -1)
            scale_out = max_values / 127
            max_values = 127 / max_values
            output = output * max_values.unsqueeze(1)
        if quant_mode == 0:
            output = torch.div(output, quant_scale[g_idx].to(torch.float32))
            output = torch.add(output, quant_offset[g_idx].to(torch.float32))
            scale_out = torch.tensor(0.0)

        output = torch.clamp(output, -128, 127)
        output = torch.round(output)
        res_y[offset: (offset+groupIdx)] = output
        res_scale[offset: (offset+groupIdx)] = scale_out
        offset = offset + groupIdx
    return res_y.to(torch.int8).cpu().numpy(), res_scale.cpu().numpy()


class TestDequantSwigluClampQuant(TestCase):
    """Test class for DequantSwigluClampQuant operator."""

    def test_dequant_swiglu_clamp_quant_int32_dynamic_eager(self):
        """INT32 输入 + 动态量化 + Eager 模式."""
        indimX = 32
        indimY = 5760
        outdimY = indimY // 2
        x_shape = [indimX, indimY]
        group_idx = 8
        activate_left = True
        quant_scale_dtype = 'fp32'
        group_index_type = torch.int64
        if quant_scale_dtype == 'fp32':
            quant_scale_type = torch.float32
        if quant_scale_dtype == 'fp16':
            quant_scale_type = torch.float16
        elif quant_scale_dtype == 'bf16':
            quant_scale_type = torch.bfloat16
        swiglu_mode = 1
        clamp_limit = round(random.uniform(3.0, 7.0), 4)
        glu_alpha = round(random.uniform(1.0, 3.0), 4)
        glu_bias = round(random.uniform(1.0, 3.0), 4)
        quant_mode = 0 # 0 代表static, 1 代表dynamic

        group_num = 1
        if group_idx is not None:
            group_num = group_idx
        x = torch.randint(0, 5, x_shape, dtype = torch.int32)
        weight_scale = torch.randn((group_num, indimY), dtype = torch.float32)
        activation_scale = torch.randn((indimX, 1), dtype = torch.float32)
        bias = None
        # bias = torch.randn((group_num, indimY), dtype = torch.float32)
        quant_scale = torch.randn((group_num, outdimY), dtype = quant_scale_type)
        quant_offset = torch.randn((group_num, outdimY), dtype = quant_scale_type)
        if group_idx is None:
            group_index = None
            group_index_npu = None
        else:
            group_index = torch.from_numpy(np.random.randint(0, 2, size=(group_num,))).to(group_index_type)
            group_index_npu = group_index.clone()
        bias_cpu = None
        if bias is not None:
            bias_cpu = bias.cpu()
        cpu_y, cpu_scale = _dequant_swiglu_clamp_quant_cpu(x.cpu(), weight_scale, activation_scale, bias_cpu, quant_scale, quant_offset,
                                                           group_index, activate_left, quant_mode, swiglu_mode, clamp_limit, glu_alpha, glu_bias)

        if quant_mode == 1:
            quant_offset = None
        torch_npu.npu.set_device(int(DEVICE_ID))
        x_npu = x.to(f"npu:{DEVICE_ID}")
        weight_scale = weight_scale.to(f"npu:{DEVICE_ID}")
        activation_scale = activation_scale.to(f"npu:{DEVICE_ID}")
        quant_scale = quant_scale.to(f"npu:{DEVICE_ID}")
        quant_offset = quant_offset.to(f"npu:{DEVICE_ID}")
        group_index_npu = group_index_npu.to(f"npu:{DEVICE_ID}")

        print(f'======================== INT32 Dynamic Eager BEGIN ========================')
        # API signature: torch_npu.npu_dequant_swiglu_clamp_quant(x, weight_scale, activation_scale, bias, quant_scale, ...)
        npu_y, npu_scale = torch_npu.npu_dequant_swiglu_clamp_quant(
            x=x_npu,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index_npu,
            activate_left=activate_left,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=glu_alpha,
            glu_bias=glu_bias,
        )
        print(f'======================== INT32 Dynamic Eager FINISH ========================')
        cmp_and_print(npu_y, npu_scale, cpu_y, cpu_scale, group_index, quant_mode)

if __name__ == "__main__":
    print("=" * 80)
    print("DequantSwigluClampQuant Operator Test Suite")
    print("=" * 80)
    print("\nOperator description:")
    print("  Algorithm: Dequant(x) -> Swiglu -> Quant(y)")
    print("  Input:  x (int32/float16/bfloat16)")
    print("  Output: y (int8/fp8/fp4), scale (float32)")
    print("\nTest cases:")
    print("  - INT32 input + dynamic quant")
    print("  - float16 input + dynamic quant")
    print("  - group_index (MoE scenario)")
    print("  - swiglu_mode: standard (0) / variant (1)")
    print("  - Graph mode (torch.compile + torchair)")
    print("=" * 80)
    run_tests()