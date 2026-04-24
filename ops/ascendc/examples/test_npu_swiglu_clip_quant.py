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

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
MAX_INT8_VALUE = 127
MIN_INT8_VALUE = -128


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1):
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


def _swiglu_clip_quant(x, group_indexs, group_alpha, activate_left=True):
    x_shape = list(x.shape)
    x_shape[-1] = x_shape[-1] // 2
    y_res = torch.zeros(x_shape, dtype=torch.float32)
    scale_res = torch.zeros(x_shape[:-1], dtype=torch.float32)
    offset = 0
    for index in range(group_indexs.shape[0]):
        group_index = group_indexs[index]
        x_part = x[offset: offset + group_index].to(torch.float32)
        if activate_left:
            x_left, x_right = torch.chunk(x_part, 2, dim=-1)
        else:
            x_right, x_left = torch.chunk(x_part, 2, dim=-1)
        
        # swiglu
        y = torch.nn.functional.silu(x_left) * x_right
        # clamp
        abs_value = torch.abs(y)
        max_values = torch.amax(abs_value, dim=-1)
        single_alpha = group_alpha[index]
        max_values = max_values * single_alpha
        min_values = max_values * -1
        y = torch.clamp(y, min=min_values.reshape(y.shape[0], 1), max=max_values.reshape(y.shape[0], 1))
        # quant
        dynamic_scale = max_values / MAX_INT8_VALUE
        y = y / dynamic_scale.unsqueeze(1)
        # clamp
        y = torch.clamp(y, min=MIN_INT8_VALUE, max=MAX_INT8_VALUE)
        # copy
        y_res[offset: offset + group_index] = y
        scale_res[offset: offset + group_index] = dynamic_scale
        #loop index
        offset += group_index

    return torch.round(y_res).to(torch.int8), scale_res


class TestCustomSwigluClipQuant(TestCase):
    def test_swiglu_clip_quant_eager(self):
        m = 5000
        n = 4096

        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-100000, 100000, (m, n))).to(torch.bfloat16)
        group_index = torch.tensor([10, 1000, 2000, 500], dtype=torch.int64)
        group_alpha = torch.tensor([0.12, 0.532, 0.927, 0.5], dtype=torch.float32)
        quant_mode = 1
        clamp_mode = 1
        activate_left = True
        total_nums = group_index.sum().item()
        cpu_y, cpu_scale = _swiglu_clip_quant(x, group_indexs=group_index, group_alpha=group_alpha, 
                                    activate_left=activate_left)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)
        group_index = group_index.to("npu:%s" % DEVICE_ID)
        group_alpha = group_alpha.to("npu:%s" % DEVICE_ID)

        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        npu_y, npu_scale = torch_npu.npu_swiglu_clip_quant(x, group_index, group_alpha,
                                                           quant_mode=quant_mode, clamp_mode=clamp_mode,
                                                           activate_left=activate_left)
        print(f'======================== PTA eager FINISH ========================')
        # compare result
        compare_y = data_compare(cpu_y[:total_nums, :].numpy(), npu_y.cpu()[:total_nums, :].numpy())
        compare_scale = data_compare(cpu_scale[:total_nums].numpy(), npu_scale.cpu()[:total_nums].numpy())
        assert(compare_y[0] == "Pass" and compare_scale[0] == "Pass")


    def test_swiglu_clip_quant_graph(self):
        m = 5000
        n = 4096

        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-100000, 100000, (m, n))).to(torch.bfloat16)
        group_index = torch.tensor([10, 1000, 2000, 500], dtype=torch.int64)
        group_alpha = torch.tensor([0.12, 0.532, 0.927, 0.5], dtype=torch.float32)
        quant_mode = 1
        clamp_mode = 1
        activate_left = True
        total_nums = group_index.sum().item()
        cpu_y, cpu_scale = _swiglu_clip_quant(x, group_index, group_alpha, 
                                              activate_left=activate_left)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)
        group_index = group_index.to("npu:%s" % DEVICE_ID)
        group_alpha = group_alpha.to("npu:%s" % DEVICE_ID)

        # start run custom ops
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, group_index, group_alpha, quant_mode=1, clamp_mode=1, activate_left=True):
                y, scale = torch_npu.npu_swiglu_clip_quant(x, group_index, group_alpha, 
                                                           quant_mode=quant_mode, clamp_mode=clamp_mode, 
                                                           activate_left=activate_left)

                return y, scale
        
        print(f'======================== PTA graph BEGIN ========================')
        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_y, npu_scale = npu_mode(x, group_index, group_alpha, quant_mode=quant_mode, clamp_mode=clamp_mode, 
                                    activate_left=activate_left)
        print(f'======================== PTA graph FINISH ========================')
        # compare result
        compare_y = data_compare(cpu_y[:total_nums, :].numpy(), npu_y.cpu()[:total_nums, :].numpy())
        compare_scale = data_compare(cpu_scale[:total_nums].numpy(), npu_scale.cpu()[:total_nums].numpy())
        assert(compare_y[0] == "Pass" and compare_scale[0] == "Pass")


if __name__ == "__main__":
    run_tests()
