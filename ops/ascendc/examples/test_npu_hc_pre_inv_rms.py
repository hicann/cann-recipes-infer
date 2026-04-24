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


def data_compare(npu_out, cpu_out, diff_thd=0.0001, pct_thd=0.0005, max_diff_hd=0.0001):
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


def _hc_pre_inv_rms(x, epsilon=1e-20):
    if x.dim() == 4: 
        x = x.flatten(2)
    elif x.dim() == 3:
        x = x.flatten(1)
    x = x.float()
    y = torch.rsqrt(x.square().mean(-1, keepdim = True) + epsilon)
    return y


class TestCustomHcPreInvRms(TestCase):
    def test_hc_pre_inv_rms_eager(self):
        b = 4
        s = 4
        hc = 4
        d = 4096
        eps = 1e-6

        np.random.seed(0)

        # start run custom ops
        print(f'======================== PTA eager BEGIN ========================')
        # float32 input test (flattened input)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.float32)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # bfloat16 input test (flattened input)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.bfloat16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # float16 input test (orgin 4 dims input)
        x = torch.tensor(np.random.uniform(-1, 1, (b * s, hc ,d))).to(torch.float16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")

        # ======================== d=7168 test cases ========================
        d = 7168
        
        # float32 input test (d=7168)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.float32)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # bfloat16 input test (d=7168)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.bfloat16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")


        # float16 input test (d=7168, 3 dims)
        x = torch.tensor(np.random.uniform(-1, 1, (b * s, hc, d))).to(torch.float16)
        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x = x.to("npu:%s" % DEVICE_ID)

        npu_y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=eps)
        
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")
        print(f'======================== PTA eager FINISH ========================')

    def test_hc_pre_inv_rms_graph(self):
        b = 4
        s = 4
        hc = 4
        d = 4096
        eps = 1e-6

        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.bfloat16)

        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x_npu = x.to("npu:%s" % DEVICE_ID)

        # start run custom ops
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, epsilon):
                y = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=epsilon)
                return y
        
        print(f'======================== PTA graph BEGIN ========================')
        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_y = npu_mode(x_npu, eps)
        print(f'======================== PTA graph FINISH ========================')
        # compare result
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")

        # ======================== d=7168 test case ========================
        d = 7168
        x = torch.tensor(np.random.uniform(-1, 1, (b, s, hc, d))).to(torch.bfloat16)

        cpu_y = _hc_pre_inv_rms(x, epsilon=eps)

        torch_npu.npu.set_device(int(DEVICE_ID))
        x_npu = x.to("npu:%s" % DEVICE_ID)

        npu_y = npu_mode(x_npu, eps)
        
        print(f'======================== PTA graph d=7168 FINISH ========================')
        compare_y = data_compare(cpu_y.float().numpy(), npu_y.cpu().float().numpy())
        assert(compare_y[0] == "Pass")

if __name__ == "__main__":
    run_tests()
