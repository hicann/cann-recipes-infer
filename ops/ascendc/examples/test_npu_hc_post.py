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

def _hc_post_cpu(x, residual, post, comb):
    data_type = x.dtype
    x = x.float()
    residual = residual.float()
    post = post.float()
    comb = comb.float()
    hc = residual.shape[2]
    out_shape = list(residual.shape)
    out = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=x.dim() - 1)
    out = out.to(data_type)
    out = out.reshape(out_shape)
    return out


class TestHcPost(TestCase):
    def test_hc_post_b4_s4(self):
        b = 4
        s = 4
        hc = 4
        d_list = [4096, 7168]
        for d in d_list:
            np.random.seed(0)
            x = torch.tensor(np.random.uniform(-10, 10, (b, s, d))).to(torch.bfloat16)
            residual = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, d))).to(torch.bfloat16)
            post = torch.tensor(np.random.uniform(-10, 10, (b, s, hc))).to(torch.bfloat16)
            comb = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, hc))).to(torch.bfloat16)
            cpuout = _hc_post_cpu(x, residual, post, comb)

            torch_npu.npu.set_device(int(DEVICE_ID))
            x = x.to("npu:%s" % DEVICE_ID)
            residual = residual.to("npu:%s" % DEVICE_ID)
            post = post.to("npu:%s" % DEVICE_ID)
            comb = comb.to("npu:%s" % DEVICE_ID)
            # start run custom ops
            print(f'======================== PTA eager BEGIN ========================')
            npu_out = torch.ops.custom.npu_hc_post(x, residual, post, comb)

            # compare result
            npu_out = npu_out.reshape(b, s, hc, d).cpu().float()
            cpuout = cpuout.reshape(b, s, hc, d).float()
            for i in range(b):
                for j in range(s):
                    for k in range(hc):
                        for l in range(d):
                            if torch.abs(npu_out[i][j][k][l] - cpuout[i][j][k][l]):
                                print("i j k l npu cpu = ", i, j, k, l, npu_out[i][j][k][l], cpuout[i][j][k][l])
            print(f'======================== PTA eager FINISH ========================')

    def test_hc_post_b4_s4_float(self):
        b = 4
        s = 4
        hc = 4
        d_list = [4096, 7168]
        for d in d_list:

            np.random.seed(0)
            x = torch.tensor(np.random.uniform(-10, 10, (b, s, d))).to(torch.float)
            residual = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, d))).to(torch.float)
            post = torch.tensor(np.random.uniform(-10, 10, (b, s, hc))).to(torch.float)
            comb = torch.tensor(np.random.uniform(-10, 10, (b, s, hc, hc))).to(torch.float)
            cpuout = _hc_post_cpu(x, residual, post, comb)

            torch_npu.npu.set_device(int(DEVICE_ID))
            x = x.to("npu:%s" % DEVICE_ID)
            residual = residual.to("npu:%s" % DEVICE_ID)
            post = post.to("npu:%s" % DEVICE_ID)
            comb = comb.to("npu:%s" % DEVICE_ID)
            # start run custom ops
            print(f'======================== PTA eager BEGIN ========================')
            npu_out = torch.ops.custom.npu_hc_post(x, residual, post, comb)

            # compare result
            npu_out = npu_out.reshape(b, s, hc, d).cpu().float()
            cpuout = cpuout.reshape(b, s, hc, d).float()
            for i in range(b):
                for j in range(s):
                    for k in range(hc):
                        for l in range(d):
                            if torch.abs(npu_out[i][j][k][l] - cpuout[i][j][k][l]) > 0.001:
                                print("i j k l npu cpu = ", i, j, k, l, npu_out[i][j][k][l], cpuout[i][j][k][l])
            print(f'======================== PTA eager FINISH ========================')
    def test_hc_post_bs16(self):
        bs = 16
        hc = 4
        d = 4096
        d_list = [4096, 7168]
        for d in d_list:

            np.random.seed(0)
            x = torch.tensor(np.random.uniform(-10, 10, (bs, d))).to(torch.bfloat16)
            residual = torch.tensor(np.random.uniform(-10, 10, (bs, hc, d))).to(torch.bfloat16)
            post = torch.tensor(np.random.uniform(-10, 10, (bs, hc))).to(torch.bfloat16)
            comb = torch.tensor(np.random.uniform(-10, 10, (bs, hc, hc))).to(torch.bfloat16)
            cpuout = _hc_post_cpu(x, residual, post, comb)

            torch_npu.npu.set_device(int(DEVICE_ID))
            x = x.to("npu:%s" % DEVICE_ID)
            residual = residual.to("npu:%s" % DEVICE_ID)
            post = post.to("npu:%s" % DEVICE_ID)
            comb = comb.to("npu:%s" % DEVICE_ID)
            # start run custom ops
            print(f'======================== PTA eager BEGIN ========================')
            npu_out = torch.ops.custom.npu_hc_post(x, residual, post, comb)

            # compare result
            npu_out = npu_out.reshape(bs, hc, d).cpu().float()
            cpuout = cpuout.reshape(bs, hc, d).float()
            for i in range(bs):
                for k in range(hc):
                    for l in range(d):
                        if torch.abs(npu_out[i][k][l] - cpuout[i][k][l]):
                            print("i k l npu cpu = ", i, k, l, npu_out[i][k][l], cpuout[i][k][l])
            print(f'======================== PTA eager FINISH ========================')

    def test_hc_post_bs16_float(self):
        bs = 16
        hc = 4
        d_list = [4096, 7168]
        for d in d_list:
            np.random.seed(0)
            x = torch.tensor(np.random.uniform(-10, 10, (bs, d))).to(torch.float)
            residual = torch.tensor(np.random.uniform(-10, 10, (bs, hc, d))).to(torch.float)
            post = torch.tensor(np.random.uniform(-10, 10, (bs, hc))).to(torch.float)
            comb = torch.tensor(np.random.uniform(-10, 10, (bs, hc, hc))).to(torch.float)
            cpuout = _hc_post_cpu(x, residual, post, comb)

            torch_npu.npu.set_device(int(DEVICE_ID))
            x = x.to("npu:%s" % DEVICE_ID)
            residual = residual.to("npu:%s" % DEVICE_ID)
            post = post.to("npu:%s" % DEVICE_ID)
            comb = comb.to("npu:%s" % DEVICE_ID)
            # start run custom ops
            print(f'======================== PTA eager BEGIN ========================')
            npu_out = torch.ops.custom.npu_hc_post(x, residual, post, comb)

            # compare result
            npu_out = npu_out.reshape(bs, hc, d).cpu().float()
            cpuout = cpuout.reshape(bs, hc, d).float()
            for i in range(bs):
                for k in range(hc):
                    for l in range(d):
                        if torch.abs(npu_out[i][k][l] - cpuout[i][k][l]) > 0.001:
                            print("i k l npu cpu = ", i, k, l, npu_out[i][k][l], cpuout[i][k][l])
            print(f'======================== PTA eager FINISH ========================')

if __name__ == "__main__":
    run_tests()