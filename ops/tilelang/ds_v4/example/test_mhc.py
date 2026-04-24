# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import torch
import logging
from torch import nn


logging.basicConfig(level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from hc_split_sinkhorn import hc_split_sinkhorn

torch.set_default_device('npu')
torch.manual_seed(42)


# golden
def hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps):
    n = mixes.shape[0]
    dtype = torch.float32
    pre_ref = torch.empty((n, hc), dtype=dtype)
    post_ref = torch.empty((n, hc), dtype=dtype)
    comb_ref = torch.empty((n, hc, hc), dtype=dtype)

    for i in range(n):
        for j in range(hc):
            pre_ref[i, j] = torch.sigmoid(mixes[i, j] * hc_scale[0] + hc_base[j]) + eps
            post_ref[i, j] = 2 * torch.sigmoid(mixes[i, j + hc] * hc_scale[1] + hc_base[j + hc])
            for k in range(hc):
                comb_ref[i, j, k] = mixes[i, j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]
        
        # comb = comb.softmax(-1) + eps
        row_max, row_max_indices = torch.max(comb_ref[i,:,:], dim=1)
        for j in range(hc):
            comb_ref[i, j, :] = torch.exp(comb_ref[i, j, :] - row_max[j])
        row_sum = torch.sum(comb_ref[i,:,:], dim=1)
        for j in range(hc):
            comb_ref[i, j, :] = comb_ref[i, j, :] / row_sum[j] + eps
        # print(f"i: {i}, comb_ref: {comb_ref[i, :, :]}")

        # comb = comb / (comb.sum(-2) + eps)
        col_sum = torch.sum(comb_ref[i,:,:], dim=0)
        # print(f"i: {i}, col_sum: {col_sum}")
        for k in range(hc):
            comb_ref[i, :, k] = comb_ref[i, :, k] / (col_sum[k] + eps)

        for _ in range(sinkhorn_iters - 1):
            # comb = comb / (comb.sum(-1) + eps)
            row_sum = torch.sum(comb_ref[i,:,:], dim=1)
            for j in range(hc):
                comb_ref[i, j, :] = comb_ref[i, j, :] / (row_sum[j] + eps)
            # comb = comb / (comb.sum(-2) + eps)
            col_sum = torch.sum(comb_ref[i,:,:], dim=0)
            for k in range(hc):
                comb_ref[i, :, k] = comb_ref[i, :, k] / (col_sum[k] + eps)
    
    return pre_ref, post_ref, comb_ref
          

def test():
    # Input data dtype and shape
    dtype = torch.float32
    B, S, hc_mult = 1, 5, 4
    mix_hc = (2 + hc_mult) * hc_mult
    N = B * S

    mixes = torch.rand((N, mix_hc), dtype=dtype)
    hc_scale = torch.rand(3, dtype=dtype)
    hc_base = torch.rand(mix_hc, dtype=dtype)

    pre = torch.empty((N, hc_mult), dtype=dtype).npu()
    post = torch.empty((N, hc_mult), dtype=dtype).npu()
    comb = torch.empty((N, hc_mult, hc_mult), dtype=dtype).npu()
    torch.npu.synchronize()

    func = hc_split_sinkhorn(hc=hc_mult, sinkhorn_iters=20, eps=1e-6)

    logging.info("init successful!")

    pre, post, comb = func(mixes, hc_scale, hc_base)

    pre_ref, post_ref, comb_ref = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, 20, 1e-6)
    torch.npu.synchronize()

    torch.testing.assert_close(pre_ref, pre, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(post_ref, post, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_ref, comb, rtol=1e-2, atol=1e-2)

    logging.info("Test passed!")


if __name__ == "__main__":
    test()
