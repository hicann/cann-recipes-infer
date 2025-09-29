# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import torch
import logging
from collections import Counter


logging.basicConfig(level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from lightning_indexer import indexer


torch.manual_seed(2)


def index_golden(q, k, weights, top_k):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k)
    score_1 = score_1.relu()
    score = score_1.permute(0, 2, 1, 3, 4)

    mul_res = score * weights
    reduce_res = torch.sum(mul_res, dim=3)
    golden_out = torch.topk(reduce_res, top_k, dim=3, largest=True, sorted=True)
    return score_1.float(), golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_mismatches_last_dim(tensor1, tensor2):
    if tensor1.shape[-1] != tensor2.shape[-1]:
        raise ValueError("The sizes of the last dimension must be the same")

    last_dim = tensor1.shape[-1]
    tensor1_flat = tensor1.view(-1, last_dim)
    tensor2_flat = tensor2.view(-1, last_dim)
    total_mismatches = 0

    for i in range(tensor1_flat.shape[0]):
        row1 = tensor1_flat[i].tolist()
        row2 = tensor2_flat[i].tolist()

        counter1 = Counter(row1)
        counter2 = Counter(row2)

        diff = (counter1 - counter2) + (counter2 - counter1)
        total_mismatches += sum(diff.values())

    return total_mismatches


def compare_tensors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        return

    diff_mask = tensor1 != tensor2

    if not torch.any(diff_mask):
        return

    diff_indices = torch.nonzero(diff_mask)

    for idx in diff_indices:
        idx_str = str(tuple(idx.tolist()))

        val1 = tensor1[tuple(idx)]
        val2 = tensor2[tuple(idx)]


def test_indexer():
    b = 2
    n2 = 1
    g = 64
    s1 = 1024
    s2 = 8192
    d = 128
    top_k = 2048

    func = indexer(b, n2, g, s1, s2, d, top_k)

    q = torch.randn(b, s1, n2, g, d).half()
    k = torch.randn(b, s2, n2, d).half()
    weights = torch.randn(b, s1, n2, g, 1).float()

    qk_res_workspace = torch.zeros(b, n2, s1, g * s2).float()
    qk_res_workspace_, golden_out = index_golden(q, k, weights, top_k)

    q_npu = q.view(b, s1, n2, -1).npu()
    k_npu = k.npu()
    weights_npu = weights.npu()
    qk_res_workspace_npu = qk_res_workspace.npu()
  
    torch.npu.synchronize()
    npu_out = func(q_npu, k_npu, qk_res_workspace_npu, weights_npu).to(torch.int32)
    torch.npu.synchronize()

    total_mismatches = count_mismatches_last_dim(golden_out.cpu(), npu_out.cpu())

    if (1 - total_mismatches / (b * s1 * n2 * top_k)) > 0.99:
        logging.info("Test passed!")
    else:
        logging.info('Test failed! The precision is not correct!')


if __name__ == "__main__":
    test_indexer()
