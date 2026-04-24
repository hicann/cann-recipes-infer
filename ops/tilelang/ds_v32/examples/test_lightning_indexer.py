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


B = 2
N2 = 1
G = 32
S1 = 512
S2 = 4096
D = 64
TOP_K = 1024


def index_golden(q, k, weights):
    score_1 = torch.einsum("bsmgd, btmd->bmsgt", q, k)
    score_1 = score_1.relu()
    score = score_1.permute(0, 2, 1, 3, 4)
    mul_res = score * weights
    reduce_res = torch.sum(mul_res, dim=3)
    golden_out = torch.topk(reduce_res, TOP_K, dim=3, largest=True, sorted=True)
    return score_1.float(), golden_out.indices.to(torch.int32).permute(0, 2, 1, 3)


def count_mismatches_last_dim(tensor1, tensor2):
    assert tensor1.shape[-1] == tensor2.shape[
        -1], "the last dimension of two tensors must be the same"
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
        print("error: two tensors have different shapes")
        print(f"tensor1 shape: {tensor1.shape}")
        print(f"tensor2 shape: {tensor2.shape}")
        return

    diff_mask = tensor1 != tensor2

    if not torch.any(diff_mask):
        print("two tensors are completely the same")
        return

    diff_indices = torch.nonzero(diff_mask)

    print(f"found {len(diff_indices)} different elements:")
    print("index\t\ttensor1 value\t\ttensor2 value")
    print("-" * 40)

    for idx in diff_indices:
        idx_str = str(tuple(idx.tolist()))

        val1 = tensor1[tuple(idx)]
        val2 = tensor2[tuple(idx)]

        print(f"{idx_str}\t{val1.item()}\t\t{val2.item()}")


def test_indexer():
    func = indexer(B, N2, G, S1, S2, D, TOP_K, 256, 16, 64, 64, 64)
    # print(f"{func.get_kernel_source()}")

    q = torch.randn(B, S1, N2, G, D).half()
    k = torch.randn(B, S2, N2, D).half()
    weights = torch.randn(B, S1, N2, G, 1).float()

    # qk_res_workspace = torch.zeros(B, N2, S1, G, S2).float()
    qk_res_workspace_, golden_out = index_golden(q, k, weights)

    q_npu = q.view(B, S1, N2, -1).npu()
    k_npu = k.npu()
    weights_npu = weights.npu()
    # qk_res_workspace_npu = qk_res_workspace.npu()
    torch.npu.synchronize()
    npu_out = func(q_npu, k_npu, weights_npu).to(torch.int32)
    torch.npu.synchronize()

    total_mismatches = count_mismatches_last_dim(golden_out.cpu(), npu_out.cpu())

    if (1 - total_mismatches / (B * S1 * N2 * TOP_K)) > 0.99:
        print("Test passed!")
    else:
        print('Test failed! The precision is not correct!')


if __name__ == "__main__":
    test_indexer()