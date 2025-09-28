# coding=utf-8
# This program is free software, you can redistribute it and/or modify.
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


logging.basicConfig(level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from sparse_flash_attention import sparse_attention_fwd


torch.set_default_device('npu')
torch.manual_seed(0)

sparse_fa_func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    top_k=2048,
    kv_stride=1,
)


def ref_sparse_attention_fwd_interface(q_param, kv_param, indices_param, q_start_index_s, kv_stride=4, sm_scale=None,
                                       is_casual=True):
    q_param = q_param.float()
    kv_param = kv_param.float()
    indices_param = indices_param.transpose(1, 2)
    batchsize, sq, head, dim_q = q_param.shape
    batchsize, sk, g, _ = kv_param.shape
    if q_start_index_s is None:
        q_start_index_s = sk * kv_stride - sq

    if kv_param.shape[-1] != 576:
        raise ValueError('you should assign dim otherwise')

    dim = 512
    k = kv_param
    v = kv_param[..., :dim]

    batchsize, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = head // g
    compare_left = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32).view(-1, 1)
    compressed_casual_mask = compare_left >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride,
                                                          dtype=torch.int32).view(1, -1)

    mask = q_param.new_zeros(batchsize, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices_param.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(batchsize, g_index, 1, sq, sk)

    q_param = q_param.view(batchsize, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q_param, k)
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(batchsize, g_index, h_index, -1, sq, sk)
    p = p.view(batchsize, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(batchsize, sq, head, dim_v)
    return o.to(torch.float16)


def test_sparse_flash_attention():
    b, s, skv, h, hkv, dqk, dv, topk = 1, 128, 32768, 128, 1, 576, 512, 2048
    dtype = torch.float16

    kv_stride = 1
    q_start_s_index = 4096 * 7

    q = torch.randn((b, s, h, dqk), dtype=dtype)
    kv = torch.randn((b, skv, hkv, dqk), dtype=dtype)
    indices = torch.full((b, s, hkv, topk), skv, dtype=torch.int32)
    for b in range(b):
        for t in range(s):
            for h in range(hkv):
                i_i = torch.randperm(max(1, ((t + q_start_s_index) // kv_stride)))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    workspace_1 = torch.zeros((256, 64, 512), dtype=dtype)
    workspace_2 = torch.zeros((256, 64, 64), dtype=dtype)
    workspace_3 = torch.zeros((256, 64, 64), dtype=torch.float)
    workspace_4 = torch.zeros((256, 64, 64), dtype=dtype)
    workspace_5 = torch.zeros((256, 64, 512), dtype=torch.float)

    torch.npu.synchronize()

    logging.info("init successful!")

    output = sparse_fa_func(q, kv, indices, workspace_1, workspace_2, workspace_3, workspace_4, workspace_5)
    torch.npu.synchronize()

    ref_output = ref_sparse_attention_fwd_interface(q, kv, indices, q_start_s_index, kv_stride)
    torch.npu.synchronize()
    torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
    logging.info("Test passed!")


if __name__ == "__main__":
    test_sparse_flash_attention()
