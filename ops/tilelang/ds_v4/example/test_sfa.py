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
from sparse_flash_attention import sparse_attn_kernel

torch.set_default_device('npu')
torch.manual_seed(42)


# golden
def sparse_attn(
    query_states: torch.Tensor,
    kv_states: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float
):
    pattern_query_list = query_states.split(64, dim=2)
    pattern_sink_list = attn_sink.split(64)
    kv_states = kv_states.unsqueeze(1)
    res = []
    for i in range(len(pattern_query_list)):
        pattern_query_states = pattern_query_list[i]
        pattern_attn_sink = pattern_sink_list[i]
        pattern_query_states = pattern_query_states.transpose(1, 2)
        attn_weights = (
            torch.matmul(pattern_query_states, kv_states.transpose(2, 3)) * softmax_scale
        )
        topk_idxs = topk_idxs.to(pattern_query_states.device)
        index_mask = torch.full((pattern_query_states.shape[0], 1, pattern_query_states.shape[2], kv_states.shape[2] + 1),
                                fill_value=torch.finfo(torch.float32).min,
                                dtype=torch.float32, device="npu").scatter_(-1, topk_idxs.unsqueeze(1), 0)
        attn_weights = attn_weights + index_mask[..., :-1]
        sinks = pattern_attn_sink.reshape(1, -1, 1, 1).expand(pattern_query_states.shape[0], -1, pattern_query_states.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores, kv_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        res.append(attn_output)
    return torch.cat(res, dim=2)

def test():
    # Input data dtype and shape
    dtype = torch.bfloat16
    b, m, n, h, d, topk = 1, 256, 256, 64, 512, 128  # Shape 1
    # b, m, n, h, d, topk = 1, 6, 6, 16, 512, 6  # Shape 2

    q = torch.rand((b, m, h, d), dtype=dtype)
    kv = torch.rand((b, n, d), dtype=dtype)
    attn_sink = torch.rand((h), dtype=torch.float32)
    topk_idxs = torch.rand((b, m, topk), dtype=torch.int32)
    output_golden = torch.zeros((b, m, h, d), dtype=dtype)
    softmax_scale = 512 ** -0.5

    func = sparse_attn_kernel(h=h, d=d, scale=softmax_scale)

    logging.info("init successful!")

    output = func(q, kv, attn_sink, topk_idxs)
    torch.npu.synchronize()

    output_golden = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    torch.testing.assert_close(output_golden, output, rtol=1e-2, atol=1e-2)
    logging.info("Test passed!")


if __name__ == "__main__":
    test()
