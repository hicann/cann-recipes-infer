# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import math
import numpy as np
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
import sys
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"
import logging
import torchair
import custom_pypto
torchair.logger.setLevel(logging.DEBUG)
np.random.seed(0)



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, weights, actual_seq_lengths_key, block_table, score_scale=1.0000, sparse_count=2048,
                actual_seq_lengths_query=None, layout_query="BSND", layout_key="PA_BSND", sparse_mode=3):
        inputs=[query, key, weights, actual_seq_lengths_key, block_table]
        topk_res = torch.ops.custom_pypto.npu_lightning_indexer_pto(
            query=query, key=key, weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            sparse_mode=sparse_mode
            )
        return topk_res

def gen_block_table(b, block_size, max_kv, act_kv):
    logging.info("Entering into gen_block_table!")
    block_num = 0
    block_num_each = []
    for cur_s in act_kv:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    shape_bt = [b, math.ceil(max_kv / block_size)]
    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * shape_bt[1]
    block_table = np.tile(block_table, (shape_bt[0], 1)).astype(np.int32)

    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    return block_num, block_table

def gen_cache_tensor(k_tensor, block_table, block_num, block_size, b):
    logging.info("Entering into gen_cache_tensor!")
    dtype = k_tensor.dtype
    b, s, n, d = k_tensor.shape
    k_cache = torch.zeros([block_num, block_size, n * d], dtype=dtype)
    k_tensor_bsh_raw = k_tensor.reshape(b, s, n * d)

    # kv padding
    k_tensor_bsh = torch.zeros(
        (b, block_table.shape[1] * block_size, n * d), dtype=dtype)
    k_tensor_bsh[:, : k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx != -1:
                k_cache[cache_block_idx, :, :] = k_tensor_bsh[b_idx,
                                                              block_offset: (block_offset + block_size), :]

    k_cache = k_cache.reshape(block_num, block_size, n, d)
    return k_cache

def gen_data_for_compute(params):
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    dtype = params.get("dtype")
    s2 = params.get("s2")
    n2 = params.get("n2")
    act_seq_len = params.get("act_seq")
    block_size = params.get("block_size")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    sparse_count = params.get("sparse_count")
    score_scale = params.get("score_scale")

    query = torch.randn([b, s1, n1, d], dtype=dtype)
    weights = torch.randn([b, s1, n1], dtype=dtype)

    k_bsnd = torch.randn([b, s2, n2, d], dtype=dtype)
    _, block_table_list = gen_block_table(b, block_size, s2, act_seq_len)
    block_table = torch.tensor(block_table_list, dtype=torch.int32)
    act_seq = torch.tensor(act_seq_len, dtype=torch.int32)

    # (block_num, block_size, n, d)
    key = gen_cache_tensor(k_bsnd, block_table_list, block_num, block_size, b)
    # construct output tensor
    topk_res = torch.zeros([b, s1, n2, sparse_count], dtype=torch.int32)

    input_data_map = {}
    input_data_map["query"] = query
    input_data_map["key"] = key
    input_data_map["weights"] = weights
    input_data_map["act_seq"] = act_seq
    input_data_map["block_table"] = block_table

    input_params = [
        b,
        s1,
        n1,
        d,
        block_num,
        block_size,
        n2,
        max_block_num,
        sparse_count,
    ]

    return input_data_map

def indexer_topk_compute(input_data_map, params):
    block_size = params.get("block_size")  # 128
    sparse_count = params.get("sparse_count")
    b = params.get("b")
    s1 = params.get("s1")
    n1 = params.get("n1")
    d = params.get("d")
    n2 = params.get("n2")
    block_num = params.get("block_num")
    max_block_num = params.get("max_block_num")
    score_scale = params.get("score_scale")
    dtype = params.get("dtype")

    # get input tensors
    query = input_data_map.get("query")
    key = input_data_map.get("key")
    weights = input_data_map.get("weights")
    act_seq = input_data_map.get("act_seq")
    block_table = input_data_map.get("block_table")

    topk_value = torch.zeros([b, s1, n2, sparse_count], dtype=torch.float32)
    topk_res = torch.zeros([b, s1, n2, sparse_count], dtype=torch.int32)
    tmp_out = torch.zeros([b*s1*n2, max_block_num * block_size], dtype=torch.float32)

    g = n1 // n2
    query = query.reshape(b * s1 * n1, d)
    key = key.reshape(block_num * block_size, n2 * d)
    weights = weights.reshape(b * s1 * n1, 1)

    for b_idx in range(b):
        cur_seq = act_seq[b_idx]
        for s_idx in range(s1):
            casual_offset = s1 - s_idx - 1
            eff_seq = cur_seq - casual_offset
            actual_block = (eff_seq + block_size - 1) // block_size
            for n2_idx in range(n2):
                local_sum = torch.zeros(
                    [1, max_block_num * block_size], dtype=torch.float32)
                for block_idx in range(actual_block):
                    remain_s2 = min(block_size, eff_seq -
                                    block_size * block_idx)
                    cur_block_idx = block_table[b_idx][block_idx]
                    q_offset = b_idx * s1 * n1 + s_idx * n1 + n2_idx * g
                    cur_q = query[q_offset: (q_offset + g), :]
                    cur_k = key[cur_block_idx * block_size: (
                        cur_block_idx * block_size + remain_s2), n2_idx * d: ((n2_idx + 1) * d)]
                    cur_w = weights[q_offset: (q_offset + g), :]

                    mm_res = torch.matmul(cur_q.to(torch.float32), cur_k.t().to(
                        torch.float32)).to(torch.float32)
                    zero_tensor = torch.zeros(
                        [g, remain_s2], dtype=torch.float32)
                    relu_res = torch.maximum(mm_res, zero_tensor)
                    mul_res = relu_res * cur_w
                    sum_res = mul_res.sum(dim=0, keepdim=True)
                    local_sum[:, block_idx * block_size: (block_idx * block_size + remain_s2)] = sum_res
                    cur_n2_idx = b_idx * s1 * n2 + s_idx * n2 + n2_idx
                    tmp_out[cur_n2_idx:(cur_n2_idx + 1), block_idx * block_size: (block_idx * block_size + remain_s2)] = sum_res
                eff_sum_res = local_sum[:, :eff_seq]
                k_num = sparse_count
                if eff_seq < sparse_count:
                    k_num = eff_seq
                cur_value, cur_index = torch.topk(eff_sum_res, k=k_num, dim=1)
                topk_value[b_idx, s_idx, n2_idx, :eff_seq] = cur_value.reshape(1, 1, 1, k_num)
                topk_res[b_idx, s_idx, n2_idx, :eff_seq] = cur_index.reshape(1, 1, 1, k_num)

                if eff_seq < sparse_count:
                    topk_value[b_idx, s_idx, n2_idx, eff_seq:] = (-float(3.40282347e38)) * \
                        torch.ones([1, 1, 1, sparse_count -
                                   eff_seq], dtype=torch.float32)
                    topk_res[b_idx, s_idx, n2_idx, eff_seq:] = -1 * \
                        torch.ones([1, 1, 1, sparse_count -
                                   eff_seq], dtype=torch.int32)

    return topk_value, topk_res, tmp_out

def build_npu_graph():

    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_device(0)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

    print("=========== start gen lightning indexer golden=============:")
    ########################  gen golden
    ### param
    dtype = torch.bfloat16

    n1, n2, d = 64, 1, 128
    block_size = 128
    sparse_count = 2048

    ##############
    b, s1 = 32, 3

    act_seq = torch.randint(1, 8193, (32,))
    print("act_seq is ", act_seq)

    s2 = max(act_seq) # s2 means max act_seq
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])
    max_block_num = (s2 + block_size - 1) // block_size

    n1_scale = 1.0 / np.sqrt(n1)
    softmax_scale = 1.0 / np.sqrt(d)
    score_scale = n1_scale * softmax_scale

    params = {}
    params["b"] = b
    params["s1"] = s1
    params["n1"] = n1
    params["d"] = d
    params["dtype"] = dtype
    params["s2"] = s2
    params["n2"] = n2
    params["act_seq"] = act_seq
    params["block_size"] = block_size
    params["block_num"] = block_num
    params["max_block_num"] = max_block_num
    params["sparse_count"] = sparse_count
    params["score_scale"] = score_scale

    input_data_map = gen_data_for_compute(params)
    topk_value, topk_res_golden, tmp_out = indexer_topk_compute(input_data_map, params)

    query = input_data_map["query"]
    key = input_data_map["key"]
    weights = input_data_map["weights"]
    act_seq = input_data_map["act_seq"]
    block_table = input_data_map["block_table"]

    topk_res =  model(query, key, weights, act_seq, block_table, 1, 2048)
    topk_res = topk_res.cpu()

    print(f" ")
    print(f"===================================topk res npu===================================")
    print(f"{topk_res}")
    error_cnt = 0
    result = True
    batch_threshold = 1
    for b_idx in range(b):
        for s1_idx in range(s1):
            topk_res_part = topk_res[b_idx, s1_idx, :, :].view(-1).sort().values
            topk_res_gloden_part = topk_res_golden[b_idx, s1_idx, :, :].view(-1).sort().values
            only_in_topk_res_npu = set(topk_res_part.tolist()) - set(topk_res_gloden_part.tolist())
            diff_cnt = len(only_in_topk_res_npu)
            error_cnt += diff_cnt
            if diff_cnt > batch_threshold:
                print(f"diff cnt in each batch is: {diff_cnt} diff threshold in each batch is: {batch_threshold}", flush=True)
                print(f"only in topk res npu {only_in_topk_res_npu}", flush=True)
                result = False
                break
        if not result:
            break
    print(f" ")
    print(f"error cnt is: {error_cnt}, error threshold is: {b * batch_threshold}, compare result is {result}", flush=True)
    if result is True:
        print(f"\033[1;32m================case lightning indexer topk run success================\033[0m \n")
    else:
        print(f"\033[1;31m================ case lightning indexer topk precision failed ================\033[0m \n")

    return topk_res

if __name__ == "__main__":
    res = build_npu_graph()
    print("===execute end ", flush=True)
