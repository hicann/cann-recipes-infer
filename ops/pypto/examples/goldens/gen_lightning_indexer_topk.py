#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from enum import Enum
import math
import os
import sys
import logging
import torch
from pathlib import Path
from typing import List

import numpy as np
import torch
from bfloat16 import bfloat16
import copy


project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)


if __name__ == "__main__":
    """单独执行时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
#     from golden_register import (
#         GoldenRegister,
#     )  # 单独执行 import 失败, 需确认上文中 '系统 import 路径' 配置正确
# else:
#     from golden_register import GoldenRegister


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == "fp16":
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == "fp32":
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == "fp64":
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == "int8":
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == "int16":
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == "int32":
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == "int64":
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == "uint8":
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == "uint16":
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == "uint32":
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == "uint64":
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == "complex64":
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == "complex128":
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == "bool":
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == "bf16":
        np.array(data_pool).astype(bfloat16).tofile(data_path)

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


def gen_data_for_compute(out_path: Path, params):
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

    # define input files
    in_params_path = Path(out_path, "input_params.bin")
    query_path = Path(out_path, "query.bin")
    key_path = Path(out_path, "key.bin")
    weights_path = Path(out_path, "weights.bin")
    act_seq_path = Path(out_path, "act_seq.bin")
    block_table_path = Path(out_path, "block_table.bin")

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

    # dump golden file
    dump_file(input_params, in_params_path, "int32")
    dump_file(query.to(torch.float32).numpy().astype(
        bfloat16), query_path, "bf16")
    dump_file(key.to(torch.float32).numpy().astype(bfloat16), key_path, "bf16")
    dump_file(weights.to(torch.float32).numpy().astype(bfloat16), weights_path, "bf16")
    dump_file(act_seq.numpy(), act_seq_path, "int32")
    dump_file(block_table.numpy(), block_table_path, "int32")

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
                    # topk_res[b_idx, s_idx, n2_idx, eff_seq:] = -1 * \
                    #     torch.ones([1, 1, 1, sparse_count -
                    #                eff_seq], dtype=torch.int32)
                    topk_res[b_idx, s_idx, n2_idx, eff_seq:] = torch.zeros([1, 1, 1, sparse_count - eff_seq], dtype=torch.int32)
 
    return topk_value, topk_res, tmp_out

# @GoldenRegister.reg_golden_func(
#     case_names=[
#         "DynamicIndexerTopk.indxer_topk_bf16_4_b_1_s1_64k_s2",
#         "DynamicIndexerTopk.indxer_topk_bf16_48_b_4_s1_128k_s2",
#         "DynamicIndexerTopk.indxer_topk_bf16_2_b_3_s1_122_s2",
#     ]
# )
def indexer_topk(case_name: str, output: Path) -> bool:
    n1, n2, d = 64, 1, 128
    block_size = 128
    dtype = torch.bfloat16
    if case_name == "DynamicIndexerTopk.indxer_topk_bf16_4_b_1_s1_64k_s2":
        b, s1 = 4, 1
        act_seq = [64*1024] * b
    elif case_name == "DynamicIndexerTopk.indxer_topk_bf16_48_b_4_s1_128k_s2":
        b, s1 = 48, 4
        act_seq = [128*1024, 64*1024, 64*1024+111, 32*1024+11, 16 *
                   1024+1, 8*1024+2, 2*1024+7, 1024+5, 512*4, 256, 127, 32] * 4
    elif case_name == "DynamicIndexerTopk.indxer_topk_bf16_2_b_3_s1_122_s2":
        b, s1 = 2, 3
        act_seq = [122] * b
    else:
        logging.error("Fail to gen golden for Case(%s)", case_name)
        return False

    s2 = max(act_seq) # s2 means max act_seq
    block_num = sum([(s + block_size - 1) // block_size for s in act_seq])
    max_block_num = (s2 + block_size - 1) // block_size
    sparse_count = 2048

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

    input_data_map = gen_data_for_compute(output, params)
    topk_value, topk_res, tmp_out = indexer_topk_compute(input_data_map, params)

    # dump golden for compare res
    topk_value_path = Path(output, "topk_value.bin")
    topk_res_path = Path(output, "topk_res.bin")
    tmp_out_path = Path(output, "tmp_out.bin")
    dump_file(topk_res.numpy(), topk_res_path, "int32")
    dump_file(tmp_out.numpy(), tmp_out_path, "fp32")
    dump_file(topk_value.numpy(), topk_value_path, "fp32")

    return True


def main() -> bool:
    """
    单独执行 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicIndexerTopk.indxer_topk_bf16_4_b_1_s1_64k_s2",
        "DynamicIndexerTopk.indxer_topk_bf16_48_b_4_s1_128k_s2",
        "DynamicIndexerTopk.indxer_topk_bf16_2_b_3_s1_122_s2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = indexer_topk(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
