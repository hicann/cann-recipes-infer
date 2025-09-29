
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

import math
import sys
import logging
from pathlib import Path
from typing import List
import time

import numpy as np
from bfloat16 import bfloat16
import os
import torch

# np.random.seed(0)
# 添加 golden 所在目录的父路径（例如项目根目录）
project_root = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
golden_parent = os.path.join(project_root, "../../../../")  # 假设 golden 在上级目录
sys.path.insert(0, golden_parent)
golden_parent2 = os.path.join(golden_parent, "cmake/scripts")
sys.path.insert(0, golden_parent2)

from . import gen_mla_prolog_golden
from gen_mla_prolog_golden import gen_prolog_input_data, mla_prolog_compute, gen_block_table
from gen_lightning_indexer_topk import indexer_topk_compute
from gen_selected_attention import compute_attention
import gen_lightning_indexer_prolog
from gen_lightning_indexer_prolog import indexer_prolog


if __name__ == "__main__":
    """ 单独执行时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    # g_src_root: Path = Path(Path(__file__).parent, "../../../../cmake/").resolve()
    # logging.debug("SrcRoot: %s", g_src_root)
    # g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    # if str(g_ctrl_path) not in sys.path:
    #     sys.path.append(str(g_ctrl_path))
#     from golden_register import GoldenRegister  # 单独执行 import 失败, 需确认上文中 '系统 import 路径' 配置正确
# else:
#     from golden_register import GoldenRegister


def dump_file(data_pool, data_path, dtype):
    np.array(data_pool).astype(dtype).tofile(data_path)


def dump_file_torch(data, data_path, dtype):
    """将PyTorch张量保存到文件，支持BFloat16类型转换"""
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.int32:
        np_dtype = np.int32
    elif dtype == torch.int64:
        np_dtype = np.int64
    elif dtype == torch.bfloat16:
        np_dtype = bfloat16
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")

    if isinstance(data, torch.Tensor):
        # 处理BFloat16类型：转换为float32后再转NumPy（NumPy不支持BFloat16）
        if data.dtype == torch.bfloat16:
            data_np = data.cpu().to(torch.float32).numpy()
        else:
            data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)

    # 确保最终类型与指定dtype一致
    data_np = data_np.astype(np_dtype)
    data_np.tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def genNsaR2TopkRes(params):
    b = params.get("b")
    s = params.get("s")
    act_seq = params.get("act_seq")
    topk = 2048
    topkResShape = (b,s,topk)
    topkRes = gen_uniform_data(topkResShape, 0, 0, dtype=np.int32)
    for b_idx in range(b):
        topkResLineShape = (s,topk)
        topkRes[b_idx,:,:] = gen_uniform_data(topkResLineShape, 0, act_seq[b_idx], dtype=np.int32)
    return topkRes


def nsaR2Gather(kv_cache, topkRes, params):
    print("Enter gather golden.")
    #     bs2n1d bs1k
    dtype = kv_cache.dtype
    b = params.get("b")
    s1 = params.get("s")
    n2 = params.get("n2")
    act_seq = params.get("act_seq")
    topk = 2048
    dn = params.get("dn")
    dr = params.get("dr")
    gatherResShape = (b, s1, topk, dn + dr)
    gatherRes = gen_uniform_data(gatherResShape, 0, 0, dtype=dtype)
    for b_idx in range(b):
        for s1_idx in range(s1):
            for n2_idx in range(n2):
                act_topk = min(max(act_seq[b_idx] - s1 + 1 + s1_idx, 0), topk)
                print("gather actual topk: ", act_topk)
                for k_idx in range(act_topk):
                    topkVal = topkRes[b_idx, s1_idx, n2_idx, k_idx]
                    gatherRes[b_idx,s1_idx,k_idx] = kv_cache[b_idx,topkVal,:,:].flatten()
    print("gather golden end.")
    return gatherRes


def gen_kv_cache(params, actual_seq_list, dtype, output_dir):
    '''
    生成kv_cache, 包括kv_nope_cache, k_rope_cache, block_table
    '''
    b = params.get("b")
    s1 = params.get("s")
    n2 = params.get("n2")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")

    block_size = params.get("block_size")

    block_num, block_table = gen_block_table(b, actual_seq_list, block_size)

    block_table_path = Path(output_dir, 'block_table.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    dump_file(block_table, block_table_path, np.int32)
    dump_file(actual_seq_list, kv_cache_actual_seq_path, np.int32)

    return block_table, block_num


def kv_cache_concat_bsnd(concat_parms, kr_cache_out, kv_cache_out, block_table, block_num, kv_cache_actual_seq, dtype):
    b = concat_parms[0]
    s = concat_parms[1]
    n2 = concat_parms[2]
    kv_lora_rank = concat_parms[3]
    rope_dim= concat_parms[4]
    block_size = concat_parms[5]

    kv_max = (max(kv_cache_actual_seq) + block_size - 1) // block_size * block_size
    k_cache = np.zeros([b, kv_max, n2, kv_lora_rank], dtype = dtype)
    v_cache = np.zeros([b, kv_max, n2, rope_dim], dtype = dtype)
    for b_idx in range(b):
        block_list = block_table[b_idx]
        kv_nope_temp_tensor = np.zeros([1, kv_max, n2, kv_lora_rank], dtype = dtype)
        kv_rope_temp_tensor = np.zeros([1, kv_max, n2, rope_dim], dtype = dtype)
        s_idx = 0
        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            kv_nope_temp_tensor[:, s_idx * block_size : (s_idx + 1) * block_size, :, :] = kv_cache_out[block_idx : block_idx + 1, :, :, :]
            kv_rope_temp_tensor[:, s_idx * block_size : (s_idx + 1) * block_size, :, :] = kr_cache_out[block_idx : block_idx + 1, :, :, :]
            s_idx += 1
        k_cache[b_idx : b_idx + 1, :, :, :] = kv_nope_temp_tensor
        v_cache[b_idx : b_idx + 1, :, :, :] = kv_rope_temp_tensor
    k_cache_bsnd = np.concatenate([k_cache, v_cache], axis = -1)
    v_cache_bsnd = k_cache
    return k_cache_bsnd, v_cache_bsnd


def mla_prolog_golden(params, is_nz, dtype, output_dir):
    # mla_prolog 数据
    kv_cache_actual_seq = params.get("kv_cache_actual_seq")
    epsilon = params.get("epsilon")
    is_quant = params.get("is_quant")
    has_smooth = params.get("is_smooth")
    cache_mode = params.get("cache_mode")
    b = params.get("b")
    s = params.get("s")
    n2 = params.get("n2")
    kv_lora_rank = params.get("kv_lora_rank")
    rope_dim = params.get("rope_dim")

    block_size = params.get("block_size")

    block_table, block_num = gen_kv_cache(params, kv_cache_actual_seq, dtype, output_dir) # 生成 block_table

    prolog_params = {
        "b": b,
        "s": s,
        "s2": params.get("s2"),
        "h": params.get("h"),
        "num_heads": params.get("n1"),
        "q_lora_rank": params.get("q_lora_rank"),
        "qk_nope_head_dim": params.get("qk_nope_head_dim"),
        "qk_rope_head_dim": rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "v_head_dim": params.get("v_head_dim"),
        "block_num": block_num,
        "block_table": block_table,
        "skv_max": max(kv_cache_actual_seq),
    }
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache = \
        gen_prolog_input_data(prolog_params, [dtype, dtype], epsilon, output_dir, (False, is_quant), is_nz, has_smooth,
                              block_size, cache_mode)

    # mla_prolog 计算
    prolog_inputs = {"dtype": dtype, "is_quant_a": False, "is_quant_b": is_quant, "has_smooth": has_smooth}
    prolog_inputs["cache_mode"] = cache_mode
    prolog_inputs["gamma_cq"] = gamma_cq
    prolog_inputs["gamma_ckv"] = gamma_ckv
    prolog_inputs["epsilon"] = epsilon
    prolog_inputs["x"] = x
    prolog_inputs["w_dq"] = w_dq
    prolog_inputs["w_uqqr"] = w_uqqr
    prolog_inputs["w_uk"] = w_uk
    prolog_inputs["w_dkvkr"] = w_dkvkr
    prolog_inputs["cos"] = cos
    prolog_inputs["sin"] = sin
    prolog_inputs["kv_cache"] = kv_cache
    prolog_inputs["kr_cache"] = kr_cache
    prolog_inputs["cache_index"] = kv_len
    if is_quant:
        prolog_inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            prolog_inputs["smooth_cq"] = smooth_cq
    # q_out: [b, s, n1, kv_lora_rank], q_rope_out: [b, s, n1, rope_dim]
    # kv_cache_out: [block_num, block_size, n2, kv_lora_rank], kr_cache_out: [block_num, block_size, n2, rope_dim]
    q_out, q_rope_out, kv_cache_out, kr_cache_out, q_a_rms_norm = mla_prolog_compute(prolog_inputs)
    q_out.tofile(Path(output_dir, 'q_golden.bin'))
    q_rope_out.tofile(Path(output_dir, 'q_rope_golden.bin'))
    kv_cache_out.tofile(Path(output_dir, 'kv_cache_golden.bin'))
    kr_cache_out.tofile(Path(output_dir, 'kr_cache_golden.bin'))
    # q_a_rms_norm.tofile(Path(output_dir, 'rms_res.bin'))

    # reshape
    kv_nope_cache = kv_cache_out.reshape([block_num * block_size, n2 * kv_lora_rank])
    k_rope_cache = kr_cache_out.reshape([block_num * block_size, n2 * rope_dim])

    q_bsnd = np.concatenate([q_out, q_rope_out], axis=-1)  # [b, s, n1, kv_lora_rank + rope_dim]
    concat_parms = [b, s, n2, kv_lora_rank, rope_dim, block_size]
    k_cache_bsnd, v_cache_bsnd = kv_cache_concat_bsnd(concat_parms, kr_cache_out, kv_cache_out, block_table, block_num, kv_cache_actual_seq, dtype)

    return x, q_a_rms_norm, cos, sin, kv_len, block_table, q_bsnd, k_cache_bsnd, v_cache_bsnd


def indexer_golden(input_data_map, params, output_dir):
    _, topk_res, tmp_out = indexer_topk_compute(input_data_map, params)
    print(f'tmp_out: {tmp_out}')
    # dump golden for compare res
    topk_res_path = Path(output_dir, "topk_res.bin")
    dump_file(topk_res.numpy(), topk_res_path, np.int32)
    tmp_out_path = Path(output_dir, "tmp_out.bin")
    dump_file(tmp_out.numpy(), tmp_out_path, np.float32)

    return topk_res


def gather_golden(k_cache, topk_indices, gather_params, output_dir):
    """ gather subgraph"""
    nsaR2GatherRes = nsaR2Gather(k_cache, topk_indices, gather_params)
    topk_indices.numpy().tofile(Path(output_dir, 'topk_2048.bin'))
    nsaR2GatherRes.tofile(Path(output_dir, 'nsaR2GatherRes.bin'))

    shape = nsaR2GatherRes.shape
    left = nsaR2GatherRes.reshape(shape[0]*shape[1]*shape[2],shape[3])
    right = np.random.uniform(-0.1, 0.1, (shape[3], 16)).astype(bfloat16)
    matmulRes = np.matmul(left, right)
    print(f'left {left.shape} right {right.shape} matmulRes {matmulRes.shape}')
    right.tofile(Path(output_dir, 'right.bin'))
    matmulRes.astype(bfloat16).tofile(Path(output_dir, 'matmulRes.bin'))

    return nsaR2GatherRes


def slc_attn_golden(params, q_bsnd, k_bsnd, dtype, output_dir):
    b = params.get("b")
    s = params.get("s")
    n1 = params.get("n1")
    kv_lora_rank = params.get("dn")
    actual_seq = params.get("kv_cache_actual_seq")
    scalar = params.get("softmax_scalar")
    topk = params.get("topk")

    atten_out_shape = [b, s, n1, kv_lora_rank]
    v_bsnd = k_bsnd[:, :, :, :kv_lora_rank]
    atten_out = compute_attention(q_bsnd, k_bsnd, v_bsnd, actual_seq, scalar, topk, atten_out_shape)

    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_rope_path = Path(output_dir, 'q_rope.bin')
    k_slc_path = Path(output_dir, 'k_slc.bin')
    v_slc_path = Path(output_dir, 'v_slc.bin')
    # actual_seq_path = Path(output_dir, 'actual_seq.bin')
    atten_out_path = Path(output_dir, 'atten_out.bin')

    # dump golden file
    dump_file_torch(q_nope, q_nope_path, dtype)
    dump_file_torch(q_rope, q_rope_path, dtype)
    dump_file_torch(k_bsnd, k_slc_path, dtype)
    dump_file_torch(v_bsnd, v_slc_path, dtype)
    # dump_file_torch(actual_seq, actual_seq_path, torch.int32)
    dump_file_torch(atten_out, atten_out_path, dtype)

    return atten_out


def gen_deepseek_dsa_golden(params, dtypes, output_dir: Path, is_nz=False):
    print("=========== start gen deepseek indexer attention golden =============:")

    dtype, w_dtype = dtypes
    logging.debug(f"gen_deepseek_dsa_golden  dtype:{dtype}, w_dtype:{w_dtype}")
    b = params.get("b")
    s = params.get("s")
    s1 = params.get("s")
    s2 = params.get("s2")
    h = params.get("h")
    n1 = params.get("n1")
    n2 = params.get("n2")
    q_dim = params.get("q_dim")
    k_dim = params.get("k_dim")
    v_dim = params.get("v_dim")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    block_size = params.get("block_size")
    epsilon = params.get("epsilon")
    cache_mode = params.get("cache_mode")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    v_head_dim = params.get("v_head_dim")
    is_quant = params.get("is_quant")
    has_smooth = params.get("is_smooth")

    # indexer参数
    idx_n_heads = params.get("idx_n_heads")
    idx_head_dim = params.get("idx_head_dim")
    topk = params.get("topk")

    # kv cache actual_seq
    kv_cache_actual_seq_p = params.get("kv_cache_actual_seq")
    if isinstance(kv_cache_actual_seq_p, int):
        kv_cache_actual_seq = [kv_cache_actual_seq_p] * b
    elif isinstance(kv_cache_actual_seq_p, list):
        if len(kv_cache_actual_seq_p) == b:
            kv_cache_actual_seq = kv_cache_actual_seq_p
        else:
            raise RuntimeError("unsupported this kv_cache_actual_seq")
    else:
        raise RuntimeError("unsupported kv_cache_actual_seq data type")
    params["kv_cache_actual_seq"] = kv_cache_actual_seq
    print("cur actual seq: ", kv_cache_actual_seq)

    softmax_scale = q_dim ** -0.5
    block_num = sum([(s + block_size - 1) // block_size for s in kv_cache_actual_seq])
    max_block_num = (max(kv_cache_actual_seq) + block_size - 1) // block_size

    # np.random.seed(int(time.time()))
    np.random.seed(42)

    # 1. 生成原始输入的data
    # _, idx_block_table_list = gen_block_table(b, kv_cache_actual_seq, block_size)
    # idx_block_table = torch.tensor(idx_block_table_list, dtype=torch.int32)
    act_seq_tensor = torch.tensor(kv_cache_actual_seq, dtype=torch.int32)
    # index_q_weights
    wq_b = torch.empty((q_lora_rank, idx_n_heads * idx_head_dim), dtype=dtype).uniform_(-1, 1)
    wq_b_nz = wq_b.reshape(q_lora_rank, idx_n_heads * idx_head_dim // 16, 16).permute(1, 0, 2)
    dump_file_torch(wq_b, Path(output_dir, "wq_b.bin"), dtype)
    dump_file_torch(wq_b_nz, Path(output_dir, "wq_b_nz.bin"), dtype)

    # index_k_weights
    wk = torch.empty((h, idx_head_dim), dtype=dtype).uniform_(-1, 1)
    wk_nz = wk.reshape(h, idx_head_dim // 16, 16).permute(1, 0, 2)
    dump_file_torch(wk, Path(output_dir, "wk.bin"), dtype)
    dump_file_torch(wk_nz, Path(output_dir, "wk_nz.bin"), dtype)

    # proj_weights
    weight_proj = torch.empty((h, idx_n_heads), dtype=dtype).uniform_(-1, 1)
    weight_proj_nz = weight_proj.reshape(h, idx_n_heads // 16, 16).permute(1, 0, 2)
    dump_file_torch(weight_proj, Path(output_dir, "weights_proj.bin"), dtype)
    dump_file_torch(weight_proj_nz, Path(output_dir, "weights_proj_nz.bin"), dtype)

    # weights_layer_norm, bias_layer_norm
    weight_ln = torch.ones((idx_head_dim), dtype=dtype)
    bias_ln = torch.zeros((idx_head_dim), dtype=dtype)
    dump_file_torch(weight_ln, Path(output_dir, "weight_layer_norm.bin"), dtype)
    dump_file_torch(bias_ln, Path(output_dir, "bias_layer_norm.bin"), dtype)

    # 2. 计算 & dump file
    # mla-prolog 子图
    print("============ mla prolog ==================")
    numpy_dtype = bfloat16 if dtype == torch.bfloat16 else np.float16
    x, q_a_rms_norm, cos, sin, k_cache_index, block_table, q_bsnd, k_cache_bsnd, v_cache_bsnd = mla_prolog_golden(params, is_nz, numpy_dtype, output_dir)
    dump_file_torch(q_a_rms_norm, Path(output_dir, "q_a_rms_norm.bin"), dtype)
    dump_file_torch(cos, Path(output_dir, "cos_idx_rope.bin"), dtype)
    dump_file_torch(sin, Path(output_dir, "sin_idx_rope.bin"), dtype)
    dump_file_torch(k_cache_index, Path(output_dir, "k_cache_index.bin"), torch.int32)


    # Lightning Indexer prolog子图
    print("============ Lightning Indexer prolog ==================")
    idx_k_cache_bsnd = torch.empty((b, s2, n2, idx_head_dim), dtype=dtype).uniform_(-1, 1)
    idx_k_cache = gen_lightning_indexer_prolog.gen_cache_tensor(idx_k_cache_bsnd, block_table, block_num, block_size)
    dump_file_torch(idx_k_cache, Path(output_dir, "idx_k_cache.bin"), dtype)

    indexer_prolog_inputs = {
        "token_x": torch.tensor(x.astype(np.float32), dtype=torch.bfloat16),
        "qr": torch.tensor(q_a_rms_norm.astype(np.float32), dtype=torch.bfloat16),
        "wq_b": wq_b,
        "wq_b_nz": wq_b_nz,
        "wk": wk,
        "wk_nz": wk_nz,
        "weights_proj": weight_proj,
        "weights_proj_nz": weight_proj_nz,
        "weight_layer_norm": weight_ln,
        "bias_layer_norm": bias_ln,
        "cos_idx_rope": torch.tensor(cos.astype(np.float32), dtype=torch.bfloat16),
        "sin_idx_rope": torch.tensor(sin.astype(np.float32), dtype=torch.bfloat16),
        "idx_k_cache": idx_k_cache,
        "idx_k_cache_index": k_cache_index,
        "idx_block_table": block_table,
    }
    indexer_prolog_params = {
        "s2": s2,
        "b": b,
        "seq": s1,
        "dim": h,
        "q_lora_rank": q_lora_rank,
        "idx_head_dim": idx_head_dim,
        "idx_n_heads": idx_n_heads,
        "rope_head_dim": rope_dim,
        "block_size": block_size,
        "block_num": block_num,
        "n_kv": n2
    }
    # params_tensor = torch.tensor(list(indexer_prolog_params.values()), dtype=torch.int32)
    # dump_file_torch(params_tensor, Path(output_dir, "input_param.bin"), torch.int32)

    indexer_outputs = indexer_prolog(indexer_prolog_inputs, indexer_prolog_params)
    indexer_query = indexer_outputs["query"]
    indexer_key = indexer_outputs["idx_k_cache_out"]
    weights = indexer_outputs["weights"] # shape is [b, s1, idx_n_heads]
    dump_file_torch(indexer_query, Path(output_dir, "query.bin"), dtype)
    dump_file_torch(indexer_key, Path(output_dir, "key.bin"), dtype)
    dump_file_torch(weights, Path(output_dir, "weights.bin"), dtype)


    # indexer子图
    print("============ Indexer ==================")
    indexer_input_data_map = {
        "query": indexer_query,
        "key": indexer_key,
        "weights": weights,
        "act_seq": act_seq_tensor,
        "block_table": block_table
    }
    n1_scale = 1.0 / np.sqrt(idx_n_heads)
    idx_softmax_scale = 1.0 / np.sqrt(idx_head_dim)
    indexer_params = {
        "b": b,
        "s1": s1,
        "n1": idx_n_heads,
        "d": idx_head_dim,
        "dtype": dtype,
        "s2": s2,
        "n2": n2,
        "act_seq": kv_cache_actual_seq,
        "block_size": block_size,
        "block_num": block_num,
        "max_block_num": max_block_num,
        "sparse_count": topk,
        "score_scale": n1_scale * idx_softmax_scale
    }
    topk_res = indexer_golden(indexer_input_data_map, indexer_params, output_dir)


    # gather 子图
    print("============ gather ==================")
    gather_params = {
        "b": b,
        "s": s,
        "n2": n2,
        "dn": kv_lora_rank,
        "dr": rope_dim,
        "act_seq": kv_cache_actual_seq
    }
    # nsaR2TopkRes = genNsaR2TopkRes(gather_params) # TODO: Indexer子图的输出
    kv_slc_cache_out = gather_golden(k_cache_bsnd, topk_res, gather_params, output_dir)

    # slc_attn子图
    print("============ slc_attn ==================")
    slc_attn_params = {
        "b": b,
        "s": s,
        "n1": n1,
        "dn": kv_lora_rank,
        "kv_cache_actual_seq": kv_cache_actual_seq,
        "softmax_scalar": softmax_scale,
        "topk": topk
    }
    q_bsnd_torch = torch.tensor(q_bsnd.astype(np.float32), dtype=torch.float32)
    kv_slc_cache_out_torch = torch.tensor(kv_slc_cache_out.astype(np.float32), dtype=torch.float32)
    slc_attn_golden(slc_attn_params, q_bsnd_torch, kv_slc_cache_out_torch, dtype, output_dir)

    return True


def deepseek_dsa_entry(dtypes, bs1s2h, quant_smooth, actual_seq, output_dir: Path):
    b, s1, s2, h = bs1s2h
    is_quant, is_smooth = quant_smooth
    kv_lora_rank = 512
    rope_dim = 64
    q_dim = kv_lora_rank + rope_dim
    k_dim = kv_lora_rank + rope_dim
    v_dim = kv_lora_rank
    v_head_dim = 128
    epsilon = 1e-5
    cache_mode = "PA_BSND"

    # index 参数
    idx_n_heads = 64
    idx_head_dim = 128
    topk = 2048

    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "n1": 128,
        "n2": 1,
        "h": h,
        "q_lora_rank": 1536,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rope_dim": rope_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "topk": topk,
        "block_size": 128,
        "kv_cache_actual_seq": actual_seq,
        "epsilon": epsilon,
        "cache_mode": cache_mode,
        "v_head_dim": v_head_dim,
        "idx_n_heads": idx_n_heads,
        "idx_head_dim": idx_head_dim,
        "is_quant": is_quant,
        "is_smooth": is_smooth
    }
    gen_deepseek_dsa_golden(params, dtypes, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = [params.get("b"), params.get("s"), params.get("s2"), params.get("n1"), params.get("n2")]
    input_params.append(1 if is_quant else 0)
    input_params.append(1 if is_smooth else 0)
    dump_file(input_params, Path(output_dir, 'input_params.bin'), np.int32)


# @GoldenRegister.reg_golden_func(
#     case_names=[
#         "DecodeSparseAttentionSTest.mini",
#         "DecodeSparseAttentionSTest.32B",
#         "DecodeSparseAttentionSTest.24B",
#         "DecodeSparseAttentionSTest.48B",
#         "DecodeSparseAttentionSTest.testGatherLogical_bfloat16",
#     ],
#     version=0,
#     timeout=0
# )
def gen_deepseek_dsa_func(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'x.bin')
    complete = input_path.exists()
    complete = False
    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DecodeSparseAttentionSTest.mini": # TODO: 用例名字
            b, s1, s2 = 4, 1, 128 * 1024
            kv_act_seq = [768, 4097, 8192, 131071]
            deepseek_dsa_entry((torch.bfloat16, torch.bfloat16), (b, s1, s2, 7168), (False, False), kv_act_seq, output)
        elif case_name == "DecodeSparseAttentionSTest.testGatherLogical_bfloat16": # TODO: 用例名字
            b, s2 = 4, 128 * 1024
            kv_act_seq = [768, 4097, 8192, 131071]
            deepseek_dsa_entry((torch.bfloat16, torch.bfloat16), (b, 2, s2, 7168), (False, False), kv_act_seq, output)
        elif case_name == "DecodeSparseAttentionSTest.32B": # TODO: 用例名字
            b, s1, s2 = 32, 1, 8192
            kv_act_seq = [s2] * b
            deepseek_dsa_entry((torch.bfloat16, torch.bfloat16), (b, s1, s2, 7168), (False, False), kv_act_seq, output)
        elif case_name == "DecodeSparseAttentionSTest.24B": # TODO: 用例名字
            b, s1, s2 = 24, 1, 4096
            kv_act_seq = [s2] * b
            deepseek_dsa_entry((torch.bfloat16, torch.bfloat16), (b, s1, s2, 7168), (False, False), kv_act_seq, output)
        elif case_name == "DecodeSparseAttentionSTest.48B": # TODO: 用例名字
            b, s1, s2 = 48, 1, 8192
            kv_act_seq = [4096] * b
            deepseek_dsa_entry((torch.bfloat16, torch.bfloat16), (b, s1, s2, 7168), (False, False), kv_act_seq, output)
        else:
            logging.error("Can't get func to gen golden, Case(%s)", case_name)
            return False
    return True


def attention_entry(cs, base) -> bool:
    # 函数调用
    ret: bool = True
    g_src_root = base
    output_dir: Path = Path(g_src_root, "", cs).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ret = gen_deepseek_dsa_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if attention_entry() else 1)
