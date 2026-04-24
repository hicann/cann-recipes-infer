#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test module for sparse_compressed_flash_attention."""


from dataclasses import dataclass
import math
import os
import torch
import torch_npu
import pypto
import logging
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, 'impl')
sys.path.insert(0, impl_path)
from sparse_compress_flash_attention_pypto import *
from compare import compare


class CompressSFA(torch.nn.Module):
    def forward(self, query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu, cmp_block_table_npu, atten_sink_npu,
                                    seqused_kv_npu, cmp_sparse_indices_npu, softmax_scale, win_size, cmp_ratio):
        return sparse_compress_flash_attention_pypto(query_npu, q_act_seqs_npu, ori_kv_npu, cmp_kv_npu, ori_block_table_npu, cmp_block_table_npu, atten_sink_npu,
                                    seqused_kv_npu, cmp_sparse_indices_npu, softmax_scale, win_size, cmp_ratio)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def compute_attention_no_flash(input_data, params, s2_tile):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    no flash 版本
    """
    q, compress_kv, origin_kv, topk_indices, block_table, actual_seq_q, actual_seq, origin_block_table, origin_actual_seq, atten_sink = input_data
    block_size, scalar, topk, d_v, win_size = params

    # 提取维度信息
    t, n1, d = q.shape
    t = actual_seq_q[-1]
    b = len(actual_seq)

    if topk_indices.ndim > 2:
        topk_indices = topk_indices.reshape(t, topk)

    atten_out_shape = [t, n1, d_v]
    input_dtype = q.dtype
    kv_dtype = compress_kv.dtype

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=input_dtype)

    atten_sink_2d = atten_sink.unsqueeze(-1)

    for b_idx in range(b):
        cur_k_seq = actual_seq[b_idx]
        origin_cur_k_seq = origin_actual_seq[b_idx]
        s1 = actual_seq_q[b_idx + 1] - actual_seq_q[b_idx]

        for s1_idx in range(s1):

            t_idx = actual_seq_q[b_idx] + s1_idx

            # win kv_cache
            cur_len = max(origin_cur_k_seq - s1 + 1 + s1_idx, 0)
            origin_cur_win_size = min(cur_len, win_size)
            valid_start_pos = cur_len - origin_cur_win_size
            valid_end_pos = cur_len - 1
            start_block = valid_start_pos // block_size
            start_offset = valid_start_pos % block_size
            end_block = valid_end_pos // block_size

            cur_seq = min(max(cur_k_seq - s1 + 1 + s1_idx, 0), topk)

            bn_per_batch = math.ceil(cur_seq / s2_tile)
            for s2_idx in range(bn_per_batch):
                s2_tile_cur = min(s2_tile, cur_seq - s2_idx * s2_tile)
                s2_start = s2_tile * s2_idx
                s2_end = s2_start + s2_tile_cur

                # compress kv_cache
                topk_indices_tmp = topk_indices[t_idx, s2_start:s2_end]
                slc_compress_kv = torch.zeros([s2_tile_cur, d_v], dtype=kv_dtype)
                # compress kvCache的offset
                offset = torch.zeros([s2_tile_cur], dtype=torch.int32)
                for cur_s2_idx in range(s2_tile_cur):
                    topk_index = topk_indices_tmp[cur_s2_idx]
                    block_idx_in_batch = topk_index // block_size
                    slc_block_idx = block_table[b_idx, block_idx_in_batch]
                    tail = topk_index % block_size
                    offset[cur_s2_idx] = slc_block_idx * block_size + tail
                # gather compress kvCache
                for cur_s2_idx in range(s2_tile_cur):
                    slc_idx = offset[cur_s2_idx]
                    slc_compress_kv[cur_s2_idx, :] = compress_kv[slc_idx, :]
                
                # win kv_cache
                kv_list = []
                for block_idx in range(start_block, end_block + 1):
                    physical_block_id = origin_block_table[b_idx, block_idx]
                    kv_block = origin_kv[physical_block_id * block_size: (physical_block_id + 1) * block_size, :]
                    kv_list.append(kv_block)
                kv_cur = torch.cat(kv_list, axis=0)
                win_kv_cache = kv_cur[start_offset : start_offset + origin_cur_win_size, :]

                # 组装新的kv_cache
                kj = torch.zeros([origin_cur_win_size + s2_tile_cur, d], dtype=kv_dtype)
                kj[0 : origin_cur_win_size, :] = win_kv_cache
                kj[origin_cur_win_size : origin_cur_win_size + s2_tile_cur, :] = slc_compress_kv

                # C1
                qi = q[t_idx, :, :].reshape(n1, d) # (n1, dk)
                sij = torch.matmul(qi.to(torch.float32), kj.transpose(1, 0).to(torch.float32)).to(torch.float32)

                sij_scale = sij * scalar # (n1, s2_tile)
                tilda_mij = sij_scale.amax(dim=-1, keepdims=True) # (n1, 1)
                t_sub = sij_scale - tilda_mij # (n1, s2_tile)
                tilda_pij = torch.exp(t_sub) # (n1, s2_tile)
                tilda_lij = tilda_pij.sum(dim=-1, keepdims=True)# (n1, 1)
                # calc attn_sink
                sink_t_sub = atten_sink_2d - tilda_mij # (n1, s2_tile)
                sink_tilda_pij = torch.exp(sink_t_sub) # (n1, s2_tile)
                tilda_lij = tilda_lij + sink_tilda_pij

                tmp_softmax = (tilda_pij / tilda_lij).to(input_dtype)
                atten_out_part = torch.matmul(tmp_softmax.to(torch.float32), kj.to(torch.float32)).to(torch.float32)

            attention_output[t_idx, :, :] = atten_out_part.to(input_dtype)

    return attention_output


def gen_block_table(act_seq, block_size):
    block_num = 0
    block_num_each = []
    b = act_seq.shape[0]
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    block_idx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    return block_num, block_table


def gen_sparse_compress_attention_golden(dtype, bn1n2s1, actual_seq_q, actual_seq, cmp_ratio):
    block_size = 128
    win_size = 128
    torch.manual_seed(42)
    b, n_q, n_kv, _ = bn1n2s1
    kv_lora_rank = 512
    topk = 512
    d_q = kv_lora_rank

    scalar = d_q ** -0.5

    if isinstance(actual_seq, int):
        origin_actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            origin_actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")
    
    # 生成压缩后的seq
    actual_seq = [i // cmp_ratio for i in origin_actual_seq]

    assert isinstance(actual_seq_q, list) and len(actual_seq_q) == b + 1, "actual_seq_q length should be b + 1"

    t = actual_seq_q[-1]
    b = len(actual_seq)

    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq_tmp in actual_seq:
        block_num_per_batch.append(math.ceil(actual_seq_tmp / block_size))
        block_num_min += math.ceil(actual_seq_tmp / block_size)
    block_num = block_num_min

    # 1. 定义shape
    shape_q = [t, n_q, d_q]
    shape_kv = [block_num, block_size, kv_lora_rank]
    shape_atten_sink = [n_q]

    max_kv_seq = max(actual_seq)
    block_num, block_table = gen_block_table(torch.tensor(actual_seq), block_size)
    origin_block_num, origin_block_table = gen_block_table(torch.tensor(origin_actual_seq), block_size)
    topk_indices = torch.zeros(t, n_kv * topk).to(torch.int32)
    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))

    for b_i in range(b):
        s_q = actual_seq_q[b_i + 1] - actual_seq_q[b_i]
        for s_q_i in range(s_q):
            t_idx = actual_seq_q[b_i] + s_q_i
            if slc_actual_seq[b_i] < topk:
                topk_indices[t_idx, :slc_actual_seq[b_i]] = torch.arange(0, slc_actual_seq[b_i])
            else:
                perm = torch.randperm(slc_actual_seq[b_i])
                topk_indices[t_idx, :] = perm[:topk]

    topk_indices = topk_indices.reshape(t, n_kv * topk)

    q_tnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kv = gen_uniform_data(shape_kv, -1, 1, dtype)

    atten_sink = gen_uniform_data(shape_atten_sink, -1, 1, torch.float32)

    # 2D
    compress_kv = kv.reshape(block_num * block_size, kv_lora_rank)

    shape_origin_kv = [origin_block_num, block_size, kv_lora_rank]
    origin_kv = gen_uniform_data(shape_origin_kv, -1, 1, dtype).reshape(origin_block_num * block_size, kv_lora_rank)

    # 3. 计算attention
    params = [block_size, scalar, topk, kv_lora_rank, win_size]
    input_data = [q_tnd, compress_kv, origin_kv, topk_indices, block_table, actual_seq_q, actual_seq, origin_block_table, origin_actual_seq, atten_sink]

    s2_tile = 512
    atten_out = compute_attention_no_flash(input_data, params, s2_tile)

    # 4.dump 数据
    # data split to [nope + rope]
    q = q_tnd.reshape(t * n_q, kv_lora_rank)
    # input params
    input_params = [b, s_q, n_q, n_kv, max_kv_seq, kv_lora_rank, block_num, block_size, win_size, topk, scalar, cmp_ratio]
    input_data_map = [q, compress_kv, origin_kv, topk_indices, block_table, origin_block_table, actual_seq_q, torch.tensor(origin_actual_seq, dtype=torch.int32), atten_sink]

    return input_params, input_data_map, atten_out


def get_case_config(case_name: str):
    # case参数配置字典，key为case名称，value为对应的参数元组(bn1n2s1, is_kn_quant, actual_seq_q, actual_seq, cmp_ratio)
    test_case_config = {
        "sfa_bf16_b1_s4_seq64K_p": (
            (1, 64, 1, 4), 0, [0, 4], [65536] * 1, 4
        ),
        "sfa_bf16_b1_s256_seq64K_p": (
            (1, 64, 1, 256), 0, [0, 256], [65536] * 1, 4
        ),
        "sfa_bf16_b4_s16_seq64K_p": (
            (4, 64, 1, 16), 0, [0, 16, 32, 48, 49], [65536] * 4, 4
        ),
        "sfa_bf16_b1_s16_seq64K_p": (
            (1, 64, 1, 16), 0, [0, 16], [130] * 1, 4
        ),
        "sfa_bf16_b64_s2_seq8K_d": (
            (64, 64, 1, 2), 0, [i * 2 for i in range(64 + 1)], [8192] * 64, 4
        ),
    }
    case_config = test_case_config.get(case_name)
    return case_config


def do_test_sparse_compress_attention_func(bn1n2s1, actual_seq, input_params, input_data, atten_out):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tile_config = SCFATileShapeConfig(
        g_tile=64,
        c1_tile_shape=[64, 64, 128, 512, 128, 128],
        v1_tile_shape=[32, 640],
        c2_tile_shape=[64, 64, 128, 640, 256, 256]
    )

    _, _, n_q, n_kv, max_kv_seq, kv_lora_rank, block_num, block_size, win_size, topk, scalar,\
        cmp_ratio = input_params
    q, compress_kv, origin_kv, topk_indices, block_table, origin_block_table, act_seq_q, origin_act_seq, atten_sink = input_data
    q_act_seqs = torch.tensor(act_seq_q, dtype=torch.int32)
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    t = act_seq_q[-1]
    calc_attention_out = torch.zeros([t * n_q, kv_lora_rank], dtype=torch.bfloat16)

    # 算子kernel接口入参名称及顺序与算子原型对齐
    q_npu = q.npu()
    query_pto = pypto.from_torch(q_npu, dynamic_axis=[0], name="q_nope")
    q_act_seqs_npu = q_act_seqs.npu()
    actual_seq_q_pto = pypto.from_torch(q_act_seqs_npu, dynamic_axis=[0], name="actual_seq_q")
    compress_kv_npu = compress_kv.npu()
    cmp_kv_pto = pypto.from_torch(compress_kv_npu, dynamic_axis=[0], name="kv")
    origin_kv_npu = origin_kv.npu()
    ori_kv_pto = pypto.from_torch(origin_kv_npu, dynamic_axis=[0], name="origin_kv")
    origin_block_table_npu = origin_block_table.npu()
    ori_block_table_pto = pypto.from_torch(origin_block_table_npu, dynamic_axis=[0], name="origin_block_table")
    topk_indices_npu = topk_indices.npu()
    cmp_sparse_indices_pto = pypto.from_torch(topk_indices_npu, dynamic_axis=[0], name="topk_indices")
    block_table_npu = block_table.npu()
    cmp_block_table_pto = pypto.from_torch(block_table_npu, dynamic_axis=[0], name="block_table")
    kv_act_seqs_npu = kv_act_seqs.npu()
    seqused_kv_pto = pypto.from_torch(kv_act_seqs_npu, dynamic_axis=[0], name="kv_act_seqs")
    atten_sink_npu = atten_sink.npu()
    atten_sink_pto = pypto.from_torch(atten_sink_npu, name="atten_sink")

    calc_attention_out_npu = calc_attention_out.npu()
    attention_out_pto = pypto.from_torch(calc_attention_out_npu, dynamic_axis=[0], name="calc_attention_out")

    pto_inputs = [query_pto, actual_seq_q_pto, ori_kv_pto, cmp_kv_pto, ori_block_table_pto,
                  cmp_block_table_pto, atten_sink_pto, seqused_kv_pto, cmp_sparse_indices_pto]
    pto_outputs = [attention_out_pto]

    print("=============== inputs && outputs =======================")
    print(*[f"{i.name}: {i.ori_shape} {i.dtype}" for i in pto_inputs if i is not None], sep="\n")
    print(*[f"{i.name}: {i.ori_shape} {i.dtype}" for i in pto_outputs], sep="\n")
    print("========================================================\n")

    sparse_compress_flash_attention_kernel(*pto_inputs, *pto_outputs, n_q, n_kv, scalar,
                                           topk, block_size, win_size, cmp_ratio, tile_config)

    pypto.runtime._device_synchronize()
    print("======================sfa compare====================")
    compare(calc_attention_out_npu.cpu(), atten_out.reshape(calc_attention_out.shape), "atten_out", atol=0.0001, rtol=0.0078125, max_error_count=100)


#acl graph测试入口
def do_test_sparse_compress_attention_func_acl_graph(bn1n2s1, actual_seq, input_params, input_data, atten_out):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    b, _, n_q, n_kv, max_kv_seq, kv_lora_rank, block_num, block_size, win_size, topk, \
        softmax_scale, cmp_ratio = input_params
    q, compress_kv, origin_kv, topk_indices, block_table, origin_block_table, act_seq_q, origin_act_seq, atten_sink = input_data
    q_act_seqs = torch.tensor(act_seq_q, dtype=torch.int32)
    kv_act_seqs = torch.tensor(actual_seq, dtype=torch.int32)

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(CompressSFA(), dynamic=False, fullgraph=True, backend=npu_backend)

    q_npu = q.npu()
    q_act_seqs_npu = q_act_seqs.npu()
    compress_kv_npu = compress_kv.npu()
    origin_kv_npu = origin_kv.npu()
    origin_block_table_npu = origin_block_table.npu()
    topk_indices_npu = topk_indices.npu()
    block_table_npu = block_table.npu()
    kv_act_seqs_npu = kv_act_seqs.npu()
    atten_sink_npu = atten_sink.npu()

    attention_out = model(q_npu, q_act_seqs_npu, origin_kv_npu, compress_kv_npu, origin_block_table_npu, block_table_npu, atten_sink_npu,
        kv_act_seqs_npu, topk_indices_npu, softmax_scale, win_size, cmp_ratio)
    pypto.runtime._device_synchronize()

    compare(attention_out.cpu(), atten_out.reshape(attention_out.shape), "atten_out", atol=0.0001, rtol=0.005, max_error_count=100)


def do_test_sfa_entry(case_name: str, is_acl_graph: bool = False):
    case_config = get_case_config(case_name)
    if not case_config:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    bn1n2s1, is_kn_quant, actual_seq_q, actual_seq, cmp_ratio = case_config

    input_params, input_data, atten_out = gen_sparse_compress_attention_golden(
        torch.bfloat16, bn1n2s1, actual_seq_q, actual_seq, cmp_ratio
    )

    if is_acl_graph:
        print("\n====================== acl_graph ===============================\n")
        do_test_sparse_compress_attention_func_acl_graph(
            bn1n2s1, actual_seq, input_params, input_data, atten_out
        )
    else:
        print("\n====================== st ===============================\n")
        do_test_sparse_compress_attention_func(
            bn1n2s1, actual_seq, input_params, input_data, atten_out
        )
    return True


def test_sfa_bf16_b1_s4_seq64K_acl_graph_p():
    '''
    scfa aclgraph测试用例
    '''
    do_test_sfa_entry("sfa_bf16_b1_s4_seq64K_p", is_acl_graph=True)


def test_sfa_bf16_b1_s256_seq64K_p():
    '''
    scfa prefill测试用例
    '''
    do_test_sfa_entry("sfa_bf16_b1_s256_seq64K_p")


def test_sfa_bf16_b4_s16_seq64K_p():
    '''
    scfa prefill测试用例, 验证多batch mtp
    '''
    do_test_sfa_entry("sfa_bf16_b4_s16_seq64K_p")


def test_sfa_bf16_b1_s16_seq64K_p():
    '''
    scfa prefill测试用例, 验证小seq
    '''
    do_test_sfa_entry("sfa_bf16_b1_s16_seq64K_p")


def test_sfa_bf16_b64_s2_seq8K_d():
    '''
    scfa decode, mtp 1
    '''
    do_test_sfa_entry("sfa_bf16_b64_s2_seq8K_d")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_sfa_bf16_b1_s256_seq64K_p()
    test_sfa_bf16_b64_s2_seq8K_d()
