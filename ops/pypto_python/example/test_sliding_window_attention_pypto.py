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
"""Test module for sliding_window_attention."""

import math
import os
import sys
import torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, 'impl')
sys.path.insert(0, impl_path)
from sliding_window_attention_pypto import npu_sliding_window_attention, get_mask


class SWA(torch.nn.Module):
    def forward(self, q, ori_block_table, ori_kv, seqused_kv, sinks, win_size, \
        mask, cu_seqlens_q):
        return torch.ops.pypto.sliding_window_attention(q, ori_block_table, ori_kv, seqused_kv, \
            sinks, win_size, mask, cu_seqlens_q)


def gen_uniform_data(data_shape, min_value, max_value, dtypes, device_id):
    if isinstance(data_shape, list):
        data_shape = tuple(data_shape)
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtypes, device=f'npu:{device_id}')
    if dtypes == torch.bool:
        return torch.randint(0, 2, size=data_shape, dtype=torch.bool, device=f'npu:{device_id}')
    return torch.rand(data_shape, dtype=dtypes, device=f'npu:{device_id}').uniform_(min_value, max_value)


def gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, block_size, seqused_kv_list, dtypes, device_id):
    torch.manual_seed(42)
    b = len(seqused_kv_list)
    new_dtype = dtypes
    actual_seq_max = max(seqused_kv_list)
    s_kv_max = actual_seq_max
    shape_q = [t * n_q, d_q]
    shape_kv = [b, s_kv_max, n_kv, d_kv]
    atten_out_shape = [t, n_q, d_kv]
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0

    sinks = gen_uniform_data([n_q], -1, 1, torch.float32, device_id)

    q = gen_uniform_data(shape_q, -1, 1, new_dtype, device_id)
    q_tnd = q.reshape(t, n_q, d_q)
    kv_bsnd = gen_uniform_data(shape_kv, -1, 1, new_dtype, device_id)

    for actual_seq in seqused_kv_list:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    block_table_shape = [b, math.ceil(s_kv_max / block_size)]
    block_num = block_num_min
    block_idx_list = torch.randperm(block_num, dtype=torch.int32)
    block_idx = 0
    
    block_table = [-1] * block_table_shape[1]

    block_table = torch.tile(torch.tensor(block_table, device=f'npu:{device_id}').\
        to(torch.int32), (block_table_shape[0], 1))
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    kv_cache = torch.zeros([block_num, block_size, n_kv, d_kv], dtype=new_dtype, device=f'npu:{device_id}')
    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                kv_valid = kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]
                kv_cache[kv_cache_blk_id, 0: kv_valid.shape[0], :, :] = \
                    kv_bsnd[b_idx, block_offset:(block_offset + block_size), :, :]

    atten_out = torch.zeros(atten_out_shape, dtype=new_dtype, device=f'npu:{device_id}')
    return q_tnd, block_table, kv_cache, sinks, atten_out


def win_atten_calc_tnd(input_params_win_attn, seqused_kv_list, sinks, q_tnd, \
    kv_cache, block_table, cu_seqlens_q, device_id):

    t = input_params_win_attn[0]
    n_q = input_params_win_attn[2]
    d_q = input_params_win_attn[3]
    win = input_params_win_attn[4]
    scalar = input_params_win_attn[5]
    atten_out_shape = [t, n_q, d_q]
    atten_out = torch.zeros(atten_out_shape, dtype=torch.bfloat16, device=f'npu:{device_id}')
    block_size = kv_cache.shape[1]
    b = len(seqused_kv_list)

    for b_index in range(b):
        cur_s_q = cu_seqlens_q[b_index + 1] - cu_seqlens_q[b_index]

        for s1_index in range(cur_s_q):

            t_index = cu_seqlens_q[b_index] + s1_index

            actual_seq = seqused_kv_list[b_index]
            q_tensor_cur = q_tnd[t_index:(t_index + 1), :, :].reshape(n_q, d_q)

            cur_loc = actual_seq - cur_s_q + s1_index + 1
            valid_len = min(cur_loc, win)
            cur_start_pos = cur_loc - valid_len
            end_pos = cur_loc
            start_block = cur_start_pos // block_size
            start_offset = cur_start_pos % block_size
            end_block = (end_pos - 1) // block_size

            kv_list = []
            for block_idx in range(start_block, end_block + 1):
                physical_block_id = block_table[b_index, block_idx]
                kv_block = kv_cache[physical_block_id, :, 0, :]
                kv_list.append(kv_block)
            
            kv_cur = torch.cat(kv_list, axis=0)
            kv_cur = kv_cur[start_offset : start_offset + valid_len, :]

            sum_exp = torch.zeros([n_q, 1], dtype=torch.float32, device=f'npu:{device_id}')
            acc_s = torch.matmul(q_tensor_cur.to(torch.float32), kv_cur.to(torch.float32).transpose(1, 0))
            acc_s = acc_s * scalar
            scores_max = torch.max(acc_s, dim=-1, keepdims=True)[0]
            acc_s = torch.exp(acc_s - scores_max)
            sum_exp = torch.sum(acc_s, dim=-1, keepdims=True)
            sum_exp += torch.exp(sinks.reshape(n_q, 1) - scores_max)
            v1_res = acc_s / sum_exp
            v1_res = v1_res.to(torch.bfloat16)
            mm2_res = torch.matmul(v1_res, kv_cur)

            atten_out[t_index:(t_index + 1), :, :] = mm2_res

    return atten_out


def test_win_atten_tnd_mask(allow_in_graph=False) -> None:

    for b in [64]:
        s_val = 2
        cu_seqlens_q = [i * s_val for i in range(b + 1)]
        t = cu_seqlens_q[-1]
        win_size = 128
        n_q = 64
        block_size = 128
        n_kv = 1
        dtypes = torch.bfloat16
        head_dim = 512
        d_q = head_dim
        d_kv = head_dim
        scalar = d_q ** -0.5
        input_params_win_attn = [t, n_kv, n_q, d_q, win_size, scalar]

        device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        torch.npu.set_device(device_id)

        seqused_kv_list = [8192] * b
        seqused_kv_list_tensor = torch.tensor(seqused_kv_list, dtype=torch.int32, device=f'npu:{device_id}')
        cu_seqlens_q_tenor = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=f'npu:{device_id}')

        q_tnd, block_table, kv_cache, sinks, _ = gen_win_attn_data_tnd(t, n_q, d_q, n_kv, d_kv, \
            block_size, seqused_kv_list, dtypes, device_id)
        mask2 = get_mask(4, n_q, device_id, block_size)

        if allow_in_graph:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
            compiler_config = CompilerConfig()
            compiler_config.mode = "reduce-overhead"
            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
            model = torch.compile(SWA(), dynamic=False, fullgraph=True, backend=npu_backend)

            q_npu = q_tnd.npu()
            ori_block_table_npu = block_table.npu()
            ori_kv_npu = kv_cache.npu()
            seqused_kv_list_tensor_npu = seqused_kv_list_tensor.npu()
            attn_sinks_npu = sinks.npu()
            mask2_npu = mask2.npu()
            cu_seqlens_q_tenor_npu = cu_seqlens_q_tenor.npu()

            atten_out = model(q_npu, ori_block_table_npu, ori_kv_npu, seqused_kv_list_tensor_npu, \
                attn_sinks_npu, win_size, mask2_npu, cu_seqlens_q_tenor_npu)

        else:
            atten_out = npu_sliding_window_attention(q_tnd, block_table, kv_cache, seqused_kv_list_tensor, \
                sinks, win_size, mask=mask2, cu_seqlens_q=cu_seqlens_q_tenor)

        golden = win_atten_calc_tnd(input_params_win_attn, seqused_kv_list, sinks, q_tnd, \
            kv_cache, block_table, cu_seqlens_q, device_id)
        from compare import compare
        compare(golden, atten_out, "SWA tnd mask version", rtol=0.0078125, atol=0.0001)


if __name__ == "__main__":
    test_win_atten_tnd_mask(False)
