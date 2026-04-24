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
"""Test module for compressed_flash_attention."""
import os
import sys
from dataclasses import dataclass
import torch
import pypto
import numpy as np
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, 'impl')
sys.path.insert(0, impl_path)

from compressed_flash_attention_pypto import npu_compress_flash_attention

np.random.seed(0)
torch.manual_seed(0)
np.set_printoptions(formatter={'float': '{:.6f}'.format})


@dataclass
class AttentionConfig:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    q_d: int
    kv_d: int
    block_size: int = 128
    cmp_ratio: int = 128
    max_blocks: int = 0
    actual_seq: torch.Tensor = None  # Changed to torch.Tensor type
    block_table_batch: int = 0
    kv_num_blocks: int = 0
    

def gen_block_table(actual_seq_len, block_size, block_table_shape, cmp_ratio=128, enable_win=False):
    block_num_per_batch = []
    block_num = 0
    
    if enable_win:
        cmp_ratio = 1
    # Process torch tensor type actual_seq_len
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq.item() // cmp_ratio / block_size))
        block_num += math.ceil(actual_seq.item() / block_size)

    # Use torch to replace numpy
    block_idx_list = torch.arange(0, block_num, dtype=torch.int32)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))]  # Random permutation

    # Create cmp_block_table tensor
    cmp_block_table = torch.full(
        block_table_shape, -1, dtype=torch.int32, device=actual_seq_len.device
    )
    block_idx = 0
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            cmp_block_table[block_table_batch_idx][j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_batch_idx += 1
    return cmp_block_table


def get_decode_case(device="cpu"):
    b = 64
    s1 = 2
    s2 = 8 * 1024
    q_d = 512
    nq = 64
    nkv = 1
    block_table_batch = b
    block_size = 128
    cmp_ratio = 128
    kv_num_blocks = b * ((s2 + block_size - 1) // block_size)
    actual_seq_values = [s2] * b
    actual_seq_tensor = torch.tensor(actual_seq_values, dtype=torch.int32, device=device)
    attn_cfg = AttentionConfig(b=b, s1=s1, s2=s2, n1=nq, n2=nkv,
                               q_d=q_d, kv_d=q_d, block_size=block_size, block_table_batch=block_table_batch,
                               kv_num_blocks=kv_num_blocks, actual_seq=actual_seq_tensor, cmp_ratio=cmp_ratio)
    attn_cfg.max_blocks = (s2 + block_size - 1) // block_size
    return attn_cfg


class MM(torch.nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        cmp_kv: torch.Tensor,
        sinks: torch.Tensor,
        cmp_block_table: torch.Tensor,
        seqused_kv: torch.Tensor,
        ori_kv: torch.Tensor,
        ori_block_table: torch.Tensor,
        cmp_ratio: int = 1,
    ):
        return torch.ops.pypto.compress_flash_attention(
            q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv,
            ori_block_table, cmp_ratio
        )


def softmax(x, sinks, is_fp16=False, is_new_sink=False):
    # Use torch's softmax implementation
    if is_fp16:
        original_dtype = x.dtype
        x = x.float()
    x_max = x.max(dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    if sinks is not None:
        if not is_new_sink:
            x_sum += sinks.unsqueeze(-1)
        else:
            x_sum += torch.exp(sinks.unsqueeze(-1) - x_max)
    ans = y / x_sum
    if is_fp16:
        ans = ans.to(original_dtype)
        x_max = x_max.to(original_dtype)
        x_sum = x_sum.to(original_dtype)

    return ans, x_max, x_sum


def matmul_proxy(left, right):
    fp32 = torch.float32
    return torch.matmul(left.to(fp32), right.to(fp32)).to(fp32)


def get_block_kv(kv_2d, cmp_block_table, b_idx, s2_idx, block_size, cur_seq):
    block_idx = cmp_block_table[b_idx][s2_idx]
    block_idx_valid = max(block_idx, 0)
    actual_s2_tile = min(block_size, cur_seq - s2_idx * block_size)
    kj_start = block_idx_valid * block_size
    kj_end = kj_start + actual_s2_tile
    kvj = kv_2d[kj_start:kj_end, :]
    return kvj


def flash_end(out, sinks, li_upd, mi_upd, oi_upd, n2g_ofs, g_tile, bs_ofs, dtype, is_new_sink=False):
    li = li_upd.unsqueeze(-1)
    if sinks is not None:
        if not is_new_sink:
            li += sinks.unsqueeze(-1)
        else:
            li += torch.exp(sinks - mi_upd).unsqueeze(-1)
    oi_final = oi_upd / li
    oi_upd_3d = oi_final.unsqueeze(0)
    attn_out_start = n2g_ofs
    attn_out_end = n2g_ofs + g_tile
    if attn_out_end > out.shape[1]:
        attn_out_end = out.shape[1]
        attn_out_start = attn_out_end - g_tile
    out[bs_ofs : bs_ofs + 1, attn_out_start:attn_out_end, :] = (
        oi_upd_3d.to(dtype)
    )


def kv_cache_concat_bsnd(kv_cache_out, cmp_block_table, actual_seqs):
    b = actual_seqs.shape[0]
    n2 = kv_cache_out.shape[2]
    d = kv_cache_out.shape[3]
    block_size = kv_cache_out.shape[1]
    dtype = kv_cache_out.dtype

    # Process torch tensor type kv_cache_actual_seq
    kv_max = (torch.max(actual_seqs).item() + block_size - 1) // block_size * block_size

    # Use torch to create tensor, keep on same device
    cmp_kv = torch.zeros([b, kv_max, n2, d], dtype=dtype).to(kv_cache_out.device)

    for b_idx in range(b):
        block_list = cmp_block_table[b_idx]
        kv_nope_temp_tensor = torch.zeros([1, kv_max, n2, d], dtype=dtype)
        s_idx = 0

        for _, block_idx in enumerate(block_list):
            if block_idx == -1:
                break
            # Use torch slice operation
            start_idx = s_idx * block_size
            end_idx = (s_idx + 1) * block_size

            kv_nope_temp_tensor[:, start_idx:end_idx, :, :] = kv_cache_out[
                block_idx : block_idx + 1, :, :, :
            ]
            s_idx += 1

        cmp_kv[b_idx : b_idx + 1, :, :, :] = kv_nope_temp_tensor

    return cmp_kv


def cfa_flash_torch(q, cmp_kv, sinks, cmp_block_table, seqused_kv, out, cmp_ratio=128, is_new_sink=False,
                    ori_kv=None, ori_block_table=None):
    """
    Args:
        q: Query [batch_size * s1, num_head, head_size]
        k: Key cache [num_blocks, block_size, kv_head_num, head_size]
        v: Value cache [num_blocks, block_size, kv_head_num, head_size]
        cmp_block_table: Block mapping table for compress cmp_kv cache [batch_size, max_num_blocks_per_query]
        start_pos: Actual start position [batch_size], satisify start_pos + s1 = original actual seq
        out: Output [batch_size * s1, num_head, head_size]
    """
    fp32 = torch.float32
    q_shape = q.shape
    device = q.device
    dtype = q.dtype
    bs1, n1, d = q_shape[0], q_shape[1], q_shape[2]
    b = seqused_kv.shape[0]
    s1 = bs1 // b
    k_shape = cmp_kv.shape
    _, block_size, n2, _ = k_shape
    g = n1 // n2
    g_tile = g
    kv_2d = cmp_kv.reshape(-1, d)
    q_2d = q.reshape(-1, d)
    scale = d ** -0.5
    win = 128

    for b_idx in range(b):
        for s1_idx in range(s1):
            cur_seq = (seqused_kv[b_idx] - (s1 - 1 - s1_idx)) // cmp_ratio
            cur_seq = max(cur_seq, 0)
            s2_loop = math.ceil(cur_seq / block_size)
            for g_idx in range(g // g_tile):
                oi_upd = torch.zeros((g_tile, d), device=device, dtype=fp32)
                li_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                mi_upd = torch.zeros(g_tile, device=device, dtype=fp32)
                bs_ofs = b_idx * s1 + s1_idx
                n2g_ofs = g_idx * g_tile
                qi_start = bs_ofs * n1 + n2g_ofs
                qi_end = qi_start + g_tile
                qi = q_2d[qi_start:qi_end, :]
                if ori_kv is not None and ori_block_table is not None:
                    kv_win_2d = ori_kv.reshape(-1, d)
                    valid_len = seqused_kv[b_idx] - (s1 - s1_idx - 1)
                    valid_win_len = min(valid_len, win)
                    valid_start_pos = valid_len - valid_win_len
                    valid_end_pos = valid_len - 1
                    start_offset = valid_start_pos % block_size

                    start_block = valid_start_pos // block_size
                    end_block = valid_end_pos // block_size
                    kv_list = []

                    for block_idx in range(start_block, end_block + 1):
                        block_idx_valid = max(ori_block_table[b_idx, block_idx], 0)
                        block_offset = block_idx_valid * block_size
                        kv_block = kv_win_2d[block_offset : block_offset + block_size, :]
                        kv_list.append(kv_block)

                    kv_cur = torch.cat(kv_list, axis=0)
                    kv_cur = kv_cur[start_offset : start_offset + valid_win_len, :]

                    mm1 = matmul_proxy(qi, kv_cur.t())
                    muls_res = mm1 * scale
                    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
                    tsub = muls_res - tilda_mij
                    tilda_pij = torch.exp(tsub)
                    tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                    oi_tmp = matmul_proxy(tilda_pij.to(dtype), kv_cur)
                    oi_upd = oi_tmp
                    li_upd = tilda_lij.squeeze(-1)
                    mi_upd = tilda_mij.squeeze(-1)
                    if s2_loop == 0:
                        flash_end(out, sinks, li_upd, mi_upd, oi_upd, n2g_ofs, g_tile, bs_ofs, dtype, \
                                is_new_sink=is_new_sink)
                for s2_idx in range(s2_loop):
                    kvj = get_block_kv(kv_2d, cmp_block_table, b_idx, s2_idx, block_size, cur_seq)
                    mm1 = matmul_proxy(qi, kvj.t())
                    muls_res = mm1 * scale
                    tilda_mij, _ = torch.max(muls_res, dim=-1, keepdim=True)
                    if s2_idx == 0 and ori_kv is None:
                        tsub = muls_res - tilda_mij
                        tilda_pij = torch.exp(tsub)
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                        oi_tmp = matmul_proxy(tilda_pij.to(dtype), kvj)
                        oi_upd = oi_tmp
                        li_upd = tilda_lij.squeeze(-1)
                        mi_upd = tilda_mij.squeeze(-1)
                    else:
                        mi = mi_upd.unsqueeze(-1)
                        max_new, _ = torch.max(
                            torch.cat([mi, tilda_mij], dim=-1), dim=-1, keepdim=True
                        )
                        tsub = muls_res - max_new
                        tilda_pij = torch.exp(tsub)
                        tilda_lij = torch.sum(tilda_pij, dim=-1, keepdim=True)
                        tsub2 = torch.sub(mi, max_new)
                        mi_upd = max_new.squeeze(-1)
                        update_mul = torch.exp(tsub2)
                        li = li_upd.unsqueeze(-1)
                        sum_new = li * update_mul + tilda_lij
                        li_upd = sum_new.squeeze(-1)
                        q1 = matmul_proxy(tilda_pij.to(dtype), kvj)
                        oi_upd = oi_upd * update_mul + q1
                    if s2_idx == s2_loop - 1:
                        flash_end(out, sinks, li_upd, mi_upd, oi_upd, n2g_ofs, g_tile, bs_ofs, dtype,
                                is_new_sink=is_new_sink)
    return out


def cfa_golden(q, kv, attn_sink, blk_cfa, seqused_kv, out, enable_flash=True, cmp_r=1, is_new_sink=False,
                kv_win=None, blk_win=None):
    if not enable_flash:
        fp64 = torch.float64
        q = q.to(fp64)
        kv = kv.to(fp64)
        b = seqused_kv.shape[0]
        bs = q.shape[0]
        s1 = bs // b
        nkv = kv.shape[2]
        d = kv.shape[3]
        softmax_scale = d ** -0.5
        compress_actual_seqs = seqused_kv // cmp_r
        kv_bsnd = kv_cache_concat_bsnd(kv, blk_cfa, compress_actual_seqs)
        if kv_win is not None and blk_win is not None:
            k_cfa_bsnd = kv_cache_concat_bsnd(kv, blk_cfa, compress_actual_seqs)
            k_win_bsnd = kv_cache_concat_bsnd(kv_win, blk_win, seqused_kv)
            kv_bsnd = torch.cat([k_cfa_bsnd], dim=1)
        for i in range(b):
            for j in range(s1):
                for n2_idx in range(nkv):
                    seq_end = seqused_kv[i] - (s1 - 1 - j)
                    seq_len = seq_end // cmp_r
                    q_bs = q[i * s1 + j]
                    kv_win_view = k_win_bsnd[i, seq_end - 128 : seq_end, :, :].reshape(-1, d)
                    kv_bs = kv_bsnd[i, :seq_len, n2_idx : n2_idx + 1].reshape(seq_len, d)
                    kv_bs = torch.cat([kv_win_view, kv_bs], dim=0)
                    qk_bmm_res = torch.matmul(q_bs, kv_bs.transpose(1, 0))
                    qk_ele_res = qk_bmm_res * softmax_scale
                    softmax_res, _, _ = softmax(qk_ele_res, attn_sink, True, is_new_sink=is_new_sink)
                    bmm2_res = torch.matmul(softmax_res, kv_bs)
                    out[i * s1 + j] = bmm2_res
        return out, kv_bs
    else:
        cfa_flash_torch(
            q=q,
            cmp_kv=kv,
            sinks=attn_sink,
            cmp_block_table=blk_cfa,
            seqused_kv=seqused_kv,
            out=out,
            cmp_ratio=cmp_r,
            is_new_sink=is_new_sink,
            ori_kv=kv_win,
            ori_block_table=blk_win,
        )


def test_compressed_flash_attention(enable_flash: bool, enable_high_perf: bool, enable_graph: bool, device: str, attn_cfg: AttentionConfig):
    torch_dtype = torch.bfloat16
    b = attn_cfg.b
    s1 = attn_cfg.s1
    d = attn_cfg.q_d
    nq = attn_cfg.n1
    nkv = attn_cfg.n2
    cmp_ratio = attn_cfg.cmp_ratio

    block_size = attn_cfg.block_size
    max_blocks = attn_cfg.max_blocks
    seqused_kv = attn_cfg.actual_seq.npu()

    q_shape = [b * s1, nq, d]
    cmp_kv_shape = [attn_cfg.kv_num_blocks, block_size, nkv, d]
    cmp_blk_tbl_shape = [attn_cfg.block_table_batch, max_blocks]
    max_actual_seq = max(seqused_kv)
    win_max_actual_seq = max(max_actual_seq, block_size + s1 - 1)
    win_max_blocks = math.ceil(win_max_actual_seq / block_size)
    ori_kv_shape = [b * win_max_blocks, block_size, nkv, d]
    ori_blk_tbl_shape = cmp_blk_tbl_shape

    empty_kwargs = {"dtype": torch_dtype, "device": device}
    q = torch.empty(q_shape, **empty_kwargs).uniform_(-1, 1).npu()
    cmp_kv = torch.empty(cmp_kv_shape, **empty_kwargs).uniform_(-1, 1).npu()
    sinks = torch.empty(nq, dtype=torch.float32, device=device).uniform_(-1, 1).npu()
    ori_kv = torch.empty(ori_kv_shape, **empty_kwargs).uniform_(-1, 1).npu()
    ori_block_table = gen_block_table(seqused_kv, block_size, ori_blk_tbl_shape, cmp_ratio=cmp_ratio,
                                        enable_win=True).npu()

    output_flash = torch.zeros(q_shape, **empty_kwargs).npu()

    cmp_block_table = gen_block_table(seqused_kv, block_size, cmp_blk_tbl_shape, cmp_ratio=cmp_ratio).npu()
    attention_out = torch.zeros(q_shape, **empty_kwargs)

    cfa_golden(q, cmp_kv, sinks, cmp_block_table, seqused_kv, output_flash, enable_flash=True, cmp_r=cmp_ratio, \
               is_new_sink=True, kv_win=ori_kv, blk_win=ori_block_table)
    threhold = 5e-4
    # acl graph mode
    if enable_graph:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        compiler_config = CompilerConfig()
        compiler_config.mode = "reduce-overhead"
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        model = torch.compile(MM(), dynamic=False, fullgraph=True, backend=npu_backend)
        for _ in range(1):
            attention_out = model(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table, cmp_ratio)
    else:
        attention_out = npu_compress_flash_attention(q, cmp_kv, sinks, cmp_block_table, seqused_kv, ori_kv, ori_block_table,
                                        cmp_ratio)

    from compare import compare
    compare(output_flash, attention_out, "golden vs npu", rtol=threhold, atol=threhold)


def test_compress_flash_attention_kernel(enable_flash: bool = False, enable_high_perf: bool = False, enable_graph: bool = True,
                    device_id: int = 0):
    device_id = max(device_id, int(os.environ.get('DEVICE_ID', 0)))
    device = f'npu:{device_id}'
    attn_cfg = get_decode_case(device=device)
    test_compressed_flash_attention(enable_flash=enable_flash, enable_high_perf=enable_high_perf, enable_graph=True, device=device,
                                  attn_cfg=attn_cfg)


if __name__ == "__main__":
    import argparse as ap
    p = ap.ArgumentParser(description="Parameter configuration")
    p.add_argument("-f", "--enable-flash", action="store_true", help="Enable flash mode")
    p.add_argument("-p", "--high-perf", action="store_true", help="Enable high performance mode")
    p.add_argument("-g", "--enable-graph", action="store_true", help="Enable high performance mode")
    p.add_argument("-c", "--device-id", type=int, default=0, help="GPU ID, default is 0")
    p.add_argument("-u", "--upper", type=int, default=6000, help="Fusion upper bound method")
    args = p.parse_args()
    test_compress_flash_attention_kernel(enable_flash=args.enable_flash, enable_high_perf=args.high_perf, enable_graph=args.enable_graph,
             device_id=args.device_id)
