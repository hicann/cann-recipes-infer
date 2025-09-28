# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import tilelang
from tilelang import DataType, language as T


tilelang.disable_cache()
os.environ["ACL_OP_INIT_MODE"] = "1"


@tilelang.jit(out_idx=[3],)
def sparse_attention_fwd(
    heads,
    dim,
    tail_dim,
    top_k,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_i=64,
):
    if dim != tilelang.math.next_power_of_2(dim):
        raise ValueError(f"haven't check padding correctness yet, dim={dim}")
    if tail_dim != tilelang.math.next_power_of_2(tail_dim):
        raise ValueError(f"haven't check padding correctness yet, dim={tail_dim}")
    if is_causal != True:
        raise ValueError("non-casual is not supporteds")
    if top_k % block_i != 0:
        raise ValueError("otherwise will load some index=0 thus causing wrong kv to be loaded")

    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
    else:
        sm_scale = sm_scale

    batch = 1
    seq_len = 128

    seq_len_kv = 32768
    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, top_k]
    indices_dtype = "int32"
    d_type = "float16"
    accum_dtype = "float"

    h_kv = head_kv

    padded_h = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_h != h_kv:
        raise ValueError("""here we solve the h padding automically,
            other wise you should handle Q copy and Output copy with your mask
            (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automically)""")

    bi = block_i
    ni = tilelang.cdiv(top_k, block_i)
    d = dim
    d_tail = tail_dim

    if head_kv > 64:
        if head_kv % 64 != 0:
            raise AssertionError('head_kv should be a multiple of 64')
        replicate_h = head_kv // 64
    else:
        replicate_h = 1

    h_per_block = padded_h if replicate_h == 1 else 64

    v_block = h_per_block // 2

    block_num = seq_len * replicate_h * batch * kv_group

    @T.prim_func
    def main(
        q: T.Tensor(q_shape, d_type),
        kv: T.Tensor(kv_shape, d_type),
        indices: T.Tensor(indices_shape, indices_dtype),
        output: T.Tensor(o_shape, d_type),

        workspace1: T.Tensor([block_num, bi, d], d_type),
        workspace2: T.Tensor([block_num, bi, d_tail], d_type),
        workspace3: T.Tensor([block_num, h_per_block, bi], accum_dtype),
        workspace4: T.Tensor([block_num, h_per_block, bi], d_type),
        workspace5: T.Tensor([block_num, h_per_block, d], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len * replicate_h)
            by = cid // (seq_len * replicate_h) % batch
            bz = cid // (seq_len * replicate_h) // batch % kv_group

            q_l1 = T.alloc_L1([h_per_block, d], d_type)
            q_tail_l1 = T.alloc_L1([h_per_block, d_tail], d_type)
            kv_l1 = T.alloc_L1([bi, d], d_type)
            kv_tail_l1 = T.alloc_L1([bi, d_tail], d_type)
            acc_s_l1 = T.alloc_L1([h_per_block, bi], d_type)

            acc_s_l0c = T.alloc_L0C([h_per_block, bi], accum_dtype)
            acc_o_l0c = T.alloc_L0C([h_per_block, d], accum_dtype)

            acc_o = T.alloc_ub([v_block, d], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            indices_ub_ = T.alloc_ub([bi], indices_dtype)
            kv_ub = T.alloc_ub([d], d_type)
            kv_tail_ub = T.alloc_ub([d_tail], d_type)
            acc_s_ub = T.alloc_ub([v_block, bi], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, bi], accum_dtype)
            tmp_ub = T.alloc_ub([3 * DataType(accum_dtype).bits // 8 * v_block * bi], "uint8")
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, bi], d_type)
            acc_o_ub = T.alloc_ub([v_block, d], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, d], d_type)

            T.annotate_address({
                q_l1: 0,
                q_tail_l1: 65536,
                kv_l1: 73728,
                kv_tail_l1: 139264,
                acc_s_l1: 139264,

                acc_s_l0c: 0,
                acc_o_l0c: 0,

                acc_o: 0,
                sumexp: 65536,
                m_i: 65664,
                indices_ub_: 65792,
                kv_ub: 66048,
                kv_tail_ub: 67072,
                acc_s_ub: 66048,
                m_i_prev: 74240,
                acc_s_ub_: 74368,
                tmp_ub: 74368,
                sumexp_i_ub: 98944,
                acc_s_half: 98944,
                acc_o_ub: 98944,
                acc_o_half: 98944
            })

            b_i = by
            g_i = bz

            s_i = (bx // replicate_h)

            h0 = g_i * padded_h + (0 if replicate_h == 1 else (bx % replicate_h) * 64)
            h1 = h0 + h_per_block

            with T.Scope("C"):
                T.copy(q[b_i, s_i, h0:h1, :d], q_l1)
                T.copy(q[b_i, s_i, h0:h1, d:], q_tail_l1)
                T.barrier_all()
                for _ in T.serial(ni):
                    T.wait_cross_flag(0)
                    T.barrier_all()
                    T.copy(workspace1[cid, 0:bi, 0:d], kv_l1)
                    T.copy(workspace2[cid, 0:bi, 0:d_tail], kv_tail_l1)
                    T.barrier_all()

                    T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()
                    T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)
                    T.barrier_all()

                    T.copy(acc_s_l0c, workspace3[cid, 0:h_per_block, 0:bi])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 1)

                    T.wait_cross_flag(2)
                    T.barrier_all()

                    T.copy(workspace4[cid, 0:h_per_block, 0:bi], acc_s_l1)
                    T.barrier_all()

                    T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)
                    T.barrier_all()

                    T.copy(acc_o_l0c, workspace5[cid, 0:h_per_block, 0:d])
                    T.barrier_all()

                    T.set_cross_flag("FIX", 3)
                    T.wait_cross_flag(4)
                T.wait_cross_flag(8)

            with T.Scope("V"):
                T.fill(acc_o, 0.0)
                T.fill(sumexp, 0.0)
                T.fill(m_i, -2.0 ** 30)
                T.barrier_all()

                for loop_i in range(ni):
                    T.copy(indices[b_i, s_i, g_i, loop_i * bi:loop_i * bi + bi], indices_ub_)
                    T.barrier_all()

                    for bi_i in range(bi // 2):
                        T.copy(kv[b_i, indices_ub_[bi_i + vid * bi // 2], g_i, :d], kv_ub)
                        T.copy(kv[b_i, indices_ub_[bi_i + vid * bi // 2], g_i, d:], kv_tail_ub)
                        T.barrier_all()
                        T.copy(kv_ub, workspace1[cid, bi_i + vid * bi // 2, :])
                        T.copy(kv_tail_ub, workspace2[cid, bi_i + vid * bi // 2, :])
                        T.barrier_all()

                    T.set_cross_flag("MTE3", 0)

                    T.fill(acc_s_ub, 0.0)
                    T.barrier_all()

                    T.copy(m_i, m_i_prev)
                    T.barrier_all()

                    T.wait_cross_flag(1)
                    T.copy(workspace3[cid, vid * v_block:vid * v_block + v_block, :], acc_s_ub_)
                    T.barrier_all()

                    T.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.barrier_all()

                    T.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.barrier_all()

                    T.reduce_max(m_i, acc_s_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    T.max(m_i, m_i, m_i_prev)
                    T.barrier_all()


                    T.sub(m_i_prev, m_i_prev, m_i)
                    T.barrier_all()

                    T.exp(m_i_prev, m_i_prev)
                    T.barrier_all()

                    for h_i in range(v_block):
                        T.barrier_all()
                        T.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                        T.barrier_all()

                    T.exp(acc_s_ub, acc_s_ub)
                    T.barrier_all()

                    T.reduce_sum(sumexp_i_ub, acc_s_ub, tmp_ub, dim=-1)
                    T.barrier_all()

                    T.mul(sumexp, sumexp, m_i_prev)
                    T.barrier_all()

                    T.add(sumexp, sumexp, sumexp_i_ub)
                    T.barrier_all()

                    for h_i in range(v_block):
                        T.barrier_all()
                        T.mul(acc_o[h_i, :], acc_o[h_i, :], m_i_prev[h_i])
                        T.barrier_all()

                    T.copy(acc_s_ub, acc_s_half)
                    T.barrier_all()

                    T.copy(acc_s_half, workspace4[cid, vid * v_block:vid * v_block + v_block, :])
                    T.barrier_all()

                    T.set_cross_flag("MTE3", 2)

                    T.wait_cross_flag(3)
                    T.barrier_all()

                    T.copy(workspace5[cid, vid * v_block:vid * v_block + v_block, :], acc_o_ub)
                    T.barrier_all()

                    T.add(acc_o, acc_o, acc_o_ub)
                    T.barrier_all()

                    T.set_cross_flag("V", 4)
                    T.barrier_all()

                for h_i in range(v_block):
                    T.barrier_all()
                    T.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])
                    T.barrier_all()

                T.copy(acc_o, acc_o_half)
                T.barrier_all()
                T.copy(acc_o_half, output[b_i, s_i, h0 + vid * v_block:h1 + vid * v_block, :])

                T.barrier_all()

                T.set_cross_flag("MTE3", 8)

    return main
