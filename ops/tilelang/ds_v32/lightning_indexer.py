# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import logging
import tilelang
import tilelang.language as T


logging.basicConfig(level=logging.INFO)

tilelang.disable_cache()

os.environ["ACL_OP_INIT_MODE"] = "1"
os.environ["TVM_BACKTRACE"] = "1"


@tilelang.jit(out_idx=[-1], workspace_idx=[-3])  # for jit
def indexer(B, N2, G, S1, S2, D, TOP_K, VECTOR_BASEN, VECTOR_BASEG, BLOCK_M, BLOCK_N, BLOCK_K, input_dtype="float16", calc_dtype="float"):

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N2, G * D), input_dtype),
        KEY: T.Tensor((B, S2, N2, D), input_dtype),
        QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
        WEIGHTS: T.Tensor((B, S1, N2, G), calc_dtype),
        OUT: T.Tensor((B, N2, S1, TOP_K), "int"),
    ):
        total_process_num = N2 * S1
        each_core_process_num = total_process_num // 2
        with T.Kernel(B * N2, is_npu=True) as (cid, vid):
            n2_id = cid % N2

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((BLOCK_M, BLOCK_K), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, BLOCK_K), input_dtype)

                C_L0 = T.alloc_L0C((BLOCK_M, BLOCK_N), calc_dtype)

                T.annotate_address(
                    {
                        # L1 address
                        Q_L1: 0,
                        K_L1: 16384,
                        # L0C address
                        C_L0: 0,
                    }
                )
                T.barrier_all()
                for n2 in T.serial(N2):
                    for g in T.serial(G):
                        for m in T.serial(S1 // BLOCK_M):
                            for n in T.serial(S2 // BLOCK_N):
                                T.barrier_all()
                                T.copy(Query[cid, m * BLOCK_M : (m + 1) * BLOCK_M, n2, g * D : (g + 1) * D], Q_L1)
                                T.barrier_all()
                                T.copy(KEY[cid, n * BLOCK_N : (n + 1) * BLOCK_N, n2, 0:D], K_L1)
                                T.barrier_all()
                                T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                                T.barrier_all()
                                T.copy(
                                    C_L0,
                                    QK_RES[
                                        cid, n2, m * BLOCK_M : (m + 1) * BLOCK_M, g, n * BLOCK_N : (n + 1) * BLOCK_N
                                    ],  # [B, N2, S1, G, S2]
                                    enable_relu=True,
                                )
                                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                mm_res_ub_flat = T.alloc_ub((VECTOR_BASEG * VECTOR_BASEN), calc_dtype)
                mm_res_ub_uint8 = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), "uint8")
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                weight_brcb_ub = T.alloc_ub((VECTOR_BASEG, 8), calc_dtype)
                reduce_tmp_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                sort_output_ub = T.alloc_ub(VECTOR_BASEN * 2, calc_dtype)
                sort_index_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                topk_global_ub1 = T.alloc_ub([TOP_K // VECTOR_BASEN, VECTOR_BASEN * 2], calc_dtype)
                topk_global_ub1_flat = T.alloc_ub(TOP_K, calc_dtype)
                topk_global_ub1_uint = T.alloc_ub([TOP_K // VECTOR_BASEN, VECTOR_BASEN * 2], "uint")
                topk_global_ub2 = T.alloc_ub(TOP_K * 2, calc_dtype)
                output_ub = T.alloc_ub(TOP_K, "int")

                T.annotate_address(
                    {
                        # ub address
                        mm_res_ub: 0,
                        mm_res_ub_flat: 0,
                        mm_res_ub_uint8: 0,
                        weight_ub: 32768,
                        weight_brcb_ub: 32832,
                        reduce_tmp_ub: 33344,
                        reduce_g_ub: 66112,
                        sort_output_ub: 67136,
                        sort_index_ub: 69184,
                        topk_global_ub1: 70208,
                        topk_global_ub1_uint: 70208,
                        topk_global_ub1_flat: 70208,
                        topk_global_ub2: 78400,
                        output_ub: 86592,
                    }
                )

                s1_start_idx = vid * each_core_process_num
                s1_end_idx = s1_start_idx + each_core_process_num

                T.wait_cross_flag(0)
                for s1_id in T.serial(s1_start_idx, s1_end_idx):
                    T.barrier_all()
                    T.tile.init_sort_buf(topk_global_ub2, TOP_K * 2, 0)
                    for s2_id in T.serial(S2 // VECTOR_BASEN):
                        T.barrier_all()
                        T.tile.fill(reduce_tmp_ub, 0)
                        T.tile.fill(reduce_g_ub, 0)
                        T.barrier_all()

                        for g_id in T.serial(G // VECTOR_BASEG):
                            T.barrier_all()
                            T.copy(
                                QK_RES[
                                    cid,
                                    n2_id,
                                    s1_id,
                                    g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG,
                                    s2_id * VECTOR_BASEN : (s2_id + 1) * VECTOR_BASEN,
                                ],
                                mm_res_ub,
                            )
                            T.barrier_all()
                            T.copy(WEIGHTS[cid, s1_id, n2_id, g_id * VECTOR_BASEG : (g_id + 1) * VECTOR_BASEG], weight_ub)
                            T.barrier_all()
                            for i in range(VECTOR_BASEG):
                                T.barrier_all()
                                T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                T.barrier_all()
                            T.barrier_all()
                            T.tile.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.barrier_all()
                        # topK
                        merge_sort_times = TOP_K // VECTOR_BASEN
                        T.barrier_all()
                        T.reduce_sum(reduce_tmp_ub, reduce_g_ub, 0)
                        T.barrier_all()
                        T.tile.sort(
                            sort_output_ub,
                            reduce_g_ub,
                            VECTOR_BASEN,
                        )
                        T.barrier_all()
                        T.tile.gather_mask(sort_index_ub, sort_output_ub, "P1010")
                        T.barrier_all()
                        T.tile.add(sort_index_ub, sort_index_ub, T.float32(s2_id * VECTOR_BASEN))
                        T.barrier_all()
                        row_idx = s2_id % merge_sort_times
                        for i in range(VECTOR_BASEN):
                            topk_global_ub1[row_idx, i * 2] = sort_output_ub[i * 2]
                            topk_global_ub1[row_idx, i * 2 + 1] = sort_index_ub[i]
                        T.barrier_all()
                        if s2_id % merge_sort_times == merge_sort_times - 1:
                            if s2_id == merge_sort_times - 1:
                                T.tile.merge_sort(
                                    topk_global_ub2,
                                    topk_global_ub1[0, :],
                                    topk_global_ub1[1, :],
                                    topk_global_ub1[2, :],
                                    topk_global_ub1[3, :],
                                )
                            else:
                                T.tile.merge_sort(
                                    reduce_tmp_ub,
                                    topk_global_ub1[0, :],
                                    topk_global_ub1[1, :],
                                    topk_global_ub1[2, :],
                                    topk_global_ub1[3, :],
                                )
                                T.barrier_all()
                                T.tile.topk(topk_global_ub2, reduce_tmp_ub, VECTOR_BASEN * merge_sort_times)
                        T.barrier_all()
                    T.barrier_all()
                    T.tile.gather_mask(topk_global_ub1, topk_global_ub2, "P1010")
                    T.barrier_all()
                    T.tile.cast(output_ub, topk_global_ub1_flat, "CAST_ROUND", TOP_K)
                    T.barrier_all()
                    T.copy(output_ub, OUT[cid, n2_id, s1_id, 0:TOP_K])
                    T.barrier_all()

    return main
