# coding=utf-8
# This program is free software, you can redistribute it and/or modify.
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


@tilelang.jit(out_idx=[-1])
def indexer(b_param, n2_param, g_param, s1_param, s2_param, d_param, 
            top_k_param, vector_basen, vector_baseg, block_m, block_n, block_k, input_dtype="float16", calc_dtype="float"):
    @T.prim_func
    def main(
        query: T.Tensor((b_param, s1_param, n2_param, g_param * d_param), input_dtype),
        key: T.Tensor((b_param, s2_param, n2_param, d_param), input_dtype),
        qk_res: T.Tensor((b_param, n2_param, s1_param, g_param * s2_param), calc_dtype),
        weights: T.Tensor((b_param, s1_param, n2_param, g_param), calc_dtype),
        out: T.Tensor((b_param, n2_param, s1_param, top_k_param), "int")
    ):
        with T.Kernel(b_param * n2_param, is_npu=True) as (cid, vid):
            b_id = cid // n2_param
            n2_id = cid % n2_param
            with T.Scope("C"):
                q_l1 = T.alloc_L1((block_m, block_k), input_dtype)
                k_l1 = T.alloc_L1((block_n, block_k), input_dtype)

                c_l0 = T.alloc_L0C((block_m, block_n), calc_dtype)

                T.annotate_address({
                    q_l1: 0,
                    k_l1: 32768,

                    c_l0: 0,
                })
                T.barrier_all()

                for n2_serial in T.serial(n2_param):
                    for g_serial in T.serial(g_param):
                        for m_serial in T.serial(s1_param // block_m):
                            for n_serial in T.serial(s2_param // block_n):
                                T.barrier_all()
                                T.copy(query[cid, m_serial * block_m, n2_serial, g_serial * d_param], q_l1)
                                T.barrier_all()
                                T.copy(key[cid, n_serial * block_n, n2_serial, 0], k_l1)
                                T.barrier_all()
                                T.gemm_v0(q_l1, k_l1, c_l0, transpose_B=True, init=True)
                                T.barrier_all()
                                T.copy(c_l0, qk_res[cid, n2_serial, m_serial * block_m, n_serial * block_n + g_serial * s2_param], srcN=-1, enable_relu=True)
                                T.barrier_all()
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub(
                    (vector_baseg, vector_basen), calc_dtype)
                mm_res_ub_flat = T.alloc_ub(
                    (vector_baseg * vector_basen), calc_dtype)
                mm_res_ub_uint8 = T.alloc_ub(
                    (vector_baseg, vector_basen), "uint8")
                weight_ub = T.alloc_ub(vector_baseg, calc_dtype)
                weight_brcb_ub = T.alloc_ub((vector_baseg, 8), calc_dtype)
                reduce_tmp_ub = T.alloc_ub(
                    (vector_baseg, vector_basen), calc_dtype)
                reduce_g_ub = T.alloc_ub(vector_basen, calc_dtype)
                sort_indice_tmp_ub = T.alloc_ub(vector_basen, "int")
                sort_indice_tmp_ub_uint = T.alloc_ub(vector_basen, "uint")
                topk_indices_tmp_ub = T.alloc_ub(vector_basen, "int")
                topk_indices_tmp_ub_uint = T.alloc_ub(vector_basen, "uint")
                topk_global_ub1 = T.alloc_ub([top_k_param // vector_basen, vector_basen * 2], calc_dtype)
                topk_global_ub1_flat = T.alloc_ub(top_k_param, "int")
                topk_global_ub1_uint = T.alloc_ub([top_k_param // vector_basen, vector_basen * 2], "uint")
                topk_global_ub2 = T.alloc_ub(top_k_param * 2, calc_dtype)

                T.annotate_address({
                    mm_res_ub: 0,
                    mm_res_ub_flat: 0,
                    mm_res_ub_uint8: 0,
                    weight_ub: 65536,
                    weight_brcb_ub: 65664,
                    reduce_tmp_ub: 66688,
                    reduce_g_ub: 132224,
                    sort_indice_tmp_ub: 134272,
                    sort_indice_tmp_ub_uint: 134272,
                    topk_indices_tmp_ub: 136320, topk_indices_tmp_ub_uint: 136320,
                    topk_global_ub1: 138368, topk_global_ub1_uint: 138368, topk_global_ub1_flat: 138368,
                    topk_global_ub2: 154752
                })

                total_process_num = n2_param * s1_param
                each_core_process_num = total_process_num // 2
                s1_start_idx = vid * each_core_process_num
                s1_end_idx = s1_start_idx + each_core_process_num

                T.wait_cross_flag(0)
                T.arith_progression(topk_indices_tmp_ub, 0, 1, vector_basen)

                for s1_id in T.serial(s1_start_idx, s1_end_idx):
                    T.barrier_all()
                    T.init_sort_buf(topk_global_ub2, top_k_param * 2, 0)
                    for s2_id in T.serial(s2_param // vector_basen):
                        T.barrier_all()
                        T.fill(reduce_tmp_ub, 0)
                        T.fill(reduce_g_ub, 0)
                        T.barrier_all()

                        for g_id in T.serial(g_param // vector_baseg):
                            T.barrier_all()
                            T.copy(
                                qk_res[cid, n2_id, s1_id, g_id * vector_baseg * s2_param + s2_id * vector_basen], mm_res_ub, s2_param)
                            T.barrier_all()
                            T.copy(
                                weights[cid, s1_id, n2_id, g_id * vector_baseg], weight_ub)
                            T.barrier_all()

                            for i in range(vector_baseg):
                                T.barrier_all()
                                T.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                T.barrier_all()

                            T.barrier_all()
                            T.add(reduce_tmp_ub, mm_res_ub, reduce_tmp_ub)
                            T.barrier_all()

                        merge_sort_times = top_k_param // vector_basen
                        T.barrier_all()
                        T.reduce_sum(reduce_g_ub, reduce_tmp_ub, mm_res_ub_uint8, 0)
                        T.barrier_all()
                        T.add(sort_indice_tmp_ub,
                                topk_indices_tmp_ub, T.int32(s2_id * vector_basen))
                        T.barrier_all()
                        T.sort(topk_global_ub1[(s2_id % merge_sort_times), :], reduce_g_ub, sort_indice_tmp_ub_uint, mm_res_ub, vector_basen // 32)
                        T.barrier_all()

                        if s2_id % merge_sort_times == merge_sort_times - 1:
                            if s2_id == merge_sort_times - 1:
                                T.merge_sort(topk_global_ub2, topk_global_ub1, vector_basen, merge_sort_times, 0)
                            else:
                                T.merge_sort(mm_res_ub, topk_global_ub1, vector_basen, merge_sort_times, 1)
                                T.barrier_all()
                                T.topk(topk_global_ub2, topk_global_ub1, mm_res_ub, vector_basen * merge_sort_times)
                        T.barrier_all()
                    T.barrier_all()
                    T.gather_mask(topk_global_ub1, topk_global_ub2, top_k_param)
                    T.barrier_all()
                    T.copy(topk_global_ub1_flat, out[cid, n2_id, s1_id, 0])
                    T.barrier_all()
    return main
