# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import tilelang
import tilelang.language as T
from tilelang import DataType
import torch
from typing import Literal


tilelang.cache.clear_cache()
torch.manual_seed(42)

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(
    out_idx=[-2],
    workspace_idx=[5],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
)
def int8_gemm_kernel_corrected(
    N: int, K: int,
    block_M: int = 32, block_N: int = 32, block_K: int = 32,
    in_dtype: Literal["int8"] = "int8",
    out_dtype: Literal["float16", "bfloat16", "float32"] = "float32",
    accum_dtype: Literal["int32"] = "int32",
    scale_dtype: Literal["float32"] = "float32"
):

    VEC_NUM = 2
    CAST_MODE = "CAST_RINT"
    M = T.symbolic("M")
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)
    block_M_2 = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
        A: T.Tensor([M, K], in_dtype),
        B: T.Tensor([N, K], in_dtype),
        scale_a: T.Tensor([M], scale_dtype),
        scale_b: T.Tensor([N], scale_dtype),
        C: T.Tensor([M, N], out_dtype),
        workspace_1: T.Tensor([M, N], accum_dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num

            with T.Scope("C"):
              A_L1 = T.alloc_L1([block_M, block_K], in_dtype)
              B_L1 = T.alloc_L1([block_N, block_K], in_dtype)
              C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

              for bk in T.serial(k_num):
                  T.copy(A[bm * block_M, bk * block_K], A_L1)
                  T.copy(B[bn * block_N, bk * block_K], B_L1)

                  T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=(bk == 0))

              T.copy(C_L0, workspace_1[bm * block_M, bn * block_N])
              T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
              c_ub = T.alloc_ub([block_M_2, block_N], accum_dtype)
              c_scale = T.alloc_ub([block_M_2, block_N], scale_dtype)
              c_out = T.alloc_ub([block_M_2, block_N], out_dtype)

              scale_a_ub = T.alloc_ub([block_M_2], scale_dtype)
              scale_b_ub = T.alloc_ub([block_N], scale_dtype)

              T.wait_cross_flag(0)
              T.copy(workspace_1[bm * block_M + vid * block_M_2, bn * block_N], c_ub)
              T.copy(scale_a[bm * block_M + vid * block_M_2], scale_a_ub)
              T.copy(scale_b[bn * block_N], scale_b_ub)

              T.tile.cast(c_scale, c_ub, mode=CAST_MODE, count=block_M_2 * block_N)

              for (i, j) in T.Parallel(block_M_2, block_N):
                c_scale[i, j] *= scale_a_ub[i]
                c_scale[i, j] *= scale_b_ub[j]

              T.tile.cast(c_out, c_scale, mode=CAST_MODE, count=block_M_2 * block_N)
              T.copy(c_out, C[bm * block_M + vid * block_M_2, bn * block_N])

    return main
