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


tilelang.cache.clear_cache()


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
)
def act_quant_kernel_int8_optimized(
    N: int,
    block_M: int = 32, block_N: int = 32,
    round_scale: bool = False
):
    M = T.symbolic("M")
    VEC_NUM = 2
    CAST_MODE = "CAST_NONE"
    
    int8_min = -128
    int8_max = 127
    int8_abs_max = 127.0
    
    m_num = M // block_M
    n_num = N // block_N
    block_M_2 = block_M // VEC_NUM

    @T.prim_func
    def main(
        X: T.Tensor([M, N], "bfloat16"),
        Y: T.Tensor([M, N], "int8"),
        S: T.Tensor([M], "float"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm = cid // n_num
            bn = cid % n_num
            
            x_ub = T.alloc_ub([block_M_2, block_N], "bfloat16")
            x_ub_half = T.alloc_ub([block_M_2, block_N], "float16")
            x_ub_fp_abs = T.alloc_ub([block_M_2, block_N], "float")
            y_ub = T.alloc_ub([block_M_2, block_N], "int8")
            
            max_ub = T.alloc_ub([block_M_2], "float")
            scale_ub = T.alloc_ub([block_M_2], "float")
            x_ub_fp = T.alloc_ub([block_M_2, block_N], "float")
            x_ub_fp_1 = T.alloc_ub([block_M_2, block_N], "float")
            scale_global = T.alloc_ub([block_M_2, 1], "float")
            
            with T.Scope("V"):
              T.copy(X[bm * block_M + vid * block_M_2, bn * block_N], x_ub)
              T.tile.fill(max_ub, 0.0)
              
              T.tile.cast(x_ub_fp, x_ub, mode=CAST_MODE, count=block_M_2 * block_N)
              T.tile.abs(x_ub_fp_abs, x_ub_fp)

              T.reduce_max(x_ub_fp_abs, max_ub, dim=-1)
            
              for i in T.Parallel(block_M_2):
                scale_ub[i] = max_ub[i] / int8_abs_max
            
              for (i, j) in T.Parallel(block_M_2, block_N):
                x_ub_fp[i, j] = x_ub_fp[i, j] / scale_ub[i]

              T.tile.clamp(x_ub_fp, x_ub_fp, -127.0, 127.0, block_M_2 * block_N)              

              T.tile.round(x_ub_fp, x_ub_fp, block_M_2 * block_N)

              T.tile.cast(x_ub_half, x_ub_fp, mode=CAST_MODE, count=block_M_2 * block_N)

              T.tile.cast(y_ub, x_ub_half, mode=CAST_MODE, count=block_M_2 * block_N)
              T.copy(y_ub, Y[bm * block_M + vid * block_M_2 : bm * block_M + vid * block_M_2 + block_M_2, bn * block_N : bn * block_N + block_N])

              T.copy(scale_ub, S[bm * block_M + vid * block_M_2 : bm * block_M + vid * block_M_2 + block_M_2])
    
    return main
