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
from tilelang import DataType, language as T


tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# kernel
@tilelang.jit(out_idx=[4, 5, 6], workspace_idx=[3], pass_configs=pass_configs)
def hc_split_sinkhorn(hc, sinkhorn_iters, eps):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    dtype = "float"

    block_M = 2
    VEC_NUM = 2

    m_num = tilelang.cdiv(n, block_M)

    hc_pad = hc
    if hc * 4 % 32 != 0:
        hc_pad = tilelang.cdiv(hc * 4, 32) * 32 // 4

    @T.prim_func
    def main(
            mixes: T.Tensor([n, mix_hc], dtype),
            hc_scale: T.Tensor([3], dtype),
            hc_base: T.Tensor([mix_hc], dtype),
            workspace: T.Tensor([n, mix_hc], dtype),
            pre: T.Tensor([n, hc], dtype),
            post: T.Tensor([n, hc], dtype),
            comb: T.Tensor([n, hc, hc], dtype),
    ):

        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            mixes_shared = T.alloc_shared(mix_hc, dtype)
            hc_base_shared = T.alloc_shared(mix_hc, dtype)
            hc_scale_shared = T.alloc_ub(mix_hc, dtype)

            comb_shared = T.alloc_shared((hc, hc_pad), dtype)
            pre_shared = T.alloc_shared(hc_pad, dtype)
            post_shared = T.alloc_shared(hc_pad, dtype)

            tmp_shared = T.alloc_shared(hc_pad, dtype)

            row_sum = T.alloc_shared(hc_pad, dtype)
            col_sum = T.alloc_shared(hc_pad, dtype)
            row_max = T.alloc_shared(hc_pad, dtype)

            if cid * block_M + vid * block_M // VEC_NUM < n:
                alpha_0 = hc_scale[0]
                alpha_1 = hc_scale[1]
                alpha_2 = hc_scale[2]
                
                for i in T.serial(mix_hc):
                    if i < hc:
                        hc_scale_shared[i] = alpha_0
                    elif i < 2 * hc:
                        hc_scale_shared[i] = alpha_1
                    else:
                        hc_scale_shared[i] = alpha_2
                T.copy(hc_base, hc_base_shared)
                T.copy(mixes[cid * block_M + vid * block_M // VEC_NUM, :], mixes_shared)

                T.tile.mul(mixes_shared, mixes_shared, hc_scale_shared)
                T.tile.add(mixes_shared, mixes_shared, hc_base_shared)
                T.copy(mixes_shared, workspace[cid * block_M + vid * block_M // VEC_NUM, :])

                # pre
                T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, :hc], tmp_shared)
                T.tile.sigmoid(pre_shared, tmp_shared)
                T.tile.add(pre_shared, pre_shared, eps)
                T.copy(pre_shared[:hc], pre[cid * block_M + vid * block_M // VEC_NUM, :hc])

                # post
                T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, hc:hc+hc_pad], tmp_shared)
                T.tile.sigmoid(post_shared, tmp_shared)
                T.tile.mul(post_shared, post_shared, 2.0)
                T.copy(post_shared[:hc], post[cid * block_M + vid * block_M // VEC_NUM, :hc])

                # comb
                for i in T.serial(hc):
                    start = 2 * hc + i * hc
                    end = 2 * hc + i * hc + hc
                    T.copy(workspace[cid * block_M + vid * block_M // VEC_NUM, start:end], tmp_shared)
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.sub(comb_shared[i,:], comb_shared[i,:], row_max[i])
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i,:], comb_shared[i,:], row_sum[i])
                T.tile.add(comb_shared, comb_shared, eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, eps)
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i,:], comb_shared[i,:], col_sum)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i,:], comb_shared[i,:], row_sum[i])
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i,:], comb_shared[i,:], col_sum)
                
                for i in T.serial(hc):
                    T.copy(comb_shared[i, :hc], comb[cid * block_M + vid * block_M // VEC_NUM, i, :])

    return main
