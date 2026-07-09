import tilelang
from tilelang import DataType, language as T
import torch
import torch_npu
import numpy as np


tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# kernel
@tilelang.jit(out_idx=[4, 5, 6], workspace_idx=[3], pass_configs=pass_configs)
def hc_split_sinkhorn(hc, sinkhorn_iters, eps):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    dtype = "float"

    block_m = 2
    vec_num = 2

    m_num = tilelang.cdiv(n, block_m)

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

            if cid * block_m + vid * block_m // vec_num < n:
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
                T.copy(mixes[cid * block_m + vid * block_m // vec_num, :], mixes_shared)

                T.tile.mul(mixes_shared, mixes_shared, hc_scale_shared)
                T.tile.add(mixes_shared, mixes_shared, hc_base_shared)
                T.copy(
                    mixes_shared, workspace[cid * block_m + vid * block_m // vec_num, :]
                )

                # pre
                T.copy(
                    workspace[cid * block_m + vid * block_m // vec_num, :hc], tmp_shared
                )
                T.tile.sigmoid(pre_shared, tmp_shared)
                T.tile.add(pre_shared, pre_shared, eps)
                T.copy(
                    pre_shared[:hc], pre[cid * block_m + vid * block_m // vec_num, :hc]
                )

                # post
                T.copy(
                    workspace[
                        cid * block_m + vid * block_m // vec_num, hc: hc + hc_pad
                    ],
                    tmp_shared,
                )
                T.tile.sigmoid(post_shared, tmp_shared)
                T.tile.mul(post_shared, post_shared, 2.0)
                T.copy(
                    post_shared[:hc],
                    post[cid * block_m + vid * block_m // vec_num, :hc],
                )

                # comb
                for i in T.serial(hc):
                    start = 2 * hc + i * hc
                    end = 2 * hc + i * hc + hc
                    T.copy(
                        workspace[cid * block_m + vid * block_m // vec_num, start:end],
                        tmp_shared,
                    )
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.sub(comb_shared[i, :], comb_shared[i, :], row_max[i])
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i, :], comb_shared[i, :], row_sum[i])
                T.tile.add(comb_shared, comb_shared, eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, eps)
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i, :], comb_shared[i, :], col_sum)

                for _ in T.serial(sinkhorn_iters - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i, :], comb_shared[i, :], row_sum[i])
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i, :], comb_shared[i, :], col_sum)

                for i in T.serial(hc):
                    T.copy(
                        comb_shared[i, :hc],
                        comb[cid * block_m + vid * block_m // vec_num, i, :],
                    )

    return main


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    b, s, _ = mixes.size()
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    pre, post, comb = kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base)
    n, hc = pre.shape
    n, hc = post.shape
    n, hc, hc = comb.shape
    pre = pre.reshape(b, n // b, hc)
    post = post.reshape(b, n // b, hc)
    comb = comb.reshape(b, n // b, hc, hc)
    # print("pre.shape: ", pre.shape)
    # print("post.shape: ", post.shape)
    # print("prcombe.shape: ", comb.shape)
    return pre, post, comb
