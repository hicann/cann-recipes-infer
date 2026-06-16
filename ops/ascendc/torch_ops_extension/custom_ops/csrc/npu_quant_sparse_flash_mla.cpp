/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

inline TensorWrapper qsmla_make_wrapper(const at::Tensor &tensor, aclDataType tensor_acltype)
{
    return {tensor, tensor_acltype};
}

std::tuple<at::Tensor, at::Tensor> construct_quant_sparse_flash_mla_atten_out_tensor(const at::Tensor &q,
                                                                                     const at::Tensor &ori_kv,
                                                                                     std::string layout_q_str,
                                                                                     bool return_softmax_lse)
{
    TORCH_CHECK(layout_q_str == "BSND" || layout_q_str == "TND",
                "The layout of query only support BSND and TND, but got ", layout_q_str);
    for (auto i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    at::SmallVector<int64_t, SIZE> atten_out_size;
    at::SmallVector<int64_t, SIZE> softmax_lse_size;
    if (layout_q_str == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be "
                    "4, but got ",
                    q.dim());
        atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3)};
    } else {
        TORCH_CHECK(q.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be "
                    "3, but got ",
                    q.dim());
        atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2)};
    }
    at::Tensor atten_out = at::empty(atten_out_size, q.options().dtype(torch::kBFloat16));
    at::Tensor softmax_lse;

    if (return_softmax_lse) {
        if (layout_q_str == "BSND") {
            // 对齐 Python: [q.shape[0], ori_kv.shape[2], q.shape[1], q.shape[2] //
            // ori_kv.shape[2]]
            softmax_lse_size = {q.size(DIM_0), ori_kv.size(DIM_2), q.size(DIM_1), q.size(DIM_2) / ori_kv.size(DIM_2)};
        } else {
            // 对齐 Python: [ori_kv.shape[1], q.shape[0], q.shape[2] //
            // ori_kv.shape[2]]
            softmax_lse_size = {ori_kv.size(DIM_1), q.size(DIM_0), q.size(DIM_2) / ori_kv.size(DIM_2)};
        }
    } else {
        // 不返回时tensor传空
        softmax_lse_size = {};
    }
    softmax_lse = at::empty(softmax_lse_size, q.options().dtype(torch::kFloat32));

    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_quant_sparse_flash_mla_npu(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &q_descale, const c10::optional<at::Tensor> &ori_kv_descale,
    const c10::optional<at::Tensor> &cmp_kv_descale, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t qkv_quant_mode,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, int64_t topk_value_mode,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(qkv_quant_mode == 1, "qkv_quant_mode only support 1.")

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    const at::Tensor &ori_kv_val = *ori_kv;
    // convert str
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> quant_sparse_flash_mla_atten_out =
        construct_quant_sparse_flash_mla_atten_out_tensor(q, ori_kv_val, layout_q_str, return_softmax_lse);
    at::Tensor atten_out = std::get<0>(quant_sparse_flash_mla_atten_out);
    at::Tensor softmax_lse = std::get<1>(quant_sparse_flash_mla_atten_out);

    if (qkv_quant_mode == 1) { // HIFLOAT8 FULL QUANT
        at::Tensor null_tensor;
        auto ori_kv_value = ori_kv.has_value() ? ori_kv.value() : null_tensor;
        auto cmp_kv_value = cmp_kv.has_value() ? cmp_kv.value() : null_tensor;
        TensorWrapper q_wrapper = qsmla_make_wrapper(q, ACL_HIFLOAT8);
        TensorWrapper ori_kv_wrapper = qsmla_make_wrapper(ori_kv_value, ACL_HIFLOAT8);
        TensorWrapper cmp_kv_wrapper = qsmla_make_wrapper(cmp_kv_value, ACL_HIFLOAT8);

        EXEC_NPU_CMD_V1(aclnnQuantSparseFlashMla, q_wrapper, ori_kv_wrapper, cmp_kv_wrapper, q_descale, ori_kv_descale,
                        cmp_kv_descale, ori_sparse_indices, cmp_sparse_indices, ori_block_table, cmp_block_table,
                        cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv,
                        cmp_residual_kv, ori_topk_length, cmp_topk_length, sinks, metadata, qkv_quant_mode,
                        softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
                        layout_q_ptr, layout_kv_ptr, topk_value_mode, return_softmax_lse, atten_out, softmax_lse);
    }

    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_quant_sparse_flash_mla_meta(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &q_descale, const c10::optional<at::Tensor> &ori_kv_descale,
    const c10::optional<at::Tensor> &cmp_kv_descale, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_ori_kv,
    const c10::optional<at::Tensor> &seqused_cmp_kv, const c10::optional<at::Tensor> &cmp_residual_kv,
    const c10::optional<at::Tensor> &ori_topk_length, const c10::optional<at::Tensor> &cmp_topk_length,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t qkv_quant_mode,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, int64_t topk_value_mode,
    bool return_softmax_lse)
{
    TORCH_CHECK(q.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(qkv_quant_mode == 1, "qkv_quant_mode only support 1.")

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    const at::Tensor &ori_kv_val = *ori_kv;
    // convert str
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> quant_sparse_flash_mla_atten_out =
        construct_quant_sparse_flash_mla_atten_out_tensor(q, ori_kv_val, layout_q_str, return_softmax_lse);
    at::Tensor atten_out = std::get<0>(quant_sparse_flash_mla_atten_out);
    at::Tensor softmax_lse = std::get<1>(quant_sparse_flash_mla_atten_out);

    return std::tuple<at::Tensor, at::Tensor>(atten_out, softmax_lse);
}
} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("npu_quant_sparse_flash_mla", &custom::npu_quant_sparse_flash_mla_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("npu_quant_sparse_flash_mla", &custom::npu_quant_sparse_flash_mla_meta);
}