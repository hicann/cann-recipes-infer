/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

// 工具函数，推导输出shape
at::Tensor construct_sparse_antiquant_infer_output_tensor(
    const at::Tensor& query, const at::Tensor& value, std::string layout_query_str,
    std::string layout_kv_str, const uint64_t &rope_head_dim)
{
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3) - rope_head_dim};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) - rope_head_dim};
    }
    at::Tensor output = at::empty(output_size, query.options().dtype(query.dtype()));

    return output;
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_sparse_flash_attention_antiquant_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t attention_mode, int64_t quant_scale_repo_mode,
    int64_t tile_size, int64_t rope_head_dim)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor output = construct_sparse_antiquant_infer_output_tensor(
        query, value, layout_query_str, layout_kv_str, rope_head_dim);
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD_V1(aclnnSparseFlashAttentionAntiquant, query,
        key, value, sparse_indices, key_dequant_scale, value_dequant_scale, block_table, actual_seq_lengths_query,
        actual_seq_lengths_kv, scale_value, sparse_block_size, key_quant_mode, value_quant_mode,
        layout_query_ptr, layout_kv_ptr, sparse_mode, attention_mode, quant_scale_repo_mode, tile_size, rope_head_dim,
        output);
    return output;
}

// step3, 为META设备实现前向接口
at::Tensor npu_sparse_flash_attention_antiquant_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t attention_mode, int64_t quant_scale_repo_mode,
    int64_t tile_size, int64_t rope_head_dim)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);
    at::Tensor output = construct_sparse_antiquant_infer_output_tensor(
        query, value, layout_query_str, layout_kv_str, rope_head_dim);

    return output;
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_flash_attention_antiquant", &custom::npu_sparse_flash_attention_antiquant_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_sparse_flash_attention_antiquant", &custom::npu_sparse_flash_attention_antiquant_meta);
}
