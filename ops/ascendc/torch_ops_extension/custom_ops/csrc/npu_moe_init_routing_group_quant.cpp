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
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

constexpr int64_t DIM_X = 2;
constexpr int64_t DIM_EXPERT_IDX = 2;
constexpr int64_t LENGTH_ACTIVE_EXPERT_RANGE = 2;
constexpr int64_t EXPERT_TOKENS_COUNT = 1;
constexpr int64_t EXPERT_TOKENS_KEY_VALUE = 2;
constexpr int64_t QUANT_MODE_UNQUANT = -1;
constexpr int64_t QUANT_MODE_STATIC = 0;
constexpr int64_t QUANT_MODE_DYNAMIC = 1;
constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
constexpr int64_t QUANT_MODE_GROUP_FP8_E5M2 = 4;
constexpr int64_t QUANT_MODE_GROUP_FP8_E4M3FN = 5;
constexpr int64_t MXQUANT_BLOCK_SIZE = 32;
constexpr int64_t GROUP_QUANT_GROUP_SIZE = 128;
constexpr int64_t PAD_TO_EVEN_FACTOR = 2;

constexpr int64_t EXPERT_NUM_V2 = 128;
constexpr int64_t EXPERT_NUM_MIN_V2 = 0;
constexpr int64_t EXPERT_NUM_MAX_V2 = 128;
constexpr int64_t HIDDEN_DIM_VAL_V2 = 2048;

inline bool IsQuantModeMXFP8(int64_t quantMode)
{
    return quantMode == QUANT_MODE_MXFP8_E5M2 || quantMode == QUANT_MODE_MXFP8_E4M3FN;
}

inline bool IsFP8GroupQuant(int64_t quantMode)
{
    return quantMode == QUANT_MODE_GROUP_FP8_E4M3FN || quantMode == QUANT_MODE_GROUP_FP8_E5M2;
}

at::IntArrayRef init_new_active_expert_range(at::IntArrayRef &active_expert_range, int64_t expert_num)
{
    if (active_expert_range.empty()) {
        static std::vector<int64_t> default_active_expert_range = {0, expert_num};
        return at::IntArrayRef(default_active_expert_range);
    } else {
        return active_expert_range;
    }
}

// step2, 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing_group_quant_npu(const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type, int64_t group_size)
{
    at::IntArrayRef current_active_expert_range = init_new_active_expert_range(active_expert_range, expert_num);
    int expert_length = current_active_expert_range[1] - current_active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
    const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });
    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];

    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;

    expanded_scale_len = (active_num <= 0) ? bs * k : std::min<int64_t>(active_num, bs * k);
    at::SmallVector<int64_t, SIZE> expanded_x_size = {expanded_scale_len, h};
    switch (quant_mode) {
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_GROUP_FP8_E5M2:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kFloat8_e5m2));
            break;
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_GROUP_FP8_E4M3FN:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kFloat8_e4m3fn));
            break;
        case QUANT_MODE_STATIC:
        case QUANT_MODE_DYNAMIC:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kChar));
            break;
        default:  // quant_mode == QUANT_MODE_UNQUANT
            expanded_x = at::empty(expanded_x_size, x.options().dtype(x.dtype()));
    }

    at::SmallVector<int64_t, SIZE> expanded_row_idx_size = {bs * k};
    at::Tensor expanded_row_idx = at::empty(expanded_row_idx_size, expert_idx.options().dtype(expert_idx.dtype()));

    at::SmallVector<int64_t, SIZE> expert_tokens_count_or_cumsum_size;
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type < EXPERT_TOKENS_KEY_VALUE) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum_size = {expert_length};
        expert_tokens_count_or_cumsum = at::empty(expert_tokens_count_or_cumsum_size, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == EXPERT_TOKENS_KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum_size = {expert_num, 2};
        expert_tokens_count_or_cumsum = at::empty(expert_tokens_count_or_cumsum_size, x.options().dtype(at::kLong));
    }

    at::Tensor expanded_scale;
    at::SmallVector<int64_t, SIZE> scale_cols_size;
    if (IsQuantModeMXFP8(quant_mode)) {
        // scale_cols为h向上整除32后向上对齐到偶数倍
        int64_t scale_cols = (h + MXQUANT_BLOCK_SIZE - 1) / MXQUANT_BLOCK_SIZE;
        scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) / PAD_TO_EVEN_FACTOR;
        scale_cols_size = {expanded_scale_len, scale_cols, PAD_TO_EVEN_FACTOR};
        expanded_scale =  at::empty(scale_cols_size, x.options().dtype(at::kFloat8_e8m0fnu));
    } else if (IsFP8GroupQuant(quant_mode)){
        int64_t scale_cols = (h + GROUP_QUANT_GROUP_SIZE - 1) / GROUP_QUANT_GROUP_SIZE;
        scale_cols_size = {expanded_scale_len, scale_cols};
        expanded_scale = at::empty(scale_cols_size, x.options().dtype(at::kFloat));
    } else {
        scale_cols_size = {expanded_scale_len};
        expanded_scale = at::empty(scale_cols_size, x.options().dtype(at::kFloat));
    }

    EXEC_NPU_CMD_V1(aclnnMoeInitRoutingGroupQuant,
        x,
        expert_idx,
        p_scale,
        p_offset,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range,
        row_idx_type,
        group_size,
        expanded_x,
        expanded_row_idx,
        expert_tokens_count_or_cumsum,
        expanded_scale);
    return std::make_tuple(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_init_routing_group_quant_meta(const at::Tensor &x, const at::Tensor &expert_idx,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &offset, int64_t active_num,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, at::IntArrayRef active_expert_range, int64_t row_idx_type, int64_t group_size)
{
    at::IntArrayRef current_active_expert_range = init_new_active_expert_range(active_expert_range, expert_num);
    int expert_length = current_active_expert_range[1] - current_active_expert_range[0];
    auto x_size = x.sizes();
    auto expert_idx_size = expert_idx.sizes();
    const at::Tensor &p_scale = c10::value_or_else(scale, [] { return at::Tensor(); });
    const at::Tensor &p_offset = c10::value_or_else(offset, [] { return at::Tensor(); });
    int bs = x_size[0];
    int h = x_size[1];
    int k = expert_idx_size[1];

    int64_t expanded_scale_len = 0;
    at::Tensor expanded_x;

    expanded_scale_len = (active_num <= 0) ? bs * k : std::min<int64_t>(active_num, bs * k);
    at::SmallVector<int64_t, SIZE> expanded_x_size = {expanded_scale_len, h};
    switch (quant_mode) {
        case QUANT_MODE_MXFP8_E5M2:
        case QUANT_MODE_GROUP_FP8_E5M2:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kFloat8_e5m2));
            break;
        case QUANT_MODE_MXFP8_E4M3FN:
        case QUANT_MODE_GROUP_FP8_E4M3FN:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kFloat8_e4m3fn));
            break;
        case QUANT_MODE_STATIC:
        case QUANT_MODE_DYNAMIC:
            expanded_x = at::empty(expanded_x_size, x.options().dtype(at::kChar));
            break;
        default:  // quant_mode == QUANT_MODE_UNQUANT
            expanded_x = at::empty(expanded_x_size, x.options().dtype(x.dtype()));
    }

    at::SmallVector<int64_t, SIZE> expanded_row_idx_size = {bs * k};
    at::Tensor expanded_row_idx = at::empty(expanded_row_idx_size, expert_idx.options().dtype(expert_idx.dtype()));

    at::SmallVector<int64_t, SIZE> expert_tokens_count_or_cumsum_size;
    at::Tensor expert_tokens_count_or_cumsum;
    if (expert_tokens_num_type < EXPERT_TOKENS_KEY_VALUE) {
        // expert_tokens_count_or_cumsum in [end-start, ]
        expert_tokens_count_or_cumsum_size = {expert_length};
        expert_tokens_count_or_cumsum = at::empty(expert_tokens_count_or_cumsum_size, x.options().dtype(at::kLong));
    } else if (expert_tokens_num_type == EXPERT_TOKENS_KEY_VALUE) {
        // key_value in [2, end-start]
        expert_tokens_count_or_cumsum_size = {expert_num, 2};
        expert_tokens_count_or_cumsum = at::empty(expert_tokens_count_or_cumsum_size, x.options().dtype(at::kLong));
    }

    at::Tensor expanded_scale;
    at::SmallVector<int64_t, SIZE> scale_cols_size;
    if (IsQuantModeMXFP8(quant_mode)) {
        // scale_cols为h向上整除32后向上对齐到偶数倍
        int64_t scale_cols = (h + MXQUANT_BLOCK_SIZE - 1) / MXQUANT_BLOCK_SIZE;
        scale_cols = (scale_cols + PAD_TO_EVEN_FACTOR - 1) / PAD_TO_EVEN_FACTOR;
        scale_cols_size = {expanded_scale_len, scale_cols, PAD_TO_EVEN_FACTOR};
        expanded_scale =  at::empty(scale_cols_size, x.options().dtype(at::kFloat8_e8m0fnu));
    } else if (IsFP8GroupQuant(quant_mode)){
        int64_t scale_cols = (h + GROUP_QUANT_GROUP_SIZE - 1) / GROUP_QUANT_GROUP_SIZE;
        scale_cols_size = {expanded_scale_len, scale_cols};
        expanded_scale = at::empty(scale_cols_size, x.options().dtype(at::kFloat));
    } else {
        scale_cols_size = {expanded_scale_len};
        expanded_scale = at::empty(scale_cols_size, x.options().dtype(at::kFloat));
    }
    return std::make_tuple(expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum, expanded_scale);
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_init_routing_group_quant", &custom::npu_moe_init_routing_group_quant_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_init_routing_group_quant", &custom::npu_moe_init_routing_group_quant_meta);
}}