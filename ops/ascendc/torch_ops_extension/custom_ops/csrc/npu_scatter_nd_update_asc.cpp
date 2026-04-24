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

const int64_t DIM_0 = 0;
const int64_t DIM_1 = 1;
const int64_t DIM_2 = 2;
const int64_t DIM_3 = 3;
const int64_t DIM_LEN = 2;

// npu tensor max size
const int64_t SIZE = 2;

// const int64_t HC_LIMIT = 4;
const int64_t B_LIMIT = 32768;

// step1，工具函数，检查输入shape
void check_scatter_nd_update_asc_shape_and_dtype(const at::Tensor& var, const at::Tensor& indices, const at::Tensor& update) {
    // check var shape: [a, b]
    TORCH_CHECK(var.dim() == 2, "Input tensor var's dim num should be 2, actual ", var.dim(), ".");
    for (size_t i = 0; i < 2; i++) {
        TORCH_CHECK(var.size(i) > 0, "Input tensor var's shape should be positive, but var.shape[", i, "] is :", var.size(i), ".");
    }
    auto a = var.size(0);
    auto b = var.size(1);
    TORCH_CHECK(b <= B_LIMIT, "The d of var only support ", B_LIMIT, ", actual ", b, ".");
    
    // check indices shape: [c, 1]
    TORCH_CHECK(indices.dim() == 2, "Input tensor indices's dim num should be 2, actual ", indices.dim(), ".");
    auto c = indices.size(0);
    TORCH_CHECK(c > 0, "Input tensor indices's shape should be positive, but var.shape[", 0, "] is :", c, ".");
    TORCH_CHECK(indices.size(1) == 1, "The indices.shape[1] should be 1, actual indices.shape[1] is ", indices.size(1), ".");
        
    // check update [c, b]
    TORCH_CHECK(update.dim() == 2, "Input tensor update's dim num should be 2, actual ", update.dim(), ".");
    TORCH_CHECK(update.size(0) == c, "The update.shape[0] should be c, actual update.shape[0] is ", update.size(0), ", c is ", c, ".");
    TORCH_CHECK(update.size(1) == b, "The update.shape[1] should be b, actual update.shape[1] is ", update.size(1), ", b is ", b, ".");
    
    // check dtype
    TORCH_CHECK(var.dtype() == at::kChar || var.dtype() == at::kHalf || var.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or kInt8.");
    TORCH_CHECK(update.dtype() == var.dtype(), "var's dtype should be equal to update's dtype.");
    TORCH_CHECK(indices.dtype() == at::kInt || indices.dtype() == at::kLong,
                "indices should be kInt32, kInt64.");
}

// step2, 为NPU设备实现前向接口
void scatter_nd_update_asc_npu(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    check_scatter_nd_update_asc_shape_and_dtype(var, indices, update);
    // construct the output tensor
    EXEC_NPU_CMD_V1(aclnnScatterNdUpdateAsc, var, indices, update);
    return;
}

// step3, 为META设备实现前向接口
void scatter_nd_update_asc_meta(
    at::Tensor& var,
    const at::Tensor& indices,
    const at::Tensor& update)
{
    check_scatter_nd_update_asc_shape_and_dtype(var, indices, update);
    // construct the output tensor
    return;
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("scatter_nd_update_asc", &custom::scatter_nd_update_asc_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("scatter_nd_update_asc", &custom::scatter_nd_update_asc_meta);
}}