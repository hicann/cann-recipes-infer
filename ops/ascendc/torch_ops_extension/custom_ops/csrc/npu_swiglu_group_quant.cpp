/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int SWIGLU_FACTOR = 2;
const int PER_BLOCK_FP16 = 128;
const int PER_MX_FP16 = 32;
const int BLOCK_QUANT_INPUT_ALIGN = PER_BLOCK_FP16 * SWIGLU_FACTOR;
const int MX_SCALE_ALIGN_FACTOR = 2;
const int BLOCK_QUANT = 0;
const int MX_QUANT = 1;

// 工具函数，获取dst_type对应的整数表示
int64_t get_type_code(at::ScalarType dst_type)
{
    int64_t dst_type_code = 35;
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            dst_type_code = 35;
            break;
        case at::ScalarType::Float8_e4m3fn:
            dst_type_code = 36;
            break;
        case at::ScalarType::Half:
            dst_type_code = 1;
            break;
        case at::ScalarType::BFloat16:
            dst_type_code = 27;
            break;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dst_type);
            break;
    }
    return dst_type_code;
}

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_swiglu_group_quant_output_tensor(
    const at::Tensor& x,
    int64_t dst_type,
    int64_t quant_mode)
{
    at::SmallVector<int64_t, SIZE> y_size(x.sizes().begin(), x.sizes().end());
    for (auto i = 0; i < x.sizes().size(); i++) {
        TORCH_CHECK(x.size(i) >= 0, "All values within query's shape should be greater or equal "
            "than 0, but shape[", i, "] is ", x.size(i));
    }
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
            "x should be FLOAT16 or BFLOAT16.");
    int64_t x_last_dim = x.sizes().back();
    TORCH_CHECK(quant_mode == BLOCK_QUANT || quant_mode == MX_QUANT,
        "Unsupported quant mode, only support ", BLOCK_QUANT, " or ", MX_QUANT, ".");
    TORCH_CHECK(x_last_dim % BLOCK_QUANT_INPUT_ALIGN == 0,
        "In group quant, the last dim of x should be divisible by ", BLOCK_QUANT_INPUT_ALIGN,
        ", actual ", x_last_dim, ".");

    // Divide the last dimension by 2
    if (!y_size.empty()) {
        y_size.back() = y_size.back() / SWIGLU_FACTOR;
    }
    int64_t y_last_dim = y_size.back();
    auto y_dtype = dst_type == 35 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));
    at::SmallVector<int64_t, SIZE> scale_out_size(y_size.begin(), y_size.end());
    if (quant_mode == BLOCK_QUANT) {
        int64_t scaleOutLastDim = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
        scale_out_size.back() = scaleOutLastDim;
    } else if (quant_mode == MX_QUANT) {
        int64_t scaleOutLastDim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_out_size.back() = scaleOutLastDim;
        // 额外地，mxFp8需要将最后一维reshape为(-1, 2)
        scaleOutLastDim = (scaleOutLastDim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_out_size.back() = scaleOutLastDim;
        scale_out_size.push_back(static_cast<int64_t>(MX_SCALE_ALIGN_FACTOR));
    }
    auto scale_out_type =  at::kFloat;
    if (quant_mode == MX_QUANT) {
        scale_out_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale_out = at::empty(scale_out_size, x.options().dtype(scale_out_type));

    at::Tensor yOrigin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale_out, yOrigin);
}

// step2, 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant_npu(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& group_index,
    at::ScalarType dst_type = at::ScalarType::Float8_e4m3fn,
    int64_t quant_mode = 0,
    int64_t block_size = 0,
    bool round_scale = false,
    const c10::optional<double>& clamp_limit = c10::nullopt,
    bool output_origin = false)
{
    int64_t dst_type_code = get_type_code(dst_type);

    // construct the output tensor
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale_out = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    // -inf means user did not pass clamp_limit
    double actualClampLimit = clamp_limit.has_value() ? clamp_limit.value()
                                                        : -std::numeric_limits<double>::infinity();
    EXEC_NPU_CMD_V1(aclnnSwigluGroupQuant, x, weight, group_index, dst_type_code, quant_mode, block_size,
        round_scale, actualClampLimit, output_origin, y, scale_out, y_origin);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale_out, y_origin);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant_meta(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& group_index,
    at::ScalarType dst_type = at::ScalarType::Float8_e4m3fn,
    int64_t quant_mode = 0,
    int64_t block_size = 0,
    bool round_scale = false,
    const c10::optional<double>& clamp_limit = c10::nullopt,
    bool output_origin = false)
{
    int64_t dst_type_code = get_type_code(dst_type);

    // construct the output tensor
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale_out = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale_out, y_origin);
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_swiglu_group_quant", &custom::npu_swiglu_group_quant_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_swiglu_group_quant", &custom::npu_swiglu_group_quant_meta);
}
