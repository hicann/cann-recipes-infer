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
#include <torch/version.h>
#include "ops_common.h"

// torch only exposes the packed fp4 dtype (Float4_e2m1fn_x2 / kFloat4_e2m1fn_x2) from 2.8 onwards.
// On older torch the fp4 quant path cannot construct an fp4 output tensor (compile-time disabled here).
#if (TORCH_VERSION_MAJOR > 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 8)
#define SWIGLU_HAS_FP4_DTYPE 1
#else
#define SWIGLU_HAS_FP4_DTYPE 0
#endif

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
// ge::DataType integer codes used by the host op (aclnn dst_type attr)
const int64_t DTYPE_CODE_FLOAT8_E5M2 = 35;
const int64_t DTYPE_CODE_FLOAT8_E4M3FN = 36;
const int64_t DTYPE_CODE_FLOAT4_E2M1 = 40;
const int64_t DTYPE_CODE_FLOAT4_E1M2 = 41;

// 工具函数，获取dst_type对应的整数表示
int64_t get_type_code(at::ScalarType dst_type)
{
    int64_t dst_type_code = 35;
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            dst_type_code = DTYPE_CODE_FLOAT8_E5M2;
            break;
        case at::ScalarType::Float8_e4m3fn:
            dst_type_code = DTYPE_CODE_FLOAT8_E4M3FN;
            break;
#if SWIGLU_HAS_FP4_DTYPE
        case at::ScalarType::Float4_e2m1fn_x2:
            // torch only exposes a packed e2m1 fp4 dtype (torch>=2.8). FLOAT4_E1M2 has no torch
            // ScalarType and is therefore not selectable from PyTorch (it stays available only
            // through the aclnn/op-def integer dst_type path), falling through to the default below.
            dst_type_code = DTYPE_CODE_FLOAT4_E2M1;
            break;
#endif
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

inline bool IsFp4Code(int64_t dstTypeCode)
{
    return dstTypeCode == DTYPE_CODE_FLOAT4_E2M1 || dstTypeCode == DTYPE_CODE_FLOAT4_E1M2;
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

    // Divide the last dimension by 2 (swiglu halves the feature dim). y_last_dim counts output elements.
    if (!y_size.empty()) {
        y_size.back() = y_size.back() / SWIGLU_FACTOR;
    }
    int64_t y_last_dim = y_size.back();
    bool isFp4 = IsFp4Code(dst_type);

    // scale_out shape is derived from the number of output elements (y_last_dim) and is identical
    // for fp8-mx and fp4-mx (one e8m0 scale per 32 elements).
    at::SmallVector<int64_t, SIZE> scale_out_size(y_size.begin(), y_size.end());
    if (quant_mode == BLOCK_QUANT) {
        int64_t scaleOutLastDim = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
        scale_out_size.back() = scaleOutLastDim;
    } else if (quant_mode == MX_QUANT) {
        int64_t scaleOutLastDim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_out_size.back() = scaleOutLastDim;
        // 额外地，mx 量化需要将最后一维 reshape 为 (-1, 2)
        scaleOutLastDim = (scaleOutLastDim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_out_size.back() = scaleOutLastDim;
        scale_out_size.push_back(static_cast<int64_t>(MX_SCALE_ALIGN_FACTOR));
    }
    auto scale_out_type = (quant_mode == MX_QUANT) ? at::kFloat8_e8m0fnu : at::kFloat;
    at::Tensor scale_out = at::empty(scale_out_size, x.options().dtype(scale_out_type));

    // y tensor: fp8 stores one byte per element; fp4 packs two elements per byte (e2m1),
    // so the packed last dim is y_last_dim / 2.
    at::Tensor y;
    if (isFp4) {
#if SWIGLU_HAS_FP4_DTYPE
        at::SmallVector<int64_t, SIZE> y_fp4_size(y_size.begin(), y_size.end());
        y_fp4_size.back() = y_last_dim / SWIGLU_FACTOR;
        y = at::empty(y_fp4_size, x.options().dtype(at::kFloat4_e2m1fn_x2));
#else
        TORCH_CHECK(false, "fp4 output requires torch>=2.8 with the Float4_e2m1fn_x2 dtype");
#endif
    } else {
        auto y_dtype = dst_type == DTYPE_CODE_FLOAT8_E5M2 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
        y = at::empty(y_size, x.options().dtype(y_dtype));
    }

    // y_origin keeps input dtype; fp4 has no meaningful y_origin but a placeholder keeps the 3-output signature.
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
    // dst_type is selected purely through the standard ScalarType; fp4 (FLOAT4_E2M1) is reachable
    // only on torch>=2.8 via at::ScalarType::Float4_e2m1fn_x2 (see get_type_code).
    int64_t dst_type_code = get_type_code(dst_type);

    // construct the output tensor
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale_out = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    // -inf means user did not pass clamp_limit
    double actualClampLimit = clamp_limit.has_value() ? clamp_limit.value()
                                                        : -std::numeric_limits<double>::infinity();

    // y already carries its acl dtype natively: fp8 via its ScalarType, and (torch>=2.8) fp4 via the
    // packed at::kFloat4_e2m1fn_x2 dtype which EXEC_NPU_CMD maps to ACL_FLOAT4_E2M1. No TensorWrapper
    // override is required. The kernel/tiling are still driven by the dst_type attr (dst_type_code).
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
