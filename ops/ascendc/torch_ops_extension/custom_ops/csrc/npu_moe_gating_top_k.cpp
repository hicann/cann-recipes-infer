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

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_moe_gating_top_k_output_tensor(
    const at::Tensor& x, 
    int64_t k, const c10::optional<at::Tensor>& bias,int64_t kGroup, int64_t groupCount, bool outFlag)
{
    // Check input tensor validity
    TORCH_CHECK(x.sym_numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(k > 0, "k should be greater than 0.");
    TORCH_CHECK(kGroup > 0, "kGroup should be greater than 0.");
    TORCH_CHECK(groupCount > 0, "groupCount should be greater than 0.");
    TORCH_CHECK(k <= x.sym_size(-1) / groupCount * kGroup, "k should be <= x_shape[-1] / groupCount * kGroup.");
    TORCH_CHECK(kGroup <= groupCount, "kGroup should be <= groupCount.");

    // Get input tensor options
    auto options = x.options();

    // 使用 sym_size/empty_symint 保留动态维（如 token 维），避免动态 shape 下
    // .sizes() 将符号维具象化；维度个数用 x.dim() 而非 x.sizes().size()。
    // Construct yOut output tensor: same shape as x but last dimension is k
    c10::SymDimVector yOut_shape;
    for (int64_t i = 0; i < x.dim() - 1; i++) {
        yOut_shape.push_back(x.sym_size(i));
    }
    yOut_shape.push_back(c10::SymInt(k));
    at::Tensor yOut = at::empty_symint(yOut_shape, options.dtype(x.dtype()));

    // Construct expertIdxOut output tensor: same shape as x but last dimension is k, dtype int32
    c10::SymDimVector expertIdxOut_shape;
    for (int64_t i = 0; i < x.dim() - 1; i++) {
        expertIdxOut_shape.push_back(x.sym_size(i));
    }
    expertIdxOut_shape.push_back(c10::SymInt(k));
    at::Tensor expertIdxOut = at::empty_symint(expertIdxOut_shape, options.dtype(at::kInt));

    return std::make_tuple(yOut, expertIdxOut);
}

// step2, 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_gating_top_k_npu(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    int64_t kGroup,
    int64_t groupCount,
    double routedScalingFactor,
    double eps,
    int64_t groupSelectMode,
    int64_t renorm,
    int64_t normType,
    bool outFlag)
{
    TORCH_CHECK(x.sym_numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(k > 0, "k should be greater than 0.");
    TORCH_CHECK(kGroup > 0, "kGroup should be greater than 0.");
    TORCH_CHECK(groupCount > 0, "groupCount should be greater than 0.");
    TORCH_CHECK(k <= x.sym_size(-1) / groupCount * kGroup, "k should be <= x_shape[-1] / groupCount * kGroup.");
    TORCH_CHECK(kGroup <= groupCount, "kGroup should be <= groupCount.");
    TORCH_CHECK(groupSelectMode == 0 || groupSelectMode == 1, "groupSelectMode should be 0 or 1.");
    TORCH_CHECK(normType == 0 || normType == 1 ||  normType == 2, "normType should be 0, 1 or 2.");
    TORCH_CHECK(renorm == 0, "renorm only supports 0.");

    // Check bias tensor if provided
    if (bias.has_value()) {
        const auto& bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.sym_numel() > 0, "Bias tensor should not be empty.");
        TORCH_CHECK(bias_tensor.dim() == 1, "Bias tensor should be 1-dimensional.");
        TORCH_CHECK(bias_tensor.sym_size(0) == x.sym_size(-1),
            "Bias tensor size should match the last dimension of x. Expected: ",
            x.sym_size(-1), ", Got: ", bias_tensor.sym_size(0));
    }

    if (input_ids.has_value()) {
        const auto& input_ids_tensor = input_ids.value();
        TORCH_CHECK(input_ids_tensor.sym_numel() > 0, "input_ids tensor should not be empty when not null.");
    }
    
    if (tid2eid.has_value()) {
        const auto& tid2eid_tensor = tid2eid.value();
        TORCH_CHECK(tid2eid_tensor.sym_numel() > 0, "tid2eid tensor should not be empty when not null.");
    }

    // Check input data types
    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");

    // construct the output tensors
    at::Tensor yOut;
    at::Tensor expertIdxOut;
    at::Tensor normOut;
    std::tie(yOut, expertIdxOut) = construct_moe_gating_top_k_output_tensor(x, k, bias, kGroup, groupCount, outFlag);

    // Create normOut tensor if outFlag is true
    c10::SymDimVector normOut_shape;
    for (int64_t i = 0; i < x.dim(); i++) {
        normOut_shape.push_back(x.sym_size(i));
    }
    normOut = at::empty_symint(normOut_shape, x.options().dtype(at::kFloat));

    // Execute the NPU operation
    EXEC_NPU_CMD_V1(aclnnMoeGatingTopKHash, x, bias, input_ids, tid2eid, k, kGroup, groupCount, groupSelectMode, renorm, normType,
                    outFlag, routedScalingFactor, eps,  yOut, expertIdxOut, normOut);
    

    return std::make_tuple(yOut, expertIdxOut, normOut);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_gating_top_k_meta(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& input_ids,
    const c10::optional<at::Tensor>& tid2eid,
    int64_t kGroup,
    int64_t groupCount,
    double routedScalingFactor,
    double eps,
    int64_t groupSelectMode,
    int64_t renorm,
    int64_t normType,
    bool outFlag)
{
    TORCH_CHECK(x.sym_numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(k > 0, "k should be greater than 0.");
    TORCH_CHECK(kGroup > 0, "kGroup should be greater than 0.");
    TORCH_CHECK(groupCount > 0, "groupCount should be greater than 0.");
    TORCH_CHECK(k <= x.sym_size(-1) / groupCount * kGroup, "k should be <= x_shape[-1] / groupCount * kGroup.");
    TORCH_CHECK(kGroup <= groupCount, "kGroup should be <= groupCount.");

    // construct the output tensors
    at::Tensor yOut;
    at::Tensor expertIdxOut;
    at::Tensor normOut;
    std::tie(yOut, expertIdxOut) = construct_moe_gating_top_k_output_tensor(x, k, bias, kGroup, groupCount, outFlag);

    // Create normOut tensor if outFlag is true
    c10::SymDimVector normOut_shape;
    for (int64_t i = 0; i < x.dim(); i++) {
        normOut_shape.push_back(x.sym_size(i));
    }
    normOut = at::empty_symint(normOut_shape, x.options().dtype(at::kFloat));
    return std::make_tuple(yOut, expertIdxOut, normOut);
}

}  // namespace custom

// step5, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_moe_gating_top_k", &custom::npu_moe_gating_top_k_npu);
}

// step6, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_moe_gating_top_k", &custom::npu_moe_gating_top_k_meta);
}