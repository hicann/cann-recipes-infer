/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

constexpr int64_t GROUP_SIZE = 32;      // MXFP8 group_size, fixed
constexpr int64_t SCALE_ALIGN = 2;      // mxscale last-axis interleave factor

// y: x with last dim halved, fp8_e4m3fn.
// mxscale: x[..:-1] + [ceil(ceil(EPR/32)/2), 2], e8m0. EPR = x[-1] / 2.
std::tuple<at::Tensor, at::Tensor> construct_grouped_situ_mx_quant_output(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() > 0 && x.size(-1) > 0 && x.size(-1) % 2 == 0,
                "x must have a positive even last dimension");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be FLOAT16 or BFLOAT16");

    auto y_shape = x.sizes().vec();
    y_shape.back() /= 2;
    const int64_t epr = y_shape.back();
    at::Tensor y = at::empty(y_shape, x.options().dtype(at::kFloat8_e4m3fn));

    auto scale_shape = x.sizes().vec();
    const int64_t groupCount = (epr + GROUP_SIZE - 1) / GROUP_SIZE;
    scale_shape.back() = (groupCount + SCALE_ALIGN - 1) / SCALE_ALIGN;
    scale_shape.push_back(SCALE_ALIGN);
    at::Tensor mxscale = at::empty(scale_shape, x.options().dtype(at::kFloat8_e8m0fnu));

    return std::tuple<at::Tensor, at::Tensor>(y, mxscale);
}

std::tuple<at::Tensor, at::Tensor> grouped_situ_mx_quant_npu(
    const at::Tensor& x, const at::Tensor& expert_tokens,
    double beta = 1.0, double alpha = 1.0, bool high_precision = false)
{
    TORCH_CHECK(expert_tokens.scalar_type() == at::kLong, "expert_tokens must be INT64");
    TORCH_CHECK(beta > 0.0 && alpha > 0.0, "beta and alpha must be positive");
    auto outs = construct_grouped_situ_mx_quant_output(x);
    at::Tensor y = std::get<0>(outs);
    at::Tensor mxscale = std::get<1>(outs);
    EXEC_NPU_CMD_V1(aclnnGroupedSituMxQuant, x, expert_tokens, beta, alpha, high_precision, y, mxscale);
    return std::tuple<at::Tensor, at::Tensor>(y, mxscale);
}

std::tuple<at::Tensor, at::Tensor> grouped_situ_mx_quant_meta(
    const at::Tensor& x, const at::Tensor& expert_tokens,
    double beta = 1.0, double alpha = 1.0, bool high_precision = false)
{
    (void)expert_tokens; (void)beta; (void)alpha; (void)high_precision;
    return construct_grouped_situ_mx_quant_output(x);
}
}  // namespace custom

TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("grouped_situ_mx_quant", &custom::grouped_situ_mx_quant_npu);
}

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("grouped_situ_mx_quant", &custom::grouped_situ_mx_quant_meta);
}
