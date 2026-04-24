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
at::Tensor construct_hc_pre_inv_rms_output_tensor(const at::Tensor& x, float epsilon=1e-20)
{
    // Check input tensor validity
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // Get input tensor options
    auto options = x.options();

    // Construct yOut output tensor
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    // The first one or two dimensions of y match those of x, and the last dimension of y is 1.
    // x: (b, s, hc, d) --> y: (b, s, 1)   or   x: (b * s, hc, d) --> y: (b * s, 1)
    for (auto i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

// step2, 为NPU设备实现前向接口
at::Tensor npu_hc_pre_inv_rms_npu(const at::Tensor& x, double epsilon=1e-20)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // Check input data types
    TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf || x.dtype() == at::kBFloat16,
                "x should be FLOAT16, BFLOAT16, or FLOAT32.");

    // construct the output tensors
    at::Tensor yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);

    // Execute the NPU operation
    EXEC_NPU_CMD_V1(aclnnHcPreInvRms, x, epsilon, yOut);
    
    return yOut;
}

// step3, 为META设备实现前向接口
at::Tensor npu_hc_pre_inv_rms_meta(const at::Tensor& x, double epsilon=1e-20)
{
    TORCH_CHECK(x.numel() > 0, "Input tensor x should not be empty.");
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // construct the output tensors
    at::Tensor yOut = construct_hc_pre_inv_rms_output_tensor(x, epsilon);

    return yOut;
}

}  // namespace custom

// step5, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_hc_pre_inv_rms", &custom::npu_hc_pre_inv_rms_npu);
}

// step6, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_hc_pre_inv_rms", &custom::npu_hc_pre_inv_rms_meta);
}