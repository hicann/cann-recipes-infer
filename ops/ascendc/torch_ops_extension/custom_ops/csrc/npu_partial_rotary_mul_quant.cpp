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

namespace custom
{
    using namespace at_npu::native;
    const int BSND_DIM_NUM = 4;
    static const std::unordered_map<std::string, int> mode_map = {
        {"half", 0},
        {"interleave", 1},
        {"quarter", 2},
        {"interleave-half", 3}
    };

    // NPU 前向实现 — 输出是新 tensor y (hifloat8/uint8)
    at::Tensor partial_rotary_mul_quant_npu(
        const at::Tensor &x,
        const at::Tensor &r1,
        const at::Tensor &r2,
        c10::string_view rotary_mode,
        at::IntArrayRef partial_slice,
        double scale = 1.0)
    {
        std::string rotaryModeStr = std::string(rotary_mode);
        auto it = mode_map.find(rotaryModeStr);
        TORCH_CHECK(it != mode_map.end(), "rotary_mode must be half/interleave/quarter/interleave-half");

        auto origin_dim_num = x.dim();
        TORCH_CHECK(origin_dim_num == BSND_DIM_NUM,
            "Input tensor x's dim num should be 4, actual ", origin_dim_num, ".");

        auto y = at::empty(x.sizes(), x.options().dtype(at::kByte));

        EXEC_NPU_CMD_V1(aclnnPartialRotaryMulQuant, x, r1, r2, it->second, partial_slice, scale, y);
        return y;
    }

    // META 前向实现
    at::Tensor partial_rotary_mul_quant_meta(
        const at::Tensor &x,
        const at::Tensor &r1,
        const at::Tensor &r2,
        c10::string_view rotary_mode,
        at::IntArrayRef partial_slice,
        double scale = 1.0)
    {
        auto origin_dim_num = x.dim();
        TORCH_CHECK(origin_dim_num == BSND_DIM_NUM, "Input tensor x's dim num should be 4, actual ",
            origin_dim_num, ".");
        return at::empty(x.sizes(), x.options().dtype(at::kByte));
    }
}

// NPU 设备注册
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("partial_rotary_mul_quant", &custom::partial_rotary_mul_quant_npu);
}

// META 设备注册
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("partial_rotary_mul_quant", &custom::partial_rotary_mul_quant_meta);
}