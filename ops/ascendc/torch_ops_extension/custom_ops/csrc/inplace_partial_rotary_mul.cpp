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
    // step2, 为NPU设备实现前向接口
    void inplace_partial_rotary_mul_npu(at::Tensor & x, const at::Tensor &r1, const at::Tensor &r2, c10::string_view rotary_mode, at::IntArrayRef partial_slice)
    {
        std::string rotary_mode_str = std::string(rotary_mode);
        auto it = mode_map.find(rotary_mode_str);
        if (it == mode_map.end())
        {
            return;
        }
        auto origin_dim_num = x.dim();
        TORCH_CHECK(origin_dim_num == BSND_DIM_NUM, "Input tensor x's dim num should be 4, actual ", origin_dim_num, ".");
        EXEC_NPU_CMD_V1(aclnnInplacePartialRotaryMul, x, r1, r2, it->second, partial_slice);
    }

    // step3, 为META设备实现前向接口
    void inplace_partial_rotary_mul_meta(
        at::Tensor &x,
        const at::Tensor &r1,
        const at::Tensor &r2,
        c10::string_view rotary_mode,
        at::IntArrayRef partial_slice)
    {
        auto origin_dim_num = x.dim();
        TORCH_CHECK(origin_dim_num == BSND_DIM_NUM, "Input tensor x's dim num should be 4, actual ", origin_dim_num, ".");
        return;
    }
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)
{
    m.impl("inplace_partial_rotary_mul", &custom::inplace_partial_rotary_mul_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m)
{
    m.impl("inplace_partial_rotary_mul", &custom::inplace_partial_rotary_mul_meta);
}
