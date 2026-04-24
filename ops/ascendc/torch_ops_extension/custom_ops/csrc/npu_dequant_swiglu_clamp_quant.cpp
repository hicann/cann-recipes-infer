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

// step1, 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_dequant_swiglu_clamp_quant_output_tensor(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode,
    int64_t resolved_dst_type,
    int64_t resolved_round_mode,
    int64_t resolved_activate_dim,
    int64_t swiglu_mode,
    double clamp_limit,
    double glu_alpha,
    double glu_bias)
{
    int64_t select_dim = resolved_activate_dim >= 0 ? resolved_activate_dim : resolved_activate_dim + x.dim();

    // 维度检查
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1");
    TORCH_CHECK(select_dim >= 0 && select_dim < x.dim(), "activate dim should less than x dim");
    TORCH_CHECK(x.size(select_dim) % 2 == 0, "x last dim should be even");
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1");

    // 构建 y_size 向量
    std::vector<int64_t> y_size;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i == select_dim) {
            y_size.push_back(static_cast<int64_t>(std::floor(x.size(i) / 2.0)));
        } else {
            y_size.push_back(x.size(i));
        }
    }

    std::vector<int64_t> scale_size(y_size.begin(), y_size.end() - 1);

    // 确定输出数据类型
    auto output_dtype = at::kChar;
    if (resolved_dst_type == 23) {
        output_dtype = at::kFloat8_e5m2;
    } else if (resolved_dst_type == 24) {
        output_dtype = at::kFloat8_e4m3fn;
    } else if (resolved_dst_type == 296 || resolved_dst_type == 297) {
        output_dtype = at::kByte;
    }

    // 创建两个空张量
    at::Tensor out1 = at::empty(y_size, x.options().dtype(output_dtype));
    at::Tensor out2 = at::empty(scale_size, x.options().dtype(at::kFloat));

    return std::make_tuple(out1, out2);
}


// step2, 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_clamp_quant_npu(const at::Tensor &x, const c10::optional<at::Tensor> &weight_scale, const c10::optional<at::Tensor> &activation_scale,
                                        const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &quant_scale, const c10::optional<at::Tensor> &quant_offset,
                                        const c10::optional<at::Tensor> &group_index, bool activate_left, int64_t quant_mode, c10::optional<int64_t> dst_type,
                                        c10::optional<int64_t> round_mode, c10::optional<int64_t> activate_dim, int64_t swiglu_mode,
                                        double clamp_limit, double glu_alpha, double glu_bias)
{
    int64_t resolved_dst_type = dst_type.has_value() ? dst_type.value() : 1;
    int64_t resolved_round_mode = round_mode.has_value() ? round_mode.value() : 0;
    int64_t resolved_activate_dim = activate_dim.has_value() ? activate_dim.value() : -1;
    at::Tensor y;
    at::Tensor scale;
    std::tie(y, scale) = construct_dequant_swiglu_clamp_quant_output_tensor(x, weight_scale, activation_scale, bias, quant_scale,
                          quant_offset, group_index, activate_left, quant_mode,
                          resolved_dst_type, resolved_round_mode, resolved_activate_dim, swiglu_mode,
                          clamp_limit, glu_alpha, glu_bias);

    std::string quant_mode_str = "static";
    if (quant_mode == 1) {
        quant_mode_str = "dynamic";
    }

    std::string round_mode_str = "rint";
    if (resolved_round_mode == 1) {
        round_mode_str = "round";
    } else if (resolved_round_mode == 2) {
        round_mode_str = "floor";
    } else if (resolved_round_mode == 3) {
        round_mode_str = "ceil";
    } else if (resolved_round_mode == 4) {
        round_mode_str = "trunc";
    }

    char *quant_mode_res = const_cast<char *>(quant_mode_str.c_str());
    char *round_mode_res = const_cast<char *>(round_mode_str.c_str());

    EXEC_NPU_CMD_V1(aclnnDequantSwigluClampQuant, x, weight_scale, activation_scale, bias, quant_scale,
                          quant_offset, group_index, activate_left, quant_mode_res,
                          resolved_dst_type, round_mode_res, resolved_activate_dim, swiglu_mode,
                          clamp_limit, glu_alpha, glu_bias, y, scale);
    return std::make_tuple(y, scale);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_clamp_quant_meta(const at::Tensor &x, const c10::optional<at::Tensor> &weight_scale, const c10::optional<at::Tensor> &activation_scale,
                                            const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &quant_scale, const c10::optional<at::Tensor> &quant_offset,
                                            const c10::optional<at::Tensor> &group_index, bool activate_left, int64_t quant_mode, c10::optional<int64_t> dst_type,
                                            c10::optional<int64_t> round_mode, c10::optional<int64_t> activate_dim, int64_t swiglu_mode,
                                            double clamp_limit, double glu_alpha, double glu_bias)
{
    int64_t resolved_dst_type = dst_type.has_value() ? dst_type.value() : 1;
    int64_t resolved_round_mode = round_mode.has_value() ? round_mode.value() : 0;
    int64_t resolved_activate_dim = activate_dim.has_value() ? activate_dim.value() : -1;
    at::Tensor y;
    at::Tensor scale;
    std::tie(y, scale) = construct_dequant_swiglu_clamp_quant_output_tensor(x, weight_scale, activation_scale, bias, quant_scale,
                          quant_offset, group_index, activate_left, quant_mode,
                          resolved_dst_type, resolved_round_mode, resolved_activate_dim, swiglu_mode,
                          clamp_limit, glu_alpha, glu_bias);
    return std::make_tuple(y, scale);
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_dequant_swiglu_clamp_quant", &custom::npu_dequant_swiglu_clamp_quant_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_dequant_swiglu_clamp_quant", &custom::npu_dequant_swiglu_clamp_quant_meta);
}