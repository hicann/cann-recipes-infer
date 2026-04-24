/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int HC_LIMIT = 4;
const int D_LIMIT = 4096;
const int D_LIMIT_EXTEND = 7168; // 扩展D类型
const int MIX_HC_LIMIT = 24;
const int SINGLE_OP_MAX_BS = 128;
const int SINGLE_OP_BS_ALIGN_FACTOR = 16;
const int FUSION_BASE_BS = 8192;
const int FUSION_SPLIT_K_MAX_BS = 512;

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor& x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3) {
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

// 工具函数，推导输出hc_pre_inv_rms_shape
at::Tensor construct_hc_pre_rsqrt_output_tensor(const at::Tensor& x, float epsilon=1e-6)
{
    // Check input tensor validity
    TORCH_CHECK(epsilon >= 0, "epsilon should be greater than 0.");

    // Get input tensor options
    auto options = x.options();

    // Construct yOut output tensor
    auto xDims = x.dim();
    c10::SmallVector<int64_t, SIZE> yOut_shape;
    for (size_t i = 0; i < xDims - 2; i++) {
        yOut_shape.push_back(x.sizes()[i]);
    }
    yOut_shape.push_back(1);
    at::Tensor yOut = at::empty(yOut_shape, options.dtype(at::kFloat));

    return yOut;
}

// 工具函数，检查输入shape
void check_hc_pre_shape_and_dtype(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    bool is_ascend950 = false) {
    // check x shape: [b, s, hc, d]
    auto xDims = x.dim();
    TORCH_CHECK(xDims == 4 || xDims == 3, "Input tensor x's dim num should be 4, actual ", xDims, ".");
    for (size_t i = 0; i < 3; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is :", x.size(i), ".");
    }
    auto hc = x.size(1);
    auto d = x.size(2);
    if (xDims == 4) {
        hc = x.size(2);
        d = x.size(3);
    }
    TORCH_CHECK(hc == HC_LIMIT, "The hc of x only support ", HC_LIMIT, ", actual ", hc, ".");
    TORCH_CHECK(
        d == D_LIMIT || d == D_LIMIT_EXTEND,
        "The d of x only support ", D_LIMIT, " or ", D_LIMIT_EXTEND, ", actual ", d, ".");
    // check hc_fn: [mix_hc, hc * d]
    TORCH_CHECK(hc_fn.dim() == 2, "Input tensor hc_fn's dim num should be 2, actual ", hc_fn.dim(), ".");
    auto mix_hc = hc_fn.size(0);
    TORCH_CHECK(mix_hc == MIX_HC_LIMIT, "The mix_hc of hc_fn only support ", MIX_HC_LIMIT, ", actual ", mix_hc, ".");
    TORCH_CHECK(hc_fn.size(1) == hc * d, "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ", hc_fn.size(1), ", hc is ", hc, ", d is ", d, ".");
    // check hc_scale: [3]
    TORCH_CHECK(hc_scale.dim() == 1, "Input tensor hc_scale's dim num should be 1, actual ", hc_scale.dim(), ".");
    TORCH_CHECK(hc_scale.size(0) == 3, "Input tensor hc_scale's shape should be [3], actual [", hc_scale.size(0), "].");
    // check hc_base: [mix_hc]
    TORCH_CHECK(hc_base.dim() == 1, "Input tensor hc_base's dim num should be 1, actual ", hc_base.dim(), ".");
    TORCH_CHECK(hc_base.size(0) == mix_hc, "The hc_base.shape[0] should be mix_hc, actual hc_base.shape[0] is ", hc_base.size(0), ", mix_hc is ", mix_hc, ".");
    // check dtype
    TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
    TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
    TORCH_CHECK(hc_scale.dtype() == at::kFloat, "hc_scale's dtype should be FLOAT32.");
    TORCH_CHECK(hc_base.dtype() == at::kFloat, "hc_base's dtype should be FLOAT32.");
}

// hc_pre 小算子拼接实现
std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_composite(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base, 
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto xDims = x.dim();
    // call hc_pre_inv_rms
    auto rsqrt = construct_hc_pre_rsqrt_output_tensor(x, norm_eps);
    EXEC_NPU_CMD_V1(aclnnHcPreInvRms, x, norm_eps, rsqrt);

    // call matmul -> get mixes
    auto original_type = x.dtype();
    at::Tensor x_float = x.to(at::kFloat);
    at::Tensor x_flattened = x_float.flatten(2, -1);
    if (xDims == 3) {
        x_flattened = x_float.flatten(1, -1);
    }
    auto mixes = at::linear(x_flattened, hc_fn);

    // call hc_pre_sinkhorn
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD_V1(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x, hc_mult, hc_sinkhorn_iters, hc_eps, 
                    y, post, comb_frag);
    y = y.to(original_type);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

// hc_pre 融合算子实现
std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_fusion(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base, 
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
                    
    EXEC_NPU_CMD_V1(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps,
                    y, post, comb_frag);

    y = y.to(x.dtype());

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

// step2 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_npu(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base, 
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    // get soc name
    static const char* socName = aclrtGetSocName();
    static const char* prefix = "Ascend950";
    const bool is_ascend950 = socName != nullptr && std::string(socName).find(prefix) == 0;

    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, is_ascend950);
    // get x shape
    auto xDims = x.dim();
    int32_t bs = x.size(0) * x.size(1);
    int32_t d = x.size(2);
    if (xDims == 3) {
        bs = x.size(0);
        d = x.size(2);
    } else if (xDims == 4) {
        d = x.size(3);
    }
    // 非Ascend950场景，d=7168路由到小算子拼接
    if (!is_ascend950 && d == D_LIMIT_EXTEND) {
        ASCEND_LOGI("For non-Ascend950 with d=%d, implemented hc_pre using a concatenation of small operators.", d);
        return hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
    }
    if (is_ascend950) { // if socVersion is Ascend950
        ASCEND_LOGI("The matched SoC version is Ascend950.");
        if (bs <= FUSION_SPLIT_K_MAX_BS || bs % FUSION_BASE_BS == 0) {
            ASCEND_LOGI("Use fused operator for current bs %d.", bs);
            return hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
        }
        ASCEND_LOGI("For current bs %d, implemented hc_pre using a concatenation of small operators.", bs);
        return hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
    } else {
        if (bs > SINGLE_OP_MAX_BS || bs % SINGLE_OP_BS_ALIGN_FACTOR != 0) {
            ASCEND_LOGI("For current bs %d, implemented hc_pre using a concatenation of small operators.", bs);
            return hc_pre_composite(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
        }
        return hc_pre_fusion(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps);
    }
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_meta(
    const at::Tensor& x, const at::Tensor& hc_fn, const at::Tensor& hc_scale, const at::Tensor& hc_base, 
    int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    // get soc name
    static const char* socName = aclrtGetSocName();
    static const char* prefix = "Ascend950";
    const bool is_ascend950 = socName != nullptr && std::string(socName).find(prefix) == 0;

    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, is_ascend950);
    // construct the output tensor
    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

} // namespace custom

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_hc_pre", &custom::npu_hc_pre_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_hc_pre", &custom::npu_hc_pre_meta);
}
