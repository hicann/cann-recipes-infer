/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>

#include "acl/acl.h"

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "op_kernel/recurrent_gated_delta_rule.h"

namespace npu_ops_transformer_ext {
namespace RecurrentGatedDeltaRule {

using namespace AscendC;

extern "C" __global__ __aicore__ void

kernel_recurrent_gated_delta_rule(GM_ADDR mixqkv, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices,
                           GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut,
                           uint32_t b, uint32_t s, uint32_t nk, uint32_t dk, uint32_t nv, uint32_t dv,
                           bool hasAcceptedTokens, bool hasGama, uint32_t vStep, uint32_t ubRestBytes, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    RGDR<bfloat16_t, bfloat16_t> op(
        b, s, nk, dk, nv, dv,
        hasAcceptedTokens, hasGama, vStep, ubRestBytes, scale
    );
    RGDRInitParams initParams{mixqkv, g, beta, state, cuSeqlens, ssmStateIndices, numAcceptedTokens,
                              out, stateOut};
    op.Init(initParams, &pipe);
    op.Process();
}


at::Tensor recurrent_gated_delta_rule(at::Tensor &mix_qkv, at::Tensor &state, at::Tensor &beta,
                                      double scale, at::Tensor &actual_seq_lengths, at::Tensor &ssm_state_indices,
                                      int64_t nk, int64_t nv,
                                      c10::optional<at::Tensor> num_accepted_tokens_opt,
                                      c10::optional<at::Tensor> g_opt,
                                      c10::optional<at::Tensor> gk_opt)
{
    uint64_t ubSize{0UL};
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint32_t coreNum = ascendcPlatform->GetCoreNum();

    int devidx = mix_qkv.device().index();
    c10_npu::set_device(devidx);

    int b = mix_qkv.size(0);
    int s = mix_qkv.size(1);
    int t = b * s;

    int state0 = state.size(0);
    int dv = state.size(2);
    int dk = state.size(3);

    // =================================Calculate the size of UB===================================
    const int64_t MAX_MTP = 8;
    const int64_t ALIGN_SIZE = 16;

    auto ceilAlign = [](int64_t value, int64_t align) {
        return (value + align - 1) & ~(align - 1);
    };

    auto ceilDiv = [](int64_t dividend, int64_t divisor) {
        return (dividend + divisor - 1) / divisor;
    };

    int64_t aNv = ceilAlign(nv, ALIGN_SIZE);
    int64_t aDv = ceilAlign(dv, ALIGN_SIZE);
    int64_t aDk = ceilAlign(dk, ALIGN_SIZE);
    int64_t usedUbBytes = MAX_MTP * (4 * aDk + 2 * aDv);  // 4 for qInQueue_ & kInQueue_, 2 for vInQueue_
    usedUbBytes += 128;                                   // reserve 128 Bytes
    usedUbBytes += MAX_MTP * (4 * aNv + 2 * aNv);
    int64_t ubRestBytes = ubSize - usedUbBytes;
    usedUbBytes += MAX_MTP * (8 * aDk + 4 * aDv + 4 * aNv);
    int64_t coeff = (2 + 2) * aDk + 4;
    coeff += (4 + 4) * aDk + 4 + 4;
    int64_t vStep = (ubSize - usedUbBytes) / coeff / 8 * 8;

    if (vStep < 8) {
        std::cerr << "ERROR: vStep should be bigger than 8, shape is too big" << std::endl;
        std::cerr << "  Calculated vStep: " << vStep << std::endl;
        std::cerr << "  Required min vStep: 8" << std::endl;
        TORCH_CHECK(false, "vStep should be bigger than 8, shape is too big");
    }
    int64_t rptime = ceilDiv(dv, static_cast<uint32_t>(vStep));

    vStep = ceilAlign(ceilDiv(dv, static_cast<uint32_t>(rptime)), 8);
    ubRestBytes -= ((2 + 2) * aDk + 4) * vStep;

    bool hasAcceptedTokens = false;
    void* numAcceptedTokensPtr = nullptr;
    at::Tensor num_accepted_tokens_int32;
    if (num_accepted_tokens_opt.has_value() && num_accepted_tokens_opt.value().defined()) {
        hasAcceptedTokens = true;
        num_accepted_tokens_int32 = num_accepted_tokens_opt.value().to(at::kInt).contiguous();
        numAcceptedTokensPtr = (void*)num_accepted_tokens_int32.storage().data();
    }

    bool hasGama = false;
    at::Tensor g_tensor;
    if (g_opt.has_value() && g_opt.value().defined()) {
        hasGama = true;
        g_tensor = g_opt.value().to(at::kFloat).contiguous();
    } else {
        g_tensor = torch::zeros({t, nv}, torch::TensorOptions().dtype(at::kFloat).device(mix_qkv.device()));
    }

    void* gkPtr = nullptr;
    at::Tensor gk_local;
    if (gk_opt.has_value() && gk_opt.value().defined()) {
        gk_local = gk_opt.value().to(at::kFloat).contiguous();
        gkPtr = (void*)gk_local.storage().data();
    }

    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    at::Tensor output = torch::empty({b, s, nv, dv}, mix_qkv.options());

    // === Kernel launch ===
    auto aclCall = [=]() -> int {
        void* gPtr = g_tensor.defined() ? (void*)g_tensor.storage().data() : nullptr;
        kernel_recurrent_gated_delta_rule<<<coreNum, nullptr, aclstream>>>(
            (GM_ADDR)(mix_qkv.storage().data()),
            (GM_ADDR)(beta.storage().data()),
            (GM_ADDR)(state.data_ptr()),
            (GM_ADDR)(actual_seq_lengths.storage().data()),
            (GM_ADDR)(ssm_state_indices.storage().data()),
            (GM_ADDR)gPtr,
            (GM_ADDR)gkPtr,
            (GM_ADDR)numAcceptedTokensPtr,
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(state.data_ptr()),
            b, s, nk, dk, nv, dv, hasAcceptedTokens, hasGama, vStep, ubRestBytes, scale
        );
        return 0;
    };

    // Register and execute the operator
    at_npu::native::OpCommand::RunOpApiV2("RecurrentGatedDeltaRule", aclCall);

    return output;
}

torch::Tensor recurrent_gated_delta_rule_meta(at::Tensor &mix_qkv, at::Tensor &state, at::Tensor &beta, double scale,
                                              at::Tensor &actual_seq_lengths, at::Tensor &ssm_state_indices,
                                              int64_t nk, int64_t nv,
                                              c10::optional<at::Tensor> num_accepted_tokens_opt,
                                              c10::optional<at::Tensor> g_opt,
                                              c10::optional<at::Tensor> gk_opt)
{
    TORCH_CHECK(mix_qkv.defined(), "MixQKV tensor must be defined");
    TORCH_CHECK(state.defined(), "State tensor must be defined");

    // Check tensor dimensions
    TORCH_CHECK(mix_qkv.dim() == 3, "MixQKV must be 3-dimensional (B, S, D)");
    TORCH_CHECK(state.dim() == 4, "State must be 4-dimensional (N, nv, dv, dk)");

    // Dimension validation
    int64_t b = mix_qkv.size(0);
    int64_t s = mix_qkv.size(1);
    int64_t d = mix_qkv.size(2);
    int64_t dv = state.size(2);
    int64_t dk = state.size(3);

    // Validate mix_qkv width D
    int64_t expectedD = 2 * nk * dk + nv * dv;
    TORCH_CHECK(d == expectedD,
        "mix_qkv width mismatch. Expected: " + std::to_string(expectedD) +
        ", Got: " + std::to_string(d) +
        ". Formula: D = nv*dv + 2*nk*dk, where nv=" + std::to_string(nv) +
        ", dv(or State.size(3))=" + std::to_string(dv) + ", nk=" + std::to_string(nk) +
        ", dk(or State.size(4))=" + std::to_string(dk));

    TORCH_CHECK(state.size(1) == nv, "State third dimension must match nv");

    TORCH_CHECK(beta.dim() == 3, "Beta must be 3-dimensional (B, S, nv)");
    TORCH_CHECK(beta.size(0) == b, "Beta batch size must match MixQKV");
    TORCH_CHECK(beta.size(1) == s, "Beta sequence length must match MixQKV");
    TORCH_CHECK(beta.size(2) == nv, "Beta last dimension must equal nv");

    TORCH_CHECK(ssm_state_indices.dim() == 2, "ssm_state_indices must be 2-dimensional (B, S)");
    TORCH_CHECK(ssm_state_indices.size(0) == b, "ssm_state_indices batch size must match MixQKV");
    TORCH_CHECK(ssm_state_indices.size(1) == s, "ssm_state_indices sequence length must match MixQKV");

    if (g_opt.has_value() && g_opt.value().defined()) {
        auto g = g_opt.value();
        TORCH_CHECK(g.dim() == 3, "g must be 3-dimensional (B, S, nv)");
        TORCH_CHECK(g.size(0) == b && g.size(1) == s && g.size(2) == nv,
            "g shape must be (B, S, nv)");
    }

    if (num_accepted_tokens_opt.has_value() && num_accepted_tokens_opt.value().defined()) {
        auto num_accepted_tokens = num_accepted_tokens_opt.value();
        TORCH_CHECK(num_accepted_tokens.dim() == 1, "num_accepted_tokens must be 1-dimensional (B)");
        TORCH_CHECK(num_accepted_tokens.size(0) == b, "num_accepted_tokens size must match batch size");
    }

    return torch::zeros({b, s, nv, dv}, mix_qkv.options());
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("recurrent_gated_delta_rule", recurrent_gated_delta_rule);
}

// Register Meta Function for Mambav2Rmsnormgated
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("recurrent_gated_delta_rule", TORCH_FN(recurrent_gated_delta_rule_meta));
}

} // namespace RecurrentGatedDeltaRule
} // namespace npu_ops_transformer_ext