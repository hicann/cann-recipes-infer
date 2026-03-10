/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/flopcount/FlopCount.h"
#include "torch_npu/csrc/flopcount/FlopCounter.h"
#include "torch_npu/csrc/custom_dtype/Init.h"
namespace custom_pypto {
using namespace at_npu::native;
const int SIZE = 6;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_npu_lightning_indexer_prolog_pto_output_tensor(
    const at::Tensor& token_x, const at::Tensor& q_norm, const at::Tensor& q_norm_scale, const at::Tensor& wq_b,
    const at::Tensor& wq_b_scale, const at::Tensor& wk, const at::Tensor& weights_proj, const at::Tensor& ln_gamma_k,
    const at::Tensor& ln_beta_k, const at::Tensor& cos_idx_rope, const at::Tensor& sin_idx_rope, const at::Tensor& hadamard_q,
    const at::Tensor& hadamard_k, const at::Tensor& idx_k_cache, const at::Tensor& idx_k_scale_cache,
    const at::Tensor& idx_k_cache_index, double layernorm_epsilon_k,
    c10::optional<c10::string_view> layout_query, c10::optional<c10::string_view> layout_key)
{
    // BSND
    at::SmallVector<int64_t, SIZE> query_size;
    query_size = {token_x.size(DIM_0), weights_proj.size(DIM_1), wk.size(DIM_1)};
    at::SmallVector<int64_t, SIZE> query_scale_size = {token_x.size(DIM_0), weights_proj.size(DIM_1), 1};
    at::SmallVector<int64_t, SIZE> weights_size = {token_x.size(DIM_0), weights_proj.size(DIM_1)};
    at::Tensor output_query = at::empty(query_size, token_x.options().dtype(at::ScalarType::Char));
    at::Tensor output_query_scale = at::empty(query_scale_size, token_x.options().dtype(at::ScalarType::Half));
    at::Tensor output_weights = at::empty(weights_size, token_x.options().dtype(at::ScalarType::Half));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output_query, output_query_scale, output_weights);
}
// step3, 为META设备实现前向接口
std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_lightning_indexer_prolog_pto_meta(
 const at::Tensor& token_x, const at::Tensor& q_norm, const at::Tensor& q_norm_scale, const at::Tensor& wq_b,
    const at::Tensor& wq_b_scale, const at::Tensor& wk, const at::Tensor& weights_proj, const at::Tensor& ln_gamma_k,
    const at::Tensor& ln_beta_k, const at::Tensor& cos_idx_rope, const at::Tensor& sin_idx_rope, const at::Tensor& hadamard_q,
    const at::Tensor& hadamard_k, const at::Tensor& idx_k_cache, const at::Tensor& idx_k_scale_cache,
    const at::Tensor& idx_k_cache_index, double layernorm_epsilon_k,
    c10::optional<c10::string_view> layout_query, c10::optional<c10::string_view> layout_key)
{
    return construct_npu_lightning_indexer_prolog_pto_output_tensor(token_x, q_norm, q_norm_scale, wq_b, wq_b_scale, wk, weights_proj,
        ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k, idx_k_cache, idx_k_scale_cache,
        idx_k_cache_index, layernorm_epsilon_k, layout_query, layout_key);

}
}
TORCH_LIBRARY_IMPL(custom_pypto, Meta, m) {
    m.impl("npu_lightning_indexer_prolog_pto", &custom_pypto::npu_lightning_indexer_prolog_pto_meta);
}