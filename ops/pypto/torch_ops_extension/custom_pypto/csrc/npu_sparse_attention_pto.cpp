/* *
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  * See LICENSE in the root of
the software repository for the full text of the License.
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

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// 工具函数，推导输出shape
at::Tensor construct_npu_sparse_attention_pto_output_tensor(
    const at::Tensor &x, const at::Tensor &w_dq, const at::Tensor &w_uq_qr, const at::Tensor &w_uk,
    const at::Tensor &w_dkv_kr, const at::Tensor &gamma_cq, const at::Tensor &gamma_ckv,
    const at::Tensor &sin, const at::Tensor &cos, const at::Tensor &cache_index,
    const at::Tensor &kv_cache, const at::Tensor &kr_cache, const at::Tensor &block_table, const at::Tensor &act_seqs,
    const at::Tensor &w_idx_qb, const at::Tensor &w_idx_k, const at::Tensor &w_idx_proj,
    const at::Tensor &in_gamma_k,const at::Tensor &in_beta_k, const at::Tensor &index_k_cache
)
{
    at::SmallVector<int64_t, SIZE> output_size;
    // BSND
    output_size = {x.size(DIM_0), x.size(DIM_1), w_uk.size(DIM_0), w_uk.size(DIM_2)};
    at::Tensor output = at::empty(output_size, x.options().dtype(x.dtype()));

    return output;
}

// step3, 为META设备实现前向接口
at::Tensor npu_sparse_attention_pto_meta(
    const at::Tensor &x, const at::Tensor &w_dq, const at::Tensor &w_uq_qr, const at::Tensor &w_uk,
    const at::Tensor &w_dkv_kr, const at::Tensor &gamma_cq, const at::Tensor &gamma_ckv,
    const at::Tensor &sin, const at::Tensor &cos, const at::Tensor &cache_index,
    const at::Tensor &kv_cache, const at::Tensor &kr_cache, const at::Tensor &block_table, const at::Tensor &act_seqs,
    const at::Tensor &w_idx_qb, const at::Tensor &w_idx_k, const at::Tensor &w_idx_proj,
    const at::Tensor &in_gamma_k,const at::Tensor &in_beta_k, const at::Tensor &index_k_cache
)
{
    // input_list = [query, key, weights, actual_seq_lengths_key, block_table]
    return construct_npu_sparse_attention_pto_output_tensor(
            x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
            sin, cos, cache_index, kv_cache, kr_cache, block_table, act_seqs,
            w_idx_qb, w_idx_k, w_idx_proj, in_gamma_k, in_beta_k, index_k_cache
    );
}
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom_pypto, Meta, m) {
    m.impl("npu_sparse_attention_pto", &custom_pypto::npu_sparse_attention_pto_meta);
}
