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
at::Tensor construct_lightning_indexer_pto_output_tensor(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, 
    const c10::optional<at::Tensor> &actual_seq_lengths_key, const c10::optional<at::Tensor> &block_table)
{
    at::SmallVector<int64_t, SIZE> output_size;
    int sparse_count = 2048;
    // BSND
    output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    at::Tensor output = at::empty(output_size, query.options().dtype(at::kInt));

    return output;
}

// step3, 为META设备实现前向接口
at::Tensor npu_lightning_indexer_pto_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    // input_list = [query, key, weights, actual_seq_lengths_key, block_table]
    return construct_lightning_indexer_pto_output_tensor(query, key, weights, actual_seq_lengths_key, block_table);
}
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom_pypto, Meta, m) {
    m.impl("npu_lightning_indexer_pto", &custom_pypto::npu_lightning_indexer_pto_meta);
}
