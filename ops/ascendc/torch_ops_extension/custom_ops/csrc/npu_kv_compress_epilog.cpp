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

// Helper function for input validation
void validate_kv_compress_epilog_inputs(
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_compress_cache)
{
    // Validate x is 2D
    TORCH_CHECK(x.dim() == 2,
        "x must be 2D tensor, but got dimensions: ", x.dim());

    // Validate x dimensions are positive
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0,
        "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1), "]");

    // Validate slot_mapping is 1D
    TORCH_CHECK(slot_mapping.dim() == 1,
        "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());

    // Validate slot_mapping size matches x's first dimension
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
        "slot_mapping size must equal x's first dimension, "
        "but got slot_mapping_size=", slot_mapping.size(0),
        ", x.dim(0)=", x.size(0));

    // Dtype validation
    TORCH_CHECK(x.dtype() == at::kBFloat16,
        "x must be BF16, but got ", x.dtype());
    TORCH_CHECK(slot_mapping.dtype() == at::kInt || slot_mapping.dtype() == at::kLong,
        "slot_mapping must be INT32 or INT64, but got ", slot_mapping.dtype());
    TORCH_CHECK(kv_compress_cache.dtype() == at::ScalarType::Float8_e5m2 ||
                kv_compress_cache.dtype() == at::ScalarType::Float8_e4m3fn,
        "kv_compress_cache must be FP8_E5M2 or FP8_E4M3, but got ", kv_compress_cache.dtype());
}

// step1, 为NPU设备实现前向接口 (In-place版本)
void kv_compress_epilog_npu(
    at::Tensor& kv_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_group_size, 
    int64_t quant_mode, 
    bool round_scale_flag)
{
    validate_kv_compress_epilog_inputs(x, slot_mapping, kv_compress_cache);

    int64_t round_scale = 0;
    if (round_scale_flag) {
       round_scale = 1;
    }

    // 调用NPU算子
    EXEC_NPU_CMD_V1(aclnnKvCompressEpilog,
                    kv_compress_cache, x, slot_mapping, quant_group_size, quant_mode, round_scale);

}

// step2, 为META设备实现前向接口 (In-place版本)
void kv_compress_epilog_meta(
    at::Tensor& kv_compress_cache,
    const at::Tensor& x,
    const at::Tensor& slot_mapping,
    int64_t quant_group_size, 
    int64_t quant_mode, 
    bool round_scale_flag)
{
    validate_kv_compress_epilog_inputs(x, slot_mapping, kv_compress_cache);

    // META设备不执行，直接返回
    return;
}

}

// step5, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("kv_compress_epilog", &custom::kv_compress_epilog_npu);
}

// step6, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("kv_compress_epilog", &custom::kv_compress_epilog_meta);
}
