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
// step2, 为NPU设备实现前向接口
void indexer_compress_epilog_npu(
    at::Tensor& indexer_compress_cache, 
    at::Tensor& indexer_compress_cache_scale, 
    const at::Tensor& x, 
    const at::Tensor& slot_mapping, 
    int64_t quant_mode=1, 
    bool round_scale=true)
{
    EXEC_NPU_CMD_V1(aclnnIndexerCompressEpilog, indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, quant_mode, round_scale);
}

// step3, 为META设备实现前向接口
void indexer_compress_epilog_meta(
    at::Tensor& indexer_compress_cache, 
    at::Tensor& indexer_compress_cache_scale, 
    const at::Tensor& x, 
    const at::Tensor& slot_mapping, 
    int64_t quant_mode=1, 
    bool round_scale=true)
{
    return;
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("indexer_compress_epilog", &custom::indexer_compress_epilog_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("indexer_compress_epilog", &custom::indexer_compress_epilog_meta);
}
