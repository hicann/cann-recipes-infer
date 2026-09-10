/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kv_compress_epilog_v2_kernel.h"
#include "kv_compress_epilog_v2_mxfp4_group16.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_compress_epilog_v2(
    GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping, GM_ADDR cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogV2TilingData, tilingDataValue, tiling);
    const KvCompressEpilogV2TilingData *tilingData = &tilingDataValue;
    TPipe pipe;
    const int64_t overflowMode =
        AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(2000)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE, false> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2001)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE, true> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2MxFp4Group16Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(overflowMode);
}
