/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_situ_mx_quant_tiling.h
 * \brief
 */

#ifndef GROUPED_SITU_MX_QUANT_TILING_H
#define GROUPED_SITU_MX_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupedSituMxQuantTilingData)
    // --- SiTU activation fields (mirror situ_and_mul_sparse) ---
    TILING_DATA_FIELD_DEF(int64_t, totalElements);        // total activation elements = totalRows * EPR
    TILING_DATA_FIELD_DEF(int64_t, outputElementsPerRow); // EPR = x_last / 2
    TILING_DATA_FIELD_DEF(int64_t, expertTokenCount);     // number of experts (expert_tokens length)
    TILING_DATA_FIELD_DEF(int64_t, chunkElements);        // min(EPR, MAX_TILE_COLS) = whole-row chunk
    TILING_DATA_FIELD_DEF(float, beta);
    TILING_DATA_FIELD_DEF(float, alpha);
    TILING_DATA_FIELD_DEF(int32_t, highPrecision);
    // --- MXFP8 quantization fields ---
    TILING_DATA_FIELD_DEF(int32_t, blockSize);            // group_size, fixed 32
    TILING_DATA_FIELD_DEF(int64_t, scaleColNum);          // e8m0 scales per row = ceil(EPR/32) rounded up to even
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupedSituMxQuant, GroupedSituMxQuantTilingData)

// Compile-time cached platform info, so tiling still works when the SuperKernel
// JIT / tiling-sink path calls TilingFor* with a null runtime PlatformInfo.
struct GroupedSituMxQuantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
};
}  // namespace optiling

#endif  // GROUPED_SITU_MX_QUANT_TILING_H
