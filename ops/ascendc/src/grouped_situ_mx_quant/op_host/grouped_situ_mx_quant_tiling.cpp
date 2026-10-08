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
 * \file grouped_situ_mx_quant_tiling.cpp
 * \brief
 */

#include "grouped_situ_mx_quant_tiling.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
// Whole-row chunk: min(EPR, 3072). With EPR=3072 every row is a single chunk,
// and 3072 = 96 * 32 keeps the quant group (32) boundaries aligned to the chunk.
constexpr int64_t MAX_TILE_COLS = 3072;
constexpr int64_t QUANT_BLOCK_SIZE = 32;
// Per activation element (one gate/up column) UB budget, T is 2 bytes:
//   gate+up in-queue (double buf): 2*2*2 = 8
//   activation staging (bf16/fp16): 2
//   y fp8 out-queue (double buf): 1*2 = 2
//   maxExp/recipScale/scale per group amortized: < 1
// -> ~13 B/elem, well under UB. Reserve 16 for headroom.
constexpr int64_t BYTES_PER_ELEMENT = 16;

ge::graphStatus TilingForGroupedSituMxQuant(gert::TilingContext* context)
{
    const auto* shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
    const auto& storage = shape->GetStorageShape();
    const int64_t rank = storage.GetDimNum();
    OPS_ERR_IF(rank < 1 || storage.GetDim(rank - 1) <= 0 || storage.GetDim(rank - 1) % 2 != 0,
        OPS_LOG_E(context->GetNodeName(), "x last dimension must be positive and even."),
        return ge::GRAPH_FAILED);
    const int64_t outputElementsPerRow = storage.GetDim(rank - 1) / 2;
    const int64_t totalElements = storage.GetShapeSize() / 2;
    const int64_t totalRows = totalElements / outputElementsPerRow;

    const auto* tokenShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context, tokenShape, return ge::GRAPH_FAILED);
    const int64_t expertTokenCount = tokenShape->GetStorageShape().GetShapeSize();

    // Prefer the runtime PlatformInfo; fall back to the TilingParse-cached
    // CompileInfo (SuperKernel JIT / tiling-sink can call this with null info).
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        coreNum = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        const auto* compileInfo = context->GetCompileInfo<GroupedSituMxQuantCompileInfo>();
        OPS_LOG_E_IF_NULL(context, compileInfo, return ge::GRAPH_FAILED);
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
    }
    OPS_ERR_IF(coreNum == 0, OPS_LOG_E(context->GetNodeName(), "AIV core number must be positive."),
        return ge::GRAPH_FAILED);
    const int64_t blockDim = std::min(static_cast<int64_t>(coreNum), totalRows);

    const auto* attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const float* beta = attrs->GetFloat(0);
    const float* alpha = attrs->GetFloat(1);
    const bool* highPrecision = attrs->GetBool(2);
    OPS_ERR_IF(beta == nullptr || alpha == nullptr || *beta <= 0.0f || *alpha <= 0.0f,
        OPS_LOG_E(context->GetNodeName(), "beta and alpha must be positive."),
        return ge::GRAPH_FAILED);

    const int64_t chunkElements = std::min(outputElementsPerRow, MAX_TILE_COLS);

    // e8m0 scales per row: ceil(EPR/32) groups, rounded up to even (interleaved layout).
    const int64_t groupCount = (outputElementsPerRow + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    const int64_t scaleColNum = (groupCount + 1) / 2 * 2;

    OPS_ERR_IF(ubSize > 0 && chunkElements * BYTES_PER_ELEMENT > static_cast<int64_t>(ubSize * 7 / 8),
        OPS_LOG_E(context->GetNodeName(), "chunk UB budget exceeds available UB."),
        return ge::GRAPH_FAILED);

    GroupedSituMxQuantTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_outputElementsPerRow(outputElementsPerRow);
    tiling.set_expertTokenCount(expertTokenCount);
    tiling.set_chunkElements(chunkElements);
    tiling.set_beta(*beta);
    tiling.set_alpha(*alpha);
    tiling.set_highPrecision(highPrecision != nullptr && *highPrecision ? 1 : 0);
    tiling.set_blockSize(QUANT_BLOCK_SIZE);
    tiling.set_scaleColNum(scaleColNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platformInfo != nullptr
        ? platform_ascendc::PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize()
        : 0;
    return ge::GRAPH_SUCCESS;
}

// TilingParse: cache platform info at compile time for the SuperKernel / GE JIT
// path, so TilingFor* has coreNum/ubSize even when PlatformInfo is null at run.
ge::graphStatus TilingPrepareForGroupedSituMxQuant(gert::TilingParseContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GroupedSituMxQuant", "Tiling parse context is null"),
        return ge::GRAPH_FAILED);
    auto* compileInfo = context->GetCompiledInfo<GroupedSituMxQuantCompileInfo>();
    OPS_ERR_IF(compileInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GroupedSituMxQuant", "Compile info is null"),
        return ge::GRAPH_FAILED);
    auto* platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GroupedSituMxQuant", "Tiling parse platform info is null"),
        return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->socVersion = platform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(GroupedSituMxQuant)
    .Tiling(TilingForGroupedSituMxQuant)
    .TilingParse<GroupedSituMxQuantCompileInfo>(TilingPrepareForGroupedSituMxQuant);
}  // namespace optiling
