/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <sstream>
#include "kv_compress_epilog_v2_tiling_arch35.h"
#include "log/ops_log.h"

namespace optiling {
namespace {
template <typename T>
T CeilDiv(T value, T divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

int64_t RoundUp(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}
}  // namespace

ge::graphStatus KvCompressEpilogV2Tiling::GetPlatformInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto *compileInfo = context_->GetCompileInfo<KvCompressEpilogV2CompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        coreNum_ = platform.GetCoreNumAiv();
    }
    OPS_CHECK(coreNum_ == 0 || ubSize_ == 0,
                OPS_LOG_E(context_->GetNodeName(), "invalid platform resources, coreNum=%lu, ubSize=%lu.",
                        coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogV2Tiling::GetShapeAndDtypeInfo()
{
    auto *cacheShape = context_->GetInputShape(KCEV2_CACHE_INPUT_INDEX);
    auto *xShape = context_->GetInputShape(KCEV2_X_INPUT_INDEX);
    auto *slotShape = context_->GetInputShape(KCEV2_SLOT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, slotShape);

    const auto &cacheStorageShape = cacheShape->GetStorageShape();
    const auto &xStorageShape = xShape->GetStorageShape();
    const auto &slotStorageShape = slotShape->GetStorageShape();
    OPS_CHECK(cacheStorageShape.GetDimNum() != 2,
                OPS_LOG_E(context_->GetNodeName(), "cache must be 2D, got rank %zu.",
                          cacheStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_CHECK(xStorageShape.GetDimNum() != 2,
                OPS_LOG_E(context_->GetNodeName(), "x must be 2D, got rank %zu.", xStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_CHECK(slotStorageShape.GetDimNum() != 1,
                OPS_LOG_E(context_->GetNodeName(), "slot_mapping must be 1D, got rank %zu.",
                          slotStorageShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    cacheRows_ = cacheStorageShape.GetDim(0);
    cacheRowStride_ = cacheStorageShape.GetDim(1);
    bs_ = xStorageShape.GetDim(0);
    d_ = xStorageShape.GetDim(1);
    OPS_CHECK(slotStorageShape.GetDim(0) != bs_,
                OPS_LOG_E(context_->GetNodeName(), "slot_mapping length must equal x dim 0, got %ld and %ld.",
                          slotStorageShape.GetDim(0), bs_),
                return ge::GRAPH_FAILED);

    auto *cacheDesc = context_->GetInputDesc(KCEV2_CACHE_INPUT_INDEX);
    auto *xDesc = context_->GetInputDesc(KCEV2_X_INPUT_INDEX);
    auto *slotDesc = context_->GetInputDesc(KCEV2_SLOT_INPUT_INDEX);
    auto *outputDesc = context_->GetOutputDesc(KCEV2_CACHE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cacheDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, slotDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    cacheDtype_ = cacheDesc->GetDataType();
    xDtype_ = xDesc->GetDataType();
    slotDtype_ = slotDesc->GetDataType();
    outputDtype_ = outputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogV2Tiling::GetAttributes()
{
    auto *attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t *groupSize = attrs->GetAttrPointer<int64_t>(KCEV2_GROUP_ATTR_INDEX);
    const int64_t *mode = attrs->GetAttrPointer<int64_t>(KCEV2_MODE_ATTR_INDEX);
    const bool *roundScale = attrs->GetAttrPointer<bool>(KCEV2_ROUND_ATTR_INDEX);
    const float *xScale = attrs->GetAttrPointer<float>(KCEV2_X_SCALE_ATTR_INDEX);
    quantGroupSize_ = groupSize == nullptr ? KCEV2_GROUP_SIZE_32 : *groupSize;
    quantMode_ = mode == nullptr ? KCEV2_MODE_MXFP8 : *mode;
    roundScale_ = roundScale == nullptr || *roundScale ? 1 : 0;
    xScale_ = xScale == nullptr ? 1.0f : *xScale;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogV2Tiling::ValidateAndCalculateLayout()
{
    OPS_CHECK(bs_ <= 0 || d_ <= 0 || cacheRows_ <= 0 || cacheRowStride_ <= 0,
                OPS_LOG_E(context_->GetNodeName(),
                        "all tensor dimensions must be positive, got bs=%ld, d=%ld, cacheRows=%ld, cacheCol=%ld.",
                        bs_, d_, cacheRows_, cacheRowStride_),
                return ge::GRAPH_FAILED);
    OPS_CHECK(d_ > 8192,
              OPS_LOG_E(context_->GetNodeName(), "x last dimension must not exceed 8192, got %ld.", d_),
              return ge::GRAPH_FAILED);
    OPS_CHECK(quantMode_ != KCEV2_MODE_MXFP8 && quantMode_ != KCEV2_MODE_MXFP4,
                OPS_LOG_E(context_->GetNodeName(), "quant_mode must be 2 or 4, got %ld.", quantMode_),
                return ge::GRAPH_FAILED);
    const bool validModeAndGroup =
        (quantMode_ == KCEV2_MODE_MXFP8 && quantGroupSize_ == KCEV2_GROUP_SIZE_32) ||
        (quantMode_ == KCEV2_MODE_MXFP4 &&
         (quantGroupSize_ == KCEV2_GROUP_SIZE_16 || quantGroupSize_ == KCEV2_GROUP_SIZE_32));
    OPS_CHECK(!validModeAndGroup,
                OPS_LOG_E(context_->GetNodeName(),
                          "invalid quant_mode and quant_group_size combination, got quant_mode=%ld, "
                          "quant_group_size=%ld; supported combinations are (2,32), (4,32), and (4,16).",
                          quantMode_, quantGroupSize_),
                return ge::GRAPH_FAILED);
    OPS_CHECK(d_ % quantGroupSize_ != 0,
                OPS_LOG_E(context_->GetNodeName(),
                          "x last dimension must be divisible by quant_group_size, got d=%ld, "
                          "quant_group_size=%ld.", d_, quantGroupSize_),
                return ge::GRAPH_FAILED);
    OPS_CHECK(xScale_ != 1.0f,
              OPS_LOG_E(context_->GetNodeName(), "x_scale is reserved and must be 1.0, got %f.", xScale_),
              return ge::GRAPH_FAILED);
    OPS_CHECK(xDtype_ != ge::DT_BF16,
                OPS_LOG_E(context_->GetNodeName(), "x dtype must be BF16, got %d.",
                          static_cast<int32_t>(xDtype_)),
                return ge::GRAPH_FAILED);
    OPS_CHECK(slotDtype_ != ge::DT_INT32 && slotDtype_ != ge::DT_INT64,
                OPS_LOG_E(context_->GetNodeName(), "slot_mapping dtype must be INT32 or INT64, got %d.",
                        static_cast<int32_t>(slotDtype_)),
                return ge::GRAPH_FAILED);
    OPS_CHECK(outputDtype_ != cacheDtype_,
                OPS_LOG_E(context_->GetNodeName(), "output cache dtype must match input cache dtype, got %d and %d.",
                        static_cast<int32_t>(outputDtype_), static_cast<int32_t>(cacheDtype_)),
                return ge::GRAPH_FAILED);
    const bool validMxFp8Dtype = cacheDtype_ == ge::DT_FLOAT8_E4M3FN || cacheDtype_ == ge::DT_FLOAT8_E5M2;
    OPS_CHECK((quantMode_ == KCEV2_MODE_MXFP8 && !validMxFp8Dtype) ||
                    (quantMode_ == KCEV2_MODE_MXFP4 && cacheDtype_ != ge::DT_UINT8),
                OPS_LOG_E(context_->GetNodeName(),
                        "cache dtype does not match quant_mode, got quant_mode=%ld, cache dtype=%d.",
                        quantMode_, static_cast<int32_t>(cacheDtype_)),
                return ge::GRAPH_FAILED);

    scaleCol_ = d_ / quantGroupSize_;
    dataCol_ = quantMode_ == KCEV2_MODE_MXFP8 ? d_ : d_ / 2;
    concatCol_ = dataCol_ + scaleCol_ * static_cast<int64_t>(sizeof(uint16_t));
    kvCacheCol_ = RoundUp(concatCol_, 32L);
    padCol_ = kvCacheCol_ - concatCol_;
    OPS_CHECK(cacheRowStride_ < kvCacheCol_,
                OPS_LOG_E(context_->GetNodeName(), "cache row width must be at least %ld bytes, got %ld.",
                        kvCacheCol_, cacheRowStride_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogV2Tiling::DoOpTiling()
{
    const int64_t rowOfFormerBlock = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(static_cast<uint64_t>(CeilDiv(bs_, rowOfFormerBlock)), coreNum_);
    const int64_t rowOfTailBlock = bs_ - (static_cast<int64_t>(usedCoreNums_) - 1) * rowOfFormerBlock;

    int64_t rowFactor = 0;
    const bool isMxFp4Group16 =
        quantMode_ == KCEV2_MODE_MXFP4 && quantGroupSize_ == KCEV2_GROUP_SIZE_16;
    for (int64_t candidate = 1; candidate <= rowOfFormerBlock; ++candidate) {
        const int64_t xBytes = candidate * RoundUp(d_, 16L) * 2 * 2;
        const int64_t yBytes = candidate * RoundUp(kvCacheCol_, 32L) * 2;
        const int64_t scaleAlignment = isMxFp4Group16 ? 32L : 64L;
        const int64_t scaleBytes = candidate * RoundUp(scaleCol_ * 2, scaleAlignment) * 2;
        const int64_t fp4ScratchBytes = isMxFp4Group16 ?
            candidate * CeilDiv(d_, 64L) * 32L + 32L : 2 * RoundUp(scaleCol_ * 2, 32L) + 32L;
        const int64_t indexBytes = RoundUp(candidate * static_cast<int64_t>(sizeof(int64_t)), 32L);
        if (xBytes + yBytes + scaleBytes + fp4ScratchBytes + indexBytes > static_cast<int64_t>(ubSize_)) {
            break;
        }
        rowFactor = candidate;
    }
    OPS_CHECK(rowFactor == 0,
                OPS_LOG_E(context_->GetNodeName(), "UB is insufficient for one row, ubSize=%lu, d=%ld, cacheCol=%ld.",
                        ubSize_, d_, kvCacheCol_),
                return ge::GRAPH_FAILED);

    const int64_t rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, rowFactor);
    const int64_t rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, rowFactor);
    const int64_t tailFormer = rowOfFormerBlock % rowFactor == 0 ? rowFactor : rowOfFormerBlock % rowFactor;
    const int64_t tailTail = rowOfTailBlock % rowFactor == 0 ? rowFactor : rowOfTailBlock % rowFactor;

    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_cacheRows(cacheRows_);
    tilingData_.set_cacheRowStride(cacheRowStride_);
    tilingData_.set_dataCol(dataCol_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_concatCol(concatCol_);
    tilingData_.set_kvCacheCol(kvCacheCol_);
    tilingData_.set_padCol(padCol_);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_perGroupSize(quantGroupSize_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock);
    tilingData_.set_rowFactor(rowFactor);
    tilingData_.set_tailRowFactorOfFormerBlock(tailFormer);
    tilingData_.set_tailRowFactorOfTailBlock(tailTail);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvCompressEpilogV2Tiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    uint64_t tilingKey = 2000UL;
    if (quantMode_ == KCEV2_MODE_MXFP4) {
        tilingKey = quantGroupSize_ == KCEV2_GROUP_SIZE_16 ? 2002UL : 2001UL;
    }
    context_->SetTilingKey(tilingKey);
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = KCEV2_WORKSPACE_SIZE;
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

void KvCompressEpilogV2Tiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "bs=" << tilingData_.get_bs() << ", d=" << tilingData_.get_d()
         << ", dataCol=" << tilingData_.get_dataCol() << ", scaleCol=" << tilingData_.get_scaleCol()
         << ", concatCol=" << tilingData_.get_concatCol() << ", kvCacheCol=" << tilingData_.get_kvCacheCol()
         << ", padCol=" << tilingData_.get_padCol() << ", quantMode=" << tilingData_.get_quantMode()
         << ", roundScale=" << tilingData_.get_roundScale() << ", rowFactor=" << tilingData_.get_rowFactor();
    OPS_LOG_I(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus KvCompressEpilogV2Tiling::RunTiling()
{
    OPS_CHECK(GetPlatformInfo() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "GetPlatformInfo failed."), return ge::GRAPH_FAILED);
    OPS_CHECK(GetShapeAndDtypeInfo() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "GetShapeAndDtypeInfo failed."), return ge::GRAPH_FAILED);
    OPS_CHECK(GetAttributes() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "GetAttributes failed."), return ge::GRAPH_FAILED);
    OPS_CHECK(ValidateAndCalculateLayout() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "ValidateAndCalculateLayout failed."), return ge::GRAPH_FAILED);
    OPS_CHECK(DoOpTiling() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "DoOpTiling failed."), return ge::GRAPH_FAILED);
    return PostTiling();
}

ge::graphStatus TilingForKvCompressEpilogV2(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    KvCompressEpilogV2Tiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForKvCompressEpilogV2(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto *compileInfo = context->GetCompiledInfo<KvCompressEpilogV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilogV2)
    .Tiling(TilingForKvCompressEpilogV2)
    .TilingParse<KvCompressEpilogV2CompileInfo>(TilingPrepareForKvCompressEpilogV2);

}  // namespace optiling
