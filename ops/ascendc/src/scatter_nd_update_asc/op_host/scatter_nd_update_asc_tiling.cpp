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
 * \file scatter_nd_update_asc_tiling.cpp
 * \brief
 */

#include "scatter_nd_update_asc_tiling.h"

namespace optiling {
constexpr int64_t DEFAULT_UB_FOR_ASCNENDC = 8192;
constexpr int64_t DEFAULT_GM_FOR_ASCNENDC = 16 * 1024 * 1024;
constexpr int64_t VEC_BLOCK_SIZE = 32;
constexpr int64_t INPUT_VAR_IDX = 0;
constexpr int64_t INPUT_INDICES_IDX = 1;
constexpr int64_t INPUT_UPDATE_IDX = 2;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t INPUT_DIM_VALUE = 2;
constexpr int64_t DB_CONST = 2;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr int64_t PART_CORE_C_THREAD = 256;
constexpr int64_t PART_CORE_NUM = 16;
constexpr int64_t B1 = 1;
constexpr int64_t B128 = 128;
constexpr int64_t B512 = 512;


ge::graphStatus ScatterNdUpdateAscTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    // DELETE UB FOR ASCENDC
    ubSize_ = ubSize_ - DEFAULT_UB_FOR_ASCNENDC;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetShapeInfo()
{
    OPS_ERR_IF(
        context_ == nullptr, OPS_LOG_E("ScatterNdUpdateAscTiling", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetInputShapeInfo()
{
    auto varInput = context_->GetInputShape(INPUT_VAR_IDX);
    OPS_ERR_IF(varInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get varInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape varShape = varInput->GetStorageShape();
    int64_t dimsN = varShape.GetDimNum();
    OPS_ERR_IF((dimsN != INPUT_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "varInput dim:%ld should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    a_ = varShape.GetDim(DIM_0);
    b_ = varShape.GetDim(DIM_1);
    OPS_ERR_IF(((b_ != B1)&&(b_ != B128)&&(b_ != B512)),
        OPS_LOG_E(context_->GetNodeName(), "varInput dim1:%ld should be {1, 128, 512}.", b_),
        return ge::GRAPH_FAILED);

    auto indicesInput = context_->GetInputShape(INPUT_INDICES_IDX);
    OPS_ERR_IF(indicesInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get indicesInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape indicesShape = indicesInput->GetStorageShape();
    dimsN = indicesShape.GetDimNum();
    OPS_ERR_IF((dimsN != INPUT_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim:%ld should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    c_ = indicesShape.GetDim(DIM_0);
    int64_t indicseRank = indicesShape.GetDim(DIM_1);
    OPS_ERR_IF((indicseRank != 1),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim1:%ld should be 1.", indicseRank),
        return ge::GRAPH_FAILED);
    
    auto updateInput = context_->GetInputShape(INPUT_UPDATE_IDX);
    OPS_ERR_IF(updateInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get updateInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape updateShape = updateInput->GetStorageShape();
    dimsN = updateShape.GetDimNum();
    OPS_ERR_IF((dimsN != INPUT_DIM_VALUE),
        OPS_LOG_E(context_->GetNodeName(), "updateInput dim:%ld should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    int64_t updateC = updateShape.GetDim(DIM_0);
    OPS_ERR_IF((updateC != c_),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim0:%ld  should be same as update dim0:%ld", updateC, c_),
        return ge::GRAPH_FAILED);
    int64_t updateB = updateShape.GetDim(DIM_1);
    OPS_ERR_IF((updateB != b_),
        OPS_LOG_E(context_->GetNodeName(), "indicesInput dim1:%ld  should be same as var dim1:%ld", updateB, b_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::GetInputDtypeInfo()
{
    auto varDesc = context_->GetInputDesc(INPUT_VAR_IDX);
    OPS_ERR_IF(varDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get varDesc nullptr."),
        return ge::GRAPH_FAILED);
    auto varDtype = varDesc->GetDataType();
    OPS_ERR_IF(
        (varDtype != ge::DT_FLOAT16 && varDtype != ge::DT_BF16 && varDtype != ge::DT_INT8),
        OPS_LOG_E(context_->GetNodeName(), "varDtype is not supported."),
        return ge::GRAPH_FAILED);
    varDtypeSize_ = varDtype == ge::DT_INT8 ? sizeof(int8_t) : sizeof(uint16_t);
    int64_t bBlockSize = VEC_BLOCK_SIZE / varDtypeSize_;
    bAlign_ = (b_ + bBlockSize - 1) / bBlockSize * bBlockSize;

    auto indicesDesc = context_->GetInputDesc(INPUT_INDICES_IDX);
    OPS_ERR_IF(indicesDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get indicesDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType indicesDtype = indicesDesc->GetDataType();
    OPS_ERR_IF(
        (indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64),
        OPS_LOG_E(context_->GetNodeName(), "indicesDtype is not supported."),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = indicesDtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);

    auto updateDesc = context_->GetInputDesc(INPUT_UPDATE_IDX);
    OPS_ERR_IF(updateDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get updateDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType updateDtype = updateDesc->GetDataType();
    OPS_ERR_IF(
        (updateDtype != varDtype),
        OPS_LOG_E(context_->GetNodeName(), "updateDtype should same with varDtype."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::DoOpTiling()
{
    // block_factor
    if (c_ <= PART_CORE_C_THREAD) {
        coreNum_ = PART_CORE_NUM;
    }
    blockFactor_ = (c_ + coreNum_ - 1) / coreNum_;
    blockNum_ = (c_ + blockFactor_ - 1) / blockFactor_;
    blockFactorTail_ = c_ - (blockNum_ - 1) * blockFactor_;

    // ub_factor condtion
    // ub only constains update
    ubFactor_ = ubSize_ / DB_CONST / bAlign_ / varDtypeSize_;
    OPS_ERR_IF(
    (ubFactor_ < 1),
    OPS_LOG_E(context_->GetNodeName(), "update length to long %ld, not support.", b_),
    return ge::GRAPH_FAILED);
    ubFactor_ = ubFactor_ > blockFactor_ ? blockFactor_ : ubFactor_;

     
    tilingData_.set_a(a_);
    tilingData_.set_b(b_);
    tilingData_.set_bAlign(bAlign_);
    tilingData_.set_c(c_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockFactorTail(blockFactorTail_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_blockNum(blockNum_);
    
    OPS_LOG_I(context_->GetNodeName(), "TilingData ScatterNdUpdateAsc a=%ld, b=%ld, bAlign=%ld, c=%ld, blockFactor=%ld, blockFactorTail=%ld, ubFactor=%ld, blockNum=%ld.",
     a_, b_, bAlign_, c_, blockFactor_, blockFactorTail_, ubFactor_, blockNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::PostTiling()
{
    context_->SetTilingKey(0);
    context_->SetBlockDim(blockNum_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_GM_FOR_ASCNENDC;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    OPS_LOG_I(context_->GetNodeName(), "TilingForScatterNdUpdateAsc leaving.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdUpdateAscTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4ScatterNdUpdateAsc(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForScatterNdUpdateAsc running.");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForScatterNdUpdateAsc", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    ScatterNdUpdateAscTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4ScatterNdUpdateAsc(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterNdUpdateAsc)
    .Tiling(Tiling4ScatterNdUpdateAsc)
    .TilingParse<ScatterNdUpdateAscCompileInfo>(TilingPrepare4ScatterNdUpdateAsc);

} // namespace optiling