/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "dsa_serve_tiling.h"

namespace optiling {
namespace {
constexpr int32_t PLAN_IDX = 0;
constexpr int32_t FULL_KV_IDX = 1;
constexpr int32_t FULL_ROPE_IDX = 2;
constexpr int32_t POOL_KV_IDX = 3;
constexpr int32_t POOL_ROPE_IDX = 4;
constexpr int32_t SELECTION_KV_IDX = 5;
constexpr int32_t SELECTION_ROPE_IDX = 6;
constexpr int32_t RAW_SEQ_ATTR_IDX = 0;
constexpr int32_t TOPK_ATTR_IDX = 1;
constexpr int32_t SELECTION_BLOCK_SIZE_ATTR_IDX = 2;
constexpr int32_t COMPACT_LAYOUT_ATTR_IDX = 3;
constexpr int64_t PLAN_WIDTH = 2;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t LEGACY_MAX_BATCH = 16;
constexpr int64_t LEGACY_MAX_RAW_SEQ = 4;
constexpr int64_t LEGACY_MAX_TOPK = 64;
constexpr int64_t LEGACY_MAX_ROWS = 256;
constexpr int64_t LEGACY_MAX_FULL_SEQ = 512;
constexpr int64_t LEGACY_MAX_POOL_SIZE = 256;
constexpr int64_t PROD_MAX_RAW_SEQ = 4;
constexpr int64_t PROD_MAX_TOPK = 2048;
constexpr int64_t PROD_MAX_POOL_SIZE = 16384;
constexpr int64_t TOPK_GRANULARITY = 8;
constexpr int64_t MIN_SELECTION_BLOCK_SIZE = 16;
constexpr int64_t MAX_SELECTION_BLOCK_SIZE = 128;
constexpr int64_t MAX_KV_DIM = 512;
constexpr int64_t MAX_ROPE_DIM = 128;
}  // namespace

ge::graphStatus DsaServeTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);
    const int64_t* rawSeqAttr = attrs->GetAttrPointer<int64_t>(RAW_SEQ_ATTR_IDX);
    rawSeq_ = (rawSeqAttr != nullptr) ? *rawSeqAttr : 1;
    const int64_t* topkAttr = attrs->GetAttrPointer<int64_t>(TOPK_ATTR_IDX);
    topK_ = (topkAttr != nullptr) ? *topkAttr : 2048;
    const int64_t* selectionBlockSizeAttr = attrs->GetAttrPointer<int64_t>(SELECTION_BLOCK_SIZE_ATTR_IDX);
    selectionBlockSize_ = (selectionBlockSizeAttr != nullptr) ? *selectionBlockSizeAttr : 128;
    const int64_t* compactLayoutAttr = attrs->GetAttrPointer<int64_t>(COMPACT_LAYOUT_ATTR_IDX);
    compactLayout_ = (compactLayoutAttr != nullptr) ? *compactLayoutAttr : 1;
    OPS_ERR_IF((compactLayout_ != 0 && compactLayout_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "DsaServe compact_layout must be 0 or 1, got %ld.", compactLayout_),
        return ge::GRAPH_FAILED);
    const bool compact = compactLayout_ != 0;
    const int64_t maxRawSeq = compact ? PROD_MAX_RAW_SEQ : LEGACY_MAX_RAW_SEQ;
    const int64_t maxTopK = compact ? PROD_MAX_TOPK : LEGACY_MAX_TOPK;
    OPS_ERR_IF((rawSeq_ <= 0 || rawSeq_ > maxRawSeq),
        OPS_LOG_E(context_->GetNodeName(), "DsaServe raw_seq %ld must be in (0, %ld] for compact_layout=%ld.",
            rawSeq_, maxRawSeq, compactLayout_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK_ <= 0 || topK_ > maxTopK || (topK_ % TOPK_GRANULARITY) != 0),
        OPS_LOG_E(context_->GetNodeName(),
            "DsaServe topK %ld must be positive, <= %ld, and divisible by %ld for compact_layout=%ld.",
            topK_, maxTopK, TOPK_GRANULARITY, compactLayout_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionBlockSize_ < MIN_SELECTION_BLOCK_SIZE ||
                   selectionBlockSize_ > MAX_SELECTION_BLOCK_SIZE ||
                   (selectionBlockSize_ & (selectionBlockSize_ - 1)) != 0),
        OPS_LOG_E(context_->GetNodeName(),
            "DsaServe selection_block_size %ld must be a power of two in [%ld, %ld].",
            selectionBlockSize_, MIN_SELECTION_BLOCK_SIZE, MAX_SELECTION_BLOCK_SIZE),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(compact && (topK_ % selectionBlockSize_) != 0,
        OPS_LOG_E(context_->GetNodeName(),
            "DsaServe compact topK %ld must be divisible by selection_block_size %ld.",
            topK_, selectionBlockSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaServeTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaServeTiling::GetShapeInfo()
{
    auto planShapeIn = context_->GetInputShape(PLAN_IDX);
    OPS_ERR_IF(planShapeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get plan shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape planShape = planShapeIn->GetStorageShape();
    OPS_ERR_IF((planShape.GetDimNum() != 2 || planShape.GetDim(1) != PLAN_WIDTH),
        OPS_LOG_E(context_->GetNodeName(), "plan must be [rows, 2]."),
        return ge::GRAPH_FAILED);
    planRows_ = planShape.GetDim(0);
    const bool compact = compactLayout_ != 0;
    const int64_t maxPoolSize = compact ? PROD_MAX_POOL_SIZE : LEGACY_MAX_POOL_SIZE;
    OPS_ERR_IF(planRows_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "DsaServe rows must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((!compact && planRows_ > LEGACY_MAX_ROWS),
        OPS_LOG_E(context_->GetNodeName(), "DsaServe rows %ld exceed legacy cap %ld for compact_layout=%ld.",
            planRows_, LEGACY_MAX_ROWS, compactLayout_),
        return ge::GRAPH_FAILED);

    auto fullKvShapeIn = context_->GetInputShape(FULL_KV_IDX);
    auto fullRopeShapeIn = context_->GetInputShape(FULL_ROPE_IDX);
    auto poolKvShapeIn = context_->GetInputShape(POOL_KV_IDX);
    auto poolRopeShapeIn = context_->GetInputShape(POOL_ROPE_IDX);
    auto selectionKvShapeIn = context_->GetInputShape(SELECTION_KV_IDX);
    auto selectionRopeShapeIn = context_->GetInputShape(SELECTION_ROPE_IDX);
    OPS_ERR_IF(fullKvShapeIn == nullptr || fullRopeShapeIn == nullptr ||
                   poolKvShapeIn == nullptr || poolRopeShapeIn == nullptr ||
                   selectionKvShapeIn == nullptr || selectionRopeShapeIn == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get DsaServe payload shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fullKvShape = fullKvShapeIn->GetStorageShape();
    gert::Shape fullRopeShape = fullRopeShapeIn->GetStorageShape();
    gert::Shape poolKvShape = poolKvShapeIn->GetStorageShape();
    gert::Shape poolRopeShape = poolRopeShapeIn->GetStorageShape();
    gert::Shape selectionKvShape = selectionKvShapeIn->GetStorageShape();
    gert::Shape selectionRopeShape = selectionRopeShapeIn->GetStorageShape();
    OPS_ERR_IF((fullKvShape.GetDimNum() != 3 || fullRopeShape.GetDimNum() != 3 ||
                   poolKvShape.GetDimNum() != 3 || poolRopeShape.GetDimNum() != 3 ||
                   selectionKvShape.GetDimNum() != 3 || selectionRopeShape.GetDimNum() != 3),
        OPS_LOG_E(context_->GetNodeName(), "full/pool/selection KV and rope tensors must be 3D."),
        return ge::GRAPH_FAILED);
    batch_ = fullKvShape.GetDim(0);
    OPS_ERR_IF(batch_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "DsaServe batch must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((!compact && batch_ > LEGACY_MAX_BATCH),
        OPS_LOG_E(context_->GetNodeName(), "DsaServe batch %ld exceeds legacy cap %ld for compact_layout=%ld.",
            batch_, LEGACY_MAX_BATCH, compactLayout_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(planRows_ != batch_ * rawSeq_ * topK_,
        OPS_LOG_E(context_->GetNodeName(), "plan rows %ld must equal batch*raw_seq*topK %ld.",
            planRows_, batch_ * rawSeq_ * topK_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((fullRopeShape.GetDim(0) != batch_ || poolKvShape.GetDim(0) != batch_ ||
                   poolRopeShape.GetDim(0) != batch_),
        OPS_LOG_E(context_->GetNodeName(), "all DsaServe payload tensors must share batch."),
        return ge::GRAPH_FAILED);
    fullSeq_ = fullKvShape.GetDim(1);
    poolSize_ = poolKvShape.GetDim(1);
    kvDim_ = fullKvShape.GetDim(2);
    ropeDim_ = fullRopeShape.GetDim(2);
    OPS_ERR_IF((fullSeq_ <= 0 || (!compact && fullSeq_ > LEGACY_MAX_FULL_SEQ)),
        OPS_LOG_E(context_->GetNodeName(), "full_seq %ld unsupported for DsaServe compact_layout=%ld.",
            fullSeq_, compactLayout_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((poolSize_ <= 0 || poolSize_ > maxPoolSize || poolRopeShape.GetDim(1) != poolSize_),
        OPS_LOG_E(context_->GetNodeName(), "pool size unsupported or mismatched for DsaServe compact_layout=%ld.",
            compactLayout_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((kvDim_ <= 0 || kvDim_ > MAX_KV_DIM || poolKvShape.GetDim(2) != kvDim_),
        OPS_LOG_E(context_->GetNodeName(), "kv dim unsupported or mismatched for DsaServe gate."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((ropeDim_ <= 0 || ropeDim_ > MAX_ROPE_DIM || poolRopeShape.GetDim(2) != ropeDim_),
        OPS_LOG_E(context_->GetNodeName(), "rope dim unsupported or mismatched for DsaServe gate."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullRopeShape.GetDim(1) != fullSeq_,
        OPS_LOG_E(context_->GetNodeName(), "full rope seq must match full kv seq."),
        return ge::GRAPH_FAILED);
    const int64_t selectionBlocks = compact ?
        ((planRows_ + selectionBlockSize_ - 1) / selectionBlockSize_) : planRows_;
    OPS_ERR_IF((selectionKvShape.GetDim(0) != selectionBlocks ||
                   selectionKvShape.GetDim(1) != selectionBlockSize_ ||
                   selectionKvShape.GetDim(2) != kvDim_),
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_cache shape mismatches DsaServe output shape."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionRopeShape.GetDim(0) != selectionBlocks ||
                   selectionRopeShape.GetDim(1) != selectionBlockSize_ ||
                   selectionRopeShape.GetDim(2) != ropeDim_),
        OPS_LOG_E(context_->GetNodeName(), "selection_k_rope shape mismatches DsaServe output shape."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaServeTiling::PostTiling()
{
    context_->SetTilingKey(0);
    context_->SetBlockDim(coreNum_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.set_planRows(planRows_);
    tilingData_.set_batch(batch_);
    tilingData_.set_rawSeq(rawSeq_);
    tilingData_.set_topK(topK_);
    tilingData_.set_fullSeq(fullSeq_);
    tilingData_.set_poolSize(poolSize_);
    tilingData_.set_selectionBlockSize(selectionBlockSize_);
    tilingData_.set_kvDim(kvDim_);
    tilingData_.set_ropeDim(ropeDim_);
    tilingData_.set_compactLayout(compactLayout_);
    tilingData_.set_usedCoreNum(coreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaServeTiling::RunTiling()
{
    ge::graphStatus ret = GetAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4DsaServe(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4DsaServe", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    DsaServeTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4DsaServe(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaServe)
    .Tiling(Tiling4DsaServe)
    .TilingParse<DsaServeCompileInfo>(TilingPrepare4DsaServe);
}  // namespace optiling
