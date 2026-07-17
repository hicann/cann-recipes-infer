/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "dsa_plan_tiling.h"

namespace optiling {
namespace {
constexpr int32_t TOPK_IDX = 0;
constexpr int32_t FULL_KV_ACTUAL_SEQ_IDX = 1;
constexpr int32_t POOL_IDS_IDX = 2;
constexpr int32_t ID_TO_SLOT_IDX = 3;
constexpr int32_t LRU_COUNTER_IDX = 4;
constexpr int32_t RAW_SEQ_ATTR_IDX = 0;
constexpr int64_t PLAN_WIDTH = 2;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t MAX_PROD_RAW_SEQ = 4;
constexpr int64_t MAX_PROD_TOPK = 2048;
constexpr int64_t MAX_PROD_POOL_SIZE = 16384;
constexpr int64_t WAYS_PER_SET = 16;
constexpr int64_t M1B_ORD_MICROTASK_K = 8;
constexpr int64_t INT32_ALIGN_ELEMS = 8;
constexpr int64_t TOPK_GRANULARITY = M1B_ORD_MICROTASK_K * INT32_ALIGN_ELEMS;
constexpr int64_t INSTALL_RECORD_INT32S = 4;
constexpr int64_t INSTALL_RECORD_BYTES = INSTALL_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t INSTALL_COUNT_BYTES = 32;
constexpr int64_t SLAB_ALIGN_BYTES = 4096;

int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

int64_t ComputeInstallRecordsRequiredBytes(int64_t batch, int64_t rawSeq, int64_t topK)
{
    const int64_t recordsPerMt = (topK + M1B_ORD_MICROTASK_K - 1) / M1B_ORD_MICROTASK_K;
    const int64_t recordStride = AlignUp(recordsPerMt * INSTALL_RECORD_BYTES, SLAB_ALIGN_BYTES);
    const int64_t totalMt = batch * rawSeq * M1B_ORD_MICROTASK_K;
    const int64_t recordsBytes = AlignUp(totalMt * recordStride, SLAB_ALIGN_BYTES);
    const int64_t countsBytes = AlignUp(totalMt * INSTALL_COUNT_BYTES, SLAB_ALIGN_BYTES);
    return recordsBytes + countsBytes;
}

int64_t NumElements(const gert::Shape& shape)
{
    int64_t total = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        total *= shape.GetDim(i);
    }
    return total;
}
}  // namespace

ge::graphStatus DsaPlanTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);

    const int64_t* rawSeqAttr = attrs->GetAttrPointer<int64_t>(RAW_SEQ_ATTR_IDX);
    rawSeq_ = (rawSeqAttr != nullptr) ? *rawSeqAttr : 1;
    OPS_ERR_IF(rawSeq_ <= 0, OPS_LOG_E(context_->GetNodeName(), "raw_seq must be positive, got %ld.", rawSeq_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(rawSeq_ > MAX_PROD_RAW_SEQ,
        OPS_LOG_E(context_->GetNodeName(), "DsaPlan raw_seq %ld exceeds production cap %ld.",
            rawSeq_, MAX_PROD_RAW_SEQ),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaPlanTiling::GetPlatformInfo()
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

ge::graphStatus DsaPlanTiling::GetShapeInfo()
{
    auto topkShapeIn = context_->GetInputShape(TOPK_IDX);
    OPS_ERR_IF(topkShapeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get topk shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape topkShape = topkShapeIn->GetStorageShape();
    OPS_ERR_IF((topkShape.GetDimNum() != 3 && topkShape.GetDimNum() != 4),
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%lu should be 3 or 4.",
            topkShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    if (topkShape.GetDimNum() == 4) {
        OPS_ERR_IF(topkShape.GetDim(1) != rawSeq_,
            OPS_LOG_E(context_->GetNodeName(), "raw_seq attr %ld mismatches topk dim1 %ld.",
                rawSeq_, topkShape.GetDim(1)),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(topkShape.GetDim(2) != 1,
            OPS_LOG_E(context_->GetNodeName(), "DsaPlan supports only HEADS=1, got %ld.",
                topkShape.GetDim(2)),
            return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(rawSeq_ != 1,
            OPS_LOG_E(context_->GetNodeName(), "3D topk layout requires raw_seq=1, got %ld.", rawSeq_),
            return ge::GRAPH_FAILED);
    }

    const int64_t topkElems = NumElements(topkShape);
    OPS_ERR_IF(topkElems <= 0, OPS_LOG_E(context_->GetNodeName(), "topk elems must be positive."),
        return ge::GRAPH_FAILED);
    batch_ = topkShape.GetDim(0);
    OPS_ERR_IF(batch_ <= 0, OPS_LOG_E(context_->GetNodeName(), "topk batch must be positive."),
        return ge::GRAPH_FAILED);
    topK_ = topkShape.GetDim(topkShape.GetDimNum() - 1);
    OPS_ERR_IF(topK_ <= 0, OPS_LOG_E(context_->GetNodeName(), "topk dim must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK_ % TOPK_GRANULARITY) != 0,
        OPS_LOG_E(context_->GetNodeName(), "DsaPlan topK %ld must be divisible by %ld.",
            topK_, TOPK_GRANULARITY),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(topK_ > MAX_PROD_TOPK,
        OPS_LOG_E(context_->GetNodeName(), "DsaPlan topK %ld exceeds production cap %ld.",
            topK_, MAX_PROD_TOPK),
        return ge::GRAPH_FAILED);
    topkPerBatch_ = rawSeq_ * topK_;
    planElems_ = topkElems * PLAN_WIDTH;
    installElems_ = ComputeInstallRecordsRequiredBytes(batch_, rawSeq_, topK_) + SLAB_ALIGN_BYTES - 1;
    actualSeqElems_ = topkElems / topkShape.GetDim(topkShape.GetDimNum() - 1);

    auto fullKvActualSeqShapeIn = context_->GetInputShape(FULL_KV_ACTUAL_SEQ_IDX);
    OPS_ERR_IF(fullKvActualSeqShapeIn == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get full_kv_actual_seq shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fullKvActualSeqShape = fullKvActualSeqShapeIn->GetStorageShape();
    OPS_ERR_IF(fullKvActualSeqShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq must be [batch], dim:%lu.",
            fullKvActualSeqShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullKvActualSeqShape.GetDim(0) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_actual_seq shape[0] %ld must equal batch %ld.",
            fullKvActualSeqShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);

    auto poolShapeIn = context_->GetInputShape(POOL_IDS_IDX);
    OPS_ERR_IF(poolShapeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get pool_ids shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape poolShape = poolShapeIn->GetStorageShape();
    OPS_ERR_IF(poolShape.GetDimNum() != 2,
        OPS_LOG_E(context_->GetNodeName(), "pool_ids must be [batch, pool_size], dim:%lu.", poolShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(poolShape.GetDim(0) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "pool_ids shape[0] %ld must equal topk batch %ld.",
            poolShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);
    poolSize_ = poolShape.GetDim(1);
    OPS_ERR_IF((poolSize_ <= 0 || poolSize_ > MAX_PROD_POOL_SIZE || (poolSize_ % WAYS_PER_SET) != 0),
        OPS_LOG_E(context_->GetNodeName(),
            "DsaPlan pool_size %ld must be positive, <= %ld, and divisible by %ld.",
            poolSize_, MAX_PROD_POOL_SIZE, WAYS_PER_SET),
        return ge::GRAPH_FAILED);
    numSets_ = poolSize_ / WAYS_PER_SET;
    OPS_ERR_IF((numSets_ & (numSets_ - 1)) != 0,
        OPS_LOG_E(context_->GetNodeName(),
            "DsaPlan pool_size %ld must produce a power-of-two set count with %ld ways.",
            poolSize_, WAYS_PER_SET),
        return ge::GRAPH_FAILED);
    poolElems_ = NumElements(poolShape);
    OPS_ERR_IF(poolElems_ <= 0, OPS_LOG_E(context_->GetNodeName(), "pool elems must be positive."),
        return ge::GRAPH_FAILED);

    auto idToSlotShapeIn = context_->GetInputShape(ID_TO_SLOT_IDX);
    OPS_ERR_IF(idToSlotShapeIn == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get id_to_slot shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape idToSlotShape = idToSlotShapeIn->GetStorageShape();
    OPS_ERR_IF(idToSlotShape.GetDimNum() != 2,
        OPS_LOG_E(context_->GetNodeName(), "id_to_slot must be [batch, id_range], dim:%lu.",
            idToSlotShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(idToSlotShape.GetDim(0) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "id_to_slot shape[0] %ld must equal batch %ld.",
            idToSlotShape.GetDim(0), batch_),
        return ge::GRAPH_FAILED);
    idRange_ = idToSlotShape.GetDim(1);
    OPS_ERR_IF(idRange_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "id_to_slot id_range %ld must be positive.", idRange_),
        return ge::GRAPH_FAILED);

    auto lruShapeIn = context_->GetInputShape(LRU_COUNTER_IDX);
    OPS_ERR_IF(lruShapeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get lru_counter shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape lruShape = lruShapeIn->GetStorageShape();
    OPS_ERR_IF(lruShape.GetDimNum() != 2,
        OPS_LOG_E(context_->GetNodeName(), "lru_counter must be [batch, num_sets], dim:%lu.",
            lruShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((lruShape.GetDim(0) != poolShape.GetDim(0) || lruShape.GetDim(1) != numSets_),
        OPS_LOG_E(context_->GetNodeName(), "lru_counter shape [%ld, %ld] must be [%ld, %ld].",
            lruShape.GetDim(0), lruShape.GetDim(1), poolShape.GetDim(0), numSets_),
        return ge::GRAPH_FAILED);
    lruElems_ = NumElements(lruShape);
    OPS_ERR_IF(lruElems_ <= 0, OPS_LOG_E(context_->GetNodeName(), "lru elems must be positive."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaPlanTiling::PostTiling()
{
    context_->SetTilingKey(0);
    context_->SetBlockDim(coreNum_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.set_planElems(planElems_);
    tilingData_.set_installElems(installElems_);
    tilingData_.set_actualSeqElems(actualSeqElems_);
    tilingData_.set_poolElems(poolElems_);
    tilingData_.set_lruElems(lruElems_);
    tilingData_.set_batch(batch_);
    tilingData_.set_rawSeq(rawSeq_);
    tilingData_.set_topK(topK_);
    tilingData_.set_numSets(numSets_);
    tilingData_.set_waysPerSet(WAYS_PER_SET);
    tilingData_.set_poolSize(poolSize_);
    tilingData_.set_idRange(idRange_);
    tilingData_.set_topkPerBatch(topkPerBatch_);
    const int64_t totalMt = batch_ * rawSeq_ * M1B_ORD_MICROTASK_K;
    const int64_t mainBsLoopNum = (totalMt + coreNum_ - 1) / coreNum_;
    const int64_t tailBsLoopNum = totalMt / coreNum_;
    const int64_t recordsPerMt = (topK_ + M1B_ORD_MICROTASK_K - 1) / M1B_ORD_MICROTASK_K;
    const int64_t recordStride = AlignUp(recordsPerMt * INSTALL_RECORD_BYTES, SLAB_ALIGN_BYTES);
    const int64_t recordsBytes = AlignUp(totalMt * recordStride, SLAB_ALIGN_BYTES);
    const int64_t requiredBytes = ComputeInstallRecordsRequiredBytes(batch_, rawSeq_, topK_);
    tilingData_.set_usedCoreNum(coreNum_);
    tilingData_.set_mainCoreBsLoopNum(mainBsLoopNum);
    tilingData_.set_tailCoreBsLoopNum(tailBsLoopNum);
    tilingData_.set_m1bOrdMicrotaskK(M1B_ORD_MICROTASK_K);
    tilingData_.set_installRecordsStride(recordStride);
    tilingData_.set_installCountsOffset(recordsBytes);
    tilingData_.set_installRecordsRequiredBytes(requiredBytes);
    tilingData_.set_installRecordsAllocatedBytes(installElems_);
    tilingData_.set_compactAivRecordsOff(0);
    tilingData_.set_compactAivCountsOff(recordsBytes);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaPlanTiling::RunTiling()
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

ge::graphStatus Tiling4DsaPlan(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4DsaPlan", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    DsaPlanTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4DsaPlan(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaPlan)
    .Tiling(Tiling4DsaPlan)
    .TilingParse<DsaPlanCompileInfo>(TilingPrepare4DsaPlan);
}  // namespace optiling
