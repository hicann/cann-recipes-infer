/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "dsa_install_tiling.h"

namespace optiling {
namespace {
constexpr int32_t INSTALL_RECORDS_IDX = 0;
constexpr int32_t SELECTION_KV_IDX = 1;
constexpr int32_t SELECTION_ROPE_IDX = 2;
constexpr int32_t SELECTION_BLOCK_TABLE_IDX = 3;
constexpr int32_t POOL_KV_IDX = 4;
constexpr int32_t POOL_ROPE_IDX = 5;
constexpr int32_t POOL_IDS_IDX = 6;
constexpr int32_t ID_TO_SLOT_IDX = 7;
constexpr int32_t LRU_COUNTER_IDX = 8;
constexpr int32_t RAW_SEQ_ATTR_IDX = 0;
constexpr int32_t TOPK_ATTR_IDX = 1;
constexpr int32_t SELECTION_BLOCK_SIZE_ATTR_IDX = 2;
constexpr int32_t METADATA_UPDATE_ATTR_IDX = 3;
constexpr int64_t MAX_PROD_RAW_SEQ = 4;
constexpr int64_t MAX_PROD_TOPK = 2048;
constexpr int64_t MAX_PROD_POOL_SIZE = 16384;
constexpr int64_t MAX_PROD_KV_DIM = 512;
constexpr int64_t MAX_PROD_ROPE_DIM = 128;
constexpr int64_t MIN_SELECTION_BLOCK_SIZE = 16;
constexpr int64_t MAX_SELECTION_BLOCK_SIZE = 128;
constexpr int64_t M1B_ORD_MICROTASK_K = 8;
constexpr int64_t INSTALL_RECORD_INT32S = 4;
constexpr int64_t STAGED_RECORD_INT32S = 5;
constexpr int64_t INSTALL_RECORD_BYTES = INSTALL_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t INSTALL_COUNT_BYTES = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t SLAB_ALIGN_BYTES = 4096;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t UB_SAFETY_PAD = 8 * 1024;

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
}  // namespace

ge::graphStatus DsaInstallTiling::GetAttrInfo()
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
    const int64_t* metadataUpdateAttr = attrs->GetAttrPointer<int64_t>(METADATA_UPDATE_ATTR_IDX);
    metadataUpdate_ = (metadataUpdateAttr != nullptr) ? *metadataUpdateAttr : 1;
    OPS_ERR_IF((rawSeq_ <= 0 || rawSeq_ > MAX_PROD_RAW_SEQ),
        OPS_LOG_E(context_->GetNodeName(), "raw_seq %ld must be in (0, %ld].", rawSeq_, MAX_PROD_RAW_SEQ),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK_ <= 0 || topK_ > MAX_PROD_TOPK || (topK_ % M1B_ORD_MICROTASK_K) != 0),
        OPS_LOG_E(context_->GetNodeName(), "topk %ld must be positive, <= %ld, and divisible by %ld.",
            topK_, MAX_PROD_TOPK, M1B_ORD_MICROTASK_K),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionBlockSize_ < MIN_SELECTION_BLOCK_SIZE ||
                   selectionBlockSize_ > MAX_SELECTION_BLOCK_SIZE ||
                   (selectionBlockSize_ & (selectionBlockSize_ - 1)) != 0),
        OPS_LOG_E(context_->GetNodeName(), "selection_block_size %ld must be power-of-two in [%ld, %ld].",
            selectionBlockSize_, MIN_SELECTION_BLOCK_SIZE, MAX_SELECTION_BLOCK_SIZE),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK_ % selectionBlockSize_) != 0,
        OPS_LOG_E(context_->GetNodeName(),
            "topk %ld must be divisible by selection_block_size %ld.", topK_, selectionBlockSize_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((metadataUpdate_ != 0 && metadataUpdate_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "metadata_update %ld must be 0 or 1.", metadataUpdate_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaInstallTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OPS_ERR_IF(ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "UB size must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaInstallTiling::GetShapeInfo()
{
    auto installRecordsShapeIn = context_->GetInputShape(INSTALL_RECORDS_IDX);
    OPS_ERR_IF(installRecordsShapeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get install_records shape nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape installRecordsShape = installRecordsShapeIn->GetStorageShape();
    OPS_ERR_IF(installRecordsShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "install_records must be 1-D int8 buffer."),
        return ge::GRAPH_FAILED);
    installRecordsAllocatedBytes_ = installRecordsShape.GetDim(0);

    auto selectionKvShapeIn = context_->GetInputShape(SELECTION_KV_IDX);
    auto selectionRopeShapeIn = context_->GetInputShape(SELECTION_ROPE_IDX);
    auto selectionBlockTableShapeIn = context_->GetInputShape(SELECTION_BLOCK_TABLE_IDX);
    auto poolKvShapeIn = context_->GetInputShape(POOL_KV_IDX);
    auto poolRopeShapeIn = context_->GetInputShape(POOL_ROPE_IDX);
    auto poolIdsShapeIn = context_->GetInputShape(POOL_IDS_IDX);
    auto idToSlotShapeIn = context_->GetInputShape(ID_TO_SLOT_IDX);
    auto lruShapeIn = context_->GetInputShape(LRU_COUNTER_IDX);
    OPS_ERR_IF(selectionKvShapeIn == nullptr || selectionRopeShapeIn == nullptr ||
                   selectionBlockTableShapeIn == nullptr || poolKvShapeIn == nullptr ||
                   poolRopeShapeIn == nullptr || poolIdsShapeIn == nullptr ||
                   idToSlotShapeIn == nullptr || lruShapeIn == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get DsaInstall input shape nullptr."),
        return ge::GRAPH_FAILED);

    gert::Shape selectionKvShape = selectionKvShapeIn->GetStorageShape();
    gert::Shape selectionRopeShape = selectionRopeShapeIn->GetStorageShape();
    gert::Shape selectionBlockTableShape = selectionBlockTableShapeIn->GetStorageShape();
    gert::Shape poolKvShape = poolKvShapeIn->GetStorageShape();
    gert::Shape poolRopeShape = poolRopeShapeIn->GetStorageShape();
    gert::Shape poolIdsShape = poolIdsShapeIn->GetStorageShape();
    gert::Shape idToSlotShape = idToSlotShapeIn->GetStorageShape();
    gert::Shape lruShape = lruShapeIn->GetStorageShape();
    OPS_ERR_IF((selectionKvShape.GetDimNum() != 3 || selectionRopeShape.GetDimNum() != 3 ||
                   poolKvShape.GetDimNum() != 3 || poolRopeShape.GetDimNum() != 3),
        OPS_LOG_E(context_->GetNodeName(), "selection/pool KV and rope tensors must be 3D."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(selectionBlockTableShape.GetDimNum() != 2,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table must be [batch*raw_seq, blocks]."),
        return ge::GRAPH_FAILED);

    selectionRows_ = selectionKvShape.GetDim(0) * selectionKvShape.GetDim(1);
    batch_ = poolKvShape.GetDim(0);
    poolSize_ = poolKvShape.GetDim(1);
    kvDim_ = selectionKvShape.GetDim(2);
    ropeDim_ = selectionRopeShape.GetDim(2);
    OPS_ERR_IF(batch_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "DsaInstall batch must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionRows_ < batch_ * rawSeq_ * topK_ ||
                   selectionRopeShape.GetDim(0) != selectionKvShape.GetDim(0) ||
                   selectionRopeShape.GetDim(1) != selectionKvShape.GetDim(1)),
        OPS_LOG_E(context_->GetNodeName(), "selection rows unsupported or mismatched for DsaInstall."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionBlockTableShape.GetDim(0) != batch_ * rawSeq_ ||
                   selectionBlockTableShape.GetDim(1) * selectionBlockSize_ < topK_),
        OPS_LOG_E(context_->GetNodeName(), "selection block table shape mismatches batch/raw_seq/topk."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((poolSize_ <= 0 || poolSize_ > MAX_PROD_POOL_SIZE || poolRopeShape.GetDim(1) != poolSize_),
        OPS_LOG_E(context_->GetNodeName(), "pool size unsupported or mismatched for DsaInstall."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((kvDim_ <= 0 || kvDim_ > MAX_PROD_KV_DIM || poolKvShape.GetDim(2) != kvDim_),
        OPS_LOG_E(context_->GetNodeName(), "kv dim unsupported or mismatched for DsaInstall."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((ropeDim_ <= 0 || ropeDim_ > MAX_PROD_ROPE_DIM || poolRopeShape.GetDim(2) != ropeDim_),
        OPS_LOG_E(context_->GetNodeName(), "rope dim unsupported or mismatched for DsaInstall."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(poolIdsShape.GetDimNum() != 2 || poolIdsShape.GetDim(0) != batch_ || poolIdsShape.GetDim(1) != poolSize_,
        OPS_LOG_E(context_->GetNodeName(), "pool_ids must be [batch, pool_size]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(idToSlotShape.GetDimNum() != 2 || idToSlotShape.GetDim(0) != batch_,
        OPS_LOG_E(context_->GetNodeName(), "id_to_slot must be [batch, id_range]."),
        return ge::GRAPH_FAILED);
    idRange_ = idToSlotShape.GetDim(1);
    OPS_ERR_IF(idRange_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "id_to_slot id_range %ld must be positive.", idRange_),
        return ge::GRAPH_FAILED);
    numSets_ = poolSize_ / 16;
    waysPerSet_ = 16;
    OPS_ERR_IF((poolSize_ % waysPerSet_) != 0 || (numSets_ & (numSets_ - 1)) != 0,
        OPS_LOG_E(context_->GetNodeName(), "pool_size %ld must equal power-of-two num_sets * 16.", poolSize_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(lruShape.GetDimNum() != 2 || lruShape.GetDim(0) != batch_ || lruShape.GetDim(1) != numSets_,
        OPS_LOG_E(context_->GetNodeName(), "lru_counter must be [batch, num_sets]."),
        return ge::GRAPH_FAILED);

    const int64_t requiredBytes = ComputeInstallRecordsRequiredBytes(batch_, rawSeq_, topK_);
    OPS_ERR_IF(installRecordsAllocatedBytes_ < requiredBytes,
        OPS_LOG_E(context_->GetNodeName(), "install_records bytes %ld smaller than required %ld.",
            installRecordsAllocatedBytes_, requiredBytes),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaInstallTiling::PostTiling()
{
    const int64_t workItemCount = batch_ * rawSeq_;
    const int64_t totalMt = workItemCount * M1B_ORD_MICROTASK_K;
    const int64_t mainBsLoopNum = (totalMt + coreNum_ - 1) / coreNum_;
    const int64_t tailBsLoopNum = totalMt / coreNum_;
    const int64_t recordsPerMt = topK_ / M1B_ORD_MICROTASK_K;
    const int64_t recordStride = AlignUp(recordsPerMt * INSTALL_RECORD_BYTES, SLAB_ALIGN_BYTES);
    const int64_t recordsBytes = AlignUp(totalMt * recordStride, SLAB_ALIGN_BYTES);
    const int64_t countsBytes = AlignUp(totalMt * INSTALL_COUNT_BYTES, SLAB_ALIGN_BYTES);
    workspaceBytes_ = DEFAULT_WORKSPACE_SIZE;

    const int64_t bitsetInt32s = (idRange_ + 31) / 32;
    const int64_t bitsetBytes = AlignUp(bitsetInt32s * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    const int64_t lruBytes = AlignUp(numSets_ * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
    const int64_t recordsUbBytes = recordStride;
    const int64_t stagedUbBytes = waysPerSet_ * STAGED_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t));
    const int64_t kvPayloadBytes = AlignUp(kvDim_ * static_cast<int64_t>(sizeof(uint16_t)), BLOCK_BYTES);
    const int64_t ropePayloadBytes = AlignUp(ropeDim_ * static_cast<int64_t>(sizeof(uint16_t)), BLOCK_BYTES);
    const int64_t kvPayloadOff = AlignUp(bitsetBytes + lruBytes + recordsUbBytes + stagedUbBytes, BLOCK_BYTES);
    installWorkBufBytes_ = kvPayloadOff + 2 * kvPayloadBytes + 2 * ropePayloadBytes;
    OPS_ERR_IF(installWorkBufBytes_ + UB_SAFETY_PAD > static_cast<int64_t>(ubSize_),
        OPS_LOG_E(context_->GetNodeName(),
            "DsaInstall UB requirement %ld bytes plus safety pad %ld exceeds platform UB %lu bytes for id_range %ld.",
            installWorkBufBytes_, UB_SAFETY_PAD, ubSize_, idRange_),
        return ge::GRAPH_FAILED);

    context_->SetTilingKey(0);
    context_->SetBlockDim(coreNum_);
    auto workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(workspaceBytes_);

    tilingData_.set_usedCoreNum(coreNum_);
    tilingData_.set_installWorkerNum(coreNum_);
    tilingData_.set_mainCoreBsLoopNum(mainBsLoopNum);
    tilingData_.set_tailCoreBsLoopNum(tailBsLoopNum);
    tilingData_.set_selTopKBlockSize(1);
    tilingData_.set_fullKvBlockSize(selectionBlockSize_);
    tilingData_.set_kRopeDim(ropeDim_);
    tilingData_.set_kvCacheDim(kvDim_);
    tilingData_.set_selKvBlockSize(selectionBlockSize_);
    tilingData_.set_fullMaxBlockNum(0);
    tilingData_.set_selMaxBlockNum((topK_ + selectionBlockSize_ - 1) / selectionBlockSize_);
    tilingData_.set_batchsize(workItemCount);
    tilingData_.set_rawSeq(rawSeq_);
    tilingData_.set_headnum(1);
    tilingData_.set_topk(topK_);
    tilingData_.set_poolSize(poolSize_);
    tilingData_.set_idRange(idRange_);
    tilingData_.set_numSets(numSets_);
    tilingData_.set_waysPerSet(waysPerSet_);
    tilingData_.set_maxRecordsPerSeq(topK_);
    tilingData_.set_stageTileSets(numSets_);
    tilingData_.set_installRecordsStride(recordStride);
    tilingData_.set_installCountsOffset(recordsBytes);
    tilingData_.set_m1bOrdMicrotaskK(M1B_ORD_MICROTASK_K);
    tilingData_.set_installRecordsRequiredBytes(recordsBytes + countsBytes);
    tilingData_.set_installRecordsAllocatedBytes(installRecordsAllocatedBytes_);
    tilingData_.set_compactAivRecordsOff(0);
    tilingData_.set_compactAivCountsOff(recordsBytes);
    tilingData_.set_metadataUpdate(metadataUpdate_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaInstallTiling::RunTiling()
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

ge::graphStatus Tiling4DsaInstall(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Tiling4DsaInstall", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    DsaInstallTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4DsaInstall(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DsaInstall)
    .Tiling(Tiling4DsaInstall)
    .TilingParse<DsaInstallCompileInfo>(TilingPrepare4DsaInstall);
}  // namespace optiling
