/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
namespace {
constexpr int32_t TOPK_IDX = 0;
constexpr int32_t FULL_KV_ACTUAL_SEQ_IDX = 1;
constexpr int32_t POOL_IDS_IDX = 2;
constexpr int32_t ID_TO_SLOT_IDX = 3;
constexpr int32_t LRU_COUNTER_IDX = 4;
constexpr int32_t RAW_SEQ_ATTR_IDX = 0;
constexpr int32_t PLAN_OUT_IDX = 0;
constexpr int32_t INSTALL_OUT_IDX = 1;
constexpr int32_t ACTUAL_SEQ_OUT_IDX = 2;
constexpr int64_t PLAN_WIDTH = 2;
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

int64_t ComputeInstallRecordsAllocatedBytes(int64_t batch, int64_t rawSeq, int64_t topK)
{
    const int64_t recordsPerMt = (topK + M1B_ORD_MICROTASK_K - 1) / M1B_ORD_MICROTASK_K;
    const int64_t recordStride = AlignUp(recordsPerMt * INSTALL_RECORD_BYTES, SLAB_ALIGN_BYTES);
    const int64_t totalMt = batch * rawSeq * M1B_ORD_MICROTASK_K;
    const int64_t recordsBytes = AlignUp(totalMt * recordStride, SLAB_ALIGN_BYTES);
    const int64_t countsBytes = AlignUp(totalMt * INSTALL_COUNT_BYTES, SLAB_ALIGN_BYTES);
    return recordsBytes + countsBytes + SLAB_ALIGN_BYTES - 1;
}

int64_t NumElements(const gert::Shape* shape)
{
    int64_t total = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        total *= shape->GetDim(i);
    }
    return total;
}
}  // namespace

static ge::graphStatus InferShape4DsaPlan(gert::InferShapeContext* context)
{
    const gert::Shape* topkShape = context->GetInputShape(TOPK_IDX);
    OPS_LOG_E_IF_NULL(context, topkShape, return ge::GRAPH_FAILED);
    const size_t dims = topkShape->GetDimNum();
    OPS_ERR_IF((dims != 3 && dims != 4),
        OPS_LOG_E(context->GetNodeName(), "selection_topk_indices dim:%lu should be 3 or 4.", dims),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* rawSeqAttr = attrs->GetAttrPointer<int64_t>(RAW_SEQ_ATTR_IDX);
    const int64_t rawSeq = (rawSeqAttr != nullptr) ? *rawSeqAttr : 1;
    OPS_ERR_IF(rawSeq <= 0, OPS_LOG_E(context->GetNodeName(), "raw_seq must be positive, got %ld.", rawSeq),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(rawSeq > MAX_PROD_RAW_SEQ,
        OPS_LOG_E(context->GetNodeName(), "DsaPlan raw_seq %ld exceeds production cap %ld.",
            rawSeq, MAX_PROD_RAW_SEQ),
        return ge::GRAPH_FAILED);
    if (dims == 4) {
        OPS_ERR_IF(topkShape->GetDim(1) != rawSeq,
            OPS_LOG_E(context->GetNodeName(), "raw_seq attr %ld mismatches topk dim1 %ld.",
                rawSeq, topkShape->GetDim(1)),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(topkShape->GetDim(2) != 1,
            OPS_LOG_E(context->GetNodeName(), "DsaPlan supports only HEADS=1, got %ld.",
                topkShape->GetDim(2)),
            return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(rawSeq != 1,
            OPS_LOG_E(context->GetNodeName(), "3D topk layout requires raw_seq=1, got %ld.", rawSeq),
            return ge::GRAPH_FAILED);
    }

    const int64_t topkElems = NumElements(topkShape);
    const int64_t batch = topkShape->GetDim(0);
    OPS_ERR_IF(batch <= 0, OPS_LOG_E(context->GetNodeName(), "topk batch must be positive."),
        return ge::GRAPH_FAILED);
    const int64_t topK = topkShape->GetDim(dims - 1);
    OPS_ERR_IF(topK <= 0,
        OPS_LOG_E(context->GetNodeName(), "topk dim must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK % TOPK_GRANULARITY) != 0,
        OPS_LOG_E(context->GetNodeName(), "DsaPlan topK %ld must be divisible by %ld.",
            topK, TOPK_GRANULARITY),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(topK > MAX_PROD_TOPK,
        OPS_LOG_E(context->GetNodeName(), "DsaPlan topK %ld exceeds production cap %ld.",
            topK, MAX_PROD_TOPK),
        return ge::GRAPH_FAILED);
    const int64_t queryRows = topkElems / topK;

    gert::Shape* planShape = context->GetOutputShape(PLAN_OUT_IDX);
    planShape->SetDimNum(2);
    planShape->SetDim(0, topkElems);
    planShape->SetDim(1, PLAN_WIDTH);
    gert::Shape* installShape = context->GetOutputShape(INSTALL_OUT_IDX);
    installShape->SetDimNum(1);
    installShape->SetDim(0, ComputeInstallRecordsAllocatedBytes(batch, rawSeq, topK));
    gert::Shape* actualSeqShape = context->GetOutputShape(ACTUAL_SEQ_OUT_IDX);
    actualSeqShape->SetDimNum(1);
    actualSeqShape->SetDim(0, queryRows);

    const gert::Shape* poolShape = context->GetInputShape(POOL_IDS_IDX);
    OPS_LOG_E_IF_NULL(context, poolShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(poolShape->GetDimNum() != 2,
        OPS_LOG_E(context->GetNodeName(), "pool_ids must be [batch, pool_size], dim:%lu.",
            poolShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(poolShape->GetDim(0) != batch,
        OPS_LOG_E(context->GetNodeName(), "pool_ids shape[0] %ld must equal topk batch %ld.",
            poolShape->GetDim(0), batch),
        return ge::GRAPH_FAILED);
    const int64_t poolSize = poolShape->GetDim(1);
    OPS_ERR_IF((poolSize <= 0 || poolSize > MAX_PROD_POOL_SIZE || (poolSize % WAYS_PER_SET) != 0),
        OPS_LOG_E(context->GetNodeName(),
            "DsaPlan pool_size %ld must be positive, <= %ld, and divisible by %ld.",
            poolSize, MAX_PROD_POOL_SIZE, WAYS_PER_SET),
        return ge::GRAPH_FAILED);
    const int64_t numSets = poolSize / WAYS_PER_SET;
    OPS_ERR_IF((numSets & (numSets - 1)) != 0,
        OPS_LOG_E(context->GetNodeName(),
            "DsaPlan pool_size %ld must produce a power-of-two set count with %ld ways.",
            poolSize, WAYS_PER_SET),
        return ge::GRAPH_FAILED);
    const gert::Shape* idToSlotShape = context->GetInputShape(ID_TO_SLOT_IDX);
    OPS_LOG_E_IF_NULL(context, idToSlotShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(idToSlotShape->GetDimNum() != 2,
        OPS_LOG_E(context->GetNodeName(), "id_to_slot must be [batch, id_range], dim:%lu.",
            idToSlotShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(idToSlotShape->GetDim(0) != batch,
        OPS_LOG_E(context->GetNodeName(), "id_to_slot shape[0] %ld must equal batch %ld.",
            idToSlotShape->GetDim(0), batch),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(idToSlotShape->GetDim(1) <= 0,
        OPS_LOG_E(context->GetNodeName(), "id_to_slot id_range %ld must be positive.",
            idToSlotShape->GetDim(1)),
        return ge::GRAPH_FAILED);

    const gert::Shape* fullKvActualSeqShape = context->GetInputShape(FULL_KV_ACTUAL_SEQ_IDX);
    OPS_LOG_E_IF_NULL(context, fullKvActualSeqShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullKvActualSeqShape->GetDimNum() != 1,
        OPS_LOG_E(context->GetNodeName(), "full_kv_actual_seq must be [batch], dim:%lu.",
            fullKvActualSeqShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullKvActualSeqShape->GetDim(0) != batch,
        OPS_LOG_E(context->GetNodeName(), "full_kv_actual_seq shape[0] %ld must equal batch %ld.",
            fullKvActualSeqShape->GetDim(0), batch),
        return ge::GRAPH_FAILED);

    const gert::Shape* lruShape = context->GetInputShape(LRU_COUNTER_IDX);
    OPS_LOG_E_IF_NULL(context, lruShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF(lruShape->GetDimNum() != 2,
        OPS_LOG_E(context->GetNodeName(), "lru_counter must be [batch, num_sets], dim:%lu.",
            lruShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((lruShape->GetDim(0) != poolShape->GetDim(0) || lruShape->GetDim(1) != numSets),
        OPS_LOG_E(context->GetNodeName(), "lru_counter shape [%ld, %ld] must be [%ld, %ld].",
            lruShape->GetDim(0), lruShape->GetDim(1), poolShape->GetDim(0), numSets),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4DsaPlan(gert::InferDataTypeContext* context)
{
    for (int32_t i = 0; i < 3; ++i) {
        context->SetOutputDataType(i, ge::DT_INT32);
    }
    context->SetOutputDataType(INSTALL_OUT_IDX, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaPlan)
    .InferShape(InferShape4DsaPlan)
    .InferDataType(InferDtype4DsaPlan);
}  // namespace ops
