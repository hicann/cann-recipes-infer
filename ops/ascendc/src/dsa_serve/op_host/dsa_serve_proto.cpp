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
constexpr int32_t SEL_KV_OUT_IDX = 0;
constexpr int32_t SEL_ROPE_OUT_IDX = 1;
constexpr int64_t PLAN_WIDTH = 2;
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

static ge::graphStatus InferShape4DsaServe(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t* rawSeqAttr = attrs->GetAttrPointer<int64_t>(RAW_SEQ_ATTR_IDX);
    const int64_t rawSeq = (rawSeqAttr != nullptr) ? *rawSeqAttr : 1;
    const int64_t* topkAttr = attrs->GetAttrPointer<int64_t>(TOPK_ATTR_IDX);
    const int64_t topK = (topkAttr != nullptr) ? *topkAttr : 2048;
    const int64_t* selectionBlockSizeAttr = attrs->GetAttrPointer<int64_t>(SELECTION_BLOCK_SIZE_ATTR_IDX);
    const int64_t selectionBlockSize = (selectionBlockSizeAttr != nullptr) ? *selectionBlockSizeAttr : 128;
    const int64_t* compactLayoutAttr = attrs->GetAttrPointer<int64_t>(COMPACT_LAYOUT_ATTR_IDX);
    const int64_t compactLayout = (compactLayoutAttr != nullptr) ? *compactLayoutAttr : 1;
    OPS_ERR_IF((compactLayout != 0 && compactLayout != 1),
        OPS_LOG_E(context->GetNodeName(), "DsaServe compact_layout must be 0 or 1, got %ld.", compactLayout),
        return ge::GRAPH_FAILED);
    const bool compact = compactLayout != 0;
    const int64_t maxRawSeq = compact ? PROD_MAX_RAW_SEQ : LEGACY_MAX_RAW_SEQ;
    const int64_t maxTopK = compact ? PROD_MAX_TOPK : LEGACY_MAX_TOPK;
    const int64_t maxPoolSize = compact ? PROD_MAX_POOL_SIZE : LEGACY_MAX_POOL_SIZE;
    OPS_ERR_IF((rawSeq <= 0 || rawSeq > maxRawSeq),
        OPS_LOG_E(context->GetNodeName(), "DsaServe raw_seq %ld must be in (0, %ld] for compact_layout=%ld.",
            rawSeq, maxRawSeq, compactLayout),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((topK <= 0 || topK > maxTopK || (topK % TOPK_GRANULARITY) != 0),
        OPS_LOG_E(context->GetNodeName(),
            "DsaServe topK %ld must be positive, <= %ld, and divisible by %ld for compact_layout=%ld.",
            topK, maxTopK, TOPK_GRANULARITY, compactLayout),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionBlockSize < MIN_SELECTION_BLOCK_SIZE ||
                   selectionBlockSize > MAX_SELECTION_BLOCK_SIZE ||
                   (selectionBlockSize & (selectionBlockSize - 1)) != 0),
        OPS_LOG_E(context->GetNodeName(),
            "DsaServe selection_block_size %ld must be a power of two in [%ld, %ld].",
            selectionBlockSize, MIN_SELECTION_BLOCK_SIZE, MAX_SELECTION_BLOCK_SIZE),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(compact && (topK % selectionBlockSize) != 0,
        OPS_LOG_E(context->GetNodeName(),
            "DsaServe compact topK %ld must be divisible by selection_block_size %ld.",
            topK, selectionBlockSize),
        return ge::GRAPH_FAILED);

    const gert::Shape* planShape = context->GetInputShape(PLAN_IDX);
    OPS_LOG_E_IF_NULL(context, planShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF((planShape->GetDimNum() != 2 || planShape->GetDim(1) != PLAN_WIDTH),
        OPS_LOG_E(context->GetNodeName(), "plan must be [rows, 2]."),
        return ge::GRAPH_FAILED);
    const int64_t planRows = planShape->GetDim(0);
    OPS_ERR_IF(planRows <= 0,
        OPS_LOG_E(context->GetNodeName(), "DsaServe rows must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((!compact && planRows > LEGACY_MAX_ROWS),
        OPS_LOG_E(context->GetNodeName(), "DsaServe rows %ld exceed legacy cap %ld for compact_layout=%ld.",
            planRows, LEGACY_MAX_ROWS, compactLayout),
        return ge::GRAPH_FAILED);

    const gert::Shape* fullKvShape = context->GetInputShape(FULL_KV_IDX);
    const gert::Shape* fullRopeShape = context->GetInputShape(FULL_ROPE_IDX);
    const gert::Shape* poolKvShape = context->GetInputShape(POOL_KV_IDX);
    const gert::Shape* poolRopeShape = context->GetInputShape(POOL_ROPE_IDX);
    const gert::Shape* selectionKvInShape = context->GetInputShape(SELECTION_KV_IDX);
    const gert::Shape* selectionRopeInShape = context->GetInputShape(SELECTION_ROPE_IDX);
    OPS_LOG_E_IF_NULL(context, fullKvShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, fullRopeShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, poolKvShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, poolRopeShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, selectionKvInShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, selectionRopeInShape, return ge::GRAPH_FAILED);
    OPS_ERR_IF((fullKvShape->GetDimNum() != 3 || fullRopeShape->GetDimNum() != 3 ||
                   poolKvShape->GetDimNum() != 3 || poolRopeShape->GetDimNum() != 3 ||
                   selectionKvInShape->GetDimNum() != 3 || selectionRopeInShape->GetDimNum() != 3),
        OPS_LOG_E(context->GetNodeName(), "full/pool/selection KV and rope tensors must be 3D."),
        return ge::GRAPH_FAILED);
    const int64_t batch = fullKvShape->GetDim(0);
    OPS_ERR_IF(batch <= 0,
        OPS_LOG_E(context->GetNodeName(), "DsaServe batch must be positive."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((!compact && batch > LEGACY_MAX_BATCH),
        OPS_LOG_E(context->GetNodeName(), "DsaServe batch %ld exceeds legacy cap %ld for compact_layout=%ld.",
            batch, LEGACY_MAX_BATCH, compactLayout),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(planRows != batch * rawSeq * topK,
        OPS_LOG_E(context->GetNodeName(), "plan rows %ld must equal batch*raw_seq*topK %ld.",
            planRows, batch * rawSeq * topK),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((fullRopeShape->GetDim(0) != batch || poolKvShape->GetDim(0) != batch ||
                   poolRopeShape->GetDim(0) != batch),
        OPS_LOG_E(context->GetNodeName(), "all DsaServe payload tensors must share batch."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullKvShape->GetDim(1) <= 0 || (!compact && fullKvShape->GetDim(1) > LEGACY_MAX_FULL_SEQ),
        OPS_LOG_E(context->GetNodeName(), "full_seq %ld unsupported for DsaServe compact_layout=%ld.",
            fullKvShape->GetDim(1), compactLayout),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(poolKvShape->GetDim(1) <= 0 || poolKvShape->GetDim(1) > maxPoolSize ||
                   poolRopeShape->GetDim(1) != poolKvShape->GetDim(1),
        OPS_LOG_E(context->GetNodeName(), "pool size unsupported or mismatched for DsaServe compact_layout=%ld.",
            compactLayout),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((fullKvShape->GetDim(2) <= 0 || fullKvShape->GetDim(2) > MAX_KV_DIM ||
                   poolKvShape->GetDim(2) != fullKvShape->GetDim(2)),
        OPS_LOG_E(context->GetNodeName(), "kv dim unsupported or mismatched for DsaServe gate."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((fullRopeShape->GetDim(2) <= 0 || fullRopeShape->GetDim(2) > MAX_ROPE_DIM ||
                   poolRopeShape->GetDim(2) != fullRopeShape->GetDim(2)),
        OPS_LOG_E(context->GetNodeName(), "rope dim unsupported or mismatched for DsaServe gate."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(fullRopeShape->GetDim(1) != fullKvShape->GetDim(1),
        OPS_LOG_E(context->GetNodeName(), "full rope seq must match full kv seq."),
        return ge::GRAPH_FAILED);

    const int64_t selectionBlocks =
        compact ? ((planRows + selectionBlockSize - 1) / selectionBlockSize) : planRows;
    OPS_ERR_IF((selectionKvInShape->GetDim(0) != selectionBlocks ||
                   selectionKvInShape->GetDim(1) != selectionBlockSize ||
                   selectionKvInShape->GetDim(2) != fullKvShape->GetDim(2)),
        OPS_LOG_E(context->GetNodeName(), "selection_kv_cache shape mismatches DsaServe output shape."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((selectionRopeInShape->GetDim(0) != selectionBlocks ||
                   selectionRopeInShape->GetDim(1) != selectionBlockSize ||
                   selectionRopeInShape->GetDim(2) != fullRopeShape->GetDim(2)),
        OPS_LOG_E(context->GetNodeName(), "selection_k_rope shape mismatches DsaServe output shape."),
        return ge::GRAPH_FAILED);
    gert::Shape* selectionKvShape = context->GetOutputShape(SEL_KV_OUT_IDX);
    *selectionKvShape = *selectionKvInShape;
    gert::Shape* selectionRopeShape = context->GetOutputShape(SEL_ROPE_OUT_IDX);
    *selectionRopeShape = *selectionRopeInShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4DsaServe(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(SEL_KV_OUT_IDX, context->GetInputDataType(SELECTION_KV_IDX));
    context->SetOutputDataType(SEL_ROPE_OUT_IDX, context->GetInputDataType(SELECTION_ROPE_IDX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaServe)
    .InferShape(InferShape4DsaServe)
    .InferDataType(InferDtype4DsaServe);
}  // namespace ops
