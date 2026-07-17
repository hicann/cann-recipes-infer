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
constexpr int32_t POOL_KV_IDX = 4;
constexpr int32_t POOL_ROPE_IDX = 5;
constexpr int32_t POOL_IDS_IDX = 6;
constexpr int32_t ID_TO_SLOT_IDX = 7;
constexpr int32_t LRU_COUNTER_IDX = 8;
constexpr int32_t POOL_KV_OUT_IDX = 0;
constexpr int32_t POOL_ROPE_OUT_IDX = 1;
constexpr int32_t POOL_IDS_OUT_IDX = 2;
constexpr int32_t ID_TO_SLOT_OUT_IDX = 3;
constexpr int32_t LRU_COUNTER_OUT_IDX = 4;
}  // namespace

static ge::graphStatus InferShape4DsaInstall(gert::InferShapeContext* context)
{
    const gert::Shape* poolKvShape = context->GetInputShape(POOL_KV_IDX);
    const gert::Shape* poolRopeShape = context->GetInputShape(POOL_ROPE_IDX);
    const gert::Shape* poolIdsShape = context->GetInputShape(POOL_IDS_IDX);
    const gert::Shape* idToSlotShape = context->GetInputShape(ID_TO_SLOT_IDX);
    const gert::Shape* lruShape = context->GetInputShape(LRU_COUNTER_IDX);
    OPS_LOG_E_IF_NULL(context, poolKvShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, poolRopeShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, poolIdsShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, idToSlotShape, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, lruShape, return ge::GRAPH_FAILED);
    *context->GetOutputShape(POOL_KV_OUT_IDX) = *poolKvShape;
    *context->GetOutputShape(POOL_ROPE_OUT_IDX) = *poolRopeShape;
    *context->GetOutputShape(POOL_IDS_OUT_IDX) = *poolIdsShape;
    *context->GetOutputShape(ID_TO_SLOT_OUT_IDX) = *idToSlotShape;
    *context->GetOutputShape(LRU_COUNTER_OUT_IDX) = *lruShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4DsaInstall(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(POOL_KV_OUT_IDX, context->GetInputDataType(POOL_KV_IDX));
    context->SetOutputDataType(POOL_ROPE_OUT_IDX, context->GetInputDataType(POOL_ROPE_IDX));
    context->SetOutputDataType(POOL_IDS_OUT_IDX, ge::DT_INT32);
    context->SetOutputDataType(ID_TO_SLOT_OUT_IDX, ge::DT_INT32);
    context->SetOutputDataType(LRU_COUNTER_OUT_IDX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaInstall)
    .InferShape(InferShape4DsaInstall)
    .InferDataType(InferDtype4DsaInstall);
}  // namespace ops
