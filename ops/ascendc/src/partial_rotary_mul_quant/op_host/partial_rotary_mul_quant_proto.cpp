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
 * \file partial_rotary_mul_quant_proto.cpp
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_
#define OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_IDX_COS = 1;
const int32_t INPUT_IDX_SIN = 2;
const int32_t INDEX_OUTPUT_Y = 0;
const int32_t INDEX_ATTR_MODE = 1;
const int32_t INDEX_ATTR_SLICE = 2;
const int32_t INDEX_ATTR_SCALE = 3;


static ge::graphStatus InferShape4PartialRotaryMulQuant(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Begin to do InferShape4PartialRotaryMulQuant.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED);

    const gert::Shape* cosShape = context->GetInputShape(INPUT_IDX_COS);
    OPS_LOG_E_IF_NULL(context, cosShape, return ge::GRAPH_FAILED);

    const gert::Shape* sinShape = context->GetInputShape(INPUT_IDX_SIN);
    OPS_LOG_E_IF_NULL(context, sinShape, return ge::GRAPH_FAILED);

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    *yShape = *xShape;

    OPS_LOG_I(context->GetNodeName(), "End to do InferShape4PartialRotaryMulQuant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4PartialRotaryMulQuant(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "InferDtype4PartialRotaryMulQuant enter");
    context->SetOutputDataType(INDEX_OUTPUT_Y, ge::DT_UINT8);
    OPS_LOG_I(context->GetNodeName(), "InferDtype4PartialRotaryMulQuant end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PartialRotaryMulQuant)
    .InferShape(InferShape4PartialRotaryMulQuant)
    .InferDataType(InferDtype4PartialRotaryMulQuant);
}  // namespace ops

#endif