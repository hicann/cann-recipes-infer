/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_antiquant_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 4;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 10;

ge::graphStatus InferShapeSparseFlashAttentionAntiquant(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttentionAntiquant", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutQueryPtr, return ge::GRAPH_FAILED);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    attentionOutShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        attentionOutShape->SetDim(0, queryShape->GetDim(0));
        attentionOutShape->SetDim(1, queryShape->GetDim(1));
        attentionOutShape->SetDim(2, queryShape->GetDim(2)); // 2:dim2
        attentionOutShape->SetDim(3, queryShape->GetDim(3) - ropeHeadDim); // 3:dim3
    } else { // TND
        attentionOutShape->SetDim(0, queryShape->GetDim(0));
        attentionOutShape->SetDim(1, queryShape->GetDim(1));
        attentionOutShape->SetDim(2, queryShape->GetDim(2) - ropeHeadDim); // 2:dim2
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseFlashAttentionAntiquant(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("SparseFlashAttentionAntiquant", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseFlashAttentionAntiquant)
    .InferShape(InferShapeSparseFlashAttentionAntiquant)
    .InferDataType(InferDataTypeSparseFlashAttentionAntiquant);
} // namespace ops

