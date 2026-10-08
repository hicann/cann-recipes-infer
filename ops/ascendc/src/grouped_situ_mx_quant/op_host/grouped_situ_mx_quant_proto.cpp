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
 * \file grouped_situ_mx_quant_proto.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "error/ops_error.h"

namespace ops {
namespace {
constexpr int64_t BLOCK_SIZE = 32;   // MXFP8 group_size, fixed
constexpr int64_t ALIGN_NUM = 2;     // mxscale last axis interleave factor
}  // namespace

// y = x with last axis halved (gate/up -> activation), dtype fp8_e4m3fn.
// mxscale follows dynamic_mx_quant tail-axis convention:
//   shape = x[..:-1] + [ceil(ceil(EPR/32)/2), 2], dtype e8m0, EPR = x[-1]/2.
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x = context->GetInputShape(0);
    gert::Shape* y = context->GetOutputShape(0);
    gert::Shape* scale = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, x, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, y, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL(context, scale, return ge::GRAPH_FAILED);

    const int64_t rank = x->GetDimNum();
    OPS_ERR_IF(rank < 1,
        OPS_LOG_E(context->GetNodeName(), "x rank must be >= 1."), return ge::GRAPH_FAILED);
    const int64_t lastDim = x->GetDim(rank - 1);
    OPS_ERR_IF(lastDim != -1 && (lastDim <= 0 || lastDim % ALIGN_NUM != 0),
        OPS_LOG_E(context->GetNodeName(), "x last dim must be positive and even."), return ge::GRAPH_FAILED);

    const int64_t epr = (lastDim == -1) ? -1 : lastDim / ALIGN_NUM;

    *y = *x;
    y->SetDim(rank - 1, epr);

    *scale = *x;
    int64_t scaleDim = -1;
    if (epr != -1) {
        scaleDim = (epr + BLOCK_SIZE - 1) / BLOCK_SIZE;      // group count = ceil(EPR/32)
        scaleDim = (scaleDim + ALIGN_NUM - 1) / ALIGN_NUM;   // interleave halve
    }
    scale->SetDim(rank - 1, scaleDim);
    scale->AppendDim(ALIGN_NUM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedSituMxQuant).InferShape(InferShape).InferDataType(InferDtype);
}  // namespace ops
