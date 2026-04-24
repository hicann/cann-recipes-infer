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
 * \file dequant_swiglu_clamp_quant_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_SCALE = 1;
constexpr int64_t CONST_UNKNOW_SHAPE = -1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t INDEX_ATTR_DST_TYPE = 2;
constexpr int64_t INDEX_ATTR_ACTIVATE_DIM = 4;
static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
                                                                        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                                        ge::DT_INT8, ge::DT_HIFLOAT8};

static ge::graphStatus InferShape4DequantSwigluClampQuant(gert::InferShapeContext* context) {
  OPS_LOG_D(context, "Begin to do InferShape4DequantSwigluClampQuant.");

  const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
  gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
  gert::Shape* scaleShape = context->GetOutputShape(OUTPUT_IDX_SCALE);
  OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

  *yShape = *xShape;

  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  const int64_t *activateDim = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);
  const int64_t activateDimNum = (activateDim == nullptr) ? -1 : *activateDim;

  // 将切分轴转换为正数
  int64_t xShapeRank = static_cast<int64_t>(xShape->GetDimNum());
  int64_t selectDim = (activateDimNum >= 0) ? activateDimNum : (activateDimNum + xShapeRank);
  OPS_CHECK(selectDim >= xShapeRank,
           OPS_LOG_E(context, "activateDim must < xShapeRank, but is %ld, xShapeRank is %ld", selectDim, xShapeRank),
           return ge::GRAPH_FAILED);
  int64_t activateShape = xShape->GetDim(selectDim);
  int64_t outActivateShape = activateShape == CONST_UNKNOW_SHAPE ? CONST_UNKNOW_SHAPE : activateShape / NUM_TWO;
  OPS_CHECK((activateShape != CONST_UNKNOW_SHAPE) && (activateShape % NUM_TWO != 0),
           OPS_LOG_E(context, "The active axis must be an even number， but is %ld", activateShape),
           return ge::GRAPH_FAILED);
  // 设置Y的shape
  yShape->SetDim(selectDim, outActivateShape);
  // 设置Scale的shape
  *scaleShape = *yShape;
  scaleShape->SetDimNum(xShapeRank - 1);
  OPS_LOG_D(context, "End to do InferShape4DequantSwigluClampQuant");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4DequantSwigluClampQuant(gert::InferDataTypeContext* context) {
  OPS_LOG_D(context, "InferDtype4DequantSwigluClampQuant enter");

  auto attrsPtr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
  const int64_t *dstDtype = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
  const int64_t dstDtypeNum = (dstDtype == nullptr) ? NUM_TWO : *dstDtype;

  ge::DataType outDtype = static_cast<ge::DataType>(dstDtypeNum);
  OPS_CHECK((outDtype != ge::DT_FLOAT4_E2M1) && (outDtype != ge::DT_FLOAT4_E1M2) && (outDtype != ge::DT_FLOAT8_E4M3FN) &&
            (outDtype != ge::DT_FLOAT8_E5M2) && (outDtype != ge::DT_INT8) && (outDtype != ge::DT_HIFLOAT8),
           OPS_LOG_E(context, "dst_type is illegal, only supports 2(INT8) 40(FLOAT4_E2M1), 41(FLOAT4_E1M2), 35(FLOAT8E5M2), 36(FLOAT8E4M3), 34(HiFloat8)"),
           return ge::GRAPH_FAILED);

  context->SetOutputDataType(OUTPUT_IDX_Y, outDtype);
  context->SetOutputDataType(OUTPUT_IDX_SCALE, DT_FLOAT);
  OPS_LOG_D(context, "InferDtype4DequantSwigluClampQuant end");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantSwigluClampQuant)
    .InferShape(InferShape4DequantSwigluClampQuant)
    .InferDataType(InferDtype4DequantSwigluClampQuant);
}  // namespace ops