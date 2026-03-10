/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
// input
constexpr uint32_t TOKEN_X_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INDEX = 2;
constexpr uint32_t WEIGHT_UK_INDEX = 3;
constexpr uint32_t ROPE_SIN_INDEX = 7;
constexpr uint32_t KV_CACHE_INDEX = 9;
constexpr uint32_t KR_CACHE_INDEX = 10;
// output
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t QUERY_ROPE_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_INDEX = 3;
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;
constexpr uint32_t QUERY_NORM_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_INDEX = 6;
// Attribute
constexpr uint32_t ATTR_QUERY_NORM_FLAG_INDEX = 3;
constexpr uint32_t ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX = 4;
constexpr uint32_t ATTR_KV_QUANT_MODE_FLAG_INDEX = 5;

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

struct MlaPrologV3ProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t N { 0 };
    int64_t Hckv { 0 };
    int64_t He { 0 };
    int64_t Dr { 0 };
    int64_t Hcq { 0 };
};

ge::graphStatus GetMlaPrologV3ShapeDim(const gert::InferShapeContext* context, MlaPrologV3ProtoShapeParam &shapeParam)
{
    auto tokenXShape = context->GetRequiredInputShape(TOKEN_X_INDEX);      // (B, S, He) | (T, He)
    OPS_LOG_E_IF_NULL(context, tokenXShape, return ge::GRAPH_FAILED)
    auto weightUkShape = context->GetRequiredInputShape(WEIGHT_UK_INDEX);  // (N, D, Hckv)
    OPS_LOG_E_IF_NULL(context, weightUkShape, return ge::GRAPH_FAILED)
    auto weightDqShape = context->GetRequiredInputShape(WEIGHT_DQ_INDEX);  // (He, Hcq)
    OPS_LOG_E_IF_NULL(context, weightDqShape, return ge::GRAPH_FAILED)
    auto ropeSinShape = context->GetRequiredInputShape(ROPE_SIN_INDEX);    // (B, S, Dr) | (T, Dr)
    OPS_LOG_E_IF_NULL(context, ropeSinShape, return ge::GRAPH_FAILED)
    auto kvCacheShape = context->GetRequiredInputShape(KV_CACHE_INDEX);    // (B, Nkv, Skv, Hckv)
    OPS_LOG_E_IF_NULL(context, kvCacheShape, return ge::GRAPH_FAILED)
    auto krCacheShape = context->GetRequiredInputShape(KR_CACHE_INDEX);    // (B, Nkv, Skv, Dr)
    OPS_LOG_E_IF_NULL(context, krCacheShape, return ge::GRAPH_FAILED)

    OPS_LOG_E_IF(((tokenXShape->GetDimNum() != DIM_NUM_3) && (tokenXShape->GetDimNum() != DIM_NUM_2)),
        context, return ge::GRAPH_FAILED, "tokenXShape is not 2 or 3, but %zu", tokenXShape->GetDimNum());

    if (tokenXShape->GetDimNum() == DIM_NUM_3) {                // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.S = tokenXShape->GetDim(DIM_INDEX_1);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        shapeParam.T = tokenXShape->GetDim(DIM_INDEX_0);
        shapeParam.Dr = ropeSinShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.N = weightUkShape->GetDim(DIM_INDEX_0);
    shapeParam.Hckv = weightUkShape->GetDim(DIM_INDEX_2);
    shapeParam.Hcq = weightDqShape->GetDim(DIM_INDEX_1);
    return GRAPH_SUCCESS;
}

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaPrologV3ProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    auto queryShape = context->GetOutputShape(QUERY_INDEX);                 // query: (B, S, N, Hckv) | (T, N, Hckv)
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    auto queryRopeShape = context->GetOutputShape(QUERY_ROPE_INDEX);        // queryRope: (B, S, N, Dr) | (T, N, Dr)
    OPS_LOG_E_IF_NULL(context, queryRopeShape, return ge::GRAPH_FAILED)
    auto kvCacheOutShape = context->GetOutputShape(KV_CACHE_OUT_INDEX);     // kvCacheOut: (B, Nkv, Skv, Hckv)
    OPS_LOG_E_IF_NULL(context, kvCacheOutShape, return ge::GRAPH_FAILED)
    auto krCacheOutShape = context->GetOutputShape(KR_CACHE_OUT_INDEX);     // krCacheOut: (B, Nkv, Skv, Dr)
    OPS_LOG_E_IF_NULL(context, krCacheOutShape, return ge::GRAPH_FAILED)

    // Set output shape
    if (!shapeParam.isBsMerge) {
        queryShape->SetDimNum(DIM_NUM_4);                   // (B, S, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_3, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_4);               // (B, S, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.B);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.S);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_3, shapeParam.Dr);
    } else {
        queryShape->SetDimNum(DIM_NUM_3);                   // (T, N, Hckv)
        queryShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryShape->SetDim(DIM_INDEX_2, shapeParam.Hckv);

        queryRopeShape->SetDimNum(DIM_NUM_3);               // (T, N, Dr)
        queryRopeShape->SetDim(DIM_INDEX_0, shapeParam.T);
        queryRopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        queryRopeShape->SetDim(DIM_INDEX_2, shapeParam.Dr);
    }
    *kvCacheOutShape = *context->GetRequiredInputShape(KV_CACHE_INDEX);
    *krCacheOutShape = *context->GetRequiredInputShape(KR_CACHE_INDEX);

    // set output shape
    auto attrs = context->GetAttrs();
    auto *ckvQuantMode = attrs->GetAttrPointer<int>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    auto *weightQuantMode = attrs->GetAttrPointer<int>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);

    // dequantScaleQNope: (B*S, N ,1) | (T, N, 1). (1) if not enabled
    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OPS_LOG_E_IF_NULL(context, dequantScaleQNopeShape, return ge::GRAPH_FAILED)

    if (*ckvQuantMode == 1 && *weightQuantMode == 2) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3);                   // (B*S, N, 1) | (T, N, 1)
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, 1);                 // 1: Fix dim 1
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    }

    // queryNorm
    gert::Shape *queryNormShape = context->GetOutputShape(QUERY_NORM_INDEX);
    OPS_LOG_E_IF_NULL(context, queryNormShape, return ge::GRAPH_FAILED)
    gert::Shape *dequantScaleQNormShape = context->GetOutputShape(DEQUANT_SCALE_Q_NORM_INDEX);
    OPS_LOG_E_IF_NULL(context, dequantScaleQNormShape, return ge::GRAPH_FAILED)

    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    auto queryNormFlagPtr = attrs->GetAttrPointer<bool>(ATTR_QUERY_NORM_FLAG_INDEX);
    const bool queryNormFlag = (queryNormFlagPtr == nullptr) ? 0 : *queryNormFlagPtr;

    if (queryNormFlag) {
        if (shapeParam.isBsMerge) {
            // [T, Hcq]
            queryNormShape->SetDimNum(DIM_NUM_2);
            queryNormShape->SetDim(DIM_INDEX_0, shapeParam.T);
            queryNormShape->SetDim(DIM_INDEX_1, shapeParam.Hcq);
        } else {
            // [B, S, Hcq]
            queryNormShape->SetDimNum(DIM_NUM_3);
            queryNormShape->SetDim(DIM_INDEX_0, shapeParam.B);
            queryNormShape->SetDim(DIM_INDEX_1, shapeParam.S);
            queryNormShape->SetDim(DIM_INDEX_2, shapeParam.Hcq);
        }

        auto weightUqQrDesc = context->GetInputDesc(WEIGHT_UQ_QR_INDEX);
        OPS_LOG_E_IF_NULL(context, weightUqQrDesc, return ge::GRAPH_FAILED)

        if (weightUqQrDesc->GetDataType() == ge::DT_INT8) {
            dequantScaleQNormShape->SetDimNum(DIM_NUM_2);
            dequantScaleQNormShape->SetDim(DIM_INDEX_0, shapeParam.T);
            dequantScaleQNormShape->SetDim(DIM_INDEX_1, DIM_NUM_1);
        } else {
            dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
            dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
        }
    } else {
        queryNormShape->SetDimNum(DIM_NUM_1);
        queryNormShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
        dequantScaleQNormShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNormShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    MlaPrologV3ProtoShapeParam shapeParam {};
    auto apiRet = GetMlaPrologV3ShapeDim(context, shapeParam);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context get input shape failed");

    apiRet = SetMlaPrologV3ShapeDim(shapeParam, context);
    OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context set output shape failed");

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "Enter MlaPrologV3 infershape impl.");

    context->SetOutputDataType(QUERY_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(QUERY_ROPE_INDEX, context->GetRequiredInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(KV_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KV_CACHE_INDEX));
    context->SetOutputDataType(KR_CACHE_OUT_INDEX, context->GetRequiredInputDataType(KR_CACHE_INDEX));

    // full quant
    auto attrs = context->GetAttrs();
    auto ckvQuantModePtr = attrs->GetAttrPointer<int>(ATTR_KV_QUANT_MODE_FLAG_INDEX);
    const int ckvQuantMode = (ckvQuantModePtr == nullptr) ? 0 : *ckvQuantModePtr;
    auto weightQuantModePtr = attrs->GetAttrPointer<int>(ATTR_WEIGHT_QUANT_MODE_FLAG_INDEX);
    const int weightQuantMode = (weightQuantModePtr == nullptr) ? 0 : *weightQuantModePtr;

    bool isQuantQuery = (ckvQuantMode == 1 && weightQuantMode == 2);

    context->SetOutputDataType(QUERY_INDEX, isQuantQuery ? ge::DT_INT8 : ge::DT_BF16);
    context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

    context->SetOutputDataType(QUERY_NORM_INDEX, (
        context->GetInputDataType(WEIGHT_UQ_QR_INDEX) == ge::DT_INT8) ? ge::DT_INT8 : ge::DT_BF16);
    context->SetOutputDataType(DEQUANT_SCALE_Q_NORM_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV3).InferShape(InferShapeMlaPrologV3).InferDataType(InferDataTypeMlaPrologV3);
}  // namespace ops