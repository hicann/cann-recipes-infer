/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "register/op_ct_impl_registry.h"
#include "nlohmann/json.hpp"

#define RED "\033[31m"
#define RESET "\033[0m"
using namespace ge;
using Json = nlohmann::json;

namespace ops {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t W_IDX_K_INPUT_INDEX = 5;
constexpr uint32_t W_IDX_PROJ_INPUT_INDEX = 6;
constexpr uint32_t W_Q_INPUT_INDEX = 3;
constexpr uint32_t W_K_INPUT_INDEX = 5;

// OUTPUT
constexpr uint32_t QUERY_OUTPUT_INDEX = 0;
constexpr uint32_t QUERY_SCALE_OUTPUT_INDEX = 1;
constexpr uint32_t WEIGHTS_INPUT_INDEX = 2;

ge::graphStatus InferShapeLightningIndexerPrologPto(gert::InferShapeContext *context) {
    const gert::Shape *tokenXShape = context->GetInputShape(TOKEN_X_INPUT_INDEX);
    if (tokenXShape == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto tokenXShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *kShape = context->GetInputShape(W_IDX_K_INPUT_INDEX);
    if (kShape == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto kShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *projShape = context->GetInputShape(W_IDX_PROJ_INPUT_INDEX);
    if (projShape == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto projShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    gert::Shape *queryOutputShape  = context->GetOutputShape(QUERY_OUTPUT_INDEX);
    if (queryOutputShape  == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto queryOutputShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    *queryOutputShape  = {tokenXShape->GetDim(0), projShape->GetDim(1), kShape->GetDim(1)};

    gert::Shape *queryScaleOutputShape  = context->GetOutputShape(QUERY_SCALE_OUTPUT_INDEX);
    if (queryScaleOutputShape  == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto queryScaleOutputShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    *queryScaleOutputShape  = {tokenXShape->GetDim(0), projShape->GetDim(1), 1};

    gert::Shape *weightsOutputShape  = context->GetOutputShape(WEIGHTS_INPUT_INDEX);
    if (weightsOutputShape  == nullptr) {
        printf("%sError:%s Failed to get InferShapeLightningIndexerPrologPto weightsOutputShape.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    *weightsOutputShape  = {tokenXShape->GetDim(0), projShape->GetDim(1)};
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeLightningIndexerPrologPto(gert::InferDataTypeContext *context) {
    if (context->GetNodeName() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(QUERY_OUTPUT_INDEX, ge::DT_INT8);
    context->SetOutputDataType(QUERY_SCALE_OUTPUT_INDEX, ge::DT_FLOAT16);
    context->SetOutputDataType(WEIGHTS_INPUT_INDEX, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferFormatLightningIndexerPrologPto(gert::InferFormatContext *context) {
    if (context->GetNodeName() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto weight_query = context->GetDynamicInputFormat(TOKEN_X_INPUT_INDEX, W_Q_INPUT_INDEX);
    weight_query->SetOriginFormat(Format::FORMAT_FRACTAL_NZ);
    weight_query->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);

    auto weight_key = context->GetDynamicInputFormat(TOKEN_X_INPUT_INDEX, W_K_INPUT_INDEX);
    weight_key->SetOriginFormat(Format::FORMAT_FRACTAL_NZ);
    weight_key->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
    auto weight_proj = context->GetDynamicInputFormat(TOKEN_X_INPUT_INDEX, W_IDX_PROJ_INPUT_INDEX);
    weight_proj->SetOriginFormat(Format::FORMAT_FRACTAL_NZ);
    weight_proj->SetStorageFormat(Format::FORMAT_FRACTAL_NZ);
    return GRAPH_SUCCESS;
}

IMPL_OP(LightningIndexerPrologPto).InferShape(InferShapeLightningIndexerPrologPto)
    .InferDataType(InferDataTypeLightningIndexerPrologPto);


ge::graphStatus GetOpspecificInfoLightningIndexerPrologPto(const gert::OpCheckContext *context, ge::AscendString &result)
{
    Json op_specific_info;
    op_specific_info["tileFwkOp"] = "true";
    result = op_specific_info.dump().c_str();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcOpParamLightningIndexerPrologPto(gert::ExeResGenerationContext *context) {
    ge::AscendString name = "pto aicpu kfc server";
    ge::AscendString reuse_key = "pto kfc_stream";
    gert::StreamInfo stream_info;
    stream_info.name = name;
    stream_info.reuse_key = reuse_key;
    std::vector<int64_t> stream_depend_value_list(0);
    stream_info.depend_value_input_indices = stream_depend_value_list;
    stream_info.required = true;
    std::vector<gert::StreamInfo> stream_info_vec(0);
    stream_info_vec.push_back(stream_info);
    context->SetAttachedStreamInfos(stream_info_vec);

    gert::SyncResInfo sync_res_info;
    sync_res_info.type = gert::SyncResType::SYNC_RES_NOTIFY;
    sync_res_info.name = name;
    sync_res_info.reuse_key = reuse_key;
    sync_res_info.required = true;
    std::vector<gert::SyncResInfo> sync_info_vec(0);
    sync_info_vec.push_back(sync_res_info);

    sync_res_info.name = "ast_tail";
    sync_res_info.reuse_key = "ast_tail_sync";
    sync_info_vec.push_back(sync_res_info);
    context->SetSyncResInfos(sync_info_vec);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_CT(LightningIndexerPrologPto)
    .GetOpSpecificInfo(GetOpspecificInfoLightningIndexerPrologPto)
    .CalcOpParam(CalcOpParamLightningIndexerPrologPto);
}  // namespace ops