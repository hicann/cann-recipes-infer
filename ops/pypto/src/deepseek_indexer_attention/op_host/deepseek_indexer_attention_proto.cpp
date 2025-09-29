/* *
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  * See LICENSE in the root of
the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "register/op_ct_impl_registry.h"
#include "nlohmann/json.hpp"

using namespace ge;
using Json = nlohmann::json;
namespace ops {
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_UK_INPUT_INDEX = 3;

// OUTPUT
constexpr uint32_t ATTEN_RES_OUTPUT_INDEX = 0;
ge::graphStatus InferShapeSparseAttentionPto(gert::InferShapeContext *context) {
    const gert::Shape *tokenXShape = context->GetInputShape(TOKEN_X_INPUT_INDEX);
    if (tokenXShape == nullptr) {
        printf("Failed to get SparseAttentionPto tokenXShape.\n");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *weightUkShape = context->GetInputShape(WEIGHT_UK_INPUT_INDEX);
    if (weightUkShape == nullptr) {
        printf("Failed to get SparseAttentionPto weightUkShape.\n");
        return ge::GRAPH_FAILED;
    }

    // output: (B, N2, S, k)
    gert::Shape *outputShape = context->GetOutputShape(ATTEN_RES_OUTPUT_INDEX);
    if (outputShape == nullptr) {
        printf("Failed to get SparseAttentionPto outputShape.\n");
        return ge::GRAPH_FAILED;
    }

    *outputShape = {tokenXShape->GetDim(0), tokenXShape->GetDim(1), weightUkShape->GetDim(0), weightUkShape->GetDim(2)};
    printf("SparseAttentionPto inferOutputShape sccuess, op node name is %s.\n", context->GetNodeName());
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseAttentionPto(gert::InferDataTypeContext *context) {
    if (context->GetNodeName() == nullptr) {
        printf("Failed to get node name.\n");
        return ge::GRAPH_FAILED;
    }
    ge::DataType tokenXDtype = context->GetInputDataType(TOKEN_X_INPUT_INDEX);
    context->SetOutputDataType(ATTEN_RES_OUTPUT_INDEX, tokenXDtype);
    printf("SparseAttentionPto infer outputDataType sccuess, op node name is %s.\n", context->GetNodeName());
    return GRAPH_SUCCESS;
}

IMPL_OP(SparseAttentionPto).InferShape(InferShapeSparseAttentionPto).InferDataType(InferDataTypeSparseAttentionPto);

ge::graphStatus GetOpspecificInfoSparseAttentionPto(const gert::OpCheckContext *context, ge::AscendString &result) {
    Json op_specific_info;
    op_specific_info["tileFwkOp"] = "true";
    result = op_specific_info.dump().c_str();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcOpParamSparseAttentionPto(gert::ExeResGenerationContext *context) {
    printf("CalcOpParamSparseAttentionPto.\n");
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

IMPL_OP_CT(SparseAttentionPto)
    .GetOpSpecificInfo(GetOpspecificInfoSparseAttentionPto)
    .CalcOpParam(CalcOpParamSparseAttentionPto);
} // namespace ops
