/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/compute_node_info.h"

#define RED "\033[31m"
#define RESET "\033[0m"

namespace tiling {
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t Q_NORM_INDEX = 1;
constexpr uint32_t Q_NORM_SCALE_INDEX = 2;
constexpr uint32_t WQB_TENSOR_INDEX = 3;
constexpr uint32_t WQB_TENSOR_SCALE_INDEX = 4;
constexpr uint32_t WK_TENSOR_INDEX = 5;
constexpr uint32_t WPROJ_TENSOR_INDEX = 6;
constexpr uint32_t GAMAK_TENSOR_INDEX = 7;
constexpr uint32_t BETA_TENSOR_INDEX = 8;
constexpr uint32_t COS_IDX_ROPE_INDEX = 9;
constexpr uint32_t SIN_IDX_ROPE_INDEX = 10;
constexpr uint32_t HADAMARDQ_INDEX = 11;
constexpr uint32_t HADAMARDK_INDEX = 12;
constexpr uint32_t KCACHE_INDEX = 13;
constexpr uint32_t KCACHESCALE_INDEX = 14;
constexpr uint32_t IDX_K_CACHE_INDEX = 15;
constexpr uint32_t IDX_BLOCK_TABLE_INDEX = 16;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_KEY_INDEX = 17;
constexpr uint32_t ACTUAL_SEQ_LENGTHS_QUERY_INDEX = 18;
constexpr uint32_t AXIS_0 = 0;
constexpr uint32_t AXIS_1 = 1;
constexpr uint32_t AXIS_2 = 2;
constexpr uint32_t DIMNUM_1 = 1;
constexpr uint32_t DIMNUM_2 = 2;
constexpr uint32_t DIMNUM_3 = 3;
constexpr uint32_t DIMNUM_4 = 4;

constexpr uint64_t Lightning_Indexer_Prolog_PTO_ConfigKey = uint64_t(100000000UL);
struct LightningIndexerPrologPtoCompileInfo {
    uint32_t block_dim_num = 0;
};

template<typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}

std::string GetDataTypeStr(ge::DataType datatype) {
    const std::string unkwon_datatype = "unknown datatype";
    const std::map<ge::DataType, std::string> dataType2Str = {
        {ge::DT_FLOAT, "float"},
        {ge::DT_INT8, "int8"},
        {ge::DT_BF16, "bfloat16"},
        {ge::DT_FLOAT16, "float16"},
        {ge::DT_INT64, "int64"},
        {ge::DT_INT32, "int32"},
    };

    if (dataType2Str.find(datatype) != dataType2Str.end()) {
        return dataType2Str.find(datatype)->second;
    } else {
        return unkwon_datatype;
    }
}

ge::graphStatus CheckTensorDimNumAndDtype(const gert::StorageShape* tensorShape, const gert::CompileTimeTensorDesc* tensorDesc,
    const uint32_t &dimNum, const ge::DataType &dataType, const uint32_t &index) {
    ge::DataType dtype = tensorDesc->GetDataType();
    if(dtype != dataType) {
        printf("%sError:%s Lightning indexer prolog pto get invalid dtype, tensor index is: %u, it shoule be %s, get tensor dataType is: %s.\n",
            RED, RESET, index, GetDataTypeStr(dataType).c_str(), GetDataTypeStr(dtype).c_str());
        return ge::GRAPH_FAILED;
    }

    if (tensorShape->GetStorageShape().GetDimNum() != dimNum) {
        printf("%sError:%s Shape error, tensor index is: %u, dimNum should be %u, get tensor dimNum is: %zu.\n",
            RED, RESET, index, dimNum, tensorShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerPrologPtoCheck(gert::TilingContext *context) {
    auto tokenXShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, TOKEN_X_INPUT_INDEX);
    auto tokenXDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, TOKEN_X_INPUT_INDEX);
    if (tokenXShape == nullptr || tokenXDesc == nullptr ||
        CheckTensorDimNumAndDtype(tokenXShape, tokenXDesc, DIMNUM_2, ge::DT_BF16, TOKEN_X_INPUT_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor tokenX.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto qNormShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, Q_NORM_INDEX);
    auto qNormDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, Q_NORM_INDEX);
    if (qNormShape == nullptr || qNormDesc == nullptr ||
        CheckTensorDimNumAndDtype(qNormShape, qNormDesc, DIMNUM_2, ge::DT_INT8, Q_NORM_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor qNorm.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto qNormScaleShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, Q_NORM_SCALE_INDEX);
    auto qNormScaleDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, Q_NORM_SCALE_INDEX);
    if (qNormScaleShape == nullptr || qNormScaleDesc == nullptr ||
        CheckTensorDimNumAndDtype(qNormScaleShape, qNormScaleDesc, DIMNUM_2, ge::DT_FLOAT, Q_NORM_SCALE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor qNormScale.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto wQbShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, WQB_TENSOR_INDEX);
    auto wQbDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, WQB_TENSOR_INDEX);
    if (wQbDesc->GetFormat().GetStorageFormat() != ge::Format::FORMAT_FRACTAL_NZ) {
        printf("%sError:%s Tensor w_qb format should be FRACTAL_NZ.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    if (wQbShape == nullptr || wQbDesc == nullptr ||
        CheckTensorDimNumAndDtype(wQbShape, wQbDesc, DIMNUM_4, ge::DT_INT8, WQB_TENSOR_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor w_qb\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto wQbScaleShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, WQB_TENSOR_SCALE_INDEX);
    auto wQbScaleDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, WQB_TENSOR_SCALE_INDEX);
    if (wQbScaleShape == nullptr || wQbScaleDesc == nullptr ||
        CheckTensorDimNumAndDtype(wQbScaleShape, wQbScaleDesc, DIMNUM_2, ge::DT_FLOAT, WQB_TENSOR_SCALE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor wQbScale.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto wkShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, WK_TENSOR_INDEX);
    auto wkDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, WK_TENSOR_INDEX);
    if (wkDesc->GetFormat().GetStorageFormat() != ge::Format::FORMAT_FRACTAL_NZ) {
        printf("%sError:%s Tensor w_k format should be FRACTAL_NZ.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    if (wkShape == nullptr || wkDesc == nullptr ||
        CheckTensorDimNumAndDtype(wkShape, wkDesc, DIMNUM_4, ge::DT_BF16, WK_TENSOR_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor wk.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto wProjShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, WPROJ_TENSOR_INDEX);
    auto wProjDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, WPROJ_TENSOR_INDEX);
    if (wProjDesc->GetFormat().GetStorageFormat() != ge::Format::FORMAT_FRACTAL_NZ) {
        printf("%sError:%s Tensor w_proj format should be FRACTAL_NZ.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }
    if (wProjShape == nullptr || wProjDesc == nullptr ||
        CheckTensorDimNumAndDtype(wProjShape, wProjDesc, DIMNUM_4, ge::DT_BF16, WPROJ_TENSOR_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor wproj.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto lnGammaKShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, GAMAK_TENSOR_INDEX);
    auto lnGammaKDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, GAMAK_TENSOR_INDEX);
    if (lnGammaKShape == nullptr || lnGammaKDesc == nullptr ||
        CheckTensorDimNumAndDtype(lnGammaKShape, lnGammaKDesc, DIMNUM_1, ge::DT_BF16, GAMAK_TENSOR_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor lnGammaK.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto lnBetaKShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, BETA_TENSOR_INDEX);
    auto lnBetaKDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, BETA_TENSOR_INDEX);
    if (lnBetaKShape == nullptr || lnBetaKDesc == nullptr ||
        CheckTensorDimNumAndDtype(lnBetaKShape, lnBetaKDesc, DIMNUM_1, ge::DT_BF16, BETA_TENSOR_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor lnBetaK.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto cosIdxRopeShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, COS_IDX_ROPE_INDEX);
    auto cosIdxRopeDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, COS_IDX_ROPE_INDEX);
    if (cosIdxRopeShape == nullptr || cosIdxRopeDesc == nullptr ||
        CheckTensorDimNumAndDtype(cosIdxRopeShape, cosIdxRopeDesc, DIMNUM_2, ge::DT_BF16, COS_IDX_ROPE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor cosIdxRope.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto sinIdxRopeShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, SIN_IDX_ROPE_INDEX);
    auto sinIdxRopeDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, SIN_IDX_ROPE_INDEX);
    if (sinIdxRopeShape == nullptr || sinIdxRopeDesc == nullptr ||
        CheckTensorDimNumAndDtype(sinIdxRopeShape, sinIdxRopeDesc, DIMNUM_2, ge::DT_BF16, SIN_IDX_ROPE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor sinIdxRope.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto hadamardQShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, HADAMARDQ_INDEX);
    auto hadamardQDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, HADAMARDQ_INDEX);
    if (hadamardQShape == nullptr || hadamardQDesc == nullptr ||
        CheckTensorDimNumAndDtype(hadamardQShape, hadamardQDesc, DIMNUM_2, ge::DT_BF16, HADAMARDQ_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor hadamardQ.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto hadamardKShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, HADAMARDK_INDEX);
    auto hadamardKDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, HADAMARDK_INDEX);
    if (hadamardKShape == nullptr || hadamardKDesc == nullptr ||
        CheckTensorDimNumAndDtype(hadamardKShape, hadamardKDesc, DIMNUM_2, ge::DT_BF16, HADAMARDK_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor hadamardK.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto kCahceShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, KCACHE_INDEX);
    auto kCahceDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, KCACHE_INDEX);
    if (kCahceShape == nullptr || kCahceDesc == nullptr ||
        CheckTensorDimNumAndDtype(kCahceShape, kCahceDesc, DIMNUM_4, ge::DT_INT8, KCACHE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor kCahce.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto kCahceScaleShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, KCACHESCALE_INDEX);
    auto kCahceScaleDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, KCACHESCALE_INDEX);
    if (kCahceScaleShape == nullptr || kCahceScaleDesc == nullptr ||
        CheckTensorDimNumAndDtype(kCahceScaleShape, kCahceScaleDesc, DIMNUM_3, ge::DT_FLOAT16, KCACHESCALE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor kCahceScale.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    auto cacheIndexShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, IDX_K_CACHE_INDEX);
    auto cacheIndexDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, IDX_K_CACHE_INDEX);
    if (cacheIndexShape == nullptr || cacheIndexDesc == nullptr ||
        CheckTensorDimNumAndDtype(cacheIndexShape, cacheIndexDesc, DIMNUM_1, ge::DT_INT64, IDX_K_CACHE_INDEX)) {
        printf("%sError:%s Lightning indexer prolog pto get invalid input tensor kCahceIndex.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    if (tokenXShape->GetStorageShape().GetDimNum() != 2 || qNormShape->GetStorageShape().GetDimNum() != 2 ||
        sinIdxRopeShape->GetStorageShape().GetDimNum() != 2 || cacheIndexShape->GetStorageShape().GetDimNum() != 1 ||
        cosIdxRopeShape->GetStorageShape().GetDimNum() != 2 || qNormScaleShape->GetStorageShape().GetDimNum() != 2) {
        printf("%sError:%s Lightning indexer prolog pto get invalid dim num.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    if (tokenXShape->GetStorageShape().GetDim(AXIS_0) != qNormShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != cosIdxRopeShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != sinIdxRopeShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != cacheIndexShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != qNormScaleShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) < 1) {
        printf("%sError:%s Lightning indexer prolog pto get invalid t axis value.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingLightningIndexerPrologPto(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetNodeName() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (LightningIndexerPrologPtoCheck(context) != ge::GRAPH_SUCCESS) {
        printf("%sError:%s Lightning indexer prolog pto get invalid tensor.\n", RED, RESET);
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = Lightning_Indexer_Prolog_PTO_ConfigKey;
    context->SetTilingKey(tilingKey);

    auto compileInfo = reinterpret_cast<const LightningIndexerPrologPtoCompileInfo*>(context->GetCompileInfo());
    context->SetBlockDim(compileInfo->block_dim_num);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseLightningIndexerPrologPto(gert::TilingParseContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfo = GetCompileInfoPtr<LightningIndexerPrologPtoCompileInfo>(context);
    compileInfo->block_dim_num = platformInfo->GetCoreNum();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(LightningIndexerPrologPto)
     .Tiling(TilingLightningIndexerPrologPto)
     .TilingParse<LightningIndexerPrologPtoCompileInfo>(TilingParseLightningIndexerPrologPto);
}