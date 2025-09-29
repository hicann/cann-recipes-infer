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
 
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
 
namespace tiling {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t WEIGHTS_INPUT_INDEX = 2;
constexpr uint32_t ACTSEQ_INPUT_INDEX = 3;
 
constexpr uint32_t IDX_N_HEADS = 64;
constexpr uint32_t IDX_HEAD_DIM = 128;
constexpr uint32_t N2_VALUE = 1;
constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t AXIS_0 = 0;
constexpr uint32_t AXIS_1 = 1;
constexpr uint32_t AXIS_2 = 2;
constexpr uint32_t AXIS_3 = 3;

constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 4;
constexpr uint64_t L1_SORT_LENGTH = 1024 * 16;
constexpr uint64_t Lightning_Indexer_PTO_ConfigKey = uint64_t(100000000UL);
struct LightningIndexerPtoCompileInfo {
    uint32_t block_dim_num = 0;
};
 
template<typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}
 
ge::graphStatus GetDataType(ge::DataType dataType, uint32_t &typeValue) {
    switch (dataType) {
        case ge::DT_FLOAT16:
            typeValue = 0;
            break;
        case ge::DT_BF16:
            typeValue = 1U;
            break;
        case ge::DT_INT8:
            typeValue = 2U;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LightningIndexerPtoCheck(gert::TilingContext *context) {
    auto blockTableShape = context->GetDynamicInputShape(QUERY_INPUT_INDEX, BLOCK_TABLE_INPUT_INDEX);
    if (blockTableShape == nullptr) {
        printf("Lightning indexer pto must support block table tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    auto queryShape = context->GetDynamicInputShape(QUERY_INPUT_INDEX, QUERY_INPUT_INDEX);
    if (queryShape == nullptr) {
        printf("Lightning indexer pto must support query tensor.\n");
        return ge::GRAPH_FAILED;
    }
 
    auto keyShape = context->GetDynamicInputShape(QUERY_INPUT_INDEX, KEY_INPUT_INDEX);
    if (keyShape == nullptr) {
        printf("Lightning indexer pto must support key tensor.\n");
        return ge::GRAPH_FAILED;
    }
 
    auto weightsShape = context->GetDynamicInputShape(QUERY_INPUT_INDEX, WEIGHTS_INPUT_INDEX);
    if (weightsShape == nullptr) {
        printf("Lightning indexer pto must support weights tensor.\n");
        return ge::GRAPH_FAILED;
    }
 
    auto actSeqShape = context->GetDynamicInputShape(QUERY_INPUT_INDEX, ACTSEQ_INPUT_INDEX);
    if (actSeqShape == nullptr) {
        printf("Lightning indexer pto must support act seq tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    if (queryShape->GetStorageShape().GetDimNum() !=4 || keyShape->GetStorageShape().GetDimNum() != 4 ||
        blockTableShape->GetStorageShape().GetDimNum() != 2) {
        printf("Lightning indexer pto get tensor with invalid dim num.\n");
        return ge::GRAPH_FAILED;
    }

    if (queryShape->GetStorageShape().GetDim(AXIS_0) != weightsShape->GetStorageShape().GetDim(AXIS_0) || 
        queryShape->GetStorageShape().GetDim(AXIS_0) != actSeqShape->GetStorageShape().GetDim(AXIS_0) ||
        queryShape->GetStorageShape().GetDim(AXIS_0) != blockTableShape->GetStorageShape().GetDim(AXIS_0) ||
        queryShape->GetStorageShape().GetDim(AXIS_0) > 128 || queryShape->GetStorageShape().GetDim(AXIS_0) < 1) {
        printf("Lightning indexer pto get invalid batch value.\n");
        return ge::GRAPH_FAILED;
    }
 
    if (queryShape->GetStorageShape().GetDim(AXIS_1) != weightsShape->GetStorageShape().GetDim(AXIS_1) || 
        queryShape->GetStorageShape().GetDim(AXIS_1) > 4 || queryShape->GetStorageShape().GetDim(AXIS_1) <= 0) {
        printf("Lightning indexer pto get invalid s1 value.\n");
        return ge::GRAPH_FAILED;
    }
 
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingLightningIndexerPto(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
 
    if (context->GetNodeName() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    if (LightningIndexerPtoCheck(context) != ge::GRAPH_SUCCESS) {
        printf("Lightning indexer pto get invalid tensor.\n");
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = Lightning_Indexer_PTO_ConfigKey;
    context->SetTilingKey(tilingKey);

    auto compileInfo = reinterpret_cast<const LightningIndexerPtoCompileInfo*>(context->GetCompileInfo());
    context->SetBlockDim(compileInfo->block_dim_num);
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus TilingParseLightningIndexerPto(gert::TilingParseContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfo = GetCompileInfoPtr<LightningIndexerPtoCompileInfo>(context);
    compileInfo->block_dim_num = platformInfo->GetCoreNum();
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP(LightningIndexerPto)
     .Tiling(TilingLightningIndexerPto)
     .TilingParse<LightningIndexerPtoCompileInfo>(TilingParseLightningIndexerPto);
}
