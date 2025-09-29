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
constexpr uint64_t SA_ENABLE_ACT_QUERY_TYPE_OFFSET = uint64_t(100000000UL);
// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t ACT_SEQ_QUERY_INPUT_INDEX = 20;
constexpr uint32_t MLA_ROPE_COS_INDEX = 7;
constexpr uint32_t MLA_ROPE_SIN_INDEX = 8;
constexpr uint32_t ACT_SEQ_KEY_INDEX = 13;
constexpr uint32_t BLOCK_TABLE_INDEX = 12;
constexpr uint32_t CACHE_TENSOR_INDEX = 9;
constexpr uint32_t KV_CACHE_INDEX = 10;
constexpr uint32_t KR_CACHE_INDEX = 11;
constexpr uint64_t AXIS_0 = 0;
constexpr uint64_t AXIS_1 = 1;
constexpr uint64_t AXIS_2 = 2;
constexpr uint64_t AXIS_3 = 3;

// OUTPUT
constexpr uint32_t ATTEN_RES_OUTPUT_INDEX = 0;
constexpr uint32_t NUM_512 = 512;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_1536 = 1536;
constexpr uint32_t NUM_128K = 128 * 1024;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t NUM_1024 = 1024;
constexpr uint32_t SELECT_COUNT_VALUE = 2048;
struct SparseAttentionPtoCompileInfo {
    uint32_t block_dim_num = 0;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context) {
    return context->GetCompiledInfo<T>();
}

ge::graphStatus SparseAttentionPtoCheck(gert::TilingContext *context) {
    auto tokenXShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, TOKEN_X_INPUT_INDEX);
    if (tokenXShape == nullptr) {
        printf("Sparse attention pto must support token x tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    auto mlaRopeCosShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, MLA_ROPE_COS_INDEX);
    if (mlaRopeCosShape == nullptr) {
        printf("Sparse attention pto must support mlaRopeCos tensor.\n");
        return ge::GRAPH_FAILED;
    }

    auto mlaRopeSinShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, MLA_ROPE_SIN_INDEX);
    if (mlaRopeSinShape == nullptr) {
        printf("Sparse attention pto must support mlaRopeSin tensor.\n");
        return ge::GRAPH_FAILED;
    }

    auto actSeqKeyShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, ACT_SEQ_KEY_INDEX);
    if (actSeqKeyShape == nullptr) {
        printf("Sparse attention pto must support actSeqKey tensor.\n");
        return ge::GRAPH_FAILED;
    }

    auto blockTableShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, BLOCK_TABLE_INDEX);
    if (blockTableShape == nullptr) {
        printf("Sparse attention pto must support block table tensor.\n");
        return ge::GRAPH_FAILED;
    }

    auto cacheIndexShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, CACHE_TENSOR_INDEX);
    if (cacheIndexShape == nullptr) {
        printf("Sparse attention pto must support cache index tensor.\n");
        return ge::GRAPH_FAILED;
    }

    auto kvCaheShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, KV_CACHE_INDEX);
    if (kvCaheShape == nullptr) {
        printf("Sparse attention pto must support kv cache tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    auto krCaheShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, KR_CACHE_INDEX);
    if (krCaheShape == nullptr) {
        printf("Sparse attention pto must support kr cache tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    if (tokenXShape->GetStorageShape().GetDimNum() != 3 || mlaRopeCosShape->GetStorageShape().GetDimNum() != 3 ||
        mlaRopeSinShape->GetStorageShape().GetDimNum() != 3 || actSeqKeyShape->GetStorageShape().GetDimNum() != 1 ||
        cacheIndexShape->GetStorageShape().GetDimNum() != 2 || kvCaheShape->GetStorageShape().GetDimNum() != 4 ||
        krCaheShape->GetStorageShape().GetDimNum() != 4) {
        printf("Sparse attention pto get tensor with invalid dim num.\n");
        return ge::GRAPH_FAILED;
    }

    if (tokenXShape->GetStorageShape().GetDim(AXIS_0) != mlaRopeCosShape->GetStorageShape().GetDim(AXIS_0) || 
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != mlaRopeSinShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != blockTableShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != actSeqKeyShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != cacheIndexShape->GetStorageShape().GetDim(AXIS_0) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_0) != 4) {
        printf("Sparse attention pto get invalid batch value.\n");
        return ge::GRAPH_FAILED;
    }

    if (tokenXShape->GetStorageShape().GetDim(AXIS_1) != mlaRopeCosShape->GetStorageShape().GetDim(AXIS_1) || 
        tokenXShape->GetStorageShape().GetDim(AXIS_1) != mlaRopeSinShape->GetStorageShape().GetDim(AXIS_1) ||
        tokenXShape->GetStorageShape().GetDim(AXIS_1) != cacheIndexShape->GetStorageShape().GetDim(AXIS_1) ||
        (tokenXShape->GetStorageShape().GetDim(AXIS_1) != 1 && tokenXShape->GetStorageShape().GetDim(AXIS_1) != 2)) {
        printf("Sparse attention pto get invalid s1 value.\n");
        return ge::GRAPH_FAILED;
    }

    if (kvCaheShape->GetStorageShape().GetDim(AXIS_0) != krCaheShape->GetStorageShape().GetDim(AXIS_0) ||
        kvCaheShape->GetStorageShape().GetDim(AXIS_0) > tokenXShape->GetStorageShape().GetDim(AXIS_0) * NUM_1024 ||
        blockTableShape->GetStorageShape().GetDim(AXIS_1) > NUM_1024) {
        printf("Sparse attention pto get invalid block num value.\n");
        return ge::GRAPH_FAILED;
    }
 
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingSparseAttentionPto(gert::TilingContext *context) {
    if (context == nullptr) {
        printf("Failed to get SparseAttentionPto tiling context.\n");
        return ge::GRAPH_FAILED;
    }

    if (context->GetNodeName() == nullptr) {
        printf("Failed to get SparseAttentionPto tiling context node name.\n");
        return ge::GRAPH_FAILED;
    }

    if (SparseAttentionPtoCheck(context) != ge::GRAPH_SUCCESS) {
        printf("SparseAttentionPto get invalid tensor.\n");
        return ge::GRAPH_FAILED;
    }
    
    auto tokenXDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, TOKEN_X_INPUT_INDEX);
    auto tokenXShape = context->GetDynamicInputShape(TOKEN_X_INPUT_INDEX, TOKEN_X_INPUT_INDEX);
    if (tokenXDesc == nullptr || tokenXShape == nullptr || tokenXShape->GetStorageShape().GetDimNum() != 3) {
        printf("Input tensor tokenX is invalid.\n");
        return ge::GRAPH_FAILED;
    }

    auto actualSeqQueryTableDesc = context->GetDynamicInputDesc(TOKEN_X_INPUT_INDEX, ACT_SEQ_QUERY_INPUT_INDEX);
    if (actualSeqQueryTableDesc == nullptr) {
        printf("SparseAttentionPto do not support actualSeqQuery.\n");
    }

    uint64_t tilingKey = SA_ENABLE_ACT_QUERY_TYPE_OFFSET;
    context->SetTilingKey(SA_ENABLE_ACT_QUERY_TYPE_OFFSET);
    auto compileInfo = reinterpret_cast<const SparseAttentionPtoCompileInfo *>(context->GetCompileInfo());
    context->SetBlockDim(compileInfo->block_dim_num);
    uint64_t b = tokenXShape->GetStorageShape().GetDim(0);
    uint64_t s = tokenXShape->GetStorageShape().GetDim(1);

    auto tokenXDataType = tokenXDesc->GetDataType();
    uint64_t dtypeSize = ge::GetSizeByDataType(tokenXDataType);

    uint64_t queryNopeOutBuffer = b * s * NUM_128 * NUM_512 * dtypeSize;
    uint64_t queryRopeOutBuffer = b * s * NUM_128 * NUM_64 * dtypeSize;
    uint64_t rmsResBuffer = b * s * NUM_1536 * dtypeSize;
    uint64_t queryOutBuffer = b * s * NUM_64 * NUM_128 * dtypeSize;
    uint64_t weightOutBuffer = b * s * NUM_64 * dtypeSize;
    uint64_t localSumBuffer = b * s * NUM_128K * 4;
    auto totalBuffer = queryNopeOutBuffer + queryRopeOutBuffer + rmsResBuffer + queryOutBuffer + weightOutBuffer + localSumBuffer;
    auto workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = totalBuffer;
    printf("SparseAttentionPto parse tiling context sccuess, core num is %d, tilingKey is %ld, totalBuffer is %ld.\n",
        compileInfo->block_dim_num, tilingKey, totalBuffer);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseSparseAttentionPto(gert::TilingParseContext *context) {
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfo = GetCompileInfoPtr<SparseAttentionPtoCompileInfo>(context);
    compileInfo->block_dim_num = platformInfo->GetCoreNum();
    printf("SparseAttentionPto parse tiling context sccuess, core num is %d.\n", platformInfo->GetCoreNum());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SparseAttentionPto)
    .Tiling(TilingSparseAttentionPto)
    .TilingParse<SparseAttentionPtoCompileInfo>(TilingParseSparseAttentionPto);
} // namespace tiling
