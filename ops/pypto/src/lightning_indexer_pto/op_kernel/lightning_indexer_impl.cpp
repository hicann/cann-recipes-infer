/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <sstream>
#include "lightning_indexer_topk.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/op_registry.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk {
constexpr uint64_t Lightning_Indexer_PTO_ConfigKey = uint64_t(100000000UL);

struct LightningIndexerPtoParams {
    int b = -1;
    int s1 = -1;
    int blockNum = -1;
    int maxBlockNum = -1;
    int blockSize = 128;
    int indexNHeads = 64;
    int indexHeadDim = 128;
    int n1 = 64;
    int n2 = 1;
    DataType dType = DT_BF16;
    int selectedCount = 2048;
};

void DynamicLightningIndexerPto(uint64_t configKey) {
    (void)configKey;
    config::SetHostOption(ONLY_CODEGEN, true);
    config::SetCodeGenOption(SUPPORT_DYNAMIC_UNALIGNED, true);
    config::SetCodeGenOption(CODEGEN_EXPRESSION_FUSION, true);
    config::SetRuntimeOption(MACHINE_SCHED_MODE, static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH) |
                                                 static_cast<uint8_t>(MachineScheduleConfig::MULTI_CORE_FAIR_SCH));
    config::SetRuntimeOption<int>(WORKSPACE_RECYCLE_PERIOD, 128);
    config::SetRuntimeOption<int>(ESTIMATED_STITCH_TASK_MAX_LOOP_NUM, 128);

    LightningIndexerPtoParams params;

    IndexerTile indexerConfig;
    indexerConfig.weightTile = {64, 128};
    indexerConfig.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    indexerConfig.v1Tile = {64, 128};
    indexerConfig.topkTile = {1, 4096};
    indexerConfig.addsTile = {1, 1, 1, 4096};

    std::vector<int64_t> queryShape = {params.b, params.s1, params.indexNHeads, params.indexHeadDim};
    std::vector<int64_t> keyShape = {params.blockNum, params.blockSize, params.n2, params.indexHeadDim};
    std::vector<int64_t> weightsShape = {params.b, params.s1, params.n1, 1};
    std::vector<int64_t> actualSeqLengthsKeyShape = {params.b};
    std::vector<int64_t> blockTableShape = {params.b, params.maxBlockNum};

    Tensor query(params.dType, queryShape, "query");
    Tensor key(params.dType, keyShape, "key");
    Tensor weights(params.dType, weightsShape, "weights");
    Tensor actualSeqLengthsKey(DT_INT32, actualSeqLengthsKeyShape, "actualSeqgiLengthsKey");
    Tensor blockTable(DT_INT32, blockTableShape, "blocktable");
    Tensor selectedIndices(DT_INT32,
        {GetInputShape(query, 0), GetInputShape(query, 1), params.n2, params.selectedCount}, "selectedIndices");

    FUNCTION("LightningIndexer", {query, key, weights, actualSeqLengthsKey, blockTable}, {selectedIndices}) {
        LightningIndexerTopk(
            query, key, weights, actualSeqLengthsKey, blockTable, selectedIndices, params.selectedCount, indexerConfig);
    }
}

REGISTER_OP(LightningIndexerPto)
    .ImplFunc({
        {Lightning_Indexer_PTO_ConfigKey, DynamicLightningIndexerPto}
});
} // namespace npu::tile_fwk
