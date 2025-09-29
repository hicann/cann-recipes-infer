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

#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/config_manager.h"
#include "gather_after_prolog.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
void GatherAfterPrologCompute(Tensor &topKIndcies, Tensor &kNopeCache, Tensor &kRopeCache, Tensor &blockTable,
    Tensor &actSeqs, Tensor &gatherRes, const NSASimpleParams &params, SymbolicScalar b, SymbolicScalar s1) {
    int dN = kNopeCache.GetShape()[kNopeCache.GetShape().size() - 1];
    int dR = kRopeCache.GetShape()[kRopeCache.GetShape().size() - 1];
    int n2 = topKIndcies.GetShape()[2];
    int blockSize = params.blockSize;
    int topk = params.topk;
    std::set<int> unrollList = {64, 32, 16, 8, 4, 2, 1};

    config::SetCodeGenConfig("SUPPORT_DYNAMIC_UNALIGNED", true);
    LOOP("loop_b_gather", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, b, 1), {}, true) {
        LOOP("loop_s1_gather", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(0, s1, 1)) {
            LOOP("loop_n2_gather", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(0, n2, 1)) {
                ConfigManager::Instance().SetSemanticLabel("gather0");
                SymbolicScalar curKvSeq = GetInputData(actSeqs, {bIdx});
                SymbolicScalar topkLoop = std::min(
                    std::max(curKvSeq - s1 + 1 + s1Idx, 0), topk); // for MTP s1!= 1 casual计算, 并且与topk取min
                LOOP("loop_k_gather", FunctionType::DYNAMIC_LOOP, topKIdx, LoopRange(0, topkLoop, 1), unrollList) {
                    SymbolicScalar topkIndex;
                    TileShape::Current().SetVecTile(1, 1, 1, 16);
#if DEBUG_DUMP_TMP_IN_OUT == 1
                    topkIndex = GetInputData(topKIndcies, {bIdx, s1Idx, n2Idx, topKIdx});
#else
                    topkIndex = GetTensorData(topKIndcies, {bIdx, s1Idx, n2Idx, topKIdx});
#endif
                    SymbolicScalar blockIdxInBatch = topkIndex / blockSize;
                    SymbolicScalar tail = topkIndex % blockSize;
                    SymbolicScalar slcBlockIdx = GetInputData(blockTable, {bIdx, blockIdxInBatch});
                    TileShape::Current().SetVecTile(1, dN);
                    auto kvSlcBlock = View(kNopeCache, {1, dN}, {slcBlockIdx * blockSize + tail, 0});
                    auto krSlcBlock = View(kRopeCache, {1, dR}, {slcBlockIdx * blockSize + tail, 0});

                    ConfigManager::Instance().SetSemanticLabel("gather1");
                    auto kvSlcBlock_fp32 = Cast(kvSlcBlock, DataType::DT_FP32);
                    auto krSlcBlock_fp32 = Cast(krSlcBlock, DataType::DT_FP32);
                    ConfigManager::Instance().SetSemanticLabel("gather2");
                    auto kvSlcBlock_fp16 = Cast(kvSlcBlock_fp32, gatherRes.GetDataType());
                    auto krSlcBlock_fp16 = Cast(krSlcBlock_fp32, gatherRes.GetDataType());
                    SymbolicScalar ofs = bIdx * s1 * n2 * topk + s1Idx * n2 * topk + n2Idx * topk + topKIdx;

                    Assemble(kvSlcBlock_fp16, {ofs, 0}, gatherRes);
                    Assemble(krSlcBlock_fp16, {ofs, dN}, gatherRes);
                }
            }
        }
    }
}

} // namespace npu::tile_fwk