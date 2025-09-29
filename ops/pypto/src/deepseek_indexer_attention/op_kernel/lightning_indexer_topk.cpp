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
 
#include <cfloat>
#include "tilefwk/tensor.h"
#include "tilefwk/tilefwk.h"
#include "parallel_sort.h"
#include "lightning_indexer_topk.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {

void LightningIndexerTopkInner(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList) {
    LightningIndexerTopkInnerImpl(
        query, key, weights, actSeqKey, blockTable, topkRes, selectedCount, tileConfig, unrollList);
}

void LightningIndexerTopkInnerImpl(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList, Tensor *tmpOut, Tensor *topkValue) {
    /*
    query: [B, S1, indexN1, indexD], bf16
    key: [blockNum, blockSize, n2, indexD] bf16
    weights: [B, S1, indexN1], bf16
    actSeqKey: [B], int32
    blockTable: [B, maxBlockNum]
    topkRes: [B, s1, N2, selectedCount], int32
    selectedCount: selectedCount num
    */

    // Symbolization
    SymbolicScalar b = GetInputShape(query, 0);
    SymbolicScalar s1 = GetInputShape(query, 1);
    SymbolicScalar blockNum = GetInputShape(key, 0);

    auto indexN1 = query.GetShape()[SHAPE_DIM2];
    auto indexD = query.GetShape()[SHAPE_DIM3];
    auto blockSize = key.GetShape()[1];
    auto n2 = key.GetShape()[SHAPE_DIM2];
    auto dtype = query.GetDataType();
    auto group = indexN1 / n2;
    auto c1Tile = tileConfig.c1Tile;
    constexpr int64_t maxBatch = 128;
    constexpr int64_t maxS1 = 4;
    constexpr int64_t maxN2 = 1;
    constexpr int64_t maxS2 = 128 * 1024;

    Tensor query2D(dtype, {b * s1 * indexN1, indexD}, "query2D");
    Tensor key2D(dtype, {blockNum * blockSize, n2 * indexD}, "key2D");
    Tensor weight2D(dtype, {b * s1 * indexN1, 1}, "weight2D");
    Tensor localSum(DT_FP32, {maxBatch * maxS1 * maxN2, maxS2}, "localSum");

    config::SetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED,true);
    LOOP("INPUT_4D_2_2D", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        ReshapeInplace(query, query2D);
        ReshapeInplace(key, key2D);
        ReshapeInplace(weights, weight2D);
    }

    LOOP("INDEX_LOOP_BATCH", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(b)) {
        auto curSeq = GetInputData(actSeqKey, {bIdx});

        LOOP("INDEX_LOOP_S1", FunctionType::DYNAMIC_LOOP, s1Idx, LoopRange(s1)) {
            // casual inference
            auto casualOffset = s1 - s1Idx - 1;
            auto effSeq = curSeq - casualOffset;
            auto actBlock = (effSeq + blockSize - 1) / blockSize;
            LOOP("INDEX_LOOP_N2", FunctionType::DYNAMIC_LOOP, n2Idx, LoopRange(n2)) {
                auto bs1n2Offset = bIdx * s1 * n2 + s1Idx * n2 + n2Idx;
                auto qOffset = bIdx * s1 * indexN1 + s1Idx * indexN1 + n2Idx * group;

                // unrolling process template
                auto unrollingProcess = [&](int unrollLength, auto &&firstBlockIdx) {
                    auto curQ = View(query2D, {group, indexD}, {qOffset, 0}); // (group, d)
                    std::vector<Tensor> concatSrcs;
                    // static unrolling
                    for (int subblockIdx = 0; subblockIdx < unrollLength; subblockIdx++) {
                        auto blockIdx = firstBlockIdx + subblockIdx;
                        SymbolicScalar curBlockIdx = GetInputData(blockTable, {bIdx, blockIdx});
                        auto curK = View(key2D, {blockSize, indexD},
                            {std::min(blockSize, effSeq - (blockIdx * blockSize)), indexD},
                            {curBlockIdx * blockSize, n2Idx * indexD});

                        TileShape::Current().SetCubeTile(
                            {c1Tile[0], c1Tile[1]}, {c1Tile[2], c1Tile[3]}, {c1Tile[4], c1Tile[5]}, false);
                        auto mmRes =
                            Matrix::Matmul<false, true>(DataType::DT_FP32, curQ, curK); // (group, superBlockSize)
                        concatSrcs.emplace_back(mmRes);
                    }

                    TileShape::Current().SetVecTile(tileConfig.weightTile);

                    auto curW = View(weight2D, {group, 1}, {qOffset, 0}); // (group, 1)
                    auto wB32 = Cast(curW, DT_FP32);                      // (group, 1)

                    auto mmRes = Concat(concatSrcs, -1); // (group, superBlockSize)

                    TileShape::Current().SetVecTile(tileConfig.v1Tile);
                    auto reluRes = Relu(mmRes);       // (group, superBlockSize)
                    auto mulRes = Mul(reluRes, wB32); // (group, superBlockSize) * (group, 1) -> (group, superBlockSize)
                    auto sumRes = RowSumSingle(mulRes, 0); // (1, superBlockSize)
                    Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, localSum);
                    if (tmpOut != nullptr) {
                        // tmpOut: [B*S1*N2, S2]
                        Assemble(sumRes, {bs1n2Offset, firstBlockIdx * blockSize}, *tmpOut);
                    }
                };

                LOOP("INDEX_LOOP_MATMUL", FunctionType::DYNAMIC_LOOP, blockIdx, LoopRange(actBlock), unrollList) {
                    for (int unrollLength : unrollList) {
                        UNROLL(unrollLength) {
                            unrollingProcess(unrollLength, blockIdx);
                        }
                    }
                }
            }
        }
    }

    ASSERT(selectedCount == 2048);
    DataType xdtype = localSum.GetDataType();
    DataType idxdtype = topkRes.GetDataType();
    const int padIdxValue = -1;
    const int tileSize = 4096;
    const bool descending = true;
    const float padValue = descending ? -FLT_MAX : FLT_MAX;
    const int length2K = selectedCount;
    const int length8K = 1024 * 8;
    const int length32K = 1024 * 32;
    TileShape::Current().SetVecTile({1, tileSize});
    LOOP("INDEX_LOOP_TOPK_bs1n2Offset", FunctionType::DYNAMIC_LOOP, bs1n2Offset, LoopRange(b * s1 * n2)) {
        auto bIdx = bs1n2Offset / (s1 * n2);
        auto s1Idx = (bs1n2Offset % (s1 * n2)) / n2;
        auto n2Idx = bs1n2Offset % n2;

        auto curSeq = GetInputData(actSeqKey, {bIdx});
        auto casualOffset = s1 - s1Idx - 1;
        auto effSeq = curSeq - casualOffset;

        auto lengthIsLE2K = effSeq <= length2K;
        auto lengthIsGT2K = effSeq > length2K;
        Tensor padX2K(xdtype, {maxBatch * maxS1 * maxN2, length2K}, "padX2K");
        TileShape::Current().SetVecTile({1, tileSize});
        LOOP("2K_TOPK", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsLE2K)) {
            (void)unused;
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, true);
            LOOP("2K_TOPK_PAD", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                (void)unused1;
                TileShape::Current().SetVecTile({1, length2K});
                auto effSumRes = View(localSum, {1, length2K}, {1, effSeq}, {bs1n2Offset, 0});
                auto ax = View(effSumRes, {1, length2K}, {1, effSeq}, {0, 0});
                auto bx = VectorDuplicate(Element(xdtype, padValue), xdtype, {1, length2K}, {1, length2K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX2K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX2K);
            }
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, false);
            LOOP("2K_TOPK_SORT", FunctionType::DYNAMIC_LOOP, unused2, LoopRange(1)) {
                (void)unused2;
                auto [res, tmp] = TopKSort(View(padX2K, {1, length2K}, {bs1n2Offset, 0}), 0);
                auto resIdx = TopKExtract(res, selectedCount, true);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(
                    View(resIdx, {1, selectedCount}, {1, effSeq}, {0, 0}), {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
                auto topkIndicesPad = VectorDuplicate(Element(idxdtype, padIdxValue), idxdtype,
                    {1, 1, 1, selectedCount}, {1, 1, 1, selectedCount - effSeq});
                Assemble(topkIndicesPad, {bIdx, s1Idx, n2Idx, effSeq}, topkRes);

                if (topkValue != nullptr) {
                    auto resValue = TopKExtract(res, selectedCount, false);
                    auto topk4DValue = Reshape(View(resValue, {1, selectedCount}, {1, effSeq}, {0, 0}),
                        {1, 1, 1, selectedCount}, {1, 1, 1, effSeq});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                    auto topkValuePad = VectorDuplicate(Element(DT_FP32, padValue), DT_FP32, {1, 1, 1, selectedCount},
                        {1, 1, 1, selectedCount - effSeq});
                    Assemble(topkValuePad, {bIdx, s1Idx, n2Idx, effSeq}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }

        auto lengthIsLE8K = effSeq <= length8K;
        auto lengthIsGT8K = effSeq > length8K;
        Tensor padX8K(xdtype, {maxBatch * maxS1 * maxN2, length8K}, "padX8K");
        TileShape::Current().SetVecTile({1, tileSize});
        LOOP("8K_TOPK", FunctionType::DYNAMIC_LOOP, unused, LoopRange(lengthIsGT2K * lengthIsLE8K)) {
            UNUSED(unused);
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, true);

            LOOP("8K_TOPK_PAD", FunctionType::DYNAMIC_LOOP, unused1, LoopRange(1)) {
                (void)unused1;
                TileShape::Current().SetVecTile({1, length2K});
                auto effSumRes = View(localSum, {1, length8K}, {1, effSeq}, {bs1n2Offset, 0});
                auto ax = View(effSumRes, {1, length8K}, {1, effSeq}, {0, 0});
                auto bx = VectorDuplicate(Element(xdtype, padValue), xdtype, {1, length8K}, {1, length8K - effSeq});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX8K);
                Assemble(bx, {bs1n2Offset, effSeq}, padX8K);
            }
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, false);

            LOOP("8K_TOPK_SORT", FunctionType::DYNAMIC_LOOP, unused2, LoopRange(1)) {
                (void)unused2;
                auto [res, tmp] = TopKSort(View(padX8K, {1, length8K}, {bs1n2Offset, 0}), 0);
                auto resIdx = TopKExtract(res, selectedCount, true);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
                Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);

                if (topkValue != nullptr) {
                    auto resValue = TopKExtract(res, selectedCount, false);
                    auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                    Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
                }
                TileShape::Current().SetVecTile({1, tileSize});
            }
        }

        // 128K TOPK
        auto totalSizeY1 = maxS2 / length8K * selectedCount;
        auto totalSizeY2 = maxS2 / length8K * selectedCount / length8K * selectedCount;
        Tensor localY1(xdtype, {maxBatch * maxS1 * maxN2, totalSizeY1 * 2}, "localY1");
        Tensor padY1(xdtype, {maxBatch * maxS1 * maxN2, length8K * 2}, "padY1");
        Tensor localY2(xdtype, {maxBatch * maxS1 * maxN2, totalSizeY2 * 2}, "localY2");
        auto maxNumOf8K = maxS2 / length8K;
        auto numOf8K = (effSeq - 1) / length8K + 1;
        auto validSizeY1 = numOf8K * selectedCount;
        auto padSizeY1 = totalSizeY1 - validSizeY1;
        auto numOf32K = (effSeq - 1) / length32K + 1;
        auto validSizeY2 = numOf32K * selectedCount;
        auto padSizeY2 = totalSizeY2 - validSizeY2;

        TileShape::Current().SetVecTile({1, tileSize});
        LOOP("128K_PAD_Y1Y2", FunctionType::DYNAMIC_LOOP, unused,
            LoopRange(1 * lengthIsGT8K * (numOf8K != maxNumOf8K))) {
            UNUSED(unused);
            Assemble(VectorDuplicate(Element(xdtype, padValue), xdtype, {1, totalSizeY1 * 2}, {1, padSizeY1 * 2}),
                {bs1n2Offset, validSizeY1 * 2}, localY1);
            Assemble(VectorDuplicate(Element(xdtype, padValue), xdtype, {1, totalSizeY2 * 2}, {1, padSizeY2 * 2}),
                {bs1n2Offset, validSizeY2 * 2}, localY2);
        }

        auto need8KPadTail = (effSeq % length8K) != 0;
        auto numOf8KFullBlock = numOf8K - need8KPadTail;
        LOOP("128K_TO_32K_FULL_SORT", FunctionType::DYNAMIC_LOOP, idx1, LoopRange(numOf8KFullBlock * lengthIsGT8K)) {
            auto effSumRes = View(localSum, {1, maxS2}, {1, effSeq}, {bs1n2Offset, 0});
            auto ax = View(effSumRes, {1, length8K}, {0, idx1 * length8K});
            auto [res, tmp] = TopKSort(ax, idx1);
            Assemble(
                Assign(View(res, {1, selectedCount * 2}, {0, 0})), {bs1n2Offset, idx1 * selectedCount * 2}, localY1);
        }

        LOOP("128K_TO_32K_TAIL", FunctionType::DYNAMIC_LOOP, unused, LoopRange(need8KPadTail * lengthIsGT8K)) {
            UNUSED(unused);
            auto xStartOffset = numOf8KFullBlock * length8K;
            auto tailBlockLength = effSeq - xStartOffset;
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, true);

            LOOP("128K_TO_32K_TAIL_PAD", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                UNUSED(unused0);
                TileShape::Current().SetVecTile({1, tileSize});
                auto effSumRes = View(localSum, {1, maxS2}, {1, effSeq}, {bs1n2Offset, 0});
                auto ax = View(effSumRes, {1, length8K}, {1, tailBlockLength}, {0, xStartOffset});
                auto bx =
                    VectorDuplicate(Element(xdtype, padValue), xdtype, {1, length8K}, {1, length8K - tailBlockLength});
                Assemble(Assign(ax), {bs1n2Offset, 0}, padX8K);
                Assemble(bx, {bs1n2Offset, tailBlockLength}, padX8K);
            }
            ConfigManager::SetProgramConfig(SG_SKIP_PARTITION, false);

            LOOP("128K_TO_32K_TAIL_SORT", FunctionType::DYNAMIC_LOOP, unused0, LoopRange(1)) {
                UNUSED(unused0);
                auto [res, tmp] = TopKSort(View(padX8K, {1, length8K}, {bs1n2Offset, 0}), numOf8KFullBlock);
                Assemble(Assign(View(res, {1, selectedCount * 2}, {0, 0})),
                    {bs1n2Offset, numOf8KFullBlock * selectedCount * 2}, localY1);
            }
        }

        LOOP("32K_TO_8K_MERGE", FunctionType::DYNAMIC_LOOP, idx2, LoopRange(numOf32K * lengthIsGT8K)) {
            auto res = TopKMerge(View(localY1, {1, length8K * 2}, {bs1n2Offset, idx2 * length8K * 2}), selectedCount);
            Assemble(
                Assign(View(res, {1, selectedCount * 2}, {0, 0})), {bs1n2Offset, idx2 * selectedCount * 2}, localY2);
        }

        TileShape::Current().SetVecTile({1, tileSize});
        LOOP("8K_TO_2K_MERGE", FunctionType::DYNAMIC_LOOP, unused, LoopRange(1 * lengthIsGT8K)) {
            UNUSED(unused);
            auto res = TopKMerge(View(localY2, {1, length8K * 2}, {bs1n2Offset, 0}), selectedCount);
            auto resIdx = TopKExtract(res, selectedCount, true);
            TileShape::Current().SetVecTile(tileConfig.addsTile);
            auto topk4D = Reshape(resIdx, {1, 1, 1, selectedCount});
            Assemble(Assign(topk4D), {bIdx, s1Idx, n2Idx, 0}, topkRes);
            if (topkValue != nullptr) {
                auto resValue = TopKExtract(res, selectedCount, false);
                TileShape::Current().SetVecTile(tileConfig.addsTile);
                auto topk4DValue = Reshape(resValue, {1, 1, 1, selectedCount});
                Assemble(Assign(topk4DValue), {bIdx, s1Idx, n2Idx, 0}, *topkValue);
            }
            TileShape::Current().SetVecTile({1, tileSize});
        }
    }
}

} // namespace npu::tile_fwk