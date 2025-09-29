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

#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <sstream>

#include "dynamic_nsa_common.h"
#include "deepseek_indexer_attention.h"

#include "tilefwk/tilefwk.h"
#include "tilefwk/op_registry.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/config_manager.h"

namespace npu::tile_fwk {

constexpr uint64_t DSA_BF16_ConfigKey = uint64_t(100000000UL);

template <bool isSmooth = false, bool nz = false>
void DynamicDecodeSparseAttentionPtoImpl(NSASimpleParams &params) {
    int b = params.b;
    int s1 = params.s1;
    int n1 = params.n1;
    int n2 = params.n2;
    int h = params.h;
    int dn = params.kv_lora_rank;
    int qLoraRank = params.q_lora_rank;
    int qkNopeHeadDim = params.qk_nope_head_dim;
    int qkRopeHeadDim = params.qk_rope_head_dim;
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    int blockSize = params.blockSize;
    int idx_n_heads = params.idx_n_heads;
    int idx_head_dim = params.idx_head_dim;
    int blockNum = params.blockNum;
    int topk = params.topk;

    int maxBlockNumPerBatch = -1;
    DataType dType = DT_BF16;

    std::vector<int64_t> xShape = {b, s1, h};
    std::vector<int64_t> wDqShape = {h, qLoraRank};
    std::vector<int64_t> wUqQrShape = {qLoraRank, n1 * qHeadDim};
    std::vector<int64_t> wDkvKrShape = {h, dn + qkRopeHeadDim};
    std::vector<int64_t> wUkShape = {n1, qkNopeHeadDim, dn};
    std::vector<int64_t> cosShape = {b, s1, qkRopeHeadDim};
    std::vector<int64_t> gammaCqShape = {qLoraRank};
    std::vector<int64_t> gammaCkvShape = {dn};
    std::vector<int64_t> kvLenShape = {b, s1};
    std::vector<int64_t> kvCacheShape = {blockNum, blockSize, n2, dn};
    std::vector<int64_t> krCacheShape = {blockNum, blockSize, n2, qkRopeHeadDim};

    std::vector<int64_t> blockTableShape = {b, maxBlockNumPerBatch};
    std::vector<int64_t> actSeqsShape = {b};
    std::vector<int64_t> tmpTopkInputShape = {b, s1, n2, topk};

    std::vector<int64_t> qWShape = {qLoraRank, idx_n_heads * idx_head_dim};
    std::vector<int64_t> kWShape = {h, idx_head_dim};
    std::vector<int64_t> projWShape = {h, idx_n_heads};
    std::vector<int64_t> lnWShape = {idx_head_dim};
    std::vector<int64_t> keyShape = {blockNum, blockSize, n2, idx_head_dim};

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;

    // input
    Tensor dynamicX(dType, xShape, "x");
    Tensor wDq(dType, wDqShape, "wDq", weightFormat);
    Tensor wUqQr(dType, wUqQrShape, "wUqQr", weightFormat);
    Tensor wDkvKr(dType, wDkvKrShape, "wDkvKr", weightFormat);
    Tensor wUk(dType, wUkShape, "wUk", TileOpFormat::TILEOP_ND);
    Tensor gammaCq(dType, gammaCqShape, "gammaCq");
    Tensor gammaCkv(dType, gammaCkvShape, "gammaCkv");
    Tensor dynamicCos(dType, cosShape, "cos");
    Tensor dynamicSin(dType, cosShape, "sin");
    Tensor dynamicCacheIndex(DT_INT32, kvLenShape, "cacheIndex");
    Tensor kvCache(dType, kvCacheShape, "kvCache");
    Tensor krCache(dType, krCacheShape, "krCache");
    Tensor dynamicBlockTable(DT_INT32, blockTableShape, "blockTable");
    Tensor dynamicActSeqs(DT_INT32, actSeqsShape, "actSeqs");
    MlaQuantInputs quantInputs;

    Tensor qW(dType, qWShape, "qW",weightFormat);
    Tensor kW(dType, kWShape, "kW", weightFormat);
    Tensor projW(dType, projWShape, "projW", weightFormat);
    Tensor lnW(dType, lnWShape, "lnW");
    Tensor lnBias(dType, lnWShape, "lnBias");
    Tensor indexKCache(dType, keyShape, "indexKCache");

    SymbolicScalar bSymbol = GetInputShape(dynamicX, 0);
    SymbolicScalar s1Symbol = GetInputShape(dynamicX, 1);
    SymbolicScalar maxBlockNumSymbol = GetInputShape(dynamicBlockTable, 1);

    // tmp input
    Tensor dynamicTmpTopkInput(DT_INT32, tmpTopkInputShape, "tmpTopkInput");

    // output
    Tensor dynamicSaOut(dType, {bSymbol, s1Symbol, n1, dn}, "saOut");

    // tmp output
    Tensor dynamicGatherRes(dType, {bSymbol * s1Symbol * topk, dn + qkRopeHeadDim}, "gatherRes");
    Tensor dynamicTmpRowSumOut(DT_FP32, {bSymbol * s1Symbol * n2, maxBlockNumSymbol * blockSize}, "tmpRowSumOut");
    Tensor dynamicTmpIndexerTopkRes(DT_INT32, {bSymbol, s1Symbol, n2, topk}, "tmpIndexerTopkRes");

    Tensor rmsResOut(dType, {bSymbol, s1Symbol, qLoraRank}, "rmsResOut");
    Tensor queryOut(dType, {bSymbol, s1Symbol, idx_n_heads, idx_head_dim}, "queryOut");
    Tensor weightOut(dType, {bSymbol, s1Symbol, idx_n_heads}, "weightOut");

    Tensor qNopeOut(dType, {bSymbol * s1Symbol , n1, dn}, "qNopeOut");
    Tensor qRopeOut(dType, {bSymbol * s1Symbol , n1, qkRopeHeadDim}, "qRopeOut");

    DeepseekIndexerAttentionPto(dynamicX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, dynamicSin, dynamicCos,
        dynamicCacheIndex, kvCache, krCache, quantInputs, dynamicBlockTable, dynamicActSeqs, qW, kW, projW, lnW, lnBias,
        indexKCache, dynamicSaOut, dynamicGatherRes, dynamicTmpTopkInput, dynamicTmpIndexerTopkRes, dynamicTmpRowSumOut,
        rmsResOut, queryOut, weightOut, qNopeOut, qRopeOut, params);
}

void SetTileConfig(NSASimpleParams &params) {
    // set mla-prolog tileConfig
    MlaTileConfig prologConfig = {4, 1};

    // set rope tileConfig
    RopeTileShapeConfig ropeTileConfigs = {
        {128, 128},
        {32, 128, 128},
        {1, 128, 128, 128}
    };

    // set lighting-indexer-prolog tileConfig
    IndexerTileShapeConfig indexerConfigs {
        {16, 16, 128, 128, 128, 128}, // c1TileShape
        {128, 128, 128, 128}, // v1TileShape
        {16, 16, 128, 128, 128, 128}, // c2TileShape
        {128, 128, 128, 128}  // v2TileShape
    };

    // set selected-attention tileConfig
    SaTileShapeConfig saTileConfig;
    const int gTile = 128;  // for gLoop split
    const int sTile = 1024; // for s2Loop split
    saTileConfig.gTile = gTile;
    saTileConfig.sKvTile = sTile;
    saTileConfig.c1TileShape = {gTile, gTile, 64, 64, 256, 256};   // (n1, dn+dr) @ (s2Tile, dn+dr) -> (n1, s2Tile)
    saTileConfig.v1TileShape = {16, 256};                          // (n1, s2Tile)
    saTileConfig.c2TileShape = {gTile, gTile, 128, 128, 128, 128}; // (n1, s2Tile) @ (s2Tile, dn) -> (n1, d)
    saTileConfig.v2TileShape = {64, 128};                          // (n1, d)

    // set lighting-indexer tileConfig
    IndexerTile indexerTile;
    indexerTile.weightTile = {64, 128};
    indexerTile.c1Tile = {64, 64, 128, 128, 128, 128}; // (m, M), (k, K), (n, N)
    indexerTile.v1Tile = {64, 128};
    indexerTile.topkTile = {1, 2048};
    indexerTile.addsTile = {1, 1, 1, 2048};

    // set tile_config to params
    params.salTileCfg = saTileConfig;
    params.mlaTileCfg = prologConfig;
    params.indexTileCfg = indexerTile;
    params.indexerTileConfigs = indexerConfigs;
    params.ropeTileConfigs = ropeTileConfigs;
}

void DynamicDecodeSparseAttentionPtoPto(uint64_t configKey) {
    NSASimpleParams params = NSASimpleParams::getDecodeParams();

    config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    config::SetCodeGenConfig(KEY_CODEGEN_EXPRESSION_FUSION, true);

    // set params
    params.b = -1;
    params.s1 = -1;
    params.blockNum = -1;
    params.s2 = 131072;
    params.n1 = 128;
    params.n2 = 1;
    params.topk = 2048;
    params.cacheMode = "PA_BSND";

    // Set tile config
    SetTileConfig(params);

    DynamicDecodeSparseAttentionPtoImpl<false, true>(params);
}

} // namespace tile_fwk end

REGISTER_OP(SparseAttentionPto).ImplFunc({
    {npu::tile_fwk::DSA_BF16_ConfigKey, npu::tile_fwk::DynamicDecodeSparseAttentionPtoPto}
});
