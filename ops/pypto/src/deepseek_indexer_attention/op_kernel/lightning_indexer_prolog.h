/* *
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  * See LICENSE in the root of
the software repository for the full text of the License.
 */

#pragma once
#ifndef INDEXER_PROLOG_H
#define INDEXER_PROLOG_H

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "dynamic_nsa_common.h"

namespace npu::tile_fwk {

struct IndexerShapeParams {
    int b;
    int seq;
    int dim;
    int qLoraRank;
    int headDim;
    int headNum;
    int ropeHeadDim;
    int blockSize; // PA block size
    int blockNum;
    int nKV;
    int s2;
    int tileBS = 2;
    IndexerTileShapeConfig indexerTileConfigs;
    RopeTileShapeConfig ropeTileConfigs;
};

struct IndexerPrologInput {
    const Tensor &x;
    const Tensor &qr;
    const Tensor &qW;
    const Tensor &kW;
    const Tensor &projW;
    const Tensor &lnW;
    const Tensor &lnBias;
    const Tensor &cos;
    const Tensor &sin;
    const Tensor &kCache;
    const Tensor &kCacheIndex;
    const Tensor &blockTable;
};

struct IndexerPrologOutput {
    Tensor &query;
    Tensor &weight;
    Tensor &kCacheOut;
};

void LightningIndexerPrologCompute(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params);

void LightningIndexerProlog(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params);

void LightningIndexerFP8(const Tensor &x, const Tensor &qr, Tensor &qFP8, Tensor &qScale, Tensor &kFP8, Tensor &kScale);

} // namespace npu::tile_fwk

#endif // INDEXER_PROLOG_H
