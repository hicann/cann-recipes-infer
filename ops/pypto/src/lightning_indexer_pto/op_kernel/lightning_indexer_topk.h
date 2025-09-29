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

#pragma once
#ifndef INDEXER_TOPK_H
#define INDEXER_TOPK_H

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/config_manager.h"

namespace npu::tile_fwk {

constexpr const int SHAPE_DIM0 = 0;
constexpr const int SHAPE_DIM1 = 1;
constexpr const int SHAPE_DIM2 = 2;
constexpr const int SHAPE_DIM3 = 3;
constexpr const int SHAPE_DIM4 = 4;
constexpr const int SHAPE_DIM5 = 5;

struct IndexerTile {
    std::vector<int64_t> weightTile;
    std::array<int64_t, TILE_CUBE_DIMS> c1Tile; // (m, M), (k, K), (n, N)
    std::vector<int64_t> v1Tile;
    std::vector<int64_t> topkTile;
    std::vector<int64_t> addsTile;
};

void LightningIndexerTopkImpl(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList = {64, 32, 16, 8, 4, 1}, Tensor *tmpOut = nullptr, Tensor *topkValue = nullptr);

void LightningIndexerTopk(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList = {64, 32, 16, 8, 4, 1});

} // namespace npu::tile_fwk

#endif // INDEXER_TOPK_H