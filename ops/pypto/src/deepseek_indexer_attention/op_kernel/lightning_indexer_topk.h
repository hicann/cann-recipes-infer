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
#include "dynamic_nsa_common.h"
#include "tilefwk/config_manager.h"

namespace npu::tile_fwk {
void LightningIndexerTopkInnerImpl(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList = {64, 32, 16, 8, 4, 1}, Tensor *tmpOut = nullptr, Tensor *topkValue = nullptr);

void LightningIndexerTopkInner(const Tensor &query, const Tensor &key, const Tensor &weights, const Tensor &actSeqKey,
    const Tensor &blockTable, Tensor &topkRes, const int selectedCount, IndexerTile tileConfig,
    std::set<int> unrollList = {64, 32, 16, 8, 4, 1});

} // namespace npu::tile_fwk

#endif // INDEXER_TOPK_H