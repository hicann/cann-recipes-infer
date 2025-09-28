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
#ifndef PARALLEL_SORT
#define PARALLEL_SORT

#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk {

std::tuple<Tensor, Tensor> L2Sort(Tensor x, int tileSize, bool descending);

std::tuple<Tensor, Tensor> L2SortWithIndex(Tensor x, Tensor idx, int tileSize, bool descending);

void ParallelSort(const Tensor &x, Tensor &y, Tensor &yIdx, int tileSize = 4096, bool descending = true);

void SingleOpTopK(const Tensor &x, Tensor &kIdx, int k = 2048, bool descending = true);

void SingleOpTopKWithIndex(const Tensor &x, const Tensor &idx, Tensor &yIdx, int k = 2048, bool descending = true);

}  // namespace npu::tile_fwk

#endif  // PARALLEL_SORT