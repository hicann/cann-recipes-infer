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
#ifndef SELECTED_ATTENTION
#define SELECTED_ATTENTION

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "dynamic_nsa_common.h"
#include "tilefwk/config_manager.h"

namespace npu::tile_fwk {

void SelectedAttentionCompute(const Tensor &qNope, const Tensor &qRope, const Tensor &kSlc, const Tensor &vSlc,
    const Tensor &kvSlcActSeqs, int nQ, int nKv, float softmaxScale, int topk, Tensor &attentionOut,
    SaTileShapeConfig tileConfig = {});

void SelectedAttention(const Tensor &qNope, const Tensor &qRope, const Tensor &kSlc, const Tensor &vSlc,
    const Tensor &kvSlcActSeqs, int nQ, int nKv, float softmaxScale, int topk, Tensor &attentionOut,
    SaTileShapeConfig tileConfig = {});
}  // namespace npu::tile_fwk

#endif  // SELECTED_ATTENTION
