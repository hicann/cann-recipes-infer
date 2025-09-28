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
#ifndef DEEPSEEK_INDEXER_ATTENTION
#define DEEPSEEK_INDEXER_ATTENTION

#include "tilefwk/tilefwk.h"
#include "dynamic_mla.h"
#include "gather_after_prolog.h"
#include "selected_attention.h"
#include "dynamic_nsa_common.h"
#include "lightning_indexer_topk.h"
#include "lightning_indexer_prolog.h"

namespace npu::tile_fwk {
void DeepseekIndexerAttentionPto(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, Tensor &blockTable,
    Tensor &actSeqs, const Tensor &qW, const Tensor &kW, const Tensor &projW, const Tensor &lnW, const Tensor &lnBias,
    const Tensor &indexKCache, Tensor &attentionOut, Tensor &gatherResTmp, Tensor &tmpTopkInput,
    Tensor &tmpIndexerTopkRes, Tensor &tmpRowSumOut, Tensor &rmsResOut, Tensor &queryOut, Tensor &weightsOut,
    Tensor &qNopeOut, Tensor &qRopeOut, const NSASimpleParams &params);

} // namespace npu::tile_fwk

#endif // DEEPSEEK_INDEXER_ATTENTION
