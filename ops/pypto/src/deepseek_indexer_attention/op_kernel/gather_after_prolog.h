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
#include "tilefwk/symbolic_scalar.h"
#ifndef GATHER_AFTER_PROLOG
#define GATHER_AFTER_PROLOG

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "dynamic_nsa_common.h"


namespace npu::tile_fwk {

void GatherAfterPrologCompute(Tensor &topKIndcies, Tensor &kNopeCache, Tensor &kRopeCache, Tensor &blockTable, Tensor &actSeqs,
   Tensor &gatherRes, const NSASimpleParams &params, SymbolicScalar b, SymbolicScalar s1);

} // namespace npu::tile_fwk

#endif // GATHER_AFTER_PROLOG
