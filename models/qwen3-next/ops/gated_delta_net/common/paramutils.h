/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"

namespace npu_ops_transformer_ext {
using namespace AscendC;

constexpr int STRIDE_FLOAT = 8;
constexpr int CAST_STRIDE_HALF = 4;

__aicore__ inline UnaryRepeatParams MakeDefaultUnaryRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = STRIDE_FLOAT;
    p.srcRepStride = STRIDE_FLOAT;
    return p;
}

__aicore__ inline UnaryRepeatParams CastHalf2FloatRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = STRIDE_FLOAT;
    p.srcRepStride = CAST_STRIDE_HALF;
    return p;
}

__aicore__ inline UnaryRepeatParams CastFloat2HalfRepeatParams()
{
    UnaryRepeatParams p;
    p.dstBlkStride = 1;
    p.srcBlkStride = 1;
    p.dstRepStride = CAST_STRIDE_HALF;
    p.srcRepStride = STRIDE_FLOAT;
    return p;
}

__aicore__ inline BinaryRepeatParams MakeDefaultBinaryRepeatParams()
{
    BinaryRepeatParams p;
    p.dstBlkStride = 1;
    p.src0BlkStride = 1;
    p.src1BlkStride = 1;
    p.dstRepStride = STRIDE_FLOAT;
    p.src0RepStride = STRIDE_FLOAT;
    p.src1RepStride = STRIDE_FLOAT;
    return p;
}

}