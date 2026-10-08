/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_situ_mx_quant.cpp
 * \brief
 */

#include "grouped_situ_mx_quant_base.h"

using namespace AscendC;
using namespace GroupedSituMxQuant;

extern "C" __global__ __aicore__ void grouped_situ_mx_quant(
    GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // Save the caller's overflow mode; the per-row Compute sets saturation (0)
    // for the e4m3 quant cast. Restore on exit so the SPR does not leak out.
#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
    TPipe pipe;
    GroupedSituMxQuantKernel<DTYPE_X> op;
    op.Init(x, expertTokens, y, mxscale, &tilingData, &pipe);
    op.Process();
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
