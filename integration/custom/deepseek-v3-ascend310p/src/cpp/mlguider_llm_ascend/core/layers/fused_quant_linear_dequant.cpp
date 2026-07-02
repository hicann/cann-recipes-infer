/**
 * Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "fused_quant_linear_dequant.h"

#include "acl_utils.h"
#include "linear_op.h"
#include "logger.h"
#include "quant_op.h"
#include "per_token_quant_op.h"
#include "per_token_per_channel_dequant_op.h"

void FusedQuantLinearDequant(aclrtStream& stream, void* input, void* weight, void* smoothFactor, void* wScale, int m, int n, int k, void* qx, void* outI32, void* output)
{
    // Quant FP16 -> Int8
    QuantizeFP16ToInt8(stream, input, smoothFactor, qx, m, k);

    // W8A8
    LinearI8(stream, qx, weight, outI32, m, n, k);

    DequantizeInt32ToFp16(stream, outI32, wScale, output, m, n);
}

void FusedQuantLinearDequantDynamic(aclrtStream& stream, void* input, void* weight, void* xScale, void* wScale, int m, int n, int k, void* qx, void* outI32, void* output)
{
    // Quant FP16 -> Int8
    PerTokenQuant(stream, input, qx, xScale, m, k);

    // W8A8
    LinearI8(stream, qx, weight, outI32, m, n, k);

    PerTokenPerChannelDequant(stream, outI32, wScale, xScale, output, m, n);
}
