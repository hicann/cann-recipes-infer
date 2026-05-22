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
 * \file rotary_position_embedding.cc
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "partial_rotary_mul_quant_tiling.h"
namespace optiling {
constexpr uint32_t MODE_ATTR_IDX = 0;

ge::graphStatus Tiling4PartialRotaryMulQuant(gert::TilingContext *context)
{
    OPS_LOG_I(context, "Tiling4PartialRotaryMulQuant start");
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Tiling4PartialRotaryMulQuant", "Tiling context is null"),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Tiling4PartialRotaryMulQuant", "Tiling platformInfo is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    OPS_ERR_IF(socVersion != platform_ascendc::SocVersion::ASCEND950,
               OPS_LOG_E(context, "Only ascend950 is supported for PartialRotaryMulQuant"),
               return ge::GRAPH_FAILED);

    std::vector<std::unique_ptr<QuantRopeRegBaseTilingClass>> regBaseTilingCases;
    regBaseTilingCases.push_back(std::unique_ptr<QuantRopeRegBaseTilingClass>(
        new QuantRopeRegBaseTilingClassAAndB(context)));
    regBaseTilingCases.push_back(std::unique_ptr<QuantRopeRegBaseTilingClass>(
        new QuantRopeRegBaseTilingClassAB(context)));
    regBaseTilingCases.push_back(std::unique_ptr<QuantRopeRegBaseTilingClass>(
        new QuantRopeRegBaseTilingClassABAAndBA(context)));
    regBaseTilingCases.push_back(std::unique_ptr<QuantRopeRegBaseTilingClass>(
        new QuantRopeRegBaseTilingClassBAB(context)));
    OPS_LOG_I(context, "Using arch35 tiling for ASCEND950");

    for (const auto& ptr : regBaseTilingCases) {
        if (ptr) {
            ge::graphStatus status = ptr->DoTiling();
            if (status != ge::GRAPH_PARAM_INVALID) {
                OPS_LOG_I(context, "Do general op tiling success priority");
                return status;
            }
            OPS_LOG_I(context, "Ignore general op tiling priority");
        }
    }
    OPS_LOG_E(context, "No matching tiling strategy found");
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingPrepareForPartialRotaryMulQuant(gert::TilingParseContext *context)
{
    OPS_LOG_I(context, "TilingPrepareForPartialRotaryMulQuant context success");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PartialRotaryMulQuant)
    .Tiling(Tiling4PartialRotaryMulQuant)
    .TilingParse<QuantRotaryPositionEmbeddingCompileInfo>(TilingPrepareForPartialRotaryMulQuant);
} // namespace optiling