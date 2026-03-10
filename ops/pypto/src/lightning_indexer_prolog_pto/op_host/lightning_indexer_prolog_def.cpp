/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "graph/operator_reg.h"

namespace ge {
REG_OP(LightningIndexerPrologPto)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_INT64}))
    .ATTR(layernorm_epsilon_k, Float, 1e-6)
    .ATTR(layout_query, String, "TND")
    .ATTR(layout_key, String, "PA_BSND")
    .OP_END_FACTORY_REG(LightningIndexerPrologPto)
}

namespace ops {
class LightningIndexerPrologPto : public OpDef {
public:
    explicit LightningIndexerPrologPto(const char *name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("outputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
}  // namespace ops