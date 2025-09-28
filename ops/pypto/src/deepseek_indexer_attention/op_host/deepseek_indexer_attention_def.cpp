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

#include "register/op_def_registry.h"
#include "graph/operator_reg.h"

namespace ge {
REG_OP(SparseAttentionPto)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_INT64, DT_INT32, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_BF16, DT_INT64, DT_INT32, DT_FLOAT16}))
    .OP_END_FACTORY_REG(SparseAttentionPto)
}

namespace ops {
class SparseAttentionPto : public OpDef {
public:
    explicit SparseAttentionPto(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("outputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
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
}