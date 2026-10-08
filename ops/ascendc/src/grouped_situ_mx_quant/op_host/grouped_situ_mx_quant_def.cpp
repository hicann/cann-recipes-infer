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
 * \file grouped_situ_mx_quant_def.cpp
 * \brief
 *
 * Fusion of situ_and_mul_sparse (sparse SiTU gated activation) and the MXFP8
 * (e4m3fn, blocksize=32) tail-axis dynamic quantization of dynamic_mx_quant.
 * Input x holds the gate/up halves concatenated on the last axis (2*EPR);
 * the fused op computes the sparse SiTU activation act[.., EPR] in registers
 * (never spilled to GM) and quantizes it to y(fp8_e4m3fn) + mxscale(e8m0)
 * with group_size=32 along the last axis. Bit-identical to the two small ops
 * chained (npu_situ_and_mul_sparse -> npu_dynamic_mx_quant, dst=e4m3fn).
 */

#include "register/op_def_registry.h"

namespace ops {
class GroupedSituMxQuant : public OpDef {
public:
    explicit GroupedSituMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expert_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("high_precision").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_situ_mx_quant");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(GroupedSituMxQuant);
}  // namespace ops
