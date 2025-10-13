/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file mla_prolog_v3_proto.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_PROTO_H
#define MLA_PROLOG_V3_PROTO_H

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/ops_log.h"

using namespace ge;

namespace ops {
//input
constexpr uint32_t TOKEN_X_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INDEX = 2;
constexpr uint32_t WEIGHT_UK_INDEX = 3;
constexpr uint32_t ROPE_SIN_INDEX = 7;
constexpr uint32_t KV_CACHE_INDEX = 10;
constexpr uint32_t KR_CACHE_INDEX = 11;
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
// output
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t QUERY_ROPE_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_INDEX = 3;
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;
constexpr uint32_t QUERY_NORM_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_INDEX = 6;
// Attribute
constexpr uint32_t ATTR_QUERY_NORM_FLAG_INDEX = 3;
// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

struct MlaPrologV3ProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t N { 0 };
    int64_t Hckv { 0 };
    int64_t He { 0 };
    int64_t Dr { 0 };
    int64_t Hcq { 0 };
};

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaPrologV3ProtoShapeParam &shapeParam, gert::InferShapeContext* context);

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext* context);


}  // namespace ops

#endif // MLA_PROLOG_V3_PROTO_H