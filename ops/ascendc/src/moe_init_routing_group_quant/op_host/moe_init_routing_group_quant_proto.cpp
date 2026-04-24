/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v3_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {
const size_t DIM_ONE = 1U;
const size_t DIM_TWO = 2U;
const size_t DIM_THREE = 3U;
const int64_t NEG_ONE = static_cast<int64_t>(-1);
const int64_t NEG_TWO = static_cast<int64_t>(-2);
const int64_t MOE_INIT_ROUTING_V3_INPUT_X = 0;
const int64_t MOE_INIT_ROUTING_V3_INPUT_EXPERT_IDX = 1;
const int64_t MOE_INIT_ROUTING_V3_INPUT_SCALE = 2;
const int64_t MOE_INIT_ROUTING_V3_INPUT_OFFSET = 3;
const int64_t MOE_INIT_ROUTING_V3_ATTR_ACTIVE_NUM = 0;
const int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_CAPACITY = 1;
const int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_NUM = 2;
const int64_t MOE_INIT_ROUTING_V3_ATTR_DROP_PAD_MODE = 3;
const int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_TYPE = 4;
const int64_t MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_FLAG = 5;
const int64_t MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE = 6;
const int64_t MOE_INIT_ROUTING_V3_ATTR_ACTIVE_EXPERT_RANGE = 7;
const int64_t MOE_INIT_ROUTING_V3_ATTR_ROW_IDX_TYPE = 8;
const int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X = 0;
const int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX = 1;
const int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT = 2;
const int64_t MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE = 3;
const int64_t MOE_INIT_ROUTING_V3_EXPERT_END_BOUND = 10240;
const int64_t KEY_VALUE_MODE_DIM0_NUM = 2;
const int64_t MX_QUANT_BLOCK_SIZE = 32;
const int64_t FP8_QUANT_GROUP_SIZE = 128;

enum DropPadMode : int8_t {
    NO_DROP_PAD = 0,
    DROP_PAD = 1,
};

enum QuantMode : int8_t {
    NON_QUANT = -1,
    STATIC_QUANT = 0,
    DYNAMIC_QUANT = 1,
    MXQUANT_FP8_E5M2 = 2,
    MXQUANT_FP8_E4M3FN = 3,
    GROUPQUANT_FP8_E5M2 = 4,
    GROUPQUANT_FP8_E4M3 = 5
};

enum ExpertTokenNumType : int8_t {
    CUMSUM = 0,
    COUNT = 1,
    KEY_VALUE = 2
};

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 <= NEG_ONE || dim2 <= NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus GetAndCheckAttrActiveExpertRange(const gert::RuntimeAttrs *attrs,
                                                        gert::InferShapeContext *context, int64_t &expertStart,
                                                        int64_t &expertEnd, int64_t &experNum)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckAttrActiveExpertRange.");
    // Check if active_expert_range size is 2 and if expert_start < expert_end
    auto activeExpertRangePtr = attrs->GetListInt(MOE_INIT_ROUTING_V3_ATTR_ACTIVE_EXPERT_RANGE);
    if (nullptr == activeExpertRangePtr) {
        OPS_LOG_E(context, "The active_expert_range should be list int. But it is none.");
        return ge::GRAPH_FAILED;
    }
    int64_t activeExpertRangeSize = activeExpertRangePtr->GetSize();
    if (activeExpertRangePtr->GetSize() == DIM_TWO) {
        expertStart = activeExpertRangePtr->GetData()[0];
        expertEnd = activeExpertRangePtr->GetData()[1];
        if (expertStart >= expertEnd || expertStart < 0 || expertEnd > MOE_INIT_ROUTING_V3_EXPERT_END_BOUND) {
            OPS_LOG_E(context,
                    "The active_expert_range should be in [0, %ld), but the active_expert_range is [%ld, %ld).",
                    MOE_INIT_ROUTING_V3_EXPERT_END_BOUND, expertStart, expertEnd);
            return ge::GRAPH_FAILED;
        }
    } else if (activeExpertRangePtr->GetSize() == 0) {
        expertStart = 0;
        expertEnd = experNum;
    } else {
        OPS_LOG_E(context, "The active_expert_range size should be 2, but its size is %ld.", activeExpertRangeSize);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrActiveExpertRange.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrActiveNum(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &activeNum, int64_t &dropPadMode)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckAttrActiveNum.");
    const int64_t *activeNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_ACTIVE_NUM);
    if (nullptr == activeNumPtr) {
        OPS_LOG_E(context, "The active_num should not be none.");
        return ge::GRAPH_FAILED;
    }
    activeNum = *activeNumPtr;
    if (dropPadMode == DropPadMode::NO_DROP_PAD && activeNum < -1) {
        OPS_LOG_E(context, "The active_num should be greater than or equal to 0. But it is %ld.", activeNum);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrActiveNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertCapacity(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                     const gert::Shape *xShape, int64_t &expertCapacity,
                                                     int64_t &dropPadMode)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckAttrExpertCapacity.");
    const int64_t *expertCapacityPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_CAPACITY);
    if (nullptr == expertCapacityPtr) {
        OPS_LOG_E(context, "The expert_capacity should not be none.");
        return ge::GRAPH_FAILED;
    }
    expertCapacity = *expertCapacityPtr;
    if (dropPadMode == DropPadMode::DROP_PAD && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
        OPS_LOG_E(context, "The expert_capacity should be between 0 and %d. But it is %ld.", xShape->GetDim(0),
                expertCapacity);
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(context, "End to do GetAndCheckAttrExpertCapacity.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertNum(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &experNum)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckexperNum.");
    const int64_t *experNumPtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_NUM);
    if (nullptr == experNumPtr) {
        OPS_LOG_E(context, "The expert_num should not be none.");
        return ge::GRAPH_FAILED;
    }
    experNum = *experNumPtr;
    if (experNum <= 0 || experNum > MOE_INIT_ROUTING_V3_EXPERT_END_BOUND) {
        OPS_LOG_E(context, "The expert_num should be greater than 0. But it is %ld.", experNum);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrExpertNum.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrDropPadMode(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                  int64_t &dropPadMode)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckAttrDropPadMode.");
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_DROP_PAD_MODE);
    if (nullptr == dropPadModePtr) {
        OPS_LOG_E(context, "The RuntimeAttrs for drop_pad_mode is none.");
        return ge::GRAPH_FAILED;
    }

    dropPadMode = *dropPadModePtr;
    if (dropPadMode < DropPadMode::NO_DROP_PAD || dropPadMode > DropPadMode::DROP_PAD) {
        OPS_LOG_E(context, "The drop_pad_mode should be %d or %d. But it is %ld.", DropPadMode::NO_DROP_PAD,
                DropPadMode::DROP_PAD, dropPadMode);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrDropPadMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumType(const gert::RuntimeAttrs *attrs,
                                                         gert::InferShapeContext *context, int64_t &experTokenNumType)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckexperTokenNumType.");
    const int64_t *experTokenNumTypePtr =
        attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_TYPE);
    if (nullptr == experTokenNumTypePtr) {
        OPS_LOG_E(context, "The expert_token_num_type should not be none.");
        return ge::GRAPH_FAILED;
    }
    experTokenNumType = *experTokenNumTypePtr;
    if (experTokenNumType < ExpertTokenNumType::CUMSUM || experTokenNumType > ExpertTokenNumType::KEY_VALUE) {
        OPS_LOG_E(context, "The expert_token_num_type should be %d, %d or %d. But it is %ld.", ExpertTokenNumType::CUMSUM,
                ExpertTokenNumType::COUNT, ExpertTokenNumType::KEY_VALUE, experTokenNumType);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrExpertTokenNumFlag(const gert::RuntimeAttrs *attrs,
                                                         gert::InferShapeContext *context, bool &experTokenNumFlag)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckexperTokenNumType.");
    const bool *experTokenNumFlagPtr = attrs->GetAttrPointer<bool>(MOE_INIT_ROUTING_V3_ATTR_EXPERT_TOKEN_NUM_FLAG);
    if (nullptr == experTokenNumFlagPtr) {
        OPS_LOG_E(context, "The expert_token_num_flag should not be none.");
        return ge::GRAPH_FAILED;
    }
    experTokenNumFlag = *experTokenNumFlagPtr;
    OPS_LOG_D(context, "End to do GetAndCheckAttrExpertTokenNumType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrQuantMode(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                int64_t &quantMode)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckQuantMode.");
    if (nullptr == attrs) {
        OPS_LOG_E(context, "The RuntimeAttrs for quant_mode is none.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OPS_LOG_E(context, "The quant_mode should not be null.");
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    if (quantMode < QuantMode::NON_QUANT || quantMode > QuantMode::GROUPQUANT_FP8_E5M2) {
        OPS_LOG_E(context, "The quant_mode should be in [%d, %d]. But it is %d.", QuantMode::NON_QUANT,
                QuantMode::NON_QUANT, QuantMode::GROUPQUANT_FP8_E5M2, quantMode);
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(context, "End to do GetAndCheckQuantMode.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckAttrRowIdxType(const gert::RuntimeAttrs *attrs, gert::InferShapeContext *context,
                                                 int64_t &rowIdxType, int64_t &dropPadMode)
{
    OPS_LOG_D(context, "Begin to do GetAndCheckAttrRowIdxType.");
    if (nullptr == attrs) {
        OPS_LOG_E(context, "The RuntimeAttrs for row_Idx_type is none.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *dropPadModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_DROP_PAD_MODE);
    dropPadMode = *dropPadModePtr;

    const int64_t *rowIdxTypePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_ROW_IDX_TYPE);
    if (nullptr == rowIdxTypePtr) {
        OPS_LOG_E(context, "The row_Idx_type should be 0 or 1. But it is none.");
        return ge::GRAPH_FAILED;
    }
    rowIdxType = *rowIdxTypePtr;
    if (dropPadMode == DropPadMode::DROP_PAD && rowIdxType != 0) {
        OPS_LOG_E(context, "The row_Idx_type should be 0 when dropPadMode is equal to 1 But it is %ld.", rowIdxType);
        return ge::GRAPH_FAILED;
    }

    if (rowIdxType < 0 || rowIdxType > 1) {
        OPS_LOG_E(context, "The row_Idx_type should be 0 or 1 But it is %ld.", rowIdxType);
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_D(context, "End to do GetAndCheckAttrRowIdxType.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputScaleShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                            const gert::Shape *scaleShape, const int64_t expertStart,
                                            const int64_t expertEnd, const int64_t quantMode)
{
    // When quant_mode is STATIC_QUANT, scale cannot be none.
    OPS_ERR_IF((nullptr == scaleShape && QuantMode::STATIC_QUANT == quantMode),
                OPS_LOG_E(context, "The scale cannot be none when quant_mode is %ld.", quantMode),
                return ge::GRAPH_FAILED);

    //  When quant_mode is NON_QUANT/DYNAMIC_QUANT/MXQUANT_FP8_E5M2/MXQUANT_FP8_E4M3FN, scale can be none.
    OPS_ERR_IF((nullptr == scaleShape &&
                 (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode ||
                  QuantMode::MXQUANT_FP8_E5M2 == quantMode || QuantMode::MXQUANT_FP8_E4M3FN == quantMode ||
                  QuantMode::GROUPQUANT_FP8_E4M3 == quantMode || QuantMode::GROUPQUANT_FP8_E5M2 == quantMode)),
                OPS_LOG_I(context, "When quant_mode is %ld , scale can be none.", quantMode), return ge::GRAPH_SUCCESS);

    if (QuantMode::NON_QUANT == quantMode) {
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OPS_ERR_IF(scaleShape->GetDim(0) < 0 && scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO,
                        OPS_LOG_E(context,
                                "When quant_mode is %ld and use scale in dynamic graph, The shape of scale should be (-1) or (-2).",
                                quantMode),
                        return ge::GRAPH_FAILED);
            OPS_ERR_IF(scaleShape->GetDim(0) > 0 && !isSameDim(scaleShape->GetDim(0), xShape->GetDim(0)),
                        OPS_LOG_E(context,
                                "When quant_mode is %ld and use scale in static graph, The shape of scale should be (%ld,).",
                                quantMode, xShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_E(context, "When quant_mode is %ld, The dimNum of scale should be 1, current shape is (%ld).", quantMode,
                    scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::STATIC_QUANT == quantMode) {
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OPS_ERR_IF(
                scaleShape->GetDim(0) != NEG_ONE && scaleShape->GetDim(0) != NEG_TWO &&
                    !isSameDim(scaleShape->GetDim(0), DIM_ONE),
                OPS_LOG_E(
                    context,
                    "When quant_mode is %ld, the shape of scale should be (-1) or (-2) or (1,).", quantMode),
                return ge::GRAPH_FAILED);
        } else {
            OPS_LOG_E(context, "When quant_mode is %ld, the dimNum of scale should be (1,), current shape is (%ld).",
                    quantMode, scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else if (QuantMode::DYNAMIC_QUANT == quantMode) {
        int64_t activeExpertRange = expertEnd - expertStart;
        if (scaleShape->GetDimNum() == DIM_ONE) {
            OPS_ERR_IF(scaleShape->GetDim(0) != NEG_TWO,
                        OPS_LOG_E(context,
                                "When quant_mode is %ld and scale dim is 1 in dynamic graph, the first dim of scale should be -2, but "
                                "its shape is (%ld).",
                                quantMode, scaleShape->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else if (scaleShape->GetDimNum() == DIM_TWO) {
            if (scaleShape->GetDim(0) > 0) {
                OPS_ERR_IF(
                    !isSameDim(scaleShape->GetDim(0), activeExpertRange) && !isSameDim(scaleShape->GetDim(0), DIM_ONE),
                    OPS_LOG_E(
                        context,
                        "When quant_mode is %ld in static graph, the first dim of scale should be 1 or %ld, but its shape is (%ld).",
                        quantMode, activeExpertRange, scaleShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
                OPS_ERR_IF(
                    !isSameDim(scaleShape->GetDim(1), xShape->GetDim(1)),
                    OPS_LOG_E(
                        context,
                        "When quant_mode is %ld in static graph, the second dim of scale should or %ld, but its shape is (%ld).",
                        quantMode, xShape->GetDim(1), scaleShape->GetDim(0)),
                    return ge::GRAPH_FAILED);
            } else {
                OPS_ERR_IF(
                    scaleShape->GetDim(0) != NEG_ONE || (scaleShape->GetDim(1) != NEG_ONE && scaleShape->GetDim(1) != xShape->GetDim(1)),
                    OPS_LOG_E(context,
                            "When quant_mode is %ld and scale dim is 2 in dynamic graph, the shape of scale should be (-1, -1) or (-1, %d).",
                            quantMode, xShape->GetDim(1)),
                    return ge::GRAPH_FAILED);
            }
        } else {
            OPS_LOG_E(
                context,
                "When quant_mode is %ld, the dimNum of scale should be 1(dynamic graph) or 2, but its shape is (%ld).",
                scaleShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputOffsetShape(gert::InferShapeContext *context, const gert::Shape *offsetShape,
                                             const int64_t expertStart, const int64_t expertEnd,
                                             const int64_t quantMode)
{
    // The shape of offset can be none.
    if (quantMode != QuantMode::STATIC_QUANT) {
        return ge::GRAPH_SUCCESS;
    } else if (nullptr == offsetShape) {
        return ge::GRAPH_FAILED;
    }

    if (offsetShape->GetDimNum() != DIM_ONE) {
        OPS_LOG_E(context, "The dimNum of offset should be 1, current shape is (%ld).", offsetShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (offsetShape->GetDim(0) != NEG_ONE && offsetShape->GetDim(0) != NEG_TWO && !isSameDim(offsetShape->GetDim(0), DIM_ONE)) {
        OPS_LOG_E(context,
                "The shape of offset should be (1,) in static graph or (-2), (-1,) in dynamic graph.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::InferShapeContext *context, const gert::Shape *xShape,
                                       const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                       const gert::Shape *offsetShape, const int64_t expertStart,
                                       const int64_t expertEnd, const int64_t quantMode)
{
    // Check the shape of input_x
    if (xShape->GetDimNum() == DIM_ONE) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context, "The dynamic dim of x should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context, "The dim of x should be 2 or dynamic");
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        OPS_LOG_E(context, "Invalid x shape");
        return ge::GRAPH_FAILED;
    }

    // Check the shape of expert_idx
    if (expertIdxShape->GetDimNum() == DIM_ONE) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OPS_LOG_E(context, "The dynamic dim of expert_idx should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        OPS_LOG_E(context, "The dim of expert_idx should be 2 or dynamic");
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        OPS_LOG_E(context, "Invalid expert_idx shape");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, expert_idx_n)) {
        OPS_LOG_E(context, "The first dim of x and expert_idx should be same.");
        return ge::GRAPH_FAILED;
    }
    // Check the shape of scale
    if (CheckInputScaleShape(context, xShape, scaleShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check the shape of offset
    if (CheckInputOffsetShape(context, offsetShape, expertStart, expertEnd, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeAndAttrInfo(gert::InferShapeContext *context, const gert::Shape *xShape,
                                      const gert::Shape *expertIdxShape, const gert::Shape *scaleShape,
                                      const gert::Shape *offsetShape, const int64_t expertStart,
                                      const int64_t expertEnd, const int64_t quantMode, const int64_t rowIdxType)
{
    // scale is optional and can be none.
    if (nullptr == scaleShape) {
        OPS_LOG_D(context, "scale_shape is: none.");
    }
    // offset is optional and can be none.
    OPS_LOG_D(context, "Begin print offset_shape.");
    if (nullptr == offsetShape) {
        OPS_LOG_D(context, "offset_shape is: none.");
    }
    OPS_LOG_D(context, "End print offset_shape.");

    // Attrs are all required.
    OPS_LOG_D(context, "active_expert_range is: [%ld, %ld).", expertStart, expertEnd);
    OPS_LOG_D(context, "quant_mode is: %ld.", quantMode);
    OPS_LOG_D(context, "row_Idx_type is: %ld.", rowIdxType);
}

static ge::graphStatus InferShape4MoeInitRoutingV3(gert::InferShapeContext *context)
{
    OPS_LOG_D(context, "Begin to do MoeInitRoutingV3Infershape.");
    // 1. Get and check input shape
    // 1.1 Get and check input_x
    const gert::Shape *xShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_X);
    OPS_LOG_E_IF_NULL(context, xShape, ge::GRAPH_FAILED);

    // 1.2 Get and check expert_idx
    const gert::Shape *expertIdxShape = context->GetInputShape(MOE_INIT_ROUTING_V3_INPUT_EXPERT_IDX);
    OPS_LOG_E_IF_NULL(context, expertIdxShape, ge::GRAPH_FAILED);

    // 1.3 Get scale shape without checking null, because scale is optional and can be none.
    const gert::Shape *scaleShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_V3_INPUT_SCALE);

    // 1.4 Get offset shape without checking null, because offset is optional and can be none.
    const gert::Shape *offsetShape = context->GetOptionalInputShape(MOE_INIT_ROUTING_V3_INPUT_OFFSET);
    // 2. Get and check attrs
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, ge::GRAPH_FAILED);

    // 2.1 Get and check expert_num attr
    int64_t experNum = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertNum(attrs, context, experNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.2 Get and check active_expert_range attr
    int64_t expertStart = static_cast<int64_t>(-1);
    int64_t expertEnd = static_cast<int64_t>(-1);
    if (GetAndCheckAttrActiveExpertRange(attrs, context, expertStart, expertEnd, experNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (nullptr == attrs) {
        OPS_LOG_E(context, "The attrs is none.");
        return ge::GRAPH_FAILED;
    }

    // 2.3 Get and check drop_pad_mode attr
    int64_t dropPadMode = static_cast<int64_t>(-1);
    if (GetAndCheckAttrDropPadMode(attrs, context, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.4 Get and check active_num attr
    int64_t activeNum = static_cast<int64_t>(-1);
    if (GetAndCheckAttrActiveNum(attrs, context, activeNum, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.5 Get and check expert_capacity attr
    int64_t expertCapacity = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertCapacity(attrs, context, xShape, expertCapacity, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.6 Get and check expert_token_num_type attr
    int64_t expertTokenNumType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrExpertTokenNumType(attrs, context, expertTokenNumType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.7 Get and check expert_token_num_type attr
    bool expertTokenNumFlag = false;
    if (GetAndCheckAttrExpertTokenNumFlag(attrs, context, expertTokenNumFlag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.8 Get and check quant_mode attr
    int64_t quantMode = static_cast<int64_t>(-1);
    if (GetAndCheckAttrQuantMode(attrs, context, quantMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 2.9 Get and check row_Idx_type attr
    int64_t rowIdxType = static_cast<int64_t>(-1);
    if (GetAndCheckAttrRowIdxType(attrs, context, rowIdxType, dropPadMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Check input shape
    if (CheckInputShape(context, xShape, expertIdxShape, scaleShape, offsetShape, expertStart, expertEnd, quantMode) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. Infer output shape
    // 3.1 Prepare output shape
    gert::Shape *expandedXShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X);
    OPS_LOG_E_IF_NULL(context, expandedXShape, ge::GRAPH_FAILED);
    gert::Shape *expandedRowIdxShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX);
    OPS_LOG_E_IF_NULL(context, expandedRowIdxShape, ge::GRAPH_FAILED);
    gert::Shape *expertTokenCumsumOrCountShape =
        context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT);
    OPS_LOG_E_IF_NULL(context, expertTokenCumsumOrCountShape, ge::GRAPH_FAILED);
    gert::Shape *expandedScaleShape = context->GetOutputShape(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE);
    OPS_LOG_E_IF_NULL(context, expandedScaleShape, ge::GRAPH_FAILED);

    int64_t x_n = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == DIM_ONE ? NEG_ONE : xShape->GetDim(1);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t k = expertIdxShape->GetDimNum() == DIM_ONE ? NEG_ONE : expertIdxShape->GetDim(1);
    int64_t n = x_n > expert_idx_n ? x_n : expert_idx_n;
    if (n > 0 && k > 0) {
        if (activeNum == 0 || activeNum == -1) {
            activeNum = n * k;
        } else {
            activeNum = std::min(activeNum, n * k);
        }
    }

    int64_t xOutDimNum = activeNum < n * k ? activeNum : n * k;
    int64_t outNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : n * k;
    int64_t xOutNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : xOutDimNum;
    // 3.2 Set output expanded_x shape
    if (dropPadMode == DropPadMode::NO_DROP_PAD) {
        expandedXShape->SetDimNum(DIM_TWO);
        expandedXShape->SetDim(0U, xOutNum);
        expandedXShape->SetDim(DIM_ONE, cols);
    } else {
        expandedXShape->SetDimNum(DIM_THREE);
        expandedXShape->SetDim(0U, experNum);
        expandedXShape->SetDim(DIM_ONE, expertCapacity);
        expandedXShape->SetDim(DIM_TWO, cols);
    }

    // 3.3 Set output expanded_row_idx shape
    expandedRowIdxShape->SetDimNum(DIM_ONE);
    expandedRowIdxShape->SetDim(0U, outNum);

    // 3.4 Set output expert_token_cumsum_or_count shape
    if (expertTokenNumFlag) {
        if (expertTokenNumType == ExpertTokenNumType::KEY_VALUE) {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_TWO);
            expertTokenCumsumOrCountShape->SetDim(0U, experNum);
            expertTokenCumsumOrCountShape->SetDim(DIM_ONE, KEY_VALUE_MODE_DIM0_NUM);
        } else {
            expertTokenCumsumOrCountShape->SetDimNum(DIM_ONE);
            expertTokenCumsumOrCountShape->SetDim(0U, expertEnd - expertStart);
        }
    }

    //  3.5 Set output expanded_scale shape
    //  When scale_shape=(b*s) and non-quant, or it is dynamic quant mode, the shape of expanded_scale should be (b*s*k)
    if (QuantMode::NON_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        expandedScaleShape->SetDimNum(DIM_ONE);
        if (dropPadMode == DropPadMode::NO_DROP_PAD) {
            expandedScaleShape->SetDim(0U, xOutNum);
        } else {
            expandedScaleShape->SetDim(0U, experNum * expertCapacity);
        }
    } else if (quantMode == QuantMode::MXQUANT_FP8_E5M2 || quantMode == QuantMode::MXQUANT_FP8_E4M3FN) {
        expandedScaleShape->SetDimNum(DIM_THREE);
        expandedScaleShape->SetDim(0U, outNum);
        if (cols == NEG_ONE) {
            expandedScaleShape->SetDim(1U, NEG_ONE);
            expandedScaleShape->SetDim(2U, 2LL);
        } else {
            expandedScaleShape->SetDim(1U, CeilAlign<int64_t>(CeilDiv<int64_t>(cols, MX_QUANT_BLOCK_SIZE), 2LL) / 2LL);
            expandedScaleShape->SetDim(2U, 2LL);
        }
    } else if (quantMode == QuantMode::GROUPQUANT_FP8_E4M3 || quantMode == QuantMode::GROUPQUANT_FP8_E5M2) {
        expandedScaleShape->SetDimNum(DIM_TWO);
        expandedScaleShape->SetDim(0U, outNum);
        int64_t dim1 = (cols == NEG_ONE) ? NEG_ONE : CeilDiv<int64_t>(cols, FP8_QUANT_GROUP_SIZE);
        expandedScaleShape->SetDim(1U, dim1);
    }

    OPS_LOG_D(context, "End to do MoeInitRoutingV3Infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingV3(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context, "Begin to do MoeInitRoutingV3InferDataType.");

    // Get and check quant_mode attr
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, ge::GRAPH_FAILED);
    int64_t quantMode = static_cast<int64_t>(-1);
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(MOE_INIT_ROUTING_V3_ATTR_QUANT_MODE);
    if (nullptr == quantModePtr) {
        OPS_LOG_E(context, "The quant_mode should be in range [%d, %d]. But it is none.", QuantMode::NON_QUANT,
                QuantMode::MXQUANT_FP8_E4M3FN);
        return ge::GRAPH_FAILED;
    }
    quantMode = *quantModePtr;
    // Infer output dtype according quant_mode
    auto xDtype = context->GetInputDataType(MOE_INIT_ROUTING_V3_INPUT_X);
    auto expandedXDtype = xDtype;           // default same as dtype(x)
    auto expandedScaleDtype = ge::DT_FLOAT; // default float32
    if (QuantMode::STATIC_QUANT == quantMode || QuantMode::DYNAMIC_QUANT == quantMode) {
        if (ge::DT_INT8 == xDtype) {
            OPS_LOG_E(context, "When quant_mode=%ld, xDtype cannot be int_8.", quantMode);
            return ge::GRAPH_FAILED;
        }
        expandedXDtype = ge::DT_INT8;
    } else if (QuantMode::MXQUANT_FP8_E5M2 == quantMode || QuantMode::MXQUANT_FP8_E4M3FN == quantMode) {
        if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16) {
            OPS_LOG_E(
                context,
                "When quant_mode=%ld, xDtype should be DT_FLOAT16 or DT_BF16. Current got unexpected dtype id of %d.",
                quantMode, xDtype);
            return ge::GRAPH_FAILED;
        }
        expandedXDtype = (QuantMode::MXQUANT_FP8_E5M2 == quantMode) ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT8_E8M0;
    } else if (QuantMode::GROUPQUANT_FP8_E4M3 == quantMode || QuantMode::GROUPQUANT_FP8_E5M2 == quantMode) {
        if (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT) {
            OPS_LOG_E(
                context,
                "When quant_mode=%ld, xDtype should be DT_FLOAT16, DT_BF16 or DT_FLOAT. Current got unexpected dtype id of %d.",
                quantMode, xDtype);
            return ge::GRAPH_FAILED;
        }
        expandedXDtype = (QuantMode::GROUPQUANT_FP8_E5M2 == quantMode) ? ge::DT_FLOAT8_E5M2 : ge::DT_FLOAT8_E4M3FN;
        expandedScaleDtype = ge::DT_FLOAT;
    }
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_X, expandedXDtype);
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPERT_TOKEN_CUMSUM_OR_COUNT, ge::DT_INT64);
    context->SetOutputDataType(MOE_INIT_ROUTING_V3_OUTPUT_EXPANDED_SCALE, expandedScaleDtype);
    OPS_LOG_D(context, "End to do MoeInitRoutingV3InferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingGroupQuant)
    .InferShape(InferShape4MoeInitRoutingV3)
    .InferDataType(InferDataType4MoeInitRoutingV3);
} // namespace ops