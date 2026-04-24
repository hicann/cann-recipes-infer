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
 * \file dequant_swiglu_clamp_quant_tiling.cpp
 * \brief dequant_swiglu_quant 算子 ClampQuant tiling 实现文件
 *        合并 Group 模式和基础模式的 tiling 逻辑
 */
#include "register/op_def_registry.h"
#include "error/ops_error.h"
#include "tiling/tiling_api.h"
#include "dequant_swiglu_clamp_quant_tiling.h"

#define CHECK_FAIL(cont, cond, ...)                      \
    do {                                                 \
        if (cond) {                                      \
            OPS_LOG_E(cont->GetNodeName(), ##__VA_ARGS__); \
            return ge::GRAPH_FAILED;                     \
        }                                                \
    } while (0)

namespace optiling {

// ============================================================
// 构造函数
// ============================================================

DequantSwigluClampQuantTiling::DequantSwigluClampQuantTiling(gert::TilingContext* context)
    : context_(context),
      isGroupMode_(false),
      tilingKey_(0),
      workspaceSize_(WORKSPACE_SIZE),
      coreNum_(0),
      ubSize_(0),
      socVersion_(platform_ascendc::SocVersion::ASCEND910B),
      // Group 模式成员初始化
      groupNum_(0),
      hasGroupIndex_(false),
      speGroupType_(false),
      hasBias_(false),
      hasWeightScale_(false),
      hasActivationScale_(false),
      hasQuantScale_(false),
      hasQuantOffset_(false),
      quantMode_(0),
      actRight_(0),
      swigluMode_(0),
      clampLimit_(CLAMP_LIMIT_DEFAULT),
      gluAlpha_(GLU_ALPHA_DEFAULT),
      gluBias_(GLU_BIAS_DEFAULT),
      maxPreCore_(0),
      inDimx_(0),
      inDimy_(0),
      outDimy_(0),
      // 基础模式成员初始化
      inputDTypeLen_(2),
      totalCore_(0),
      totalAvailableCore_(0),
      totalUsedCoreNum_(0),
      baseRowLen_(0),
      baseColLen_(0),
      maxTileLen_(0),
      ubMinBlockLen_(0),
      cacheLineLen_(0),
      alignPackLen_(0),
      rowLen_(0),
      colLen_(0),
      optTotalTileNum_(0),
      optBaseSize_(0),
      optBaseTileNum_(0),
      quantScaleShapeSize_(0),
      xInputDataType_(ge::DT_FLOAT),
      biasDataType_(ge::DT_FLOAT),
      isPerfBranch_(false)
{
}

// ============================================================
// 主入口方法
// ============================================================

ge::graphStatus DequantSwigluClampQuantTiling::DoTiling()
{
    // 获取平台信息
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 根据是否有 group_index 判断模式
    if (IsGroupMode()) {
        return DoTilingGroupMode();
    } else {
        return DoTilingBaseMode();
    }
}

// ============================================================
// 模式判断方法
// ============================================================

bool DequantSwigluClampQuantTiling::IsGroupMode()
{
    auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
    isGroupMode_ = (shapeGroupIndex != nullptr);
    return isGroupMode_;
}

// ============================================================
// 公共方法：获取平台信息
// ============================================================

ge::graphStatus DequantSwigluClampQuantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<DequantSwigluClampQuantCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    totalCore_ = static_cast<uint32_t>(coreNum_);
    totalAvailableCore_ = totalCore_;
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Group 模式实现（从 dequant_swiglu_quant_tiling.cpp 迁移）
// ============================================================

bool DequantSwigluClampQuantTiling::CheckOptionalShapeExisting(const gert::StorageShape* storageShape)
{
    if (storageShape == nullptr) {
        return false;
    }
    int64_t shapeSize = storageShape->GetOriginShape().GetShapeSize();
    if (shapeSize <= 0) {
        return false;
    }
    return true;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckXAndGroupIndexDtype()
{
    auto xPtr = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    auto xDtype = xPtr->GetDataType();
    OP_CHECK_IF((SUPPORT_DTYPE.find(xDtype) == SUPPORT_DTYPE.end()),
                    OPS_LOG_E(context_->GetNodeName(), "x dtype only support int32 or bfloat16, please check."),
                    return ge::GRAPH_FAILED);
    tilingData_.set_groupIndexDtype(-1);
    if (hasGroupIndex_) {
        auto groupIndexPtr = context_->GetOptionalInputDesc(INPUT_GROUP_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexPtr);
        auto groupIndexDtype = groupIndexPtr->GetDataType();
        bool dtypeInValid = groupIndexDtype != ge::DT_INT64;
        OP_CHECK_IF(
            dtypeInValid,
            OPS_LOG_E(context_->GetNodeName(), "group_index dtype only support int64, please check!"),
            return ge::GRAPH_FAILED);
        tilingData_.set_groupIndexDtype(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckBias()
{
    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr) {
        hasBias_ = true;
        OP_CHECK_IF(CheckScaleShapeWithDim(BIAS_INDEX, inDimy_) != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "bias shape check failed."),
                  return ge::GRAPH_FAILED);
    } else {
        hasBias_ = false;
    }

    auto biasPtr = context_->GetOptionalInputDesc(BIAS_INDEX);
    if (biasPtr != nullptr && hasBias_ == true) {
        auto biasDtype = biasPtr->GetDataType();
        bool dtypeInValid = (biasDtype != ge::DT_INT32 && biasDtype != ge::DT_FLOAT &&
                            biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_BF16);
        OP_CHECK_IF(
          dtypeInValid,
          OPS_LOG_E(context_->GetNodeName(), "bias dtype only support bf16, fp16, float, int32, please check!"),
          return ge::GRAPH_FAILED);
        if (biasDtype == ge::DT_BF16) {
            tilingData_.set_biasDtype(BIAS_DTYPE_BF16);
        } else if (biasDtype == ge::DT_FLOAT16) {
            tilingData_.set_biasDtype(BIAS_DTYPE_FP16);
        } else if (biasDtype == ge::DT_FLOAT) {
            tilingData_.set_biasDtype(BIAS_DTYPE_FP32);
        } else if (biasDtype == ge::DT_INT32) {
            tilingData_.set_biasDtype(BIAS_DTYPE_INT32);
        }
    } else {
        tilingData_.set_biasDtype(0);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckWeightScale()
{
    auto weightScalePtr = context_->GetOptionalInputDesc(WEIGHT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightScalePtr);
    auto weightScaleDtype = weightScalePtr->GetDataType();
    bool dtypeInValid = weightScaleDtype != ge::DT_FLOAT;
    OP_CHECK_IF(dtypeInValid,
                    OPS_LOG_E(context_->GetNodeName(),
                                                    "weight_scale dtype only support float32, please check."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckScaleShapeWithDim(WEIGHT_SCALE_INDEX, inDimy_) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "weight scale shape check failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckActivationScale()
{
    auto activationScaleShapePtr = context_->GetOptionalInputShape(ACTIVATION_SCALE_INDEX);
    if (CheckOptionalShapeExisting(activationScaleShapePtr)) {
        auto activationScalePtr = context_->GetOptionalInputDesc(ACTIVATION_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, activationScalePtr);
        auto activationScaleDtype = activationScalePtr->GetDataType();
        bool dtypeInValid = activationScaleDtype != ge::DT_FLOAT;

        OP_CHECK_IF(dtypeInValid,
                        OPS_LOG_E(context_->GetNodeName(),
                                                        "activation_scale dtype only support float32, please check."),
                        return ge::GRAPH_FAILED);
        OP_CHECK_NULL_WITH_CONTEXT(context_, activationScaleShapePtr);
        auto activationScaleShape = activationScaleShapePtr->GetStorageShape();
        int64_t activationScaleNum = activationScaleShape.GetShapeSize();

        OP_CHECK_IF(
            activationScaleNum != inDimx_,
            OPS_LOG_E(
                context_->GetNodeName(),
                "activation_scale num(%ld) must be equal to the tokens num(%ld), please check.",
                activationScaleNum, inDimx_),
            return ge::GRAPH_FAILED);
        tilingData_.set_activationScaleIsEmpty(0);
    } else {
        tilingData_.set_activationScaleIsEmpty(1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckForDequant()
{
    // check weight scale, activation scale and bias
    auto xPtr = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    auto xDtype = xPtr->GetDataType();
    if (xDtype == ge::DT_INT32) {
        OP_CHECK_IF(CheckWeightScale() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "weight scale check failed."),
              return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckActivationScale() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "activation scale check failed."),
              return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckBias() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "bias check failed."),
              return ge::GRAPH_FAILED);
    }

    if (xDtype == ge::DT_BF16 && hasGroupIndex_) {
        auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
        const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
        OP_CHECK_IF(inputShapeGroupIndex.GetDimNum() != 1,
                        OPS_LOG_E(context_->GetNodeName(),
                                                        "groupIndex only support 1D Tensor now, please check."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckForDynamicQuant()
{
    auto offsetPtr = context_->GetOptionalInputShape(QUANT_OFFSET_INDEX);
    OP_CHECK_IF(offsetPtr != nullptr,
                OPS_LOG_E(context_->GetNodeName(),
                                                "quantOffSet only support None in dynamic quantization of group mode now, please check."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckScaleShapeWithDim(QUANT_SCALE_INDEX, outDimy_) != ge::GRAPH_SUCCESS,
            OPS_LOG_E(context_->GetNodeName(), "quant scale shape check failed."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckForStaticQuant()
{
    // check quantOffset dtype
    auto quantOffsetDescPtr = context_->GetOptionalInputDesc(QUANT_OFFSET_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantOffsetDescPtr);
    auto quantScaleDescPtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantScaleDescPtr);
    auto quantOffsetDtype = quantOffsetDescPtr->GetDataType();
    auto quantScaleDtype = quantScaleDescPtr->GetDataType();
    OP_CHECK_IF(quantOffsetDtype != quantScaleDtype,
                    OPS_LOG_E(context_->GetNodeName(),
                                                    "quantOffset dtype is different from quantScale, please check."),
                    return ge::GRAPH_FAILED);

    int64_t quantScaleColLen = 0;
    int64_t quantOffsetColLen = 0;
    OP_CHECK_IF(CheckStaticQuantShape(QUANT_SCALE_INDEX, quantScaleColLen) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "quant scale shape check failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckStaticQuantShape(QUANT_OFFSET_INDEX, quantOffsetColLen) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "quant offset shape check failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(quantScaleColLen != quantOffsetColLen,
            OPS_LOG_E(context_->GetNodeName(), "quant offset shape is different from quant scale."),
            return ge::GRAPH_FAILED);
    if (quantScaleColLen == 1) {
        tilingData_.set_quantIsOne(1);
    } else {
        tilingData_.set_quantIsOne(0);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckForQuant()
{
    // check and set quant scale dtype
    OP_CHECK_IF(CheckQuantScaleDtype() != ge::GRAPH_SUCCESS,
            OPS_LOG_E(context_->GetNodeName(), "Check QuantScale Dtype failed."),
            return ge::GRAPH_FAILED);

    // check quant offset and quant scale shape in dynamic scenario
    if (quantMode_ == QUANT_MODE_DYNAMIC) {
        OP_CHECK_IF(CheckForDynamicQuant() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "Check For Dynamic Quant failed."),
              return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(CheckForStaticQuant() != ge::GRAPH_SUCCESS,
              OPS_LOG_E(context_->GetNodeName(), "Check For Static Quant failed."),
              return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckQuantScaleDtype()
{
    bool dtypeInValid = false;

    auto quantScaleShapePtr = context_->GetOptionalInputShape(QUANT_SCALE_INDEX);
    if (quantScaleShapePtr == nullptr) {
        tilingData_.set_quantScaleDtype(0);
        tilingData_.set_needSmoothScale(0);
    } else {
        auto quantScalePtr = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, quantScalePtr);
        tilingData_.set_needSmoothScale(1);
        auto quantScaleDtype = quantScalePtr->GetDataType();
        dtypeInValid =
            quantScaleDtype != ge::DT_FLOAT && quantScaleDtype != ge::DT_FLOAT16 && quantScaleDtype != ge::DT_BF16;
        OP_CHECK_IF(
            dtypeInValid,
            OPS_LOG_E(context_->GetNodeName(),
                                            "quant_scale dtype only support float32 or float16 or bfloat16, please check."),
            return ge::GRAPH_FAILED);
        tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_BF16);
        if (quantScaleDtype == ge::DT_FLOAT) {
            tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_FP32);
        } else if (quantScaleDtype == ge::DT_FLOAT16) {
            tilingData_.set_quantScaleDtype(QUANT_SCALE_DTYPE_FP16);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::GetAttrGroupMode()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
    actRight_ = (attrActivateLeft == nullptr || *attrActivateLeft == false) ? 1 : 0;
    std::string quantMode = attrs->GetAttrPointer<char>(ATTR_QUANT_MODE_INDEX);
    auto it = SUPPORT_QUANT_MODE.find(quantMode);
    OP_CHECK_IF(it == SUPPORT_QUANT_MODE.end(),
                    OPS_LOG_E(context_->GetNodeName(),
                                                    "attr quant_mode only support dynamic(1) and static(0) currently, please check."),
                    return ge::GRAPH_FAILED);
    quantMode_ = it->second;

    auto* swigluMode = attrs->GetAttrPointer<int>(SWIGLU_MODE_INDEX);
    auto* clampLimit = attrs->GetAttrPointer<float>(CLAMP_LIMIT_INDEX);
    auto* gluAlpha = attrs->GetAttrPointer<float>(GLU_ALPHA_INDEX);
    auto* gluBias = attrs->GetAttrPointer<float>(GLU_BIAS_INDEX);

    swigluMode_ = swigluMode == nullptr ? 0 : *swigluMode;
    clampLimit_ = clampLimit == nullptr ? CLAMP_LIMIT_DEFAULT : *clampLimit;
    gluAlpha_ = gluAlpha == nullptr ? GLU_ALPHA_DEFAULT : *gluAlpha;
    gluBias_ = gluBias == nullptr ? GLU_BIAS_DEFAULT : *gluBias;

    OP_CHECK_IF(swigluMode_ != 0 && swigluMode_ != 1,
                    OPS_LOG_E(context_->GetNodeName(), "swigluMode only support 1 or 0, value is %ld, please check!", swigluMode_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckScaleShapeWithDim(const int64_t scaleInputIdx,
                                                                    const int64_t expectDim)
{
    auto scalePtr = context_->GetOptionalInputShape(scaleInputIdx);
    if (scalePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleShape = scalePtr->GetStorageShape();
    OP_CHECK_IF(scaleShape.GetDimNum() < 1,
                    OPS_LOG_E(context_->GetNodeName(), "Scale shape len must > 0"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleShape.GetDim(scaleShape.GetDimNum() - 1) != expectDim,
                    OPS_LOG_E(context_->GetNodeName(), "ScaleShape[-1] must be %ld, but is %ld",
                                                    expectDim, scaleShape.GetDim(scaleShape.GetDimNum() - 1)),
                    return ge::GRAPH_FAILED);
    if (groupNum_ > 1) {
        // check with group index
        OP_CHECK_IF(
            scaleShape.GetDimNum() != DIM_SIZE_2,
            OPS_LOG_E(context_->GetNodeName(),
                                            "Scaleshape len must be 2d(like [groupNum, %ld] in groupmode"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            scaleShape.GetDim(0) != groupNum_,
            OPS_LOG_E(context_->GetNodeName(), "ScaleShape[0] be groupNum[%ld], but is [%ld]",
                                            groupNum_, scaleShape.GetDim(0)),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            scaleShape.GetDimNum() > DIM_SIZE_2,
            OPS_LOG_E(context_->GetNodeName(), "ScaleShape Len must <= 2"),
            return ge::GRAPH_FAILED);
        int64_t groupNumFromScale = scaleShape.GetDimNum() <= 1 ? 1 : scaleShape.GetDim(0);
        OP_CHECK_IF(
            groupNumFromScale != 1,
            OPS_LOG_E(context_->GetNodeName(), "ScaleShape must be [1, 2H] or [2H]"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckStaticQuantShape(const int64_t quantInputIdx, int64_t& colLen)
{
    // check quant scale and quant offset shape
    auto quantPtr = context_->GetOptionalInputShape(quantInputIdx);
    if (quantPtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto quantShape = quantPtr->GetStorageShape();
    OP_CHECK_IF(quantShape.GetDimNum() < 1,
                    OPS_LOG_E(context_->GetNodeName(), "Quant shape len must > 0"),
                    return ge::GRAPH_FAILED);
    colLen = quantShape.GetDim(quantShape.GetDimNum() - 1);
    if (quantShape.GetDimNum() == 1) {
        OP_CHECK_IF(colLen != groupNum_,
              OPS_LOG_E(context_->GetNodeName(), "QuantShape must be [%ld, ] or [%ld, %ld]",
                                              groupNum_, groupNum_, outDimy_), return ge::GRAPH_FAILED);
        colLen = 1;
    } else {
        OP_CHECK_IF(colLen != outDimy_ || quantShape.GetDim(0) != groupNum_,
            OPS_LOG_E(context_->GetNodeName(), "QuantShape must be [%ld, ] or [%ld, %ld]",
                                            groupNum_, groupNum_, outDimy_), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::CheckIllegalParam()
{
    // if hasbias, speGroupType_ must be false
    if (hasBias_) {
        OP_CHECK_IF(speGroupType_ == true,
                        OPS_LOG_E(context_->GetNodeName(), "speGroupType_ only support false when using bias"),
                        return ge::GRAPH_FAILED);
    }

    // if swigluMode is 1, speGroupType_ must be false
    if (swigluMode_) {
        OP_CHECK_IF(speGroupType_ == true,
                    OPS_LOG_E(context_->GetNodeName(), "speGroupType_ only support false when swiglu mode is 1"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool DequantSwigluClampQuantTiling::IsPerformanceAndGroupIndexBrach()
{
    auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
    if (shapeGroupIndex != nullptr) {
        return true;
    }
    return false;
}

ge::graphStatus DequantSwigluClampQuantTiling::GetShapeAttrsInfoGroupModeInner()
{
    if (!IsPerformanceAndGroupIndexBrach()) {
        return ge::GRAPH_SUCCESS;
    }
    // get 2H from x, get H from y, check if 2H can be divided by 64
    auto shapeX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& inputShapeX = shapeX->GetStorageShape();
    int64_t inputShapeXTotalNum = inputShapeX.GetShapeSize();
    int64_t inputShapeXRank = inputShapeX.GetDimNum();
    inDimy_ = inputShapeX.GetDim(inputShapeXRank - 1);
    inDimx_ = inputShapeXTotalNum / inDimy_;
    auto shapeY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeY);
    const gert::Shape& outputShapeY = shapeY->GetStorageShape();
    outDimy_ = outputShapeY.GetDim(inputShapeXRank - 1);
    OP_CHECK_IF(inDimy_ % (BLOCK_SIZE * SWI_FACTOR) != 0,
                    OPS_LOG_E(context_->GetNodeName(),
                                                    "only support lastdimSize being divided by 64, but is %ld", inDimy_),
                    return ge::GRAPH_FAILED);

    // set the relevant param of group, hasGroupIndex_, groupNum_ and speGroupType_
    auto shapeGroupIndex = context_->GetOptionalInputShape(INPUT_GROUP_INDEX);
    hasGroupIndex_ = shapeGroupIndex != nullptr;
    groupNum_ = 0;
    speGroupType_ = false;
    if (hasGroupIndex_) {
        const gert::Shape& inputShapeGroupIndex = shapeGroupIndex->GetStorageShape();
        groupNum_ = inputShapeGroupIndex.GetDimNum() == 0 ? 1 : inputShapeGroupIndex.GetDim(0);
        speGroupType_ = inputShapeGroupIndex.GetDimNum() == DIM_SIZE_2;
    }

    OP_CHECK_IF(CheckXAndGroupIndexDtype() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttrGroupMode() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckForDequant() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "check for dequant failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckForQuant() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "check for quant failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckIllegalParam() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "check illegal param failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void DequantSwigluClampQuantTiling::CountTilingKey()
{
    auto xPtr = context_->GetInputDesc(X_INDEX);
    auto xDtype = xPtr->GetDataType();
    tilingKey_ = hasGroupIndex_ ? TILING_KEY_HAS_GROUP : TILING_KEY_NO_GROUP;
    // add quant scale offset to tilingKey_
    tilingKey_ += TILING_KEY_QS_DTYPE * tilingData_.get_quantScaleDtype();
    // add bias offset to tilingKey_
    tilingKey_ += TILING_KEY_BIAS_DTYPE * tilingData_.get_biasDtype();
    // tiling based on groupnum, pre cut num by coreNum_ and total tokens
    bool cond1 = speGroupType_ &&
                 (groupNum_ >= CUT_GROUP_LARGE_THAN_64) &&
                 (inDimx_ / groupNum_ <= EACH_GROUP_TOKEN_LESS_THAN);
    bool cond2 = !speGroupType_ &&
                 (groupNum_ >= CUT_GROUP_LARGE_THAN_32) &&
                 (inDimx_ / groupNum_ <= EACH_GROUP_TOKEN_LESS_THAN) &&
                 !tilingData_.get_biasDtype() && !tilingData_.get_quantScaleDtype() &&
                 (xDtype == ge::DT_INT32);
    if (cond1 || cond2) {
        tilingKey_ += TILING_KEY_CUT_GROUP;
        maxPreCore_ = std::min(static_cast<int64_t>(coreNum_), static_cast<int64_t>(inDimx_));
    }
}

ge::graphStatus DequantSwigluClampQuantTiling::CountMaxDim(int64_t& ubFactorDimx)
{
    int64_t db = 2;
    int64_t maxOutDimy = 0;
    int64_t biasBufferY = hasBias_ == false ? 0 : static_cast<int64_t>(SWI_FACTOR * sizeof(float));
    int64_t biasBufferX = hasBias_ == false ? 0 : outDimy_ * SWI_FACTOR * static_cast<int64_t>(sizeof(float));

    int64_t SweiGLUBufferY = swigluMode_ == 0 ? 0 : static_cast<int64_t>(sizeof(int8_t) + sizeof(int32_t));
    int64_t SweiGLUBufferX = swigluMode_ == 0 ? 0 : outDimy_ * static_cast<int64_t>(sizeof(int8_t)) + outDimy_ * static_cast<int64_t>(sizeof(int32_t));

    int64_t quantOffsetSpace = quantMode_ == QUANT_MODE_DYNAMIC ? 0 : static_cast<int64_t>(sizeof(float));

    // UbFactorDimx is 1, compute maxOutDimy
    int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - BLOCK_SIZE - db * BLOCK_SIZE - static_cast<int64_t>(sizeof(float));
    int64_t denominator =
        5 * static_cast<int64_t>(sizeof(float)) + db * SWI_FACTOR * static_cast<int64_t>(sizeof(float)) + static_cast<int64_t>(sizeof(int8_t)) + biasBufferY + SweiGLUBufferY + quantOffsetSpace;
    maxOutDimy = static_cast<int64_t>(numerator / denominator);
    maxOutDimy = maxOutDimy / BLOCK_SIZE * BLOCK_SIZE;
    int64_t maxInDimy = static_cast<int64_t>(maxOutDimy * SWI_FACTOR);
    OPS_LOG_I(context_->GetNodeName(), "Get maxInDimy[%ld]", maxInDimy);
    OP_CHECK_IF(inDimy_ > maxInDimy,
                    OPS_LOG_E(context_->GetNodeName(),
                                                    "only support lastdimSize <= %ld, but is %ld", maxInDimy, inDimy_),
                    return ge::GRAPH_FAILED);

    // compute ubFactorDimx
    quantOffsetSpace = quantMode_ == QUANT_MODE_DYNAMIC ? 0 : outDimy_ * sizeof(float);
    numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - outDimy_ * static_cast<int64_t>(sizeof(float)) - BLOCK_SIZE - SWI_FACTOR * outDimy_ * static_cast<int64_t>(sizeof(float)) - biasBufferX - quantOffsetSpace;

    denominator = db * (outDimy_ * SWI_FACTOR + BLOCK_ELEM) * static_cast<int64_t>(sizeof(float)) + outDimy_ * static_cast<int64_t>(sizeof(int8_t)) + static_cast<int64_t>(sizeof(float)) +
                  outDimy_ * SWI_FACTOR * static_cast<int64_t>(sizeof(float)) + SweiGLUBufferX;
    ubFactorDimx = static_cast<int64_t>(numerator / denominator);
    ubFactorDimx = std::min(ubFactorDimx, inDimx_);
    OPS_LOG_I(context_->GetNodeName(), "Get ubFactorDimx[%ld]", ubFactorDimx);

    // special ub cut for 2048 4096
    if (swigluMode_ == 0 && hasBias_ == false) {
        ubFactorDimx =
        (inDimy_ == PERFORMANCE_H_2048 || inDimy_ == PERFORMANCE_H_4096) ? PERFORMANCE_UB_FACTOR / inDimy_ : ubFactorDimx;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::DoOpTilingGroupMode()
{
    if (GetShapeAttrsInfoGroupModeInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    auto inputShapeX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShapeX);

    int64_t ubFactorDimx = 0;
    OP_CHECK_IF(CountMaxDim(ubFactorDimx) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "Count MaxDim failed."),
                return ge::GRAPH_FAILED);

    maxPreCore_ = (inDimx_ + ubFactorDimx - 1) / ubFactorDimx;
    maxPreCore_ = std::min(maxPreCore_, static_cast<int64_t>(PERFORMANCE_CORE_NUM));
    maxPreCore_ = std::min(maxPreCore_, static_cast<int64_t>(coreNum_));

    CountTilingKey();

    tilingData_.set_inDimx(inDimx_);
    tilingData_.set_inDimy(inDimy_);
    tilingData_.set_outDimy(outDimy_);
    tilingData_.set_UbFactorDimx(ubFactorDimx);
    tilingData_.set_UbFactorDimy(outDimy_);
    tilingData_.set_usedCoreNum(maxPreCore_);
    tilingData_.set_maxCoreNum(maxPreCore_);
    tilingData_.set_inGroupNum(groupNum_);
    tilingData_.set_quantMode(quantMode_);
    tilingData_.set_actRight(actRight_);
    tilingData_.set_speGroupType(static_cast<int64_t>(speGroupType_));
    tilingData_.set_hasBias(hasBias_);

    tilingData_.set_swigluMode(swigluMode_);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_gluAlpha(gluAlpha_);
    tilingData_.set_gluBias(gluBias_);
    return ge::GRAPH_SUCCESS;
}

uint64_t DequantSwigluClampQuantTiling::GetTilingKeyGroupMode() const
{
    return tilingKey_;
}

ge::graphStatus DequantSwigluClampQuantTiling::GetWorkspaceSizeGroupMode()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::PostTilingGroupMode()
{
    context_->SetTilingKey(GetTilingKeyGroupMode());
    context_->SetBlockDim(maxPreCore_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::DoTilingGroupMode()
{
    if (DoOpTilingGroupMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSizeGroupMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (PostTilingGroupMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

void DequantSwigluClampQuantTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "inDimx_: " << tilingData_.get_inDimx();
    info << ", inDimy_: " << tilingData_.get_inDimy();
    info << ", outDimy: " << tilingData_.get_outDimy();
    info << ", UbFactorDimx: " << tilingData_.get_UbFactorDimx();
    info << ", UbFactorDimy: " << tilingData_.get_UbFactorDimy();
    info << ", usedCoreNum: " << tilingData_.get_usedCoreNum();
    info << ", maxCoreNum: " << tilingData_.get_maxCoreNum();
    info << ", inGroupNum: " << tilingData_.get_inGroupNum();
    info << ", quantMode: " << tilingData_.get_quantMode();
    info << ", actRight: " << tilingData_.get_actRight();
    info << ", tilingKey: " << tilingKey_;
    info << ", hasBias: " << hasBias_;
    info << ", swigluMode: " << tilingData_.get_swigluMode();
    info << ", clampLimit: " << tilingData_.get_clampLimit();
    info << ", gluAlpha: " << tilingData_.get_gluAlpha();
    info << ", gluBias: " << tilingData_.get_gluBias();

    OPS_LOG_I(context_->GetNodeName(), "%s", info.str().c_str());
}

// ============================================================
// 基础模式实现（从 dequant_swiglu_quant_tiling_base.cpp 迁移）
// ============================================================

ge::graphStatus DequantSwigluClampQuantTiling::SetTotalShape(gert::TilingContext* cont, const gert::Shape& inShape)
{
    int64_t shapeBefore = 1;
    int64_t shapeAfter = 1;
    int64_t dimNum = inShape.GetDimNum();
    CHECK_FAIL(cont, dimNum <= 1, "The shape dim of x can not be less than 2");

    int64_t splitDim = dimNum - 1; // inDim default -1
    for (int64_t i = 0; i < splitDim; i++) {
        shapeBefore *= inShape.GetDim(i);
    }
    shapeAfter = inShape.GetDim(splitDim);
    // 如果shape不是2的倍数,返回

    CHECK_FAIL(cont, shapeAfter % 2 != 0, "The shape dim of x dim must be even number");

    rowLen_ = static_cast<uint64_t>(shapeBefore);
    // colLen为原shape除以2
    colLen_ = static_cast<uint64_t>(shapeAfter / 2);

    tilingData_.set_rowLen(rowLen_);
    tilingData_.set_colLen(colLen_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::checkWeightBiasActivate(gert::TilingContext* context)
{
    auto biasShapeShapePtr = context->GetOptionalInputShape(3);
    if (biasShapeShapePtr != nullptr) {
        auto biasInputDesc = context->GetOptionalInputDesc(3);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasInputDesc);
        biasDataType_ = biasInputDesc->GetDataType();

        bool checkBiasRes = biasDataType_ != ge::DT_INT32 && biasDataType_ != ge::DT_FLOAT &&
                            biasDataType_ != ge::DT_FLOAT16 && biasDataType_ != ge::DT_BF16;
        CHECK_FAIL(context, checkBiasRes, "DequantSwilguQuant check bias dtype failed");

        uint64_t biasShapeSize = biasShapeShapePtr->GetStorageShape().GetShapeSize();
        CHECK_FAIL(
            context, biasShapeSize != colLen_ * 2,
            "The shape of the bias is not equal to the last dimension of the xshape.");
    }
    tilingData_.set_biasIsEmpty(biasShapeShapePtr == nullptr ? 1 : 0);
    // int32时 weight_scale为必选项
    auto weightScaleShapePtr = context->GetOptionalInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShapePtr);

    auto weightScaleInputDesc = context->GetOptionalInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleInputDesc);
    ge::DataType weightScaleDataType = weightScaleInputDesc->GetDataType();
    CHECK_FAIL(context, weightScaleDataType != ge::DT_FLOAT, "The dtype of weightscale must be float.");

    uint64_t weightScaleShapeSize = weightScaleShapePtr->GetStorageShape().GetShapeSize();
    CHECK_FAIL(
        context, weightScaleShapeSize != colLen_ * 2,
        "The shape of the weight scale is not equal to the last dimension of the xshape.");

    // int32时 activate_scale为可选项
    auto activateScaleShapePtr = context->GetOptionalInputShape(2);
    if (activateScaleShapePtr != nullptr) {
        auto activateScaleInputDesc = context->GetOptionalInputDesc(2);
        OP_CHECK_NULL_WITH_CONTEXT(context, activateScaleInputDesc);
        ge::DataType activateScaleDataType = activateScaleInputDesc->GetDataType();
        CHECK_FAIL(context, activateScaleDataType != ge::DT_FLOAT, "The dtype of activateScale must be float");

        uint64_t activateScaleShapeSize = activateScaleShapePtr->GetStorageShape().GetShapeSize();
        CHECK_FAIL(
            context, activateScaleShapeSize != rowLen_,
            "The shape of the activat scale is not equal to xshape divided by the total number of the last "
            "dimensions.");
    }
    tilingData_.set_activateScaleIsEmpty(activateScaleShapePtr == nullptr ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::checkInputShape(gert::TilingContext* context, ge::DataType xDataType)
{
    if (xDataType == ge::DT_INT32) {
        if (checkWeightBiasActivate(context) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // quant_scale
    auto quantScaleShapePtr = context->GetOptionalInputShape(4); // 3: bias idx
    if (quantScaleShapePtr == nullptr) {
        tilingData_.set_quantScaleIsEmpty(1);
        return ge::GRAPH_SUCCESS;
    }
    auto quantScaleInputDesc = context->GetOptionalInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantScaleInputDesc);
    ge::DataType quantScaleDataType = quantScaleInputDesc->GetDataType();
    CHECK_FAIL(
        context, quantScaleDataType != ge::DT_FLOAT, "The dType of quantscale must be float. type: %d",
        quantScaleDataType);
    quantScaleShapeSize_ = quantScaleShapePtr->GetStorageShape().GetShapeSize();
    bool checkQuantScaleSize = (quantScaleShapeSize_ != colLen_) && (quantScaleShapeSize_ != 1);
    CHECK_FAIL(
        context, checkQuantScaleSize,
        "The shape of the quant scale is not equal to the last dimension of the xshape.");
    if (quantMode_ == 0) {
        auto quantOffsetShapePtr = context->GetOptionalInputShape(5);
        auto quantOffsetInputDesc = context->GetOptionalInputDesc(5);
        OP_CHECK_NULL_WITH_CONTEXT(context, quantOffsetInputDesc);
        ge::DataType quantOffsetDataType = quantOffsetInputDesc->GetDataType();
        CHECK_FAIL(context, quantOffsetDataType != ge::DT_FLOAT, "quant offset must be float.");
        uint64_t quantOffsetShapeSize = quantOffsetShapePtr->GetStorageShape().GetShapeSize();
        bool checkQuantOffsetSize = (quantOffsetShapeSize != colLen_) && (quantOffsetShapeSize != 1);
        CHECK_FAIL(
            context, checkQuantOffsetSize,
            "The shape of the quant offset is not equal to the last dimension of the xshape.");
    }
    return ge::GRAPH_SUCCESS;
}

bool DequantSwigluClampQuantTiling::SetAttr(const gert::RuntimeAttrs* attrs)
{
    auto isActivateLeftAttr = *(attrs->GetBool(0));
    auto str = attrs->GetStr(1);
    std::string quantModeAttr{str};
    std::transform(quantModeAttr.begin(), quantModeAttr.end(), quantModeAttr.begin(), ::tolower);

    if ((quantModeAttr != "static") && (quantModeAttr != "dynamic")) {
        OPS_LOG_E(
            "CalcTiling",
            "dequant_swiglu_quant quant_mode "
            "should be static or dynamic with case insensitive, current: %s",
            quantModeAttr.c_str());
        return false;
    }
    uint32_t activateLeft = (isActivateLeftAttr ? 1 : 0);
    quantMode_ = ((quantModeAttr == "static") ? 0 : 1);
    tilingData_.set_activateLeft(activateLeft);
    return true;
}

bool DequantSwigluClampQuantTiling::GetBufferNumAndDataLenPerUB(uint64_t ubSize, int32_t dtype, uint64_t& dataLenPerUB)
{
    uint32_t singleDataSize = 1;
    if (quantMode_ == 1) {
        if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
            singleDataSize = DYNAMIC_BF16_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                             static_cast<uint32_t>(sizeof(int8_t));
        } else if (dtype == ge::DT_INT32) {
            if ((biasDataType_ == ge::DT_INT32 || biasDataType_ == ge::DT_FLOAT)) {
                singleDataSize = DYNAMIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 static_cast<uint32_t>(sizeof(int8_t));
            } else {
                singleDataSize = DYNAMIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 DYNAMIC_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(int16_t)) +
                                 static_cast<uint32_t>(sizeof(int8_t));
            }
        }
    }
    if (quantMode_ == 0) {
        if (dtype == ge::DT_INT32) {
            if ((biasDataType_ == ge::DT_INT32 || biasDataType_ == ge::DT_FLOAT)) {
                singleDataSize = STATIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 static_cast<uint32_t>(sizeof(int8_t)); /* 11 -> float 块数量 */
            } else {
                singleDataSize = STATIC_BF16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                                 DYNAMIC_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(int16_t)) +
                                 static_cast<uint32_t>(sizeof(int8_t)); /* 11 -> float 块数量 */
            }
        } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
            singleDataSize = STATIC_BF16_INT16_TBUF_NUM_HALF * static_cast<uint32_t>(sizeof(float)) +
                             static_cast<uint32_t>(sizeof(int8_t));
        }
    }
    dataLenPerUB = ubSize / singleDataSize;
    return true;
}

bool DequantSwigluClampQuantTiling::CalcUbMaxTileLen(uint64_t ubSize, int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // get buffernum and maxTileLen
    uint64_t maxTileLenPerUB = 1;
    if (!GetBufferNumAndDataLenPerUB(ubSize, dtype, maxTileLenPerUB)) {
        OPS_LOG_E("DequantSwigluClampQuant", "CalcTiling Get maxTileLenPerUB %lu failed", maxTileLenPerUB);
        return false;
    }
    optTiling.maxTileLen = AlignDown<uint64_t>(maxTileLenPerUB, ALIGN_UINT_IN_CACHE_32B); // 32个元素对齐
    OPS_LOG_I("DequantSwigluClampQuant", "CalcTiling ubSize:%lu, maxTileLenPerUB:%u", ubSize, optTiling.maxTileLen);
    return true;
}

uint32_t DequantSwigluClampQuantTiling::getBaseColLenUpBound(GluSingleTilingOptParam& optTiling)
{
    uint32_t upBound = std::min(static_cast<uint32_t>(colLen_), static_cast<uint32_t>(optTiling.maxTileLen));
    if (tilingData_.get_is32BAligned() == 1) {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN));
    } else {
        upBound = std::min(upBound, static_cast<uint32_t>(DISCONTINE_COPY_MAX_BLOCKLEN / inputDTypeLen_));
    }

    if (upBound < colLen_ && upBound > cacheLineLen_) {
        // 该种场景，每一个colLen至少被切割成2块，需要保证baseColLen为512B整数倍才高效
        return AlignDown<uint32_t>(upBound, cacheLineLen_);
    } else {
        return upBound;
    }
}

void DequantSwigluClampQuantTiling::SaveOptBaseShape(
    uint32_t baseRowLen_, uint32_t baseColLen_, GluSingleTilingOptParam& optTiling)
{
    uint64_t totalTileNum =
        std::min(static_cast<uint64_t>(rowLen_), static_cast<uint64_t>(totalAvailableCore_));
    uint64_t baseSize = static_cast<uint64_t>(baseRowLen_ * baseColLen_);
    if (static_cast<int32_t>(baseRowLen_) == 0 || static_cast<int32_t>(baseColLen_) == 0) {
        OPS_LOG_I("SaveOptBaseShape", "baseRowLen_:%u or baseColLen:%u is zero.", baseRowLen_, baseColLen_);
        return;
    }
    uint64_t baseTileNum = (baseRowLen_ == 0 ? 0 : (rowLen_ / baseRowLen_)) *
                           (baseColLen_ == 0 ? 0 : (colLen_ / baseColLen_));
    totalUsedCoreNum_ = std::min(static_cast<uint32_t>(totalTileNum), static_cast<uint32_t>(totalAvailableCore_));
    if (colLen_ < PERFORMANCE_COL_LEN && rowLen_ < PERFORMANCE_ROW_LEN) {
        totalUsedCoreNum_ = std::min(totalUsedCoreNum_, static_cast<uint32_t>(MIN_CORE));
    }
    optTiling.optBaseRowLen = baseRowLen_;
    optTiling.optBaseColLen = baseColLen_;
    optTiling.optTotalTileNum = static_cast<uint32_t>(totalTileNum);
    optTiling.optBaseSize = baseSize;
    optTiling.optBaseTileNum = static_cast<uint32_t>(baseTileNum);
    optTiling.totalUsedCoreNum = totalUsedCoreNum_;
    optTiling.tileNumPerCore = DivCeil<uint64_t>(totalTileNum, totalUsedCoreNum_);
}

bool DequantSwigluClampQuantTiling::CalcOptBaseShape(GluSingleTilingOptParam& optTiling, int32_t dtype)
{
    uint32_t baseColLen_ = getBaseColLenUpBound(optTiling);
    uint32_t baseRowlen_ = 1;
    if ((quantMode_ == 1) && (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16)) {
        baseRowlen_ = std::min(
            optTiling.maxTileLen / AlignUp<uint32_t>(baseColLen_, ALIGN_UINT_IN_CACHE_32B),
            static_cast<uint32_t>(rowLen_));
        baseRowlen_ = std::min(DivCeil<uint32_t>(static_cast<uint32_t>(rowLen_), totalAvailableCore_), baseRowlen_);
    }
    SaveOptBaseShape(baseRowlen_, baseColLen_, optTiling);
    return true;
}

bool DequantSwigluClampQuantTiling::CalcOptTiling(
    const uint64_t ubSize, const int32_t dtype, GluSingleTilingOptParam& optTiling)
{
    // 计算maxTilingLen
    if (!CalcUbMaxTileLen(ubSize, dtype, optTiling)) {
        return false;
    }
    // 计算最优的base块形状
    if (!CalcOptBaseShape(optTiling, dtype)) {
        return false;
    }
    return true;
}

bool DequantSwigluClampQuantTiling::CalcTiling(
    const uint32_t totalCores, const uint64_t ubSize, const platform_ascendc::SocVersion socVersion_)
{
    totalAvailableCore_ = totalCores;
    if (!GetLengthByType(xInputDataType_, inputDTypeLen_)) {
        OPS_LOG_I("DequantSwigluClampQuant", "CalcTiling Unsupported input data type %d", xInputDataType_);
        return false;
    }
    ubMinBlockLen_ = ALIGN_UINT_IN_CACHE_32B / inputDTypeLen_; // min block size
    cacheLineLen_ = PACK_UINT_IN_CACHE_512B / inputDTypeLen_;  // bandwidth max efficiency
    alignPackLen_ = cacheLineLen_;                             // 默认512对齐，策略可调整
    OPS_LOG_I(
        "DequantSwigluClampQuant", "CalcTiling GetLengthByType:%u ubMinBlockLen:%u cacheLineLen:%u alignPackLen:%u",
        inputDTypeLen_, ubMinBlockLen_, cacheLineLen_, alignPackLen_);
    // Is 32-byte aligned for split colLen?
    tilingData_.set_is32BAligned(colLen_ % ubMinBlockLen_ == 0 ? 1 : 0);
    // 310p not support Non-64B
    const uint32_t blockSizeOf64B = ALIGN_UINT_IN_CACHE_64B / inputDTypeLen_;
    if (((socVersion_ == platform_ascendc::SocVersion::ASCEND310P)) &&
        (colLen_ % blockSizeOf64B != 0)) {
        OPS_LOG_E("DequantSwigluClampQuant", "input shape is not support Non-64B aligned");
        return false;
    }
    GluSingleTilingOptParam optTilingDb;
    if (!CalcOptTiling(ubSize, xInputDataType_, optTilingDb)) {
        return false;
    }
    const GluSingleTilingOptParam* const optTiling = &optTilingDb;
    // 记录最优的结果
    baseRowLen_ = optTiling->optBaseRowLen;
    baseColLen_ = optTiling->optBaseColLen;
    tilingData_.set_baseRowLen(baseRowLen_);
    tilingData_.set_baseColLen(baseColLen_);
    totalUsedCoreNum_ = optTiling->totalUsedCoreNum;
    tilingData_.set_usedCoreNum(totalUsedCoreNum_);
    OPS_LOG_I(
        "DequantSwigluClampQuant", "CalcTilingRES baseRowLen:%u baseColLen:%u", optTiling->optBaseRowLen,
        optTiling->optBaseColLen);
    return true;
}

ge::graphStatus DequantSwigluClampQuantTiling::GetShapeAttrsInfoBaseModeInner()
{
    // 获取输入shape
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const gert::Shape xShape = xShapePtr->GetStorageShape();
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    xInputDataType_ = inputDesc->GetDataType();
    if (SetTotalShape(context_, xShape) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入属性
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    if (!SetAttr(attrs)) {
        return ge::GRAPH_FAILED;
    }

    if (checkInputShape(context_, xInputDataType_) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    const gert::Shape yShape = yShapePtr->GetStorageShape();

    int32_t dimNum = xShape.GetDimNum();
    CHECK_FAIL(context_, xShape.GetDimNum() != yShape.GetDimNum(), "The shape of y must be equal to The shape of x.");
    CHECK_FAIL(
        context_, xShape.GetDim(dimNum - 1) != yShape.GetDim(dimNum - 1) * 2,
        "The last dimension of y must be half of x_shape last dim.");

    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    const gert::Shape scaleShape = scaleShapePtr->GetStorageShape();

    CHECK_FAIL(
        context_, static_cast<uint64_t>(scaleShape.GetShapeSize()) != rowLen_,
        "scale shape must be row length.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::DoOpTilingBaseMode()
{
    if (GetShapeAttrsInfoBaseModeInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (!CalcTiling(totalCore_, ubSize_, socVersion_)) {
        return ge::GRAPH_FAILED;
    }
    isPerfBranch_ = isPerformanceBranch();
    return ge::GRAPH_SUCCESS;
}

int64_t DequantSwigluClampQuantTiling::getTilingKeyStatic(
    const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const
{
    if (inputDtype != ge::DT_INT32) {
        if (scaleSize == 1) {
            if (inputDtype == ge::DT_FLOAT16) {
                return STATIC_FLOAT16_X;
            } else {
                return STATIC_BFLOAT16_X;
            }
        } else {
            if (inputDtype == ge::DT_FLOAT16) {
                return STATIC_FLOAT16_XD;
            } else {
                return STATIC_BFLOAT16_XD;
            }
        }
    }
    if (scaleSize == 1) {
        if (biasType == ge::DT_INT32) {
            return STATIC_INT_X_INT_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT) {
            return STATIC_INT_X_FLOAT32_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT16) {
            return STATIC_INT_X_FLOAT16_BIAS_QUANT_ONE;
        } else {
            return STATIC_INT_X_BFLOAT16_BIAS_QUANT_ONE;
        }
    } else {
        if (biasType == ge::DT_INT32) {
            return STATIC_INT_X_INT_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT) {
            return STATIC_INT_X_FLOAT32_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT16) {
            return STATIC_INT_X_FLOAT16_BIAS_QUANT_D;
        } else {
            return STATIC_INT_X_BFLOAT16_BIAS_QUANT_D;
        }
    }
}

int64_t DequantSwigluClampQuantTiling::getTilingKeyDynamic(
    const int32_t inputDtype, const ge::DataType biasType, const int64_t scaleSize) const
{
    if (inputDtype != ge::DT_INT32) {
        if (inputDtype == ge::DT_FLOAT16) {
            if (scaleSize == 1) {
                return DYNAMIC_FLOAT16_X;
            } else {
                return DYNAMIC_FLOAT16_XD;
            }
        } else {
            if (scaleSize == 1) {
                return DYNAMIC_BFLOAT16_X;
            } else {
                return DYNAMIC_BFLOAT16_XD;
            }
        }
    }
    if (scaleSize == 1) {
        if (biasType == ge::DT_INT32) {
            return DYNAMIC_INT_X_INT_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT) {
            return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_ONE;
        } else if (biasType == ge::DT_FLOAT16) {
            return DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_ONE;
        } else {
            return DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_ONE;
        }
    } else {
        if (biasType == ge::DT_INT32) {
            return DYNAMIC_INT_X_INT_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT) {
            if (isPerfBranch_) {
                return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D_PERFORMANCE;
            }
            return DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D;
        } else if (biasType == ge::DT_FLOAT16) {
            return DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_D;
        } else {
            return DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_D;
        }
    }
}

bool DequantSwigluClampQuantTiling::isPerformanceBranch()
{
    if (tilingData_.get_is32BAligned() == 1 &&
        colLen_ <= PERFORMANCE_COL_LEN &&
        baseRowLen_ == 1 &&
        baseColLen_ == colLen_ &&
        tilingData_.get_biasIsEmpty() == 1 &&
        tilingData_.get_activateScaleIsEmpty() == 0) {
        return true;
    }
    return false;
}

uint64_t DequantSwigluClampQuantTiling::GetTilingKeyBaseMode() const
{
    if (quantMode_ == 0) { // static
        return static_cast<uint64_t>(getTilingKeyStatic(xInputDataType_, biasDataType_, quantScaleShapeSize_));
    } else { // dynamic
        return static_cast<uint64_t>(getTilingKeyDynamic(xInputDataType_, biasDataType_, quantScaleShapeSize_));
    }
}

ge::graphStatus DequantSwigluClampQuantTiling::GetWorkspaceSizeBaseMode()
{
    // 计算workspace大小，无需workspace临时空间，不存在多核同步，预留固定大小即可
    workspaceSize_ = USER_WORKSPACE;
    if (quantMode_ == 1 && (colLen_ > baseColLen_)) {
        workspaceSize_ += (totalUsedCoreNum_ * colLen_ * sizeof(float));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::PostTilingBaseMode()
{
    context_->SetTilingKey(GetTilingKeyBaseMode());
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(totalUsedCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DequantSwigluClampQuantTiling::DoTilingBaseMode()
{
    if (DoOpTilingBaseMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSizeBaseMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (PostTilingBaseMode() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

void DequantSwigluClampQuantTiling::ShowTilingData()
{
    OPS_LOG_I(context_->GetNodeName(),
            "rowLen: %lu, colLen: %lu, baseRowLen: %u, baseColLen: %u, usedCoreNum: %u",
            rowLen_, colLen_, baseRowLen_, baseColLen_, totalUsedCoreNum_);
}

// ============================================================
// 全局入口函数
// ============================================================

ge::graphStatus TilingForDequantSwigluClampQuant(gert::TilingContext* context)
{
    DequantSwigluClampQuantTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForDequantSwigluClampQuant(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4DequantSwigluClampQuant enter.");
    auto compileInfo = context->GetCompiledInfo<DequantSwigluClampQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                    OPS_LOG_E(context->GetNodeName(), "Get core num failed, core num: %u",
                                                    static_cast<uint32_t>(compileInfo->coreNum)),
                    return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                    OPS_LOG_E(context->GetNodeName(), "Get ub size failed, ub size: %u",
                                                    static_cast<uint32_t>(compileInfo->ubSize)),
                    return ge::GRAPH_FAILED);

    OPS_LOG_D(context, "TilingPrepare4DequantSwigluClampQuant exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DequantSwigluClampQuant)
    .Tiling(TilingForDequantSwigluClampQuant)
    .TilingParse<DequantSwigluClampQuantCompileInfo>(TilingPrepareForDequantSwigluClampQuant);

}  // namespace optiling