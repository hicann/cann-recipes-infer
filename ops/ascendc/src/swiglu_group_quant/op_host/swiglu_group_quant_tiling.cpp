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
 * \file swiglu_group_quant_tiling.cpp
 * \brief
 */

#include <cmath>
#include <sstream>
#include "swiglu_group_quant_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t D_LIMIT = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t FP8_BYTES = 1;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t FP8_ALIGN_NUM = BLOCK_SIZE / FP8_BYTES;
constexpr int64_t B16_ALIGN_NUM = BLOCK_SIZE / B16_BYTES;
constexpr int64_t B32_ALIGN_NUM = BLOCK_SIZE / B32_BYTES;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t BLOCK_QUANT = 0;
constexpr int64_t MX_QUANT = 1;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_BLOCK_SIZE = 2;
constexpr size_t ATTR_INDEX_ROUND_SCALE = 3;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 4;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 5;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t OUTPUT_INDEX_SCALE_OUT = 1;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 2;
constexpr size_t CACHE_LINE_SIZE = 128;
constexpr int64_t BLOCK_QUANT_TILING_KEY = 1000;
constexpr int64_t MX_QUANT_TILING_KEY = 2100;
constexpr int64_t MX_QUANT_YORIGIN_TILING_KEY = 2200;

}

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<SwigluGroupQuantCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto dstTypeAttr = attrs->GetAttrPointer<int>(ATTR_INDEX_DST_TYPE);
    dstType_ = dstTypeAttr == nullptr ? ge::DT_FLOAT8_E4M3FN : static_cast<ge::DataType>(*dstTypeAttr);
    OPS_ERR_IF((dstType_ != ge::DT_FLOAT8_E4M3FN && dstType_ != ge::DT_FLOAT8_E5M2),
        OPS_LOG_E(context_->GetNodeName(), "attr dst_type only support FLOAT8_E4M3FN or FLOAT8_E5M2, got %d.",
                  static_cast<int>(dstType_)),
        return ge::GRAPH_FAILED);

    auto quantModeAttr = attrs->GetAttrPointer<int>(ATTR_INDEX_QUANT_MODE);
    quantMode_ = quantModeAttr == nullptr ? BLOCK_QUANT : *quantModeAttr;
    OPS_ERR_IF((quantMode_ != BLOCK_QUANT && quantMode_ != MX_QUANT),
        OPS_LOG_E(context_->GetNodeName(), "attr quant_mode only support 0(block_quant) or 1(mx_quant), got %ld.",
                  quantMode_),
        return ge::GRAPH_FAILED);

    auto blockSizeAttr = attrs->GetAttrPointer<int>(ATTR_INDEX_BLOCK_SIZE);
    int64_t blockSize = blockSizeAttr == nullptr ? 0 : *blockSizeAttr;
    int64_t expectedBlockSize = quantMode_ == MX_QUANT ? PER_MX_FP16 : PER_BLOCK_FP16;
    OPS_ERR_IF((blockSize != 0 && blockSize != expectedBlockSize),
        OPS_LOG_E(context_->GetNodeName(),
                  "attr block_size should be 0 or %ld when quant_mode is %ld, got %ld.",
                  expectedBlockSize, quantMode_, blockSize),
        return ge::GRAPH_FAILED);

    splitFactor_ = expectedBlockSize;

    auto roundScaleAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_ROUND_SCALE);
    if (roundScaleAttr != nullptr) {
        roundScale_ = (*roundScaleAttr) ? 1 : 0;
    }
    OPS_ERR_IF((quantMode_ == BLOCK_QUANT && roundScale_ != 0),
        OPS_LOG_E(context_->GetNodeName(), "attr round_scale should be false when quant_mode is 0."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((quantMode_ == MX_QUANT && roundScale_ != 1),
        OPS_LOG_E(context_->GetNodeName(), "attr round_scale should be true when quant_mode is 1."),
        return ge::GRAPH_FAILED);

    auto outputOriginAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginAttr != nullptr) {
        outputOrigin_ = (*outputOriginAttr) ? 1 : 0;
    }

    auto clampLimitAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    if (clampLimitAttr != nullptr) {
        // -inf means user did not pass clamp_limit.
        if (!(std::isinf(*clampLimitAttr) && *clampLimitAttr < 0.0f)) {
            OPS_ERR_IF(!(*clampLimitAttr >= 0.0f),
                OPS_LOG_E(context_->GetNodeName(), "attr clamp_limit should be greater than or equal to 0.0, got %f.",
                          *clampLimitAttr),
                return ge::GRAPH_FAILED);
            clampLimit_ = *clampLimitAttr;
            hasClampLimit_ = 1;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mix)
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OPS_LOG_E_IF_NULL(context_, shapeX, return ge::GRAPH_FAILED);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OPS_LOG_E_IF_NULL(context_, xDesc, return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    auto xDimNum = xStorageShape.GetDimNum();
    OPS_ERR_IF((xDimNum == 0),
        OPS_LOG_E(context_->GetNodeName(), "input x dim num should be greater than 0."),
        return ge::GRAPH_FAILED);
    bs_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        bs_ = bs_ * xStorageShape.GetDim(i);
    }
    d_ = xStorageShape.GetDim(xDimNum - 1);
    OPS_ERR_IF((d_ < D_LIMIT || d_ % D_LIMIT != 0),
        OPS_LOG_E(context_->GetNodeName(),
                  "input x last dim should be greater than or equal to %ld and divisible by %ld, got %ld.",
                  D_LIMIT, D_LIMIT, d_),
        return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
        if (weightShape != nullptr) {
            auto weightStorageShape = weightShape->GetStorageShape();
            if (weightStorageShape.GetDimNum() != 0) {
                hasWeight_ = true;
            }
        }
    }

    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            g_ = 1;
            for (size_t i = 0; i < groupIndexStorageShape.GetDimNum(); i++) {
                g_ = g_ * groupIndexStorageShape.GetDim(i);
            }
            hasGroupIndex_ = true;
        }
    }

    // Get Attrs
    if (GetAttr() == ge::GRAPH_FAILED) {
        OPS_LOG_E(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }

    auto yDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y);
    OPS_LOG_E_IF_NULL(context_, yDesc, return ge::GRAPH_FAILED);
    auto yDtype = yDesc->GetDataType();
    OPS_ERR_IF((yDtype != dstType_),
        OPS_LOG_E(context_->GetNodeName(), "output y dtype should be same as dst_type, got y dtype %d, dst_type %d.",
                  static_cast<int>(yDtype), static_cast<int>(dstType_)),
        return ge::GRAPH_FAILED);

    auto scaleOutDesc = context_->GetOutputDesc(OUTPUT_INDEX_SCALE_OUT);
    OPS_LOG_E_IF_NULL(context_, scaleOutDesc, return ge::GRAPH_FAILED);
    auto scaleOutDtype = scaleOutDesc->GetDataType();
    auto expectedScaleOutDtype = quantMode_ == MX_QUANT ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT;
    OPS_ERR_IF((scaleOutDtype != expectedScaleOutDtype),
        OPS_LOG_E(context_->GetNodeName(),
                  "output scale_out dtype should be %d when quant_mode is %ld, got %d.",
                  static_cast<int>(expectedScaleOutDtype), quantMode_, static_cast<int>(scaleOutDtype)),
        return ge::GRAPH_FAILED);

    auto yOriginDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    OPS_LOG_E_IF_NULL(context_, yOriginDesc, return ge::GRAPH_FAILED);
    auto yOriginDtype = yOriginDesc->GetDataType();
    OPS_ERR_IF((yOriginDtype != xDtype),
        OPS_LOG_E(context_->GetNodeName(),
                  "output y_origin dtype should be same as input x, got y_origin dtype %d, x dtype %d.",
                  static_cast<int>(yOriginDtype), static_cast<int>(xDtype)),
        return ge::GRAPH_FAILED);

    splitD_ = d_ / 2;
    scaleCol_ = CeilDiv(splitD_, splitFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcGroupIndexTiling()
{
    if (hasGroupIndex_) {
        gFactor_ = g_;
        int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
        int64_t groupIndexSumSize = BLOCK_SIZE;
        if (groupIndexSize + groupIndexSumSize <= ubSize_) {
            gLoop_ = 1;
            tailGFactor_ = gFactor_;
        } else {
            int64_t base = 2;
            while(1) {
                gFactor_ = CeilDiv(g_, base);
                groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
                if (groupIndexSize + groupIndexSumSize < ubSize_) {
                    break;
                }
                base++;
            }
            if (gFactor_ > CACHE_LINE_SIZE / sizeof(int64_t)) {
                gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / sizeof(int64_t));
            }
            gLoop_ = CeilDiv(g_, gFactor_);
            tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxQuantOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t x0Size = rowOnceLoop * RoundUp(splitD_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowOnceLoop * RoundUp(splitD_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(splitD_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowOnceLoop * scaleCol_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t yFp32Size = rowOnceLoop * RoundUp(splitD_, B32_ALIGN_NUM) * B32_BYTES;
    int64_t invScaleSize =
        rowOnceLoop * RoundUp(CeilDiv(splitD_, B32_ALIGN_NUM), B32_ALIGN_NUM) * B32_BYTES;

    int64_t totalSize = x0Size + x1Size + ySize + scaleSize + yFp32Size + invScaleSize;

    int64_t weightSize = RoundUp(rowOnceLoop, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
    totalSize = hasWeight_ ? totalSize + weightSize : totalSize;

    int64_t yOriginSize = rowOnceLoop * RoundUp(splitD_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    totalSize = outputOrigin_ ? totalSize + yOriginSize : totalSize;

    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = splitD_;
        tailDFactor_ = dFactor_;
    } else {
        dFactor_ = splitD_;
        int64_t base = 1;
        while (totalSize < ubSize_) {
            dFactor_ = base * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            x0Size = rowOnceLoop * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            x1Size = rowOnceLoop * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowOnceLoop * scaleCol_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            yFp32Size = rowOnceLoop * RoundUp(dFactor_, B32_ALIGN_NUM) * B32_BYTES;
            invScaleSize =
                rowOnceLoop * RoundUp(CeilDiv(dFactor_, B32_ALIGN_NUM), B32_ALIGN_NUM) * B32_BYTES;
            totalSize = x0Size + x1Size + ySize + scaleSize + yFp32Size + invScaleSize;
            if (hasWeight_) {
                totalSize += weightSize;
            }
            if (outputOrigin_) {
                yOriginSize = rowOnceLoop * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
                totalSize += yOriginSize;
            }
            base++;
        }
        dFactor_ = (base - 1) * splitFactor_;
        scaleCol_ = CeilDiv(dFactor_, splitFactor_);
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == splitD_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            x0Size = rowFactor_ * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            x1Size = rowFactor_ * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            ySize = rowFactor_ * RoundUp(dFactor_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowFactor_ * scaleCol_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            yFp32Size = rowFactor_ * RoundUp(dFactor_, B32_ALIGN_NUM) * B32_BYTES;
            invScaleSize =
                rowFactor_ * RoundUp(CeilDiv(dFactor_, B32_ALIGN_NUM), B32_ALIGN_NUM) * B32_BYTES;
            totalSize = x0Size + x1Size + ySize + scaleSize + yFp32Size + invScaleSize;
            if (hasWeight_) {
                weightSize = RoundUp(rowFactor_, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
                totalSize += weightSize;
            }
            if (outputOrigin_) {
                yOriginSize = rowFactor_ * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
                totalSize += yOriginSize;
            }
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcBlockQuantOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t x0Size = rowOnceLoop * RoundUp(splitD_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowOnceLoop * RoundUp(splitD_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(splitD_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowOnceLoop * scaleCol_, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;

    int64_t totalSize = x0Size + x1Size + ySize + scaleSize;

    int64_t weightSize = RoundUp(rowOnceLoop, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
    totalSize = hasWeight_ ? totalSize + weightSize : totalSize;

    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = splitD_;
        tailDFactor_ = dFactor_;
    } else {
        dFactor_ = splitD_;
        int64_t base = 1;
        while (totalSize < ubSize_) {
            dFactor_ = base * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            x0Size = rowOnceLoop * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            x1Size = rowOnceLoop * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowOnceLoop * scaleCol_, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasWeight_) {
                totalSize += weightSize;
            }
            base++;
        }
        dFactor_ = (base - 1) * splitFactor_;
        scaleCol_ = CeilDiv(dFactor_, splitFactor_);
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == splitD_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            x0Size = rowFactor_ * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            x1Size = rowFactor_ * RoundUp(dFactor_, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
            ySize = rowFactor_ * RoundUp(dFactor_, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowFactor_ * scaleCol_, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasWeight_) {
                weightSize = RoundUp(rowFactor_, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
                totalSize += weightSize;
            }
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::SetTilingData()
{
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_splitD(splitD_);
    tilingData_.set_scaleCol(scaleCol_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_outputOrigin(outputOrigin_);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_g(g_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_gLoop(gLoop_);
    tilingData_.set_gFactor(gFactor_);
    tilingData_.set_tailGFactor(tailGFactor_);
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_hasClampLimit(hasClampLimit_);
}

ge::graphStatus SwigluGroupQuantTiling::CalcOpTiling() {
    ge::graphStatus status;
    status = CalcGroupIndexTiling();
    if (status == ge::GRAPH_FAILED) {
        return status;
    }
    if (quantMode_ == BLOCK_QUANT) {
        status = CalcBlockQuantOpTiling();
    } else if (quantMode_ == MX_QUANT) {
        status = CalcMxQuantOpTiling();
    }
    return status;
}

ge::graphStatus SwigluGroupQuantTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    int64_t tilingKey = quantMode_;
    if (quantMode_ == BLOCK_QUANT) {
        tilingKey = BLOCK_QUANT_TILING_KEY;
    } else if (quantMode_ == MX_QUANT) {
        if (outputOrigin_) {
            tilingKey = MX_QUANT_YORIGIN_TILING_KEY;
        } else {
            tilingKey = MX_QUANT_TILING_KEY;
        }
    }
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::PostTiling()
{
    if (hasGroupIndex_) {
        context_->SetBlockDim(coreNum_);
    } else {
        context_->SetBlockDim(usedCoreNums_);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSwigluGroupQuant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluGroupQuant(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SwigluGroupQuant", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    SwigluGroupQuantTiling SwigluGroupQuantTiling(context);
    return SwigluGroupQuantTiling.DoOpTiling();
}

IMPL_OP_OPTILING(SwigluGroupQuant)
    .Tiling(TilingForSwigluGroupQuant)
    .TilingParse<SwigluGroupQuantCompileInfo>(TilingPrepareForSwigluGroupQuant);
}  // namespace optiling
