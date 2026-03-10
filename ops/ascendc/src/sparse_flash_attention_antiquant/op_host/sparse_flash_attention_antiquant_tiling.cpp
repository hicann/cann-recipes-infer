/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_antiquant_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_flash_attention_antiquant_template_tiling_key.h"
#include "sparse_flash_attention_antiquant_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_INT8, ge::DT_INT8}},
    {VALUE_NAME,                  {ge::DT_INT8, ge::DT_INT8}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}}
};

const std::map<std::string, std::vector<SFAALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {SFAALayout::BSND, SFAALayout::TND}},
    {KEY_NAME,               {SFAALayout::BSND, SFAALayout::TND, SFAALayout::PA_BSND}},
    {VALUE_NAME,             {SFAALayout::BSND, SFAALayout::TND, SFAALayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {SFAALayout::BSND, SFAALayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

struct SparseFlashAttentionAntiquantCompileInfo {
    int64_t coreNum;
};

static const std::map<SFAALayout, std::vector<SFAAAxis>> SFAA_LAYOUT_AXIS_MAP = {
    {SFAALayout::BSND, {SFAAAxis::B, SFAAAxis::S, SFAAAxis::N, SFAAAxis::D}},
    {SFAALayout::TND, {SFAAAxis::T, SFAAAxis::N, SFAAAxis::D}},
    {SFAALayout::PA_BSND, {SFAAAxis::Bn, SFAAAxis::Bs, SFAAAxis::N, SFAAAxis::D}},
};

static const std::map<SFAALayout, size_t> SFAA_LAYOUT_DIM_MAP = {
    {SFAALayout::BSND, DIM_NUM_FOUR},
    {SFAALayout::TND, DIM_NUM_THREE},
    {SFAALayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string SFAADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("SparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string SFAATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << SFAAShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << SFAAShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
            static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
            .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string SFAADebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << SFAATensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << SFAATensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string SFAALayoutToSerialString(SFAALayout layout)
{
    switch (layout) {
        case SFAALayout::BSND: return "BSND";
        case SFAALayout::TND: return "TND";
        case SFAALayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus SFAAMlaTiling::SetBlockDim(uint32_t blockDim)
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAMlaTiling::SetTilingKey(uint64_t tilingKey)
{
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize)
{
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAMlaTiling::SetTilingData(TilingDef &tilingData)
{
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAMlaTiling::GetPlatformInfo()
{
    OPS_ERR_IF(sfaaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SFAAMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaaInfo_->kvLayout);

    tilingKey_ = GET_TPL_TILING_KEY(0U, layoutQuery, layoutKV, perfMode_ == SFAAPerfMode::V_TEMPLATE_MODE);

    OPS_LOG_I(sfaaInfo_->opName, "SFAA tilingKey_: %lu.", tilingKey_);
}

void SFAAMlaTiling::ZeroTensorProcess()
{
    if (sfaaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaaInfo_->s2Size = 1024;
    }
}

void SFAAMlaTiling::InitParams()
{
    perfMode_ = SFAAPerfMode::V_TEMPLATE_MODE;
    coreNum_ = aicNum_;

    headDimAlign_ = Align(sfaaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void SFAAMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaaInfo_->gSize * sfaaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaaInfo_->gSize * (headDimAlign_ + 64U) * sfaaInfo_->s1Size;
}

void SFAAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void SFAAMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaaInfo_->qLayout != SFAALayout::TND) {
        if (s2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (s2Size > 256 && s2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void SFAAMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaaInfo_->s2Size);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaaInfo_->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void SFAAMlaTiling::Split()
{
    SplitBalanced();
}

void SFAAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaaInfo_->n1Size / sfaaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaaInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(sfaaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaaInfo_->sparseBlockCount);
}

// for flash decode
void SFAAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaaInfo_->n2Size * mBaseSize_ * headDimAlign_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaaInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void SFAAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void SFAAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void SFAAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t SFAAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)
{
    return actCoreNum * 2 * sfaaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void SFAAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaaInfo_->blockTypeSize); // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaaInfo_->blockTypeSize;
        if (sfaaInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void SFAAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void SFAAMlaTiling::GetWorkspaceSize()
{
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
    uint32_t softmaxSumElemSize = 4;   // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    uint32_t actCoreNum = coreNum_;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
        qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize; // vec2ResGm
    // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
    //              bufNum  s2Base   D   dRope  sizeOf(half)
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum; // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
    // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数

    CalcFDWorkSpace(actCoreNum);
}

void SFAAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OPS_LOG_I(sfaaInfo_->opName, "SFAA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus SFAAMlaTiling::DoOpTiling(SFAATilingInfo *sfaaInfo)
{
    sfaaInfo_ = sfaaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseFlashAttentionAntiquant(gert::TilingContext *context)
{
    SFAATilingInfo sfaaInfo;
    SFAAInfoParser sfaaInfoParser(context);
    if (sfaaInfoParser.Parse(sfaaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFAATilingCheck tilingChecker(sfaaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SFAAMlaTiling tiling(context);
    return tiling.DoOpTiling(&sfaaInfo);
}

ge::graphStatus TilingPrepareForSparseFlashAttentionAntiquant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const SFAATilingShapeCompareParam &param, const SFAALayout &layout) const
{
    if (layout == SFAALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == SFAALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == SFAALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OPS_LOG_E(opName_, "layout %s is unsupported", SFAALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CompareShape(SFAATilingShapeCompareParam &param,
    const gert::Shape &shape, const SFAALayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OPS_LOG_E(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OPS_LOG_E(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), SFAALayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void SFAATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SFAADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), SFAADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus SFAATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OPS_ERR_IF(it == DTYPE_SUPPORT_MAP.end(),
            OPS_LOG_E(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OPS_ERR_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SFAATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OPS_LOG_E(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void SFAATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SFAATilingCheck::CheckDimNumInLayoutSupport(const SFAALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SFAA_LAYOUT_DIM_MAP.find(layout);
    OPS_ERR_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OPS_LOG_E(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            SFAALayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


void SFAATilingCheck::LogErrorLayoutSupport(const std::vector<SFAALayout> &expectLayoutList,
    const SFAALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << SFAALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), SFAALayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus SFAATilingCheck::CheckLayoutSupport(const SFAALayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OPS_ERR_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OPS_LOG_E(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OPS_ERR_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, keyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaSparseMode() const
{
    OPS_ERR_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OPS_LOG_E(opName_, "sparseMode must == 0/3, but got: %ld.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OPS_ERR_IF((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 16),
        OPS_LOG_E(opName_, "sparseBlockSize should be in range [1, 16], but got: %ld.", *opParamInfo_.sparseBlockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckDequantScaleNotExistence()
{
    if (quantScaleRepoMode_ == 1) {
        OPS_ERR_IF((opParamInfo_.keyDequantScale.tensor == nullptr || opParamInfo_.valueDequantScale.tensor == nullptr),
            OPS_LOG_E(opName_,
            "When quant_scale_repo_mode is 1(combine), key_dequant_scale and value_dequant_scale should be both null."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer == nullptr,
        OPS_LOG_E(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer != nullptr,
        OPS_LOG_E(opName_, "%s should be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
    std::map<std::string, const void *> &notExistMap) const
{
    if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus SFAATilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OPS_LOG_E(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OPS_LOG_E(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckParaExistenceMlaAntiquant() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    std::map<std::string, const void *> mlaAntiquantParamExistMap = {
        {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
        {"blockTable", opParamInfo_.blockTable.tensor},
    };
    std::map<std::string, const void *> mlaAntiquantParamNotExistMap = {};
    if (CheckExistenceByMap(mlaAntiquantParamExistMap, mlaAntiquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaAntiquant();
}

ge::graphStatus SFAATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckDequantScaleNotExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus SFAATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const SFAALayout &layoutQuery, const std::string &name)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            SFAALayoutToSerialString(layoutQuery).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

void SFAATilingCheck::SetSFAAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
}

ge::graphStatus SFAATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor != nullptr,
            OPS_LOG_E(opName_, "when the layout_kv is %s, %s should be null",
                SFAALayoutToSerialString(kvLayout_).c_str(), BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OPS_ERR_IF(blockTableBatch != bSize_,
        OPS_LOG_E(opName_, "%s's first dimension(%u) should be equal to batch size(%u)",
            BLOCK_TABLE_NAME.c_str(), blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OPS_LOG_E(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            SFAADataTypeToSerialString(expectDtype).c_str(),
            SFAADataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckTopkShape()
{
    SFAATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus SFAATilingCheck::CheckAttenOutShape()
{
    SFAATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = 512; // 512:输出的head_dim
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckKVShapeForBatchContinuous()
{
    SFAATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint32_t SFAATilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus SFAATilingCheck::CheckKVShapeForPageAttention()
{
    uint32_t kvBlockElemNum = 32 / GetTypeSize(inputKvType_);

    int64_t blockNum = keyShapeCmp_.GetDim(0);
    SFAATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.T = kvTSize_;
    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckKVShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKVShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckKVShapeForPageAttention();
    }

    OPS_LOG_E(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus SFAATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            SFAADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            SFAADataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckMultiParaConsistency()
{
    SetSFAAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquantShape() const
{
    OPS_ERR_IF(bSize_ <= 0,
        OPS_LOG_E(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qTSize_ <= 0 && (qLayout_ == SFAALayout::TND),
            OPS_LOG_E(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ <= 0,
            OPS_LOG_E(opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
            return ge::GRAPH_FAILED);

    OPS_ERR_IF(n2Size_ != 1,
        OPS_LOG_E(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ % n2Size_ != 0,
        OPS_LOG_E(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OPS_LOG_E(opName_, "group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != 576,
        OPS_LOG_E(opName_, "qk_head_dim only support 576, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OPS_ERR_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OPS_LOG_E(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquantDtype() const
{
    OPS_ERR_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OPS_LOG_E(opName_, "query dtype only support %s and %s, but got %s",
            SFAADataTypeToSerialString(ge::DT_BF16).c_str(), SFAADataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            SFAADataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquantAttr() const
{
    OPS_ERR_IF(attentionMode_ != 2, // 2:MLA-absorb
        OPS_LOG_E(opName_, "attention_mode should be 2(MLA-absorb), but got %u",
        attentionMode_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(keyQuantMode_ != 2, // 2:per-tile
        OPS_LOG_E(opName_, "key_quant_mode should be 2(per-tile), but got %u",
        keyQuantMode_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(valueQuantMode_ != 2, // 2:per-tile
        OPS_LOG_E(opName_, "value_quant_mode should be 2(per-tile), but got %u",
        valueQuantMode_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(quantScaleRepoMode_ != 1, // 1:combine
        OPS_LOG_E(opName_, "quant_scale_repo_mode should be 1(combine), but got %u",
        quantScaleRepoMode_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(tileSize_ != 128, // 128:当前不泛化
        OPS_LOG_E(opName_, "tile_size should be 128, but got %u",
        tileSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(ropeHeadDim_ != 64, // 64:当前不泛化
        OPS_LOG_E(opName_, "rope_head_dim should be 64, but got %u",
        ropeHeadDim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    OPS_ERR_IF(blockSize_ % 16 > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);

    OPS_ERR_IF(blockSize_ % sparseBlockSize_ > 0,
        OPS_LOG_E(opName_,
            "when page attention is enabled, block_size(%d) must be divided by sparse_block_size(%d), but now the remainder is %d.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMlaAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaAntiquant();
}

ge::graphStatus SFAATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void SFAATilingCheck::Init()
{
    opName_ = sfaaInfo_.opName;
    platformInfo_ = sfaaInfo_.platformInfo;
    opParamInfo_ = sfaaInfo_.opParamInfo;
    socVersion_ = sfaaInfo_.socVersion;

    bSize_ = sfaaInfo_.bSize;
    n1Size_ = sfaaInfo_.n1Size;
    n2Size_ = sfaaInfo_.n2Size;
    s1Size_ = sfaaInfo_.s1Size;
    s2Size_ = sfaaInfo_.s2Size;
    gSize_ = sfaaInfo_.gSize;
    qkHeadDim_ = sfaaInfo_.qkHeadDim;
    vHeadDim_ = sfaaInfo_.vHeadDim;
    ropeHeadDim_ = sfaaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = sfaaInfo_.maxBlockNumPerBatch;
    qTSize_ = sfaaInfo_.qTSize;
    kvTSize_ = sfaaInfo_.kvTSize;
    blockSize_ = sfaaInfo_.blockSize;
    sparseBlockCount_ = sfaaInfo_.sparseBlockCount;
    sparseBlockSize_ = sfaaInfo_.sparseBlockSize;

    attentionMode_ = sfaaInfo_.attentionMode;
    keyQuantMode_ = sfaaInfo_.keyQuantMode;
    valueQuantMode_ = sfaaInfo_.valueQuantMode;
    quantScaleRepoMode_ = sfaaInfo_.quantScaleRepoMode;
    tileSize_ = sfaaInfo_.tileSize;

    inputQType_ = sfaaInfo_.inputQType;
    inputKvType_ = sfaaInfo_.inputKvType;
    outputType_ = sfaaInfo_.outputType;

    qLayout_ = sfaaInfo_.qLayout;
    topkLayout_ = sfaaInfo_.topkLayout;
    kvLayout_ = sfaaInfo_.kvLayout;
    outLayout_ = sfaaInfo_.outLayout;

    kvStorageMode_ = sfaaInfo_.kvStorageMode;
    l2CacheSize_ = sfaaInfo_.l2CacheSize;
}

ge::graphStatus SFAATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool SFAAInfoParser::HasAxis(const SFAAAxis &axis, const SFAALayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = SFAA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SFAA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SFAAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    const auto& dimIt = SFAA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == SFAA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t SFAAInfoParser::GetAxisIdx(const SFAAAxis &axis, const SFAALayout &layout) const
{
    const std::vector<SFAAAxis>& axes = SFAA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SFAAInfoParser::GetAxisNum(const gert::Shape &shape, const SFAAAxis &axis, const SFAALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus SFAAInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr,
        OPS_LOG_E(opName_, "Shape of tensor sparseIndices is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr,
        OPS_LOG_E(opName_, "Desc of tensor sparseIndices is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layoutQuery == nullptr, OPS_LOG_E(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.layoutKV == nullptr, OPS_LOG_E(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr, OPS_LOG_E(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.scaleValue == nullptr, OPS_LOG_E(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    SFAALayout &layout, const std::string &name)
{
    if ((tensor == nullptr)) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            SFAALayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus SFAAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("SparseFlashAttentionAntiquant", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void SFAAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.keyDequantScale.tensor = context_->GetOptionalInputTensor(KEY_DEQUANT_SCALE_INPUT_INDEX);
    opParamInfo_.valueDequantScale.tensor = context_->GetOptionalInputTensor(VALUE_DEQUANT_SCALE_INPUT_INDEX);
}

void SFAAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void SFAAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus SFAAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int64_t>(KEY_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.valueQuantMode = attrs->GetAttrPointer<int64_t>(VALUE_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
    opParamInfo_.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
    opParamInfo_.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SFAALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, SFAAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus SFAAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SFAALayout::TND) ? GetAxisNum(queryShape_, SFAAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以key的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == SFAALayout::TND) ? GetAxisNum(keyShape_, SFAAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(queryShape_, SFAAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SFAALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, SFAAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, SFAAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == SFAALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准值
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetKvLayout()
{
    const map<string, SFAALayout> layoutKVMap = {
        {"BSND",        SFAALayout::BSND},
        {"PA_BSND",     SFAALayout::PA_BSND},
        {"TND",         SFAALayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_LOG_E(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != SFAALayout::PA_BSND && qLayout_ != kvLayout_) {
        OPS_LOG_E(opName_, "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == SFAALayout::PA_BSND && keyDimNum != 4U) {
        OPS_LOG_E(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == SFAALayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, SFAAAxis::S, kvLayout_);
    } else if (kvLayout_ == SFAALayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, SFAAAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_LOG_E(opName_, "the layout_kv is %s, blockTable must be provided.",
            SFAALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != DIM_NUM_TWO) {
        OPS_LOG_E(opName_, "the dim num of block_table is %u, it should be %u.", dimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OPS_LOG_E(opName_, "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(), opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, SFAAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, SFAAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus SFAAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, SFAAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<SFAALayout, SFAALayout>> layoutMap = {
        {"BSND",        {SFAALayout::BSND,    SFAALayout::BSND}},
        {"TND",         {SFAALayout::TND,     SFAALayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OPS_LOG_E(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, SFAAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, SFAAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void SFAAInfoParser::SetSFAAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
}

ge::graphStatus SFAAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SFAAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

void SFAAInfoParser::GenerateInfo(SFAATilingInfo &sfaaInfo)
{
    sfaaInfo.opName = opName_;
    sfaaInfo.platformInfo = platformInfo_;
    sfaaInfo.opParamInfo = opParamInfo_;
    sfaaInfo.socVersion = socVersion_;

    sfaaInfo.bSize = bSize_;
    sfaaInfo.n1Size = n1Size_;
    sfaaInfo.n2Size = n2Size_;
    sfaaInfo.s1Size = s1Size_;
    sfaaInfo.s2Size = s2Size_;
    sfaaInfo.gSize = gSize_;
    sfaaInfo.qkHeadDim = qkHeadDim_;
    sfaaInfo.vHeadDim = vHeadDim_;
    sfaaInfo.qTSize = qTSize_;
    sfaaInfo.kvTSize = kvTSize_;
    sfaaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    sfaaInfo.sparseBlockCount = sparseBlockCount_;

    sfaaInfo.inputQType = inputQType_;
    sfaaInfo.inputKvType = inputKvType_;
    sfaaInfo.outputType = outputType_;

    sfaaInfo.kvStorageMode = kvStorageMode_;
    sfaaInfo.l2CacheSize = l2CacheSize_;

    sfaaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    sfaaInfo.scaleValue = *opParamInfo_.scaleValue;
    sfaaInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    sfaaInfo.blockSize = blockSize_;
    sfaaInfo.blockTypeSize =  sizeof(float);
    sfaaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    sfaaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaaInfo.maxActualseq = maxActualseq_;
    sfaaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    sfaaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaaInfo.isSameActualseq = isSameActualseq_;

    sfaaInfo.sparseMode = *opParamInfo_.sparseMode;
    sfaaInfo.attentionMode = *opParamInfo_.attentionMode;
    sfaaInfo.keyQuantMode = *opParamInfo_.keyQuantMode;
    sfaaInfo.valueQuantMode = *opParamInfo_.valueQuantMode;
    sfaaInfo.quantScaleRepoMode = *opParamInfo_.quantScaleRepoMode;
    sfaaInfo.tileSize = *opParamInfo_.tileSize;
    sfaaInfo.ropeHeadDim = *opParamInfo_.ropeHeadDim;

    sfaaInfo.qLayout = qLayout_;
    sfaaInfo.topkLayout = topkLayout_;
    sfaaInfo.kvLayout = kvLayout_;
    sfaaInfo.outLayout = outLayout_;
}

ge::graphStatus SFAAInfoParser::Parse(SFAATilingInfo &sfaaInfo)
{
    if (context_ == nullptr) {
        OPS_LOG_E("SparseFlashAttentionAntiquant", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_FULL(DLOG_INFO, "SparseFlashAttentionAntiquant", "TilingContext: %s",
        SFAADebugTilingContext(context_).c_str());
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetSFAAShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(sfaaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseFlashAttentionAntiquant)
    .Tiling(TilingSparseFlashAttentionAntiquant)
    .TilingParse<SparseFlashAttentionAntiquantCompileInfo>(TilingPrepareForSparseFlashAttentionAntiquant);
} // namespace optiling
