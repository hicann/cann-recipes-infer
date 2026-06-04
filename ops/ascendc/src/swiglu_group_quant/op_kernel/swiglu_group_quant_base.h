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
 * \file swiglu_group_quant_base.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_BASE_H
#define SWIGLU_GROUP_QUANT_BASE_H

#include "kernel_operator.h"

namespace SwigluGroupQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr int32_t PER_MX_FP16 = 32;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float TOPK_WEIGHT_DEFAULT = 1.0f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
constexpr int16_t BF16_EXP_SHR_BITS = 7;
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
#define FLOAT_OVERFLOW_MODE_CTRL 60
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait castTraitU32toU8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr AscendC::MicroAPI::CastTrait traitB16ToB32Layout0 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait traitB16ToB32Layout1 = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOuputDataUnalign(
    RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst, MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        DataCopyUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void StoreMxFp8Scale(
    __local_mem__ T* dst, RegTensor<int32_t>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<uint8_t> tmp1;
    Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp1, src, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}

__aicore__ inline void VFSwiGlu(RegTensor<float>& y, RegTensor<float>& x0, RegTensor<float>& x1,
    RegTensor<float>& one, RegTensor<float>& vreg, MaskReg pregLoop)
{
    Muls(vreg, x0, static_cast<float>(-1.0f), pregLoop);
    Exp(vreg, vreg, pregLoop);
    Adds(vreg, vreg, static_cast<float>(1.0f), pregLoop);
    Div(vreg, x0, vreg, pregLoop);
    Mul(y, vreg, x1, pregLoop);
}

template <typename T0, typename T1, typename T2, bool hasWeight = false, bool hasClampLimit = false>
__aicore__ inline void VFProcessSwigluGroupQuant(const LocalTensor<T0>& yLocal, const LocalTensor<T2>& scaleLocal,
    const LocalTensor<T1>& x0Local, const LocalTensor<T1>& x1Local, const LocalTensor<float> &weightLocal,
    float coeff, const uint16_t curRowNum, const uint32_t curColNum, float clampLimit)
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T2* scaleLocalAddr = (__local_mem__ T2*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* x0LocalAddr = (__local_mem__ T1*)x0Local.GetPhyAddr();
    __local_mem__ T1* x1LocalAddr = (__local_mem__ T1*)x1Local.GetPhyAddr();
    __local_mem__ float* weightLocalAddr = hasWeight ? (__local_mem__ float*)weightLocal.GetPhyAddr() : nullptr;
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailRemider = curColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNum = (curColNum + 128 - 1) / 128;
    uint32_t sregNum = loopCountReminder == 0 ? curColNum - loopCountFoldTwo * VL_FP32 : loopCountFoldTwo * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> weight;
        RegTensor<float> xLeft;
        RegTensor<float> xRight;
        RegTensor<float> x0Left;
        RegTensor<float> x0Right;
        RegTensor<float> x1Left;
        RegTensor<float> x1Right;
        RegTensor<float> xAbsLeft;
        RegTensor<float> xAbsRight;
        RegTensor<float> xMax;
        RegTensor<float> tmp;
        RegTensor<float> dupScale;
        RegTensor<float> scale;
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        RegTensor<float> inf;
        RegTensor<float> one;
        RegTensor<float> zero;
        RegTensor<uint32_t> coeffReg;
        UnalignReg uScale;
        MaskReg pregLoop = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        Duplicate(coeffReg, maxValueInt, pregLoop);
        Duplicate(zero, 0.0f);
        Duplicate(inf, 1.0f);
        Div<float, &mode>(inf, inf, zero, pregLoop);
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg compareLeft;
        MaskReg compareRight;
        MaskReg compareScalar;
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasWeight) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(weight, weightLocalAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(x0Left, x0LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x0Right, x0LocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Left, x1LocalAddr, pregMain, 2 * j * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Right, x1LocalAddr, pregLoop, (2 * j + 1) * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampLimit) {
                    Mins(x0Left, x0Left, clampLimit, pregMain);
                    Mins(x0Right, x0Right, clampLimit, pregLoop);
                    Maxs(x1Left, x1Left, -clampLimit, pregMain);
                    Mins(x1Left, x1Left, clampLimit, pregMain);
                    Maxs(x1Right, x1Right, -clampLimit, pregLoop);
                    Mins(x1Right, x1Right, clampLimit, pregLoop);
                }
                VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregMain);
                VFSwiGlu(xRight, x0Right, x1Right, one, tmp, pregLoop);
                if constexpr (hasWeight) {
                    Mul(xLeft, xLeft, weight, pregMain);
                    Mul(xRight, xRight, weight, pregLoop);
                }
                Muls(xAbsLeft, xLeft, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, xAbsLeft, xAbsLeft, pregMain);
                MaskNot(compareLeft, compareLeft, pregMain);
                Abs(xAbsLeft, xLeft, compareLeft);
                ReduceMax(scale0, xAbsLeft, pregMain);
                Muls(xAbsRight, xRight, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareRight, xAbsRight, xAbsRight, pregLoop);
                MaskNot(compareRight, compareRight, pregLoop);
                Abs(xAbsRight, xRight, compareRight);
                ReduceMax(scale1, xAbsRight, pregLoop);
                Max(scale, scale0, scale1, preg1);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregMain);
                StoreOuputDataUnalign(scale, scaleLocalAddr, uScale, preg1, 1);
                Div<float, &mode>(x0Left, xLeft, dupScale, pregMain);
                Muls(x1Left, x0Left, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregMain);
                Select(xLeft, xLeft, x0Left, compareLeft);
                Div<float, &mode>(x0Right, xRight, dupScale, pregLoop);
                Muls(x1Right, x0Right, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareRight, x1Right, x1Right, pregLoop);
                Select(xRight, xRight, x0Right, compareRight);
                StoreOutputData<T0>(yLocalAddr, xLeft, pregMain, 2 * j * VL_FP32 + i * dstCurColNumAlign);
                StoreOutputData<T0>(yLocalAddr, xRight, pregLoop, (2 * j + 1) * VL_FP32 + i * dstCurColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            pregLoop = UpdateMask<float>(tailRemider);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T1>(x0Left, x0LocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                LoadInputData<T1>(x1Left, x1LocalAddr, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * curColNumAlign);
                if constexpr (hasClampLimit) {
                    Mins(x0Left, x0Left, clampLimit, pregLoop);
                    Maxs(x1Left, x1Left, -clampLimit, pregLoop);
                    Mins(x1Left, x1Left, clampLimit, pregLoop);
                }
                VFSwiGlu(xLeft, x0Left, x1Left, one, tmp, pregLoop);
                if constexpr (hasWeight) {
                    Mul(xLeft, xLeft, weight, pregLoop);
                }
                Abs(xAbsLeft, xLeft, pregLoop);
                ReduceMax(scale, xAbsLeft, pregLoop);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                StoreOuputDataUnalign(scale, scaleLocalAddr, uScale, preg1, 1);
                Div<float, &mode>(x0Left, xLeft, dupScale, pregLoop);
                Muls(x1Left, x0Left, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregLoop);
                Select(xLeft, xLeft, x0Left, compareLeft);
                StoreOutputData(yLocalAddr, xLeft, pregLoop, loopCountFoldTwo * 2 * VL_FP32 + i * dstCurColNumAlign);
            }
        }
        DataCopyUnAlignPost(scaleLocalAddr, uScale, 0);
    }
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
    uint16_t curColNum)
{
    __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fourLoopCount = loopCount / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount % FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        __VEC_SCOPE__
        {
            RegTensor<T> x;
            RegTensor<T> sum;
            MaskReg pregMain = CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<T, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(sum, static_cast<T>(0), pregMain);
            uint32_t sreg = curColNum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<T>(sreg);
                DataCopy(x, xLocalAddr + i * vlLen);
                Adds(x, x, static_cast<T>(0), pregLoop);
                Add(sum, sum, x, pregMain);
            }
            ReduceSum(sum, sum, pregMain);
            if (withUbReduce) {
                RegTensor<T> origin;
                DataCopy(origin, yLocalAddr);
                Add(sum, sum, origin, pregMerge);
            }
            DataCopy(yLocalAddr, sum, pregMerge);
        }
    } else {
        __VEC_SCOPE__
        {
            RegTensor<T> x0;
            RegTensor<T> x1;
            RegTensor<T> x2;
            RegTensor<T> x3;
            RegTensor<T> sum0;
            RegTensor<T> sum1;
            RegTensor<T> sum2;
            RegTensor<T> sum3;
            MaskReg pregMain = CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<T, AscendC::MicroAPI::MaskPattern::VL1>();
            Duplicate(sum0, static_cast<T>(0), pregMain);
            Duplicate(sum1, static_cast<T>(0), pregMain);
            Duplicate(sum2, static_cast<T>(0), pregMain);
            Duplicate(sum3, static_cast<T>(0), pregMain);
            MaskReg pregLoop;
            for (uint16_t i = 0; i < fourLoopCount; i++) {
                DataCopy(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
                Add(sum0, sum0, x0, pregMain);
                DataCopy(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
                Add(sum1, sum1, x1, pregMain);
                DataCopy(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
                Add(sum2, sum2, x2, pregMain);
                DataCopy(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
                Add(sum3, sum3, x3, pregMain);
            }
            uint32_t sreg = tailReminder;
            for (uint16_t i = 0; i < tailLoopNum; i++) {
                pregLoop = UpdateMask<T>(sreg);
                DataCopy(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
                Adds(x0, x0, static_cast<T>(0), pregLoop);
                Add(sum0, sum0, x0, pregMain);
            }
            Add(sum0, sum0, sum1, pregMain);
            Add(sum2, sum2, sum3, pregMain);
            Add(sum0, sum0, sum2, pregMain);
            ReduceSum(sum0, sum0, pregMain);
            if (withUbReduce) {
                RegTensor<T> origin;
                DataCopy(origin, yLocalAddr);
                Add(sum0, sum0, origin, pregMerge);
            }
            DataCopy(yLocalAddr, sum0, pregMerge);
        }
    }
}


template <typename T0, typename T1, bool hasClampLimit = false, bool hasOutput = false, bool hasWeight = false>
__aicore__ inline void VFProcessSwigluMxFp8InvScale(const LocalTensor<T0>& yOriginLocal,
                                                    const LocalTensor<float>& yLocal,
                                                    const LocalTensor<uint8_t>& scaleLocal,
                                                    const LocalTensor<float>& invScaleLocal,
                                                    const LocalTensor<T0>& x0Local, const LocalTensor<T0>& x1Local,
                                                    const LocalTensor<float>& weightLocal,
                                                    float clampLimit, const uint16_t curRowNum,
                                                    const uint32_t curColNum)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ T0* yOriginLocalAddr = hasOutput ? (__local_mem__ T0*)yOriginLocal.GetPhyAddr() : nullptr;
    __local_mem__ uint8_t* scaleOriginLocalAddr = (__local_mem__ uint8_t*)scaleLocal.GetPhyAddr();
    __local_mem__ float* invScaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();
    __local_mem__ T0* x0LocalAddr = (__local_mem__ T0*)x0Local.GetPhyAddr();
    __local_mem__ T0* x1LocalAddr = (__local_mem__ T0*)x1Local.GetPhyAddr();
    __local_mem__ float* weightLocalAddr = hasWeight ? (__local_mem__ float*)weightLocal.GetPhyAddr() : nullptr;
    __local_mem__ uint8_t* scaleLocalAddr = scaleOriginLocalAddr;
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t vlLen = REPEAT_SIZE / sizeof(T0);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t curColNumAlignT = RoundUp<T0>(curColNum);
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum);
    uint32_t scaleColNum = CeilDiv(curColNum, 32);
    uint32_t invScaleColNumAlign = RoundUp<float>(curColNumAlignT / 8);

    __VEC_SCOPE__
    {
        RegTensor<float> weight;
        RegTensor<float> one;
        RegTensor<uint32_t> oneUint32;
        RegTensor<float> zero;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<T0> x0;
        RegTensor<T0> x1;
        RegTensor<float> x0Layout0;
        RegTensor<float> x0Layout1;
        RegTensor<float> x1Layout0;
        RegTensor<float> x1Layout1;
        RegTensor<float> yLayout0; // 奇偶交错 偶数部分
        RegTensor<float> yLayout1; // 奇偶交错 奇数部分
        RegTensor<float> y0; // 还原交错逻辑 高64位
        RegTensor<float> y1; // 还原交错逻辑 低64位
        RegTensor<float> yMax0;
        RegTensor<float> yMax1;
        RegTensor<float> yMax1Layout0;
        RegTensor<float> yMax1Layout1;
        RegTensor<float> scale;
        RegTensor<float> clampScale;
        RegTensor<float> invScale;
        RegTensor<float> invScale0;
        RegTensor<float> invScale1;
        RegTensor<float> dupInvScale;
        RegTensor<float> scale0;
        RegTensor<float> scale1;
        RegTensor<uint32_t> scale2;
        RegTensor<uint32_t> scale3;
        RegTensor<int32_t> scale4;
        RegTensor<int32_t> scale5;
        RegTensor<uint8_t> scale6;
        RegTensor<uint16_t> scale7;
        RegTensor<uint32_t> coeff;
        RegTensor<float> tmp;
        RegTensor<uint32_t> tmp0;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> tmp2;
        RegTensor<uint32_t> tmp3;
        RegTensor<int32_t> tmp4;
        RegTensor<uint8_t> tmp5;
        UnalignReg uReg;
        uint32_t sreg = (REPEAT_SIZE / sizeof(T0)) / 32;
        uint32_t sreg1 = 4 * (REPEAT_SIZE / sizeof(T0)) / 32;
        MaskReg pregMain0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMain1 = CreateMask<T0, MaskPattern::ALL>();
        MaskReg pregMerge = UpdateMask<float>(sreg);
        MaskReg pregMerge1 = UpdateMask<float>(sreg1);
        MaskReg compareMask0;
        Duplicate(one, 1.0f, pregMain0);
        Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain0);
        Duplicate(zero, 0.0f, pregMain0);
        Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain0);
        Duplicate(coeff, maxValueInt, pregMain0);
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        Duplicate(tmp4, static_cast<int32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            if constexpr (hasWeight) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(weight, weightLocalAddr + i);
            }
            for (uint16_t j = 0; j < loopCount; j++) {
                DataCopy(x0, x0LocalAddr + i * curColNumAlignT + j * vlLen);
                DataCopy(x1, x1LocalAddr + i * curColNumAlignT + j * vlLen);
                Cast<float, T0, traitB16ToB32Layout0>(x0Layout0, x0, pregMain1);
                Cast<float, T0, traitB16ToB32Layout1>(x0Layout1, x0, pregMain1);
                Cast<float, T0, traitB16ToB32Layout0>(x1Layout0, x1, pregMain1);
                Cast<float, T0, traitB16ToB32Layout1>(x1Layout1, x1, pregMain1);

                if constexpr (hasClampLimit) {
                    Mins(x0Layout0, x0Layout0, clampLimit, pregMain0);
                    Mins(x0Layout1, x0Layout1, clampLimit, pregMain0);
                    Maxs(x1Layout0, x1Layout0, -clampLimit, pregMain0);
                    Mins(x1Layout0, x1Layout0, clampLimit, pregMain0);
                    Maxs(x1Layout1, x1Layout1, -clampLimit, pregMain0);
                    Mins(x1Layout1, x1Layout1, clampLimit, pregMain0);
                }
                VFSwiGlu(yLayout0, x0Layout0, x1Layout0, one, tmp, pregMain0);
                VFSwiGlu(yLayout1, x0Layout1, x1Layout1, one, tmp, pregMain0);
                if constexpr (hasWeight) {
                    Mul(yLayout0, yLayout0, weight, pregMain0);
                    Mul(yLayout1, yLayout1, weight, pregMain0);
                }
                Add(yLayout0, yLayout0, zero, pregMain0);
                Add(yLayout1, yLayout1, zero, pregMain0);
                // 合并奇偶位置
                Interleave(y0, y1, yLayout0, yLayout1);
                if constexpr (hasOutput) {
                    StoreOutputData<T0>(yOriginLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignT);
                    StoreOutputData<T0>(yOriginLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignT);
                }
                StoreOutputData<float>(yLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
                StoreOutputData<float>(yLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);

                Abs(yLayout0, yLayout0, pregMain0);
                Abs(yLayout1, yLayout1, pregMain0);

                // fp32场景，32个数对应4个Block；先做一次Max，接着ReduceMaxWithBlock
                // 然后DeInterLeave并且Max，得到每4个Block的最大值
                Max(yMax0, yLayout0, yLayout1, pregMain0); // 4 --> 2
                ReduceMaxWithDataBlock(yMax1, yMax0, pregMain0);
                DeInterleave(yMax1Layout0, yMax1Layout1, yMax1, yMax1);

                Max(scale, yMax1Layout0, yMax1Layout1, pregMerge);

                Maxs(clampScale, scale, 0.0001f, pregMerge); // amax
                Mul(scale, clampScale, (RegTensor<float>&)coeff, pregMerge); // sf = amax / 448.0

                // 量化逻辑
                ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                And(scale2, scale2, tmp0, pregMerge); // exp
                And(scale3, (RegTensor<uint32_t>&)scale, tmp1, pregMerge); // man_bits
                Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                Select(tmp2, oneUint32, zeroUint32, compareMask0); // man_bits != 0
                Sub(scale3, scale2, tmp3, pregMerge); // exp - 127
                // exp_scale-uint32 = exp - 127 + (man_bits != 0)
                Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge);
                Adds(scale5, scale4, 127, pregMerge); // sf_uint32

                Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, scale5, pregMerge);
                Pack(scale7, (RegTensor<uint32_t>&)tmp5);
                Pack(scale6, scale7);
                DataCopyUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(scaleLocalAddr, scale6, uReg, 4);

                Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
                // ((127 - exp_scale) << 23).view(float32)
                ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge);
                Interleave(invScale0, invScale1, invScale, invScale);
                Interleave(invScale1, invScale, invScale0, invScale0);
                StoreOutputData<float>(invScaleLocalAddr, invScale1, pregMerge1, j * 16 + i * invScaleColNumAlign);
            }
            DataCopyUnAlignPost(scaleLocalAddr, uReg, 0);
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessSwigluMxFp8Quant(const LocalTensor<T>& yQuantLocal,
                                                 const LocalTensor<float>& yLocal,
                                                 const LocalTensor<float>& invScaleLocal, const uint16_t curRowNum,
                                                 const uint32_t curColNum)
{
    __local_mem__ T* yQuantLocalAddr = (__local_mem__ T*)yQuantLocal.GetPhyAddr();
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T>(curColNum);
    uint32_t invScaleColNumAlign = RoundUp<float>(CeilDiv(curColNum, 8));
    __VEC_SCOPE__
    {
        RegTensor<float> y;
        RegTensor<float> invScale;
        RegTensor<float> dupInvScale;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            for (uint16_t j = 0; j < loopCount; j++) {
                LoadInputData<float>(y, yLocalAddr, pregMain, i * curColNumAlign + j * VL_FP32);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(
                    invScale, scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
                Mul(y, y, invScale, pregMain);
                StoreOutputData<T>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}
} // namespace SwigluGroupQuant

#endif
