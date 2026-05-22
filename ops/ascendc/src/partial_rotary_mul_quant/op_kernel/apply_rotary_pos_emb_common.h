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
 * \file apply_rotary_pos_emb_common.h
 * \brief  All data (in/cos/sin) use type T, output is hifloat8 (uint8_t* + dAlignH8Out stride).
 *         VF handles T->f32 and f32->hifloat8 internally.
 */
#ifndef APPLY_ROTARY_POS_EMB_COMMON_H
#define APPLY_ROTARY_POS_EMB_COMMON_H

#include "partial_rotary_mul_quant_common.h"

using namespace AscendC;

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if defined(__DAV_C310__)
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr uint32_t VL_FLOAT32_SIZE = GetVRegSize() / sizeof(float);
constexpr uint32_t VL_FLOAT16_SIZE = GetVRegSize() / sizeof(half);
constexpr uint32_t BLOCK_TYPE_SIZE = GetUbBlockSize();
constexpr uint32_t HALF_INTERLEAVE_COEF = 2;
constexpr uint32_t QUARTER_MODE_COEF = 4;
constexpr uint32_t DOUBLE_BUFFER = 2;

enum class RotaryPosEmbeddingMode : int64_t {
    HALF = 0,
    INTERLEAVE = 1,
    QUARTER = 2,
    DEEPSEEK_INTERLEAVE = 3
};

/*
    qOut[0] = q[0] * cos[0] - q[1] * sin[0]
    qOut[1] = q[1] * cos[1] + q[0] * sin[1]
*/
template <typename T>
__aicore__ inline void HalfAlignVF(
    __local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in, const LocalTensor<uint8_t>& outTensor,
    uint32_t dLen, uint32_t dAlign, uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum, uint16_t currDNum,
    float scale)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = ops::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = ops::CeilDiv(halfD, VL_FLOAT32_SIZE);
    __local_mem__ T* currInUb;
    __local_mem__ hifloat8_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn;
        MicroAPI::RegTensor<float> vregHalfIn;
        MicroAPI::RegTensor<float> vregSin;
        MicroAPI::RegTensor<float> vregHalfSin;
        MicroAPI::RegTensor<float> vregCos;
        MicroAPI::RegTensor<float> vregHalfCos;
        MicroAPI::RegTensor<float> vregOut;
        MicroAPI::RegTensor<float> vregHalfOut;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * cosSinAlign;
            currCosUb = cosUb + sIdx * cosSinAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlignH8Out;
                uint32_t updateCnt = halfD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregIn, vregHalfIn, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, preg, preg, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, preg, preg, offset, halfOffset);

                    Mul(vregSin, vregSin, vregHalfIn, preg);
                    Mul(vregHalfOut, vregHalfSin, vregIn, preg);
                    Mul(vregCos, vregCos, vregIn, preg);
                    Sub(vregOut, vregCos, vregSin, preg);
                    Mul(vregHalfCos, vregHalfCos, vregHalfIn, preg);
                    Add(vregHalfOut, vregHalfOut, vregHalfCos, preg);

                    Muls(vregOut, vregOut, scale, preg);
                    Muls(vregHalfOut, vregHalfOut, scale, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregOut, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregHalfOut, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + offset, tmp1, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + halfOffset, tmp2, preg);
                }
            }
        }
    }
}

/*
    qOut[0] = q[0] * cos[0] - q[1] * sin[0]
    qOut[1] = q[1] * cos[1] + q[0] * sin[1]
    qOut[2] = q[2] * cos[2] - q[3] * sin[2]
    qOut[3] = q[3] * cos[3] + q[2] * sin[3]
*/
template <typename T>
__aicore__ inline void QuarterAlignVF(
    __local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in, const LocalTensor<uint8_t>& outTensor,
    uint32_t dLen, uint32_t dAlign, uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum, uint16_t currDNum,
    float scale)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t quarterD = dLen / QUARTER_MODE_COEF;
    uint32_t quarterDAlign = ops::CeilAlign(quarterD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint16_t repeatTimes = ops::CeilDiv(quarterD, VL_FLOAT32_SIZE);
    __local_mem__ T* currInUb;
    __local_mem__ hifloat8_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn;
        MicroAPI::RegTensor<float> vregQ1In;
        MicroAPI::RegTensor<float> vregQ2In;
        MicroAPI::RegTensor<float> vregQ3In;
        MicroAPI::RegTensor<float> vregSin;
        MicroAPI::RegTensor<float> vregQ1Sin;
        MicroAPI::RegTensor<float> vregQ2Sin;
        MicroAPI::RegTensor<float> vregQ3Sin;
        MicroAPI::RegTensor<float> vregCos;
        MicroAPI::RegTensor<float> vregQ1Cos;
        MicroAPI::RegTensor<float> vregQ2Cos;
        MicroAPI::RegTensor<float> vregQ3Cos;
        MicroAPI::RegTensor<float> vregOut;
        MicroAPI::RegTensor<float> vregQ1Out;
        MicroAPI::RegTensor<float> vregQ2Out;
        MicroAPI::RegTensor<float> vregQ3Out;
        MicroAPI::RegTensor<hifloat8_t> tmp1, tmp2, tmp3, tmp4;
        MicroAPI::MaskReg preg;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * cosSinAlign;
            currCosUb = cosUb + sIdx * cosSinAlign;
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * dAlignH8Out;
                uint32_t updateCnt = quarterD;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = MicroAPI::UpdateMask<float>(updateCnt);
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t q1Offset = offset + quarterDAlign;
                    uint32_t q2Offset = q1Offset + quarterDAlign;
                    uint32_t q3Offset = q2Offset + quarterDAlign;
                    ops::LoadTwoTensorForDtypeT<T>(currInUb, currInUb, vregIn, vregQ1In, preg, preg, offset, q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregQ2In, vregQ3In, preg, preg, q2Offset, q3Offset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregQ1Sin, preg, preg, offset, q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregQ2Sin, vregQ3Sin, preg, preg, q2Offset, q3Offset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregQ1Cos, preg, preg, offset, q1Offset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregQ2Cos, vregQ3Cos, preg, preg, q2Offset, q3Offset);

                    Mul(vregSin, vregSin, vregQ1In, preg);
                    Mul(vregQ1Out, vregQ1Sin, vregIn, preg);
                    Mul(vregQ2Sin, vregQ2Sin, vregQ3In, preg);
                    Mul(vregQ3Out, vregQ3Sin, vregQ2In, preg);
                    Mul(vregCos, vregCos, vregIn, preg);
                    Sub(vregOut, vregCos, vregSin, preg);
                    Mul(vregQ1Cos, vregQ1Cos, vregQ1In, preg);
                    Add(vregQ1Out, vregQ1Out, vregQ1Cos, preg);
                    Mul(vregQ2Cos, vregQ2Cos, vregQ2In, preg);
                    Sub(vregQ2Out, vregQ2Cos, vregQ2Sin, preg);
                    Mul(vregQ3Cos, vregQ3Cos, vregQ3In, preg);
                    Add(vregQ3Out, vregQ3Out, vregQ3Cos, preg);

                    Muls(vregOut, vregOut, scale, preg);
                    Muls(vregQ1Out, vregQ1Out, scale, preg);
                    Muls(vregQ2Out, vregQ2Out, scale, preg);
                    Muls(vregQ3Out, vregQ3Out, scale, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregOut, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregQ1Out, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp3, vregQ2Out, preg);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp4, vregQ3Out, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + offset, tmp1, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + q1Offset, tmp2, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + q2Offset, tmp3, preg);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + q3Offset, tmp4, preg);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void InterleaveModeVF(
    __local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in, const LocalTensor<uint8_t>& outTensor,
    uint32_t dLen, uint32_t dAlign, uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum, uint16_t currDNum,
    float scale)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint16_t repeatTimes = dLen / VL_FLOAT32_SIZE;
    uint32_t dAlignLen = dAlign;
    uint16_t loopNum = repeatTimes / 2;
    uint32_t tailNum = dLen - loopNum * 2 * VL_FLOAT32_SIZE;
    uint16_t tailTwoVL = tailNum / VL_FLOAT32_SIZE;
    uint16_t tailOneVL = (tailTwoVL == 1) ? 0 : 1;
    uint32_t tailLen = tailNum % VL_FLOAT32_SIZE;
    __local_mem__ T* currInUb;
    __local_mem__ hifloat8_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;
    __local_mem__ T* tailSinUb;
    __local_mem__ T* tailCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregFormerCos;
        MicroAPI::RegTensor<float> vregLatterCos;
        MicroAPI::RegTensor<float> vregFormerSin;
        MicroAPI::RegTensor<float> vregLatterSin;
        MicroAPI::RegTensor<float> vregFormerIn;
        MicroAPI::RegTensor<float> vregLatterIn;
        MicroAPI::RegTensor<float> vregOdd;
        MicroAPI::RegTensor<float> vregEven;
        MicroAPI::RegTensor<float> vregFormerOut;
        MicroAPI::RegTensor<float> vregLatterOut;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregTail;
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * cosSinAlign;
            currCosUb = cosUb + sIdx * cosSinAlign;
            for (uint16_t idxD = 0; idxD < currDNum; idxD++) {
                uint32_t updateCnt = dLen;
                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen;
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignH8Out;
                pregLoop = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
                for (uint16_t i = 0; i < loopNum; i++) {
                    uint32_t evenOffSet = (i * 2) * VL_FLOAT32_SIZE;
                    uint32_t oddOffset = evenOffSet + VL_FLOAT32_SIZE;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregFormerCos, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currCosUb, vregLatterCos, pregLoop, oddOffset);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregFormerSin, pregLoop, evenOffSet);
                    ops::LoadOneTensorForDtypeT<T>(currSinUb, vregLatterSin, pregLoop, oddOffset);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregLoop);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregLoop);
                    Add(vregLatterCos, vregLatterCos, vregLatterSin, pregLoop);
                    Muls(vregFormerCos, vregFormerCos, scale, pregLoop);
                    Muls(vregLatterCos, vregLatterCos, scale, pregLoop);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregFormerCos, pregLoop);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregLatterCos, pregLoop);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + evenOffSet, tmp1, pregLoop);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + oddOffset, tmp2, pregLoop);
                }

                currInUb = inUb + (sIdx * currDNum + idxD) * dAlignLen + (loopNum * 2 * VL_FLOAT32_SIZE);
                currOutUb = outUb + (sIdx * currDNum + idxD) * dAlignH8Out + (loopNum * 2 * VL_FLOAT32_SIZE);
                tailSinUb = sinUb + sIdx * cosSinAlign + loopNum * 2 * VL_FLOAT32_SIZE;
                tailCosUb = cosUb + sIdx * cosSinAlign + loopNum * 2 * VL_FLOAT32_SIZE;
                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregLatterIn, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregLatterCos, pregTail, VL_FLOAT32_SIZE);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregLoop, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregLatterSin, pregTail, VL_FLOAT32_SIZE);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregLoop);
                    Mul(vregLatterCos, vregLatterCos, vregLatterIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregLoop);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregLoop);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregLoop);
                    Mul(vregLatterSin, vregLatterSin, vregLatterIn, pregTail);
                    Add(vregLatterCos, vregLatterCos, vregLatterSin, pregTail);
                    Muls(vregFormerCos, vregFormerCos, scale, pregLoop);
                    Muls(vregLatterCos, vregLatterCos, scale, pregTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregFormerCos, pregLoop);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregLatterCos, pregTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb, tmp1, pregLoop);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + VL_FLOAT32_SIZE, tmp2, pregTail);
                }

                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t updateCnt = tailLen;
                    pregTail = MicroAPI::UpdateMask<float>(updateCnt);
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregFormerIn, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailCosUb, vregFormerCos, pregTail, 0);
                    ops::LoadOneTensorForDtypeT<T>(tailSinUb, vregFormerSin, pregTail, 0);
                    Mul(vregFormerCos, vregFormerCos, vregFormerIn, pregTail);
                    MicroAPI::DeInterleave<float>(vregEven, vregOdd, vregFormerIn, vregLatterIn);
                    Muls(vregOdd, vregOdd, float(-1.0), pregTail);
                    MicroAPI::Interleave<float>(vregFormerIn, vregLatterIn, vregOdd, vregEven);
                    Mul(vregFormerSin, vregFormerSin, vregFormerIn, pregTail);
                    Add(vregFormerCos, vregFormerCos, vregFormerSin, pregTail);
                    Muls(vregFormerCos, vregFormerCos, scale, pregTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregFormerCos, pregTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb, tmp1, pregTail);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void DeepSeekInterleaveModeVF(
    __local_mem__ T* sin, __local_mem__ T* cos, __local_mem__ T* in, const LocalTensor<uint8_t>& outTensor,
    uint32_t dLen, uint32_t dAlign, uint32_t cosSinAlign, uint32_t dAlignH8Out, uint16_t currSNum, uint16_t currDNum,
    float scale)
{
    __local_mem__ T* sinUb = sin;
    __local_mem__ T* cosUb = cos;
    __local_mem__ T* inUb = in;
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t halfD = dLen / HALF_INTERLEAVE_COEF;
    uint32_t halfDAlign = ops::CeilAlign(halfD, static_cast<uint32_t>(BLOCK_TYPE_SIZE / sizeof(T)));
    uint32_t halfCosSinAlign = cosSinAlign / HALF_INTERLEAVE_COEF;
    uint16_t repeatTimes = halfD / VL_FLOAT32_SIZE;
    uint32_t tailTwoNum = dLen - repeatTimes * VL_FLOAT32_SIZE * HALF_INTERLEAVE_COEF;
    uint16_t tailTwoVL = tailTwoNum > VL_FLOAT32_SIZE ? 1 : 0;
    uint16_t tailOneVL = tailTwoNum > 0 ? (1 - tailTwoVL) : 0;
    uint32_t halfTailNum = tailTwoNum / HALF_INTERLEAVE_COEF;
    uint32_t tailNum = tailTwoNum - tailTwoVL * VL_FLOAT32_SIZE;
    __local_mem__ T* currInUb;
    __local_mem__ hifloat8_t* currOutUb;
    __local_mem__ T* currSinUb;
    __local_mem__ T* currCosUb;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregIn;
        MicroAPI::RegTensor<float> vregHalfIn;
        MicroAPI::RegTensor<float> vregSin;
        MicroAPI::RegTensor<float> vregHalfSin;
        MicroAPI::RegTensor<float> vregCos;
        MicroAPI::RegTensor<float> vregHalfCos;
        MicroAPI::RegTensor<float> vregOut;
        MicroAPI::RegTensor<float> vregHalfOut;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg pregTail;
        MicroAPI::MaskReg pregHalfTail;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        for (uint16_t sIdx = 0; sIdx < currSNum; sIdx++) {
            currSinUb = sinUb + sIdx * halfCosSinAlign * HALF_INTERLEAVE_COEF;
            currCosUb = cosUb + sIdx * halfCosSinAlign * HALF_INTERLEAVE_COEF;
            uint32_t updateTailNum = tailNum;
            uint32_t updateHalfTailNum = halfTailNum;
            pregTail = MicroAPI::UpdateMask<float>(updateTailNum);
            pregHalfTail = MicroAPI::UpdateMask<float>(updateHalfTailNum);
            for (uint16_t row = 0; row < currDNum; row++) {
                currInUb = inUb + (sIdx * currDNum + row) * dAlign;
                currOutUb = outUb + (sIdx * currDNum + row) * halfDAlign * HALF_INTERLEAVE_COEF;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    uint32_t offset = i * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregIn, vregHalfIn, pregFull, pregFull, inOffset,
                        inOffset + VL_FLOAT32_SIZE);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, pregFull, pregFull, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, pregFull, pregFull, offset, halfOffset);

                    DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);

                    Mul(vregOut, vregCos, vregIn, pregFull);
                    Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregFull);
                    Muls(vregHalfIn, vregHalfIn, float(-1.0), pregFull);
                    Mul(vregSin, vregSin, vregHalfIn, pregFull);
                    Add(vregOut, vregOut, vregSin, pregFull);
                    Mul(vregHalfSin, vregHalfSin, vregIn, pregFull);
                    Add(vregHalfOut, vregHalfOut, vregHalfSin, pregFull);

                    Muls(vregOut, vregOut, scale, pregFull);
                    Muls(vregHalfOut, vregHalfOut, scale, pregFull);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregOut, pregFull);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregHalfOut, pregFull);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + offset, tmp1, pregFull);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + halfOffset, tmp2, pregFull);
                }

                for (uint16_t i = 0; i < tailTwoVL; i++) {
                    uint32_t offset = repeatTimes * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadTwoTensorForDtypeT<T>(
                        currInUb, currInUb, vregIn, vregHalfIn, pregFull, pregTail, inOffset,
                        inOffset + VL_FLOAT32_SIZE);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, pregHalfTail, pregHalfTail, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, pregHalfTail, pregHalfTail, offset, halfOffset);

                    DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);

                    Mul(vregOut, vregCos, vregIn, pregHalfTail);
                    Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregHalfTail);
                    Muls(vregHalfIn, vregHalfIn, float(-1.0), pregHalfTail);
                    Mul(vregSin, vregSin, vregHalfIn, pregHalfTail);
                    Add(vregOut, vregOut, vregSin, pregHalfTail);
                    Mul(vregHalfSin, vregHalfSin, vregIn, pregHalfTail);
                    Add(vregHalfOut, vregHalfOut, vregHalfSin, pregHalfTail);

                    Muls(vregOut, vregOut, scale, pregHalfTail);
                    Muls(vregHalfOut, vregHalfOut, scale, pregHalfTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregOut, pregHalfTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregHalfOut, pregHalfTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + offset, tmp1, pregHalfTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + halfOffset, tmp2, pregHalfTail);
                }

                for (uint16_t i = 0; i < tailOneVL; i++) {
                    uint32_t offset = repeatTimes * VL_FLOAT32_SIZE;
                    uint32_t halfOffset = offset + halfDAlign;
                    uint32_t inOffset = offset * HALF_INTERLEAVE_COEF;
                    ops::LoadOneTensorForDtypeT<T>(currInUb, vregIn, pregTail, inOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currSinUb, currSinUb, vregSin, vregHalfSin, pregHalfTail, pregHalfTail, offset, halfOffset);
                    ops::LoadTwoTensorForDtypeT<T>(
                        currCosUb, currCosUb, vregCos, vregHalfCos, pregHalfTail, pregHalfTail, offset, halfOffset);

                    DeInterleave<float>(vregIn, vregHalfIn, vregIn, vregHalfIn);
                    Mul(vregOut, vregCos, vregIn, pregHalfTail);
                    Mul(vregHalfOut, vregHalfCos, vregHalfIn, pregHalfTail);
                    Muls(vregHalfIn, vregHalfIn, float(-1.0), pregHalfTail);
                    Mul(vregSin, vregSin, vregHalfIn, pregHalfTail);
                    Add(vregOut, vregOut, vregSin, pregHalfTail);
                    Mul(vregHalfSin, vregHalfSin, vregIn, pregHalfTail);
                    Add(vregHalfOut, vregHalfOut, vregHalfSin, pregHalfTail);
                    Muls(vregOut, vregOut, scale, pregHalfTail);
                    Muls(vregHalfOut, vregHalfOut, scale, pregHalfTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, vregOut, pregHalfTail);
                    Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, vregHalfOut, pregHalfTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + offset, tmp1, pregHalfTail);
                    DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + halfOffset, tmp2, pregHalfTail);
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchHalfAlignVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, const LocalTensor<uint8_t>& outTensor,
    uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
    int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN, float scale)
{
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t dHalfSize = d / HALF_INTERLEAVE_COEF;
    uint16_t dLoopCount = (dHalfSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;
    uint32_t cosHalfOffset = cosSinAlign / HALF_INTERLEAVE_COEF;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ T *currInUb;
        __local_mem__ hifloat8_t *currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dHalfSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = outUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currInUb, inPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + dHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currCosUb, cosPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + cosHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currSinUb, sinPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + cosHalfOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
                        Muls(cosPart1Reg, cosPart1Reg, scale, pregLoop);
                        Muls(cosPart2Reg, cosPart2Reg, scale, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, cosPart1Reg, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, cosPart2Reg, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE, tmp1, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dHalfOffset, tmp2, pregLoop);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchQuarterAlignVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, const LocalTensor<uint8_t>& outTensor,
    uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
    int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN, float scale)
{
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t dQuarterSize = d / QUARTER_MODE_COEF;
    uint16_t dLoopCount = (dQuarterSize + VL_FLOAT32_SIZE - 1) / VL_FLOAT32_SIZE;
    uint32_t dQuarterOffset = dAlign / QUARTER_MODE_COEF;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;
    uint32_t dThreeQuarterOffset = dQuarterOffset + dHalfOffset;

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> inPart3Reg;
        MicroAPI::RegTensor<float> inPart4Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> cosPart3Reg;
        MicroAPI::RegTensor<float> cosPart4Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<float> sinPart3Reg;
        MicroAPI::RegTensor<float> sinPart4Reg;
        MicroAPI::RegTensor<hifloat8_t> tmp1, tmp2, tmp3, tmp4;
        MicroAPI::MaskReg pregLoop;
        __local_mem__ T *currInUb;
        __local_mem__ hifloat8_t *currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t count = dQuarterSize;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = outUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(count);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currInUb, currInUb, inPart1Reg, inPart2Reg, pregLoop, pregLoop, i * VL_FLOAT32_SIZE,
                            i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currInUb, currInUb, inPart3Reg, inPart4Reg, pregLoop, pregLoop,
                            i * VL_FLOAT32_SIZE + dHalfOffset, i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currCosUb, currCosUb, cosPart1Reg, cosPart2Reg, pregLoop, pregLoop, i * VL_FLOAT32_SIZE,
                            i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currCosUb, currCosUb, cosPart3Reg, cosPart4Reg, pregLoop, pregLoop,
                            i * VL_FLOAT32_SIZE + dHalfOffset, i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currSinUb, currSinUb, sinPart1Reg, sinPart2Reg, pregLoop, pregLoop, i * VL_FLOAT32_SIZE,
                            i * VL_FLOAT32_SIZE + dQuarterOffset);
                        ops::LoadTwoTensorForDtypeT<T>(
                            currSinUb, currSinUb, sinPart3Reg, sinPart4Reg, pregLoop, pregLoop,
                            i * VL_FLOAT32_SIZE + dHalfOffset, i * VL_FLOAT32_SIZE + dThreeQuarterOffset);
                        Mul(cosPart1Reg, inPart1Reg, cosPart1Reg, pregLoop);
                        Mul(sinPart1Reg, inPart2Reg, sinPart1Reg, pregLoop);
                        Sub(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Mul(cosPart2Reg, inPart2Reg, cosPart2Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
                        Mul(cosPart3Reg, inPart3Reg, cosPart3Reg, pregLoop);
                        Mul(sinPart3Reg, inPart4Reg, sinPart3Reg, pregLoop);
                        Sub(cosPart3Reg, cosPart3Reg, sinPart3Reg, pregLoop);
                        Mul(cosPart4Reg, inPart4Reg, cosPart4Reg, pregLoop);
                        Mul(sinPart4Reg, sinPart4Reg, inPart3Reg, pregLoop);
                        Add(cosPart4Reg, cosPart4Reg, sinPart4Reg, pregLoop);
                        Muls(cosPart1Reg, cosPart1Reg, scale, pregLoop);
                        Muls(cosPart2Reg, cosPart2Reg, scale, pregLoop);
                        Muls(cosPart3Reg, cosPart3Reg, scale, pregLoop);
                        Muls(cosPart4Reg, cosPart4Reg, scale, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, cosPart1Reg, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, cosPart2Reg, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp3, cosPart3Reg, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp4, cosPart4Reg, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE, tmp1, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dQuarterOffset, tmp2, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dHalfOffset, tmp3, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dThreeQuarterOffset, tmp4, pregLoop);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchInterleaveModeVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, const LocalTensor<uint8_t>& outTensor,
    uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
    int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN, float scale)
{
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;

    uint32_t halfNum = d / 2;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = d - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        __local_mem__ T *currInUb;
        __local_mem__ hifloat8_t *currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = outUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(
                            currCosUb, cosPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(
                            currSinUb, sinPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregPart1);
                        Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregPart2);
                        MicroAPI::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        MicroAPI::Interleave<float>(inPart1Reg, inPart2Reg, inPart2Reg, inPart1Reg);
                        Mul(sinPart1Reg, sinPart1Reg, inPart1Reg, pregPart1);
                        Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregPart1);
                        Mul(sinPart2Reg, sinPart2Reg, inPart2Reg, pregPart2);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregPart2);
                        Muls(cosPart1Reg, cosPart1Reg, scale, pregPart1);
                        Muls(cosPart2Reg, cosPart2Reg, scale, pregPart2);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, cosPart1Reg, pregPart1);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, cosPart2Reg, pregPart2);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * loopSize, tmp1, pregPart1);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * loopSize + VL_FLOAT32_SIZE, tmp2, pregPart2);
                    }
                }
            }
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void BatchDeepSeekInterleaveModeVF(
    __local_mem__ T* in, __local_mem__ T* cos, __local_mem__ T* sin, const LocalTensor<uint8_t>& outTensor,
    uint16_t sLength, uint16_t bLength, uint16_t nLength, int64_t d, int64_t dAlign, int64_t cosSinAlign,
    int64_t dAlignH8Out, int64_t ubFactorS, int64_t ubFactorN, float scale)
{
    LocalTensor<hifloat8_t> outH8 = outTensor.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* outUb = (__local_mem__ hifloat8_t*)outH8.GetPhyAddr();
    uint32_t loopSize = 2 * VL_FLOAT32_SIZE;
    uint16_t dLoopCount = (d + loopSize - 1) / loopSize;
    uint32_t dHalfOffset = dAlign / HALF_INTERLEAVE_COEF;
    uint32_t cosHalfOffset = cosSinAlign / HALF_INTERLEAVE_COEF;

    uint32_t halfNum = d / 2;
    uint32_t part1Num = (dLoopCount - 1) * VL_FLOAT32_SIZE;
    uint32_t part2Num = part1Num;
    uint32_t tailNum = d - part1Num - part2Num;
    if (tailNum > VL_FLOAT32_SIZE) {
        part1Num += VL_FLOAT32_SIZE;
        part2Num += (tailNum - VL_FLOAT32_SIZE);
    } else {
        part1Num += tailNum;
    }

    int32_t bStepUb = ubFactorN * ubFactorS * dAlign;
    int32_t nStepUb = ubFactorS * dAlign;
    int32_t bStepOut = ubFactorN * ubFactorS * dAlignH8Out;
    int32_t nStepOut = ubFactorS * dAlignH8Out;
    int32_t nStepCosSin = ubFactorS * cosSinAlign;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inPart1Reg;
        MicroAPI::RegTensor<float> inPart2Reg;
        MicroAPI::RegTensor<float> cosPart1Reg;
        MicroAPI::RegTensor<float> cosPart2Reg;
        MicroAPI::RegTensor<float> sinPart1Reg;
        MicroAPI::RegTensor<float> sinPart2Reg;
        MicroAPI::RegTensor<hifloat8_t> tmp1;
        MicroAPI::RegTensor<hifloat8_t> tmp2;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg pregPart1;
        MicroAPI::MaskReg pregPart2;
        __local_mem__ T *currInUb;
        __local_mem__ hifloat8_t *currOutUb;
        __local_mem__ T *currSinUb, *currCosUb;
        for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
            for (uint16_t nIdx = 0; nIdx < nLength; nIdx++) {
                for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
                    uint32_t halfCnt = halfNum;
                    uint32_t part1Cnt = part1Num;
                    uint32_t part2Cnt = part2Num;
                    currInUb = in + bIdx * bStepUb + nIdx * nStepUb + sIdx * dAlign;
                    currOutUb = outUb + bIdx * bStepOut + nIdx * nStepOut + sIdx * dAlignH8Out;
                    if constexpr (IsBBoardcast) {
                        currCosUb = cos + sIdx * cosSinAlign;
                        currSinUb = sin + sIdx * cosSinAlign;
                    } else {
                        currCosUb = cos + bIdx * nStepCosSin + sIdx * cosSinAlign;
                        currSinUb = sin + bIdx * nStepCosSin + sIdx * cosSinAlign;
                    }
                    for (uint16_t i = 0; i < dLoopCount; i++) {
                        pregLoop = MicroAPI::UpdateMask<float>(halfCnt);
                        pregPart1 = MicroAPI::UpdateMask<float>(part1Cnt);
                        pregPart2 = MicroAPI::UpdateMask<float>(part2Cnt);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart1Reg, pregPart1, i * loopSize);
                        ops::LoadOneTensorForDtypeT<T>(currInUb, inPart2Reg, pregPart2, i * loopSize + VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(currCosUb, cosPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currCosUb, cosPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + cosHalfOffset);
                        ops::LoadOneTensorForDtypeT<T>(currSinUb, sinPart1Reg, pregLoop, i * VL_FLOAT32_SIZE);
                        ops::LoadOneTensorForDtypeT<T>(
                            currSinUb, sinPart2Reg, pregLoop, i * VL_FLOAT32_SIZE + cosHalfOffset);
                        MicroAPI::DeInterleave<float>(inPart1Reg, inPart2Reg, inPart1Reg, inPart2Reg);
                        Mul(cosPart1Reg, cosPart1Reg, inPart1Reg, pregLoop);
                        Mul(cosPart2Reg, cosPart2Reg, inPart2Reg, pregLoop);
                        Muls(inPart2Reg, inPart2Reg, float(-1.0), pregLoop);
                        Mul(sinPart1Reg, sinPart1Reg, inPart2Reg, pregLoop);
                        Add(cosPart1Reg, cosPart1Reg, sinPart1Reg, pregLoop);
                        Mul(sinPart2Reg, sinPart2Reg, inPart1Reg, pregLoop);
                        Add(cosPart2Reg, cosPart2Reg, sinPart2Reg, pregLoop);
                        Muls(cosPart1Reg, cosPart1Reg, scale, pregLoop);
                        Muls(cosPart2Reg, cosPart2Reg, scale, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp1, cosPart1Reg, pregLoop);
                        Cast<hifloat8_t, float, ops::castTraitF32toh8>(tmp2, cosPart2Reg, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE, tmp1, pregLoop);
                        DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                            currOutUb + i * VL_FLOAT32_SIZE + dHalfOffset, tmp2, pregLoop);
                    }
                }
            }
        }
    }
}

#endif // APPLY_ROTARY_POS_EMB_COMMON_H