/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_COMPRESS_EPILOG_V2_COMMON_H
#define KV_COMPRESS_EPILOG_V2_COMMON_H

#include "kernel_operator.h"

namespace KvCompressEpilogV2Ops {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;

constexpr int32_t KCEV2_BLOCK_BYTES = 32;
constexpr int32_t KCEV2_GROUP_ELEMS = 32;
constexpr int32_t KCEV2_VL_FP32 = 64;
constexpr float KCEV2_FP8_E5M2_MAX = 57344.0f;
constexpr float KCEV2_FP8_E4M3_MAX = 448.0f;
constexpr uint32_t KCEV2_FAST_LOG_SHIFT = 23U;
constexpr uint32_t KCEV2_EXP_MASK = 0xFFU;
constexpr uint32_t KCEV2_MANTISSA_MASK = (1U << 23U) - 1U;
constexpr uint16_t KCEV2_BF16_EXP_MASK = 0x7F80U;
constexpr uint16_t KCEV2_BF16_NAN = 0x7F81U;
constexpr uint16_t KCEV2_BF16_INV_BIAS = 0x7F00U;
constexpr uint16_t KCEV2_FP4_E2M1_MAX_EXP = 0x0100U;
constexpr uint16_t KCEV2_FP4_SPECIAL_INV = 0x0040U;

#define FLOAT_OVERFLOW_MODE_CTRL 60

__aicore__ inline int32_t CeilDiv(int32_t value, int32_t divisor)
{
    return divisor == 0 ? value : (value + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t count)
{
    const int32_t blockElems = KCEV2_BLOCK_BYTES / sizeof(T);
    return CeilDiv(count, blockElems) * blockElems;
}

constexpr AscendC::MicroAPI::CastTrait KCEV2_B16_TO_F32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait KCEV2_F32_TO_B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait KCEV2_F32_TO_FP8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::MicroAPI::CastTrait KCEV2_B16_TO_FP4 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadB16AsFloat(RegTensor<float> &dst, __local_mem__ T *src, MaskReg mask)
{
    RegTensor<T> tmp;
    DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src);
    Cast<float, T, KCEV2_B16_TO_F32>(dst, tmp, mask);
}

template <typename T>
__aicore__ inline void StoreFloatAsFp8(__local_mem__ T *dst, RegTensor<float> &src, MaskReg mask)
{
    RegTensor<T> tmp;
    Cast<T, float, KCEV2_F32_TO_FP8>(tmp, src, mask);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst, tmp, mask);
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &src, const LocalTensor<T> &dst, uint32_t count)
{
    DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyOutBytes(const LocalTensor<uint8_t> &src, const GlobalTensor<uint8_t> &dst, uint32_t count)
{
    DataCopyExtParams copyParams{1, count, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

template <typename TCache, bool roundScale>
__aicore__ inline void VFProcessMxFp8(
    const LocalTensor<uint8_t> &output, const LocalTensor<bfloat16_t> &input, uint16_t rowCount,
    uint32_t d, uint32_t dataCol, uint32_t concatCol, uint32_t cacheCol)
{
    __local_mem__ TCache *outputData = reinterpret_cast<__local_mem__ TCache *>(output.GetPhyAddr());
    __local_mem__ bfloat16_t *outputScale = reinterpret_cast<__local_mem__ bfloat16_t *>(output.GetPhyAddr());
    __local_mem__ bfloat16_t *inputData = reinterpret_cast<__local_mem__ bfloat16_t *>(input.GetPhyAddr());
    const uint32_t inputStride = RoundUp<bfloat16_t>(d);
    const uint16_t groupCount = static_cast<uint16_t>(d / KCEV2_GROUP_ELEMS);
    const float fp8Max = IsSameType<TCache, fp8_e5m2_t>::value ? KCEV2_FP8_E5M2_MAX : KCEV2_FP8_E4M3_MAX;
    const float fp8Min = -fp8Max;
    const float coeff = 1.0f / fp8Max;

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> amax;
        RegTensor<float> scale;
        RegTensor<float> scaleDup;
        RegTensor<bfloat16_t> scaleBf16;
        RegTensor<uint32_t> exp;
        RegTensor<uint32_t> mantissa;
        RegTensor<uint32_t> expMask;
        RegTensor<uint32_t> mantissaMask;
        RegTensor<uint32_t> roundUp;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<int32_t> unbiasedExp;
        RegTensor<uint8_t> zeroBytes;
        MaskReg groupMask;
        MaskReg hasMantissa;
        MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
        MaskReg byteMask = CreateMask<uint8_t, MaskPattern::ALL>();
        UnalignReg paddingReg;
        Duplicate(zero, static_cast<uint32_t>(0), scalarMask);
        Duplicate(one, static_cast<uint32_t>(1), scalarMask);
        Duplicate(expMask, KCEV2_EXP_MASK, scalarMask);
        Duplicate(mantissaMask, KCEV2_MANTISSA_MASK, scalarMask);
        Duplicate(zeroBytes, static_cast<uint8_t>(0), byteMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            for (uint16_t group = 0; group < groupCount; ++group) {
                uint32_t valid = KCEV2_GROUP_ELEMS;
                groupMask = UpdateMask<float>(valid);
                LoadB16AsFloat(x, inputData + row * inputStride + group * KCEV2_GROUP_ELEMS, groupMask);
                Abs(xAbs, x, groupMask);
                ReduceMax(amax, xAbs, groupMask);
                Maxs(scale, amax, 1.0e-4f, scalarMask);
                Muls(scale, scale, coeff, scalarMask);
                if constexpr (roundScale) {
                    ShiftRights(exp, reinterpret_cast<RegTensor<uint32_t> &>(scale),
                                static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), scalarMask);
                    And(exp, exp, expMask, scalarMask);
                    And(mantissa, reinterpret_cast<RegTensor<uint32_t> &>(scale), mantissaMask, scalarMask);
                    Compare<uint32_t, CMPMODE::NE>(hasMantissa, mantissa, zero, scalarMask);
                    Select(roundUp, one, zero, hasMantissa);
                    Adds(unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(exp), -127, scalarMask);
                    Add(unbiasedExp, unbiasedExp, reinterpret_cast<RegTensor<int32_t> &>(roundUp), scalarMask);
                    Adds(unbiasedExp, unbiasedExp, 127, scalarMask);
                    ShiftLefts(reinterpret_cast<RegTensor<int32_t> &>(scale), unbiasedExp,
                               static_cast<int16_t>(KCEV2_FAST_LOG_SHIFT), scalarMask);
                }
                Duplicate(scaleDup, scale, groupMask);
                Div(x, x, scaleDup, groupMask);
                Maxs(x, x, fp8Min, groupMask);
                Mins(x, x, fp8Max, groupMask);
                StoreFloatAsFp8(outputData + row * cacheCol + group * KCEV2_GROUP_ELEMS, x, groupMask);
                Cast<bfloat16_t, float, KCEV2_F32_TO_B16>(scaleBf16, scale, scalarMask);
                DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
                    outputScale + (row * cacheCol + dataCol) / sizeof(bfloat16_t) + group, scaleBf16, scalarMask);
            }
            const uint32_t padCount = cacheCol - concatCol;
            if (padCount > 0) {
                __local_mem__ uint8_t *pad = reinterpret_cast<__local_mem__ uint8_t *>(output.GetPhyAddr()) +
                    row * cacheCol + concatCol;
                DataCopyUnAlign(pad, zeroBytes, paddingReg, padCount);
                DataCopyUnAlignPost(pad, paddingReg, 0);
            }
        }
    }
}

constexpr int64_t KCEV2_FP4_OUT_ELEMS_PER_BLOCK = 64;
constexpr int64_t KCEV2_FP4_TWO = 2;

template <typename T>
__simd_vf__ inline void VFComputeMaxExpMxFp4(
    __ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCount,
    uint16_t loopNum, uint32_t vlForB16, uint32_t blocksPerVreg)
{
    using namespace AscendC::Reg;
    {
        RegTensor<T> even;
        RegTensor<T> odd;
        RegTensor<uint16_t> evenExp;
        RegTensor<uint16_t> oddExp;
        RegTensor<uint16_t> expMask;
        RegTensor<uint16_t> maxExp;
        MaskReg evenMask;
        MaskReg oddMask;
        UnalignRegForStore storeReg;
        Duplicate(expMask, KCEV2_BF16_EXP_MASK);
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            evenMask = UpdateMask<T>(totalCount);
            oddMask = UpdateMask<T>(totalCount);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                even, odd, srcAddr, vlForB16 * KCEV2_FP4_TWO);
            And(evenExp, reinterpret_cast<RegTensor<uint16_t> &>(even), expMask, evenMask);
            And(oddExp, reinterpret_cast<RegTensor<uint16_t> &>(odd), expMask, evenMask);
            Max(maxExp, evenExp, oddExp, evenMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(maxExp, maxExp, evenMask);
            StoreUnAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, maxExp, storeReg, blocksPerVreg);
        }
        StoreUnAlignPost(maxExpAddr, storeReg, 0);
    }
}

__simd_vf__ inline void VFComputeScaleMxFp4(
    __ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *scaleAddr,
    __ubuf__ uint16_t *halfScaleAddr, uint32_t scaleCount,
    uint16_t loopNum, uint32_t vlForB16)
{
    using namespace AscendC::Reg;
    {
        RegTensor<uint16_t> expMask;
        RegTensor<uint16_t> maxExp;
        RegTensor<uint16_t> fp4MaxExp;
        RegTensor<uint16_t> sharedExp;
        RegTensor<uint16_t> scaleValue;
        RegTensor<uint16_t> invBias;
        RegTensor<uint16_t> halfScale;
        RegTensor<uint16_t> zero;
        RegTensor<uint16_t> nan;
        RegTensor<uint16_t> specialInv;
        MaskReg finiteMask;
        MaskReg nonZeroMask;
        MaskReg clampMask;
        MaskReg specialMask;
        MaskReg scaleMask;
        Duplicate(expMask, KCEV2_BF16_EXP_MASK);
        Duplicate(fp4MaxExp, KCEV2_FP4_E2M1_MAX_EXP);
        Duplicate(invBias, KCEV2_BF16_INV_BIAS);
        Duplicate(zero, static_cast<uint16_t>(0));
        Duplicate(nan, KCEV2_BF16_NAN);
        Duplicate(specialInv, KCEV2_FP4_SPECIAL_INV);
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            scaleMask = UpdateMask<uint16_t>(scaleCount);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpAddr, vlForB16);
            Compare<uint16_t, CMPMODE::NE>(finiteMask, maxExp, expMask, scaleMask);
            Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, scaleMask);
            Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, fp4MaxExp, scaleMask);
            Select<uint16_t>(maxExp, fp4MaxExp, maxExp, clampMask);
            Sub(sharedExp, maxExp, fp4MaxExp, scaleMask);
            Select<uint16_t>(scaleValue, sharedExp, nan, finiteMask);
            Select<uint16_t>(scaleValue, scaleValue, zero, nonZeroMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(scaleAddr, scaleValue, vlForB16, scaleMask);

            Sub(halfScale, invBias, sharedExp, scaleMask);
            Select<uint16_t>(halfScale, halfScale, nan, finiteMask);
            Select<uint16_t>(halfScale, halfScale, zero, nonZeroMask);
            Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, invBias, scaleMask);
            Select<uint16_t>(halfScale, specialInv, halfScale, specialMask);
            StoreAlign<uint16_t, PostLiteral::POST_MODE_UPDATE>(halfScaleAddr, halfScale, vlForB16, scaleMask);
        }
    }
}

template <typename T>
__simd_vf__ inline void VFComputeDataMxFp4(
    __ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleAddr,
    __ubuf__ int8_t *outputAddr, uint32_t totalCount,
    uint16_t loopNum, uint32_t vlForB16, uint32_t blocksPerVreg)
{
    using namespace AscendC::Reg;
    {
        RegTensor<uint16_t> halfScale;
        RegTensor<T> even;
        RegTensor<T> odd;
        RegTensor<fp4x2_e2m1_t> evenFp4;
        RegTensor<fp4x2_e2m1_t> oddFp4;
        MaskReg dataMask;
        static constexpr AscendC::Reg::CastTrait fp4CastTrait = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
            AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            dataMask = UpdateMask<T>(totalCount);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                even, odd, srcAddr, vlForB16 * KCEV2_FP4_TWO);
            LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                halfScale, halfScaleAddr, blocksPerVreg);
            Mul(even, even, reinterpret_cast<RegTensor<T> &>(halfScale), dataMask);
            Mul(odd, odd, reinterpret_cast<RegTensor<T> &>(halfScale), dataMask);
            Interleave(even, odd, even, odd);
            AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(evenFp4, even, dataMask);
            AscendC::Reg::Cast<fp4x2_e2m1_t, T, fp4CastTrait>(oddFp4, odd, dataMask);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outputAddr, reinterpret_cast<RegTensor<int8_t> &>(evenFp4),
                KCEV2_FP4_OUT_ELEMS_PER_BLOCK, dataMask);
            StoreAlign<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                outputAddr, reinterpret_cast<RegTensor<int8_t> &>(oddFp4),
                KCEV2_FP4_OUT_ELEMS_PER_BLOCK, dataMask);
        }
    }
}

__aicore__ inline uint32_t ScaleRowBytesMxFp4(uint32_t scaleCol)
{
    return ((scaleCol * sizeof(bfloat16_t) + 63U) / 64U) * 64U;
}

__aicore__ inline void VFProcessMxFp4Verified(
    const LocalTensor<int8_t> &output, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<bfloat16_t> &input, const LocalTensor<uint16_t> &maxExp,
    const LocalTensor<uint16_t> &halfScale, uint16_t rowCount, uint32_t d)
{
    constexpr uint32_t vregBytes = KCEV2_VL_FP32 * sizeof(float);
    constexpr uint32_t ubBlockBytes = 32U;
    const uint32_t vlForB16 = vregBytes / sizeof(bfloat16_t);
    const uint32_t blocksPerVreg = vregBytes / ubBlockBytes;
    const uint32_t scaleCol = d / KCEV2_GROUP_ELEMS;
    const uint32_t xStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<int8_t>(d / 2);
    const uint32_t scaleStrideBytes = ScaleRowBytesMxFp4(scaleCol);
    const uint16_t xLoops = static_cast<uint16_t>(
        (d + vlForB16 * KCEV2_FP4_TWO - 1) / (vlForB16 * KCEV2_FP4_TWO));
    const uint16_t scaleLoops = static_cast<uint16_t>((scaleCol + vlForB16 - 1) / vlForB16);
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputAddr = reinterpret_cast<__ubuf__ int8_t *>(output.GetPhyAddr());
    auto *scaleByteAddr = reinterpret_cast<__ubuf__ uint8_t *>(scale.GetPhyAddr());
    auto *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExp.GetPhyAddr());
    auto *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScale.GetPhyAddr());
    for (uint16_t row = 0; row < rowCount; ++row) {
        VFComputeMaxExpMxFp4(inputAddr + row * xStride, maxExpAddr, d, xLoops, vlForB16, blocksPerVreg);
        VFComputeScaleMxFp4(maxExpAddr,
                            reinterpret_cast<__ubuf__ uint16_t *>(scaleByteAddr + row * scaleStrideBytes),
                            halfScaleAddr, scaleCol, scaleLoops, vlForB16);
        VFComputeDataMxFp4(inputAddr + row * xStride, halfScaleAddr,
                           outputAddr + row * outputStride, d, xLoops, vlForB16, blocksPerVreg);
    }
}

}  // namespace KvCompressEpilogV2Ops

#endif
