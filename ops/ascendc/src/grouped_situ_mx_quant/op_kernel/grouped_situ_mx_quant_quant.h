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
 * \file grouped_situ_mx_quant_quant.h
 * \brief MXFP8 (e4m3fn, group_size=32) tail-axis OCP quantization, ported
 *        verbatim (Reg API) from dynamic_mx_quant's DynamicMxQuantTailAxisFP8
 *        (SCALE_ALG=0) so the numerics are bit-identical to npu_dynamic_mx_quant.
 *        Operates on a UB block of already-activated bf16/fp16 values.
 *        VF bodies are __simd_vf__ functions, invoked via asc_vf_call.
 */

#ifndef GROUPED_SITU_MX_QUANT_QUANT_H
#define GROUPED_SITU_MX_QUANT_QUANT_H

#include "kernel_operator.h"

namespace grouped_situ {
using namespace AscendC;

// --- dav-3510 vector register geometry (VReg=256B, UB block=32B) ---
constexpr uint32_t VREG_SIZE = 256U;
constexpr uint32_t UB_BLK = 32U;
constexpr uint32_t VF_LEN_16 = VREG_SIZE / sizeof(uint16_t);        // 128
constexpr uint32_t VF_LEN_16_DOUBLE = VF_LEN_16 * 2U;               // 256
constexpr uint32_t VF_LEN_32 = VREG_SIZE / sizeof(uint32_t);        // 64
constexpr uint16_t ELEMENT_AFTER_REDUCE = VREG_SIZE / UB_BLK;       // 8

// --- constants mirrored from dynamic_mx_quant_common.h (e4m3 / OCP subset) ---
constexpr uint16_t BF16_MAX_EXP = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t BF16_NAN_CUSTOM = 0x7f81;
constexpr uint16_t BF16_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t BF16_SHR_NUM = 7;
constexpr uint16_t FP8_DEFAULT_MAX_EXP = 0x00ff;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;   // elem_emax >> 7 (BF16 E8M7)
constexpr uint16_t FP16_INVALID = 0x7c00;

// Pass 1: per-group (32) max exponent of |x|. bf16 path.
template <typename T>
__simd_vf__ inline void ComputeMaxExpOcpVF(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t loopNum2VF)
{
    Reg::RegTensor<T> x0;
    Reg::RegTensor<T> x1;
    Reg::RegTensor<uint16_t> xMaxExp;
    Reg::RegTensor<uint16_t> xExpExtract0;
    Reg::RegTensor<uint16_t> xExpExtract1;
    Reg::RegTensor<uint16_t> expMaskBF16;
    Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);
    Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::UnalignRegForStore ureg;

    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<uint16_t> xExpSelect0;
        Reg::RegTensor<uint16_t> xExpSelect1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, FP16_INVALID);
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpSelect0, (Reg::RegTensor<uint16_t>&)x0, invalidMaskFP16, Mask);
            Reg::And(xExpSelect1, (Reg::RegTensor<uint16_t>&)x1, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, xExpSelect0, invalidMaskFP16, Mask);
            Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, xExpSelect1, invalidMaskFP16, Mask);
            Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x0BF16, x0, Mask);
            Reg::Cast<bfloat16_t, half, castTraitHalf2Bf16>(x1BF16, x1, Mask);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, Mask);
            Reg::Select<uint16_t>(xExpExtract0, xExpExtract0, expMaskBF16, invalidDataMask0);
            Reg::Select<uint16_t>(xExpExtract1, xExpExtract1, expMaskBF16, invalidDataMask1);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceDataBlock<ReduceType::MAX>(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE);
        }
    } else {
        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, Mask);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceDataBlock<ReduceType::MAX>(xMaxExp, xMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE);
        }
    }
    Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
}

// Pass 2: group max exponent -> e8m0 scale (mxScale) + reciprocal scale (recipScale).
__simd_vf__ inline void ComputeScaleOcpVF(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                          __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t loopNum1VF,
                                          uint32_t totalScaleInUB)
{
    Reg::RegTensor<uint16_t> xMaxExp;
    Reg::RegTensor<uint16_t> sharedExp;
    Reg::RegTensor<uint16_t> scaleValue;
    Reg::RegTensor<uint16_t> halfScale;

    Reg::RegTensor<uint16_t> expMask;
    Reg::Duplicate(expMask, BF16_MAX_EXP);
    Reg::RegTensor<uint16_t> maxExpValue;
    Reg::Duplicate(maxExpValue, FP8_E4M3_MAX_EXP);
    Reg::RegTensor<uint16_t> scaleBias;
    Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
    Reg::RegTensor<uint16_t> fp8NanU16;
    Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
    Reg::RegTensor<uint16_t> zeroU16;
    Reg::Duplicate(zeroU16, 0);
    Reg::RegTensor<uint16_t> nanU16;
    Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
    Reg::RegTensor<uint16_t> specialExpU16;
    Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg preMaskScale;
    Reg::MaskReg invalidDataMask;
    Reg::MaskReg specialDataMask;

    for (uint16_t i = 0; i < loopNum1VF; i++) {
        preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
        Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale);          // INF/NAN
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);
        Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);

        Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
        Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);
        Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);
        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
            mxScaleLocalAddr, scaleValue, VF_LEN_32, preMaskScale);

        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
        Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
        Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
        Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
        Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
        Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);
        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(recipScaleLocalAddr, halfScale, VF_LEN_16,
                                                                      preMaskScale);
    }
}

// Pass 3: y = quantize(x * recipScale) -> e4m3fn, interleaved-packed store.
template <typename T, typename U>
__simd_vf__ inline void ComputeDataVF(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr,
                                      __ubuf__ int8_t* yLocalAddr, uint16_t loopNum2VF)
{
    Reg::MaskReg dataMask1 = Reg::CreateMask<T>();
    Reg::MaskReg dataMask2 = Reg::CreateMask<T>();
    Reg::MaskReg dataMask3 = Reg::CreateMask<T>();
    Reg::MaskReg dataMask4 = Reg::CreateMask<T>();
    Reg::MaskReg dataMask5 = Reg::CreateMask<U>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> halfScaleForMul;
    Reg::RegTensor<float> floatScaleForMul;
    Reg::RegTensor<T> x0;
    Reg::RegTensor<T> x1;
    Reg::RegTensor<float> x0ZeroFP32;
    Reg::RegTensor<float> x0OneFP32;
    Reg::RegTensor<float> x1ZeroFP32;
    Reg::RegTensor<float> x1OneFP32;
    Reg::RegTensor<U> x0ZeroFP8;
    Reg::RegTensor<U> x0OneFP8;
    Reg::RegTensor<U> x1ZeroFP8;
    Reg::RegTensor<U> x1OneFP8;

    static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitBf16ToFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTrait32to80 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTrait32to81 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTrait32to82 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTrait32to83 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    for (uint16_t i = 0; i < loopNum2VF; i++) {
        Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE);
        if constexpr (IsSameType<T, half>::value) {
            Reg::Cast<float, T, castTraitZero>(x0ZeroFP32, x0, dataMask1);
            Reg::Cast<float, T, castTraitOne>(x0OneFP32, x0, dataMask1);
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFloat>(floatScaleForMul,
                                                               (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, maskB16);
            Reg::Mul(x0ZeroFP32, x0ZeroFP32, floatScaleForMul, dataMask3);
            Reg::Mul(x0OneFP32, x0OneFP32, floatScaleForMul, dataMask4);
            Reg::Cast<float, T, castTraitZero>(x1ZeroFP32, x1, dataMask1);
            Reg::Cast<float, T, castTraitOne>(x1OneFP32, x1, dataMask1);
            Reg::Mul(x1ZeroFP32, x1ZeroFP32, floatScaleForMul, dataMask3);
            Reg::Mul(x1OneFP32, x1OneFP32, floatScaleForMul, dataMask4);
        } else {
            Reg::Mul(x0, x0, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            Reg::Mul(x1, x1, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
            Reg::Cast<float, T, castTraitZero>(x0ZeroFP32, x0, dataMask1);
            Reg::Cast<float, T, castTraitOne>(x0OneFP32, x0, dataMask1);
            Reg::Cast<float, T, castTraitZero>(x1ZeroFP32, x1, dataMask2);
            Reg::Cast<float, T, castTraitOne>(x1OneFP32, x1, dataMask2);
        }
        Reg::Cast<U, float, castTrait32to80>(x0ZeroFP8, x0ZeroFP32, dataMask3);
        Reg::Cast<U, float, castTrait32to81>(x1ZeroFP8, x1ZeroFP32, dataMask4);
        Reg::Cast<U, float, castTrait32to82>(x0OneFP8, x0OneFP32, dataMask3);
        Reg::Cast<U, float, castTrait32to83>(x1OneFP8, x1OneFP32, dataMask4);

        Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                 (Reg::RegTensor<uint8_t>&)x0OneFP8, dataMask5);
        Reg::Add((Reg::RegTensor<uint8_t>&)x1ZeroFP8, (Reg::RegTensor<uint8_t>&)x1ZeroFP8,
                 (Reg::RegTensor<uint8_t>&)x1OneFP8, dataMask5);
        Reg::Add((Reg::RegTensor<uint8_t>&)x0ZeroFP8, (Reg::RegTensor<uint8_t>&)x0ZeroFP8,
                 (Reg::RegTensor<uint8_t>&)x1ZeroFP8, dataMask5);

        Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
            yLocalAddr, (Reg::RegTensor<int8_t>&)x0ZeroFP8, VF_LEN_16_DOUBLE, dataMask5);
    }
}

}  // namespace grouped_situ
#endif  // GROUPED_SITU_MX_QUANT_QUANT_H
