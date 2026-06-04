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
 * \file moe_v3_common.h
 * \brief
 */
#ifndef MOE_V3_COMMON_H_REGBASE
#define MOE_V3_COMMON_H_REGBASE

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t GATHER = 0;
constexpr int64_t SCATTER = 1;

constexpr uint16_t FLOAT_REG_TENSOR_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr int64_t GROUP_QUANT_SIZE = 128;
constexpr int32_t UB_BLOCK_SIZE = 32;
constexpr int32_t VFLEN_FP32 = 64;
constexpr float FP8_E5M2_MAX = 57344.0f;
constexpr float FP8_E4M3FN_MAX = 448.0f;
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t REPEAT_SIZE = 256;
constexpr int32_t VL_FP32 = 64;

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitU32toU8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait traitB16ToB32Layout0 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait traitB16ToB32Layout1 = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

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
    int32_t elemNum = UB_BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

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
__aicore__ inline void StoreOutputData(__local_mem__ T* dst,
                                       RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
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
__aicore__ inline void StoreOuputDataUnalign(RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst,
                                             MaskReg pregLoop, uint32_t postUpdateStride)
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

template <typename T, typename U>
__aicore__ inline void VFProcessSwigluMxFp8InvScale(const LocalTensor<float>& yLocal,
                                                    const LocalTensor<uint8_t>& scaleLocal,
                                                    const LocalTensor<float>& invScaleLocal,
                                                    const LocalTensor<T>& x0Local,
                                                    const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ uint8_t* scaleOriginLocalAddr = (__local_mem__ uint8_t*)scaleLocal.GetPhyAddr();
    __local_mem__ float* invScaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();
    __local_mem__ T* x0LocalAddr = (__local_mem__ T*)x0Local.GetPhyAddr();
    __local_mem__ uint8_t* scaleLocalAddr = scaleOriginLocalAddr;
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t curColNumAlignT = RoundUp<T>(curColNum);
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum);
    uint32_t scaleColNum = CeilDiv(curColNum, 32);
    uint32_t invScaleColNumAlign = RoundUp<float>(curColNumAlignT / 8);

    __VEC_SCOPE__
    {
        RegTensor<float> one;
        RegTensor<uint32_t> oneUint32;
        RegTensor<float> zero;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<T> x0;
        RegTensor<float> x0Layout0;
        RegTensor<float> x0Layout1;
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
        uint32_t sreg = (REPEAT_SIZE / sizeof(T)) / 32;
        uint32_t sreg1 = 4 * (REPEAT_SIZE / sizeof(T)) / 32;
        MaskReg pregMain0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMain1 = CreateMask<T, MaskPattern::ALL>();
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
            for (uint16_t j = 0; j < loopCount; j++) {
                DataCopy(x0, x0LocalAddr + i * curColNumAlignT + j * vlLen);
                Cast<float, T, traitB16ToB32Layout0>(x0Layout0, x0, pregMain1);
                Cast<float, T, traitB16ToB32Layout1>(x0Layout1, x0, pregMain1);

                // 合并奇偶位置
                Interleave(y0, y1, x0Layout0, x0Layout1);
                StoreOutputData<float>(yLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
                StoreOutputData<float>(yLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);

                Abs(x0Layout0, x0Layout0, pregMain0);
                Abs(x0Layout1, x0Layout1, pregMain0);

                // fp32场景，32个数对应4个Block；先做一次Max，接着ReduceMaxWithBlock，然后DeInterLeave并且Max，得到每4个Block的最大值
                Max(yMax0, x0Layout0, x0Layout1, pregMain0); // 4 --> 2
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
                Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge);
                Adds(scale5, scale4, 127, pregMerge); // sf_uint32

                Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, scale5, pregMerge);
                Pack(scale7, (RegTensor<uint32_t>&)tmp5);
                Pack(scale6, scale7);
                DataCopyUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(scaleLocalAddr, scale6, uReg, 4);

                Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
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
__aicore__ inline void VFProcessSwigluMxFp8Quant(const LocalTensor<T>& yQuantLocal, const LocalTensor<float>& yLocal,
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
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(invScale,
                         scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
                Mul(y, y, invScale, pregMain);
                StoreOutputData<T>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COMMON_H_REGBASE