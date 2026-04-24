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
 * \file moe_v3_gather_fp8_group_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_FP8_GROUP_QUANT_GATHER_H_REGBASE
#define MOE_V3_GATHER_FP8_GROUP_QUANT_GATHER_H_REGBASE

#include "moe_v3_common.h"
#include "platform_util.h"

namespace MoeInitRoutingOptimize {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace MoeInitRoutingV3;

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
    int32_t elemNum = UB_BLOCK_SIZE / sizeof(T);
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

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T, typename U>
class MoeGatherOutFp8GroupQuantGather {
public:
    __aicore__ inline MoeGatherOutFp8GroupQuantGather(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t nIndex);
    __aicore__ inline void CopyExpandedXandMXQuant(int64_t nIndex);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);
    __aicore__ inline void DoGroupQuant(LocalTensor<U>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<T>& xLocal, float coeff, uint16_t curRowNum, uint32_t curColNum);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<U> expandedXOutGm_;
    GlobalTensor<float> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t validScaleCols_; // 一个token实际有意义的scale有多少个元素（列）
    int64_t scaleCols_; // 一个token的scale有多少个元素（列），即CeilAlign(CeliDiv(h,32),2)，在actualScaleCols_为奇数时，scaleCols_=validScaleCols_+1
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopValidScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    // 会赋值为计算maxExp时，maxExp对应不同的目标fp8类型U的下限
    float lowerBoundOfB16MaxExp_ = 0;

    // 一个RegTensor的长度Bytes
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // ub的一个块大小，通常是32B
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    // 一个RegTensor能塞下多少T(B16)类型的元素
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    // 一个RegTensor有几个UbBlock，即ReduceWithBlock后的元素个数（256B/32B=8）
    // 这里如果ubBlockSize_获取的值为0，应该让kernel挂掉
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_; // 8个元素
    // 每个ubBlock对应多少T类型的元素
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                          GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                          GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                          const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ U *)expandedXAddr);
    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr);
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ float *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(U)) * 4);
    pipe_->InitBuffer(scaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(float)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(T)));
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = static_cast<float>(1.0) / FP8_E4M3FN_MAX;
    } else {
        lowerBoundOfB16MaxExp_ = static_cast<float>(1.0) / FP8_E5M2_MAX;
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    scaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, GROUP_QUANT_SIZE);
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    perCoreRow_ = Ceil(n_, tilingData->coreNum);
    needCoreNum_ = Ceil(n_, perCoreRow_);
    int64_t lastCoreDealN = n_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreDealN;
    } else {
        coreRows_ = perCoreRow_;
    }

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / GROUP_QUANT_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        for (int64_t loop = 0; loop < coreRows_; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandMXQuant(loop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::CopyInExpandedExpertIdx(int64_t nIndex)
{
    indicesOffset_ = (blockIdx_ * perCoreRow_ + nIndex) * k_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::CopyExpandedXandMXQuant(int64_t nIndex)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t j = 0; j < colLoops_; j++) {
        // 每行切分成cols，按cols读入-计算量化-拷出
        int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
        uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
        uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
        int32_t srcIdx = (blockIdx_ * perCoreRow_ + nIndex) * cols_;
        CopyIn(srcIdx, j, loopCols);
        Compute(loopCols, loopScaleCols, loopValidScaleCols);
        for (int64_t k = 0; k < k_; k++) {
            int32_t indicesIdx = indicesLocal.GetValue(k);
            if (indicesIdx < 0) {
                continue;
            }
            CopyOut(indicesIdx, j, loopCols, loopScaleCols);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = loopCols % GROUP_QUANT_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcIdx + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    // alloc outputs
    LocalTensor<U> xQuantLocal = xQuantOutQueue_.AllocTensor<U>();
    LocalTensor<float> scaleLocal = scaleOutQueue_.AllocTensor<float>();
    // get tmp buffers
    DoGroupQuant(xQuantLocal, scaleLocal, xLocal, lowerBoundOfB16MaxExp_, 1, xElemNum);
    // free input
    xInQueue_.FreeTensor(xLocal);
    // enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    scaleOutQueue_.EnQue(scaleLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                             int64_t loopScaleCols)
{
    LocalTensor<float> scaleLocal = scaleOutQueue_.DeQue<float>();
    LocalTensor<U> outLocal = xQuantOutQueue_.DeQue<U>();

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(U)), 0, 0, 0};
    DataCopyPad<U>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(float)), 0, 0, 0};
    DataCopyPad(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], scaleLocal,
                         copyScaleParams);

    xQuantOutQueue_.FreeTensor(outLocal);
    scaleOutQueue_.FreeTensor(scaleLocal);
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
__aicore__ inline void StoreOutputData(__local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
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
__aicore__ inline void StoreOuputDataUnalign(RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst, MaskReg pregLoop, uint32_t postUpdateStride)
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
__aicore__ inline void MoeGatherOutFp8GroupQuantGather<T, U>::DoGroupQuant(
    LocalTensor<U>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<T>& xLocal, float coeff,
    uint16_t curRowNum, uint32_t curColNum)
{
    __local_mem__ U* yLocalAddr = (__local_mem__ U*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t loopCount = CeilDiv(curColNum, VFLEN_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<U>(curColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailRemider = curColNum - (loopCount - 1) * VFLEN_FP32;
    uint32_t scaleRowNum = (curRowNum + 128 - 1) / 128;
    uint32_t sregNum = loopCountReminder == 0 ? curRowNum - loopCountFoldTwo * VFLEN_FP32 : loopCountFoldTwo * VFLEN_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> xLeft;
        RegTensor<float> xRight;
        RegTensor<float> x1Left;
        RegTensor<float> x1Right;
        RegTensor<float> xAbsLeft;
        RegTensor<float> xAbsRight;
        RegTensor<float> xMax;
        RegTensor<float> xReduceMax;
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
        MaskReg pregMain = CreateMask<float>();
        Duplicate(one, static_cast<float>(1.0f), pregLoop);
        Duplicate(coeffReg, maxValueInt, pregLoop);
        Duplicate(zero, 0.0f);
        Duplicate(inf, 1.0f);
        Div<float, &mode>(inf, inf, zero, pregLoop);
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg compareLeft;
 	    MaskReg compareRight;
        MaskReg compareScalar;
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCountFoldTwo; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T>(xLeft, xLocalAddr, pregMain, 2 * j * VFLEN_FP32 + i * curColNumAlign);
                LoadInputData<T>(xRight, xLocalAddr, pregLoop, (2 * j + 1) * VFLEN_FP32 + i * curColNumAlign);
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
                Div<float, &mode>(xAbsLeft, xLeft, dupScale, pregMain);
                Muls(x1Left, xAbsLeft, 0.0f, pregMain);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregMain);
                Select(xLeft, xLeft, xAbsLeft, compareLeft);
                Div<float, &mode>(xAbsRight, xRight, dupScale, pregLoop);
                Muls(x1Right, xAbsRight, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareRight, x1Right, x1Right, pregLoop);
                Select(xRight, xRight, xAbsRight, compareRight);
                StoreOutputData<U>(yLocalAddr, xLeft, pregMain, 2 * j * VFLEN_FP32 + i * dstCurColNumAlign);
                StoreOutputData<U>(yLocalAddr, xRight, pregLoop, (2 * j + 1) * VFLEN_FP32 + i * dstCurColNumAlign);
            }
            // 处理尾块, 这里只有一个for循环
            pregLoop = UpdateMask<float>(tailRemider);
            for (uint16_t j = 0; j < loopCountReminder; j++) {
                LoadInputData<T>(xLeft, xLocalAddr, pregLoop, loopCountFoldTwo * 2 * VFLEN_FP32 + i * curColNumAlign);
                Abs(xAbsLeft, xLeft, pregLoop);
                ReduceMax(scale, xAbsLeft, pregLoop);
                CompareScalar<float, CMPMODE::NE>(compareScalar, scale, (float)0.0, preg1);
                Mul(scale, scale, (RegTensor<float>&)coeffReg, compareScalar);
                Min(scale, scale, inf, preg1);
                Duplicate(dupScale, scale, pregLoop);
                StoreOuputDataUnalign(scale, scaleLocalAddr, uScale, preg1, 1);
                Div<float, &mode>(xAbsLeft, xLeft, dupScale, pregLoop);
                Muls(x1Left, xAbsLeft, 0.0f, pregLoop);
                Compare<float, CMPMODE::NE>(compareLeft, x1Left, x1Left, pregLoop);
                Select(xLeft, xLeft, xAbsLeft, compareLeft);
                StoreOutputData(yLocalAddr, xLeft, pregLoop, loopCountFoldTwo * VFLEN_FP32 + i * dstCurColNumAlign);
            }
        }
        DataCopyUnAlignPost(scaleLocalAddr, uScale, 0);
    }
}

} // namespace MoeInitRoutingOptimize
#endif // MOE_V3_GATHER_FP8_GROUP_QUANT_GATHER_H_REGBASE