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
 * \file moe_v3_gather_mxfp8_gather_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_MXFP8_QUANT_GATHER_H_REGBASE
#define MOE_V3_GATHER_MXFP8_QUANT_GATHER_H_REGBASE

#include "moe_v3_common.h"
#include "platform_util.h"
#include "math_util.h"

namespace MoeInitRoutingOptimize {
using namespace AscendC;
using namespace MoeInitRoutingV3;

// MX量化一个块的元素数量为32个，即1个scale对应32个x的元素
constexpr int64_t MX_BLOCK_SIZE = 32LL;
// 一个VL放多少fp32->fp8的元素个数，即按fp32算的一个VL能放几个元素
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
// fp16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
// bf16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
// bfloat16的nan值（与inf值不同）
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
// 对于x的目标类型为e5m2的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
// 对于x的目标类型为e4m3的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
// e8m0的inf/nan值（按定义应该是nan值）
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
// e8m0的极小值，用于写0
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
// bf16指数位的偏移位数
constexpr int16_t BF16_EXP_SHR_BITS = 7;
// 用于计算halfScale=BF16_EXP_INVSUB-sharedExp，得到的halfScale就是1/realScale，可以用于量化xQuant=x*halfScale
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;

template <typename T, typename U>
class MoeGatherOutMxfp8QuantGather {
public:
    __aicore__ inline MoeGatherOutMxfp8QuantGather(){};
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

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<uint8_t> expandedScaleOutGm_;
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
    uint16_t lowerBoundOfB16MaxExp_ = 0;

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
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                          GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                          GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                          const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr);
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(k_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    pipe_->InitBuffer(mxScaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(float)));
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2); // CeilDiv(h, 32)后再向上到2的倍数（偶数）
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
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        for (int64_t loop = 0; loop < coreRows_; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandMXQuant(loop);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::CopyInExpandedExpertIdx(int64_t nIndex)
{
    indicesOffset_ = (blockIdx_ * perCoreRow_ + nIndex) * k_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::CopyExpandedXandMXQuant(int64_t nIndex)
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
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                             int64_t loopScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);//(N*K,H)

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);//(N*K, M)

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}


template <typename T, typename U>
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = loopCols % MX_BLOCK_SIZE;
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
__aicore__ inline void MoeGatherOutMxfp8QuantGather<T, U>::Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    // alloc outputs
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    LocalTensor<U> xQuantRecastLocal = xQuantLocal.ReinterpretCast<U>();
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint8_t>();
    // get tmp buffers
    LocalTensor<float> maxExpLocal = maxExpBuffer_.Get<float>();
    LocalTensor<float> invScaleLocal = invScaleBuffer_.Get<float>();

    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ - 1) / vlForB16_;
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VFProcessSwigluMxFp8InvScale<T, U>(maxExpLocal, mxScaleLocal, invScaleLocal, xLocal, 1, xElemNum);
    VFProcessSwigluMxFp8Quant<U>(xQuantRecastLocal, maxExpLocal, invScaleLocal, 1, xElemNum);

    // free input
    xInQueue_.FreeTensor(xLocal);
    // enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    mxScaleOutQueue_.EnQue(mxScaleLocal);
}
} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_MXFP8_QUANT_GATHER_H_REGBASE