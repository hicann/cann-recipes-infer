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
 * \file rotary_position_embedding_reg_a_and_b.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H
#define ROTARY_POSITION_EMBEDDING_REG_A_AND_B_H

#include "apply_rotary_pos_emb_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T, bool IsBoardCast>
class RotaryPositionEmbeddingAAndB {
public:
    __aicore__ inline RotaryPositionEmbeddingAAndB(){};
    __aicore__ inline ~RotaryPositionEmbeddingAAndB(){};
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace,
        const QuantRopeRegbaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    __aicore__ inline void ProcessInLoop(LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInCosAndSin(int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInNope(GlobalTensor<T>& source, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInRope(GlobalTensor<T>& source, int64_t bStart, int64_t bLength);
    __aicore__ inline void QuantizeNopeRegion(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t bLength, int64_t sliceStart);
    __aicore__ inline void ComputeAndQuantizeRope(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cos, LocalTensor<T>& sin,
        int64_t bLength, int64_t sliceLen);
    __aicore__ inline void CopyOutRegion(
        TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
        GlobalTensor<uint8_t>& target, int64_t bStart, int64_t bLength,
        int64_t dAlignH8, int64_t dCount, int64_t dStart);

private:
    constexpr static uint32_t COS_DB_BUFFER = IsBoardCast ? 1 : DOUBLE_BUFFER;

    TPipe* pipe_;

    GlobalTensor<T> qGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> qOutGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> nopeInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> ropeInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, COS_DB_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> nopeOutQueue_;
    TQue<QuePosition::VECOUT, 1> ropeOutQueue_;

    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;

    const QuantRopeRegbaseTilingData* tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignNope_ = 0;
    int64_t dAlignH8_ = 0;
    int64_t dAlignH8Nope_ = 0;
    int64_t sliceAlign_ = 0;
    float scale_ = 1.0f;

    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1;
    uint64_t ubCopyInStride = 0;
};

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Init(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace, const QuantRopeRegbaseTilingData* tilingData,
    TPipe* pipe)
{
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->InitAllGlobalBuffer(q, cos, sin, qOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllGlobalBuffer(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ T*)q);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ uint8_t*)qOut);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->D_ = this->tilingData_->D;
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->copyInQSplitCoef_ = 1;
    this->dAlign_ = ops::CeilAlign<int64_t>(D_, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignNope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignH8_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->dAlignH8Nope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->sliceAlign_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
        BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    this->scale_ = tilingData_->scale;

    this->pipe_->InitBuffer(this->nopeInQueue_, DOUBLE_BUFFER, ubFactorB_ * dAlignNope_ * sizeof(T));
    this->pipe_->InitBuffer(this->ropeInQueue_, DOUBLE_BUFFER, ubFactorB_ * sliceAlign_ * sizeof(T));
    this->pipe_->InitBuffer(this->nopeOutQueue_, 1, ubFactorB_ * dAlignH8Nope_ * sizeof(uint8_t));
    this->pipe_->InitBuffer(this->ropeOutQueue_, 1, ubFactorB_ * dAlignH8_ * sizeof(uint8_t));
    if constexpr (IsBoardCast) {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, sliceAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, COS_DB_BUFFER, ubFactorB_ * sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, COS_DB_BUFFER, ubFactorB_ * sliceAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::InitLoopParams()
{
    this->bBlockLength_ = tilingData_->blockFactorB;
    if (blockIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    this->bBlockStart_ = blockIdx_ * tilingData_->blockFactorB;
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::Process()
{
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBoardCast) {
        this->CopyInCosAndSin(0, 1);
        LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInLoop(
                cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
        }
        this->cosInQueue_.FreeTensor(cosUb);
        this->sinInQueue_.FreeTensor(sinUb);
    } else {
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(
                bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
            LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
            this->ProcessInLoop(
                cosUb, sinUb, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::ProcessInLoop(
    LocalTensor<T>& cos, LocalTensor<T>& sin, int64_t bUbStart, int64_t bUbLength)
{
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;

    CopyInNope(qGm_, bUbStart, bUbLength);
    LocalTensor<T> nopeInUb = this->nopeInQueue_.template DeQue<T>();
    CopyInRope(qGm_, bUbStart, bUbLength);
    LocalTensor<T> ropeInUb = this->ropeInQueue_.template DeQue<T>();

    LocalTensor<uint8_t> nopeUb = nopeOutQueue_.template AllocTensor<uint8_t>();
    QuantizeNopeRegion(nopeInUb, nopeUb, bUbLength, sliceStart);
    CopyOutRegion(nopeOutQueue_, nopeUb, qOutGm_, bUbStart, bUbLength, dAlignH8Nope_, sliceStart, 0);

    LocalTensor<uint8_t> ropeUb = ropeOutQueue_.template AllocTensor<uint8_t>();
    ComputeAndQuantizeRope(ropeInUb, ropeUb, cos, sin, bUbLength, sliceLen);
    CopyOutRegion(ropeOutQueue_, ropeUb, qOutGm_, bUbStart, bUbLength, dAlignH8_, sliceLen, sliceStart);

    this->nopeInQueue_.FreeTensor(nopeInUb);
    this->ropeInQueue_.FreeTensor(ropeInUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInCosAndSin(int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = this->cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = this->sinInQueue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / dSplitCoef_ / ops::BLOCK_BYTES;
    DataCopyPad(cosUb, this->cosGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb, this->sinGm_[bStart * tilingData_->sliceLength], copyExtParams, copyPadExtparams);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInNope(
    GlobalTensor<T>& source, int64_t bStart, int64_t bLength)
{
    LocalTensor<T> target = this->nopeInQueue_.template AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength;
    copyExtParams.blockLen = sliceStart * sizeof(T);
    copyExtParams.srcStride = (D_ - sliceStart) * sizeof(T);
    copyExtParams.dstStride = (dAlignNope_ - sliceStart) * sizeof(T) / ops::BLOCK_BYTES;
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(target, source[bStart * D_], copyExtParams, copyPadExtparams);
    this->nopeInQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyInRope(
    GlobalTensor<T>& source, int64_t bStart, int64_t bLength)
{
    LocalTensor<T> target = this->ropeInQueue_.template AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength;
    copyExtParams.blockLen = sliceLen * sizeof(T);
    copyExtParams.srcStride = (D_ - sliceLen) * sizeof(T);
    copyExtParams.dstStride = (sliceAlign_ - sliceLen) * sizeof(T) / ops::BLOCK_BYTES;
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(target, source[bStart * D_ + sliceStart], copyExtParams, copyPadExtparams);
    this->ropeInQueue_.template EnQue(target);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::QuantizeNopeRegion(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t bLength, int64_t sliceStart)
{
    LocalTensor<hifloat8_t> nopeH8 = nopeUb.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* nopeAddr = (__local_mem__ hifloat8_t*)nopeH8.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    uint16_t nopeLoop = ops::CeilDiv(static_cast<int32_t>(sliceStart), static_cast<int32_t>(VL_FLOAT32_SIZE));
    uint16_t rows = static_cast<uint16_t>(bLength);
    __VEC_SCOPE__ {
        RegTensor<float> xReg;
        RegTensor<hifloat8_t> h8Tmp;
        MaskReg preg;
        for (uint16_t i = 0; i < rows; i++) {
            uint32_t sreg = static_cast<uint32_t>(sliceStart);
            for (uint16_t j = 0; j < nopeLoop; j++) {
                preg = UpdateMask<float>(sreg);
                ops::LoadOneTensorForDtypeT<T>(inAddr, xReg, preg, i * dAlignNope_ + j * VL_FLOAT32_SIZE);
                Muls(xReg, xReg, scale_, preg);
                Cast<hifloat8_t, float, ops::castTraitF32toh8>(h8Tmp, xReg, preg);
                DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    nopeAddr + i * dAlignH8Nope_ + j * VL_FLOAT32_SIZE, h8Tmp, preg);
            }
        }
    }
    nopeOutQueue_.template EnQue(nopeUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::ComputeAndQuantizeRope(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cos, LocalTensor<T>& sin,
    int64_t bLength, int64_t sliceLen)
{
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    __local_mem__ T* sinAddr = (__local_mem__ T*)sin.GetPhyAddr();
    __local_mem__ T* cosAddr = (__local_mem__ T*)cos.GetPhyAddr();
    if constexpr (IsBoardCast) {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            HalfAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb, sliceLen,
            sliceAlign_, sliceAlign_, dAlignH8_, 1, bLength, scale_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            InterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb, sliceLen,
            sliceAlign_, sliceAlign_, dAlignH8_, 1, bLength, scale_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            QuarterAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb, sliceLen,
            sliceAlign_, sliceAlign_, dAlignH8_, 1, bLength, scale_);
        } else {
            DeepSeekInterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb, sliceLen,
            sliceAlign_, sliceAlign_, dAlignH8_, 1, bLength, scale_);
        }
    } else {
        if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
            BatchHalfAlignVF<T, IsBoardCast>(
                inAddr, cosAddr, sinAddr, ropeUb, bLength, 1, 1, sliceLen,
                sliceAlign_, sliceAlign_, dAlignH8_, ubFactorB_, 1, scale_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
            BatchInterleaveModeVF<T, IsBoardCast>(
                inAddr, cosAddr, sinAddr, ropeUb, bLength, 1, 1, sliceLen,
                sliceAlign_, sliceAlign_, dAlignH8_, ubFactorB_, 1, scale_);
        } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
            BatchQuarterAlignVF<T, IsBoardCast>(
                inAddr, cosAddr, sinAddr, ropeUb, bLength, 1, 1, sliceLen,
                sliceAlign_, sliceAlign_, dAlignH8_, ubFactorB_, 1, scale_);
        } else {
            BatchDeepSeekInterleaveModeVF<T, IsBoardCast>(
                inAddr, cosAddr, sinAddr, ropeUb, bLength, 1, 1, sliceLen,
                sliceAlign_, sliceAlign_, dAlignH8_, ubFactorB_, 1, scale_);
        }
    }
    ropeOutQueue_.template EnQue(ropeUb);
}

template <typename T, bool IsBoardCast>
__aicore__ inline void RotaryPositionEmbeddingAAndB<T, IsBoardCast>::CopyOutRegion(
    TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
    GlobalTensor<uint8_t>& target, int64_t bStart, int64_t bLength,
    int64_t dAlignH8, int64_t dCount, int64_t dStart)
{
    srcUb = outQueue.template DeQue<uint8_t>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = bLength;
    copyExtParams.blockLen = dCount * sizeof(uint8_t);
    copyExtParams.srcStride = (dAlignH8 - dCount) * sizeof(uint8_t) / ops::BLOCK_BYTES;
    copyExtParams.dstStride = (D_ - dCount) * sizeof(uint8_t);
    DataCopyPad(target[bStart * D_ + dStart], srcUb, copyExtParams);
    outQueue.FreeTensor(srcUb);
}

} // namespace PartialRotaryMulQuantOps

#endif