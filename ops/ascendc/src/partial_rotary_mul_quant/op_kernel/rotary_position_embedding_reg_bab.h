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
 * \file rotary_position_embedding_reg_bab.h
 * \brief
 */

#ifndef ROTARY_POSITION_EMBEDDING_REG_BAB_H
#define ROTARY_POSITION_EMBEDDING_REG_BAB_H

#include "apply_rotary_pos_emb_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T>
class RotaryPositionEmbeddingBAB {
public:
    __aicore__ inline RotaryPositionEmbeddingBAB(TPipe* pipe, const QuantRopeRegbaseTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y);
    __aicore__ inline void Process();

private:
    constexpr static int32_t bufferNum = 2;
    const QuantRopeRegbaseTilingData* tilingData_;
    TPipe* pipe_;
    int64_t blockIdx_ = 0;
    int64_t dSplitCoef_ = 1;
    uint32_t dSplitSize_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignNope_ = 0;
    int64_t bIdx_ = 0;
    int64_t sIdx_ = 0;
    int64_t bNum_ = 0;
    int64_t sNum_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> yOutGm_;

    TQue<QuePosition::VECIN, bufferNum> nopeInQue_;
    TQue<QuePosition::VECIN, bufferNum> ropeInQue_;
    TQue<QuePosition::VECIN, bufferNum> cosInQue_;
    TQue<QuePosition::VECIN, bufferNum> sinInQue_;
    TQue<QuePosition::VECOUT, 1> nopeOutQue_;
    TQue<QuePosition::VECOUT, 1> ropeOutQue_;

private:
    int64_t dAlignH8_ = 0;
    int64_t dAlignH8Nope_ = 0;
    int64_t sliceAlign_ = 0;
    float scale_ = 1.0f;
    __aicore__ inline void PrePareParams();
    __aicore__ inline void ProcessNLoop(const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum);
    __aicore__ inline void ProcessN(
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const uint32_t bIdx, const uint32_t sIdx,
        const uint32_t currSNum);
    __aicore__ inline void CopyInNope(LocalTensor<T>& xTensor, int64_t offset, int64_t currSNum, int64_t currDNum);
    __aicore__ inline void CopyInRope(LocalTensor<T>& xTensor, int64_t offset, int64_t currSNum, int64_t currDNum);
    __aicore__ inline void QuantizeNopeRegion(
        LocalTensor<T>& xTensor, LocalTensor<uint8_t>& nopeUb, int64_t totalRows, int64_t sliceStart);
    __aicore__ inline void ComputeAndQuantizeRope(
        LocalTensor<T>& xTensor, LocalTensor<uint8_t>& ropeUb,
        const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
        int64_t currSNum, int64_t currDNum, int64_t sliceLen);
    __aicore__ inline void CopyOutRegion(
        TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
        int64_t offset, int64_t currSNum, int64_t currDNum,
        int64_t dAlignH8, int64_t dCount, int64_t dStart);
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y)
{
    this->blockIdx_ = GetBlockIdx();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF) ||
        tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE)) {
        this->dSplitCoef_ = HALF_INTERLEAVE_COEF;
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        this->dSplitCoef_ = QUARTER_MODE_COEF;
    }
    this->dSplitSize_ = tilingData_->sliceLength / dSplitCoef_ * sizeof(T);
    this->dAlign_ = ops::CeilAlign<int64_t>(tilingData_->D, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignNope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(T));
    this->dAlignH8_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->dAlignH8Nope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, BLOCK_TYPE_SIZE / sizeof(uint8_t));
    this->sliceAlign_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength / dSplitCoef_,
        BLOCK_TYPE_SIZE / sizeof(T)) * dSplitCoef_;
    this->scale_ = tilingData_->scale;
    ubFactorN_ = tilingData_->ubFactorN;
    ubFactorS_ = tilingData_->ubFactorS;
    this->xGm_.SetGlobalBuffer((__gm__ T*)x);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->yOutGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    this->pipe_->InitBuffer(nopeInQue_, bufferNum, ubFactorS_ * ubFactorN_ * dAlignNope_ * sizeof(T));
    this->pipe_->InitBuffer(ropeInQue_, bufferNum, ubFactorS_ * ubFactorN_ * sliceAlign_ * sizeof(T));
    this->pipe_->InitBuffer(cosInQue_, bufferNum, ubFactorS_ * sliceAlign_ * sizeof(T));
    this->pipe_->InitBuffer(sinInQue_, bufferNum, ubFactorS_ * sliceAlign_ * sizeof(T));
    this->pipe_->InitBuffer(nopeOutQue_, 1, ubFactorS_ * ubFactorN_ * dAlignH8Nope_ * sizeof(uint8_t));
    this->pipe_->InitBuffer(ropeOutQue_, 1, ubFactorS_ * ubFactorN_ * dAlignH8_ * sizeof(uint8_t));
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::PrePareParams()
{
    bIdx_ = blockIdx_ % tilingData_->blockNumB;
    sIdx_ = blockIdx_ / tilingData_->blockNumB;
    bNum_ = tilingData_->blockFactorB;
    sNum_ = tilingData_->blockFactorS;
    if (bIdx_ == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        bNum_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx_ == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        sNum_ = tilingData_->S % tilingData_->blockFactorS;
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::Process()
{
    PrePareParams();
    uint32_t bIdxStart = bIdx_ * tilingData_->blockFactorB;
    for (uint32_t bIdx = bIdxStart; bIdx < bIdxStart + bNum_; bIdx++) {
        uint32_t sIdxStart = sIdx_ * tilingData_->blockFactorS;
        uint32_t sLoopCnt = ops::CeilDiv(sNum_, ubFactorS_);
        for (uint32_t loopIdx = 0; loopIdx < sLoopCnt; loopIdx++) {
            uint32_t currSNum = (loopIdx != sLoopCnt - 1) ? ubFactorS_ : sNum_ - loopIdx * ubFactorS_;
            ProcessNLoop(bIdx, sIdxStart + loopIdx * ubFactorS_, currSNum);
        }
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::ProcessNLoop(
    const uint32_t bIdx, const uint32_t sIdx, const uint32_t currSNum)
{
    LocalTensor<T> sinTensor = sinInQue_.AllocTensor<T>();
    LocalTensor<T> cosTensor = cosInQue_.AllocTensor<T>();
    int64_t offset = sIdx * tilingData_->sliceLength;
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(currSNum * dSplitCoef_),
        dSplitSize_,
        0,
        static_cast<uint32_t>((sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / dSplitCoef_ / ops::BLOCK_BYTES),
        0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(sinTensor, sinGm_[offset], copyParams, padParams);
    DataCopyPad(cosTensor, cosGm_[offset], copyParams, padParams);
    sinInQue_.EnQue(sinTensor);
    cosInQue_.EnQue(cosTensor);
    sinTensor = sinInQue_.DeQue<T>();
    cosTensor = cosInQue_.DeQue<T>();
    ProcessN(sinTensor, cosTensor, bIdx, sIdx, currSNum);
    sinInQue_.FreeTensor(sinTensor);
    cosInQue_.FreeTensor(cosTensor);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::ProcessN(
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor, const uint32_t bIdx, const uint32_t sIdx,
    const uint32_t currSNum)
{
    int64_t baseOffset = (bIdx * tilingData_->S + sIdx) * tilingData_->N * tilingData_->D;
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    for (uint32_t idxN = 0; idxN < tilingData_->ubLoopNumN; idxN++) {
        int64_t currDNum = (idxN == tilingData_->ubLoopNumN - 1) ? tilingData_->ubTailFactorN : ubFactorN_;
        int64_t offset = baseOffset + idxN * ubFactorN_ * tilingData_->D;
        int64_t totalRows = currSNum * currDNum;

        LocalTensor<T> nopeTensor = nopeInQue_.AllocTensor<T>();
        LocalTensor<T> ropeTensor = ropeInQue_.AllocTensor<T>();
        CopyInNope(nopeTensor, offset, currSNum, currDNum);
        CopyInRope(ropeTensor, offset, currSNum, currDNum);
        nopeInQue_.EnQue(nopeTensor);
        ropeInQue_.EnQue(ropeTensor);
        nopeTensor = nopeInQue_.DeQue<T>();
        ropeTensor = ropeInQue_.DeQue<T>();

        LocalTensor<uint8_t> nopeUb = nopeOutQue_.AllocTensor<uint8_t>();
        QuantizeNopeRegion(nopeTensor, nopeUb, totalRows, sliceStart);
        CopyOutRegion(nopeOutQue_, nopeUb, offset, currSNum, currDNum, dAlignH8Nope_, sliceStart, 0);

        LocalTensor<uint8_t> ropeUb = ropeOutQue_.AllocTensor<uint8_t>();
        ComputeAndQuantizeRope(ropeTensor, ropeUb, sinTensor, cosTensor, currSNum, currDNum, sliceLen);
        CopyOutRegion(ropeOutQue_, ropeUb, offset, currSNum, currDNum, dAlignH8_, sliceLen, sliceStart);

        nopeInQue_.FreeTensor(nopeTensor);
        ropeInQue_.FreeTensor(ropeTensor);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::CopyInNope(
    LocalTensor<T>& xTensor, int64_t offset, int64_t currSNum, int64_t currDNum)
{
    int64_t sliceStart = tilingData_->sliceStart;
    DataCopyExtParams copyInParams{
        static_cast<uint16_t>(currSNum * currDNum),
        static_cast<uint32_t>(sliceStart * sizeof(T)),
        static_cast<uint32_t>((tilingData_->D - sliceStart) * sizeof(T)),
        static_cast<uint32_t>((dAlignNope_ - sliceStart) * sizeof(T) / ops::BLOCK_BYTES), 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xTensor, xGm_[offset], copyInParams, padParams);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::CopyInRope(
    LocalTensor<T>& xTensor, int64_t offset, int64_t currSNum, int64_t currDNum)
{
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    DataCopyExtParams copyInParams{static_cast<uint16_t>(currSNum * currDNum),
        static_cast<uint32_t>(sliceLen * sizeof(T)),
        static_cast<uint32_t>((tilingData_->D - sliceLen) * sizeof(T)),
        static_cast<uint32_t>((sliceAlign_ - sliceLen) * sizeof(T) / ops::BLOCK_BYTES), 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xTensor,
        xGm_[static_cast<uint64_t>(offset) + static_cast<uint64_t>(sliceStart)], copyInParams, padParams);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::QuantizeNopeRegion(
    LocalTensor<T>& xTensor, LocalTensor<uint8_t>& nopeUb, int64_t totalRows, int64_t sliceStart)
{
    LocalTensor<hifloat8_t> nopeH8 = nopeUb.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* nopeAddr = (__local_mem__ hifloat8_t*)nopeH8.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
    uint16_t nopeLoop = ops::CeilDiv(static_cast<int32_t>(sliceStart), static_cast<int32_t>(VL_FLOAT32_SIZE));
    uint16_t rows = static_cast<uint16_t>(totalRows);
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
    nopeOutQue_.template EnQue(nopeUb);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::ComputeAndQuantizeRope(
    LocalTensor<T>& xTensor, LocalTensor<uint8_t>& ropeUb,
    const LocalTensor<T>& sinTensor, const LocalTensor<T>& cosTensor,
    int64_t currSNum, int64_t currDNum, int64_t sliceLen)
{
    __local_mem__ T* inAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
    __local_mem__ T* sinAddr = (__local_mem__ T*)sinTensor.GetPhyAddr();
    __local_mem__ T* cosAddr = (__local_mem__ T*)cosTensor.GetPhyAddr();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, currSNum, currDNum, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, currSNum, currDNum, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, currSNum, currDNum, scale_);
    } else {
        DeepSeekInterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, currSNum, currDNum, scale_);
    }
    ropeOutQue_.template EnQue(ropeUb);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingBAB<T>::CopyOutRegion(
    TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
    int64_t offset, int64_t currSNum, int64_t currDNum,
    int64_t dAlignH8, int64_t dCount, int64_t dStart)
{
    srcUb = outQueue.template DeQue<uint8_t>();
    DataCopyExtParams copyOutParams{
        static_cast<uint16_t>(currSNum * currDNum),
        static_cast<uint32_t>(dCount * sizeof(uint8_t)),
        static_cast<uint32_t>((dAlignH8 - dCount) * sizeof(uint8_t) / ops::BLOCK_BYTES),
        static_cast<uint32_t>((tilingData_->D - dCount) * sizeof(uint8_t)), 0};
    DataCopyPad(yOutGm_[static_cast<uint64_t>(offset) + static_cast<uint64_t>(dStart)], srcUb, copyOutParams);
    outQueue.FreeTensor(srcUb);
}

} // namespace PartialRotaryMulQuantOps
#endif // ROTARY_POSITION_EMBEDDING_REG_BAB_H