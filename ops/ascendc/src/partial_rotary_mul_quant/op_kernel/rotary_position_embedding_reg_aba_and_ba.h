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
 * \file rotary_position_embedding_reg_aba_and_ba.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_ABA_AND_BA_H
#define ROTARY_POSITION_EMBEDDING_REG_ABA_AND_BA_H

#include "apply_rotary_pos_emb_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T, bool IsBBoardcast>
class RotaryPositionEmbeddingABAAndBA {
public:
    __aicore__ inline RotaryPositionEmbeddingABAAndBA(){};

    __aicore__ inline ~RotaryPositionEmbeddingABAAndBA(){};

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace,
        const QuantRopeRegbaseTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void Process();

private:
    // Init过程中使用的内部函数
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    // 各个层级的Process函数
    __aicore__ inline void ProcessInSLoop(
        int64_t sUbStart,
        int64_t sUbLength); // 第一重循环体，给定S范围，沿B轴进行遍历处理
    __aicore__ inline void ProcessInSBLoop(
        int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, LocalTensor<T>& cos,
        LocalTensor<T>& sin); // 第二重循环体，给定BS范围，沿Q和K的N轴进行遍历处理
    __aicore__ inline void ProcessInSBNLoop(
        int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
        int64_t nTotalSize, LocalTensor<T>& cos, LocalTensor<T>& sin, GlobalTensor<T>& in,
        GlobalTensor<uint8_t>& out);
    __aicore__ inline void CopyInCosAndSin(int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength);
    __aicore__ inline void CopyInNope(
        GlobalTensor<T>& source, int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
        int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void CopyInRope(
        GlobalTensor<T>& source, int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
        int64_t nLength, int64_t nTotalSize);
    __aicore__ inline void QuantizeNopeRegion(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t totalRows, int64_t sliceStart);
    __aicore__ inline void ComputeAndQuantizeRope(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cos, LocalTensor<T>& sin,
        int64_t sLength, int64_t bLength, int64_t nLength, int64_t sliceLen);
    __aicore__ inline void CopyOutRegion(
        TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
        GlobalTensor<uint8_t>& out, int64_t sUbStart, int64_t sUbLength,
        int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
        int64_t nTotalSize, int64_t dAlignH8, int64_t dCount, int64_t dStart);

private:
    TPipe* pipe_;

    // GlobalMemory
    GlobalTensor<T> qGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> qOutGm_;

    // UB
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> nopeInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> ropeInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> cosInQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> nopeOutQueue_;                // hifloat8 output for [0, sliceStart)
    TQue<QuePosition::VECOUT, 1> ropeOutQueue_;                // hifloat8 output for [sliceStart, sliceEnd)

    // Split core info
    int64_t blockIdx_ = 0;
    int64_t bBlockStart_ = 0;
    int64_t bBlockLength_ = 0;
    int64_t sBlockStart_ = 0;
    int64_t sBlockLength_ = 0;

    // TilingData
    const QuantRopeRegbaseTilingData* tilingData_;
    int64_t ubFactorB_ = 0;
    int64_t ubFactorS_ = 0;
    int64_t ubFactorN_ = 0;
    int64_t D_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignNope_ = 0;
    int64_t dAlignH8_ = 0;
    int64_t dAlignH8Nope_ = 0;
    int64_t sliceAlign_ = 0;
    float scale_ = 1.0f;

    // 拷贝参数
    uint8_t dSplitCoef_ = 1;
    uint8_t copyInQSplitCoef_ = 1; // 拷贝q时使用的splitCoef
    uint64_t ubCopyInStride = 0;   // 输入在ub中的stride，deepseek_interleave中不为0
};

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::Init(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut, GM_ADDR workspace, const QuantRopeRegbaseTilingData* tilingData,
    TPipe* pipe)
{
    this->tilingData_ = tilingData;
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    this->InitAllGlobalBuffer(q, cos, sin, qOut);
    this->InitAllBuffer();
    this->InitLoopParams();
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::InitAllGlobalBuffer(
    GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR qOut)
{
    this->qGm_.SetGlobalBuffer((__gm__ T*)q);
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin);
    this->qOutGm_.SetGlobalBuffer((__gm__ uint8_t*)qOut);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::InitAllBuffer()
{
    this->ubFactorB_ = this->tilingData_->ubFactorB;
    this->ubFactorS_ = this->tilingData_->ubFactorS;
    this->ubFactorN_ = this->tilingData_->ubFactorN;
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
    this->pipe_->InitBuffer(this->nopeInQueue_, DOUBLE_BUFFER,
        ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlignNope_ * sizeof(T));
    this->pipe_->InitBuffer(this->ropeInQueue_, DOUBLE_BUFFER,
        ubFactorB_ * ubFactorS_ * ubFactorN_ * sliceAlign_ * sizeof(T));
    this->pipe_->InitBuffer(this->nopeOutQueue_, 1,
        ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlignH8Nope_ * sizeof(uint8_t));
    this->pipe_->InitBuffer(this->ropeOutQueue_, 1, ubFactorB_ * ubFactorS_ * ubFactorN_ * dAlignH8_ * sizeof(uint8_t));
    if constexpr (IsBBoardcast) {
        this->pipe_->InitBuffer(this->cosInQueue_, DOUBLE_BUFFER, ubFactorS_ * sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, DOUBLE_BUFFER, ubFactorS_ * sliceAlign_ * sizeof(T));
    } else {
        this->pipe_->InitBuffer(this->cosInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * sliceAlign_ * sizeof(T));
        this->pipe_->InitBuffer(this->sinInQueue_, DOUBLE_BUFFER, ubFactorB_ * ubFactorS_ * sliceAlign_ * sizeof(T));
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::InitLoopParams()
{
    int64_t bIdx = blockIdx_ % tilingData_->blockNumB;
    int64_t sIdx = blockIdx_ / tilingData_->blockNumB;
    this->bBlockLength_ = tilingData_->blockFactorB;
    this->sBlockLength_ = tilingData_->blockFactorS;
    if (bIdx == tilingData_->blockNumB - 1 && tilingData_->B % tilingData_->blockFactorB != 0) {
        this->bBlockLength_ = tilingData_->B % tilingData_->blockFactorB;
    }
    if (sIdx == tilingData_->blockNumS - 1 && tilingData_->S % tilingData_->blockFactorS != 0) {
        this->sBlockLength_ = tilingData_->S % tilingData_->blockFactorS;
    }
    this->bBlockStart_ = bIdx * tilingData_->blockFactorB;
    this->sBlockStart_ = sIdx * tilingData_->blockFactorS;
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::Process()
{
    // 在S轴进行循环
    int64_t ubLoopCount = ops::CeilDiv(sBlockLength_, ubFactorS_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
        this->ProcessInSLoop(
            sBlockStart_ + ubLoopIdx * ubFactorS_,
            ubLoopIdx != ubLoopCount - 1 ? ubFactorS_ : sBlockLength_ - ubLoopIdx * ubFactorS_);
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::ProcessInSLoop(
    int64_t sUbStart, int64_t sUbLength)
{
    // 在B轴进行循环
    int64_t ubLoopCount = ops::CeilDiv(bBlockLength_, ubFactorB_);
    if constexpr (IsBBoardcast) {
        // cos和sin需要在B轴广播的情况
        this->CopyInCosAndSin(sUbStart, sUbLength, 0, 1);
        LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
        LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->ProcessInSBLoop(
                sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_, cosUb, sinUb);
        }
        this->sinInQueue_.FreeTensor(sinUb);
        this->cosInQueue_.FreeTensor(cosUb);
    } else {
        // sin和cos无需在B轴广播的情况
        for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoopCount; ubLoopIdx++) {
            this->CopyInCosAndSin(
                sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_);
            LocalTensor<T> cosUb = this->cosInQueue_.template DeQue<T>();
            LocalTensor<T> sinUb = this->sinInQueue_.template DeQue<T>();
            this->ProcessInSBLoop(
                sUbStart, sUbLength, bBlockStart_ + ubLoopIdx * ubFactorB_,
                ubLoopIdx != ubLoopCount - 1 ? ubFactorB_ : bBlockLength_ - ubLoopIdx * ubFactorB_, cosUb, sinUb);
            this->cosInQueue_.FreeTensor(cosUb);
            this->sinInQueue_.FreeTensor(sinUb);
        }
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::ProcessInSBLoop(
    int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, LocalTensor<T>& cos, LocalTensor<T>& sin)
{
    // 循环处理Q
    int64_t qUbLoopCount = ops::CeilDiv(tilingData_->N, ubFactorN_);
    for (int64_t ubLoopIdx = 0; ubLoopIdx < qUbLoopCount; ubLoopIdx++) {
        this->ProcessInSBNLoop(
            sUbStart, sUbLength, bUbStart, bUbLength, ubLoopIdx * ubFactorN_,
            ubLoopIdx != qUbLoopCount - 1 ? ubFactorN_ : tilingData_->N - ubLoopIdx * ubFactorN_, tilingData_->N, cos,
            sin, qGm_, qOutGm_);
    }
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::ProcessInSBNLoop(
    int64_t sUbStart, int64_t sUbLength, int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
    int64_t nTotalSize, LocalTensor<T>& cos, LocalTensor<T>& sin, GlobalTensor<T>& in, GlobalTensor<uint8_t>& out)
{
    CopyInNope(in, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
    LocalTensor<T> nopeInUb = this->nopeInQueue_.template DeQue<T>();
    CopyInRope(in, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength, nTotalSize);
    LocalTensor<T> ropeInUb = this->ropeInQueue_.template DeQue<T>();
    int64_t totalRows = sUbLength * bUbLength * nUbLength;
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;

    LocalTensor<uint8_t> nopeUb = nopeOutQueue_.template AllocTensor<uint8_t>();
    QuantizeNopeRegion(nopeInUb, nopeUb, totalRows, sliceStart);
    CopyOutRegion(nopeOutQueue_, nopeUb, out, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength,
                  nTotalSize, dAlignH8Nope_, sliceStart, 0);

    LocalTensor<uint8_t> ropeUb = ropeOutQueue_.template AllocTensor<uint8_t>();
    ComputeAndQuantizeRope(ropeInUb, ropeUb, cos, sin, sUbLength, bUbLength, nUbLength, sliceLen);
    CopyOutRegion(ropeOutQueue_, ropeUb, out, sUbStart, sUbLength, bUbStart, bUbLength, nUbStart, nUbLength,
                  nTotalSize, dAlignH8_, sliceLen, sliceStart);

    this->nopeInQueue_.FreeTensor(nopeInUb);
    this->ropeInQueue_.FreeTensor(ropeInUb);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::CopyInCosAndSin(
    int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength)
{
    LocalTensor<T> cosUb = this->cosInQueue_.template AllocTensor<T>();
    LocalTensor<T> sinUb = this->sinInQueue_.template AllocTensor<T>();
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = bLength;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1SrcStride = tilingData_->S * tilingData_->sliceLength * sizeof(T);
    loopParams.loop1DstStride = ubFactorS_ * sliceAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength * dSplitCoef_;
    copyExtParams.blockLen = tilingData_->sliceLength * sizeof(T) / dSplitCoef_;
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = (sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / dSplitCoef_ / ops::BLOCK_BYTES;
    DataCopyPad(cosUb,
        this->cosGm_[bStart * tilingData_->S * tilingData_->sliceLength + sStart * tilingData_->sliceLength],
        copyExtParams, copyPadExtparams);
    DataCopyPad(sinUb,
        this->sinGm_[bStart * tilingData_->S * tilingData_->sliceLength + sStart * tilingData_->sliceLength],
        copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->cosInQueue_.template EnQue(cosUb);
    this->sinInQueue_.template EnQue(sinUb);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::CopyInNope(
    GlobalTensor<T>& source, int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
    int64_t nLength, int64_t nTotalSize)
{
    LocalTensor<T> target = this->nopeInQueue_.template AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = nTotalSize * tilingData_->S * D_ * sizeof(T);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorS_ * dAlignNope_ * sizeof(T);
    loopParams.loop1SrcStride = tilingData_->S * D_ * sizeof(T);
    loopParams.loop1DstStride = ubFactorS_ * dAlignNope_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength;
    copyExtParams.blockLen = sliceStart * sizeof(T);
    copyExtParams.srcStride = (D_ - sliceStart) * sizeof(T);
    copyExtParams.dstStride = (dAlignNope_ - sliceStart) * sizeof(T) / ops::BLOCK_BYTES;
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(
        target, source[bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_],
        copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->nopeInQueue_.template EnQue(target);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::CopyInRope(
    GlobalTensor<T>& source, int64_t sStart, int64_t sLength, int64_t bStart, int64_t bLength, int64_t nStart,
    int64_t nLength, int64_t nTotalSize)
{
    LocalTensor<T> target = this->ropeInQueue_.template AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    LoopModeParams loopParams;
    loopParams.loop2Size = bLength;
    loopParams.loop1Size = nLength;
    loopParams.loop2SrcStride = nTotalSize * tilingData_->S * D_ * sizeof(T);
    loopParams.loop2DstStride = ubFactorN_ * ubFactorS_ * sliceAlign_ * sizeof(T);
    loopParams.loop1SrcStride = tilingData_->S * D_ * sizeof(T);
    loopParams.loop1DstStride = ubFactorS_ * sliceAlign_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = sLength;
    copyExtParams.blockLen = sliceLen * sizeof(T);
    copyExtParams.srcStride = (D_ - sliceLen) * sizeof(T);
    copyExtParams.dstStride = (sliceAlign_ - sliceLen) * sizeof(T) / ops::BLOCK_BYTES;
    DataCopyPadExtParams<T> copyPadExtparams;
    copyPadExtparams.isPad = false;
    copyPadExtparams.leftPadding = 0;
    copyPadExtparams.rightPadding = 0;
    copyPadExtparams.paddingValue = 0;
    DataCopyPad(
        target,
        source[bStart * nTotalSize * tilingData_->S * D_ + nStart * tilingData_->S * D_ + sStart * D_ + sliceStart],
        copyExtParams, copyPadExtparams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    this->ropeInQueue_.template EnQue(target);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::QuantizeNopeRegion(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t totalRows, int64_t sliceStart)
{
    LocalTensor<hifloat8_t> nopeH8 = nopeUb.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* nopeAddr = (__local_mem__ hifloat8_t*)nopeH8.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
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
    this->nopeOutQueue_.template EnQue(nopeUb);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::ComputeAndQuantizeRope(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cos, LocalTensor<T>& sin,
    int64_t sLength, int64_t bLength, int64_t nLength, int64_t sliceLen)
{
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        BatchHalfAlignVF<T, IsBBoardcast>(
            inAddr, (__local_mem__ T*)cos.GetPhyAddr(), (__local_mem__ T*)sin.GetPhyAddr(), ropeUb,
            sLength, bLength, nLength, sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, ubFactorS_, ubFactorN_, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        BatchInterleaveModeVF<T, IsBBoardcast>(
            inAddr, (__local_mem__ T*)cos.GetPhyAddr(), (__local_mem__ T*)sin.GetPhyAddr(), ropeUb,
            sLength, bLength, nLength, sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, ubFactorS_, ubFactorN_, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        BatchQuarterAlignVF<T, IsBBoardcast>(
            inAddr, (__local_mem__ T*)cos.GetPhyAddr(), (__local_mem__ T*)sin.GetPhyAddr(), ropeUb,
            sLength, bLength, nLength, sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, ubFactorS_, ubFactorN_, scale_);
    } else {
        BatchDeepSeekInterleaveModeVF<T, IsBBoardcast>(
            inAddr, (__local_mem__ T*)cos.GetPhyAddr(), (__local_mem__ T*)sin.GetPhyAddr(), ropeUb,
            sLength, bLength, nLength, sliceLen, sliceAlign_, sliceAlign_, dAlignH8_, ubFactorS_, ubFactorN_, scale_);
    }
    this->ropeOutQueue_.template EnQue(ropeUb);
}

template <typename T, bool IsBBoardcast>
__aicore__ inline void RotaryPositionEmbeddingABAAndBA<T, IsBBoardcast>::CopyOutRegion(
    TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
    GlobalTensor<uint8_t>& out, int64_t sUbStart, int64_t sUbLength,
    int64_t bUbStart, int64_t bUbLength, int64_t nUbStart, int64_t nUbLength,
    int64_t nTotalSize, int64_t dAlignH8, int64_t dCount, int64_t dStart)
{
    srcUb = outQueue.template DeQue<uint8_t>();
    LoopModeParams lp;
    lp.loop2Size = bUbLength; lp.loop1Size = nUbLength;
    lp.loop2DstStride = nTotalSize * tilingData_->S * D_ * sizeof(uint8_t);
    lp.loop2SrcStride = ubFactorN_ * ubFactorS_ * dAlignH8 * sizeof(uint8_t);
    lp.loop1DstStride = tilingData_->S * D_ * sizeof(uint8_t);
    lp.loop1SrcStride = ubFactorS_ * dAlignH8 * sizeof(uint8_t);
    SetLoopModePara(lp, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams cpExt;
    cpExt.blockCount = sUbLength;
    cpExt.blockLen = dCount * sizeof(uint8_t);
    cpExt.srcStride = (dAlignH8 - dCount) * sizeof(uint8_t) / ops::BLOCK_BYTES;
    cpExt.dstStride = (D_ - dCount) * sizeof(uint8_t);
    DataCopyPad(
        out[bUbStart * nTotalSize * tilingData_->S * D_ + nUbStart * tilingData_->S * D_ + sUbStart * D_ + dStart],
        srcUb, cpExt);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outQueue.FreeTensor(srcUb);
}

} // namespace PartialRotaryMulQuantOps

#endif