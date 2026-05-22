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
 * \file rotary_position_embedding_reg_ab.h
 * \brief
 */
#ifndef ROTARY_POSITION_EMBEDDING_REG_AB_H
#define ROTARY_POSITION_EMBEDDING_REG_AB_H

#include "apply_rotary_pos_emb_common.h"

namespace PartialRotaryMulQuantOps {
using namespace AscendC::MicroAPI;
using namespace AscendC;

template <typename T>
class RotaryPositionEmbeddingAB {
public:
    __aicore__ inline RotaryPositionEmbeddingAB(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, const QuantRopeRegbaseTilingData* tilingData,
        TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessLoop(
        int64_t xGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer, int64_t ubIdx, int64_t bsCount,
        int64_t nCount);
    __aicore__ inline void CopyInNope(GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx, int64_t totalCount);
    __aicore__ inline void CopyInRope(GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx, int64_t totalCount);
    __aicore__ inline void QuantizeNopeRegion(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t totalCount, int64_t sliceStart);
    __aicore__ inline void ComputeAndQuantizeRope(
        LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cosBuffer, LocalTensor<T>& sinBuffer,
        int64_t bsCount, int64_t nCount, int64_t sliceLen);
    __aicore__ inline void CopyOutRegion(
        TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
        GlobalTensor<uint8_t>& out, int64_t xGmOffset, int64_t ubIdx,
        int64_t totalCount, int64_t dAlignH8, int64_t dCount, int64_t dStart);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> nopeInQueue_;
    TQue<QuePosition::VECIN, 1> ropeInQueue_;
    TQue<QuePosition::VECIN, 1> cosInQueue_;
    TQue<QuePosition::VECIN, 1> sinInQueue_;
    TQue<QuePosition::VECOUT, 1> nopeOutQueue_;
    TQue<QuePosition::VECOUT, 1> ropeOutQueue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> cosGm_;
    GlobalTensor<T> sinGm_;
    GlobalTensor<uint8_t> yGm_;
    const QuantRopeRegbaseTilingData* tilingData_;
    DataCopyPadExtParams<T> padParams_ = {false, 0, 0, static_cast<T>(0)};
    uint8_t DB_FLAG = 2;
    uint32_t dSplitSize_ = 0;
    int64_t bsBlockCount_ = 0;
    int64_t nBlockCount_ = 0;
    int64_t dAlign_ = 0;
    int64_t dAlignNope_ = 0;
    int64_t dAlignH8_ = 0;
    int64_t dAlignH8Nope_ = 0;
    int64_t sliceAlign_ = 0;
    float scale_ = 1.0f;
};

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Init(
    GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace, const QuantRopeRegbaseTilingData* tilingData,
    TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    scale_ = tilingData_->scale;
    dSplitSize_ = tilingData_->sliceLength / tilingData_->dSplitCoef * sizeof(T);
    int64_t blockDimBS = GetBlockIdx() / tilingData_->blockNumN;
    int64_t blockDimN = GetBlockIdx() % tilingData_->blockNumN;
    bsBlockCount_ = (blockDimBS == tilingData_->blockNumBS - 1) ? tilingData_->blockTailBS : tilingData_->blockFactorBS;
    nBlockCount_ = (blockDimN == tilingData_->blockNumN - 1) ? tilingData_->blockTailN : tilingData_->blockFactorN;

    int64_t cosOffset = blockDimBS * tilingData_->blockFactorBS * tilingData_->sliceLength;
    int64_t offset = blockDimBS * tilingData_->blockFactorBS * tilingData_->D;
    int64_t xOffset = offset * tilingData_->N + blockDimN * tilingData_->blockFactorN * tilingData_->D;
    this->cosGm_.SetGlobalBuffer((__gm__ T*)cos + cosOffset);
    this->sinGm_.SetGlobalBuffer((__gm__ T*)sin + cosOffset);
    this->xGm_.SetGlobalBuffer((__gm__ T*)x + xOffset);
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xOffset);

    dAlign_ = ops::CeilAlign<int64_t>(tilingData_->D, GetUbBlockSize() / sizeof(T));
    dAlignNope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, GetUbBlockSize() / sizeof(T));
    dAlignH8_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength, GetUbBlockSize() / sizeof(uint8_t));
    dAlignH8Nope_ = ops::CeilAlign<int64_t>(tilingData_->sliceStart, GetUbBlockSize() / sizeof(uint8_t));
    sliceAlign_ = ops::CeilAlign<int64_t>(tilingData_->sliceLength / tilingData_->dSplitCoef, GetUbBlockSize() /
                    sizeof(T)) * tilingData_->dSplitCoef;
    int64_t nopeInSize = dAlignNope_ * sizeof(T) * tilingData_->ubFactorBS;
    int64_t ropeInSize = sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS;
    int64_t nopeOutSize = dAlignH8Nope_ * sizeof(uint8_t) * tilingData_->ubFactorBS;
    int64_t ropeOutSize = dAlignH8_ * sizeof(uint8_t) * tilingData_->ubFactorBS;
    pipe_->InitBuffer(nopeInQueue_, DB_FLAG, nopeInSize * tilingData_->ubFactorN);
    pipe_->InitBuffer(ropeInQueue_, DB_FLAG, ropeInSize * tilingData_->ubFactorN);
    pipe_->InitBuffer(cosInQueue_, DB_FLAG, sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS);
    pipe_->InitBuffer(sinInQueue_, DB_FLAG, sliceAlign_ * sizeof(T) * tilingData_->ubFactorBS);
    pipe_->InitBuffer(nopeOutQueue_, 1, nopeOutSize * tilingData_->ubFactorN);
    pipe_->InitBuffer(ropeOutQueue_, 1, ropeOutSize * tilingData_->ubFactorN);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::Process()
{
    uint32_t bsLoopCnt = ops::CeilDiv(bsBlockCount_, tilingData_->ubFactorBS);
    uint32_t nLoopCnt = ops::CeilDiv(nBlockCount_, tilingData_->ubFactorN);
    for (uint32_t bsLoopIdx = 0; bsLoopIdx < bsLoopCnt; bsLoopIdx++) {
        int64_t xGmOffset = bsLoopIdx * tilingData_->ubFactorBS * tilingData_->N * tilingData_->D;
        uint32_t currBSNum = (bsLoopIdx != bsLoopCnt - 1) ? tilingData_->ubFactorBS :
                                                            bsBlockCount_ - (bsLoopIdx * tilingData_->ubFactorBS);

        DataCopyExtParams cosParams = {
            static_cast<uint16_t>(currBSNum * tilingData_->dSplitCoef), dSplitSize_, 0,
            static_cast<uint32_t>((sliceAlign_ - tilingData_->sliceLength) * sizeof(T) / tilingData_->dSplitCoef) /
                                                            ops::BLOCK_BYTES, 0};

        LocalTensor<T> cosBuffer = cosInQueue_.AllocTensor<T>();
        LocalTensor<T> sinBuffer = sinInQueue_.AllocTensor<T>();
        DataCopyPad(cosBuffer, cosGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosParams,
                    padParams_);
        cosInQueue_.EnQue(cosBuffer);
        cosBuffer = cosInQueue_.DeQue<T>();
        DataCopyPad(sinBuffer, sinGm_[bsLoopIdx * tilingData_->ubFactorBS * tilingData_->sliceLength], cosParams,
                    padParams_);
        sinInQueue_.EnQue(sinBuffer);
        sinBuffer = sinInQueue_.DeQue<T>();

        for (int64_t nLoopIdx = 0; nLoopIdx < nLoopCnt; nLoopIdx++) {
            int64_t currNNum = (nLoopIdx != nLoopCnt - 1) ? tilingData_->ubFactorN :
                                                            nBlockCount_ - (nLoopIdx * tilingData_->ubFactorN);
            ProcessLoop(xGmOffset, cosBuffer, sinBuffer, nLoopIdx, currBSNum, currNNum);
        }

        cosInQueue_.FreeTensor(cosBuffer);
        sinInQueue_.FreeTensor(sinBuffer);
    }
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ProcessLoop(
    int64_t xGmOffset, LocalTensor<T> cosBuffer, LocalTensor<T> sinBuffer, int64_t ubIdx, int64_t bsCount,
    int64_t nCount)
{
    int64_t totalCount = bsCount * nCount;
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;

    CopyInNope(xGm_, xGmOffset, ubIdx, totalCount);
    LocalTensor<T> nopeInUb = nopeInQueue_.DeQue<T>();
    CopyInRope(xGm_, xGmOffset, ubIdx, totalCount);
    LocalTensor<T> ropeInUb = ropeInQueue_.DeQue<T>();

    LocalTensor<uint8_t> nopeUb = nopeOutQueue_.AllocTensor<uint8_t>();
    QuantizeNopeRegion(nopeInUb, nopeUb, totalCount, sliceStart);
    CopyOutRegion(nopeOutQueue_, nopeUb, yGm_, xGmOffset, ubIdx, totalCount, dAlignH8Nope_, sliceStart, 0);

    LocalTensor<uint8_t> ropeUb = ropeOutQueue_.AllocTensor<uint8_t>();
    ComputeAndQuantizeRope(ropeInUb, ropeUb, cosBuffer, sinBuffer, bsCount, nCount, sliceLen);
    CopyOutRegion(ropeOutQueue_, ropeUb, yGm_, xGmOffset, ubIdx, totalCount, dAlignH8_, sliceLen, sliceStart);

    nopeInQueue_.FreeTensor(nopeInUb);
    ropeInQueue_.FreeTensor(ropeInUb);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::CopyInNope(
    GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx, int64_t totalCount)
{
    LocalTensor<T> target = nopeInQueue_.AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(sliceStart * sizeof(T)),
        static_cast<uint32_t>((tilingData_->D - sliceStart) * sizeof(T)),
        static_cast<uint32_t>((dAlignNope_ - sliceStart) * sizeof(T) / ops::BLOCK_BYTES), 0};
    DataCopyPad(target, source[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D], inParams, padParams_);
    nopeInQueue_.EnQue(target);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::CopyInRope(
    GlobalTensor<T>& source, int64_t xGmOffset, int64_t ubIdx, int64_t totalCount)
{
    LocalTensor<T> target = ropeInQueue_.AllocTensor<T>();
    int64_t sliceStart = tilingData_->sliceStart;
    int64_t sliceLen = tilingData_->sliceLength;
    DataCopyExtParams inParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(sliceLen * sizeof(T)),
        static_cast<uint32_t>((tilingData_->D - sliceLen) * sizeof(T)),
        static_cast<uint32_t>((sliceAlign_ - sliceLen) * sizeof(T) / ops::BLOCK_BYTES), 0};
    DataCopyPad(target,
        source[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D + sliceStart], inParams, padParams_);
    ropeInQueue_.EnQue(target);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::QuantizeNopeRegion(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& nopeUb, int64_t totalCount, int64_t sliceStart)
{
    LocalTensor<hifloat8_t> nopeH8 = nopeUb.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* nopeAddr = (__local_mem__ hifloat8_t*)nopeH8.GetPhyAddr();
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    uint16_t nopeLoop = ops::CeilDiv(static_cast<int32_t>(sliceStart), static_cast<int32_t>(VL_FLOAT32_SIZE));
    uint16_t rows = static_cast<uint16_t>(totalCount);
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

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::ComputeAndQuantizeRope(
    LocalTensor<T>& inUb, LocalTensor<uint8_t>& ropeUb, LocalTensor<T>& cosBuffer, LocalTensor<T>& sinBuffer,
    int64_t bsCount, int64_t nCount, int64_t sliceLen)
{
    __local_mem__ T* inAddr = (__local_mem__ T*)inUb.GetPhyAddr();
    __local_mem__ T* sinAddr = (__local_mem__ T*)sinBuffer.GetPhyAddr();
    __local_mem__ T* cosAddr = (__local_mem__ T*)cosBuffer.GetPhyAddr();
    if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::HALF)) {
        HalfAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            tilingData_->sliceLength, sliceAlign_, sliceAlign_, dAlignH8_, bsCount, nCount, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::INTERLEAVE)) {
        InterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            tilingData_->sliceLength, sliceAlign_, sliceAlign_, dAlignH8_, bsCount, nCount, scale_);
    } else if (tilingData_->rotaryMode == static_cast<int64_t>(RotaryPosEmbeddingMode::QUARTER)) {
        QuarterAlignVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            tilingData_->sliceLength, sliceAlign_, sliceAlign_, dAlignH8_, bsCount, nCount, scale_);
    } else {
        DeepSeekInterleaveModeVF<T>(sinAddr, cosAddr, inAddr, ropeUb,
            tilingData_->sliceLength, sliceAlign_, sliceAlign_, dAlignH8_, bsCount, nCount, scale_);
    }
    ropeOutQueue_.template EnQue(ropeUb);
}

template <typename T>
__aicore__ inline void RotaryPositionEmbeddingAB<T>::CopyOutRegion(
    TQue<QuePosition::VECOUT, 1>& outQueue, LocalTensor<uint8_t>& srcUb,
    GlobalTensor<uint8_t>& out, int64_t xGmOffset, int64_t ubIdx,
    int64_t totalCount, int64_t dAlignH8, int64_t dCount, int64_t dStart)
{
    srcUb = outQueue.template DeQue<uint8_t>();
    DataCopyExtParams outParams = {static_cast<uint16_t>(totalCount), static_cast<uint32_t>(dCount * sizeof(uint8_t)),
        static_cast<uint32_t>((dAlignH8 - dCount) * sizeof(uint8_t) / ops::BLOCK_BYTES),
        static_cast<uint32_t>((tilingData_->D - dCount) * sizeof(uint8_t)), 0};
    DataCopyPad(out[xGmOffset + ubIdx * tilingData_->ubFactorN * tilingData_->D + dStart], srcUb, outParams);
    outQueue.FreeTensor(srcUb);
}

} // namespace PartialRotaryMulQuantOps

#endif // ROTARY_POSITION_EMBEDDING_REG_AB_H