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
 * \file swiglu_group_quant_perf.h
 * \brief
 */

#ifndef SWIGLU_GROUP_QUANT_PERF_H
#define SWIGLU_GROUP_QUANT_PERF_H

#include "kernel_operator.h"
#include "swiglu_group_quant_base.h"

namespace SwigluGroupQuant {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class SwigluGroupQuantPerf {
public:
    __aicore__ inline SwigluGroupQuantPerf()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
        const SwigluGroupQuantTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        yGm.SetGlobalBuffer((__gm__ T1*)y);
        scaleGm.SetGlobalBuffer((__gm__ T2*)scale);

        pipe->InitBufPool(tBufPool, tilingData->ubSize);
        if (groupIndex != nullptr) {
            hasGroupIndex_ = true;
            groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
            tBufPool.InitBuffer(
                groupIndexQue, DOUBLE_BUFFER_NUM, RoundUp<int64_t>(tilingData->gFactor) * sizeof(int64_t));
            tBufPool.InitBuffer(groupIndexSumBuf, BLOCK_SIZE);
            groupSumLocal = groupIndexSumBuf.Get<int64_t>();
            for (int64_t idx = 0; idx < tilingData->gLoop; idx++) {
                int64_t curGFactor = (idx == tilingData->gLoop - 1) ? tilingData->tailGFactor : tilingData->gFactor;
                groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
                CopyIn(groupIndexGm[idx * tilingData->gFactor], groupIndexLocal, 1, curGFactor);
                groupIndexQue.template EnQue(groupIndexLocal);
                groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
                if (idx == 0) {
                    VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGFactor);
                } else {
                    VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGFactor);
                }
                groupIndexQue.template FreeTensor(groupIndexLocal);
            }
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            int64_t realBs = groupSumLocal.GetValue(0) > tilingData->bs ? tilingData->bs : groupSumLocal.GetValue(0);

            rowOfFormerBlock = CeilDiv(realBs, static_cast<int64_t>(tilingData->coreNum));
            usedCoreNums = CeilDiv(realBs, rowOfFormerBlock) < tilingData->coreNum
                ? CeilDiv(realBs, rowOfFormerBlock)
                : tilingData->coreNum;
            rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

            rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
            rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
            tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0
                ? tilingData->rowFactor
                : rowOfFormerBlock % tilingData->rowFactor;
            tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0
                ? tilingData->rowFactor
                : rowOfTailBlock % tilingData->rowFactor;
            tBufPool.Reset();
        } else {
            rowOfFormerBlock = tilingData->rowOfFormerBlock;
            rowOfTailBlock = tilingData->rowOfTailBlock;
            rowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
            rowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
            tailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
            tailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
            usedCoreNums = GetBlockNum();
        }

        if (weight != nullptr) {
            hasWeight_ = true;
            weightGm.SetGlobalBuffer((__gm__ float*)weight);
            tBufPool.InitBuffer(
                weightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        }

        tBufPool.InitBuffer(
            x0Que, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(
            x1Que, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(
            yQue, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T1>(tilingData->dFactor) * sizeof(T1));
        // scale 在ub内连续写，拷出时采用Compact模式进行搬出
        int64_t scaleColNum = CeilDiv(tilingData->dFactor, PER_BLOCK_FP16);
        tBufPool.InitBuffer(
            scaleQue, DOUBLE_BUFFER_NUM, RoundUp<T2>(tilingData->rowFactor * scaleColNum) * sizeof(T2));
        hasClampLimit_ = (tilingData->hasClampLimit == 1);
        clampLimit_ = tilingData->clampLimit;
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= usedCoreNums) {
            return;
        }
        SetMaxValue();
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == usedCoreNums - 1) ? rowLoopOfTailBlock :rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums - 1) ? tailRowFactorOfTailBlock :
                                                                     tailRowFactorOfFormerBlock;
        int64_t x0GmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->d;
        int64_t x1GmBaseOffset = x0GmBaseOffset + tilingData->splitD;
        int64_t yGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->splitD;
        int64_t scaleGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->scaleCol;
        int64_t weightGmBaseOffset = curBlockIdx * rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            // copy in weight
            if (hasWeight_) {
                weightLocal = weightQue.template AllocTensor<float>();
                CopyIn(weightGm[weightGmBaseOffset + rowOuterIdx * tilingData->rowFactor],
                    weightLocal, 1, curRowFactor);
                weightQue.template EnQue(weightLocal);
                weightLocal = weightQue.template DeQue<float>();
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                int64_t scaleDFactor = CeilDiv(curDFactor, PER_BLOCK_FP16);
                int64_t xBaseOffset =
                    rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor;
                x0Local = x0Que.template AllocTensor<T0>();
                CopyIn(
                    xGm[x0GmBaseOffset + xBaseOffset],
                    x0Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x0Que.template EnQue(x0Local);
                x0Local = x0Que.template DeQue<T0>();

                x1Local = x1Que.template AllocTensor<T0>();
                CopyIn(
                    xGm[x1GmBaseOffset + xBaseOffset],
                    x1Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x1Que.template EnQue(x1Local);
                x1Local = x1Que.template DeQue<T0>();

                yLocal = yQue.template AllocTensor<T1>();
                scaleLocal = scaleQue.template AllocTensor<T2>();
                if (hasWeight_) {
                    if (hasClampLimit_) {
                        VFProcessSwigluGroupQuant<T1, T0, T2, true, true>(
                            yLocal, scaleLocal, x0Local, x1Local, weightLocal, maxValue, curRowFactor,
                            curDFactor, clampLimit_);
                    } else {
                        VFProcessSwigluGroupQuant<T1, T0, T2, true, false>(
                            yLocal, scaleLocal, x0Local, x1Local, weightLocal, maxValue, curRowFactor,
                            curDFactor, clampLimit_);
                    }
                } else {
                    if (hasClampLimit_) {
                        VFProcessSwigluGroupQuant<T1, T0, T2, false, true>(
                            yLocal, scaleLocal, x0Local, x1Local, weightLocal, maxValue, curRowFactor,
                            curDFactor, clampLimit_);
                    } else {
                        VFProcessSwigluGroupQuant<T1, T0, T2, false, false>(
                            yLocal, scaleLocal, x0Local, x1Local, weightLocal, maxValue, curRowFactor,
                            curDFactor, clampLimit_);
                    }
                }
                x0Que.template FreeTensor(x0Local);
                x1Que.template FreeTensor(x1Local);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T1>();
                CopyOut(yLocal, yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor *
                    tilingData->splitD + dLoopIdx * tilingData->dFactor], curRowFactor, curDFactor,
                    tilingData->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);

                scaleQue.template EnQue(scaleLocal);
                scaleLocal = scaleQue.template DeQue<T2>();
                CopyOut<T2, AscendC::PaddingMode::Compact>(scaleLocal,
                    scaleGm[scaleGmBaseOffset + rowOuterIdx * tilingData->rowFactor *
                        tilingData->scaleCol + dLoopIdx * CeilDiv(tilingData->dFactor, PER_BLOCK_FP16)],
                    curRowFactor, scaleDFactor, tilingData->scaleCol - scaleDFactor);
                scaleQue.template FreeTensor(scaleLocal);
            }
            if (hasWeight_) {
                weightQue.template FreeTensor(weightLocal);
            }
        }
    }

    __aicore__ inline void SetMaxValue() {
        if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupQuantTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int64_t> groupIndexGm;
    GlobalTensor<T1> yGm;
    GlobalTensor<T2> scaleGm;
    GlobalTensor<float> weightGm;

    TQue<QuePosition::VECIN, 1> x0Que;
    TQue<QuePosition::VECIN, 1> x1Que;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> scaleQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TQue<QuePosition::VECIN, 1> weightQue;

    LocalTensor<T0> x0Local;
    LocalTensor<T0> x1Local;
    LocalTensor<T1> yLocal;
    LocalTensor<T2> scaleLocal;

    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;
    LocalTensor<float> weightLocal;

    float maxValue = 0.0f;
    bool hasGroupIndex_ = false;
    bool hasWeight_ = false;
    float clampLimit_ = 448.0f;
    bool hasClampLimit_ = false;
    TBufPool<QuePosition::VECCALC, 12> tBufPool;

    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};
} // namespace SwigluGroupQuant

#endif
