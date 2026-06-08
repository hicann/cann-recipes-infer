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
 * \file swiglu_mx_quant_perf.h
 * \brief
 */

#ifndef SWIGLU_MXFP4_QUANT_PERF_H
#define SWIGLU_MXFP4_QUANT_PERF_H

#include "kernel_operator.h"
#include "swiglu_group_quant_base.h"

namespace SwigluGroupQuant {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class SwigluMxFp4QuantPerf {
public:
    __aicore__ inline SwigluMxFp4QuantPerf()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
        const SwigluGroupQuantTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        // y is packed fp4 (2 values per byte); address it as raw int8 bytes. fp4x2_e2m1_t/fp4x2_e1m2_t
        // have sizeof()==2, so a GlobalTensor<T1> would double every byte offset/length on CopyOut.
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
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

        if (topkWeight != nullptr) {
            hasTopkWeight_ = true;
            topkWeightGm.SetGlobalBuffer((__gm__ float*)topkWeight);
            tBufPool.InitBuffer(
                topkWeightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        }

        tBufPool.InitBuffer(
            xQue, DOUBLE_BUFFER_NUM,
            tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0) * DIGIT_TWO);
        // fp4 output is packed 2 elements per byte: per-row valid bytes = CeilDiv(dFactor, FP4_PACK_NUM)
        tBufPool.InitBuffer(yQue, DOUBLE_BUFFER_NUM,
            tilingData->rowFactor * RoundUp<int8_t>(CeilDiv(tilingData->dFactor, FP4_PACK_NUM)) * sizeof(T1));
        scaleColNum = CeilDiv(tilingData->dFactor, PER_MX_FP16);
        tBufPool.InitBuffer(
            scaleQue, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T2>(scaleColNum) * sizeof(T2));

        tBufPool.InitBuffer(swigluBuf, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(maxExpBuf, tilingData->rowFactor * RoundUp<uint16_t>(scaleColNum) * sizeof(uint16_t));
        tBufPool.InitBuffer(invScaleBuf, tilingData->rowFactor * RoundUp<uint16_t>(scaleColNum) * sizeof(uint16_t));
    
        swigluLocal = swigluBuf.Get<T0>();
        maxExpLocal = maxExpBuf.Get<uint16_t>();
        invScaleLocal = invScaleBuf.Get<uint16_t>();

        hasClampValue_ = (tilingData->hasClampLimit == 1);
        clampValue_ = tilingData->clampLimit;
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
        int64_t yGmBaseOffset = curBlockIdx * rowOfFormerBlock * (tilingData->splitD / FP4_PACK_NUM);
        int64_t scaleGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->scaleCol;
        int64_t topkWeightGmBaseOffset = curBlockIdx * rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            // copy in topkWeight
            if (hasTopkWeight_) {
                topkWeightLocal = topkWeightQue.template AllocTensor<float>();
                CopyIn(topkWeightGm[topkWeightGmBaseOffset + rowOuterIdx * tilingData->rowFactor],
                    topkWeightLocal, 1, curRowFactor);
                topkWeightQue.template EnQue(topkWeightLocal);
                topkWeightLocal = topkWeightQue.template DeQue<float>();
            }
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                int64_t scaleDFactor = CeilDiv(curDFactor, PER_MX_FP16);
                int64_t xBaseOffset =
                    rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T0>();
                CopyIn(
                    xGm[x0GmBaseOffset + xBaseOffset],
                    xLocal, curRowFactor, curDFactor, tilingData->d - curDFactor);

                CopyIn(
                    xGm[x1GmBaseOffset + xBaseOffset],
                    xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)],
                    curRowFactor, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T0>();
                if (hasTopkWeight_) {
                    if (hasClampValue_) {
                        VFProcessSwiglu<T0, true, true>(swigluLocal, xLocal,
                            xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal,
                            curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwiglu<T0, true, false>(swigluLocal, xLocal,
                            xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal,
                            curRowFactor, curDFactor, clampValue_);
                    }
                } else {
                    if (hasClampValue_) {
                        VFProcessSwiglu<T0, false, true>(swigluLocal, xLocal,
                            xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal,
                            curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwiglu<T0, false, false>(swigluLocal, xLocal,
                            xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal,
                            curRowFactor, curDFactor, clampValue_);
                    }
                }
                VFComputeMaxExpMXFP4(swigluLocal, maxExpLocal, curRowFactor, curDFactor);
                scaleLocal = scaleQue.template AllocTensor<T2>();
                VFComputeScaleMXFP4(maxExpLocal, scaleLocal.template ReinterpretCast<uint16_t>(),
                    invScaleLocal, curRowFactor, curDFactor, maxValue);

                scaleQue.template EnQue(scaleLocal);
                scaleLocal = scaleQue.template DeQue<T2>();
                // e8m0 scale is written contiguously (packed, scaleDFactor bytes/row, not 32B-aligned) in UB,
                // so it must be copied out with PaddingMode::Compact (same as the fp8-mx path). Default
                // (Normal) mode assumes each row starts at a 32B UB block boundary and misreads rows > 0.
                CopyOut<T2, AscendC::PaddingMode::Compact>(scaleLocal,
                    scaleGm[scaleGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->scaleCol +
                        dLoopIdx * CeilDiv(tilingData->dFactor, PER_MX_FP16)],
                    curRowFactor, scaleDFactor, tilingData->scaleCol - scaleDFactor);
                scaleQue.template FreeTensor(scaleLocal);
        
                yLocal = yQue.template AllocTensor<T1>();
                if constexpr (IsSameType<T1, fp4x2_e2m1_t>::value) {
                    VFComputeDataMXFP4<T0, fp4x2_e2m1_t>(swigluLocal, invScaleLocal,
                        yLocal.template ReinterpretCast<int8_t>(), curRowFactor, curDFactor);
                } else {
                    VFComputeDataMXFP4<T0, fp4x2_e1m2_t>(swigluLocal, invScaleLocal,
                        yLocal.template ReinterpretCast<int8_t>(), curRowFactor, curDFactor);
                }

                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);

                yLocal = yQue.template DeQue<T1>();
                // fp4 output is packed two values per byte: lengths/strides are halved. The GM tensor and
                // CopyOut must be addressed in BYTES (int8): fp4x2_e2m1_t/fp4x2_e1m2_t have sizeof()==2, so
                // indexing yGm as T1 would double every offset/length and overrun adjacent rows. The
                // official swiglu_mx_quant uses a uint8 y GM for exactly this reason.
                CopyOut(yLocal.template ReinterpretCast<int8_t>(),
                    yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor * (tilingData->splitD / FP4_PACK_NUM) +
                        dLoopIdx * (tilingData->dFactor / FP4_PACK_NUM)],
                    curRowFactor, CeilDiv(curDFactor, FP4_PACK_NUM),
                    tilingData->splitD / FP4_PACK_NUM - CeilDiv(curDFactor, FP4_PACK_NUM));
                yQue.template FreeTensor(yLocal);
            }
            if (hasTopkWeight_) {
                topkWeightQue.template FreeTensor(topkWeightLocal);
            }
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<T1, fp4x2_e2m1_t>::value) {
            maxValue = FP4_E2M1_BF16_MAX_EXP;
        } else {
            maxValue = FP4_E1M2_MAX_EXP;
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupQuantTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int64_t> groupIndexGm;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<T2> scaleGm;
    GlobalTensor<float> topkWeightGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> scaleQue;

    TBuf<QuePosition::VECCALC> swigluBuf;
    TBuf<QuePosition::VECCALC> maxExpBuf;
    TBuf<QuePosition::VECCALC> invScaleBuf;
    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TQue<QuePosition::VECIN, 1> topkWeightQue;
    TBufPool<QuePosition::VECCALC, TBUF_POOL_NUM> tBufPool;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> yLocal;
    LocalTensor<T2> scaleLocal;
    LocalTensor<T0> swigluLocal;
    LocalTensor<uint16_t> maxExpLocal;
    LocalTensor<uint16_t> invScaleLocal;
    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;
    LocalTensor<float> topkWeightLocal;

    uint16_t maxValue = 0;
    int64_t scaleColNum = 0;
    float clampValue_ = 448.0f;
    bool hasClampValue_ = false;
    bool hasGroupIndex_ = false;
    bool hasTopkWeight_ = false;

    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};

} // namespace HCPre

#endif