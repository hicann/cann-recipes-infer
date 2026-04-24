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
 * \file hc_pre_m_split_core.h
 * \brief
 */

#ifndef HC_PRE_M_SPLIT_CORE_H
#define HC_PRE_M_SPLIT_CORE_H

#include "kernel_operator.h"
#include "hc_pre_base.h"

namespace HcPre {
using namespace AscendC;
template <typename T>
class HcPreMembaseMSplitCorePerf {
public:
    __aicore__ inline HcPreMembaseMSplitCorePerf()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag, GM_ADDR workspace,
                                const HcPreTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        // InQue
        int64_t mixesQue01Size = tilingData->rowFactor * tilingData->hcMultAlign * 2 * sizeof(float);
        pipe->InitBuffer(mixesQue01, 2, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, 2,
                         tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        int64_t xQueNum1 = tilingData->rowFactor * RoundUp<T>(tilingData->kFactor) * sizeof(T);
        int64_t xQueNum2 = tilingData->rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T);
        int64_t xQueNum = xQueNum1 > xQueNum2 ? xQueNum1 : xQueNum2; 
        pipe->InitBuffer(xQue, 2, xQueNum * sizeof(T));

        // OutQue
        pipe->InitBuffer(yQue, 2, tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, 2, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, 2,
                         tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->rowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, tilingData->rowFactor * RoundUp<float>(tilingData->kFactor) * sizeof(float) * DOUBLE_BUFFER);
        pipe->InitBuffer(yCastBuf, tilingData->rowFactor * RoundUp<float>(tilingData->dFactor) * sizeof(float));
        pipe->InitBuffer(rsqrtBuf, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
    }

    __aicore__ inline void Process() 
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t totalBlockNum = GetBlockNum();

        int64_t rowOuterLoop =
            (curBlockIdx == totalBlockNum - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == totalBlockNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;

        CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        int64_t xGmBlockBaseOffsetPart1 = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
        int64_t workspaceSize1 = CeilAlign(tilingData->rowFactor * RoundUp<float>(tilingData->kFactor) * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
        int64_t workspaceSize2 = CeilAlign(tilingData->bs * (tilingData->hcMult * 2 + tilingData->hcMult * tilingData->hcMult) * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
        // bs上的循环
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) { 
            int64_t xGmBsBaseOffsetPart1 = rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->d;
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            for (int64_t kLoopIdx = 0; kLoopIdx < tilingData->kLoop; kLoopIdx++) {
                int64_t curKFactor =
                    (kLoopIdx == tilingData->kLoop - 1) ? tilingData->tailKFactor : tilingData->kFactor;
                xLocal = xQue.template AllocTensor<T>();
                int64_t curGmOffset = xGmBlockBaseOffsetPart1 + xGmBsBaseOffsetPart1 + kLoopIdx * tilingData->kFactor;
                CopyIn(xGm[curGmOffset],
                       xLocal, curRowFactor, curKFactor, tilingData->k - curKFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                xCastLocal = mmInQue.AllocTensor<float>();
                CastTwoDim(xCastLocal, xLocal, curRowFactor, curKFactor);
                xQue.template FreeTensor(xLocal);
                mmInQue.template EnQue(xCastLocal);
                xCastLocal = mmInQue.template DeQue<float>();
                CopyOut(xCastLocal,
                        workspaceGm[curGmOffset],
                        curRowFactor, curKFactor, tilingData->k - curKFactor);
                mmInQue.FreeTensor(xCastLocal);
                // MM计算
            }

            squareSumOutLocal = squareSumQue.AllocTensor<float>();
            CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 + curBlockIdx * tilingData->rowOfFormerBlock * 16 + rowOuterIdx * tilingData->rowFactor * 16], 
                   squareSumOutLocal, 1, curRowFactor * 16);
            squareSumQue.EnQue(squareSumOutLocal);
            squareSumOutLocal = squareSumQue.DeQue<float>();
            // TODO:: 这里需要使用GatherMask将斜对角的元素在UB内变成连续的一行
            // GatherMask(rsqrtLocal
            squareSumQue.template FreeTensor(squareSumOutLocal);
            float coeff = 1.0f / static_cast<float>(tilingData->k);
            Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
            PipeBarrier<PIPE_V>();
            Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
            PipeBarrier<PIPE_V>();
            Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
            Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
            PipeBarrier<PIPE_V>();
            Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);

            mxies01Local = mixesQue01.AllocTensor<float>();
            CopyIn(workspaceGm[workspaceSize1 + curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix], 
                   mxies01Local, curRowFactor, tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);
            CopyIn(
                workspaceGm[workspaceSize1 + curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix + tilingData->hcMult],
                mxies01Local[tilingData->rowFactor * tilingData->hcMultAlign], curRowFactor, tilingData->hcMult,
                tilingData->hcMix - tilingData->hcMult);
            mixesQue01.EnQue(mxies01Local);
            mxies01Local = mixesQue01.DeQue<float>();
            ProcessPre(mxies01Local, mxies01Local, hcBase0Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                       hcScaleGm.GetValue(0), tilingData->hcEps, curRowFactor, tilingData->hcMult);
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T>();
                CopyIn(xGm[xGmBlockBaseOffsetPart1 + xGmBsBaseOffsetPart1 +
                           dLoopIdx * tilingData->dFactor],
                       xLocal, tilingData->rowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                yLocal = yQue.template AllocTensor<T>();
                ProcessY(yLocal, xLocal, mxies01Local, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                         tilingData->hcMult, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal,
                        yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d +
                            rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                        curRowFactor, curDFactor, tilingData->d - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            // post
            postLocal = postQue.AllocTensor<float>();
            ProcessPost(postLocal, mxies01Local[tilingData->rowFactor * tilingData->hcMultAlign], hcBase1Local,
                        rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor,
                        tilingData->hcMult);
            mixesQue01.template FreeTensor(mxies01Local);
            postQue.EnQue(postLocal);
            postLocal = postQue.DeQue<float>();
            CopyOut(postLocal,
                    postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                           rowOuterIdx * tilingData->rowFactor * tilingData->hcMult],
                    curRowFactor, tilingData->hcMult);
            postQue.FreeTensor(postLocal);

            // combFrag
            mixes2Local = mixesQue2.AllocTensor<float>();
            CopyInWithOuterFor(workspaceGm[workspaceSize1 + curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix * 2],
                               mixes2Local, curRowFactor, tilingData->hcMult, tilingData->hcMult, tilingData->hcMix);
            mixesQue2.EnQue(mixes2Local);
            mixes2Local = mixesQue2.DeQue<float>();

            combFragLocal = combFragQue.AllocTensor<float>();

            MulABLastDimBrcInline<float, false>(mixes2Local, mixes2Local, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
            Muls(mixes2Local, mixes2Local, hcScaleGm.GetValue(2),
                 curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
            PipeBarrier<PIPE_V>();
            AddBAFirstDimBrcInline<float>(mixes2Local, mixes2Local, hcBase2Local, curRowFactor,
                                          tilingData->hcMult * tilingData->hcMultAlign);
            SoftmaxFP32Perf(mixes2Local, mixes2Local, reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                            tilingData->hcMult, tilingData->hcEps);
            ReduceSumARAPerf(reduceLocal, mixes2Local, curRowFactor, tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
            PipeBarrier<PIPE_V>();
            DivABABrcInline(combFragLocal, mixes2Local, reduceLocal, curRowFactor, tilingData->hcMult,
                            tilingData->hcMult);
            for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                                   curRowFactor * tilingData->hcMult, tilingData->hcMult);
                ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);
            }
            mixesQue2.FreeTensor(mixes2Local);
            squareSumQue.template FreeTensor(squareSumOutLocal);

            combFragQue.EnQue(combFragLocal);
            combFragLocal = combFragQue.DeQue<float>();
            CopyOut(combFragLocal,
                    combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                               rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->hcMult],
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
            combFragQue.FreeTensor(combFragLocal);
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TQue<QuePosition::VECIN, 1> squareSumQue;
    TQue<QuePosition::VECOUT, 1> mmInQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;

    LocalTensor<float> mxies01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> mmOutLocal;
    LocalTensor<float> squareSumOutLocal;
};

} // namespace HcPreSinkhorn

#endif