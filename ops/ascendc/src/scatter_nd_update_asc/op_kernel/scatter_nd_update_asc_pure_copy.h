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
 * \file scatter_nd_update_asc_pure_copy.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_ASC_PURE_COPY_H
#define SCATTER_ND_UPDATE_ASC_PURE_COPY_H

#include "kernel_operator.h"

namespace ScatterNdUpdateAsc {
using namespace AscendC;

template <typename DVAR, typename DINDICES>
class ScatterNdUpdateAscPureCopy {
public:
    // static constexpr bool ifGroupAlpha_ = true;
    __aicore__ inline ScatterNdUpdateAscPureCopy(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR update, GM_ADDR y, const ScatterNdUpdateAscTilingData* tilingData) {
        // init tilingdata
        a_ = tilingData->a;
        b_ = tilingData->b;
        bAlign_ = tilingData->bAlign;
        c_ = tilingData->c;
        ubFactor_ = tilingData->ubFactor;
        blockFactor_ = tilingData->blockFactor;
        blockFactorTail_ = tilingData->blockFactorTail;
        blockNum_ = tilingData->blockNum;

        // init
        blockIdx_ = GetBlockIdx();
        curBlockFactor_ = blockFactor_;
        if (blockIdx_ == (blockNum_ - 1)) {
            curBlockFactor_ = blockFactorTail_;
        }
        curUBLoops_ = (curBlockFactor_ + ubFactor_ - 1) / ubFactor_;
        curUbFactorTail_ = curBlockFactor_ - (curUBLoops_ - 1) * ubFactor_;
        
        // int gm tensor
        varGm_.SetGlobalBuffer((__gm__ DVAR*)var);
        indicesGm_.SetGlobalBuffer((__gm__ DINDICES*)indices + blockIdx_ * blockFactor_);
        updateGm_.SetGlobalBuffer((__gm__ DVAR*)update + blockIdx_ * blockFactor_ * b_);
        yGm_.SetGlobalBuffer((__gm__ DVAR*)y);

        // init ub tensor
        pipe_->InitBuffer(updateQueue_, 2, ubFactor_ * bAlign_ * sizeof(DVAR));
    }

    __aicore__ inline void Process() {

        for(int64_t i = 0; i < curUBLoops_; i++) {
            curUbFactor_ = (i == (curUBLoops_ - 1)) ? curUbFactorTail_ : ubFactor_;

            // main process
            LocalTensor<DVAR> updateLocal = updateQueue_.AllocTensor<DVAR>();
            DataCopyPadExtParams<DVAR> dataCopyPadExtParamsVar;
            dataCopyPadExtParamsVar.isPad = false;
            dataCopyPadExtParamsVar.leftPadding = 0;
            dataCopyPadExtParamsVar.rightPadding = 0;
            dataCopyPadExtParamsVar.paddingValue = 0;
            DataCopyExtParams copyInParamsVar;
            copyInParamsVar.blockCount = curUbFactor_;
            copyInParamsVar.blockLen =  b_ * sizeof(DVAR);
            copyInParamsVar.srcStride = 0;
            copyInParamsVar.dstStride = 0;
            DataCopyPad(updateLocal, updateGm_[i * ubFactor_ * b_], copyInParamsVar, dataCopyPadExtParamsVar);

            updateQueue_.EnQue(updateLocal);
            updateQueue_.DeQue<DVAR>();

            DataCopyExtParams copyInParamsY;
            copyInParamsY.blockCount = 1;
            copyInParamsY.blockLen =  b_ * sizeof(DVAR);
            copyInParamsY.srcStride = 0;
            copyInParamsY.dstStride = 0;
            
            int64_t fourLoops = curUbFactor_ / UNROLL;
            int64_t fourLoopstIails = curUbFactor_ % UNROLL;
            int64_t updateoffset0 = 0;
            int64_t updateoffset1 = 0;
            int64_t updateoffset2 = 0;
            int64_t updateoffset3 = 0;
            for (int64_t j = 0; j < fourLoops; j++) {
                updateoffset0 = static_cast<int64_t>(indicesGm_(i * ubFactor_ + j * UNROLL)) * b_;
                updateoffset1 = static_cast<int64_t>(indicesGm_(i * ubFactor_ + j * UNROLL + UNROLL_IDX1)) * b_;
                updateoffset2 = static_cast<int64_t>(indicesGm_(i * ubFactor_ + j * UNROLL + UNROLL_IDX2)) * b_;
                updateoffset3 = static_cast<int64_t>(indicesGm_(i * ubFactor_ + j * UNROLL + UNROLL_IDX3)) * b_;
                if (updateoffset0 >= 0) {
                    DataCopyPad(yGm_[updateoffset0], updateLocal[(j * UNROLL) * bAlign_], copyInParamsY);
                }
                if (updateoffset1 >= 0) {
                    DataCopyPad(yGm_[updateoffset1], updateLocal[(j * UNROLL + UNROLL_IDX1) * bAlign_], copyInParamsY);
                }
                if (updateoffset2 >= 0) {
                    DataCopyPad(yGm_[updateoffset2], updateLocal[(j * UNROLL + UNROLL_IDX2) * bAlign_], copyInParamsY);
                }
                if (updateoffset3 >= 0) {
                    DataCopyPad(yGm_[updateoffset3], updateLocal[(j * UNROLL + UNROLL_IDX3) * bAlign_], copyInParamsY);
                }
            }

            int64_t updateoffset = 0;
            for (int64_t k = 0; k < fourLoopstIails; k++) {
                updateoffset = static_cast<int64_t>(indicesGm_(i * ubFactor_ + fourLoops * UNROLL + k)) * b_;
                if (updateoffset >= 0) {
                    DataCopyPad(yGm_[updateoffset], updateLocal[(fourLoops * UNROLL + k) * bAlign_], copyInParamsY);
                }
            }
            updateQueue_.FreeTensor(updateLocal);
        }

    }

protected:
    // global mem
    GlobalTensor<DVAR> varGm_;
    GlobalTensor<DINDICES> indicesGm_;
    GlobalTensor<DVAR> updateGm_;
    GlobalTensor<DVAR> yGm_;

    // ub memory tensor
    TPipe* pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> updateQueue_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t a_ = 0;
    int64_t b_ = 0;
    int64_t bAlign_ = 0;
    int64_t c_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t ubFactor_ = 0;

    int64_t curBlockFactor_ = 0;
    int64_t curUbFactor_ = 0;
    int64_t curUbFactorTail_ = 0;
    int64_t curUBLoops_ = 0;

    static constexpr int64_t UNROLL = 4;
    static constexpr int64_t UNROLL_IDX1 = 1;
    static constexpr int64_t UNROLL_IDX2 = 2;
    static constexpr int64_t UNROLL_IDX3 = 3;
};

} 
#endif