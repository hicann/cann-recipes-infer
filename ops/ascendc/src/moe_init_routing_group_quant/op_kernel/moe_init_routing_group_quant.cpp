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
 * \file moe_init_routing_group_quant.cpp
 * \brief
 */
#include "moe_v3_mrgsort_out.h"
#include "moe_v3_mrgsort.h"
#include "moe_v3_sort_one_core.h"
#include "moe_v3_sort_multi_core.h"
#include "moe_v3_expert_tokens_count.h"
#include "moe_v3_row_idx_gather.h"
#include "moe_v3_gather_out.h"
#include "moe_v3_gather_dynamic_quant.h"
#include "moe_v3_gather_mxfp8_quant.h"
#include "moe_v3_gather_mxfp8_quant_gather.h"
#include "moe_v3_gather_fp8_group_quant.h"
#include "moe_v3_gather_fp8_group_quant_gather.h"

/*
 * 非量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_GATHER 1000000    // 单核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER 1001000   // 单核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER 1100000  // 多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER 1101000 // 多核排序、非量化、SCATTER索引

/*
 * 动态量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER 1020000    // 单核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER 1021000   // 单核排序、动态量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER 1120000  // 多核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER 1121000 // 多核排序、动态量化、SCATTER索引

/*
 * MXFP8量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER 1030000    // 单核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER 1031000   // 单核排序、MXFP8量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER 1130000  // 多核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER 1131000 // 多核排序、MXFP8量化、SCATTER索引

/*
 * FP8Block量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_GATHER 1050000    // 单核排序、FP8Block量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_SCATTER 1051000   // 单核排序、FP8Block量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_GATHER 1150000  // 多核排序、FP8Block量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_SCATTER 1151000 // 多核排序、FP8Block量化、SCATTER索引

using namespace AscendC;
using namespace MoeInitRoutingV3;
using namespace MoeInitRoutingOptimize;
extern "C" __global__ __aicore__ void moe_init_routing_group_quant(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                          GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                          GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MoeInitRoutingV3Arch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeInitRoutingV3Arch35TilingData, tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();

    auto t = &tilingData;

    // 1.排序阶段，计算SortedExpertIdx和SortedRowIdx。若rowIdxType=1(Scatter)，则SortedRowIdx直接写到输出expandedRowIdx。
    TPipe sortPipe;
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_SCATTER)) {
        // 单核排序
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_SCATTER)) {
        // 多核排序
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    // 2.TokensCount阶段，计算输出expertTokensCountOrCumsum
    TPipe histogramPipe;
    ExpertTokensCount countOp;
    countOp.Init(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
    countOp.Process();
    histogramPipe.Destroy();

    // 3.若rowIdxType=0(Gather)，映射计算输出expandedRowIdx；否则该输出在阶段1就被写出
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_GATHER)) {
        // GATHER索引
        TPipe rowIdxPipe;
        RowIdxGather rowIdxGatherOp;
        rowIdxGatherOp.Init(expandedRowIdx, userWS, t, &rowIdxPipe);
        rowIdxGatherOp.Process();
        rowIdxPipe.Destroy();
    }
    SyncAll();

    // 4.直接搬运或是搬运的过程中对x进行量化
    if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER)) {
        // 非量化
        TPipe gatherPipe;
        MoeGatherOut<DTYPE_X> gatherOp;
        gatherOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
        gatherOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER)) {
        // 动态量化
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            MoeGatherOutDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
            gatherDynamicQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherDynamicQuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER)) {
        // MXFP8量化。由于MXFP8模板用到的指令不支持DTYPE_X为int8_t等类型，因此需要constexpr-if来规避编译
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) {
            TPipe gatherPipe;
            MoeGatherOutMxfp8Quant<DTYPE_X, DTYPE_EXPANDED_X> gatherMxfp8QuantOp;
            gatherMxfp8QuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherMxfp8QuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER)) {
        // MXFP8量化。由于MXFP8模板用到的指令不支持DTYPE_X为int8_t等类型，因此需要constexpr-if来规避编译
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value) {
            TPipe gatherPipe;
            MoeGatherOutMxfp8QuantGather<DTYPE_X, DTYPE_EXPANDED_X> gatherMxfp8QuantOp;
            gatherMxfp8QuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherMxfp8QuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_SCATTER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_SCATTER)) {
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, float>::value) {
            TPipe gatherPipe;
            MoeGatherOutFp8GroupQuant<DTYPE_X, DTYPE_EXPANDED_X> gatherFp8GroupQuantOp;
            gatherFp8GroupQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherFp8GroupQuantOp.Process();
            gatherPipe.Destroy();
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTONECORE_FP8_GROUP_QUANT_GATHER) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_V3_SORTMULTICORE_FP8_GROUP_QUANT_GATHER)) {
        // MXFP8量化。由于MXFP8模板用到的指令不支持DTYPE_X为int8_t等类型，因此需要constexpr-if来规避编译
        if constexpr (IsSameType<DTYPE_X, bfloat16_t>::value || IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, float>::value) {
            TPipe gatherPipe;
            MoeGatherOutFp8GroupQuantGather<DTYPE_X, DTYPE_EXPANDED_X> gatherFp8GroupQuantOp;
            gatherFp8GroupQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherFp8GroupQuantOp.Process();
            gatherPipe.Destroy();
        }
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}