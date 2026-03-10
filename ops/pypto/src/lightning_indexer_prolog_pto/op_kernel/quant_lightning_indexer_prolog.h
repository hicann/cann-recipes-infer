/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_prolog.h
 * \brief
 */

#pragma once
#ifndef QUANT_LIGHTNING_INDEXER_PROLOG_H
#define QUANT_LIGHTNING_INDEXER_PROLOG_H

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk/config.h"

namespace npu::tile_fwk {
constexpr const char *SUPPORT_DYNAMIC_UNALIGNED = "support_dynamic_unaligned";
constexpr const char *ONLY_CODEGEN = "only_codegen";

constexpr const int TILE_CUBE_DIM = 6;
constexpr const int64_t CHUNK_SIZE = 2;
constexpr size_t Q_PARAM_DIM = 2;
constexpr size_t COS_SIN_DIM = 2;
constexpr const int L0M_INDEX = 0;
constexpr const int L1M_INDEX = 1;
constexpr const int L0K_INDEX = 2;
constexpr const int L1K_INDEX = 3;
constexpr const int L0N_INDEX = 4;
constexpr const int L1N_INDEX = 5;
constexpr const int SCATTER_DIM = -2;

constexpr const int64_t VEC_TILE_256 = 256;
constexpr const int64_t VEC_TILE_128 = 128;
constexpr const int64_t VEC_TILE_64 = 64;
constexpr const int VEC_TILE_8 = 8;
constexpr const int VEC_TILE_4 = 4;

constexpr size_t NZ_DIM = 4;
constexpr const int64_t NZ_FIRST_DIM = 16;
constexpr const int64_t NZ_B8_C0 = 32;
constexpr const int64_t NZ_B16_C0 = 16;

struct QuantIndexerConfigs {
    // Tile params
    std::array<int, TILE_CUBE_DIM> qLinear;
    std::array<int, TILE_CUBE_DIM> qHd;
    std::array<int, TILE_CUBE_DIM> kLinear;
    std::array<int, TILE_CUBE_DIM> wLinear;

    // Config params
    std::set<int> unrollList = {32, 16, 8, 4, 2, 1};
    std::map<int64_t, int64_t> l1ReuseParam = {
        {1, 4}
    };
    int copyInThreshold = 2 * 1024 * 1024;
    int cycleUpperBound = 8192;
    int blockSize = 128;
};

struct QuantIndexerPrologInput {
    const Tensor &x;          // BF16, (t, h)
    const Tensor &qNorm;      // INT8, (t, qLoraRank)
    const Tensor &qNormScale; // FP32, (t, 1)
    const Tensor &wQb; // INT8, (headNum * headDim / NZ_B8_C0, qLoraRank / NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B8_C0), NZ
    const Tensor &wQbScale;    // FP32, (headNum * headDim, 1)
    const Tensor &wk;          // BF16, (headDim / NZ_B16_C0, h / NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    const Tensor &wProj;       // BF16, (headNum / NZ_B16_C0, h / NZ_FIRST_DIM, NZ_FIRST_DIM, NZ_B16_C0), NZ
    const Tensor &lnGammaK;    // BF16, (headDim,)
    const Tensor &lnBetaK;     // BF16, (headDim,)
    const Tensor &cosIdxRope;  // BF16, (t, ropeHeadDim)
    const Tensor &sinIdxRope;  // BF16, (t, ropeHeadDim)
    const Tensor &hadamardQ;   // BF16, (headDim, headDim)
    const Tensor &hadamardK;   // BF16, (headDim, headDim)
    const Tensor &kCache;      // INT8, (blockNum, blockSize, nKv, headDim)
    const Tensor &kCacheScale; // FP16, (blockNum, blockSize, nKv, 1)
    const Tensor &kCacheIndex; // INT64, (t,)
};

struct QuantIndexerPrologOutput {
    Tensor &qInt8;   // INT8, (t, headNum, headDim)
    Tensor &qScale;  // FP16, (t, headNum, 1)
    Tensor &kInt8;   // INT8, (blockNum, blockSize, nKV, headDim)
    Tensor &kScale;  // FP16, (blockNum, blockSize, nKV, 1)
    Tensor &weights; // FP16, (t, headNum)
};

struct QuantIndexerPrologAttr {
    float eps = 1e-6f;
    std::string layeroutQuery = "TND";
    std::string layeroutKey = "PA_BSND";
};

void QuantLightningIndexerPrologCompute(const QuantIndexerPrologInput &inputs, QuantIndexerPrologOutput &outputs,
    QuantIndexerPrologAttr &attrs, const QuantIndexerConfigs &configs);

void QuantLightningIndexerProlog(const QuantIndexerPrologInput &inputs, QuantIndexerPrologOutput &outputs,
    QuantIndexerPrologAttr &attrs, const QuantIndexerConfigs &configs);

} // namespace npu::tile_fwk

#endif