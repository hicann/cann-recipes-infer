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
 * \file quant_lightning_indexer_prolog_impl.cpp
 * \brief
 */

#include <iostream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <sstream>
#include "tilefwk/tilefwk.h"
#include "tilefwk/op_registry.h"
#include "tilefwk/tilefwk_op.h"
#include "quant_lightning_indexer_prolog.h"

namespace npu::tile_fwk {
constexpr uint64_t Lightning_Indexer_Prolog_ConfigKey = uint64_t(100000000UL);

struct LightningIndexerPrologParams {
    int t = -1;
    int blockNum = -1;
    int h = 7168;
    int qLoraRank = 1536;
    int headDim = 128;
    int headNum = 64;
    int ropeHeadDim = 64;
    int blockSize = 128;
    int nKV = 1;
    int s2 = 65536;
    // configs
    QuantIndexerConfigs quantIndexerPrologConfigs;
    // perf
    bool enableNz{true};
};

void SetPerfConfig() {
    config::SetHostOption(ONLY_CODEGEN, true);
}

void SetTileConfigParams(LightningIndexerPrologParams &params) {
    QuantIndexerConfigs configs;
    configs.qLinear = {32, 32, 512, 512, 128, 128};
    configs.qHd = {64, 64, 128, 128, 128, 128};
    configs.kLinear = {16, 16, 512, 512, 64, 64};
    configs.wLinear = {16, 16, 1024, 1024, 32, 32};
    params.quantIndexerPrologConfigs = configs;
}

template <bool nz = false>
void DynamicLightningIndexerPrologImpl(LightningIndexerPrologParams &params) {
    DataType dType = DT_BF16;
    int t = params.t;
    int blockNum = params.blockNum;
    int h = params.h;
    int qLoraRank = params.qLoraRank;
    int headDim = params.headDim;
    int headNum = params.headNum;
    int ropeHeadDim = params.ropeHeadDim;
    int blockSize = params.blockSize;
    int nKV = params.nKV;

    constexpr int64_t nzFirstDim = 16;
    constexpr int64_t b16C0Dim = 16;
    constexpr int64_t b8C0Dim = 32;

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, {t, h}, "x");
    Tensor qNorm(DT_INT8, {t, qLoraRank}, "qNorm");
    Tensor qNormScale(DT_FP32, {t, 1}, "qNormScale");
    Tensor wQb(DT_INT8, {headNum * headDim / b8C0Dim, qLoraRank / nzFirstDim, nzFirstDim, b8C0Dim}, "wQb", weightFormat);
    Tensor wQbScale(DT_FP32, {headNum * headDim, 1}, "wQbScale");
    Tensor wk(dType, {headDim / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wk", weightFormat);
    Tensor wProj(dType, {headNum / b16C0Dim, h / nzFirstDim, nzFirstDim, b16C0Dim}, "wProj", weightFormat);
    Tensor lnGammaK(dType, {headDim}, "lnGammaK");
    Tensor lnBetaK(dType, {headDim}, "lnBetaK");
    Tensor cosIdxRope(dType, {t, ropeHeadDim}, "cosIdxRope");
    Tensor sinIdxRope(dType, {t, ropeHeadDim}, "sinIdxRope");
    Tensor hadamardQ(dType, {headDim, headDim}, "hadamardQ");
    Tensor hadamardK(dType, {headDim, headDim}, "hadamardK");
    Tensor kCache(DT_INT8, {blockNum, blockSize, nKV, headDim}, "kCache");
    Tensor kCacheScale(DT_FP16, {blockNum, blockSize, nKV, 1}, "kCacheScale");
    Tensor kCacheIndex(DT_INT64, {t}, "kCacheIndex");

    QuantIndexerPrologInput input{x, qNorm, qNormScale, wQb, wQbScale, wk, wProj, lnGammaK, lnBetaK, cosIdxRope,
        sinIdxRope, hadamardQ, hadamardK, kCache, kCacheScale, kCacheIndex};

    // outputs
    auto symT = GetInputShape(x, 0);
    auto symBlockNum = GetInputShape(kCache, 0);
    Tensor qInt8Out(DT_INT8, {symT, headNum, headDim}, "qInt8");
    Tensor qScaleOut(DT_FP16, {symT, headNum, 1}, "qScale");
    Tensor kInt8Out(DT_INT8, {symBlockNum, blockSize, nKV, headDim}, "kInt8");
    Tensor kScaleOut(DT_FP16, {symBlockNum, blockSize, nKV, 1}, "kScale");
    Tensor weightsOut(DT_FP16, {symT, headNum}, "weights");
    QuantIndexerPrologOutput output{qInt8Out, qScaleOut, kInt8Out, kScaleOut, weightsOut};

    QuantIndexerPrologAttr attrs;
    attrs.eps = 1e-6f;
    attrs.layeroutKey = "PA_BSND";
    attrs.layeroutQuery = "TND";

    QuantLightningIndexerProlog(input, output, attrs, params.quantIndexerPrologConfigs);
}

void DynamicLightningIndexerPrologPto(uint64_t configKey) {
    (void)configKey;

    SetPerfConfig();

    LightningIndexerPrologParams params;
    SetTileConfigParams(params);

    DynamicLightningIndexerPrologImpl<true>(params); // enable nz
}

REGISTER_OP(LightningIndexerPrologPto)
    .ImplFunc({
        {Lightning_Indexer_Prolog_ConfigKey, DynamicLightningIndexerPrologPto}
});
} // namespace npu::tile_fwk
