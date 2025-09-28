/* *
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.  * See LICENSE in the root of
the software repository for the full text of the License.
 */

#include "dynamic_mla.h"
#include "tilefwk/tensor.h"

namespace npu::tile_fwk {
std::vector<Tensor> mlaPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs, bool splitK, bool isSmooth) {
    // quant
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;

    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1];
    int h = tokenX.GetShape()[2];
    int bs = b * s;
    int q_lora_rank = wDq.GetShape()[1];

    DataType dType = tokenX.GetDataType();
    DataType dTypeQuantOut = isQuant ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    Tensor input = Reshape(tokenX, {bs, h}); // [b,s,h] -> [b*s,h]

    /******** q ********/
    int c0 = 16;
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;
    int tieM = std::min(32, m);
    TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {64, 64}); // 256, 64
    // [b*s,h] * [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qMmRes;
    if (splitK) {
        Tensor tmpC(DT_FP32, {bs, q_lora_rank}, "tmp_q");
        TileShape::Current().SetVecTile(std::min(32, bs), 128);
        tmpC = MulS(tmpC, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult;
        auto kSplit = 7;
        auto kSplitSize = h / kSplit;
        for (int ki = 0; ki < kSplit; ki++) {
            auto input_mk = View(input, {bs, kSplitSize}, {0, ki * kSplitSize});
            auto input_kn = View(wDq, {kSplitSize, q_lora_rank}, {ki * kSplitSize, 0});
            auto tmp = Matrix::Matmul(DT_FP32, input_mk, input_kn, tmpC); // [b*s,h/2] * [h/2,q_lora_rank]
            matmulResult.emplace_back(tmp);
        }
        Tensor qMmResF32 = npu::tile_fwk::Reduce(matmulResult, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(std::min(32, bs), 128);
        qMmRes = Cast(qMmResF32, dType);
    } else {
        qMmRes = Matrix::Matmul(dType, input, wDq);
    }

    TileShape::Current().SetVecTile(std::min(8, bs), q_lora_rank);
    Tensor normRes = RmsNorm(qMmRes, gammaCq, epsilonCq);

    Tensor normDequantScale;
    std::tuple<Tensor, Tensor> normQuantRes;
    if (isQuant) {
        if (isSmooth) {
            normQuantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            normQuantRes = Quant(normRes);
        }
        normRes = std::get<0>(normQuantRes);
        normDequantScale = std::get<1>(normQuantRes);
        TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {256, 256});
    } else {
        TileShape::Current().SetCubeTile({tieM, tieM}, {256, 256}, {64, 64});
    }
    // [b*s,qLoraRank] * [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor q = Matrix::Matmul(dTypeQuantOut, normRes, wUqQr); // bf16  // quant: A8W8O32 -> bf16
    qkvPreRes.emplace_back(q);

    /******** kv ********/
    TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
    // [b*s,h] * [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (splitK) {
        TileShape::Current().SetVecTile(std::min(32, bs), 64);
        int kv_n = wDkvKr.GetShape()[1];
        Tensor tmpC_kv(DT_FP32, {bs, kv_n}, "tmp_kv");
        tmpC_kv = MulS(tmpC_kv, Element(DataType::DT_FP32, 0.0f));
        std::vector<Tensor> matmulResult_kv;
        auto kSplit_kv = 7;
        auto kSplitSize_kv = h / kSplit_kv;
        for (int ki = 0; ki < kSplit_kv; ki++) {
            auto input_mk = View(input, {bs, kSplitSize_kv}, {0, ki * kSplitSize_kv});
            auto input_kn = View(wDkvKr, {kSplitSize_kv, kv_n}, {ki * kSplitSize_kv, 0});
            auto tmp = Matrix::Matmul(DT_FP32, input_mk, input_kn, tmpC_kv); // [b*s,h/2] * [h/2,kv_n] = [b*s,kv_n]
            matmulResult_kv.emplace_back(tmp);
        }
        Tensor kvMmResF32 = npu::tile_fwk::Reduce(matmulResult_kv, ReduceMode::ATOMIC_ADD);
        TileShape::Current().SetVecTile(std::min(32, bs), 64);
        compressedKv = Cast(kvMmResF32, dType);
    } else {
        compressedKv = Matrix::Matmul(dType, input, wDkvKr);
    }
    Tensor compressedKvRes = Reshape(compressedKv, {b, s, (int)wDkvKr.GetShape()[1]});
    qkvPreRes.emplace_back(compressedKvRes);

    if (isQuant) {
        qkvPreRes.emplace_back(normDequantScale);
    }

    return qkvPreRes;
}

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq, float epsilonCkv,
    std::string cacheMode, bool splitK, bool isSmooth) {
    // params check
    assert(tokenX.GetShape().size() == SHAPE_DIM3 && wUk.GetShape().size() == SHAPE_DIM3 &&
           sin.GetShape().size() == SHAPE_DIM3);
    assert(cacheMode == "BNSD" || cacheMode == "PA_BSND" || cacheMode == "PA_NZ");
    DataType dType = tokenX.GetDataType();
    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1];
    int h = tokenX.GetShape()[2];
    int s2 = kvCache.GetShape()[2];
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk.GetShape()[0];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim]
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;

    int tileB = b;
    int tileBS = tileB * s;
    SymbolicScalar bLoop = b / tileB;

    FunctionConfig funConfig;
    FUNCTION("main", funConfig,
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs.dequantScaleWUqQr, quantInputs.smoothScalesCq},
        {queryOut, queryRopeOut, kvCacheOut, krCacheOut}) {
        LOOP("LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1)) {
            SymbolicScalar bOffset = bIdx * tileB;
            std::vector<SymbolicScalar> outputOffset = {bOffset, 0, 0, 0};

            Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
            bool isQuant = (dequantScaleWUqQr.GetStorage() != nullptr);

            auto xView = View(tokenX, {tileB, s, h}, {bOffset, 0, 0});
            auto qKv = mlaPre(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs, splitK, isSmooth);
            Tensor q = qKv[0];     // [b*s, n*qHeadDim]
            Tensor kvTmp = qKv[1]; // [b,s,kvLoraRank+qkRopeHeadDim]

            // dequant: int32 -> fp32 -> *scale -> fp16/bf16
            if (isQuant) {
                std::vector<int64_t> tileShape = {std::min(32, tileBS), 64};
                TileShape::Current().SetVecTile(tileShape);
                auto qTmpFp32 = Cast(q, DataType::DT_FP32);
                auto qTmpDequantScale = qKv[2];
                auto qTmpDequantPerToken = Mul(qTmpFp32, qTmpDequantScale);
                auto qTmpDequantChannel = Mul(qTmpDequantPerToken, dequantScaleWUqQr);

                q = Cast(qTmpDequantChannel, dType);
            }

            auto qTmp = Reshape(q, {tileB, s, n, qHeadDim});
            std::vector<int64_t> tileShape = {std::min(32, tileB), 1, 1, 64};
            TileShape::Current().SetVecTile(tileShape);

            /******** q ********/
            Tensor qNope = View(qTmp, {tileB, s, n, qkNopeHeadDim}, {0, 0, 0, 0}); // [b,s,n,qkNopeHeadDim]
            tileShape = {tileB, 1, 1, 128};
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeRes = Reshape(qNope, {tileBS, n, qkNopeHeadDim}); // [bs,n,qkNopeHeadDim]
            tileShape = {std::min(32, tileBS), 1, qkNopeHeadDim};         // {2, 32, qkNopeHeadDim}
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeTrans = Transpose(qNopeRes, {0, 1}); // [n,bs,qkNopeHeadDim]

            int c0 = 16;
            int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;
            TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128});
            // bmm: (n,bs,qkNopeHeadDim) * (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
            Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

            tileShape = {1, std::min(32, tileBS), kvLoraRank};
            TileShape::Current().SetVecTile(tileShape);
            Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1});                     // [bs,n,kvLoraRank]
            auto queryOutDview = Reshape(qNopeNewTrans, {tileB, s, n, kvLoraRank}); // [b,s,n,kvLoraRank], output1

            /******** kv ********/
            Tensor compressedKv = View(kvTmp, {tileB, s, kvLoraRank}, {0, 0, 0}); // [b,s,kvLoraRank]
            tileShape = {2, 1, 512};
            TileShape::Current().SetVecTile(tileShape);
            Tensor compressedKvNorm = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b,s,kvLoraRank]
            Tensor kNope = Reshape(compressedKvNorm, {tileB, 1, s, kvLoraRank});   // [b,1,s,kvLoraRank]

            /******** RoPE ********/
            Tensor kPeView = View(kvTmp, {tileB, s, qkRopeHeadDim}, {0, 0, kvLoraRank}); // [b,s,qkRopeHeadDim]
            tileShape = {std::min(32, tileB), 1, qkRopeHeadDim};
            TileShape::Current().SetVecTile(tileShape);
            Tensor kPeRes = Reshape(kPeView, {tileB, s, 1, qkRopeHeadDim}); // [b,s,1,qkRopeHeadDim]
            Tensor qPeView = View(qTmp, {tileB, s, n, qkRopeHeadDim}, {0, 0, 0, qkNopeHeadDim});
            Tensor cosView = View(cos, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor sinView = View(sin, {tileB, s, qkRopeHeadDim}, {bOffset, 0, 0});
            Tensor kRopeView(kPeRes.GetDataType(), {tileB, s, 1, qkRopeHeadDim}, "kRopeView"); // [b,1,s,qkRopeHeadDim]
            Tensor qRopeView(kPeRes.GetDataType(), {tileB, s, n, qkRopeHeadDim}, "qRopeView");
            ApplyRotaryPosEmbV2(qPeView, kPeRes, cosView, sinView, qRopeView, kRopeView, 2, ropeConfig);
            Tensor kvCacheOutDview, krCacheOutDview;
            if (cacheMode != "BNSD") {
                int blockNum = kvCache.GetShape()[0];
                int blockSize = kvCache.GetShape()[1];
                int n2 = kvCache.GetShape()[2];
                Tensor kvCacheRes = Reshape(kvCache, {blockNum * blockSize * n2, kvLoraRank});
                Tensor krCacheRes = Reshape(krCache, {blockNum * blockSize * n2, qkRopeHeadDim});
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                kNope = Reshape(kNope, {tileB * s, kvLoraRank}); // [b*s,kvLoraRank]
                Tensor kRopeRes = Reshape(kRopeView, {tileB * s * 1, qkRopeHeadDim});

                /******** kvCache ********/
                tileShape = {1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
                kvCacheOutDview = ScatterUpdate(kvCacheRes, cacheIndexDview, kNope, -2, cacheMode, blockSize);

                /******** krCache ********/
                tileShape = {1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
                krCacheOutDview = ScatterUpdate(krCacheRes, cacheIndexDview, kRopeRes, -2, cacheMode, blockSize);

                kvCacheOut = Reshape(kvCacheOutDview, {blockNum, blockSize, n2, kvLoraRank});
                krCacheOut = Reshape(krCacheOutDview, {blockNum, blockSize, n2, qkRopeHeadDim});
            } else {
                Tensor kRopeRes = Reshape(kRopeView, {tileB, 1, s, qkRopeHeadDim});
                auto cacheIndexDview = View(cacheIndex, {tileB, s}, {bOffset, 0});
                tileShape = {1, 1, 1, kvLoraRank};
                TileShape::Current().SetVecTile(tileShape);
                auto kvCacheDview = View(kvCache, {tileB, 1, s2, kvLoraRank}, {bOffset, 0, 0, 0});
                kvCacheOut = ScatterUpdate(kvCacheDview, cacheIndexDview, kNope, -2);

                tileShape = {1, 1, 1, qkRopeHeadDim};
                TileShape::Current().SetVecTile(tileShape);
                auto krCacheDview = View(krCache, {tileB, 1, s2, qkRopeHeadDim}, {bOffset, 0, 0, 0});
                krCacheOut = ScatterUpdate(krCacheDview, cacheIndexDview, kRopeRes, -2);
            }
            Assemble(queryOutDview, outputOffset, queryOut);
            Assemble(qRopeView, outputOffset, queryRopeOut);
        }
    }
}

Tensor DeQuant(DataType dType, const Tensor &input, const Tensor &scale, const Tensor &wScale) {
    Tensor dequantRes = Cast(input, DataType::DT_FP32);
    dequantRes = Mul(dequantRes, scale);
    dequantRes = Mul(dequantRes, wScale);
    return Cast(dequantRes, dType);
}

Tensor RopeV2(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM2 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    auto seqSize = x.GetShape()[NUM_VALUE_0];
    auto dR = x.GetShape()[NUM_VALUE_1];
    auto xDtype = x.GetDataType();

    TileShape::Current().SetVecTile(tileConfig.twoDim[NUM_VALUE_0], tileConfig.twoDim[NUM_VALUE_1]);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        // for reshape and view
        castX = AddS(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);

    auto xView = Reshape(castX, {1, seqSize, dR / NUM_VALUE_2, NUM_VALUE_2});
    TileShape::Current().SetVecTile(tileConfig.fourDim[NUM_VALUE_0], tileConfig.fourDim[NUM_VALUE_1],
        tileConfig.fourDim[NUM_VALUE_2], tileConfig.fourDim[NUM_VALUE_3]);
    auto xTrans = Transpose(xView, {NUM_VALUE_2, NUM_VALUE_3});
    auto xReSecond = Reshape(xTrans, {seqSize, dR});

    TileShape::Current().SetVecTile(tileConfig.twoDim[NUM_VALUE_0], tileConfig.twoDim[NUM_VALUE_1]);
    if (!(x.GetShape()[0] == cos.GetShape()[0] && x.GetShape()[1] == cos.GetShape()[1])) {
        castCos = Expand(castCos, x.GetShape());
        castSin = Expand(castSin, x.GetShape());
    }
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}

Tensor Rope3DV2(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM3 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    TileShape::Current().SetVecTile(1, 32, 128);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
        // for reshape and view
        castX = AddS(castX, Element(DT_FP32, 0.0f));
    }
    auto castCos = Cast(cos, DT_FP32);
    auto castSin = Cast(sin, DT_FP32);
    castCos = Expand(Reshape(castCos, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]}), x.GetShape());
    castSin = Expand(Reshape(castSin, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]}), x.GetShape());

    auto xView = Reshape(castX,
        {x.GetShape()[NUM_VALUE_0], x.GetShape()[NUM_VALUE_1], x.GetShape()[NUM_VALUE_2] / NUM_VALUE_2, NUM_VALUE_2});
    TileShape::Current().SetVecTile(1, 32, 128, 128);
    auto xTrans = Transpose(xView, {NUM_VALUE_2, NUM_VALUE_3});
    auto xReSecond = Reshape(xTrans, x.GetShape());
    TileShape::Current().SetVecTile(1, 32, 128, 128);
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, x.GetDataType());
    return res;
}

std::vector<Tensor> PreCompute2D(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs) {
    // quant
    Tensor dequantScaleWDq = quantInputs.dequantScaleWDq;
    Tensor dequantScaleWDkvKr = quantInputs.dequantScaleWDkvKr;
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuantA = (dequantScaleWDq.GetStorage() != nullptr) && (dequantScaleWDkvKr.GetStorage() != nullptr);
    bool isQuantB = dequantScaleWUqQr.GetStorage() != nullptr;
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool isSmooth = (smoothScalesCq.GetStorage() != nullptr);

    int bs = tokenX.GetShape()[0];
    int q_lora_rank = wDq.GetShape()[1];

    DataType dType = tokenX.GetDataType();
    DataType dTypeQuantAOut = isQuantA ? DataType::DT_INT32 : dType;
    DataType dTypeQuantBOut = isQuantB ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    ConfigManager::Instance().SetSemanticLabel("pre_reshape");
    Tensor inputQuant, inputQuantScale;

    /******** q ********/
    int c0 = 16;
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;
    int mv = std::min(8, bs);
    // [b*s,h] @ [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qAProj;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        ConfigManager::Instance().SetSemanticLabel("Quant_x");
        auto quantRes = Quant(tokenX);
        inputQuant = std::get<0>(quantRes);
        inputQuantScale = std::get<1>(quantRes);
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_qa");
        qAProj = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDq);
        ConfigManager::Instance().SetSemanticLabel("Dequant_qa");
        qAProj = DeQuant(dType, qAProj, inputQuantScale, dequantScaleWDq);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_qa");
        qAProj = Matrix::Matmul(dType, tokenX, wDq);
    }

    // rmsnorm
    TileShape::Current().SetVecTile(mv, q_lora_rank);
    ConfigManager::Instance().SetSemanticLabel("RmsNorm_qa");
    Tensor normRes = RmsNorm(qAProj, gammaCq, epsilonCq);

    // [b*s,qLoraRank] @ [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor qBProj;
    if (isQuantB) {
        Tensor normQuant, normQuantScale;
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        ConfigManager::Instance().SetSemanticLabel("Quant_qMmRes");
        std::tuple<Tensor, Tensor> quantRes;
        if (isSmooth) {
            quantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            quantRes = Quant(normRes, true, false);
        }
        normQuant = std::get<0>(quantRes);
        normQuantScale = std::get<1>(quantRes);
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_qb");
        qBProj = Matrix::Matmul(dTypeQuantBOut, normQuant, wUqQr);
        ConfigManager::Instance().SetSemanticLabel("Dequant_qb");
        qBProj = DeQuant(dType, qBProj, normQuantScale, dequantScaleWUqQr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_qb");
        qBProj = Matrix::Matmul(dType, normRes, wUqQr);
    }
    qkvPreRes.emplace_back(qBProj);

    /******** kv ********/
    // [b*s,h] @ [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_kva");
        compressedKv = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDkvKr);
        ConfigManager::Instance().SetSemanticLabel("Dequant_kva");
        compressedKv = DeQuant(dType, compressedKv, inputQuantScale, dequantScaleWDkvKr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_kva");
        compressedKv = Matrix::Matmul(dType, tokenX, wDkvKr);
    }
    qkvPreRes.emplace_back(compressedKv);
    qkvPreRes.emplace_back(normRes);

    return qkvPreRes;
}

std::vector<Tensor> PreCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs) {
    // quant
    Tensor dequantScaleWDq = quantInputs.dequantScaleWDq;
    Tensor dequantScaleWDkvKr = quantInputs.dequantScaleWDkvKr;
    Tensor dequantScaleWUqQr = quantInputs.dequantScaleWUqQr;
    bool isQuantA = (dequantScaleWDq.GetStorage() != nullptr) && (dequantScaleWDkvKr.GetStorage() != nullptr);
    bool isQuantB = dequantScaleWUqQr.GetStorage() != nullptr;
    Tensor smoothScalesCq = quantInputs.smoothScalesCq;
    bool isSmooth = (smoothScalesCq.GetStorage() != nullptr);

    int b = tokenX.GetShape()[0];
    int s = tokenX.GetShape()[1];
    int h = tokenX.GetShape()[2];
    int bs = b * s;
    int q_lora_rank = wDq.GetShape()[1];

    DataType dType = tokenX.GetDataType();
    DataType dTypeQuantAOut = isQuantA ? DataType::DT_INT32 : dType;
    DataType dTypeQuantBOut = isQuantB ? DataType::DT_INT32 : dType;
    std::vector<Tensor> qkvPreRes;

    ConfigManager::Instance().SetSemanticLabel("pre_reshape");
    Tensor input = Reshape(tokenX, {bs, h}); // [b,s,h] -> [b*s,h]
    Tensor inputQuant, inputQuantScale;

    /******** q ********/
    int c0 = 16;
    int m = (std::min(32, bs) + c0 - 1) / c0 * c0;
    int mv = std::min(8, bs);
    // [b*s,h] @ [h,q_lora_rank] = [b*s,q_lora_rank]
    Tensor qAProj;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        ConfigManager::Instance().SetSemanticLabel("Quant_x");
        auto quantRes = Quant(input);
        inputQuant = std::get<0>(quantRes);
        inputQuantScale = std::get<1>(quantRes);
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_qa");
        qAProj = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDq);
        ConfigManager::Instance().SetSemanticLabel("Dequant_qa");
        qAProj = DeQuant(dType, qAProj, inputQuantScale, dequantScaleWDq);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_qa");
        qAProj = Matrix::Matmul(dType, input, wDq);
    }

    // rmsnorm
    TileShape::Current().SetVecTile(mv, q_lora_rank);
    ConfigManager::Instance().SetSemanticLabel("RmsNorm_qa");
    Tensor normRes = RmsNorm(qAProj, gammaCq, epsilonCq);

    // [b*s,qLoraRank] @ [qLoraRank, n*qHeadDim] = [b*s, n*qHeadDim]
    Tensor qBProj;
    if (isQuantB) {
        Tensor normQuant, normQuantScale;
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        ConfigManager::Instance().SetSemanticLabel("Quant_qMmRes");
        std::tuple<Tensor, Tensor> quantRes;
        if (isSmooth) {
            quantRes = Quant(normRes, true, true, smoothScalesCq);
        } else {
            quantRes = Quant(normRes, true, false);
        }
        normQuant = std::get<0>(quantRes);
        normQuantScale = std::get<1>(quantRes);
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_qb");
        qBProj = Matrix::Matmul(dTypeQuantBOut, normQuant, wUqQr);
        ConfigManager::Instance().SetSemanticLabel("Dequant_qb");
        qBProj = DeQuant(dType, qBProj, normQuantScale, dequantScaleWUqQr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_qb");
        qBProj = Matrix::Matmul(dType, normRes, wUqQr);
    }
    qkvPreRes.emplace_back(qBProj);

    /******** kv ********/
    // [b*s,h] @ [h,kvLoraRank+qkRopeHeadDim] = [b*s,kvLoraRank+qkRopeHeadDim]
    Tensor compressedKv;
    if (isQuantA) {
        TileShape::Current().SetVecTile(mv, q_lora_rank);
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {256, 256});
        // no smooth
        ConfigManager::Instance().SetSemanticLabel("QuantMatmul_kva");
        compressedKv = Matrix::Matmul(dTypeQuantAOut, inputQuant, wDkvKr);
        ConfigManager::Instance().SetSemanticLabel("Dequant_kva");
        compressedKv = DeQuant(dType, compressedKv, inputQuantScale, dequantScaleWDkvKr);
    } else {
        TileShape::Current().SetCubeTile({m, m}, {256, 256}, {64, 64});
        ConfigManager::Instance().SetSemanticLabel("Matmul_kva");
        compressedKv = Matrix::Matmul(dType, input, wDkvKr);
    }
    qkvPreRes.emplace_back(compressedKv);
    qkvPreRes.emplace_back(normRes);

    return qkvPreRes;
}

// NSA MlaProlog, b and s is dynamic, support:
// b: 16, 32, 64, 24, 48, 96
// s: 1, 2
void MlaPrologCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    Tensor &rmsRes, float epsilonCq, float epsilonCkv, std::string cacheMode) {
    // params check
    assert(tokenX.GetShape().size() == 3 && wUk.GetShape().size() == 3 && sin.GetShape().size() == 3);
    assert(kvCache.GetShape().size() == 4 && krCache.GetShape().size() == 4);
    assert(cacheMode == "PA_BSND" || cacheMode == "PA_NZ");
    DataType dType = tokenX.GetDataType();
    int h = tokenX.GetShape()[2];
    // [n, qkNopeHeadDim, kvLoraRank]
    int n = wUk.GetShape()[0];
    int q_lora_rank = wDq.GetShape()[1];
    int qkNopeHeadDim = wUk.GetShape()[1];
    int kvLoraRank = wUk.GetShape()[2];
    int qkRopeHeadDim = sin.GetShape()[2]; // [b,s,qkRopeHeadDim], 2
    int qHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    // kvCache: [block_num, block_size, n2, kv_lora_rank], n2=1
    SymbolicScalar blockNum = GetInputShape(kvCache, 0);
    int blockSize = kvCache.GetShape()[1];
    int n2 = kvCache.GetShape()[2];
    assert(qkNopeHeadDim == 128 || qkRopeHeadDim == 64);

    int tileB = tileConfig.tileB;
    int tileS = tileConfig.tileS;
    int tileBS = tileB * tileS;

    RoPETileShapeConfigNew ropeConfig {
        {tileB, tileS, qkRopeHeadDim}, // (b,s,d)
        {tileB, tileS, 1, qkRopeHeadDim}, // (b,s,n,d) Q
        {tileB, tileS, 1, qkRopeHeadDim}, // (b,s,1,d) K
        {tileB, tileS, 1, qkRopeHeadDim / 2, 2}  // (b,s,n,d//2,2)
    };

    RopeTileShapeConfig ropeCfg{
        {128, 128},
        {32, 128, 128},
        {16, 128, 128, 128}
    };

    SymbolicScalar b = GetInputShape(tokenX, 0);
    SymbolicScalar s = GetInputShape(tokenX, 1);
    SymbolicScalar bLoop = b / tileB;
    SymbolicScalar sLoop = s / tileS;
    SymbolicScalar bsLoop = (b * s + tileBS - 1) / tileBS;

    Tensor x2D(tokenX.GetDataType(), {b * s, h}, "x2D");
    Tensor cos2D(cos.GetDataType(), {b * s, qkRopeHeadDim}, "cos2D");
    Tensor sin2D(sin.GetDataType(), {b * s, qkRopeHeadDim}, "sin2D");
    Tensor kCacheIndex2D(cacheIndex.GetDataType(), {b * s, 1}, "kCacheIndex2D");

    LOOP("LOOP_MLA_RESHAPE", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
        (void)batchId;
        ReshapeInplace(tokenX, x2D);
        ReshapeInplace(cos, cos2D);
        ReshapeInplace(sin, sin2D);
        ReshapeInplace(cacheIndex, kCacheIndex2D);
    }

    Tensor kvCacheRes(kvCache.GetDataType(), {blockNum * blockSize * n2, kvLoraRank}, "kvCacheRes");
    Tensor krCacheRes(krCache.GetDataType(), {blockNum * blockSize * n2, qkRopeHeadDim}, "krCacheRes");
    LOOP("MLA_RESHAPE", FunctionType::DYNAMIC_LOOP, unUsedIdx, LoopRange(1)) {
        (void)unUsedIdx;
        ReshapeInplace(kvCache, kvCacheRes);
        ReshapeInplace(krCache, krCacheRes);
    }

    LOOP("MLA_BS_Loop", FunctionType::DYNAMIC_LOOP, bsIdx, LoopRange(0, bsLoop, 1)) {
        SymbolicScalar bsOffset = bsIdx * tileBS;
        std::vector<SymbolicScalar> outputOffset = {bsOffset, 0, 0};
        TileShape::Current().SetVecTile({tileBS, 128});
        auto xView = View(x2D, {tileBS, h}, {bsOffset, 0});
        xView = Cast(Cast(xView, DataType::DT_FP32), dType);

        auto qKv = PreCompute2D(xView, wDq, wUqQr, wDkvKr, gammaCq, epsilonCq, quantInputs);
        Tensor q = qKv[0];     // [b*s, n*qHeadDim]
        Tensor kvTmp = qKv[1]; // [b*s, kvLoraRank+qkRopeHeadDim]
        auto qTmp = Reshape(q, {tileBS, n, qHeadDim});

        /******** q ********/
        ConfigManager::Instance().SetSemanticLabel("Prepare_qNope");
        Tensor qNope = View(qTmp, {tileBS, n, qkNopeHeadDim}, {0, 0, 0}); // [b,s,n,qkNopeHeadDim]
        std::vector<int64_t> tileShape = {std::min(32, tileBS), 1, qkNopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);
        Tensor qNopeTrans = Transpose(qNope, {0, 1}); // [n,bs,qkNopeHeadDim]

        int c0 = 16;
        int m = (std::min(32, tileBS) + c0 - 1) / c0 * c0;
        ConfigManager::Instance().SetSemanticLabel("Matmul_qNope_wUk");
        TileShape::Current().SetCubeTile({m, m}, {128, 128}, {128, 128});
        // bmm: (n,bs,qkNopeHeadDim) @ (n, qkNopeHeadDim, kvLoraRank) = (n, bs, kvLoraRank)
        Tensor qNopeNew = Matrix::BatchMatmul(dType, qNopeTrans, wUk);

        ConfigManager::Instance().SetSemanticLabel("queryOut");
        tileShape = {1, std::min(32, tileBS), kvLoraRank};
        TileShape::Current().SetVecTile(tileShape);
        Tensor qNopeNewTrans = Transpose(qNopeNew, {0, 1}); // [bs,n,kvLoraRank]
        ConfigManager::Instance().SetSemanticLabel("Assemble_queryOut");
        TileShape::Current().SetVecTile({1, 32, 128});
        Assemble(qNopeNewTrans, outputOffset, queryOut); // output1

        Tensor qPeView = View(qTmp, {tileBS, n, qkRopeHeadDim}, {0, 0, qkNopeHeadDim});
        cos2D = View(cos2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0});
        sin2D = View(sin2D, {tileBS, qkRopeHeadDim}, {bsOffset, 0});
        auto qRopeView = Rope3DV2(qPeView, cos2D, sin2D, ropeCfg);
        ConfigManager::Instance().SetSemanticLabel("Assemble_qRope");
        TileShape::Current().SetVecTile({1, 32, 64});
        Assemble(qRopeView, outputOffset, queryRopeOut); // output2

        /******** RoPE ********/
        TileShape::Current().SetVecTile({2, 512});
        ConfigManager::Instance().SetSemanticLabel("RotaryPosEmb");
        Tensor kPeView = View(kvTmp, {tileBS, qkRopeHeadDim}, {0, kvLoraRank}); // [b*s,qkRopeHeadDim]
        auto kRopeView = RopeV2(kPeView, cos2D, sin2D, ropeCfg);

        Tensor kRopeRes = Reshape(kRopeView, {tileBS, 1, 1, qkRopeHeadDim});
        /******** krCache ********/
        ConfigManager::Instance().SetSemanticLabel("ScatterUpdate_krCache");
        tileShape = {1, qkRopeHeadDim};
        TileShape::Current().SetVecTile(tileShape);

        auto index = View(kCacheIndex2D, {tileBS, 1}, {bsOffset, 0});
        // krCache: [blockNum * blockSize * n2, qkRopeHeadDim], output4
        TileShape::Current().SetVecTile(4, 128, 128, 128);
        krCacheOut = ScatterUpdate(krCache, index, kRopeRes, -2, cacheMode, blockSize);

        Tensor compressedKv = View(kvTmp, {tileBS, kvLoraRank}, {0, 0}); // [b*s,kvLoraRank]
        tileShape = {2, 512};
        ConfigManager::Instance().SetSemanticLabel("RmsNorm_compressedKv");
        TileShape::Current().SetVecTile(tileShape);
        Tensor kNope = RmsNorm(compressedKv, gammaCkv, epsilonCkv); // [b*s,kvLoraRank]
        kNope = Reshape(kNope, {tileBS, 1, 1, kvLoraRank});

        /******** kvCache ********/
        ConfigManager::Instance().SetSemanticLabel("ScatterUpdate_kvCache");
        TileShape::Current().SetVecTile(4, 128, 128, 512);
        // kvCache: [blockNum * blockSize * n2, kvLoraRank], output3
        kvCacheOut = ScatterUpdate(kvCache, index, kNope, -2, cacheMode, blockSize);

        TileShape::Current().SetVecTile({tileBS, q_lora_rank});
        auto rms3D = Cast(Cast(qKv[2], DataType::DT_FP32), dType);
        Assemble(rms3D, {bsOffset, 0}, rmsRes);
    }
}

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const MlaTileConfig &tileConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq, float epsilonCkv,
    std::string cacheMode) {
    FunctionConfig funConfig;
    FUNCTION("main", funConfig,
        {tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs.dequantScaleWDq, quantInputs.dequantScaleWDkvKr, quantInputs.dequantScaleWUqQr,
            quantInputs.smoothScalesCq},
        {queryOut, queryRopeOut, kvCacheOut, krCacheOut}) {
        // compute
        Tensor rmsRes(
            tokenX.GetDataType(), {GetInputShape(tokenX, 0), GetInputShape(tokenX, 1), wDq.GetShape()[1]}, "rmsRes");
        MlaPrologCompute(tokenX, wDq, wUqQr, wUk, wDkvKr, gammaCq, gammaCkv, sin, cos, cacheIndex, kvCache, krCache,
            quantInputs, tileConfig, queryOut, queryRopeOut, kvCacheOut, krCacheOut, rmsRes, epsilonCq, epsilonCkv,
            cacheMode);
    }
}

} // namespace npu::tile_fwk
