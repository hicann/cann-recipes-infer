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

#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"
#include "tilefwk/element.h"
#include "tilefwk/function.h"
#include "tilefwk/tensor.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "lightning_indexer_prolog.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk {

Tensor LayerNorm(const Tensor &x, const Tensor &weight, const Tensor &bias, const int dim) {
    ASSERT(dim == (int)(x.GetShape().size() - 1) || dim == -1) << "We only support LayerNorm for the last dimension";
    ASSERT(x.GetDataType() == DT_FP32);
    constexpr float epsilon = 1e-6f;
    int actualDim = dim < 0 ? dim + x.GetShape().size() : dim;

    // do division first to avoid overflow
    auto xScaled = MulS(x, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto mean = RowSumSingle(xScaled, -1);

    auto diff = Sub(x, mean);
    auto squaredDiff = Mul(diff, diff);
    auto squaredDiffScaled = MulS(squaredDiff, Element(DataType::DT_FP32, 1.0f / x.GetShape()[actualDim]));
    auto var = RowSumSingle(squaredDiffScaled, -1);
    // add epsilon to avoid division by zero
    auto varEps = AddS(var, Element(DT_FP32, epsilon));
    auto stdVar = Sqrt(varEps);
    auto res32 = Div(diff, stdVar);

    auto weight32 = Cast(weight, DT_FP32);
    auto bias32 = Cast(bias, DT_FP32);

    return Add(Mul(res32, weight32), bias32);
}

Tensor RotateHalfValidShape(const Tensor &input) {
    auto shape = input.GetShape();
    auto shapeSize = shape.size();
    ASSERT(shapeSize >= 1) << "rope rotate_half input dim less than 1";
    ASSERT(shape[shapeSize - 1] % NUM2 == 0) << "rope rotate_half last dim shape is even.";

    shape[shapeSize - 1] /= NUM2;
    std::vector<SymbolicScalar> offset1(shapeSize, 0);
    std::vector<SymbolicScalar> offset2(shapeSize, 0);
    offset2[shapeSize - 1] = shape[shapeSize - 1];

    std::vector<SymbolicScalar> validShape = GetDynValidShape(input);
    validShape[shapeSize - 1] = validShape[shapeSize - 1] / NUM2;

    // x1 = [..., : x.shape[-1] // 2]
    // x2 = [..., x.shape[-1] // 2 :]
    Tensor x1 = View(input, shape, validShape, offset1);
    Tensor x2 = View(input, shape, validShape, offset2);

    // cat((-x2, x1), -1)
    return Concat({MulS(x2, Element(x2.GetDataType(), -1.0)), AddS(x1, Element(x1.GetDataType(), 0.0))}, -1);
}

Tensor Rope3D(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
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
    castCos = Reshape(castCos, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]});
    castSin = Reshape(castSin, {x.GetShape()[NUM_VALUE_0], 1, x.GetShape()[NUM_VALUE_2]});

    std::vector<SymbolicScalar> xValidShape = GetDynValidShape(x);
    auto xView = Reshape(castX,
        {x.GetShape()[NUM_VALUE_0], x.GetShape()[NUM_VALUE_1], x.GetShape()[NUM_VALUE_2] / NUM_VALUE_2, NUM_VALUE_2},
        {xValidShape[NUM_VALUE_0], xValidShape[NUM_VALUE_1], xValidShape[NUM_VALUE_2] / NUM_VALUE_2, NUM_VALUE_2});
    TileShape::Current().SetVecTile(1, 32, 128, 128);
    auto xTrans = Transpose(xView, {NUM_VALUE_2, NUM_VALUE_3});
    auto xReSecond = Reshape(xTrans, x.GetShape(), xValidShape);

    TileShape::Current().SetVecTile(1, 32, 128, 128);
    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalfValidShape(xReSecond), castSin));
    auto res = Cast(xEmbed, x.GetDataType());
    return res;
}

Tensor Rope(const Tensor &x, const Tensor &cos, const Tensor &sin, const RopeTileShapeConfig &tileConfig) {
    (void)tileConfig;
    ASSERT(x.GetShape().size() == SHAPE_DIM2 && cos.GetShape().size() == SHAPE_DIM2 &&
           sin.GetShape().size() == SHAPE_DIM2);

    auto seqSize = x.GetShape()[NUM_VALUE_0];
    auto dR = x.GetShape()[NUM_VALUE_1];
    auto xDtype = x.GetDataType();

    TileShape::Current().SetVecTile(tileConfig.twoDim[NUM_VALUE_0], tileConfig.twoDim[NUM_VALUE_1]);
    auto castX = Cast(x, DT_FP32);
    if (x.GetDataType() == DT_FP32) {
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

    auto xEmbed = Add(Mul(xReSecond, castCos), Mul(RotateHalf(xReSecond), castSin));
    auto res = Cast(xEmbed, xDtype);
    return res;
}

void LightningIndexerPrologCompute(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params) {
    SymbolicScalar b = GetInputShape(inputs.x, 0);
    SymbolicScalar seq = GetInputShape(inputs.x, 1);
    int headDim = params.headDim;
    int ropeHeadDim = params.ropeHeadDim;
    int qLoraRank = params.qLoraRank;
    int dim = params.dim;
    int headNum = params.headNum;

    Tensor x2D(inputs.x.GetDataType(), {b * seq, dim}, "x2D");
    Tensor qr2D(inputs.qr.GetDataType(), {b * seq, qLoraRank}, "qr2D");
    Tensor cos2D(inputs.cos.GetDataType(), {b * seq, ropeHeadDim}, "cos2D");
    Tensor sin2D(inputs.sin.GetDataType(), {b * seq, ropeHeadDim}, "sin2D");
    Tensor kCacheIndex2D(inputs.kCacheIndex.GetDataType(), {b * seq, 1}, "kCacheIndex2D");
    Tensor lnW2D(inputs.lnW.GetDataType(), {1, inputs.lnW.GetShape()[0]});
    Tensor lnBias2D(inputs.lnBias.GetDataType(), {1, inputs.lnBias.GetShape()[0]});

    LOOP("LOOP_RESHAPE_IN", FunctionType::DYNAMIC_LOOP, batchId, LoopRange(1)) {
        (void)batchId;
        ReshapeInplace(inputs.x, x2D);
        ReshapeInplace(inputs.qr, qr2D);
        ReshapeInplace(inputs.cos, cos2D);
        ReshapeInplace(inputs.sin, sin2D);
        ReshapeInplace(inputs.kCacheIndex, kCacheIndex2D);
        ReshapeInplace(inputs.lnW, lnW2D);
        ReshapeInplace(inputs.lnBias, lnBias2D);
    }

    std::set<int> unrollList = {1, 2, 4, 8, 16, 32};
    LOOP("IndexerPrologLoop", FunctionType::DYNAMIC_LOOP, bsIdx, LoopRange(b * seq), unrollList) {
        for (int unrollLength : unrollList) {
            UNROLL(unrollLength) {
                int tileBS = unrollLength;
                SymbolicScalar actBS = tileBS;
                auto c1Tile = params.indexerTileConfigs.c1TileShape;
                auto v1Tile = params.indexerTileConfigs.v1TileShape;

                ConfigManager::Instance().SetSemanticLabel("QMatmul");
                TileShape::Current().SetCubeTile({c1Tile[NUM_VALUE_0], c1Tile[NUM_VALUE_1]},
                    {c1Tile[NUM_VALUE_2], c1Tile[NUM_VALUE_3]}, {c1Tile[NUM_VALUE_4], c1Tile[NUM_VALUE_5]}, true);
                auto qrBlock = View(qr2D, {tileBS, qLoraRank}, {actBS, qLoraRank}, {bsIdx, 0});
                // {tileBS, qLoraRank} * {qLoraRank, headNum * headDim} = {tileBS, headNum * headDim}
                auto q32 = Matrix::Matmul<false, false>(DT_FP32, qrBlock, inputs.qW);

                ConfigManager::Instance().SetSemanticLabel("QCast");
                TileShape::Current().SetVecTile(std::min(tileBS, 4), 32, v1Tile[NUM_VALUE_1]);
                auto q = Cast(Reshape(q32, {tileBS, headNum, headDim}), qrBlock.GetDataType());
                Tensor qRope = View(q, {tileBS, headNum, ropeHeadDim}, {actBS, headNum, ropeHeadDim}, {0, 0, 0});
                Tensor qNope = View(q, {tileBS, headNum, headDim - ropeHeadDim},
                    {actBS, headNum, headDim - ropeHeadDim}, {0, 0, ropeHeadDim});
                qNope = Cast(Cast(qNope, DT_FP32), qNope.GetDataType());

                ConfigManager::Instance().SetSemanticLabel("KMatmul");
                auto c2Tile = params.indexerTileConfigs.c2TileShape;
                TileShape::Current().SetCubeTile({c2Tile[NUM_VALUE_0], c2Tile[NUM_VALUE_1]},
                    {c2Tile[NUM_VALUE_2], c2Tile[NUM_VALUE_3]}, {c2Tile[NUM_VALUE_4], c2Tile[NUM_VALUE_5]}, true);
                TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1], v1Tile[NUM_VALUE_1]);
                auto xBlock = View(x2D, {tileBS, dim}, {actBS, dim}, {bsIdx, 0});
                // {tileBS, dim} * {dim, headNum} = {tileBS, headNum}
                auto weights = Matrix::Matmul<false, false>(inputs.x.GetDataType(), xBlock, inputs.projW);
                Assemble(weights, {bsIdx, 0}, outputs.weight);

                // {tileBS, dim} * {dim, headDim} = {tileBS, headDim}
                auto k = Matrix::Matmul<false, false>(DT_FP32, xBlock, inputs.kW);
                k = Cast(LayerNorm(k, lnW2D, lnBias2D, -1), xBlock.GetDataType()); // {tileBS, headDim}
                Tensor kRope = View(k, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {0, 0});
                Tensor kNope =
                    View(k, {tileBS, headDim - ropeHeadDim}, {actBS, headDim - ropeHeadDim}, {0, ropeHeadDim});

                TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1], v1Tile[NUM_VALUE_2]);
                cos2D = View(cos2D, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {bsIdx, 0});
                sin2D = View(sin2D, {tileBS, ropeHeadDim}, {actBS, ropeHeadDim}, {bsIdx, 0});
                ConfigManager::Instance().SetSemanticLabel("QRope");
                // qRope{tileBS * headNum, ropeHeadDim}  cos{tileBS, ropeHeadDim}   sin{tileBS, ropeHeadDim}
                ConfigManager::Instance().SetSemanticLabel("KRope");
                auto qRoped = Rope3D(qRope, cos2D, sin2D, params.ropeTileConfigs); // {tileBS, headNum, ropeHeadDim}
                TileShape::Current().SetVecTile(v1Tile[NUM_VALUE_0], v1Tile[NUM_VALUE_1]);
                // kRope{tileBS, ropeHeadDim}  cos{tileBS, ropeHeadDim}   sin{tileBS, ropeHeadDim}
                auto kRoped = Rope(kRope, cos2D, sin2D, params.ropeTileConfigs); // {tileBS, ropeHeadDim}

                ConfigManager::Instance().SetSemanticLabel("KAssemble");
                TileShape::Current().SetVecTile(tileBS, 128, 128, 128);
                Assemble(qRoped, {bsIdx, 0, 0}, outputs.query);
                Assemble(qNope, {bsIdx, 0, ropeHeadDim}, outputs.query);

                TileShape::Current().SetVecTile(tileBS, 256);
                auto kType = kNope.GetDataType();
                kNope = Cast(Cast(kNope, DT_FP32), kType);
                auto kUpdate = Concat({kRoped, kNope}, -1); // {tileBS, headDim}
                auto kUpdate4D = Reshape(kUpdate, {tileBS, 1, 1, headDim});
                auto index = View(kCacheIndex2D, {tileBS, 1}, {actBS, 1}, {bsIdx, 0});

                TileShape::Current().SetVecTile(tileBS, 128, 128, 128);
                outputs.kCacheOut = ScatterUpdate(inputs.kCache, index, kUpdate4D, -2, "PA_BSND", params.blockSize);
            }
        }
    }
}

void LightningIndexerProlog(
    const IndexerPrologInput &inputs, IndexerPrologOutput &outputs, const IndexerShapeParams &params) {
    config::SetCodeGenConfig("SUPPORT_DYNAMIC_UNALIGNED", true);
    FunctionConfig funConfig;
    FUNCTION("LightningIndexerProlog", funConfig,
        {
            inputs.x, inputs.qr, inputs.qW, inputs.kW, inputs.projW, inputs.lnW, inputs.lnBias, inputs.cos, inputs.sin,
            inputs.kCache, inputs.kCacheIndex, inputs.blockTable
    },
        {outputs.query, outputs.weight}, {{outputs.kCacheOut, inputs.kCache}}) {
        LightningIndexerPrologCompute(inputs, outputs, params);
    }
}

void LightningIndexerFP8(
    const Tensor &x, const Tensor &qr, Tensor &qFP8, Tensor &qScale, Tensor &kFP8, Tensor &kScale) {
    (void)x;
    (void)qr;
    (void)qFP8;
    (void)qScale;
    (void)kFP8;
    (void)kScale;
}

} // namespace npu::tile_fwk
