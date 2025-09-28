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

#pragma once
#ifndef MLA_DYNAMIC
#define MLA_DYNAMIC

#include "tilefwk/tilefwk.h"
#include "dynamic_nsa_common.h"

namespace npu::tile_fwk {

struct MlaQuantInputs {
    Tensor dequantScaleX;
    Tensor dequantScaleWDq;
    Tensor dequantScaleWUqQr;
    Tensor dequantScaleWDkvKr;
    Tensor quantScaleCkv;
    Tensor quantScaleCkr;
    Tensor smoothScalesCq;
};

struct SimpleParams {
    int b;
    int s;
    int s2;
    int d;
    int m;
    int k;
    int n;
    int n2;
    int right;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    std::string cacheMode;
    int blockSize;
    std::vector<int> vecTile;
    std::vector<int> cubeMTile;
    std::vector<int> cubeKTile;
    std::vector<int> cubeNTile;
    int tileB;
    static SimpleParams getCommonParams() {
        SimpleParams params;
        params.n2 = 1;
        params.s = 1;
        params.h = 7168;               // 7168
        params.q_lora_rank = 1536;     // 1536
        params.kv_lora_rank = 512;     // 512
        params.qk_rope_head_dim = 64;  // 64
        params.qk_nope_head_dim = 128; // 128
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.cacheMode = "BNSD";
        params.blockSize = 128; // 128
        return params;
    }

    static SimpleParams getLowParams() {
        SimpleParams params = getCommonParams();
        params.b = 4;    // 4
        params.n = 32;   // 32
        params.s2 = 256; // 256
        return params;
    }

    static SimpleParams getHighParams() {
        SimpleParams params = getCommonParams();
        params.b = 32;    // 32
        params.n = 128;   // 128
        params.s2 = 4096; // 4096
        return params;
    }
};

std::vector<Tensor> mlaPre(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs, bool splitK = false,
    bool isSmooth = true);

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const RoPETileShapeConfigNew &ropeConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq = 1e-5f,
    float epsilonCkv = 1e-5f, std::string cacheMode = "BNSD", bool splitK = false, bool isSmooth = true);

std::vector<Tensor> PreCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wDkvKr,
    const Tensor &gammaCq, float epsilonCq, const MlaQuantInputs &quantInputs);

void MlaPrologCompute(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut,
    Tensor &rmsRes, float epsilonCq, float epsilonCkv, std::string cacheMode);

void MlaProlog(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, const MlaTileConfig &tileConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, float epsilonCq = 1e-5f,
    float epsilonCkv = 1e-5f, std::string cacheMode = "PA_NZ");

} // namespace npu::tile_fwk

#endif // MLA_DYNAMIC
