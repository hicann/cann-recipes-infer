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
#ifndef DYNAMIC_NSA_COMMON
#define DYNAMIC_NSA_COMMON

#include <cmath>
#include "tilefwk/tilefwk.h"
#include "tilefwk/config_manager.h"
#include "tilefwk/tilefwk_op.h"

namespace npu::tile_fwk {
#define DEBUG_DUMP_TMP_IN_OUT 0

constexpr int NUM_65536 = 65536;
constexpr int SCATTER_UPADATE_DIM = -2;
constexpr int NUM_1 = 1;
constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_4 = 4;
constexpr int NUM_8 = 8;
constexpr int NUM_16 = 16;
constexpr int NUM_20 = 20;
constexpr int NUM_24 = 24;
constexpr int NUM_32 = 32;
constexpr int NUM_48 = 48;
constexpr int NUM_64 = 64;
constexpr int NUM_128 = 128;
constexpr int NUM_256 = 256;
constexpr int NUM_384 = 384;
constexpr int NUM_512 = 512;
constexpr int NUM_1024 = 1024;
constexpr int NUM_1536 = 1536;
constexpr int NUM_1792 = 1792;
constexpr int NUM_4096 = 4096;
constexpr int NUM_6144 = 6144;
constexpr int NUM_8192 = 8192;
constexpr int NUM_7168 = 7168;
constexpr float F_1 = 1.0;
constexpr float F_0 = 0.0;
constexpr float F_NEGA_1 = -1.0;
constexpr double DF_1E_20 = 1e-20;

constexpr const int SHAPE_DIM0 = 0;
constexpr const int SHAPE_DIM1 = 1;
constexpr const int SHAPE_DIM2 = 2;
constexpr const int SHAPE_DIM3 = 3;
constexpr const int SHAPE_DIM4 = 4;
constexpr const int SHAPE_DIM5 = 5;

constexpr int32_t NUM_VALUE_0 = 0;
constexpr int32_t NUM_VALUE_1 = 1;
constexpr int32_t NUM_VALUE_2 = 2;
constexpr int32_t NUM_VALUE_3 = 3;
constexpr int32_t NUM_VALUE_4 = 4;
constexpr int32_t NUM_VALUE_5 = 5;
constexpr int32_t NUM_VALUE_8 = 8;
constexpr int32_t NUM_VALUE_16 = 16;
constexpr int32_t NUM_VALUE_31 = 31;
constexpr int32_t NUM_VALUE_32 = 32;
constexpr int32_t NUM_VALUE_64 = 64;

constexpr const int NUM2 = 2;

const std::string SG_PARALLEL_NUM = "parallel_threshold";
const std::string SG_CYCLE_UPPER_BOUND = "cycle_upper_bound";
const std::string SG_CYCLE_LOWER_BOUND = "cycle_lower_bound";
const std::string L1_REUSE = "l1_reuse";
const std::string L1_REUSE_MAP = "l1_reuse_map";
const std::string CUBE_NBUFFER = "cube_nbuffer";
const std::string CUBE_NBUFFER_MAP = "cube_nbuffer_map";
const std::string COPYIN_THRESHOLD = "copyin_threshold";
const std::string MACHINE_CONFIG = "machine_config";
const std::string OOO_PRESCHEDULE_METHOD_DEFAULT = "ooo_preschedule_method_default";
const std::string OOO_PRESCHEDULE_METHOD = "ooo_preschedule_method";
const std::string NBUFFER_MERGE_MODE = "nbuffer_merge_mode";
const std::string VEC_NBUFFER_MAP = "vec_nbuffer_map";
const std::string SG_CUBE_PARALLEL_NUM = "sg_cube_parallel_num";
const std::string SG_VEC_PARALLEL_NUM = "sg_vec_parallel_num";
const std::string SG_SKIP_PARTITION = "sg_skip_partition";
const std::string NBUFFER_NUM = "nbuffer_num";
const std::string L1_REUSE_NUM = "l1_reuse_num";
const std::string CUBE_NBUFFER_NUM = "cube_nbuffer_num";
const std::string DB_TYPE = "db_type";

struct MlaTileConfig {
    int tileB = 8;
    int tileS = 1;
};

struct SaTileShapeConfig {
    int gTile;
    int sKvTile;
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, TILE_VEC_DIMS> v2TileShape;
};

struct IndexerTile {
    std::vector<int64_t> weightTile;
    std::array<int64_t, TILE_CUBE_DIMS> c1Tile; // (m, M), (k, K), (n, N)
    std::vector<int64_t> v1Tile;
    std::vector<int64_t> topkTile;
    std::vector<int64_t> addsTile;
};

struct IndexerTileShapeConfig {
    std::array<int, TILE_CUBE_DIMS> c1TileShape; // (m, M), (k, K), (n, N)
    std::array<int, SHAPE_DIM4> v1TileShape;
    std::array<int, TILE_CUBE_DIMS> c2TileShape; // (m, M), (k, K), (n, N)
    std::array<int, SHAPE_DIM4> v2TileShape;
};

struct RopeTileShapeConfig {
    std::array<int, SHAPE_DIM2> twoDim;
    std::array<int, SHAPE_DIM3> threeDim;
    std::array<int, SHAPE_DIM4> fourDim;
};

struct NSASimpleParams {
    int b;
    int s1;
    int s2;
    int n1;
    int n2;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    int rope_dim;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int front;
    int near;
    int topk;
    std::string cacheMode;
    int blockSize;
    int winSize;
    int vHeadDim;
    int idx_n_heads;
    int idx_head_dim;
    float softmaxScale;
    float scoreScale;

    int idxHeadDim;
    int idxHeadNum;
    int blockNum;

    float eps;
    MlaTileConfig mlaTileCfg;
    SaTileShapeConfig salTileCfg;
    IndexerTile indexTileCfg;
    IndexerTileShapeConfig indexerTileConfigs;
    RopeTileShapeConfig ropeTileConfigs;

    static NSASimpleParams getCommonParams() {
        NSASimpleParams params;
        params.h = NUM_7168;
        params.q_lora_rank = NUM_1536;
        params.kv_lora_rank = NUM_512;
        params.qk_rope_head_dim = NUM_64;
        params.qk_nope_head_dim = NUM_128;
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.rope_dim = NUM_64;
        params.cmpBlockSize = NUM_32;
        params.cmpStride = NUM_16;
        params.slcBlockSize = NUM_64;
        params.front = NUM_1;
        params.near = NUM_2;
        params.topk = NUM_16;
        params.cacheMode = "BSND";
        params.blockSize = NUM_128;
        params.winSize = NUM_512;
        params.vHeadDim = NUM_128;
        params.eps = 1e-5f;
        params.idx_n_heads = 64;
        params.idx_head_dim = 128;
        params.softmaxScale = static_cast<float>(1.0 / sqrtf((params.kv_lora_rank + params.rope_dim)));
        params.scoreScale = (1.0f / sqrtf(params.idx_n_heads)) * (1.0f / sqrtf(params.idx_head_dim));

        return params;
    }

    static NSASimpleParams getDecodeParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_1;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }

    static NSASimpleParams getMTPParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_2;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }
};

} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA_COMMON
