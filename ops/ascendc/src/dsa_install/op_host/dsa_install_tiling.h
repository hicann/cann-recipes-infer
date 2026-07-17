/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef DSA_INSTALL_TILING_H_
#define DSA_INSTALL_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DsaInstallTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, installWorkerNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, selTopKBlockSize);
TILING_DATA_FIELD_DEF(int64_t, fullKvBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kRopeDim);
TILING_DATA_FIELD_DEF(int64_t, kvCacheDim);
TILING_DATA_FIELD_DEF(int64_t, selKvBlockSize);
TILING_DATA_FIELD_DEF(int64_t, fullMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, selMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, batchsize);
TILING_DATA_FIELD_DEF(int64_t, rawSeq);
TILING_DATA_FIELD_DEF(int64_t, headnum);
TILING_DATA_FIELD_DEF(int64_t, topk);
TILING_DATA_FIELD_DEF(int64_t, poolSize);
TILING_DATA_FIELD_DEF(int64_t, idRange);
TILING_DATA_FIELD_DEF(int64_t, numSets);
TILING_DATA_FIELD_DEF(int64_t, waysPerSet);
TILING_DATA_FIELD_DEF(int64_t, maxRecordsPerSeq);
TILING_DATA_FIELD_DEF(int64_t, stageTileSets);
TILING_DATA_FIELD_DEF(int64_t, installRecordsStride);
TILING_DATA_FIELD_DEF(int64_t, installCountsOffset);
TILING_DATA_FIELD_DEF(int64_t, m1bOrdMicrotaskK);
TILING_DATA_FIELD_DEF(int64_t, installRecordsRequiredBytes);
TILING_DATA_FIELD_DEF(int64_t, installRecordsAllocatedBytes);
TILING_DATA_FIELD_DEF(int64_t, compactAivRecordsOff);
TILING_DATA_FIELD_DEF(int64_t, compactAivCountsOff);
TILING_DATA_FIELD_DEF(int64_t, metadataUpdate);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DsaInstall, DsaInstallTilingData)

struct DsaInstallCompileInfo {};

class DsaInstallTiling {
public:
    explicit DsaInstallTiling(gert::TilingContext* context) : context_(context) {}
    ~DsaInstallTiling() {}

    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus PostTiling();

    DsaInstallTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 1;
    int64_t rawSeq_ = 1;
    int64_t topK_ = 2048;
    int64_t selectionBlockSize_ = 128;
    int64_t metadataUpdate_ = 1;
    int64_t batch_ = 0;
    int64_t selectionRows_ = 0;
    int64_t poolSize_ = 0;
    int64_t idRange_ = 0;
    int64_t kvDim_ = 0;
    int64_t ropeDim_ = 0;
    int64_t numSets_ = 512;
    int64_t waysPerSet_ = 16;
    int64_t installRecordsAllocatedBytes_ = 0;
    int64_t workspaceBytes_ = 0;
    int64_t installWorkBufBytes_ = 0;
    uint64_t ubSize_ = 0;
};
}  // namespace optiling

#endif  // DSA_INSTALL_TILING_H_
