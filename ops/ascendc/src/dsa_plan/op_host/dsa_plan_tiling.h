/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef DSA_PLAN_TILING_H_
#define DSA_PLAN_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DsaPlanTilingData)
TILING_DATA_FIELD_DEF(int64_t, planElems);
TILING_DATA_FIELD_DEF(int64_t, installElems);
TILING_DATA_FIELD_DEF(int64_t, actualSeqElems);
TILING_DATA_FIELD_DEF(int64_t, poolElems);
TILING_DATA_FIELD_DEF(int64_t, lruElems);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, rawSeq);
TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF(int64_t, numSets);
TILING_DATA_FIELD_DEF(int64_t, waysPerSet);
TILING_DATA_FIELD_DEF(int64_t, poolSize);
TILING_DATA_FIELD_DEF(int64_t, idRange);
TILING_DATA_FIELD_DEF(int64_t, topkPerBatch);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, m1bOrdMicrotaskK);
TILING_DATA_FIELD_DEF(int64_t, installRecordsStride);
TILING_DATA_FIELD_DEF(int64_t, installCountsOffset);
TILING_DATA_FIELD_DEF(int64_t, installRecordsRequiredBytes);
TILING_DATA_FIELD_DEF(int64_t, installRecordsAllocatedBytes);
TILING_DATA_FIELD_DEF(int64_t, compactAivRecordsOff);
TILING_DATA_FIELD_DEF(int64_t, compactAivCountsOff);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DsaPlan, DsaPlanTilingData)

struct DsaPlanCompileInfo {};

class DsaPlanTiling {
public:
    explicit DsaPlanTiling(gert::TilingContext* context) : context_(context) {}
    ~DsaPlanTiling() {}

    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus PostTiling();

    DsaPlanTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 1;
    int64_t planElems_ = 0;
    int64_t installElems_ = 0;
    int64_t actualSeqElems_ = 0;
    int64_t poolElems_ = 0;
    int64_t lruElems_ = 0;
    int64_t batch_ = 0;
    int64_t topK_ = 0;
    int64_t rawSeq_ = 1;
    int64_t numSets_ = 512;
    int64_t poolSize_ = 0;
    int64_t idRange_ = 0;
    int64_t topkPerBatch_ = 0;
};
}  // namespace optiling

#endif  // DSA_PLAN_TILING_H_
