/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef DSA_SERVE_TILING_H_
#define DSA_SERVE_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DsaServeTilingData)
TILING_DATA_FIELD_DEF(int64_t, planRows);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, rawSeq);
TILING_DATA_FIELD_DEF(int64_t, topK);
TILING_DATA_FIELD_DEF(int64_t, fullSeq);
TILING_DATA_FIELD_DEF(int64_t, poolSize);
TILING_DATA_FIELD_DEF(int64_t, selectionBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kvDim);
TILING_DATA_FIELD_DEF(int64_t, ropeDim);
TILING_DATA_FIELD_DEF(int64_t, compactLayout);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DsaServe, DsaServeTilingData)

struct DsaServeCompileInfo {};

class DsaServeTiling {
public:
    explicit DsaServeTiling(gert::TilingContext* context) : context_(context) {}
    ~DsaServeTiling() {}

    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeInfo();
    ge::graphStatus PostTiling();

    DsaServeTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    int64_t coreNum_ = 1;
    int64_t batch_ = 0;
    int64_t rawSeq_ = 1;
    int64_t topK_ = 1;
    int64_t planRows_ = 0;
    int64_t fullSeq_ = 0;
    int64_t poolSize_ = 0;
    int64_t selectionBlockSize_ = 16;
    int64_t kvDim_ = 0;
    int64_t ropeDim_ = 0;
    int64_t compactLayout_ = 0;
};
}  // namespace optiling

#endif  // DSA_SERVE_TILING_H_
