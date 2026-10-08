#ifndef SITU_AND_MUL_SPARSE_TILING_H
#define SITU_AND_MUL_SPARSE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SituAndMulSparseTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalElements);
    TILING_DATA_FIELD_DEF(int64_t, outputElementsPerRow);
    TILING_DATA_FIELD_DEF(int64_t, expertTokenCount);
    TILING_DATA_FIELD_DEF(int64_t, chunkElements);
    TILING_DATA_FIELD_DEF(float, beta);
    TILING_DATA_FIELD_DEF(float, alpha);
    TILING_DATA_FIELD_DEF(int32_t, highPrecision);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SituAndMulSparse, SituAndMulSparseTilingData)

struct SituAndMulSparseCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
};
}  // namespace optiling

#endif
