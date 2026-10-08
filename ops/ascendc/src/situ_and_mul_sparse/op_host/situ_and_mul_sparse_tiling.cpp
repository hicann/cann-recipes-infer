#include "situ_and_mul_sparse_tiling.h"
#include "register/op_impl_registry.h"
#include "error/ops_error.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
// fc6 (scheme r1-restore-full-blockdim-dbuf-rowpipeline-tilecols3072):
// 2048 -> 3072, so chunkElements = min(EPR, 3072) gives every EPR = 3072 row
// a single whole-row chunk. The tileRows bound formula below is unchanged
// and stays self-consistent (ubTileLimit = (ubSize*7/8)/30 = 7645, so at
// tileCols = 3072 the tileRows bound is 7645/3072 = 2 >= 1); UB feasibility:
// 30 B/elem * min(EPR, 3072) <= 92160 B <= ubSize*7/8 = 229376 B.
constexpr int64_t MAX_TILE_COLS = 3072;
constexpr int64_t MAX_TILE_ROWS = 4;
// Per tile element (one gate element): in queue depth 2 x (gate+up), out queue
// depth 2, 4 float work spaces and 1 round buffer; both supported dtypes are
// 2 bytes wide.
constexpr int64_t BYTES_PER_TILE_ELEMENT = 2 * 2 * 2 + 2 * 2 + 4 * 4 + 2;

ge::graphStatus TilingForSituAndMulSparse(gert::TilingContext* context)
{
    const auto* shape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, shape, return ge::GRAPH_FAILED);
    const auto& storage = shape->GetStorageShape();
    const int64_t rank = storage.GetDimNum();
    OPS_ERR_IF(rank < 1 || storage.GetDim(rank - 1) <= 0 || storage.GetDim(rank - 1) % 2 != 0,
        OPS_LOG_E(context->GetNodeName(), "x last dimension must be positive and even."),
        return ge::GRAPH_FAILED);
    const int64_t outputElementsPerRow = storage.GetDim(rank - 1) / 2;
    const int64_t totalElements = storage.GetShapeSize() / 2;
    const int64_t totalRows = totalElements / outputElementsPerRow;
    const auto* tokenShape = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context, tokenShape, return ge::GRAPH_FAILED);
    const int64_t expertTokenCount = tokenShape->GetStorageShape().GetShapeSize();
    auto platformInfo = context->GetPlatformInfo();
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        coreNum = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        const auto* compileInfo = context->GetCompileInfo<SituAndMulSparseCompileInfo>();
        OPS_LOG_E_IF_NULL(context, compileInfo, return ge::GRAPH_FAILED);
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
    }
    OPS_ERR_IF(coreNum == 0, OPS_LOG_E(context->GetNodeName(), "AIV core number must be positive."),
        return ge::GRAPH_FAILED);
    const int64_t blockDim = std::min(static_cast<int64_t>(coreNum), totalRows);
    const auto* attrs = context->GetAttrs();
    const float* beta = attrs->GetFloat(0);
    const float* alpha = attrs->GetFloat(1);
    const bool* highPrecision = attrs->GetBool(2);
    OPS_ERR_IF(beta == nullptr || alpha == nullptr || *beta <= 0.0f || *alpha <= 0.0f,
        OPS_LOG_E(context->GetNodeName(), "beta and alpha must be positive."),
        return ge::GRAPH_FAILED);
    const int64_t tileCols = std::min(outputElementsPerRow, MAX_TILE_COLS);
    int64_t tileRows = MAX_TILE_ROWS;
    if (ubSize > 0) {
        const int64_t ubTileLimit = static_cast<int64_t>(ubSize * 7 / 8) / BYTES_PER_TILE_ELEMENT;
        tileRows = std::min(tileRows, std::max<int64_t>(1, ubTileLimit / tileCols));
    }
    tileRows = std::min(tileRows, std::max<int64_t>(1, (totalRows + blockDim - 1) / blockDim));
    SituAndMulSparseTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_outputElementsPerRow(outputElementsPerRow);
    tiling.set_expertTokenCount(expertTokenCount);
    // Registered mapping (pre-existing frozen-tree inconsistency, documented
    // in the run1 conversion evidence): the frozen tiling.cpp filled
    // tileRows/tileCols while tiling.h/kernel define and consume
    // chunkElements; the accepted direct-invoke baseline maps
    // chunkElements := tileCols = min(outputElementsPerRow, MAX_TILE_COLS),
    // and this integration keeps that mapping. tileRows stays a frozen-tiling
    // legacy computation (not consumed by the kernel).
    tiling.set_chunkElements(tileCols);
    tiling.set_beta(*beta);
    tiling.set_alpha(*alpha);
    tiling.set_highPrecision(highPrecision != nullptr && *highPrecision ? 1 : 0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platformInfo != nullptr
        ? platform_ascendc::PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize()
        : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSituAndMulSparse(gert::TilingParseContext* context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SituAndMulSparse", "Tiling parse context is null"),
        return ge::GRAPH_FAILED);
    auto* compileInfo = context->GetCompiledInfo<SituAndMulSparseCompileInfo>();
    OPS_ERR_IF(compileInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SituAndMulSparse", "Compile info is null"),
        return ge::GRAPH_FAILED);
    auto* platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("SituAndMulSparse", "Tiling parse platform info is null"),
        return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->coreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->socVersion = platform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(SituAndMulSparse)
    .Tiling(TilingForSituAndMulSparse)
    .TilingParse<SituAndMulSparseCompileInfo>(TilingPrepareForSituAndMulSparse);
}  // namespace optiling
