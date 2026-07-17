/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#ifndef DSA_PLAN_META_ONLY
#include "ops_common.h"
#endif

namespace custom {
#ifndef DSA_PLAN_META_ONLY
using namespace at_npu::native;
#endif

namespace {
constexpr int64_t PLAN_WIDTH = 2;
constexpr int64_t LEGACY_MAX_BATCH = 16;
constexpr int64_t LEGACY_MAX_RAW_SEQ = 4;
constexpr int64_t LEGACY_MAX_TOPK = 64;
constexpr int64_t LEGACY_MAX_ROWS = 256;
constexpr int64_t LEGACY_MAX_FULL_SEQ = 512;
constexpr int64_t LEGACY_MAX_POOL_SIZE = 256;
constexpr int64_t PROD_MAX_RAW_SEQ = 4;
constexpr int64_t PROD_MAX_TOPK = 2048;
constexpr int64_t PROD_MAX_POOL_SIZE = 16384;
constexpr int64_t TOPK_GRANULARITY = 8;
constexpr int64_t MIN_SELECTION_BLOCK_SIZE = 16;
constexpr int64_t MAX_SELECTION_BLOCK_SIZE = 128;
constexpr int64_t MAX_KV_DIM = 512;
constexpr int64_t MAX_ROPE_DIM = 128;

void CheckDsaServeInputs(
    const at::Tensor& plan,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullKRope,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t compactLayout)
{
    TORCH_CHECK(compactLayout == 0 || compactLayout == 1,
        "DsaServe compact_layout must be 0 or 1.");
    const bool compact = (compactLayout != 0);
    const int64_t maxRawSeq = compact ? PROD_MAX_RAW_SEQ : LEGACY_MAX_RAW_SEQ;
    const int64_t maxTopK = compact ? PROD_MAX_TOPK : LEGACY_MAX_TOPK;
    const int64_t maxPoolSize = compact ? PROD_MAX_POOL_SIZE : LEGACY_MAX_POOL_SIZE;
    TORCH_CHECK(plan.dtype() == at::kInt, "plan must be int32.");
    TORCH_CHECK(fullKvCache.dtype() == fullKRope.dtype() &&
                    fullKvCache.dtype() == poolKvCache.dtype() &&
                    fullKvCache.dtype() == poolKRope.dtype() &&
                    fullKvCache.dtype() == selectionKvCache.dtype() &&
                    fullKvCache.dtype() == selectionKRope.dtype(),
        "DsaServe requires full/pool/selection KV and rope tensors to share dtype.");
    TORCH_CHECK(fullKvCache.dtype() == at::kHalf || fullKvCache.dtype() == at::kBFloat16,
        "DsaServe supports FP16/BF16 payload only.");
    TORCH_CHECK(rawSeq > 0 && rawSeq <= maxRawSeq,
        "DsaServe raw_seq ", rawSeq, " must be in (0, ", maxRawSeq, "] for compact_layout=", compactLayout, ".");
    TORCH_CHECK(topK > 0 && topK <= maxTopK && (topK % TOPK_GRANULARITY) == 0,
        "DsaServe topK ", topK, " must be positive, <= ", maxTopK,
        ", and divisible by ", TOPK_GRANULARITY, " for compact_layout=", compactLayout, ".");
    TORCH_CHECK(selectionBlockSize >= MIN_SELECTION_BLOCK_SIZE &&
                    selectionBlockSize <= MAX_SELECTION_BLOCK_SIZE &&
                    (selectionBlockSize & (selectionBlockSize - 1)) == 0,
        "DsaServe selection_block_size ", selectionBlockSize,
        " must be a power of two in [", MIN_SELECTION_BLOCK_SIZE, ", ", MAX_SELECTION_BLOCK_SIZE, "].");
    TORCH_CHECK(!compact || (topK % selectionBlockSize) == 0,
        "DsaServe compact topK must be divisible by selection_block_size.");
    TORCH_CHECK(plan.dim() == 2 && plan.size(1) == PLAN_WIDTH,
        "plan must be [rows, 2].");
    TORCH_CHECK(plan.size(0) > 0, "DsaServe rows must be positive.");
    TORCH_CHECK(compact || plan.size(0) <= LEGACY_MAX_ROWS,
        "DsaServe rows ", plan.size(0), " exceed legacy cap ", LEGACY_MAX_ROWS,
        " for compact_layout=", compactLayout, ".");
    TORCH_CHECK(fullKvCache.dim() == 3 && fullKRope.dim() == 3 &&
                    poolKvCache.dim() == 3 && poolKRope.dim() == 3 &&
                    selectionKvCache.dim() == 3 && selectionKRope.dim() == 3,
        "full/pool/selection KV and rope tensors must be 3D [batch, entries, dim].");
    const int64_t batch = fullKvCache.size(0);
    TORCH_CHECK(batch > 0, "DsaServe batch must be positive.");
    TORCH_CHECK(compact || batch <= LEGACY_MAX_BATCH,
        "DsaServe batch ", batch, " exceeds legacy cap ", LEGACY_MAX_BATCH,
        " for compact_layout=", compactLayout, ".");
    TORCH_CHECK(plan.size(0) == batch * rawSeq * topK,
        "plan rows must equal batch*raw_seq*topK.");
    TORCH_CHECK(fullKRope.size(0) == batch && poolKvCache.size(0) == batch && poolKRope.size(0) == batch,
        "all DsaServe payload tensors must share batch.");
    TORCH_CHECK(fullKvCache.size(1) > 0 && (compact || fullKvCache.size(1) <= LEGACY_MAX_FULL_SEQ),
        "DsaServe full_seq unsupported for compact_layout=", compactLayout, ".");
    TORCH_CHECK(fullKRope.size(1) == fullKvCache.size(1),
        "full rope seq must match full kv seq.");
    TORCH_CHECK(poolKvCache.size(1) > 0 && poolKvCache.size(1) <= maxPoolSize &&
                    poolKRope.size(1) == poolKvCache.size(1),
        "DsaServe pool size unsupported or mismatched for compact_layout=", compactLayout, ".");
    TORCH_CHECK(fullKvCache.size(2) > 0 && fullKvCache.size(2) <= MAX_KV_DIM &&
                    poolKvCache.size(2) == fullKvCache.size(2),
        "DsaServe kv dim unsupported or mismatched.");
    TORCH_CHECK(fullKRope.size(2) > 0 && fullKRope.size(2) <= MAX_ROPE_DIM &&
                    poolKRope.size(2) == fullKRope.size(2),
        "DsaServe rope dim unsupported or mismatched.");
    const int64_t selectionBlocks =
        compact ? ((plan.size(0) + selectionBlockSize - 1) / selectionBlockSize) : plan.size(0);
    TORCH_CHECK(selectionKvCache.size(0) == selectionBlocks &&
                    selectionKvCache.size(1) == selectionBlockSize &&
                    selectionKvCache.size(2) == fullKvCache.size(2),
        "DsaServe selection_kv_cache shape must match computed selected KV output.");
    TORCH_CHECK(selectionKRope.size(0) == selectionBlocks &&
                    selectionKRope.size(1) == selectionBlockSize &&
                    selectionKRope.size(2) == fullKRope.size(2),
        "DsaServe selection_k_rope shape must match computed selected rope output.");
}

std::tuple<at::Tensor, at::Tensor> ConstructDsaServeOutputs(
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope)
{
    return std::make_tuple(at::empty_like(selectionKvCache), at::empty_like(selectionKRope));
}
}  // namespace

#ifndef DSA_PLAN_META_ONLY
void dsa_serve_npu(
    const at::Tensor& plan,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullKRope,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t compactLayout)
{
    CheckDsaServeInputs(plan, fullKvCache, fullKRope, poolKvCache, poolKRope, selectionKvCache, selectionKRope,
        rawSeq, topK, selectionBlockSize, compactLayout);
    EXEC_NPU_CMD_V1(aclnnDsaServe,
        plan,
        fullKvCache,
        fullKRope,
        poolKvCache,
        poolKRope,
        selectionKvCache,
        selectionKRope,
        rawSeq,
        topK,
        selectionBlockSize,
        compactLayout);
}

std::tuple<at::Tensor, at::Tensor> dsa_serve_functional_npu(
    const at::Tensor& plan,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullKRope,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t compactLayout)
{
    CheckDsaServeInputs(plan, fullKvCache, fullKRope, poolKvCache, poolKRope, selectionKvCache, selectionKRope,
        rawSeq, topK, selectionBlockSize, compactLayout);
    auto outputs = ConstructDsaServeOutputs(selectionKvCache, selectionKRope);
    EXEC_NPU_CMD_V1(aclnnDsaServe,
        plan,
        fullKvCache,
        fullKRope,
        poolKvCache,
        poolKRope,
        std::get<0>(outputs),
        std::get<1>(outputs),
        rawSeq,
        topK,
        selectionBlockSize,
        compactLayout);
    return outputs;
}
#endif

void dsa_serve_meta(
    const at::Tensor& plan,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullKRope,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t compactLayout)
{
    CheckDsaServeInputs(plan, fullKvCache, fullKRope, poolKvCache, poolKRope, selectionKvCache, selectionKRope,
        rawSeq, topK, selectionBlockSize, compactLayout);
}

std::tuple<at::Tensor, at::Tensor> dsa_serve_functional_meta(
    const at::Tensor& plan,
    const at::Tensor& fullKvCache,
    const at::Tensor& fullKRope,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t compactLayout)
{
    CheckDsaServeInputs(plan, fullKvCache, fullKRope, poolKvCache, poolKRope, selectionKvCache, selectionKRope,
        rawSeq, topK, selectionBlockSize, compactLayout);
    return std::make_tuple(at::empty_like(selectionKvCache), at::empty_like(selectionKRope));
}
}  // namespace custom

#ifndef DSA_PLAN_META_ONLY
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("dsa_serve", &custom::dsa_serve_npu);
    m.impl("dsa_serve_functional", &custom::dsa_serve_functional_npu);
}
#endif

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("dsa_serve", &custom::dsa_serve_meta);
    m.impl("dsa_serve_functional", &custom::dsa_serve_functional_meta);
}
