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
constexpr int64_t MAX_PROD_RAW_SEQ = 4;
constexpr int64_t MAX_PROD_TOPK = 2048;
constexpr int64_t MAX_PROD_POOL_SIZE = 16384;
constexpr int64_t MAX_PROD_KV_DIM = 512;
constexpr int64_t MAX_PROD_ROPE_DIM = 128;
constexpr int64_t WAYS_PER_SET = 16;
constexpr int64_t MIN_SELECTION_BLOCK_SIZE = 16;
constexpr int64_t MAX_SELECTION_BLOCK_SIZE = 128;

void CheckDsaInstallInputs(
    const at::Tensor& installRecords,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    const at::Tensor& selectionKvBlockTable,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t metadataUpdate)
{
    TORCH_CHECK(installRecords.dtype() == at::kChar, "install_records must be int8.");
    TORCH_CHECK(selectionKvCache.dtype() == selectionKRope.dtype() &&
                    selectionKvCache.dtype() == poolKvCache.dtype() &&
                    selectionKvCache.dtype() == poolKRope.dtype(),
        "DsaInstall requires selection/pool KV and rope tensors to share dtype.");
    TORCH_CHECK(selectionKvCache.dtype() == at::kHalf || selectionKvCache.dtype() == at::kBFloat16,
        "DsaInstall supports FP16/BF16 payload only.");
    TORCH_CHECK(selectionKvBlockTable.dtype() == at::kInt,
        "selection_kv_block_table must be int32.");
    TORCH_CHECK(poolIds.dtype() == at::kInt, "pool_ids must be int32.");
    TORCH_CHECK(idToSlot.dtype() == at::kInt, "id_to_slot must be int32.");
    TORCH_CHECK(lruCounter.dtype() == at::kInt, "lru_counter must be int32.");
    TORCH_CHECK(installRecords.dim() == 1 && installRecords.size(0) > 0,
        "install_records must be a non-empty 1D int8 buffer.");
    TORCH_CHECK(rawSeq > 0 && rawSeq <= MAX_PROD_RAW_SEQ,
        "DsaInstall raw_seq ", rawSeq, " must be in (0, ", MAX_PROD_RAW_SEQ, "].");
    TORCH_CHECK(topK > 0 && topK <= MAX_PROD_TOPK && (topK % 8) == 0,
        "DsaInstall topk ", topK, " must be positive, <= ", MAX_PROD_TOPK,
        ", and divisible by 8.");
    TORCH_CHECK(selectionBlockSize >= MIN_SELECTION_BLOCK_SIZE &&
                    selectionBlockSize <= MAX_SELECTION_BLOCK_SIZE &&
                    (selectionBlockSize & (selectionBlockSize - 1)) == 0,
        "DsaInstall selection_block_size must be a power of two in [",
        MIN_SELECTION_BLOCK_SIZE, ", ", MAX_SELECTION_BLOCK_SIZE, "].");
    TORCH_CHECK(topK % selectionBlockSize == 0,
        "DsaInstall topk must be divisible by selection_block_size.");
    TORCH_CHECK(metadataUpdate == 0 || metadataUpdate == 1,
        "DsaInstall metadata_update must be 0 or 1.");
    TORCH_CHECK(selectionKvCache.dim() == 3 && selectionKRope.dim() == 3 &&
                    poolKvCache.dim() == 3 && poolKRope.dim() == 3,
        "selection/pool KV and rope tensors must be 3D.");
    TORCH_CHECK(selectionKvBlockTable.dim() == 2,
        "selection_kv_block_table must be [batch*raw_seq, blocks].");

    const int64_t batch = poolKvCache.size(0);
    const int64_t poolSize = poolKvCache.size(1);
    const int64_t selectionRows = selectionKvCache.size(0) * selectionKvCache.size(1);
    const int64_t requiredSelectionRows = batch * rawSeq * topK;
    TORCH_CHECK(batch > 0, "DsaInstall batch must be positive.");
    TORCH_CHECK(poolSize > 0 && poolSize <= MAX_PROD_POOL_SIZE && (poolSize % WAYS_PER_SET) == 0,
        "DsaInstall pool size unsupported.");
    TORCH_CHECK(selectionRows >= requiredSelectionRows,
        "DsaInstall selection rows must cover batch*raw_seq*topk.");
    TORCH_CHECK(selectionKRope.size(0) == selectionKvCache.size(0) &&
                    selectionKRope.size(1) == selectionKvCache.size(1),
        "selection KV and rope row layout must match.");
    TORCH_CHECK(selectionKvBlockTable.size(0) == batch * rawSeq &&
                    selectionKvBlockTable.size(1) * selectionBlockSize >= topK,
        "selection_kv_block_table shape must match batch/raw_seq/topk.");
    TORCH_CHECK(poolKRope.size(0) == batch && poolKRope.size(1) == poolSize,
        "pool KV and rope shape must match.");
    TORCH_CHECK(selectionKvCache.size(2) > 0 && selectionKvCache.size(2) <= MAX_PROD_KV_DIM &&
                    poolKvCache.size(2) == selectionKvCache.size(2),
        "DsaInstall kv dim unsupported or mismatched.");
    TORCH_CHECK(selectionKRope.size(2) > 0 && selectionKRope.size(2) <= MAX_PROD_ROPE_DIM &&
                    poolKRope.size(2) == selectionKRope.size(2),
        "DsaInstall rope dim unsupported or mismatched.");
    TORCH_CHECK(poolIds.dim() == 2 && poolIds.size(0) == batch && poolIds.size(1) == poolSize,
        "pool_ids must be [batch, pool_size].");
    TORCH_CHECK(idToSlot.dim() == 2 && idToSlot.size(0) == batch,
        "id_to_slot must be [batch, id_range].");
    TORCH_CHECK(idToSlot.size(1) > 0,
        "DsaInstall id_range ", idToSlot.size(1), " must be positive.");
    const int64_t numSets = poolSize / WAYS_PER_SET;
    TORCH_CHECK((numSets & (numSets - 1)) == 0,
        "DsaInstall requires power-of-two num_sets.");
    TORCH_CHECK(lruCounter.dim() == 2 && lruCounter.size(0) == batch && lruCounter.size(1) == numSets,
        "lru_counter must be [batch, num_sets].");
}

#ifndef DSA_PLAN_META_ONLY
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> ConstructDsaInstallFunctionalOutputs(
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter)
{
    return std::make_tuple(
        poolKvCache.clone(),
        poolKRope.clone(),
        poolIds.clone(),
        idToSlot.clone(),
        lruCounter.clone());
}
#endif
}  // namespace

#ifndef DSA_PLAN_META_ONLY
void dsa_install_npu(
    const at::Tensor& installRecords,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    const at::Tensor& selectionKvBlockTable,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t metadataUpdate)
{
    CheckDsaInstallInputs(installRecords, selectionKvCache, selectionKRope, selectionKvBlockTable, poolKvCache,
        poolKRope, poolIds, idToSlot, lruCounter, rawSeq, topK, selectionBlockSize, metadataUpdate);
    EXEC_NPU_CMD_V1(aclnnDsaInstall,
        installRecords,
        selectionKvCache,
        selectionKRope,
        selectionKvBlockTable,
        poolKvCache,
        poolKRope,
        poolIds,
        idToSlot,
        lruCounter,
        rawSeq,
        topK,
        selectionBlockSize,
        metadataUpdate);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dsa_install_functional_npu(
    const at::Tensor& installRecords,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    const at::Tensor& selectionKvBlockTable,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t metadataUpdate)
{
    CheckDsaInstallInputs(installRecords, selectionKvCache, selectionKRope, selectionKvBlockTable, poolKvCache,
        poolKRope, poolIds, idToSlot, lruCounter, rawSeq, topK, selectionBlockSize, metadataUpdate);
    auto outputs = ConstructDsaInstallFunctionalOutputs(poolKvCache, poolKRope, poolIds, idToSlot, lruCounter);
    EXEC_NPU_CMD_V1(aclnnDsaInstall,
        installRecords,
        selectionKvCache,
        selectionKRope,
        selectionKvBlockTable,
        std::get<0>(outputs),
        std::get<1>(outputs),
        std::get<2>(outputs),
        std::get<3>(outputs),
        std::get<4>(outputs),
        rawSeq,
        topK,
        selectionBlockSize,
        metadataUpdate);
    return outputs;
}
#endif

void dsa_install_meta(
    const at::Tensor& installRecords,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    const at::Tensor& selectionKvBlockTable,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t metadataUpdate)
{
    CheckDsaInstallInputs(installRecords, selectionKvCache, selectionKRope, selectionKvBlockTable, poolKvCache,
        poolKRope, poolIds, idToSlot, lruCounter, rawSeq, topK, selectionBlockSize, metadataUpdate);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dsa_install_functional_meta(
    const at::Tensor& installRecords,
    const at::Tensor& selectionKvCache,
    const at::Tensor& selectionKRope,
    const at::Tensor& selectionKvBlockTable,
    const at::Tensor& poolKvCache,
    const at::Tensor& poolKRope,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t topK,
    int64_t selectionBlockSize,
    int64_t metadataUpdate)
{
    CheckDsaInstallInputs(installRecords, selectionKvCache, selectionKRope, selectionKvBlockTable, poolKvCache,
        poolKRope, poolIds, idToSlot, lruCounter, rawSeq, topK, selectionBlockSize, metadataUpdate);
    return std::make_tuple(
        at::empty_like(poolKvCache),
        at::empty_like(poolKRope),
        at::empty_like(poolIds),
        at::empty_like(idToSlot),
        at::empty_like(lruCounter));
}
}  // namespace custom

#ifndef DSA_PLAN_META_ONLY
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("dsa_install", &custom::dsa_install_npu);
    m.impl("dsa_install_functional", &custom::dsa_install_functional_npu);
}
#endif

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("dsa_install", &custom::dsa_install_meta);
    m.impl("dsa_install_functional", &custom::dsa_install_functional_meta);
}
