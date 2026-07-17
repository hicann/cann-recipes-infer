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
constexpr int64_t SRC_INVALID = 0;
constexpr int64_t MAX_PROD_RAW_SEQ = 4;
constexpr int64_t MAX_PROD_TOPK = 2048;
constexpr int64_t MAX_PROD_POOL_SIZE = 16384;
constexpr int64_t WAYS_PER_SET = 16;
constexpr int64_t M1B_ORD_MICROTASK_K = 8;
constexpr int64_t INT32_ALIGN_ELEMS = 8;
constexpr int64_t TOPK_GRANULARITY = M1B_ORD_MICROTASK_K * INT32_ALIGN_ELEMS;
constexpr int64_t INSTALL_RECORD_INT32S = 4;
constexpr int64_t INSTALL_RECORD_BYTES = INSTALL_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t));
constexpr int64_t INSTALL_COUNT_BYTES = 32;
constexpr int64_t SLAB_ALIGN_BYTES = 4096;

int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

int64_t ComputeInstallRecordsAllocatedBytes(int64_t batch, int64_t rawSeq, int64_t topK)
{
    const int64_t recordsPerMt = (topK + M1B_ORD_MICROTASK_K - 1) / M1B_ORD_MICROTASK_K;
    const int64_t recordStride = AlignUp(recordsPerMt * INSTALL_RECORD_BYTES, SLAB_ALIGN_BYTES);
    const int64_t totalMt = batch * rawSeq * M1B_ORD_MICROTASK_K;
    const int64_t recordsBytes = AlignUp(totalMt * recordStride, SLAB_ALIGN_BYTES);
    const int64_t countsBytes = AlignUp(totalMt * INSTALL_COUNT_BYTES, SLAB_ALIGN_BYTES);
    return recordsBytes + countsBytes + SLAB_ALIGN_BYTES - 1;
}

void CheckDsaPlanInputs(
    const at::Tensor& selectionTopkIndices,
    const at::Tensor& fullKvActualSeq,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq)
{
    TORCH_CHECK(selectionTopkIndices.dtype() == at::kInt,
        "selection_topk_indices must be int32.");
    TORCH_CHECK(fullKvActualSeq.dtype() == at::kInt,
        "full_kv_actual_seq must be int32.");
    TORCH_CHECK(poolIds.dtype() == at::kInt, "pool_ids must be int32.");
    TORCH_CHECK(idToSlot.dtype() == at::kInt, "id_to_slot must be int32.");
    TORCH_CHECK(lruCounter.dtype() == at::kInt, "lru_counter must be int32.");
    TORCH_CHECK(selectionTopkIndices.dim() == 3 || selectionTopkIndices.dim() == 4,
        "selection_topk_indices dim must be 3 or 4, actual ", selectionTopkIndices.dim(), ".");
    const int64_t batch = selectionTopkIndices.size(0);
    TORCH_CHECK(batch > 0, "selection_topk_indices batch must be positive.");
    TORCH_CHECK(rawSeq > 0, "raw_seq must be positive.");
    TORCH_CHECK(rawSeq <= MAX_PROD_RAW_SEQ,
        "DsaPlan raw_seq ", rawSeq, " exceeds production cap ", MAX_PROD_RAW_SEQ, ".");
    if (selectionTopkIndices.dim() == 4) {
        TORCH_CHECK(selectionTopkIndices.size(1) == rawSeq,
            "raw_seq must match selection_topk_indices.size(1) for 4D topk.");
        TORCH_CHECK(selectionTopkIndices.size(2) == 1,
            "DsaPlan supports only HEADS=1 for 4D selection_topk_indices.");
    } else {
        TORCH_CHECK(rawSeq == 1, "3D selection_topk_indices layout requires raw_seq=1.");
    }
    const int64_t topK = selectionTopkIndices.size(selectionTopkIndices.dim() - 1);
    TORCH_CHECK(topK > 0,
        "selection_topk_indices topK must be positive.");
    TORCH_CHECK(topK % TOPK_GRANULARITY == 0,
        "DsaPlan topK must be divisible by ", TOPK_GRANULARITY, ".");
    TORCH_CHECK(topK <= MAX_PROD_TOPK,
        "DsaPlan topK ", topK,
        " exceeds production cap ", MAX_PROD_TOPK, ".");
    TORCH_CHECK(fullKvActualSeq.dim() == 1 && fullKvActualSeq.size(0) == batch,
        "full_kv_actual_seq must be [batch] and match selection_topk_indices.size(0).");
    TORCH_CHECK(poolIds.dim() == 2, "pool_ids must be [batch, pool_size].");
    TORCH_CHECK(poolIds.size(0) == batch,
        "pool_ids batch must match selection_topk_indices.size(0).");
    const int64_t poolSize = poolIds.size(1);
    TORCH_CHECK(poolSize > 0 && poolSize <= MAX_PROD_POOL_SIZE && (poolSize % WAYS_PER_SET) == 0,
        "DsaPlan pool_size must be positive, <= ", MAX_PROD_POOL_SIZE,
        ", and divisible by ", WAYS_PER_SET, ".");
    const int64_t numSets = poolSize / WAYS_PER_SET;
    TORCH_CHECK((numSets & (numSets - 1)) == 0,
        "DsaPlan pool_size must produce a power-of-two set count with ", WAYS_PER_SET, " ways.");
    TORCH_CHECK(idToSlot.dim() == 2 && idToSlot.size(0) == batch,
        "id_to_slot must be [batch, id_range].");
    TORCH_CHECK(idToSlot.size(1) > 0,
        "DsaPlan id_range ", idToSlot.size(1), " must be positive.");
    TORCH_CHECK(lruCounter.dim() == 2, "lru_counter must be [batch, num_sets].");
    TORCH_CHECK(lruCounter.size(0) == poolIds.size(0) && lruCounter.size(1) == numSets,
        "lru_counter must be [batch, num_sets].");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
ConstructDsaPlanOutputs(
    const at::Tensor& selectionTopkIndices,
    int64_t rawSeq)
{
    int64_t topkElems = selectionTopkIndices.numel();
    (void)SRC_INVALID;
    int64_t topk = selectionTopkIndices.size(selectionTopkIndices.dim() - 1);
    int64_t queryRows = topkElems / topk;
    at::Tensor plan = at::zeros({topkElems, PLAN_WIDTH}, selectionTopkIndices.options());
    at::Tensor installRecords = at::zeros(
        {ComputeInstallRecordsAllocatedBytes(selectionTopkIndices.size(0), rawSeq, topk)},
        selectionTopkIndices.options().dtype(at::kChar));
    at::Tensor selectionKvActualSeq = at::empty({queryRows}, selectionTopkIndices.options());
    return std::make_tuple(plan, installRecords, selectionKvActualSeq);
}
}  // namespace

#ifndef DSA_PLAN_META_ONLY
std::tuple<at::Tensor, at::Tensor, at::Tensor> dsa_plan_npu(
    const at::Tensor& selectionTopkIndices,
    const at::Tensor& fullKvActualSeq,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t groupId,
    int64_t ownerLayer,
    int64_t groupKind)
{
    (void)groupId;
    (void)ownerLayer;
    (void)groupKind;
    CheckDsaPlanInputs(selectionTopkIndices, fullKvActualSeq, poolIds, idToSlot, lruCounter, rawSeq);
    auto outputs = ConstructDsaPlanOutputs(selectionTopkIndices, rawSeq);
    EXEC_NPU_CMD_V1(aclnnDsaPlan,
        selectionTopkIndices,
        fullKvActualSeq,
        poolIds,
        idToSlot,
        lruCounter,
        rawSeq,
        groupId,
        ownerLayer,
        groupKind,
        std::get<0>(outputs),
        std::get<1>(outputs),
        std::get<2>(outputs));
    return outputs;
}
#endif

std::tuple<at::Tensor, at::Tensor, at::Tensor> dsa_plan_meta(
    const at::Tensor& selectionTopkIndices,
    const at::Tensor& fullKvActualSeq,
    const at::Tensor& poolIds,
    const at::Tensor& idToSlot,
    const at::Tensor& lruCounter,
    int64_t rawSeq,
    int64_t groupId,
    int64_t ownerLayer,
    int64_t groupKind)
{
    (void)groupId;
    (void)ownerLayer;
    (void)groupKind;
    CheckDsaPlanInputs(selectionTopkIndices, fullKvActualSeq, poolIds, idToSlot, lruCounter, rawSeq);
    return ConstructDsaPlanOutputs(selectionTopkIndices, rawSeq);
}
}  // namespace custom

#ifndef DSA_PLAN_META_ONLY
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("dsa_plan", &custom::dsa_plan_npu);
}
#endif

TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("dsa_plan", &custom::dsa_plan_meta);
}
