/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

namespace {
constexpr int32_t SRC_POOL_HIT = 1;
constexpr int32_t SRC_FULL_OLD_MISS = 2;
constexpr int32_t SRC_DECODE_WINDOW = 3;
constexpr int32_t PLAN_WIDTH = 2;
constexpr int32_t PLAN_COL_SRC_KIND = 0;
constexpr int32_t PLAN_COL_SRC_INDEX = 1;
constexpr int64_t BLOCK_BYTES = 32;
#ifndef DSA_SERVE_COMPACT_BATCH_ROWS_TUNING
#define DSA_SERVE_COMPACT_BATCH_ROWS_TUNING 64
#endif
static_assert(DSA_SERVE_COMPACT_BATCH_ROWS_TUNING > 0, "DsaServe row tile must be positive.");
constexpr int32_t DSA_SERVE_COMPACT_BATCH_ROWS = DSA_SERVE_COMPACT_BATCH_ROWS_TUNING;

template <HardEvent event>
__aicore__ inline void SetWaitEvent(HardEvent evt)
{
    const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline int64_t MaxInt64(int64_t lhs, int64_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

template <typename T>
class DsaServeNonInt8PayloadConsumer {
public:
    __aicore__ inline DsaServeNonInt8PayloadConsumer() {}

    __aicore__ inline void Init(
        GM_ADDR plan,
        GM_ADDR fullKvCache,
        GM_ADDR fullKRope,
        GM_ADDR poolKvCache,
        GM_ADDR poolKRope,
        GM_ADDR selectionKvCache,
        GM_ADDR selectionKRope,
        TPipe* pipe,
        const DsaServeTilingData* tilingData)
    {
        planGm_.SetGlobalBuffer((__gm__ int32_t*)plan);
        fullKvGm_.SetGlobalBuffer((__gm__ T*)fullKvCache);
        fullRopeGm_.SetGlobalBuffer((__gm__ T*)fullKRope);
        poolKvGm_.SetGlobalBuffer((__gm__ T*)poolKvCache);
        poolRopeGm_.SetGlobalBuffer((__gm__ T*)poolKRope);
        selectionKvGm_.SetGlobalBuffer((__gm__ T*)selectionKvCache);
        selectionRopeGm_.SetGlobalBuffer((__gm__ T*)selectionKRope);
        pipe_ = pipe;
        tilingData_ = tilingData;

        planRows_ = tilingData_->planRows;
        rawSeq_ = tilingData_->rawSeq;
        topK_ = tilingData_->topK;
        fullSeq_ = tilingData_->fullSeq;
        poolSize_ = tilingData_->poolSize;
        selectionBlockSize_ = tilingData_->selectionBlockSize;
        kvDim_ = tilingData_->kvDim;
        ropeDim_ = tilingData_->ropeDim;
        compactLayout_ = tilingData_->compactLayout != 0;
        topkPerBatch_ = rawSeq_ * topK_;

        usedCoreNum_ = tilingData_->usedCoreNum > 0 ? tilingData_->usedCoreNum : 1;

        kvEntryBytes_ = AlignUp(kvDim_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES);
        ropeEntryBytes_ = AlignUp(ropeDim_ * static_cast<int64_t>(sizeof(T)), BLOCK_BYTES);
        kvEntryAligned_ = kvEntryBytes_ / static_cast<int64_t>(sizeof(T));
        ropeEntryAligned_ = ropeEntryBytes_ / static_cast<int64_t>(sizeof(T));
        const int64_t planBytes =
            AlignUp(DSA_SERVE_COMPACT_BATCH_ROWS * PLAN_WIDTH * static_cast<int64_t>(sizeof(int32_t)), BLOCK_BYTES);
        pipe_->InitBuffer(planBuf_, static_cast<uint32_t>(planBytes));
        pipe_->InitBuffer(kvBatchBuf_, static_cast<uint32_t>(DSA_SERVE_COMPACT_BATCH_ROWS * kvEntryBytes_));
        pipe_->InitBuffer(ropeBatchBuf_, static_cast<uint32_t>(DSA_SERVE_COMPACT_BATCH_ROWS * ropeEntryBytes_));
    }

    __aicore__ inline void Process()
    {
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx() / GetTaskRation());
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        const int64_t rowStart = (planRows_ * coreIdx) / usedCoreNum_;
        const int64_t rowEnd = (planRows_ * (coreIdx + 1)) / usedCoreNum_;
        for (int64_t row = rowStart; row < rowEnd; row += DSA_SERVE_COMPACT_BATCH_ROWS) {
            const int64_t count = MinInt64(static_cast<int64_t>(DSA_SERVE_COMPACT_BATCH_ROWS), rowEnd - row);
            ProcessRowBatch(row, count);
        }
    }

private:
    __aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
    {
        return (lhs < rhs) ? lhs : rhs;
    }

    __aicore__ inline void ProcessRowBatch(int64_t rowStart, int64_t count)
    {
        LocalTensor<int32_t> planLocal = planBuf_.Get<int32_t>();
        LocalTensor<T> kvBatchLocal = kvBatchBuf_.Get<T>();
        LocalTensor<T> ropeBatchLocal = ropeBatchBuf_.Get<T>();
        DataCopyExtParams planCopyParams{
            1,
            static_cast<uint32_t>(count * PLAN_WIDTH * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        DataCopyPadExtParams<int32_t> planPadParams{false, 0, 0, 0};
        DataCopyPad(planLocal, planGm_[rowStart * PLAN_WIDTH], planCopyParams, planPadParams);
        SetWaitEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        int64_t dstKvAddrs[DSA_SERVE_COMPACT_BATCH_ROWS];
        int64_t dstRopeAddrs[DSA_SERVE_COMPACT_BATCH_ROWS];
        DataCopyExtParams kvCopyParams{1, static_cast<uint32_t>(kvDim_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ropeCopyParams{1, static_cast<uint32_t>(ropeDim_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        bool hasZeroFill = false;
        for (int64_t i = 0; i < count; ++i) {
            const int64_t row = rowStart + i;
            const int64_t base = i * PLAN_WIDTH;
            const int32_t srcKind = planLocal.GetValue(base + PLAN_COL_SRC_KIND);
            const int32_t srcIndex = planLocal.GetValue(base + PLAN_COL_SRC_INDEX);
            const int64_t batch = row / topkPerBatch_;
            LocalTensor<T> kvDst = kvBatchLocal[i * kvEntryAligned_];
            LocalTensor<T> ropeDst = ropeBatchLocal[i * ropeEntryAligned_];
            if (srcKind == SRC_POOL_HIT && srcIndex >= 0 && srcIndex < poolSize_) {
                const int64_t srcRow = batch * poolSize_ + srcIndex;
                DataCopyPad(kvDst, poolKvGm_[srcRow * kvDim_], kvCopyParams, padParams);
                DataCopyPad(ropeDst, poolRopeGm_[srcRow * ropeDim_], ropeCopyParams, padParams);
            } else if ((srcKind == SRC_FULL_OLD_MISS || srcKind == SRC_DECODE_WINDOW) &&
                       srcIndex >= 0 && srcIndex < fullSeq_) {
                const int64_t srcRow = batch * fullSeq_ + srcIndex;
                DataCopyPad(kvDst, fullKvGm_[srcRow * kvDim_], kvCopyParams, padParams);
                DataCopyPad(ropeDst, fullRopeGm_[srcRow * ropeDim_], ropeCopyParams, padParams);
            } else {
                Duplicate(kvDst, static_cast<T>(0), static_cast<uint32_t>(kvEntryAligned_));
                Duplicate(ropeDst, static_cast<T>(0), static_cast<uint32_t>(ropeEntryAligned_));
                hasZeroFill = true;
            }
            dstKvAddrs[i] = SelectionOffset(row, kvDim_);
            dstRopeAddrs[i] = SelectionOffset(row, ropeDim_);
        }

        FlushPayloadBatch(kvBatchLocal, ropeBatchLocal, dstKvAddrs, dstRopeAddrs,
            static_cast<int32_t>(count), hasZeroFill, kvCopyParams, ropeCopyParams);
    }

    __aicore__ inline void FlushPayloadBatch(LocalTensor<T>& kvBatchLocal, LocalTensor<T>& ropeBatchLocal,
        const int64_t dstKvAddrs[DSA_SERVE_COMPACT_BATCH_ROWS],
        const int64_t dstRopeAddrs[DSA_SERVE_COMPACT_BATCH_ROWS],
        int32_t count, bool hasZeroFill, DataCopyExtParams& kvCopyParams, DataCopyExtParams& ropeCopyParams)
    {
        if (count <= 0) {
            return;
        }
        bool kvContiguous = (kvEntryAligned_ == kvDim_);
        bool ropeContiguous = (ropeEntryAligned_ == ropeDim_);
        for (int32_t i = 1; i < count; ++i) {
            if (dstKvAddrs[i] != dstKvAddrs[0] + static_cast<int64_t>(i) * kvEntryAligned_) {
                kvContiguous = false;
            }
            if (dstRopeAddrs[i] != dstRopeAddrs[0] + static_cast<int64_t>(i) * ropeEntryAligned_) {
                ropeContiguous = false;
            }
        }
        SetWaitEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        if (hasZeroFill) {
            SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }
        PipeBarrier<PIPE_V>();
        if (kvContiguous) {
            DataCopyExtParams kvBatchCopyParams{
                1,
                static_cast<uint32_t>(static_cast<int64_t>(count) * kvEntryAligned_ * sizeof(T)),
                0, 0, 0
            };
            DataCopyPad(selectionKvGm_[dstKvAddrs[0]], kvBatchLocal, kvBatchCopyParams);
        } else {
            for (int32_t i = 0; i < count; ++i) {
                DataCopyPad(selectionKvGm_[dstKvAddrs[i]], kvBatchLocal[static_cast<int64_t>(i) * kvEntryAligned_],
                    kvCopyParams);
            }
        }
        if (ropeContiguous) {
            DataCopyExtParams ropeBatchCopyParams{
                1,
                static_cast<uint32_t>(static_cast<int64_t>(count) * ropeEntryAligned_ * sizeof(T)),
                0, 0, 0
            };
            DataCopyPad(selectionRopeGm_[dstRopeAddrs[0]], ropeBatchLocal, ropeBatchCopyParams);
        } else {
            for (int32_t i = 0; i < count; ++i) {
                DataCopyPad(selectionRopeGm_[dstRopeAddrs[i]], ropeBatchLocal[static_cast<int64_t>(i) * ropeEntryAligned_],
                    ropeCopyParams);
            }
        }
        SetWaitEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    __aicore__ inline int64_t SelectionOffset(int64_t dstRow, int64_t dim)
    {
        if (compactLayout_) {
            return dstRow * dim;
        }
        return dstRow * selectionBlockSize_ * dim;
    }

    TPipe* pipe_ = nullptr;
    const DsaServeTilingData* tilingData_ = nullptr;
    TBuf<TPosition::VECCALC> planBuf_;
    TBuf<TPosition::VECCALC> kvBatchBuf_;
    TBuf<TPosition::VECCALC> ropeBatchBuf_;
    GlobalTensor<int32_t> planGm_;
    GlobalTensor<T> fullKvGm_;
    GlobalTensor<T> fullRopeGm_;
    GlobalTensor<T> poolKvGm_;
    GlobalTensor<T> poolRopeGm_;
    GlobalTensor<T> selectionKvGm_;
    GlobalTensor<T> selectionRopeGm_;

    int64_t planRows_ = 0;
    int64_t rawSeq_ = 1;
    int64_t topK_ = 1;
    int64_t fullSeq_ = 0;
    int64_t poolSize_ = 0;
    int64_t selectionBlockSize_ = 16;
    int64_t kvDim_ = 0;
    int64_t ropeDim_ = 0;
    int64_t topkPerBatch_ = 1;
    int64_t usedCoreNum_ = 1;
    int64_t kvEntryBytes_ = 0;
    int64_t ropeEntryBytes_ = 0;
    int64_t kvEntryAligned_ = 0;
    int64_t ropeEntryAligned_ = 0;
    bool compactLayout_ = false;
};
}  // namespace

extern "C" __global__ __aicore__ void dsa_serve(
    GM_ADDR plan,
    GM_ADDR full_kv_cache,
    GM_ADDR full_k_rope,
    GM_ADDR pool_kv_cache,
    GM_ADDR pool_k_rope,
    GM_ADDR selection_kv_cache_in,
    GM_ADDR selection_k_rope_in,
    GM_ADDR selection_kv_cache_out,
    GM_ADDR selection_k_rope_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    (void)selection_kv_cache_in;
    (void)selection_k_rope_in;
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    DsaServeNonInt8PayloadConsumer<DTYPE_FULL_KV_CACHE> op;
    op.Init(plan, full_kv_cache, full_k_rope, pool_kv_cache, pool_k_rope, selection_kv_cache_out, selection_k_rope_out,
        &pipe, &tilingData);
    op.Process();
}
