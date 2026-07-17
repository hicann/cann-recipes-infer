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
constexpr int32_t SRC_INVALID = 0;
constexpr int32_t SRC_POOL_HIT = 1;
constexpr int32_t SRC_FULL_OLD_MISS = 2;
constexpr int32_t SRC_DECODE_WINDOW = 3;
constexpr int32_t SRC_FUTURE_SKIP = 4;
constexpr int32_t SRC_NEGATIVE_STOP = 5;
constexpr int32_t PLAN_WIDTH = 2;
constexpr int32_t RECORD_INT32S = 4;
constexpr int64_t INT32_ALIGN_ELEMS = 8;
constexpr int64_t COUNT_STRIDE_INT32S = INT32_ALIGN_ELEMS;

__aicore__ inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline int32_t MaxInt32(int32_t lhs, int32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

__aicore__ inline int64_t MaxInt64(int64_t lhs, int64_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <HardEvent event>
__aicore__ inline void SetWaitEvent(HardEvent evt)
{
    const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

class DsaPlanServeMetadata {
public:
    __aicore__ inline DsaPlanServeMetadata() {}

    __aicore__ inline void Init(
        GM_ADDR selectionTopkIndices,
        GM_ADDR fullKvActualSeq,
        GM_ADDR poolIds,
        GM_ADDR idToSlot,
        GM_ADDR lruCounter,
        GM_ADDR plan,
        GM_ADDR installRecords,
        GM_ADDR selectionKvActualSeq,
        TPipe* pipe,
        const DsaPlanTilingData* tilingData)
    {
        (void)lruCounter;
        selectionTopkIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)selectionTopkIndices);
        fullKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)fullKvActualSeq);
        poolIdsGm_.SetGlobalBuffer((__gm__ int32_t*)poolIds);
        idToSlotGm_.SetGlobalBuffer((__gm__ int32_t*)idToSlot);
        planGm_.SetGlobalBuffer((__gm__ int32_t*)plan);
        selectionKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selectionKvActualSeq);

        const uint64_t rawAddr = reinterpret_cast<uint64_t>(installRecords);
        const uint64_t alignedAddr = (rawAddr + 4095u) & ~static_cast<uint64_t>(4095u);
        const int64_t alignedOff = static_cast<int64_t>(alignedAddr - rawAddr);
        ASSERT(alignedOff >= 0 && alignedOff < 4096);
        ASSERT(alignedOff + tilingData->installRecordsRequiredBytes <= tilingData->installRecordsAllocatedBytes);
        GM_ADDR alignedBase = reinterpret_cast<GM_ADDR>(alignedAddr);
        installRecordsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(alignedBase + tilingData->compactAivRecordsOff));
        installCountsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(alignedBase + tilingData->compactAivCountsOff));

        pipe_ = pipe;
        tilingData_ = tilingData;
        batch_ = tilingData_->batch;
        rawSeq_ = tilingData_->rawSeq;
        topK_ = tilingData_->topK;
        numSets_ = tilingData_->numSets;
        waysPerSet_ = tilingData_->waysPerSet;
        poolSize_ = tilingData_->poolSize;
        idRange_ = tilingData_->idRange;
        topkPerBatch_ = tilingData_->topkPerBatch;
        usedCoreNum_ = tilingData_->usedCoreNum > 0 ? tilingData_->usedCoreNum : 1;
        kMicro_ = tilingData_->m1bOrdMicrotaskK > 0 ? tilingData_->m1bOrdMicrotaskK : 1;
        ordsPerMt_ = topK_ / kMicro_;
        recordStrideInt32_ = tilingData_->installRecordsStride / static_cast<int64_t>(sizeof(int32_t));
        maxMtPerCore_ = MaxInt64(tilingData_->mainCoreBsLoopNum, tilingData_->tailCoreBsLoopNum);

        ASSERT(topK_ > 0);
        ASSERT(kMicro_ > 0);
        ASSERT((topK_ % kMicro_) == 0);
        ASSERT((ordsPerMt_ % INT32_ALIGN_ELEMS) == 0);
        ASSERT((numSets_ & (numSets_ - 1)) == 0);

        const int64_t topkElems = AlignUpInt64(topK_, INT32_ALIGN_ELEMS);
        const int64_t planElems = AlignUpInt64(kMicro_ * ordsPerMt_ * PLAN_WIDTH, INT32_ALIGN_ELEMS);
        const int64_t recordElems = AlignUpInt64(kMicro_ * recordStrideInt32_, INT32_ALIGN_ELEMS);
        const int64_t countElems = AlignUpInt64(maxMtPerCore_ * COUNT_STRIDE_INT32S, INT32_ALIGN_ELEMS);
        const int64_t actualElems = AlignUpInt64((maxMtPerCore_ + kMicro_ - 1) / kMicro_ + 1, INT32_ALIGN_ELEMS);
        pipe_->InitBuffer(topkBuf_, static_cast<uint32_t>(topkElems * sizeof(int32_t)));
        pipe_->InitBuffer(planBuf_, static_cast<uint32_t>(planElems * sizeof(int32_t)));
        pipe_->InitBuffer(recordsBuf_, static_cast<uint32_t>(recordElems * sizeof(int32_t)));
        pipe_->InitBuffer(countsBuf_, static_cast<uint32_t>(countElems * sizeof(int32_t)));
        pipe_->InitBuffer(actualSeqBuf_, static_cast<uint32_t>(actualElems * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx() / GetTaskRation());
        if (coreIdx >= usedCoreNum_) {
            return;
        }

        const int64_t totalMt = batch_ * rawSeq_ * kMicro_;
        const int64_t main = tilingData_->mainCoreBsLoopNum;
        const int64_t tail = tilingData_->tailCoreBsLoopNum;
        const int64_t r = (main == tail) ? usedCoreNum_ : (totalMt - usedCoreNum_ * tail) / (main - tail);
        const int64_t mtCount = (coreIdx < r) ? main : tail;
        const int64_t mtStart = (coreIdx < r) ? coreIdx * main : r * main + (coreIdx - r) * tail;
        if (mtCount <= 0) {
            return;
        }
        const int64_t mtEnd = mtStart + mtCount;

        LocalTensor<int32_t> countsLocal = countsBuf_.Get<int32_t>();
        const int64_t countElems = AlignUpInt64(mtCount * COUNT_STRIDE_INT32S, INT32_ALIGN_ELEMS);
        Duplicate<int32_t>(countsLocal, 0, static_cast<uint32_t>(countElems));
        PipeBarrier<PIPE_V>();

        int64_t prevWorkItem = -1;
        int64_t currentWorkMtStart = mtStart;
        int64_t actualSeqStartWorkItem = -1;
        int32_t actualSeqCount = 0;
        for (int64_t mt = mtStart; mt < mtEnd; ++mt) {
            const int64_t wi = mt / kMicro_;
            const int64_t ord = mt - wi * kMicro_;
            const int64_t batchIdx = wi / rawSeq_;
            const int64_t seqIdx = wi - batchIdx * rawSeq_;
            if (wi != prevWorkItem) {
                if (prevWorkItem >= 0) {
                    FlushWorkItem(prevWorkItem, currentWorkMtStart, mt, mtStart, countsLocal);
                }
                currentWorkMtStart = mt;
                PrepareWorkItem(wi, batchIdx, seqIdx, mtStart, mtEnd,
                    actualSeqStartWorkItem, actualSeqCount);
                prevWorkItem = wi;
            }
            ProcessMicrotask(wi, batchIdx, seqIdx, ord, mt - mtStart, countsLocal);
        }
        if (prevWorkItem >= 0) {
            FlushWorkItem(prevWorkItem, currentWorkMtStart, mtEnd, mtStart, countsLocal);
        }
        FlushCounts(mtStart, mtCount, countsLocal);
        FlushActualSeq(actualSeqStartWorkItem, actualSeqCount);
    }

private:
    __aicore__ inline void LoadTopkRange(int64_t wi, int64_t ordFirst, int64_t ordLastExcl)
    {
        const int64_t elemFirst = ordFirst * ordsPerMt_;
        const int64_t elemLast = ordLastExcl * ordsPerMt_;
        const int64_t elemCount = elemLast - elemFirst;
        if (elemCount <= 0) {
            return;
        }
        LocalTensor<int32_t> topkLocal = topkBuf_.Get<int32_t>();
        DataCopyExtParams topkCopyParams{
            1,
            static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(topkLocal[elemFirst], selectionTopkIndicesGm_[wi * topK_ + elemFirst],
            topkCopyParams, padParams);
    }

    __aicore__ inline int32_t LookupPoolSlot(int64_t batchIdx, int32_t topkId)
    {
        if (topkId < 0 || topkId >= idRange_) {
            return -1;
        }
        const int32_t poolSlot = idToSlotGm_.GetValue(batchIdx * idRange_ + topkId);
        if (poolSlot < 0 || poolSlot >= poolSize_) {
            return -1;
        }
        return poolSlot;
    }

    __aicore__ inline bool HasNegativeBefore(int64_t wi, int64_t ordStart)
    {
        if (ordStart <= 0) {
            return false;
        }
        return selectionTopkIndicesGm_.GetValue(wi * topK_ + ordStart - 1) < 0;
    }

    __aicore__ inline int32_t ComputeActualSeq(int64_t batchIdx, int64_t seqIdx,
        LocalTensor<int32_t>& topkLocal)
    {
        const int32_t curLen = fullKvActualSeqGm_.GetValue(batchIdx);
        if (curLen <= 0) {
            return 0;
        }
        const int32_t seqOffset = static_cast<int32_t>((rawSeq_ - 1) - seqIdx);
        const int32_t maxSelection = curLen - seqOffset - 1;
        if (maxSelection < 0) {
            return 0;
        }
        int32_t validRows = 0;
        for (int64_t dstPos = 0; dstPos < topK_; ++dstPos) {
            const int32_t topkId = topkLocal.GetValue(dstPos);
            if (topkId < 0) {
                break;
            }
            if (topkId <= maxSelection) {
                ++validRows;
            }
        }
        return validRows;
    }

    __aicore__ inline void AppendActualSeq(int64_t wi, int32_t value, int64_t& startWorkItem, int32_t& count)
    {
        if (startWorkItem < 0) {
            startWorkItem = wi;
        } else if (wi != startWorkItem + count) {
            FlushActualSeq(startWorkItem, count);
            startWorkItem = wi;
            count = 0;
        }
        LocalTensor<int32_t> actualLocal = actualSeqBuf_.Get<int32_t>();
        actualLocal.SetValue(count, value);
        ++count;
    }

    __aicore__ inline void FlushActualSeq(int64_t& startWorkItem, int32_t& count)
    {
        if (startWorkItem < 0 || count <= 0) {
            return;
        }
        LocalTensor<int32_t> actualLocal = actualSeqBuf_.Get<int32_t>();
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(static_cast<int64_t>(count) * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        SetWaitEvent<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(selectionKvActualSeqGm_[startWorkItem], actualLocal, copyParams);
        SetWaitEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        startWorkItem = -1;
        count = 0;
    }

    __aicore__ inline void WritePlanLocal(LocalTensor<int32_t>& planLocal, int64_t localRow,
        int32_t srcKind, int32_t topkId, int32_t dstPos, int32_t setIdx,
        int32_t hitWay, int32_t poolSlot, int32_t install)
    {
        (void)dstPos;
        (void)setIdx;
        (void)hitWay;
        (void)install;
        const int64_t base = localRow * PLAN_WIDTH;
        planLocal.SetValue(base + 0, srcKind);
        planLocal.SetValue(base + 1, (srcKind == SRC_POOL_HIT) ? poolSlot : topkId);
    }

    __aicore__ inline void PrepareWorkItem(int64_t wi, int64_t batchIdx, int64_t seqIdx,
        int64_t mtStart, int64_t mtEnd, int64_t& actualSeqStartWorkItem, int32_t& actualSeqCount)
    {
        const int64_t workMtStart = wi * kMicro_;
        const int64_t workMtEnd = workMtStart + kMicro_;
        const int64_t firstMt = MaxInt64(mtStart, workMtStart);
        const int64_t lastMt = MinInt64(mtEnd, workMtEnd);
        const int64_t ordFirst = firstMt - workMtStart;
        const int64_t ordLast = lastMt - workMtStart;
        if (ordFirst == 0) {
            LoadTopkRange(wi, 0, kMicro_);
        } else {
            LoadTopkRange(wi, ordFirst, ordLast);
        }
        SetWaitEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        if (ordFirst == 0) {
            LocalTensor<int32_t> topkLocal = topkBuf_.Get<int32_t>();
            AppendActualSeq(wi, ComputeActualSeq(batchIdx, seqIdx, topkLocal),
                actualSeqStartWorkItem, actualSeqCount);
        }
    }

    __aicore__ inline void FlushWorkItem(int64_t wi, int64_t firstMt, int64_t lastMt,
        int64_t coreMtStart, LocalTensor<int32_t>& countsLocal)
    {
        if (lastMt <= firstMt) {
            return;
        }
        const int64_t workMtStart = wi * kMicro_;
        const int64_t ordFirst = firstMt - workMtStart;
        const int64_t ordLast = lastMt - workMtStart;
        const int64_t mtCount = ordLast - ordFirst;
        const int64_t rowCount = mtCount * ordsPerMt_;
        LocalTensor<int32_t> planLocal = planBuf_.Get<int32_t>();
        DataCopyExtParams planCopyParams{
            1,
            static_cast<uint32_t>(rowCount * PLAN_WIDTH * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        const int64_t localPlanOff = ordFirst * ordsPerMt_ * PLAN_WIDTH;
        const int64_t gmRow = wi * topK_ + ordFirst * ordsPerMt_;
        SetWaitEvent<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(planGm_[gmRow * PLAN_WIDTH], planLocal[localPlanOff], planCopyParams);
        SetWaitEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        LocalTensor<int32_t> recordsLocal = recordsBuf_.Get<int32_t>();
        bool fenced = false;
        for (int64_t mt = firstMt; mt < lastMt; ++mt) {
            const int32_t installCount =
                countsLocal.GetValue((mt - coreMtStart) * COUNT_STRIDE_INT32S);
            if (installCount <= 0) {
                continue;
            }
            if (!fenced) {
                SetWaitEvent<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                fenced = true;
            }
            const int64_t ord = mt - workMtStart;
            DataCopyExtParams recordCopyParams{
                1,
                static_cast<uint32_t>(
                    installCount * RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t))),
                0, 0, 0
            };
            DataCopyPad(installRecordsGm_[mt * recordStrideInt32_],
                recordsLocal[ord * recordStrideInt32_], recordCopyParams);
        }
        if (fenced) {
            SetWaitEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
    }

    __aicore__ inline void FlushCounts(int64_t mtStart, int64_t mtCount, LocalTensor<int32_t>& countsLocal)
    {
        if (mtCount <= 0) {
            return;
        }
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(mtCount * COUNT_STRIDE_INT32S * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        SetWaitEvent<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(installCountsGm_[mtStart * COUNT_STRIDE_INT32S], countsLocal, copyParams);
        SetWaitEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessMicrotask(int64_t wi, int64_t batchIdx, int64_t seqIdx, int64_t ord,
        int64_t localMtIdx, LocalTensor<int32_t>& countsLocal)
    {
        const int64_t ordStart = ord * ordsPerMt_;
        const int64_t rowBase = wi * topK_ + ordStart;
        LocalTensor<int32_t> topkLocal = topkBuf_.Get<int32_t>();
        LocalTensor<int32_t> planLocal = planBuf_.Get<int32_t>();
        LocalTensor<int32_t> recordsLocal = recordsBuf_.Get<int32_t>();

        const int32_t curLen = fullKvActualSeqGm_.GetValue(batchIdx);
        const int32_t maxValid = MaxInt32(curLen - static_cast<int32_t>(rawSeq_) - 1, 0);
        const int32_t seqOffset = static_cast<int32_t>((rawSeq_ - 1) - seqIdx);
        const int32_t maxSelection = curLen - seqOffset - 1;
        const bool seqEmpty = (curLen <= 0) || (maxSelection < 0);
        bool stopped = HasNegativeBefore(wi, ordStart);
        int32_t installCount = 0;

        for (int64_t i = 0; i < ordsPerMt_; ++i) {
            const int32_t dstPos = static_cast<int32_t>(ordStart + i);
            const int64_t localRow = ord * ordsPerMt_ + i;
            if (stopped) {
                WritePlanLocal(planLocal, localRow, SRC_INVALID, -1, dstPos, -1, -1, -1, 0);
                continue;
            }

            const int32_t topkId = topkLocal.GetValue(ordStart + i);
            if (topkId < 0) {
                WritePlanLocal(planLocal, localRow, SRC_NEGATIVE_STOP, topkId, dstPos, -1, -1, -1, 0);
                stopped = true;
                continue;
            }
            if (seqEmpty || topkId > maxSelection) {
                WritePlanLocal(planLocal, localRow, SRC_FUTURE_SKIP, topkId, dstPos, -1, -1, -1, 0);
                continue;
            }

            const int32_t setIdx = topkId & static_cast<int32_t>(numSets_ - 1);
            const int32_t poolSlot = LookupPoolSlot(batchIdx, topkId);
            if (poolSlot >= 0 && topkId <= maxValid) {
                const int32_t hitWay = poolSlot & static_cast<int32_t>(waysPerSet_ - 1);
                WritePlanLocal(planLocal, localRow, SRC_POOL_HIT, topkId, dstPos, setIdx, hitWay, poolSlot, 0);
                continue;
            }
            if (topkId > maxValid) {
                WritePlanLocal(planLocal, localRow, SRC_DECODE_WINDOW, topkId, dstPos, setIdx, -1, -1, 0);
                continue;
            }

            WritePlanLocal(planLocal, localRow, SRC_FULL_OLD_MISS, topkId, dstPos, setIdx, -1, -1, 1);
            const int64_t recBase = ord * recordStrideInt32_ + static_cast<int64_t>(installCount) * RECORD_INT32S;
            recordsLocal.SetValue(recBase + 0, setIdx);
            recordsLocal.SetValue(recBase + 1, topkId);
            recordsLocal.SetValue(recBase + 2, static_cast<int32_t>(seqIdx));
            recordsLocal.SetValue(recBase + 3, dstPos);
            ++installCount;
        }

        (void)rowBase;
        countsLocal.SetValue(localMtIdx * COUNT_STRIDE_INT32S, installCount);
    }

    TPipe* pipe_ = nullptr;
    const DsaPlanTilingData* tilingData_ = nullptr;
    TBuf<TPosition::VECCALC> topkBuf_;
    TBuf<TPosition::VECCALC> planBuf_;
    TBuf<TPosition::VECCALC> recordsBuf_;
    TBuf<TPosition::VECCALC> countsBuf_;
    TBuf<TPosition::VECCALC> actualSeqBuf_;

    GlobalTensor<int32_t> selectionTopkIndicesGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> poolIdsGm_;
    GlobalTensor<int32_t> idToSlotGm_;
    GlobalTensor<int32_t> planGm_;
    GlobalTensor<int32_t> installRecordsGm_;
    GlobalTensor<int32_t> installCountsGm_;
    GlobalTensor<int32_t> selectionKvActualSeqGm_;

    int64_t batch_ = 0;
    int64_t rawSeq_ = 1;
    int64_t topK_ = 0;
    int64_t numSets_ = 0;
    int64_t waysPerSet_ = 0;
    int64_t poolSize_ = 0;
    int64_t idRange_ = 0;
    int64_t topkPerBatch_ = 0;
    int64_t usedCoreNum_ = 1;
    int64_t kMicro_ = 1;
    int64_t ordsPerMt_ = 1;
    int64_t recordStrideInt32_ = 0;
    int64_t maxMtPerCore_ = 1;
};
}  // namespace

extern "C" __global__ __aicore__ void dsa_plan(
    GM_ADDR selection_topk_indices,
    GM_ADDR full_kv_actual_seq,
    GM_ADDR pool_ids,
    GM_ADDR id_to_slot,
    GM_ADDR lru_counter,
    GM_ADDR plan,
    GM_ADDR install_records,
    GM_ADDR selection_kv_actual_seq,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    DsaPlanServeMetadata op;
    op.Init(selection_topk_indices, full_kv_actual_seq, pool_ids, id_to_slot, lru_counter, plan, install_records,
        selection_kv_actual_seq, &pipe, &tilingData);
    op.Process();
}
