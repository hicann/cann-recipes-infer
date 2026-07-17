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
constexpr int32_t INSTALL_RECORD_INT32S = 4;
constexpr int32_t STAGED_RECORD_INT32S = 5;
constexpr int32_t COUNT_STRIDE_INT32S = 8;
constexpr int64_t BLOCK_BYTES = 32;

__aicore__ inline int64_t Align32(int64_t bytes)
{
    return ((bytes + 31) / 32) * 32;
}

template <HardEvent event>
__aicore__ inline void SetWaitEvent(HardEvent evt)
{
    const event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
class DsaInstallNonInt8Consumer {
public:
    __aicore__ inline DsaInstallNonInt8Consumer(TPipe* pipe, const DsaInstallTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR selectionKvCache,
        GM_ADDR selectionKRope,
        GM_ADDR selectionKvBlockTable,
        GM_ADDR poolKvCache,
        GM_ADDR poolKRope,
        GM_ADDR poolIds,
        GM_ADDR idToSlot,
        GM_ADDR lruCounter,
        GM_ADDR installRecordsAlignedBase)
    {
        blkIdx_ = static_cast<int32_t>(GetBlockIdx() / GetTaskRation());
        batchsize_ = tiling_->batchsize;
        rawSeq_ = tiling_->rawSeq;
        origBatch_ = batchsize_ / rawSeq_;
        numSets_ = tiling_->numSets;
        waysPerSet_ = tiling_->waysPerSet;
        maxRecordsPerSeq_ = tiling_->maxRecordsPerSeq;
        installRecordsStride_ = tiling_->installRecordsStride;
        kvCacheDim_ = tiling_->kvCacheDim;
        kRopeDim_ = tiling_->kRopeDim;
        selKvBlockSize_ = tiling_->selKvBlockSize;
        poolSize_ = tiling_->poolSize;
        idRange_ = tiling_->idRange;
        bitsetInt32s_ = (idRange_ + 31) / 32;
        poolBlocksPerBatch_ = poolSize_ / selKvBlockSize_;
        selKvBlockStride_ = selKvBlockSize_ * kvCacheDim_;
        selRopeBlockStride_ = selKvBlockSize_ * kRopeDim_;
        headnum_ = tiling_->headnum;
        selMaxBlockNum_ = tiling_->selMaxBlockNum;
        metadataUpdate_ = tiling_->metadataUpdate;

        const int64_t bitsetBytes = Align32(bitsetInt32s_ * static_cast<int64_t>(sizeof(int32_t)));
        const int64_t lruBytes = Align32(numSets_ * static_cast<int64_t>(sizeof(int32_t)));
        const int64_t recordsUbBytes = installRecordsStride_;
        const int64_t stagedUbBytes =
            waysPerSet_ * STAGED_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t));
        const int64_t kvPayloadBytes = Align32(kvCacheDim_ * static_cast<int64_t>(sizeof(T)));
        const int64_t ropePayloadBytes = Align32(kRopeDim_ * static_cast<int64_t>(sizeof(T)));
        bitsetOff_ = 0;
        lruOff_ = bitsetOff_ + bitsetBytes;
        recordsOff_ = lruOff_ + lruBytes;
        stagedOff_ = recordsOff_ + recordsUbBytes;
        kvPayloadOff_ = Align32(stagedOff_ + stagedUbBytes);
        ropePayloadOff_ = kvPayloadOff_ + 2 * kvPayloadBytes;
        pipe_->InitBuffer(workBuf_, ropePayloadOff_ + 2 * ropePayloadBytes);

        selectionKvGm_.SetGlobalBuffer((__gm__ T*)selectionKvCache);
        selectionRopeGm_.SetGlobalBuffer((__gm__ T*)selectionKRope);
        sharedPoolKvGm_.SetGlobalBuffer((__gm__ T*)poolKvCache);
        sharedPoolRopeGm_.SetGlobalBuffer((__gm__ T*)poolKRope);
        selBlkTableGm_.SetGlobalBuffer((__gm__ int32_t*)selectionKvBlockTable);
        poolIdsGm_.SetGlobalBuffer((__gm__ int32_t*)poolIds);
        idToSlotGm_.SetGlobalBuffer((__gm__ int32_t*)idToSlot);
        lruCounterGm_.SetGlobalBuffer((__gm__ int32_t*)lruCounter);
        installRecordsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(installRecordsAlignedBase + tiling_->compactAivRecordsOff));
        installCountsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(installRecordsAlignedBase + tiling_->compactAivCountsOff));
    }

    __aicore__ inline void ProcessParallel()
    {
        LocalTensor<int32_t> workBase = workBuf_.Get<int32_t>();
        LocalTensor<int32_t> bitsetUb = workBase[bitsetOff_ / static_cast<int64_t>(sizeof(int32_t))];
        LocalTensor<int32_t> lruUb = workBase[lruOff_ / static_cast<int64_t>(sizeof(int32_t))];
        LocalTensor<int32_t> recsUb = workBase[recordsOff_ / static_cast<int64_t>(sizeof(int32_t))];
        LocalTensor<int32_t> stagedUb = workBase[stagedOff_ / static_cast<int64_t>(sizeof(int32_t))];
        const int32_t workerNum = static_cast<int32_t>(
            tiling_->installWorkerNum > 0 ? tiling_->installWorkerNum : GetBlockNum());
        for (int32_t b = blkIdx_; b < static_cast<int32_t>(origBatch_); b += workerNum) {
            ProcessOneBatch(b, bitsetUb, lruUb, recsUb, stagedUb);
        }
    }

private:
    __aicore__ inline int64_t SelKvSrcAddr(int32_t b, int32_t originSeq, int32_t destPos)
    {
        const int32_t logicalBlock = destPos / static_cast<int32_t>(selKvBlockSize_);
        const int32_t blockOffset = destPos % static_cast<int32_t>(selKvBlockSize_);
        const int64_t compactBlock = (static_cast<int64_t>(b) * rawSeq_ + originSeq) * selMaxBlockNum_
            + logicalBlock;
        return compactBlock * selKvBlockStride_
             + static_cast<int64_t>(blockOffset) * kvCacheDim_;
    }

    __aicore__ inline int64_t SelRopeSrcAddr(int32_t b, int32_t originSeq, int32_t destPos)
    {
        const int32_t logicalBlock = destPos / static_cast<int32_t>(selKvBlockSize_);
        const int32_t blockOffset = destPos % static_cast<int32_t>(selKvBlockSize_);
        const int64_t compactBlock = (static_cast<int64_t>(b) * rawSeq_ + originSeq) * selMaxBlockNum_
            + logicalBlock;
        return compactBlock * selRopeBlockStride_
             + static_cast<int64_t>(blockOffset) * kRopeDim_;
    }

    __aicore__ inline void LoadLruLocal(int32_t b, LocalTensor<int32_t>& lruUb)
    {
        DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(numSets_ * static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0
        };
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(lruUb, lruCounterGm_[static_cast<int64_t>(b) * numSets_], copyParams, padParams);
        SetWaitEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    }

    __aicore__ inline void ProcessOneBatch(int32_t b, LocalTensor<int32_t>& bitsetUb,
        LocalTensor<int32_t>& lruUb,
        LocalTensor<int32_t>& recsUb, LocalTensor<int32_t>& stagedUb)
    {
        Duplicate<int32_t>(bitsetUb, 0, static_cast<uint32_t>(bitsetInt32s_));
        PipeBarrier<PIPE_ALL>();
        if (metadataUpdate_ == 0) {
            LoadLruLocal(b, lruUb);
        }
        int32_t stagedCount = 0;
        const int64_t innerCount = tiling_->m1bOrdMicrotaskK > 0 ? tiling_->m1bOrdMicrotaskK : 1;
        for (int32_t seq = 0; seq < static_cast<int32_t>(rawSeq_); ++seq) {
        for (int32_t oc = 0; oc < static_cast<int32_t>(innerCount); ++oc) {
            const int64_t slabIdx = (static_cast<int64_t>(b) * rawSeq_ + seq) * innerCount + oc;
            int32_t nRecs = installCountsGm_.GetValue(slabIdx * COUNT_STRIDE_INT32S);
            if (nRecs <= 0) {
                continue;
            }
            const int32_t slabCap = static_cast<int32_t>(
                installRecordsStride_ / (INSTALL_RECORD_INT32S * static_cast<int64_t>(sizeof(int32_t))));
            const int32_t cap = (slabCap < static_cast<int32_t>(maxRecordsPerSeq_))
                ? slabCap : static_cast<int32_t>(maxRecordsPerSeq_);
            if (nRecs > cap) {
                nRecs = cap;
            }
            const int64_t recordsInt32Off =
                slabIdx * (installRecordsStride_ / static_cast<int64_t>(sizeof(int32_t)));
            const int64_t copyInt32s = static_cast<int64_t>(nRecs) * INSTALL_RECORD_INT32S;
            const int64_t copyInt32sAligned = ((copyInt32s + 7) / 8) * 8;
            DataCopy(recsUb, installRecordsGm_[recordsInt32Off], copyInt32sAligned);
            SetWaitEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            for (int32_t r = 0; r < nRecs; ++r) {
                const int32_t recBase = r * INSTALL_RECORD_INT32S;
                const int32_t setIdx = recsUb.GetValue(recBase + 0);
                const int32_t topKId = recsUb.GetValue(recBase + 1);
                const int32_t originSeq = recsUb.GetValue(recBase + 2);
                const int32_t destPos = recsUb.GetValue(recBase + 3);
                if (topKId < 0 || topKId >= idRange_ ||
                    setIdx < 0 || setIdx >= numSets_ ||
                    originSeq < 0 || originSeq >= rawSeq_ ||
                    destPos < 0 || destPos >= tiling_->topk) {
                    continue;
                }
                const int32_t word = topKId >> 5;
                const int32_t bit = topKId & 31;
                const int32_t mask = 1 << bit;
                const int32_t cur = bitsetUb.GetValue(word);
                if ((cur & mask) != 0) {
                    continue;
                }
                bitsetUb.SetValue(word, cur | mask);

                int32_t stageSet = setIdx;
                int32_t way = 0;
                const int64_t lruIdx = static_cast<int64_t>(b) * numSets_ + setIdx;
                if (metadataUpdate_ != 0) {
                    const int32_t head = lruCounterGm_.GetValue(lruIdx);
                    way = head & static_cast<int32_t>(waysPerSet_ - 1);
                    lruCounterGm_.SetValue(lruIdx, (head + 1) & static_cast<int32_t>(waysPerSet_ - 1));
                } else {
                    const int32_t head = lruUb.GetValue(setIdx);
                    way = head & static_cast<int32_t>(waysPerSet_ - 1);
                    lruUb.SetValue(setIdx, (head + 1) & static_cast<int32_t>(waysPerSet_ - 1));
                }

                const int32_t localPos = stagedCount & 15;
                const int32_t stagedBase = localPos * STAGED_RECORD_INT32S;
                stagedUb.SetValue(stagedBase + 0, stageSet);
                stagedUb.SetValue(stagedBase + 1, way);
                stagedUb.SetValue(stagedBase + 2, topKId);
                stagedUb.SetValue(stagedBase + 3, originSeq);
                stagedUb.SetValue(stagedBase + 4, destPos);
                ++stagedCount;
                if ((stagedCount & 15) == 0) {
                    MutatePoolChunkFromUb(b, 16, stagedUb);
                }
            }
        }
        }
        const int32_t tail = stagedCount & 15;
        if (tail != 0) {
            MutatePoolChunkFromUb(b, tail, stagedUb);
        }
    }

    __aicore__ inline void PrefetchStagedRecord(int32_t b, LocalTensor<int32_t>& stagedUb,
        int32_t localIdx, LocalTensor<T>& kvBuf, LocalTensor<T>& ropeBuf,
        int64_t& dstKvAddr, int64_t& dstRopeAddr)
    {
        const int32_t base = localIdx * STAGED_RECORD_INT32S;
        const int32_t setIdx = stagedUb.GetValue(base + 0);
        const int32_t way = stagedUb.GetValue(base + 1);
        const int32_t topKId = stagedUb.GetValue(base + 2);
        const int32_t originSeq = stagedUb.GetValue(base + 3);
        const int32_t destPos = stagedUb.GetValue(base + 4);
        const int32_t poolSlot = setIdx * static_cast<int32_t>(waysPerSet_) + way;
        const int64_t poolBlock = static_cast<int64_t>(poolSlot) / selKvBlockSize_;
        const int64_t poolBlockOff = static_cast<int64_t>(poolSlot) % selKvBlockSize_;
        dstKvAddr = static_cast<int64_t>(b) * poolBlocksPerBatch_ * selKvBlockStride_
            + poolBlock * selKvBlockStride_ + poolBlockOff * kvCacheDim_;
        dstRopeAddr = static_cast<int64_t>(b) * poolBlocksPerBatch_ * selRopeBlockStride_
            + poolBlock * selRopeBlockStride_ + poolBlockOff * kRopeDim_;
        if (metadataUpdate_ != 0) {
            const int64_t poolIdOff = static_cast<int64_t>(b) * poolSize_ + poolSlot;
            const int32_t oldId = poolIdsGm_.GetValue(poolIdOff);
            if (oldId >= 0 && oldId < idRange_) {
                idToSlotGm_.SetValue(static_cast<int64_t>(b) * idRange_ + oldId, -1);
            }
            poolIdsGm_.SetValue(poolIdOff, topKId);
            if (topKId >= 0 && topKId < idRange_) {
                idToSlotGm_.SetValue(static_cast<int64_t>(b) * idRange_ + topKId, poolSlot);
            }
        }

        DataCopyExtParams kvExt{static_cast<uint16_t>(1),
            static_cast<uint32_t>(kvCacheDim_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyExtParams ropeExt{static_cast<uint16_t>(1),
            static_cast<uint32_t>(kRopeDim_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(kvBuf, selectionKvGm_[SelKvSrcAddr(b, originSeq, destPos)], kvExt, padParams);
        DataCopyPad(ropeBuf, selectionRopeGm_[SelRopeSrcAddr(b, originSeq, destPos)], ropeExt, padParams);
    }

    __aicore__ inline void MutatePoolChunkFromUb(int32_t b, int32_t chunk, LocalTensor<int32_t>& stagedUb)
    {
        if (chunk <= 0) {
            return;
        }
        const int64_t kvPayloadBytes = Align32(kvCacheDim_ * static_cast<int64_t>(sizeof(T)));
        const int64_t ropePayloadBytes = Align32(kRopeDim_ * static_cast<int64_t>(sizeof(T)));
        const int64_t kvEntries = kvPayloadBytes / static_cast<int64_t>(sizeof(T));
        const int64_t ropeEntries = ropePayloadBytes / static_cast<int64_t>(sizeof(T));
        LocalTensor<T> kvA = workBuf_.Get<T>()[kvPayloadOff_ / sizeof(T)];
        LocalTensor<T> kvB = workBuf_.Get<T>()[kvPayloadOff_ / sizeof(T) + kvEntries];
        LocalTensor<T> ropeA = workBuf_.Get<T>()[ropePayloadOff_ / sizeof(T)];
        LocalTensor<T> ropeB = workBuf_.Get<T>()[ropePayloadOff_ / sizeof(T) + ropeEntries];

        DataCopyExtParams kvExt{static_cast<uint16_t>(1),
            static_cast<uint32_t>(kvCacheDim_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyExtParams ropeExt{static_cast<uint16_t>(1),
            static_cast<uint32_t>(kRopeDim_ * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        int64_t curDstKvAddr = 0;
        int64_t curDstRopeAddr = 0;
        PrefetchStagedRecord(b, stagedUb, 0, kvA, ropeA, curDstKvAddr, curDstRopeAddr);
        for (int32_t i = 0; i < chunk; ++i) {
            const bool useA = ((i & 1) == 0);
            LocalTensor<T>& curKv = useA ? kvA : kvB;
            LocalTensor<T>& nextKv = useA ? kvB : kvA;
            LocalTensor<T>& curRope = useA ? ropeA : ropeB;
            LocalTensor<T>& nextRope = useA ? ropeB : ropeA;
            SetWaitEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            const int64_t writeDstKvAddr = curDstKvAddr;
            const int64_t writeDstRopeAddr = curDstRopeAddr;
            if (i + 1 < chunk) {
                if (i >= 1) {
                    SetWaitEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                }
                PrefetchStagedRecord(b, stagedUb, i + 1, nextKv, nextRope, curDstKvAddr, curDstRopeAddr);
            }
            DataCopyPad(sharedPoolKvGm_[writeDstKvAddr], curKv, kvExt);
            DataCopyPad(sharedPoolRopeGm_[writeDstRopeAddr], curRope, ropeExt);
        }
        SetWaitEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    TPipe* pipe_;
    const DsaInstallTilingData* tiling_;
    int32_t blkIdx_ = 0;
    int64_t batchsize_ = 0;
    int64_t rawSeq_ = 0;
    int64_t origBatch_ = 0;
    int64_t numSets_ = 0;
    int64_t waysPerSet_ = 0;
    int64_t maxRecordsPerSeq_ = 0;
    int64_t installRecordsStride_ = 0;
    int64_t kvCacheDim_ = 0;
    int64_t kRopeDim_ = 0;
    int64_t selKvBlockSize_ = 0;
    int64_t poolSize_ = 0;
    int64_t idRange_ = 0;
    int64_t bitsetInt32s_ = 0;
    int64_t poolBlocksPerBatch_ = 0;
    int64_t selKvBlockStride_ = 0;
    int64_t selRopeBlockStride_ = 0;
    int64_t headnum_ = 0;
    int64_t selMaxBlockNum_ = 0;
    int64_t metadataUpdate_ = 1;
    int64_t bitsetOff_ = 0;
    int64_t lruOff_ = 0;
    int64_t recordsOff_ = 0;
    int64_t stagedOff_ = 0;
    int64_t kvPayloadOff_ = 0;
    int64_t ropePayloadOff_ = 0;
    TBuf<TPosition::VECCALC> workBuf_;
    GlobalTensor<T> selectionKvGm_;
    GlobalTensor<T> selectionRopeGm_;
    GlobalTensor<T> sharedPoolKvGm_;
    GlobalTensor<T> sharedPoolRopeGm_;
    GlobalTensor<int32_t> selBlkTableGm_;
    GlobalTensor<int32_t> poolIdsGm_;
    GlobalTensor<int32_t> idToSlotGm_;
    GlobalTensor<int32_t> lruCounterGm_;
    GlobalTensor<int32_t> installRecordsGm_;
    GlobalTensor<int32_t> installCountsGm_;
};
}  // namespace

extern "C" __global__ __aicore__ void dsa_install(
    GM_ADDR install_records,
    GM_ADDR selection_kv_cache,
    GM_ADDR selection_k_rope,
    GM_ADDR selection_kv_block_table,
    GM_ADDR pool_kv_cache,
    GM_ADDR pool_k_rope,
    GM_ADDR pool_ids,
    GM_ADDR id_to_slot,
    GM_ADDR lru_counter,
    GM_ADDR pool_kv_next,
    GM_ADDR pool_k_rope_next,
    GM_ADDR pool_ids_next,
    GM_ADDR id_to_slot_next,
    GM_ADDR lru_counter_next,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    (void)pool_kv_next;
    (void)pool_k_rope_next;
    (void)pool_ids_next;
    (void)id_to_slot_next;
    (void)lru_counter_next;
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    const uint64_t rawAddr = reinterpret_cast<uint64_t>(install_records);
    const uint64_t alignedAddr = (rawAddr + 4095u) & ~static_cast<uint64_t>(4095u);
    const int64_t alignedOff = static_cast<int64_t>(alignedAddr - rawAddr);
    ASSERT(alignedOff >= 0 && alignedOff < 4096);
    ASSERT(alignedOff + tilingData.installRecordsRequiredBytes <= tilingData.installRecordsAllocatedBytes);
    GM_ADDR alignedBase = reinterpret_cast<GM_ADDR>(alignedAddr);

    TPipe consumerPipe;
    DsaInstallNonInt8Consumer<DTYPE_SELECTION_KV_CACHE> consumer(&consumerPipe, &tilingData);
    consumer.Init(selection_kv_cache, selection_k_rope, selection_kv_block_table,
        pool_kv_cache, pool_k_rope, pool_ids, id_to_slot, lru_counter, alignedBase);
    consumer.ProcessParallel();

    // Match the proven non-int8 Split Install contract: publish payload and
    // metadata writes before GE exposes the ref outputs to later graph users.
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
}
