/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_COMPRESS_EPILOG_V2_MXFP4_GROUP16_H
#define KV_COMPRESS_EPILOG_V2_MXFP4_GROUP16_H

#include "kv_compress_epilog_v2_common.h"

namespace KvCompressEpilogV2Ops {

constexpr uint32_t KCEV2_G16_GROUP_ELEMS = 16U;
constexpr uint32_t KCEV2_G16_CHUNK_ELEMS = 64U;
constexpr uint32_t KCEV2_G16_GROUPS_PER_CHUNK = 4U;
constexpr uint32_t KCEV2_G16_SCRATCH_ELEMS_PER_CHUNK = 16U;
constexpr uint16_t KCEV2_G16_BF16_ABS_MASK = 0x7FFFU;

constexpr AscendC::Reg::CastTrait KCEV2_G16_B16_TO_FP4 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__simd_callee__ inline void ComputeMxFp4Group16Scale(
    __ubuf__ uint16_t *scaleAddr, __ubuf__ uint16_t *scratchAddr,
    AscendC::Reg::RegTensor<uint16_t> &amaxBits, uint32_t validGroups)
{
    using namespace AscendC::Reg;
    RegTensor<uint16_t> expMask;
    RegTensor<uint16_t> maxExp;
    RegTensor<uint16_t> fp4MaxExp;
    RegTensor<uint16_t> sharedExp;
    RegTensor<uint16_t> scaleValue;
    RegTensor<uint16_t> invBias;
    RegTensor<uint16_t> halfScale;
    RegTensor<uint16_t> zero;
    RegTensor<uint16_t> nan;
    RegTensor<uint16_t> specialInv;
    MaskReg finiteMask;
    MaskReg nonZeroMask;
    MaskReg clampMask;
    MaskReg specialMask;
    uint32_t groupCount = validGroups;
    MaskReg groupMask = UpdateMask<uint16_t>(groupCount);
    UnalignRegForStore scaleStore;

    Duplicate(expMask, KCEV2_BF16_EXP_MASK);
    Duplicate(fp4MaxExp, KCEV2_FP4_E2M1_MAX_EXP);
    Duplicate(invBias, KCEV2_BF16_INV_BIAS);
    Duplicate(zero, static_cast<uint16_t>(0));
    Duplicate(nan, KCEV2_BF16_NAN);
    Duplicate(specialInv, KCEV2_FP4_SPECIAL_INV);

    And(maxExp, amaxBits, expMask, groupMask);
    Compare<uint16_t, CMPMODE::NE>(finiteMask, maxExp, expMask, groupMask);
    Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, groupMask);
    Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, fp4MaxExp, groupMask);
    Select<uint16_t>(maxExp, fp4MaxExp, maxExp, clampMask);
    Sub(sharedExp, maxExp, fp4MaxExp, groupMask);
    Select<uint16_t>(scaleValue, sharedExp, nan, finiteMask);
    Select<uint16_t>(scaleValue, scaleValue, zero, nonZeroMask);
    StoreUnAlign(scaleAddr, scaleValue, scaleStore, validGroups);
    StoreUnAlignPost(scaleAddr, scaleStore, 0);

    Sub(halfScale, invBias, sharedExp, groupMask);
    Select<uint16_t>(halfScale, halfScale, nan, finiteMask);
    Select<uint16_t>(halfScale, halfScale, zero, nonZeroMask);
    Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, invBias, groupMask);
    Select<uint16_t>(halfScale, specialInv, halfScale, specialMask);
    StoreAlign(scratchAddr, halfScale, groupMask);
}

__simd_vf__ inline void VFProcessMxFp4Group16VF(
    __ubuf__ int8_t *outputAddr, __ubuf__ uint16_t *scaleAddr,
    __ubuf__ bfloat16_t *inputAddr, __ubuf__ uint16_t *scratchAddr,
    uint16_t rowCount, uint32_t d, uint32_t xStride, uint32_t outputStride,
    uint32_t scaleStride, uint32_t chunkCount, uint32_t scratchRowStride)
{
    using namespace AscendC::Reg;
    {
        RegTensor<bfloat16_t> xChunk;
        RegTensor<uint16_t> absBits;
        RegTensor<uint16_t> absMask;
        RegTensor<uint16_t> amaxBits;
        RegTensor<bfloat16_t> halfScale;
        RegTensor<bfloat16_t> xQuant;
        RegTensor<fp4x2_e2m1_t> fp4Output;
        Duplicate(absMask, KCEV2_G16_BF16_ABS_MASK);

        for (uint16_t row = 0; row < rowCount; ++row) {
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                const uint32_t chunkOffset = chunk * KCEV2_G16_CHUNK_ELEMS;
                uint32_t validElems =
                    d - chunkOffset > KCEV2_G16_CHUNK_ELEMS ? KCEV2_G16_CHUNK_ELEMS : d - chunkOffset;
                const uint32_t validGroups = validElems / KCEV2_G16_GROUP_ELEMS;
                MaskReg dataMask = UpdateMask<bfloat16_t>(validElems);
                LoadAlign(xChunk, inputAddr + row * xStride + chunkOffset);
                And(absBits, reinterpret_cast<RegTensor<uint16_t> &>(xChunk), absMask, dataMask);
                ReduceDataBlock<AscendC::Reg::ReduceType::MAX, uint16_t>(amaxBits, absBits, dataMask);
                ComputeMxFp4Group16Scale(
                    scaleAddr + row * scaleStride + chunk * KCEV2_G16_GROUPS_PER_CHUNK,
                    scratchAddr + row * scratchRowStride + chunk * KCEV2_G16_SCRATCH_ELEMS_PER_CHUNK,
                    amaxBits, validGroups);
            }

            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                const uint32_t chunkOffset = chunk * KCEV2_G16_CHUNK_ELEMS;
                uint32_t validElems =
                    d - chunkOffset > KCEV2_G16_CHUNK_ELEMS ? KCEV2_G16_CHUNK_ELEMS : d - chunkOffset;
                MaskReg dataMask = UpdateMask<bfloat16_t>(validElems);
                LoadAlign(xChunk, inputAddr + row * xStride + chunkOffset);
                LoadAlign<uint16_t, LoadDist::DIST_E2B_B16>(
                    reinterpret_cast<RegTensor<uint16_t> &>(halfScale),
                    scratchAddr + row * scratchRowStride + chunk * KCEV2_G16_SCRATCH_ELEMS_PER_CHUNK);
                Mul(xQuant, xChunk, halfScale, dataMask);
                Cast<fp4x2_e2m1_t, bfloat16_t, KCEV2_G16_B16_TO_FP4>(fp4Output, xQuant, dataMask);
                StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(
                    outputAddr + row * outputStride + chunkOffset / 2U,
                    reinterpret_cast<RegTensor<int8_t> &>(fp4Output), dataMask);
            }
        }
    }
}

__aicore__ inline void VFProcessMxFp4Group16(
    const LocalTensor<int8_t> &output, const LocalTensor<bfloat16_t> &scale,
    const LocalTensor<bfloat16_t> &input, const LocalTensor<uint16_t> &scratch,
    uint16_t rowCount, uint32_t d)
{
    auto *inputAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    auto *outputAddr = reinterpret_cast<__ubuf__ int8_t *>(output.GetPhyAddr());
    auto *scaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(scale.GetPhyAddr());
    auto *scratchAddr = reinterpret_cast<__ubuf__ uint16_t *>(scratch.GetPhyAddr());
    const uint32_t xStride = RoundUp<bfloat16_t>(d);
    const uint32_t outputStride = RoundUp<int8_t>(d / 2U);
    const uint32_t scaleStride = RoundUp<uint16_t>(d / KCEV2_G16_GROUP_ELEMS);
    const uint32_t chunkCount = static_cast<uint32_t>(CeilDiv(d, KCEV2_G16_CHUNK_ELEMS));
    const uint32_t scratchRowStride = chunkCount * KCEV2_G16_SCRATCH_ELEMS_PER_CHUNK;
    VFProcessMxFp4Group16VF(outputAddr, scaleAddr, inputAddr, scratchAddr, rowCount, d,
                            xStride, outputStride, scaleStride, chunkCount, scratchRowStride);
}

template <typename TX, typename TSlot>
class KvCompressEpilogV2MxFp4Group16Kernel {
public:
    __aicore__ inline explicit KvCompressEpilogV2MxFp4Group16Kernel(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping, const KvCompressEpilogV2TilingData *tilingData)
    {
        tilingData_ = tilingData;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(x));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TSlot *>(slotMapping));
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(cache));
        pipe_->InitBuffer(xQueue_, 2,
                          tilingData_->rowFactor * RoundUp<TX>(tilingData_->d) * sizeof(TX));
        pipe_->InitBuffer(outputQueue_, 2,
                          tilingData_->rowFactor * RoundUp<int8_t>(tilingData_->dataCol));
        pipe_->InitBuffer(scaleQueue_, 2,
                          tilingData_->rowFactor * RoundUp<uint16_t>(tilingData_->scaleCol) * sizeof(uint16_t));
        const uint32_t chunkCount = CeilDiv(tilingData_->d, KCEV2_G16_CHUNK_ELEMS);
        pipe_->InitBuffer(scratchBuffer_, tilingData_->rowFactor * chunkCount * KCEV2_BLOCK_BYTES);
        pipe_->InitBuffer(paddingBuffer_, KCEV2_BLOCK_BYTES);
        pipe_->InitBuffer(indexBuffer_, RoundUp<TSlot>(tilingData_->rowFactor) * sizeof(TSlot));
        indexLocal_ = indexBuffer_.Get<TSlot>();
        paddingLocal_ = paddingBuffer_.Get<uint8_t>();
        Duplicate(paddingLocal_, static_cast<uint8_t>(0), KCEV2_BLOCK_BYTES);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        const int64_t block = GetBlockIdx();
        const int64_t rowLoops = block == GetBlockNum() - 1 ?
            tilingData_->rowLoopOfTailBlock : tilingData_->rowLoopOfFormerBlock;
        const int64_t tailRows = block == GetBlockNum() - 1 ?
            tilingData_->tailRowFactorOfTailBlock : tilingData_->tailRowFactorOfFormerBlock;
        const int64_t blockRowBase = block * tilingData_->rowOfFormerBlock;

        for (int64_t loop = 0; loop < rowLoops; ++loop) {
            const int64_t currentRows = loop == rowLoops - 1 ? tailRows : tilingData_->rowFactor;
            LocalTensor<TX> xLocal = xQueue_.template AllocTensor<TX>();
            int64_t validRows = 0;
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t inputRow = blockRowBase + loop * tilingData_->rowFactor + row;
                const int64_t slot = static_cast<int64_t>(slotGm_.GetValue(inputRow));
                if (slot < 0 || slot >= tilingData_->cacheRows) {
                    continue;
                }
                CopyIn(xGm_[inputRow * tilingData_->d],
                       xLocal[validRows * RoundUp<TX>(tilingData_->d)], tilingData_->d);
                indexLocal_.SetValue(validRows, static_cast<TSlot>(slot));
                ++validRows;
            }
            xQueue_.template EnQue<TX>(xLocal);
            xLocal = xQueue_.template DeQue<TX>();
            if (validRows == 0) {
                xQueue_.FreeTensor(xLocal);
                continue;
            }

            LocalTensor<int8_t> outputLocal = outputQueue_.template AllocTensor<int8_t>();
            LocalTensor<bfloat16_t> scaleLocal = scaleQueue_.template AllocTensor<bfloat16_t>();
            LocalTensor<uint16_t> scratchLocal = scratchBuffer_.Get<uint16_t>();
            VFProcessMxFp4Group16(outputLocal, scaleLocal, xLocal, scratchLocal,
                                  static_cast<uint16_t>(validRows), tilingData_->d);
            xQueue_.FreeTensor(xLocal);
            outputQueue_.template EnQue<int8_t>(outputLocal);
            scaleQueue_.template EnQue<bfloat16_t>(scaleLocal);
            outputLocal = outputQueue_.template DeQue<int8_t>();
            scaleLocal = scaleQueue_.template DeQue<bfloat16_t>();
            for (int64_t row = 0; row < validRows; ++row) {
                const int64_t slot = static_cast<int64_t>(indexLocal_.GetValue(row));
                const int64_t cacheOffset = slot * tilingData_->cacheRowStride;
                CopyOutBytes(outputLocal.template ReinterpretCast<uint8_t>()[
                                 row * RoundUp<int8_t>(tilingData_->dataCol)],
                             cacheGm_[cacheOffset], tilingData_->dataCol);
                CopyOutBytes(scaleLocal.template ReinterpretCast<uint8_t>()[
                                 row * RoundUp<uint16_t>(tilingData_->scaleCol) * sizeof(uint16_t)],
                             cacheGm_[cacheOffset + tilingData_->dataCol],
                             tilingData_->scaleCol * sizeof(bfloat16_t));
                if (tilingData_->padCol > 0) {
                    CopyOutBytes(paddingLocal_, cacheGm_[cacheOffset + tilingData_->concatCol],
                                 tilingData_->padCol);
                }
            }
            outputQueue_.FreeTensor(outputLocal);
            scaleQueue_.FreeTensor(scaleLocal);
        }
    }

private:
    TPipe *pipe_ = nullptr;
    const KvCompressEpilogV2TilingData *tilingData_ = nullptr;
    GlobalTensor<TX> xGm_;
    GlobalTensor<TSlot> slotGm_;
    GlobalTensor<uint8_t> cacheGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;
    TBuf<QuePosition::VECCALC> scratchBuffer_;
    TBuf<QuePosition::VECCALC> paddingBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    LocalTensor<TSlot> indexLocal_;
    LocalTensor<uint8_t> paddingLocal_;
};

}  // namespace KvCompressEpilogV2Ops

#endif
