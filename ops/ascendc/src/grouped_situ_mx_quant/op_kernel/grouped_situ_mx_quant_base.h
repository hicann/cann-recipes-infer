/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_situ_mx_quant_base.h
 * \brief Fused sparse-SiTU activation + MXFP8(e4m3fn, group=32) quantization.
 *
 * Backbone: situ_and_mul_sparse (per-row pipeline, sparse validRows, register
 * SiTU activation VF). The activation result stays in UB (never spilled to GM)
 * and is quantized in place by the ported dynamic_mx_quant tail-axis e4m3 OCP
 * three-pass VF. Bit-identical (within validRows) to the two small ops chained
 * npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn).
 *
 * VF form: __simd_vf__ functions invoked via asc_vf_call (the arch35 built-in
 * convention, e.g. dawsn/acos). asc_vf_call carries the MTE/VF boundary sync, so
 * the activation VF's actBuf store is ordered before the quant VF's load without
 * a manual drain.
 */

#ifndef GROUPED_SITU_MX_QUANT_BASE_H
#define GROUPED_SITU_MX_QUANT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grouped_situ_mx_quant_quant.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace GroupedSituMxQuant {
using namespace AscendC;

constexpr uint32_t VF_TILE_FLOATS = 64U;  // one full fp32 vector register (256B / 4)
constexpr int64_t QUANT_BLOCK = 32;       // MXFP8 group_size

__aicore__ inline int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }
__aicore__ inline int64_t CeilAlignI(int64_t a, int64_t b) { return CeilDivI(a, b) * b; }

// ---------------------------------------------------------------------------
// SiTU activation VF (__simd_vf__; ported verbatim from situ_and_mul_sparse).
// 64-fp32 tile per iter. Output is T (bf16/fp16) written to `out`; when
// kRoundToInput, the fp32->T->fp32 round-trip matches the small-op activated
// tensor exactly. Invoked via asc_vf_call.
// ---------------------------------------------------------------------------
template <typename T, bool kRoundToInput>
__simd_vf__ inline void SituActivationVF(__ubuf__ T* gate, __ubuf__ T* up, __ubuf__ T* out,
                                         uint32_t elements, float neg2OverBeta, float negBeta,
                                         float neg2OverAlpha, float negAlpha, float beta, float alpha)
{
    static constexpr Reg::CastTrait kF32ToTCastTrait = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    const uint16_t tileCount = static_cast<uint16_t>((elements + VF_TILE_FLOATS - 1U) / VF_TILE_FLOATS);
    uint32_t remaining = elements;
    Reg::RegTensor<float> gateF;
    Reg::RegTensor<float> upF;
    Reg::RegTensor<float> tmpF;
    Reg::RegTensor<float> branchF;
    Reg::RegTensor<T> packedF;
    Reg::MaskReg preg;
    for (uint16_t i = 0; i < tileCount; ++i) {
        preg = Reg::UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(i) * VF_TILE_FLOATS;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, gate + offset);
        Reg::Cast<float, T, kTToF32CastTrait>(gateF, packedF, preg);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, up + offset);
        Reg::Cast<float, T, kTToF32CastTrait>(upF, packedF, preg);

        // gate subchain
        Reg::Muls(tmpF, gateF, neg2OverBeta, preg);
        Reg::Exp(tmpF, tmpF, preg);
        Reg::Adds(tmpF, tmpF, 1.0f, preg);
        Reg::Duplicate(branchF, beta, preg);
        Reg::Div(branchF, branchF, tmpF, preg);
        Reg::Muls(branchF, branchF, 2.0f, preg);
        Reg::Adds(branchF, branchF, negBeta, preg);
        Reg::Muls(tmpF, gateF, -1.0f, preg);
        Reg::Exp(tmpF, tmpF, preg);
        Reg::Adds(tmpF, tmpF, 1.0f, preg);
        Reg::Div(branchF, branchF, tmpF, preg);
        if constexpr (kRoundToInput) {
            Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
            Reg::Cast<float, T, kTToF32CastTrait>(branchF, packedF, preg);
        }

        // up subchain
        Reg::Muls(tmpF, upF, neg2OverAlpha, preg);
        Reg::Exp(tmpF, tmpF, preg);
        Reg::Adds(tmpF, tmpF, 1.0f, preg);
        Reg::Duplicate(upF, alpha, preg);
        Reg::Div(upF, upF, tmpF, preg);
        Reg::Muls(upF, upF, 2.0f, preg);
        Reg::Adds(upF, upF, negAlpha, preg);
        if constexpr (kRoundToInput) {
            Reg::Cast<T, float, kF32ToTCastTrait>(packedF, upF, preg);
            Reg::Cast<float, T, kTToF32CastTrait>(upF, packedF, preg);
        }
        Reg::Mul(branchF, branchF, upF, preg);

        // exit: fp32 -> T then packed store
        Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(out + offset, packedF, preg);
    }
}

// ---------------------------------------------------------------------------
// Fused kernel class.
// ---------------------------------------------------------------------------
template <typename T>
class GroupedSituMxQuantKernel {
    using U = fp8_e4m3fn_t;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y, GM_ADDR mxscale,
                                const GroupedSituMxQuantTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        epr_ = tiling_->outputElementsPerRow;
        chunk_ = tiling_->chunkElements;
        groupCount_ = epr_ / QUANT_BLOCK;
        scaleColNum_ = tiling_->scaleColNum;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokens);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxscale);

        const int64_t scaleBufBytes = CeilAlignI(groupCount_, grouped_situ::VF_LEN_16) * sizeof(uint16_t);
        pipe->InitBuffer(gateupQueue_, 2, chunk_ * sizeof(T) * 2);
        pipe->InitBuffer(yQueue_, 2, chunk_ * sizeof(uint8_t));
        pipe->InitBuffer(mxScaleQueue_, 2, CeilAlignI(scaleColNum_, grouped_situ::VF_LEN_16) * sizeof(uint8_t));
        pipe->InitBuffer(actBuf_, chunk_ * sizeof(T));
        pipe->InitBuffer(maxExpBuf_, scaleBufBytes);
        pipe->InitBuffer(recipScaleBuf_, scaleBufBytes);

        // Assert saturation once for the e4m3 SAT cast in the quant pass (clamp to
        // 448 on overflow), matching dynamic_mx_quant / swiglu_group_quant which
        // set this at Init. The kernel entry saves/restores the original mode.
#if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    }

    __aicore__ inline void Process()
    {
        const int64_t totalRows = tiling_->totalElements / epr_;
        const int64_t validRows = LoadValidRows(totalRows);
        const int64_t block = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();
        for (int64_t row = block; row < validRows; row += blockCount) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline int64_t LoadValidRows(int64_t totalRows)
    {
        int64_t validRows = 0;
        for (int64_t i = 0; i < tiling_->expertTokenCount; ++i) {
            validRows += expertTokensGm_.GetValue(i);
        }
        return validRows > totalRows ? totalRows : (validRows > 0 ? validRows : 0);
    }

    __aicore__ inline void CopyIn(int64_t row)
    {
        LocalTensor<T> both = gateupQueue_.AllocTensor<T>();
        const int64_t inputRowOffset = row * epr_ * 2;
        // whole-row single chunk (chunk_ == epr_); gate/up halves adjacent in GM
        if ((2 * epr_ * static_cast<int64_t>(sizeof(T))) % 32 == 0) {
            DataCopy(both, xGm_[inputRowOffset], epr_ * 2);
        } else {
            DataCopy(both, xGm_[inputRowOffset], epr_);
            DataCopy(both[chunk_], xGm_[inputRowOffset + epr_], epr_);
        }
        gateupQueue_.EnQue(both);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> both = gateupQueue_.DeQue<T>();
        LocalTensor<uint8_t> y = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint16_t> scale = mxScaleQueue_.AllocTensor<uint16_t>();
        LocalTensor<T> act = actBuf_.template Get<T>();
        LocalTensor<uint16_t> maxExp = maxExpBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> recip = recipScaleBuf_.template Get<uint16_t>();

        __ubuf__ T* gate = (__ubuf__ T*)both.GetPhyAddr();
        __ubuf__ T* up = gate + chunk_;
        __ubuf__ T* actAddr = (__ubuf__ T*)act.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxExp.GetPhyAddr();
        __ubuf__ uint16_t* recipAddr = (__ubuf__ uint16_t*)recip.GetPhyAddr();
        __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)scale.GetPhyAddr();
        __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)y.GetPhyAddr();

        // --- SiTU activation -> act (bf16/fp16), same round-to-input as situ ---
        const uint32_t count = static_cast<uint32_t>(epr_);
        const float neg2OverBeta = -2.0f / tiling_->beta;
        const float negBeta = -tiling_->beta;
        const float neg2OverAlpha = -2.0f / tiling_->alpha;
        const float negAlpha = -tiling_->alpha;
        if (tiling_->highPrecision == 0) {
            asc_vf_call<SituActivationVF<T, true>>(gate, up, actAddr, count, neg2OverBeta, negBeta,
                                                   neg2OverAlpha, negAlpha, tiling_->beta, tiling_->alpha);
        } else {
            asc_vf_call<SituActivationVF<T, false>>(gate, up, actAddr, count, neg2OverBeta, negBeta,
                                                    neg2OverAlpha, negAlpha, tiling_->beta, tiling_->alpha);
        }

        // --- MXFP8 e4m3 OCP quantization (asc_vf_call, ported from dynamic_mx_quant) ---
        const uint16_t loopNum2VF = static_cast<uint16_t>(CeilDivI(groupCount_, grouped_situ::ELEMENT_AFTER_REDUCE));
        const uint16_t loopNum1VF = static_cast<uint16_t>(CeilDivI(groupCount_, grouped_situ::VF_LEN_16));
        asc_vf_call<grouped_situ::ComputeMaxExpOcpVF<T>>(actAddr, maxExpAddr, loopNum2VF);
        asc_vf_call<grouped_situ::ComputeScaleOcpVF>(maxExpAddr, scaleAddr, recipAddr, loopNum1VF,
                                                     static_cast<uint32_t>(groupCount_));
        asc_vf_call<grouped_situ::ComputeDataVF<T, U>>(actAddr, recipAddr, yAddr, loopNum2VF);

        yQueue_.EnQue(y);
        mxScaleQueue_.EnQue(scale);
        gateupQueue_.FreeTensor(both);
    }

    __aicore__ inline void CopyOut(int64_t row)
    {
        LocalTensor<uint16_t> scale = mxScaleQueue_.DeQue<uint16_t>();
        LocalTensor<uint8_t> scaleU8 = scale.template ReinterpretCast<uint8_t>();
        DataCopyExtParams scaleParams = {1U, static_cast<uint32_t>(scaleColNum_ * sizeof(uint8_t)), 0U, 0U, 0U};
        DataCopyPad(scaleGm_[row * scaleColNum_], scaleU8, scaleParams);
        mxScaleQueue_.FreeTensor(scale);

        LocalTensor<uint8_t> y = yQueue_.DeQue<uint8_t>();
        DataCopy(yGm_[row * epr_], y, epr_);
        yQueue_.FreeTensor(y);
    }

    const GroupedSituMxQuantTilingData* tiling_;
    int64_t epr_ = 0;
    int64_t chunk_ = 0;
    int64_t groupCount_ = 0;
    int64_t scaleColNum_ = 0;
    GlobalTensor<T> xGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    TQue<QuePosition::VECIN, 2> gateupQueue_;
    TQue<QuePosition::VECOUT, 2> yQueue_;
    TQue<QuePosition::VECOUT, 2> mxScaleQueue_;
    TBuf<QuePosition::VECCALC> actBuf_;
    TBuf<QuePosition::VECCALC> maxExpBuf_;
    TBuf<QuePosition::VECCALC> recipScaleBuf_;
};

}  // namespace GroupedSituMxQuant
#endif  // GROUPED_SITU_MX_QUANT_BASE_H
