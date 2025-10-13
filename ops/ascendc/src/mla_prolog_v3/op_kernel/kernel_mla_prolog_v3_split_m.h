/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file kernel_mla_prolog_v3_split_n.h
 * \brief
 */

#ifndef KERNEL_MLA_PROLOG_V3_SPLIT_M_H
#define KERNEL_MLA_PROLOG_V3_SPLIT_M_H

#include "mla_prolog_v3_comm.h"
#include "mla_prolog_v3_vector_comm.h"
#include "service_matmul.h"
#include "service_rms_norm.h"
#include "service_gather_sin_cos.h"
#include "service_rotary_position_embedding.h"
#include "service_scatter_cache.h"
#include "service_dequant.h"
#include "service_dynamic_quant_qn_mul_qr.h"

namespace MlaPrologV3 {
template <typename MLAPT>
class MlaPrologV3SplitM {
public:
    __aicore__ inline MlaPrologV3SplitM(TPipe* pipe, const MlaPrologV3TilingData* __restrict tilingData,
                                          const MlaPrologV3BaseParams* __restrict baseParams)
        : pipe_(pipe), tilingData_(tilingData), baseParams_(baseParams) {}

    __aicore__ inline void Init(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr,
                                __gm__ uint8_t *weightUk, __gm__ uint8_t *weightDkvKr,
                                __gm__ uint8_t *rmsnormGammaCq, __gm__ uint8_t *rmsnormGammaCkv,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos,
                                __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache,
                                __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t * dequantScaleWDkvkr,
                                __gm__ uint8_t *actualSeqLen,
                                __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,__gm__ uint8_t *smoothScaleCq,
                                __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *dequantScaleQNopeOut,
                                __gm__ uint8_t *queryNormOut, __gm__ uint8_t *dequantScaleQNormOut, __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyGlobalParams();
    __aicore__ inline void ScaleInit( __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,__gm__ uint8_t *deqScaleQcQrW,
        __gm__ uint8_t *dequantScaleWDkvkr, __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScaleCq);
    __aicore__ inline void WorkspaceInit(__gm__ uint8_t *workspace);
    __aicore__ inline void MmParamInit();
    __aicore__ inline void MmCqParamInit();
    __aicore__ inline void MmCkvKrParamInit();
    __aicore__ inline void MmQcQrParamInit();
    __aicore__ inline void MmQnParamInit();
    __aicore__ inline void CubeBufferInit();
    __aicore__ inline void VectorBufferInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curMSize);
    __aicore__ inline void ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset);
    __aicore__ inline void ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset);
    __aicore__ inline void AicProcess(AicOffset &aicOffset, int64_t batchOffset, int64_t mmQnLoops);
    __aicore__ inline void AivProcess(AivOffset &aivOffset, int64_t batchOffset, int64_t curMSize, int64_t numHeadOffset, int64_t mmQnLoops);
    template <typename T, typename O, bool needCheckEmptyTensor = false, bool needCheckAFullLoad = false, bool isContinuousCopy = true>
    __aicore__ inline void MatmulSplitN(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm, 
                                        const MMParams &mmPara, const UsedBlockParams &mmBlockParams);
    template <typename T, typename O, bool needCheckEmptyTensor = false, bool needCheckAFullLoad = false, bool isContinuousCopy = true>
    __aicore__ inline void MatmulSplitM(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm, 
                                        const MMParams &mmPara, const UsedBlockParams &mmBlockParams);
    __aicore__ inline void MatmulAndSyncQcQr(AicOffset &aicOffset);
    __aicore__ inline void MatmulQcQrSyncDequant(int64_t weightUqQrOffset,int64_t qcQrResOffset);
    __aicore__ inline void PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops);
    __aicore__ inline void MatmulQnWeightPreload(int64_t weightUkOffset, int64_t mmQnLoops);
    __aicore__ inline void MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset, int64_t qnResOffset, int64_t mmQnLoops);
    __aicore__ inline void CopyInSinCos(int64_t tokenIndex, int64_t curVecToken, int64_t batchOffset, int64_t curMSize);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void RopeAndScatterKr(LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                            LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr, LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                            CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void RmsNormAndScatterCkv(LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                                LocalTensor<typename MLAPT::ropeComputType>& cosLocalCkvKr, LocalTensor<typename MLAPT::ropeComputType>& sinLocalCkvKr,
                                                CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset, int64_t curVecToken);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void DequantQc(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t curVecToken, int64_t curBlockTokenOffset);
    // 低时延算力分组场景
    __aicore__ inline void DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t qcQrScaleOffset);
    __aicore__ inline void RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantQcQrSplitN(const DequantQcQrSplitNParams& dequantQcQrSplitN);
    __aicore__ inline void RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams);
    __aicore__ inline void QcQrSplit(int64_t curVecToken, int64_t curBlockTokenOffset, int64_t curMSize, int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                            int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curMSize, int64_t curBlockTokenOffset, int64_t numHeadOffset, int64_t mmQnLoops);

public:
    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using kvCacheType = typename MLAPT::kvCacheType;
    using krCacheType = typename MLAPT::krCacheType;
    using dequantScaleQNopeType = typename MLAPT::dequantScaleQNopeType;

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;
    MMParams mmQnParam_;

    static constexpr bool needQnDynamicQuant = IsSameType<mmInputType, int8_t>::value;

private:
    TPipe* pipe_;
    const MlaPrologV3TilingData* __restrict tilingData_;
    const MlaPrologV3BaseParams* __restrict baseParams_;
    uint32_t blockIdx_ = 0U;
    uint32_t cubeBlockIdx_ = 0U; // AIV上使用AIC的blockIdx
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;
    bool enableSmoothScalesCq_;

    uint32_t mSubSize_ = 0;
    uint32_t mOffsetStart_ = 0;

    struct DequantTool{
        GlobalTensor<float> deQuantScaleCqGm_;
        TBuf<TPosition::VECCALC> deQuantScaleCqBuffer_;  // 用于临时存储每一行的Scale，以及汇总最终每一行的Scale参数
        LocalTensor<float> deQuantScaleCqLocal_;
        __aicore__ inline DequantTool() {}
    };

    // 算子分组开关
    DequantTool dequantTool_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<kvCacheType> kvCacheGm_;
    GlobalTensor<krCacheType> krCacheGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;

    GlobalTensor<float> dequantScaleXGm_;
    GlobalTensor<float> dequantScaleWDqGm_;
    GlobalTensor<float> dequantScaleWDkvkrGm_;
    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<float> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;
    GlobalTensor<dequantScaleQNopeType> dequantScaleQNopeGm_;
    GlobalTensor<kvCacheType> queryOutGm_;

    // UB
    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDqBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDkvKrBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCqBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCkvBuffer_;
    TBuf<TPosition::VECCALC> smoothScaleCqBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkvBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkrBuffer_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;
    LocalTensor<float> dequantScaleWDqLocal_;
    LocalTensor<float> dequantScaleWDkvKrLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCqLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCkvLocal_;
    LocalTensor<float> smoothScaleCqLocal_;
    LocalTensor<float> quantScaleCkvLocal_;
    LocalTensor<float> quantScaleCkrLocal_;

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
};

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::Init(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk, __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv, __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache, __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr, __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *actualSeqLen, __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut, __gm__ uint8_t *dequantScaleQNormOut, __gm__ uint8_t *workspace) {
    blockIdx_ = GetBlockIdx(); // cube:0-23  vec:0-47
    if ASCEND_IS_AIV {
        cubeBlockIdx_ = blockIdx_ >> 1;
    } else {
        cubeBlockIdx_ = blockIdx_;
    }
    curVectorBlockNum_ = static_cast<int64_t>(baseParams_->stepBatchSize);
    vectorCoreNum_ = static_cast<int64_t>(baseParams_->vectorBlockNum); // aivNum 48
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;
    enableSmoothScalesCq_ = smoothScaleCq == nullptr ? false : true;
    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType *)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDq);   // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)weightUqQr);   // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType *)weightUk);
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDkvKr);  // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCkv);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeCos);
    cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t *)cacheIndex);
    kvCacheGm_.SetGlobalBuffer((__gm__ kvCacheType *)kvCache);
    krCacheGm_.SetGlobalBuffer((__gm__ krCacheType *)krCache);
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType *)queryRopeOut);
    if constexpr (std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) {
        dequantScaleQNopeGm_.SetGlobalBuffer((__gm__ dequantScaleQNopeType *)dequantScaleQNopeOut);
        queryOutGm_.SetGlobalBuffer((__gm__ kvCacheType *)queryOut);
    } else {
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)queryOut);
    }
    // if (blockIdx_ == 0) {
    //     printf(
    //         "blockIdx_:%d, "
    //         "curVectorBlockNum_:%d, "
    //         "vectorCoreNum_:%d, "
    //         "curVecTokenMax_:%d, "
    //         "baseParams_->stepBatchSize:%d, "
    //         "baseParams_->mSubSize:%d, "
    //         "baseParams_->mSubCoreNum:%d\n"
    //         , blockIdx_
    //         , curVectorBlockNum_
    //         , vectorCoreNum_
    //         , curVecTokenMax_
    //         , baseParams_->stepBatchSize
    //         , baseParams_->mSubSize
    //         , baseParams_->mSubCoreNum
    //     );
    // }
    ScaleInit(dequantScaleX, dequantScaleWDq, deqScaleQcQrW, dequantScaleWDkvkr, quantScaleCkv, quantScaleCkr, smoothScaleCq);
    MmParamInit();
    WorkspaceInit(workspace);
    if ASCEND_IS_AIV {
        VectorBufferInit();
    } else {
        CubeBufferInit();
    }

}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ScaleInit( __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr, __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScaleCq) {
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        dequantScaleXGm_.SetGlobalBuffer((__gm__ float *)dequantScaleX);
        dequantScaleWDqGm_.SetGlobalBuffer((__gm__ float *)dequantScaleWDq);
        dequantScaleWDkvkrGm_.SetGlobalBuffer((__gm__ float *)dequantScaleWDkvkr);
    }
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        smoothScaleCqGm_.SetGlobalBuffer((__gm__ float *)smoothScaleCq);
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ float *)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkv);
        quantScaleCkrGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkr);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmParamInit() {
    MmCqParamInit();
    MmCkvKrParamInit();
    MmQcQrParamInit();
    MmQnParamInit();
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmCqParamInit() {
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    mmCqParam_.k = baseParams_->headSizeX; // 7168
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKc = baseParams_->headSizeCq;  // 1536
    mmCqParam_.baseK = (sizeof(mmInputType) == sizeof(int8_t)) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCqParam_.baseN = 128;
    mmCqParam_.stepK = 4;
    mmCqParam_.kL1StepSize = mmCqParam_.baseK * mmCqParam_.stepK;
    // mmCqParam_.tailStepK = (mmCqParam_.k % mmCqParam_.kL1StepSize != 0) ? (
    //     (mmCqParam_.k % mmCqParam_.kL1StepSize) / mmCqParam_.baseK) : mmCqParam_.stepK;
    // mmCqParam_.kL1TailStepSize = mmCqParam_.tailStepK * mmCqParam_.baseK;
    // if (blockIdx_ == 0) {
    //     printf(
    //         "mmCqParam_.m:%d, "
    //         "mmCqParam_.n:%d, "
    //         "mmCqParam_.k:%d, "
    //         "mmCqParam_.orgM:%d, "
    //         "mmCqParam_.orgN:%d, "
    //         "mmCqParam_.orgKa:%d, "
    //         "mmCqParam_.orgKb:%d, "
    //         "mmCqParam_.orgKc:%d, "
    //         "mmCqParam_.baseM:%d, "
    //         "mmCqParam_.baseN:%d, "
    //         "mmCqParam_.baseK:%d, "
    //         "mmCqParam_.stepK:%d, "
    //         "mmCqParam_.needSetOrgShape:%d, "
    //         "mmCqParam_.kL1StepSize:%d, "
    //         "mmCqParam_.tailStepK:%d, "
    //         "mmCqParam_.kL1TailStepSize:%d \n"
    //         , mmCqParam_.m
    //         , mmCqParam_.n
    //         , mmCqParam_.k
    //         , mmCqParam_.orgM
    //         , mmCqParam_.orgN
    //         , mmCqParam_.orgKa
    //         , mmCqParam_.orgKb
    //         , mmCqParam_.orgKc
    //         , mmCqParam_.baseM
    //         , mmCqParam_.baseN
    //         , mmCqParam_.baseK
    //         , mmCqParam_.stepK
    //         , mmCqParam_.needSetOrgShape
    //         , mmCqParam_.kL1StepSize
    //         , mmCqParam_.tailStepK
    //         , mmCqParam_.kL1TailStepSize
    //     );
    // }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmCkvKrParamInit() {
    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 32
    mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKc = (baseParams_->headSizeCkv + baseParams_->dimHeadRope);  // 576
    mmCkvKrParam_.baseK = (sizeof(mmInputType) == sizeof(int8_t)) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCkvKrParam_.baseN = 128;
    mmCkvKrParam_.stepK = 4;
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;
    // mmCkvKrParam_.tailStepK = (mmCkvKrParam_.k % mmCkvKrParam_.kL1StepSize != 0) ? (
    //     (mmCkvKrParam_.k % mmCkvKrParam_.kL1StepSize) / mmCkvKrParam_.baseK) : mmCkvKrParam_.stepK;
    // mmCkvKrParam_.kL1TailStepSize = mmCkvKrParam_.tailStepK * mmCkvKrParam_.baseK;
    // if (blockIdx_ == 0) {
    //     printf(
    //         "mmCkvKrParam_.m:%d, "
    //         "mmCkvKrParam_.n:%d, "
    //         "mmCkvKrParam_.k:%d, "
    //         "mmCkvKrParam_.orgM:%d, "
    //         "mmCkvKrParam_.orgN:%d, "
    //         "mmCkvKrParam_.orgKa:%d, "
    //         "mmCkvKrParam_.orgKb:%d, "
    //         "mmCkvKrParam_.orgKc:%d, "
    //         "mmCkvKrParam_.baseM:%d, "
    //         "mmCkvKrParam_.baseN:%d, "
    //         "mmCkvKrParam_.baseK:%d, "
    //         "mmCkvKrParam_.stepK:%d, "
    //         "mmCkvKrParam_.needSetOrgShape:%d, "
    //         "mmCkvKrParam_.kL1StepSize:%d, "
    //         "mmCkvKrParam_.tailStepK:%d, "
    //         "mmCkvKrParam_.kL1TailStepSize:%d \n"
    //         , mmCkvKrParam_.m
    //         , mmCkvKrParam_.n
    //         , mmCkvKrParam_.k
    //         , mmCkvKrParam_.orgM
    //         , mmCkvKrParam_.orgN
    //         , mmCkvKrParam_.orgKa
    //         , mmCkvKrParam_.orgKb
    //         , mmCkvKrParam_.orgKc
    //         , mmCkvKrParam_.baseM
    //         , mmCkvKrParam_.baseN
    //         , mmCkvKrParam_.baseK
    //         , mmCkvKrParam_.stepK
    //         , mmCkvKrParam_.needSetOrgShape
    //         , mmCkvKrParam_.kL1StepSize
    //         , mmCkvKrParam_.tailStepK
    //         , mmCkvKrParam_.kL1TailStepSize
    //     );
    // }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmQcQrParamInit() {
    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32
    mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? 128 : 64;
    if constexpr (MLAPT::enableGroupComputeOpt) {
        mmQcQrParam_.baseN = 128;
    } else {
        mmQcQrParam_.baseN = 128;
    }
    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    // mmQcQrParam_.tailStepK = (mmQcQrParam_.k % mmQcQrParam_.kL1StepSize != 0) ? (
    //     (mmQcQrParam_.k % mmQcQrParam_.kL1StepSize) / mmQcQrParam_.baseK) : mmQcQrParam_.stepK;
    // mmQcQrParam_.kL1TailStepSize = mmQcQrParam_.tailStepK * mmQcQrParam_.baseK;
    // if (blockIdx_ == 0) {
    //     printf(
    //         "mmQcQrParam_.m:%d, "
    //         "mmQcQrParam_.n:%d, "
    //         "mmQcQrParam_.k:%d, "
    //         "mmQcQrParam_.orgM:%d, "
    //         "mmQcQrParam_.orgN:%d, "
    //         "mmQcQrParam_.orgKa:%d, "
    //         "mmQcQrParam_.orgKb:%d, "
    //         "mmQcQrParam_.orgKc:%d, "
    //         "mmQcQrParam_.baseM:%d, "
    //         "mmQcQrParam_.baseN:%d, "
    //         "mmQcQrParam_.baseK:%d, "
    //         "mmQcQrParam_.stepK:%d, "
    //         "mmQcQrParam_.needSetOrgShape:%d, "
    //         "mmQcQrParam_.kL1StepSize:%d, "
    //         "mmQcQrParam_.tailStepK:%d, "
    //         "mmQcQrParam_.kL1TailStepSize:%d \n"
    //         , mmQcQrParam_.m
    //         , mmQcQrParam_.n
    //         , mmQcQrParam_.k
    //         , mmQcQrParam_.orgM
    //         , mmQcQrParam_.orgN
    //         , mmQcQrParam_.orgKa
    //         , mmQcQrParam_.orgKb
    //         , mmQcQrParam_.orgKc
    //         , mmQcQrParam_.baseM
    //         , mmQcQrParam_.baseN
    //         , mmQcQrParam_.baseK
    //         , mmQcQrParam_.stepK
    //         , mmQcQrParam_.needSetOrgShape
    //         , mmQcQrParam_.kL1StepSize
    //         , mmQcQrParam_.tailStepK
    //         , mmQcQrParam_.kL1TailStepSize
    //     );
    // }
}

// template<typename MLAPT>
// __aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmQnParamInit() {
//     mmQnParam_.m = baseParams_->stepBatchSize; // 32
//     mmQnParam_.n = baseParams_->headSizeCkv; // 512
//     mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
//     mmQnParam_.needSetOrgShape = 1;
//     mmQnParam_.orgM = mmQnParam_.m;
//     mmQnParam_.orgN = mmQnParam_.n;
//     if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
//         mmQnParam_.orgKa = baseParams_->headSizeQc;
//     } else {
//         mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
//     }
//     mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
//     mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;

//     mmQnParam_.baseK = mmQnParam_.k;
//     mmQnParam_.baseN = 128;
//     mmQnParam_.stepK = 1;
//     mmQnParam_.kL1StepSize = mmQnParam_.baseK * mmQnParam_.stepK;
//     mmQnParam_.tailStepK = mmQnParam_.stepK;
//     mmQnParam_.kL1TailStepSize = mmQnParam_.tailStepK * mmQnParam_.baseK;
//     if (blockIdx_ == 0) {
//         printf(
//             "mmQnParam_.m:%d, "
//             "mmQnParam_.n:%d, "
//             "mmQnParam_.k:%d, "
//             "mmQnParam_.orgM:%d, "
//             "mmQnParam_.orgN:%d, "
//             "mmQnParam_.orgKa:%d, "
//             "mmQnParam_.orgKb:%d, "
//             "mmQnParam_.orgKc:%d, "
//             "mmQnParam_.baseM:%d, "
//             "mmQnParam_.baseN:%d, "
//             "mmQnParam_.baseK:%d, "
//             "mmQnParam_.stepK:%d, "
//             "mmQnParam_.needSetOrgShape:%d, "
//             "mmQnParam_.kL1StepSize:%d, "
//             "mmQnParam_.tailStepK:%d, "
//             "mmQnParam_.kL1TailStepSize:%d \n"
//             , mmQnParam_.m
//             , mmQnParam_.n
//             , mmQnParam_.k
//             , mmQnParam_.orgM
//             , mmQnParam_.orgN
//             , mmQnParam_.orgKa
//             , mmQnParam_.orgKb
//             , mmQnParam_.orgKc
//             , mmQnParam_.baseM
//             , mmQnParam_.baseN
//             , mmQnParam_.baseK
//             , mmQnParam_.stepK
//             , mmQnParam_.needSetOrgShape
//             , mmQnParam_.kL1StepSize
//             , mmQnParam_.tailStepK
//             , mmQnParam_.kL1TailStepSize
//         );
//     }
// }

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmQnParamInit() {
    mmQnParam_.m = baseParams_->stepBatchSize; // 32
    mmQnParam_.n = baseParams_->headSizeCkv; // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgM = mmQnParam_.m;
    mmQnParam_.orgN = mmQnParam_.n;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
    }
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
    mmQnParam_.kL1StepSize = mmQnParam_.k;
}


template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::VectorBufferInit() {

    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        pipe_->InitBuffer(dequantScaleWDqBuffer_, baseParams_->headSizeCq * sizeof(float)); // [1, 1536]
        dequantScaleWDqLocal_ = dequantScaleWDqBuffer_.Get<float>();

        pipe_->InitBuffer(dequantScaleWDkvKrBuffer_, (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float)); // [1, 512 + 64]
        dequantScaleWDkvKrLocal_ = dequantScaleWDkvKrBuffer_.Get<float>();
    }

    pipe_->InitBuffer(rmsnormGammaCqBuffer_, baseParams_->headSizeCq * sizeof(rmsNormGammaType)); // [1, 1536] bf16
    rmsnormGammaCqLocal_ = rmsnormGammaCqBuffer_.Get<rmsNormGammaType>();

    pipe_->InitBuffer(rmsnormGammaCkvBuffer_, baseParams_->headSizeCkv * sizeof(rmsNormGammaType)); // [1, 512] bf16
    rmsnormGammaCkvLocal_ = rmsnormGammaCkvBuffer_.Get<rmsNormGammaType>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value){
        if (enableSmoothScalesCq_) {
            pipe_->InitBuffer(smoothScaleCqBuffer_, baseParams_->headSizeCq * sizeof(float)); // [1, 1536]
            smoothScaleCqLocal_ = smoothScaleCqBuffer_.Get<float>();
        }
    }

    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        pipe_->InitBuffer(quantScaleCkrBuffer_, baseParams_->dimHeadRope * sizeof(float)); // [1, 64]
        quantScaleCkrLocal_ = quantScaleCkrBuffer_.Get<float>();
    }

    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            pipe_->InitBuffer(quantScaleCkvBuffer_, ALIGN_BLOCK_SIZE);
        } else {
            pipe_->InitBuffer(quantScaleCkvBuffer_, baseParams_->headSizeCkv * sizeof(float)); // [1, 512]
        }
        quantScaleCkvLocal_ = quantScaleCkvBuffer_.Get<float>();
    }

    // 预留brcb的空间
    pipe_->InitBuffer(dequantTool_.deQuantScaleCqBuffer_, (baseParams_->stepBatchSize + 7) * ALIGN_BLOCK_SIZE);
    dequantTool_.deQuantScaleCqLocal_ = dequantTool_.deQuantScaleCqBuffer_.template Get<float>();

    if constexpr (MLAPT::enableDequantOpt) {
        // 在ropeQr进行切N处理后，会复用shareBuffer的内存，不需要额外申请
        // 开启开关后会按照head切分rope qr，此时需要加载一半batchsize数量的sin和cos值
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * ((baseParams_->stepBatchSize + 1) >> 1));
    } else {
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * curVecTokenMax_); // [2, 64] float
    }

    uint64_t usedAddr;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)];
        usedAddr = reinterpret_cast<uint64_t>(sinLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)].GetPhyAddr());
    } else {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
        usedAddr =  reinterpret_cast<uint64_t>(sinLocal_[baseParams_->dimHeadRope * curVecTokenMax_].GetPhyAddr());
    }

    // 由于shareBuffer属于各个vector操作临时申请内存的区域内存使用不固定，建议shareBuffer始终放在最后，防止写入shareBuffer越界导致前面固定申请的UB内存被踩。
    pipe_->InitBuffer(shareBuffer_, MAX_UB_SIZE - usedAddr); // 除 sincos 外的 sharebuffer 共享，不会同时使用
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CubeBufferInit() {
    // cube相关Buffer初始化
    pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
    pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);

    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    aL1Tensor_ = aBufL1_.Get<mmInputType>();
    bL1Tensor_ = bBufL1_.Get<mmInputType>();
    bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
    bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());

    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128K

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    bufParam_.aL0BufAddr = aBufL0_.GetBufferAddr(aBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.bL0BufAddr = bBufL0_.GetBufferAddr(bBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.cL0BufAddr = cBufL0_.GetBufferAddr(cBufL0_.Get<float>().GetBufferHandle());
}

/*
 * workspace管理
 * 1. 常驻：dequantTool_.deQuantScaleCqGm_ stepBs * 32 Byte
 * 2. 中间结果：
 *      tokenXGm_──────>mmCkvKrResGm_
 *          |           [stepBS, HCkv + Dr]
 *          |           (bf16 | int32)
 *          └─────────>mmCqResGm_──────>rmsNormCqResGm_──────>mmQcQrResGm_──────>mmQcQrResDequantGm_──────>mmQnResGm_
 *                     [stepBS, HCq]    [stepBS, HCq]     [stepBS, N1, D + Dr]   [stepBS, N1, D]           [stepBS, N1, HCkv]
 *                     (bf16 | int32)   (bf16 | int8)     (bf16 | int32)         (bf16)                    (bf16)
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::WorkspaceInit(__gm__ uint8_t *workspace) {
    int64_t workspaceOffset = 0;
    if constexpr (MLAPT::enableGroupComputeOpt || MLAPT::enableDequantOpt) {
        dequantTool_.deQuantScaleCqGm_.SetGlobalBuffer((__gm__ float *)(workspace + workspaceOffset));
        workspaceOffset += baseParams_->stepBatchSize * ALIGN_BLOCK_SIZE;
    }

    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType *)(workspace + workspaceOffset)); // aicOffset.cqResOffset
    if constexpr (!std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
        workspaceOffset +=  static_cast<int64_t>(baseParams_->stepBatchSize) *
                            static_cast<int64_t>(baseParams_->headSizeCq) *
                            baseParams_->mm1BlockNum *
                            sizeof(mmCqOutputType);
    }

    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType *)(workspace + workspaceOffset));
    workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                       static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                       baseParams_->mm2BlockNum *
                       sizeof(mmCkvKrOutputType);

    rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType *)(workspace + workspaceOffset)); // rmsNormCqResGmOffset
    workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                       static_cast<int64_t>(baseParams_->headSizeCq) *
                       baseParams_->mm1BlockNum *
                       sizeof(rmsNormCqOutputType);

    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType *)(workspace + workspaceOffset));  // aicOffset.qcQrResOffset
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeQc + baseParams_->headSizeQr) *
                           baseParams_->mm3BlockNum *
                           sizeof(mmQcQrOutputType);
    }

    mmQcQrResDequantGm_.SetGlobalBuffer((__gm__ mmQnInputType *)(workspace + workspaceOffset));  // qcOffset mmQnPreDequantResOffset
    if constexpr (std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) {
        workspaceOffset +=  static_cast<int64_t>(baseParams_->stepBatchSize) *
                            static_cast<int64_t>(baseParams_->numHeadSize) *
                            static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
                            baseParams_->mm3BlockNum *
                            sizeof(mmQnInputType);
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)(workspace + workspaceOffset));   // aicOffset.qnResOffset
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::UpdateStepBatchParams(int64_t curMSize) {
    mmCqParam_.m = curMSize;
    mmCkvKrParam_.m = curMSize;
    mmQcQrParam_.m = curMSize;
    mmQnParam_.m = curMSize;
    curVectorBlockNum_ = curMSize;
}

/*
 * MlaPrologV3算子计算&CV流水同步流程
 *                    ┌───────────────── token_x ─────────────────┐
 *                    |                                           ▼
 *                    |                                      MatmulCkvKr
 *                    ▼                                           | wait mm CkvKr(0x1)
 *                MatmulCq                                        ▼
 *                    | wait mm Cq(0x1)                  ┌─────────────────┐
 *                    ▼                                  ▼                 ▼
 *               RmsNorm(Cq)                         RmsNorm(Ckv)       Rope(Kr)
 *                    | wait rmsNorm cq(0x1)             |                 |
 *                    ▼                                  ▼                 ▼
 *        ┌───────MatmulQcQr───────┐                 Scatter(Ckv)      Scatter(Kr)
 *        | wait mm Qc(0x1)        |                     |                 |
 *        ▼                        |                     ▼                 ▼
 *    DequantQc                    |                 kv_cache_out      kr_cache_out
 *        | wait dequant qc(0x1)   | wait mm Qr(0x2)
 *        ▼                        ▼
 *     MatmulQn                 Rope(Qr)
 *        | wait mm Qn(0x1)        |
 *   DynamicQuantQn──┐             |
 *        |          ▼             |
 *        ▼   dequant_scale_out    ▼
 *    query_out               query_rope_out
 * 注：仅为表明基本计算与CV同步流程，仅包含了影响CV同步的量化分支，其余量化分支应参考设计文档。
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::Process() {
    int64_t numHeadOffset = 0;
    int64_t mmQnLoops = baseParams_->mm4SingleCoreBatch;

    if (cubeBlockIdx_ < baseParams_->mSubCoreNum) {
        mSubSize_ = baseParams_->mSubSize;
        mOffsetStart_ = mSubSize_ * cubeBlockIdx_;
    } else {
        mSubSize_ = baseParams_->mSubSize - 1;
        mOffsetStart_ = baseParams_->mSubSize * baseParams_->mSubCoreNum +
                   mSubSize_ * (cubeBlockIdx_ - baseParams_->mSubCoreNum);
    }

    // AIC的offset参数
    AicOffset aicOffset;
    // ComputeAicOffset(aicOffset, mOffsetStart_);

    // // AIV的offset参数
    AivOffset aivOffset;
    // ComputeAivOffset(aivOffset, 0);
    uint32_t mloop = 0;
    // 需要考虑BS合轴的尾块情况
    int64_t maxMOffset = mOffsetStart_ + mSubSize_;
    for (int64_t mOffset = mOffsetStart_; mOffset < maxMOffset; mOffset += baseParams_->stepBatchSize) {
        int64_t curMSize = (maxMOffset - mOffset) < baseParams_->stepBatchSize ? 
                           (maxMOffset - mOffset) : baseParams_->stepBatchSize;

        // printf("mloop:%d, blockIdx_:%d, mOffset:%d, mOffsetStart_:%d, curMSize:%d, maxMOffset:%d \n",
        //         mloop, blockIdx_, mOffset, mOffsetStart_, curMSize, maxMOffset);

        if ASCEND_IS_AIC {
            AicProcess(aicOffset, mOffset, mmQnLoops);
        }
        if ASCEND_IS_AIV {
            AivProcess(aivOffset, mOffset, curMSize, numHeadOffset, mmQnLoops);
        }
        mloop++;
        if (mloop > 1) {
            // break;
        }
    }

    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::AicProcess(AicOffset &aicOffset, int64_t mOffset, int64_t mmQnLoops) {
    if(cubeBlockIdx_ > 2) {
        // return;
    }
    int64_t tokenXOffset = mOffset * baseParams_->headSizeX;
    aicOffset.weightDqOffset = 0;
    aicOffset.cqResOffset = baseParams_->stepBatchSize * 
                            baseParams_->headSizeCq *
                            cubeBlockIdx_;
    // printf("mmCq: tokenXOffset:%d, weightDqOffset:%d, cqResOffset:%d\n",
            // tokenXOffset, aicOffset.weightDqOffset, aicOffset.cqResOffset);

    // MatmulCq ──> RmsNorm(Cq)
    // [32, 7168] * [7168, 1536] = [32, 1536]
    MatmulSplitM<mmInputType, mmCqOutputType>(  mmCqResGm_[aicOffset.cqResOffset], 
                                                tokenXGm_[tokenXOffset], 
                                                weightDqGm_[aicOffset.weightDqOffset],
                                                mmCqParam_, 
                                                UsedBlockParams{0, baseParams_->mm1BlockNum});

    // DumpTensor(mmCqResGm_[aicOffset.cqResOffset], 1111, baseParams_->stepBatchSize * baseParams_->headSizeCq);

    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);
    // printf("CrossCoreSetFlag FINISH_MM_CQ\n");

    aicOffset.weightDkvKrOffset = 0;
    aicOffset.ckvKrResOffset = baseParams_->stepBatchSize *
                               (baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                               cubeBlockIdx_;
    // printf("mmCkvKr: tokenXOffset:%d, weightDqOffset:%d, cqResOffset:%d\n",
    //         tokenXOffset, aicOffset.weightDkvKrOffset, aicOffset.ckvKrResOffset);
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    // [32, 7168] * [7168, 512+64] = [32, 576]
    MatmulSplitM<mmInputType, mmCkvKrOutputType, true>( mmCkvKrResGm_[aicOffset.ckvKrResOffset],
                                                        tokenXGm_[tokenXOffset],
                                                        weightDkvKrGm_[aicOffset.weightDkvKrOffset],
                                                        mmCkvKrParam_,
                                                        UsedBlockParams{0, baseParams_->mm2BlockNum});

    // DumpTensor(mmCkvKrResGm_[aicOffset.ckvKrResOffset], 1222, baseParams_->stepBatchSize * (baseParams_->headSizeCkv + baseParams_->dimHeadRope));

    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
    // printf("CrossCoreWaitFlag FINISH_VEC_RMSNORM_CQ\n");
    CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);

    // MatmulAndSyncQcQr(aicOffset);
    // int64_t rmsNormCqResGmOffset = baseParams_->stepBatchSize *
    //                                baseParams_->headSizeCq *
    //                                cubeBlockIdx_;
    aicOffset.weightUqQrOffset = 0;
    aicOffset.qcQrResOffset =   baseParams_->stepBatchSize *
                                (baseParams_->headSizeQc + baseParams_->headSizeQr) *
                                cubeBlockIdx_;
    MatmulQcQrSyncDequant(aicOffset.weightUqQrOffset, aicOffset.qcQrResOffset);
    // MatmulSplitM<rmsNormCqOutputType, mmQcQrOutputType>( mmQcQrResGm_[aicOffset.qcQrResOffset],
    //                                                      rmsNormCqResGm_[rmsNormCqResGmOffset],
    //                                                      weightUqQrGm_[aicOffset.weightUqQrOffset],
    //                                                      mmQcQrParam_,
    //                                                      UsedBlockParams{0, baseParams_->mm3BlockNum});

    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]
    if constexpr (!needQnDynamicQuant) {
        // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
        aicOffset.qnResOffset = mOffset *
                                baseParams_->headSizeCkv *
                                baseParams_->numHeadSize *
                                baseParams_->numHeadSize;
    } else {
        aicOffset.qnResOffset = baseParams_->stepBatchSize *
                                baseParams_->numHeadSize *
                                baseParams_->headSizeCkv *
                                cubeBlockIdx_;
    }

    aicOffset.qcOffset = baseParams_->stepBatchSize *
                         baseParams_->numHeadSize *
                         baseParams_->dimHeadSizeQc *
                         cubeBlockIdx_;
    aicOffset.weightUkOffset = 0;

    // PreloadQnAndSync(aicOffset, mmQnLoops);
    MatmulQnSyncDynamicQuantAndMulQr(aicOffset.qcOffset, aicOffset.weightUkOffset, aicOffset.qnResOffset, mmQnLoops);
    

    if constexpr (!needQnDynamicQuant) {
        // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
        aicOffset.qnResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->headSizeCkv) *
                    static_cast<int64_t>(baseParams_->numHeadSize);
    }

    return;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::AivProcess(AivOffset &aivOffset, int64_t mOffset, int64_t curMSize, int64_t numHeadOffset, int64_t mmQnLoops) {
    if(cubeBlockIdx_ > 2) {
        // return;
    }
    if (mOffset == mOffsetStart_) {
        // 只需要搬运一次
        CopyGlobalParams();
    }

    uint64_t isBackVec = blockIdx_ % 2;
    curStepVecBackToken_ = curMSize / 2;
    curStepVecFrontListNum_ = curMSize % 2;

    aivOffset.curVecToken = curStepVecBackToken_ + isBackVec * curStepVecFrontListNum_;   // 当前核中每个vec处理多少个token
    aivOffset.curBlockTokenOffset = isBackVec * curStepVecBackToken_;                     // 当前核中每个vec在这个m中的偏移

    int64_t tokenIndex = mOffset + aivOffset.curBlockTokenOffset;
    int64_t curTokenStart = mOffset + aivOffset.curBlockTokenOffset;
    int64_t curTokenEnd = mOffset + aivOffset.curBlockTokenOffset + aivOffset.curVecToken;
    // printf("blockIdx_:%d, curTokenStart:%d, curVecToken:%d, mOffset:%d, curMSize:%d\n", blockIdx_, curTokenStart, aivOffset.curVecToken, mOffset, curMSize);
    for (int64_t curTokenIndex = curTokenStart; curTokenIndex < curTokenEnd; curTokenIndex++) {
        CopyInSinCos(curTokenIndex, aivOffset.curVecToken, mOffset, curMSize);
    }

    CrossCoreWaitFlag(FINISH_MM_CQ);
    // printf("CrossCoreWaitFlag FINISH_MM_CQ \n");
    // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);

    aivOffset.rmsNormCqOffset = baseParams_->stepBatchSize * baseParams_->headSizeCq * cubeBlockIdx_
                                + baseParams_->headSizeCq * aivOffset.curBlockTokenOffset;
    // printf("RmsNormCq: curTokenStart:%d, rmsNormCqOffset:%d, curVecToken:%d, curBlockTokenOffset:%d\n", curTokenStart, aivOffset.rmsNormCqOffset, aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
    RmsNormCq(curTokenStart, aivOffset.rmsNormCqOffset, aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
    
    // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
    // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
    // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);

    // 聚合全部scale结果
    // if constexpr (MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) {
    //     SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    //     WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    //     DataCopy(dequantTool_.deQuantScaleCqLocal_, dequantTool_.deQuantScaleCqGm_, ALIGN_BLOCK_SIZE / sizeof(float) * baseParams_->stepBatchSize);
    //     SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    //     WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // }
    // DumpTensor(dequantTool_.deQuantScaleCqGm_[aivOffset.curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], 233, 128);
    // DumpTensor(dequantTool_.deQuantScaleCqLocal_, 1234, 128);
    // printf("CrossCoreSetFlag FINISH_VEC_RMSNORM_CQ\n");
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
    CrossCoreWaitFlag(FINISH_MM_CKVKR);

    // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    // aivOffset.rmsNormCkvOffset = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * aivOffset.curBlockTokenOffset;  // (512 + 64) * idx

    aivOffset.rmsNormCkvOffset = baseParams_->stepBatchSize *
                               (baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                               cubeBlockIdx_ +
                               (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * aivOffset.curBlockTokenOffset;

    aivOffset.ropeKrOffset = baseParams_->headSizeCkv + aivOffset.rmsNormCkvOffset;  // 512 + (512 + 64) * idx
    // printf("RmsNormRopeScatterCkvKr: rmsNormCkvOffset:%d, ropeKrOffset:%d, curVecToken:%d \n", aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken);
    RmsNormRopeScatterCkvKr(tokenIndex, aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken);


    if constexpr (MLAPT::enableDequantOpt) {
        aivOffset.mmQnPreDequantOffset = ((baseParams_->stepBatchSize) *
                                            (baseParams_->headSizeQc + baseParams_->headSizeQr) *
                                            cubeBlockIdx_) +
                                            (baseParams_->headSizeQc + baseParams_->headSizeQr) *
                                            aivOffset.curBlockTokenOffset;

        aivOffset.mmQnPreDequantResOffset = baseParams_->stepBatchSize *
                                            baseParams_->numHeadSize *
                                            baseParams_->dimHeadSizeQc *
                                            cubeBlockIdx_ +
                                            baseParams_->dimHeadSizeQc *
                                            baseParams_->numHeadSize *
                                            aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrOffset = (baseParams_->stepBatchSize *
                                  (baseParams_->headSizeQc + baseParams_->headSizeQr) * cubeBlockIdx_) + 
                                  baseParams_->dimHeadSizeQc + 
                                  static_cast<int64_t>(baseParams_->numHeadSize) *
                                  static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                  aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrResOffset = static_cast<int64_t>(baseParams_->headSizeQr) * (aivOffset.curBlockTokenOffset + mOffset); //32 * 64 * idx;    // 按BS合轴切分step

        QcQrSplit( aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset,
                   curMSize,
                    aivOffset.mmQnPreDequantOffset,
                    aivOffset.mmQnPreDequantResOffset,
                    aivOffset.ropeQrOffset,
                    aivOffset.ropeQrResOffset);
        // DequantAndRopeSplitNSyncMMQcQr(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
        //         aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset);
    } else if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        // CrossCoreWaitFlag(FINISH_MM_QCQR);
        // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        // DequantQc(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset, aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
        // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        // CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
        // RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
    } else {
        // CrossCoreWaitFlag(FINISH_MM_QCQR);
        // WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        // RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken, aivOffset.curBlockTokenOffset);
    }

    // if constexpr (needQnDynamicQuant && !(MLAPT::enableDequantOpt)) {
    //     // 非切N场景，需要等待全部rope的结果做完并搬到GM
    //     WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    // }
    if constexpr (needQnDynamicQuant) {
        DynamicQuantQnAndMulQrSyncMMQn(mOffset, curMSize, aivOffset.curBlockTokenOffset, numHeadOffset, mmQnLoops);
    }
    aivOffset.ropeQrResOffset += static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->headSizeQr);

    return;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset) {
    aicOffset.weightDqOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.cqResOffset;  // 7168 * 64 * idx
    aicOffset.cqResOffset = baseParams_->stepBatchSize * blockIdx_;  // 64 * idx

    aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_;  //  (512 + 64) / 9 * idx  = 64 * idx
    aicOffset.weightDkvKrOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.ckvKrResOffset;  // 7168 * (512 + 64) / 9 * idx = 7168 * 64 * idx

    if constexpr (MLAPT::enableGroupComputeOpt) {
        aicOffset.weightUqOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                               (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * blockIdx_;  // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.weightQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                               (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * 2 * (blockIdx_ - QC_CORE_NUM) +
                               static_cast<int64_t>(baseParams_->headSizeCq) * baseParams_->dimHeadSizeQc;  
        aicOffset.qCResOffset = baseParams_->dimHeadSizeQc * blockIdx_;
        aicOffset.qRResOffset = baseParams_->stepBatchSize * baseParams_->dimHeadSizeQc * QC_CORE_NUM +
                      baseParams_->dimHeadRope * 2 * (blockIdx_ - QC_CORE_NUM); // m=baseParams_->stepBatchSize
    } else {
        aicOffset.qcQrResOffset = baseParams_->mm3SingleCoreN * blockIdx_;  // (32 * 128 + 32 * 64) / 24 * idx = 256 * idx
        aicOffset.weightUqQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) * aicOffset.qcQrResOffset;  // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
    }

    if (cubeBlockIdx_ < baseParams_->mm4BlockNum) {
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) * numHeadOffset; // 128 * idx
        } else {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * numHeadOffset; // (128 + 64) * idx
        }
        aicOffset.weightUkOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset;   // (128 * 512) * idx
        aicOffset.qnResOffset = static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset; // BS, N, Dq
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset) {
    curStepVecBackToken_ = curVectorBlockNum_ / static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontListNum_ = curVectorBlockNum_ % static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ == 0 ? curStepVecBackToken_ : curStepVecBackToken_ + 1;

    aivOffset.curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    aivOffset.curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ? blockIdx_ * aivOffset.curVecToken : blockIdx_ * aivOffset.curVecToken + curStepVecFrontListNum_;
    aivOffset.rmsNormCqOffset = baseParams_->headSizeCq * aivOffset.curBlockTokenOffset;  //  1536 * idx
    aivOffset.rmsNormCkvOffset = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * aivOffset.curBlockTokenOffset;  // (512 + 64) * idx
    aivOffset.ropeKrOffset = baseParams_->headSizeCkv + aivOffset.rmsNormCkvOffset;  // 512 + (512 + 64) * idx
    if constexpr (!MLAPT::enableDequantOpt) {
        aivOffset.mmQnPreDequantOffset = (baseParams_->headSizeQc + baseParams_->headSizeQr) * aivOffset.curBlockTokenOffset;
        aivOffset.mmQnPreDequantResOffset = baseParams_->headSizeQc * aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrOffset = baseParams_->dimHeadSizeQc + static_cast<int64_t>(baseParams_->numHeadSize) *
                        static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * aivOffset.curBlockTokenOffset; // 128 + 32 * (128 + 64) * idx
        aivOffset.ropeQrResOffset = static_cast<int64_t>(baseParams_->headSizeQr) * (aivOffset.curBlockTokenOffset + batchOffset); //32 * 64 * idx;    // 按BS合轴切分step
    }

    // 以下的Offset均是与batchOffset无关的，仅初始化一次即可。
    if (batchOffset == 0) {
        if constexpr (MLAPT::enableDequantOpt) {
            aivOffset.mmQnPreDequantOffset = baseParams_->mm3SingleCoreN * cubeBlockIdx_; //qcQrResOffset
            aivOffset.mmQnPreDequantResOffset = baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc +
                baseParams_->dimHeadRope) * baseParams_->dimHeadSizeQc * cubeBlockIdx_;
            aivOffset.ropeQrOffset = aivOffset.mmQnPreDequantOffset + baseParams_->dimHeadSizeQc;
            aivOffset.ropeQrResOffset = (baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc +
                baseParams_->dimHeadRope)) * baseParams_->dimHeadRope * cubeBlockIdx_;
        }

        if constexpr (MLAPT::enableGroupComputeOpt) {
            aivOffset.qcScaleOffsetSplitN = (baseParams_->headSizeQc + baseParams_->headSizeQr) / QC_CORE_NUM * (blockIdx_ >> 1);
            aivOffset.mmQnPreDequantResOffset = (baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ >> 1);
            aivOffset.mmQnPreDequantOffset = aivOffset.mmQnPreDequantResOffset;
            aivOffset.ropeQrResSplitNOffset = (blockIdx_ - QC_CORE_NUM * 2) * baseParams_->dimHeadRope;
            aivOffset.ropeQrSplitNOffset = baseParams_->headSizeQc + aivOffset.ropeQrResSplitNOffset;
        }
    }
}

// Mlaprolog 支持int8进int32出, 参考MatmulQcQr
template<typename MLAPT>
template<typename T, typename O, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulSplitN(
    const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,  
    const MMParams &mmPara, const UsedBlockParams &mmBlockParams) {
    if constexpr (needCheckEmptyTensor) {
        if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
            return;
        }
    }
    if (blockIdx_ < mmBlockParams.blockStartIdx || blockIdx_ >= mmBlockParams.blockEndIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t mSize = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O, isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_); 
            return;
        }
    }
    uint32_t nInput = mmPara.n;
    uint32_t nL1SplitSize = mmPara.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        MatmulSplitK<T, O>(tensorResGm, tensorAGm, tensorBGm,
            mmPara, bufParam_, nL1 * nL1SplitSize, subNL1SplitSize);
    }
}

// Mlaprolog 支持int8进int32出, 参考MatmulQcQr
template<typename MLAPT>
template<typename T, typename O, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulSplitM(
    const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,  
    const MMParams &mmPara, const UsedBlockParams &mmBlockParams) {
    if constexpr (needCheckEmptyTensor) {
        if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
            return;
        }
    }
    if (blockIdx_ < mmBlockParams.blockStartIdx || blockIdx_ >= mmBlockParams.blockEndIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t mSize = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O, isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_); 
            return;
        }
    }

    uint32_t nInput = mmPara.n;
    uint32_t nL1SplitSize = mmPara.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    // printf("MatmulSplitM: nL1SplitSize:%d, subNL1SplitSize:%d, nInput:%d, nL1loops:%d\n", nL1SplitSize, subNL1SplitSize, nInput, nL1loops);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        // printf("nL1:%d, nL1Offset:%d, subNL1SplitSize:%d\n", nL1, nL1 * nL1SplitSize, subNL1SplitSize);
        MatmulSplitK<T, O>(tensorResGm, tensorAGm, tensorBGm,
            mmPara, bufParam_, nL1 * nL1SplitSize, subNL1SplitSize);
    }
}


template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulQcQrSyncDequant(int64_t weightUqQrOffset, int64_t qcQrResOffset) {

    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    // [32, 1536] * [1536, 32*(128+64)] = [32, 32*192]

    uint32_t nInput = mmQcQrParam_.n;
    uint32_t nL1SplitSize = mmQcQrParam_.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    int64_t rmsNormCqResGmOffset = baseParams_->stepBatchSize *
                                   baseParams_->headSizeCq *
                                   cubeBlockIdx_;

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - nL1 * nL1SplitSize;
        }

        MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType>(
            mmQcQrResGm_[qcQrResOffset],
            rmsNormCqResGm_[rmsNormCqResGmOffset],
            weightUqQrGm_[weightUqQrOffset],
            mmQcQrParam_,
            bufParam_,
            nL1 * nL1SplitSize,
            subNL1SplitSize);
        // break;
        if constexpr (MLAPT::enableDequantOpt) {
            // CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);
            // printf("CrossCoreSetFlag FINISH_MM_QCQR_SPLIT_N %d\n", nL1);
        }
        // DumpTensor(rmsNormCqResGm_[rmsNormCqResGmOffset], 3301, baseParams_->headSizeCq);
        // DumpTensor(weightUqQrGm_[weightUqQrOffset], 3302, baseParams_->headSizeCq);
        // DumpTensor(mmQcQrResGm_[qcQrResOffset], 3303, baseParams_->headSizeCq);
    }
    CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);  // FINISH_MM_QCQR_SPLIT_N
    // CrossCoreSetFlag<0x2, PIPE_FIX>(0x2);  // FINISH_MM_QCQR_SPLIT_N
    // DumpTensor(rmsNormCqResGm_[rmsNormCqResGmOffset], 3331, 1024);
    // DumpTensor(weightUqQrGm_[weightUqQrOffset], 3332, 1024);
    // DumpTensor(mmQcQrResGm_[qcQrResOffset], 3333, 8192);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops) {
    MatmulQnWeightPreload(aicOffset.weightUkOffset, mmQnLoops);
    if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QR);
    } else if constexpr (MLAPT::enableDequantOpt) {
        // enableDequantOpt分支的CV同步由更细粒度的子函数控制
        return;
    } else {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR);
    }

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC);
    } else {
        WaitAllCore<SYNC_MODE_ALL_CUBE, PIPE_FIX>(FINISH_MM_ALL);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulQnWeightPreload(int64_t weightUkOffset, int64_t subLoopTimes) {
    if constexpr (MLAPT::enableGroupComputeOpt) {
        if (blockIdx_ >= QC_CORE_NUM) {
            return;
        }
    } else {
        if (blockIdx_ >= baseParams_->mm4BlockNum) {
            return;
        }
    }
    int64_t weightOffset = weightUkOffset;
    for (int32_t i = 0; i < subLoopTimes; ++i) {
        if (i < 2) { // preload double buffer
            LoadL1B<mmQnInputType, DataFormat::ND, false>(weightUkGm_[weightOffset], 
                mmQnParam_.n, mmQnParam_.k, mmQnParam_.k, bufParam_);
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
            bufParam_.bL1BufIter++;
            weightOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv);
        }
    }
    if (subLoopTimes == 1) {
        bufParam_.bL1BufIter--;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                            int64_t qnResOffset, int64_t subLoopTimes) {
    if constexpr (MLAPT::enableGroupComputeOpt) {
        if (blockIdx_ >= QC_CORE_NUM) {
            return;
        }
    } else {
        if (blockIdx_ >= baseParams_->mm4BlockNum) {
            return;
        }
    }
    int64_t resdumpt = qnResOffset;
    // printf("subLoopTimes:%d\n", subLoopTimes);
    CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N);
    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]
    for (int64_t i = 0; i < subLoopTimes; i++) {
        if constexpr (MLAPT::enableDequantOpt) {
            // printf("CrossCoreWaitFlag FINISH_VEC_DEQUANT_QC_SPLIT_N %d\n", i);
            // CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N);
        }

        MatmulFullLoad<mmQnInputType, mmQnOutputType, false, true>(
                mmQnResGm_[qnResOffset],
                mmQcQrResDequantGm_[qcOffset],
                weightUkGm_[weightUkOffset],
                mmQnParam_,
                bufParam_);

        // DumpTensor(mmQnResGm_[qnResOffset], 5551, baseParams_->headSizeCkv);
        // DumpTensor(mmQcQrResDequantGm_[qcOffset], 5552, baseParams_->dimHeadSizeQc);
        // DumpTensor(weightUkGm_[weightUkOffset], 5553, baseParams_->dimHeadSizeQc);
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        // } else {
        //     qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        }
        weightUkOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv);
        qnResOffset += baseParams_->headSizeCkv;

        if constexpr (needQnDynamicQuant) {
            // printf("CrossCoreSetFlag FINISH_MM_QN_SPLIT_N %d\n", i);
            // CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
        }
    }
    // DumpTensor(mmQnResGm_[resdumpt], 5555, 128 * baseParams_->dimHeadSizeQc);
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CopyInSinCos(int64_t tokenIndex, int64_t curVecToken, int64_t batchOffset, int64_t curMSize) {
    // printf("blockIdx_:%d, tokenIndex:%d, curVecToken:%d, batchOffset:%d, curMSize:%d\n", blockIdx_, tokenIndex, curVecToken, batchOffset, curMSize);

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt) {
        // 如果是切N场景，mm3的每个C核都会做rope
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
        // >> 1 是将curStepBatchSize分到每个vec核上；
        uint32_t subBlockIdx_ = blockIdx_ & 1u;
        int64_t offset = (curMSize >> 1) * subBlockIdx_ + batchOffset;
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, offset, (curMSize + 1) >> 1,
                    shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    } else {
        if (blockIdx_ >= curVectorBlockNum_) {
            return;
        }
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken,
                    shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CopyGlobalParams() {
    // dequantScaleWDq
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDqLocal_, dequantScaleWDqGm_, dequantCopyParams, dequantPadParams); 
    }
    
    // rmsnormGammaCq
    DataCopy(rmsnormGammaCqLocal_, rmsnormGammaCqGm_, baseParams_->headSizeCq);

    // rmsnormGammaCkv
    DataCopy(rmsnormGammaCkvLocal_, rmsnormGammaCkvGm_, baseParams_->headSizeCkv);

    // smoothScaleCq
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value){
        if (enableSmoothScalesCq_) {
            DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
            DataCopyPad(smoothScaleCqLocal_, smoothScaleCqGm_, smoothCopyParams, smoothPadParams); 
        }
    }

    
    // dequantScaleWDkvKr
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{1, static_cast<uint32_t>((baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float)), 
                                            0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDkvKrLocal_, dequantScaleWDkvkrGm_, dequantCopyParams, dequantPadParams); 
    }
    

    // quantScaleCkv
    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            DataCopyExtParams quantCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
            DataCopyPad(quantScaleCkvLocal_, quantScaleCkvGm_, quantCopyParams, quantPadParams); 
        } else {
            DataCopy(quantScaleCkvLocal_, quantScaleCkvGm_, baseParams_->headSizeCkv);
        }
    }
    

    // quantScaleCkr
    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        DataCopyExtParams quantCopyParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(float)), 
                                            0, 0, 0};
        DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
        DataCopyPad(quantScaleCkrLocal_, quantScaleCkrGm_, quantCopyParams, quantPadParams); 
    }
}


/**
 * @brief RmsNormCq流程，融合了dynamicquant
          内部所需空间约为 curVecToken(128) * 8*4 + 8*4 + (4*vectorRow_*baseParams_->headSizeCq + 8)*4 = 28.0625K
 */
template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t curVecToken,
                                                             int64_t curBlockTokenOffset) {
    int64_t rmsNormCqResOffset = rmsNormCqOffset;
    LocalTensor<rmsNormCqOutputType> outputLocal = shareBuffer_.Get<rmsNormCqOutputType>();
    LocalTensor<float> dequantScaleQcQr = outputLocal[baseParams_->headSizeCq].template ReinterpretCast<float>();
    LocalTensor<float> dequantScaleXLocal = dequantScaleQcQr[curVecToken * FP32_BLOCK_ELEMENT_NUM];
    LocalTensor<uint8_t> shareTmpUb = dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<uint8_t>();

    // DumpTensor(mmCqResGm_[rmsNormCqOffset], 1999, 1024);
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
        // printf("in RmsNormCq: blockIdx_:%d, curVecTokenIdx:%d, tokenIndex:%d, rmsNormCqOffset:%d, curVecToken:%d \n", blockIdx_, curVecTokenIdx,  tokenIndex, rmsNormCqOffset, curVecToken);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish

        // dequantScaleXGm_  [BS , 1] 每个每个token对应一个系数，此处扩展为一个DataBlock
        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        uint64_t scaleOffset = curVecTokenIdx * FP32_BLOCK_ELEMENT_NUM;
        RmsNormParam rmsNormParams ={baseParams_->reciprocalCq, // reciprocal
            baseParams_->epsilonCq,  // epsilon
            (uint32_t)vectorRow_, //row
            baseParams_->headSizeCq //col
            };

        if constexpr (std::is_same<rmsNormCqOutputType, int8_t>::value) {
            RmsNormDynamicQuant<mmCqOutputType, rmsNormGammaType, float, rmsNormComputType>(
                                outputLocal,
                                dequantScaleQcQr[scaleOffset],
                                mmCqResGm_[rmsNormCqOffset],
                                rmsnormGammaCqLocal_,
                                smoothScaleCqLocal_,
                                dequantScaleWDqLocal_,
                                dequantScaleXLocal,
                                shareTmpUb,
                                rmsNormParams,
                                enableSmoothScalesCq_);
        } else {
            RmsNormNormal<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType>(
                outputLocal, mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_, dequantScaleWDqLocal_,
                dequantScaleXLocal, shareTmpUb, rmsNormParams);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqOffset], outputLocal, baseParams_->headSizeCq);

        // DumpTensor(dequantScaleQcQr[scaleOffset], 2220, baseParams_->headSizeCq);
        // DumpTensor(outputLocal, 2221, baseParams_->headSizeCq);
        // DumpTensor(mmCqResGm_[rmsNormCqOffset], 1999, baseParams_->headSizeCq);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        rmsNormCqOffset += baseParams_->headSizeCq;
        tokenIndex++;
    }
    // DumpTensor(rmsNormCqResGm_[rmsNormCqResOffset], 2222, curVecToken * baseParams_->headSizeCq);

    Brcb(dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], dequantScaleQcQr, curVecToken, {1, 1});
    // DumpTensor(dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], 231, 128);

    if constexpr (MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dequantTool_.deQuantScaleCqGm_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 FP32_BLOCK_ELEMENT_NUM * curVecToken);
    }
    // DumpTensor(dequantTool_.deQuantScaleCqGm_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], 232, 128);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormAndScatterCkv(LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                                            LocalTensor<ropeComputType>& cosLocalCkvKr, LocalTensor<ropeComputType>& sinLocalCkvKr,
                                                            CkvkrParams rmsNormAndScatterCkvParams) {
    // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv)
    LocalTensor<kvCacheType> outputLocal = shareTmpUb.ReinterpretCast<kvCacheType>();
    LocalTensor<uint8_t> rmsNormShareTmpUb = shareTmpUb[baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();

    RmsNormParam rmsNormParams ={
        baseParams_->reciprocalCkv, // reciprocal
        baseParams_->epsilonCkv,  // epsilon
        (uint32_t)vectorRow_, //row
        baseParams_->headSizeCkv //col
    };
 
    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        // row = vectorRow_ = 1     col = baseParams_->headSizeCkv
        LocalTensor<float> tmpOut = rmsNormShareTmpUb.ReinterpretCast<float>();
        LocalTensor<uint8_t> sharedBuf = shareTmpUb.ReinterpretCast<uint8_t>()[vectorRow_ * baseParams_->headSizeCkv * sizeof(float)];
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, float>(
            tmpOut, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
            dequantScaleXLocal, sharedBuf, rmsNormParams);
        
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        Rectangle rectangleParams {
            (uint32_t)vectorRow_,    // row
            (uint32_t)baseParams_->headSizeCkv,// col
            (uint32_t)baseParams_->headSizeCkv // columnStride
        }; 
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            QuantPerTensor(outputLocal, tmpOut, quantScaleCkvLocal_, sharedBuf, rectangleParams);
            pipe_barrier(PIPE_V);
        } else {
            QuantPerChannel(outputLocal, tmpOut, quantScaleCkvLocal_, sharedBuf, rectangleParams);
        }

    } else {
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType>(
            outputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
            dequantScaleXLocal, rmsNormShareTmpUb, rmsNormParams);
    }

    // DumpTensor(outputLocal, 2322, baseParams_->headSizeCkv);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // Scatter(Ckv)
    // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
    int64_t paTokenIndex = cacheIndexGm_(rmsNormAndScatterCkvParams.tokenIndex);
    ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(kvCacheGm_, outputLocal,
        ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->headSizeCkv});
    
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RopeAndScatterKr(
                                                            LocalTensor<float>& dequantScaleXLocal, LocalTensor<uint8_t>& shareTmpUb,
                                                            LocalTensor<ropeComputType>& cosLocalCkvKr, LocalTensor<ropeComputType>& sinLocalCkvKr,
                                                            CkvkrParams ropeAndScatterKrParams) {   
    // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr) 
    LocalTensor<krCacheType> outputKrLocal = shareTmpUb.ReinterpretCast<krCacheType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputKrLocal[baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();
    int64_t stride = baseParams_->headSizeCkv + baseParams_->headSizeKr; // 512 + 64

    LocalTensor<ropeComputType> cosLocal;
    LocalTensor<ropeComputType> sinLocal;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal = cosLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    } else {
        cosLocal = cosLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    }
    Rectangle ropeParams{
        (uint32_t)vectorRow_, // row
        (uint32_t)baseParams_->dimHeadRope, // col
        (uint32_t)stride// stride
    };
    if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value && std::is_same<krCacheType, bfloat16_t>::value) {
        LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, 
            sharedBuf, ropeParams,
            dequantScaleWDkvKrLocal_[baseParams_->headSizeCkv], dequantScaleXLocal);
    } else if constexpr (std::is_same<krCacheType, int8_t>::value) {
        LocalTensor<ropeSinCosType> tmpOut = ropeShareTmpUb.ReinterpretCast<ropeSinCosType>();
        LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, ropeSinCosType>(
            tmpOut, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, 
            sharedBuf, ropeParams);
        RopePostQuantPerChannel(outputKrLocal, tmpOut, quantScaleCkrLocal_, sharedBuf, vectorRow_ * baseParams_->dimHeadRope);
    } else {
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType>(outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset],
            cosLocal, sinLocal, ropeShareTmpUb, ropeParams);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // scatter(Kr)
    // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
    int64_t paTokenIndex = cacheIndexGm_(ropeAndScatterKrParams.tokenIndex);
    ScatterCache<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(krCacheGm_, outputKrLocal,
        ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->dimHeadRope});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex, 
                                                                           int64_t rmsNormCkvOffset,
                                                                           int64_t ropeKrOffset,
                                                                           int64_t curVecToken) {
    // if (blockIdx_ >= curVectorBlockNum_) {
    //     return;
    // }
    LocalTensor<float> dequantScaleXLocal = shareBuffer_.Get<float>();
    LocalTensor<ropeComputType> cosLocalCkvKr = dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<ropeComputType>();
    LocalTensor<ropeComputType> sinLocalCkvKr = cosLocalCkvKr[baseParams_->dimHeadRope * curVecToken];
    LocalTensor<uint8_t> shareTmpUb = sinLocalCkvKr[baseParams_->dimHeadRope * curVecToken].template ReinterpretCast<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt) {
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocalCkvKr, sinLocalCkvKr, ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken,
                    shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
    if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        printf("curVecTokenIdx:%d, tokenIndex:%d, curVecToken:%d\n", curVecTokenIdx, tokenIndex, curVecToken);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        CkvkrParams rmsNormAndScatterCkvParams {
            tokenIndex,
            rmsNormCkvOffset, 
            curVecTokenIdx
        };

        RmsNormAndScatterCkv(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, rmsNormAndScatterCkvParams);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        
        CkvkrParams ropeAndScatterKrParams {
            tokenIndex,
            ropeKrOffset, 
            curVecTokenIdx
        };
        RopeAndScatterKr(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, ropeAndScatterKrParams);

        tokenIndex += 1;
        rmsNormCkvOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
        ropeKrOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken,
                                                          int64_t curBlockTokenOffset) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    uint64_t stride = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;


    LocalTensor<uint8_t> ropeShareTmpUb;
    LocalTensor<ropeOutputType> outputLocal;
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value){
        uint64_t row = baseParams_->numHeadSize;
        uint64_t col = baseParams_->dimHeadRope;
        DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                        static_cast<uint32_t>((stride - col) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[baseParams_->dimHeadSizeQc], copyParams, padParams); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        outputLocal = channelDeqScaleLocal [row * col].template ReinterpretCast<ropeOutputType>();
        ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();
    } else {
        outputLocal = shareBuffer_.Get<ropeOutputType>();
        ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();
    }

    Rectangle ropeParams{(uint32_t)baseParams_->numHeadSize, // row
        (uint32_t)baseParams_->dimHeadRope, // col
        (uint32_t)stride// stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Rope(Qr) ──> query_rope_out
        
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal, mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb,
                ropeParams,
                channelDeqScaleLocal, dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM]);
        } else {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal, mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, baseParams_->headSizeQr);

        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) * 
                            static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

// 用于enableGroupComputeOpt场景 BS = 1 ， G = 8
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset) {
    if (blockIdx_ < QC_CORE_NUM * 2 || blockIdx_ >= (QC_CORE_NUM + QR_CORE_NUM) * 2) {
        return;
    }

    uint32_t row = baseParams_->numHeadSize / QC_CORE_NUM;
    uint32_t col = baseParams_->dimHeadRope;
    int64_t stride = col;
    int64_t strideScale = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;
    int64_t deqScaleOffset = baseParams_->dimHeadSizeQc + row * (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * (blockIdx_ - QC_CORE_NUM * 2);
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)),
                    static_cast<uint32_t>((strideScale - col) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[deqScaleOffset], copyParams, padParams); // 复用内存
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    LocalTensor<ropeOutputType> outputLocal = channelDeqScaleLocal [row * col].template ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    Rectangle ropeParams{
        (uint32_t)col, // row
        (uint32_t)col, // col
        (uint32_t)stride// stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVectorBlockNum_; curVecTokenIdx++) {

        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        // sharetmp共用，待V算完再执行MTE2搬运规避风险
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, curVecTokenIdx * baseParams_->dimHeadRope, 1,
            ropeShareTmpUb, vectorRow_, baseParams_->dimHeadRope);
        
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        
        RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocal, mmQcQrResGm_[ropeQrOffset],
            cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams, channelDeqScaleLocal, dequantTool_.deQuantScaleCqLocal_);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, row * col);

                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        
        ropeQrOffset += static_cast<int64_t>(baseParams_->numHeadSize) * baseParams_->dimHeadRope;
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RopeQrSplitN(const RopeQrSplitNParams& ropeQrSplitNParams){
    uint32_t subBlockIdx_ = blockIdx_ & 1;
    uint32_t ropeCnt = (mmQcQrParam_.m + subBlockIdx_) >> 1;
    uint32_t colQr = baseParams_->dimHeadRope;                                      
    uint32_t ropeCntDown= mmQcQrParam_.m >> 1; // 向下取整，处理奇数核的偏移
    uint32_t deQuantScaleCqOffset = ropeCntDown * subBlockIdx_ * FP32_BLOCK_ELEMENT_NUM;

    DataCopyParams outputRopeParams {
    static_cast<uint16_t>(ropeCnt),
    static_cast<uint16_t>(colQr * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE),
    0,
    static_cast<uint16_t>(ropeQrSplitNParams.ropeDstStride * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        ropeCnt, // row
        colQr, // col
        (uint32_t)ropeQrSplitNParams.ropeStride// stride
    };

    GlobalTensor<mmQcQrOutputType> inputGmRope = mmQcQrResGm_[ropeQrSplitNParams.ropeQrOffset];
    GlobalTensor<float> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
    GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrSplitNParams.ropeQrResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocalRope[ropeCnt * colQr].template ReinterpretCast<uint8_t>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
        cosLocal_, sinLocal_,ropeShareTmpUb, ropeParams, ropeQrSplitNParams.ropeStride,  deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
        dequantTool_.deQuantScaleCqLocal_[deQuantScaleCqOffset]);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    DataCopy(outputGmRope[ropeQrSplitNParams.outputOffsetRope], outputLocalRope, outputRopeParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::DequantQcQrSplitN(const DequantQcQrSplitNParams& dequantQcQrSplitN){

    uint32_t row = mmQcQrParam_.m;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colQcSingle = colQc >> 1;
    uint32_t count = row * colQcSingle;
    DataCopyParams inputCopyParams {
        static_cast<uint16_t>(row),
        static_cast<uint16_t>(colQcSingle * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>(dequantQcQrSplitN.srcStride * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        0};

    DataCopyParams outputCopyParams {
        static_cast<uint16_t>(row),
        static_cast<uint16_t>(colQcSingle * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE),
        0,
        static_cast<uint16_t>(dequantQcQrSplitN.dstStride * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};

    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[dequantQcQrSplitN.mmQnPreDequantOffset];
    // DumpTensor(inputGm, 5554, 1024);
    GlobalTensor<float> scale1Gm = deqScaleQcQrW_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[dequantQcQrSplitN.mmQnPreDequantResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<float> scale2Local = dequantTool_.deQuantScaleCqLocal_;
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    // 可以和inputLocal共享内存地址，减少UB使用
    LocalTensor<float> computeLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> scaleLocal =  inputLocal[count + FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<float>();
    // outputLocal比scaleLocal占用UB少，且不会同时使用，故可以复用UB内存
    LocalTensor<mmQnInputType> outputLocal = scaleLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{
        row, //  row
        colQcSingle, // col
        colQcSingle  // columnStride
        };
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    DataCopy(inputLocal, inputGm[dequantQcQrSplitN.inputOffset], inputCopyParams);
    DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQc >> 1);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // cast
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Dequant(computeLocal, inputLocal, scaleLocal, scale2Local, dequantParams);
    PipeBarrier<PIPE_V>();
    // cast
    Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(outputGm[dequantQcQrSplitN.outputOffset], outputLocal, outputCopyParams);

    // DumpTensor(outputGm[dequantQcQrSplitN.outputOffset], 5555, 1024);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::QcQrSplit(
    int64_t curVecToken, int64_t curBlockTokenOffset, int64_t curMSize, int64_t mmQcQrOffset, int64_t mmQnPreDequantResOffset, int64_t ropeQrOffset, int64_t ropeQrResOffset)
{
    if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }

    uint32_t resdump = mmQnPreDequantResOffset;
    uint32_t resdump1 = mmQcQrOffset;

    DataCopyParams inputCopyParams {
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadSizeQc * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>((baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->dimHeadSizeQc) * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        0};

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<float> scale2Local = dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM];
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    LocalTensor<float> computeLocal = inputLocal.template ReinterpretCast<float>();
    LocalTensor<float> scaleLocal =  inputLocal[curVecToken * baseParams_->dimHeadSizeQc + FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<float>();
    LocalTensor<mmQnInputType> outputLocal = scaleLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{
        static_cast<uint16_t>(curVecToken), //  row
        baseParams_->dimHeadSizeQc, // col
        baseParams_->dimHeadSizeQc  // columnStride
        };
    DataCopyParams outputCopyParams {
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadSizeQc * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE),
        0,
        static_cast<uint16_t>((baseParams_->headSizeQc - baseParams_->dimHeadSizeQc) * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE)};
    
    DataCopyParams outputRopeParams {
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadRope * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE),
        0,
        static_cast<uint16_t>((baseParams_->headSizeQr - baseParams_->dimHeadRope) * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        static_cast<uint16_t>(curVecToken), //  row
        baseParams_->dimHeadRope, // col
        (uint32_t)(baseParams_->numHeadSize * (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope)) // stride
        };
    int64_t ropeStride = static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
    uint32_t deqScaleOffset =0;

    int64_t deqScaleQcQrWOffset = 0;
    uint32_t colOffsetCube = 0;
    uint32_t colOffsetVec = 0;
    uint32_t colOffsetVecEnd = 0;
    uint32_t colOffsetRope = 0;
    // cube一次处理row*colCube，对应的两个vec一次处理row*colQc，两vec之间切row
    // 等cube生产足够数据了以后，vec开始消费
    // CrossCoreWaitFlag(0x2);  // FINISH_MM_QCQR_SPLIT_N
    CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_N);  // FINISH_MM_QCQR_SPLIT_N
    int32_t i = 0;
    for (uint32_t colOffsetCube = 0; colOffsetCube < mmQcQrParam_.n; colOffsetCube += mmQcQrParam_.baseN) {
        colOffsetVecEnd = colOffsetCube + mmQcQrParam_.baseN;
        if (colOffsetVecEnd > mmQcQrParam_.n) {   // 当oriCol不被colCube整除时，mm最后一个base块需要刷新col end
            colOffsetVecEnd = mmQcQrParam_.n;
        }

        // CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_N);
        // printf("CrossCoreWaitFlag FINISH_MM_QCQR_SPLIT_N %d\n", colOffsetCube / mmQcQrParam_.baseN);

        while (colOffsetVec + baseParams_->dimHeadSizeQc <= colOffsetVecEnd) {   // 循环singleNumHeadSize次
            // DequantQcQrSplitN(DequantQcQrSplitNParams{mmQcQrOffset, mmQnPreDequantResOffset, inputOffset, outputOffset, srcStride, dstStride});

            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            // DumpTensor(mmQcQrResGm_[mmQcQrOffset], 4400, baseParams_->dimHeadSizeQc);
            DataCopy(inputLocal, mmQcQrResGm_[mmQcQrOffset], inputCopyParams);
            // DumpTensor(inputLocal, 4401, 2 * baseParams_->dimHeadSizeQc);
            DataCopy(scaleLocal, deqScaleQcQrW_[colOffsetVec], baseParams_->dimHeadSizeQc);
            // DumpTensor(scaleLocal, 4402, baseParams_->dimHeadSizeQc);
            // DumpTensor(scale2Local, 4502, baseParams_->dimHeadSizeQc);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Dequant(computeLocal, inputLocal, scaleLocal, scale2Local, dequantParams);

            // DumpTensor(computeLocal, 4403, 2 * baseParams_->dimHeadSizeQc);

            PipeBarrier<PIPE_V>();
            Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, curVecToken * baseParams_->dimHeadSizeQc);
            // DumpTensor(outputLocal, 4405, baseParams_->dimHeadSizeQc);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

            DataCopy(mmQcQrResDequantGm_[mmQnPreDequantResOffset], outputLocal, outputCopyParams);
            // DumpTensor(outputLocal, 4404, baseParams_->dimHeadSizeQc);


    // DumpTensor(deqScaleQcQrW_[colOffsetVec], 4442, baseParams_->dimHeadSizeQc);
    // DumpTensor(mmQcQrResDequantGm_[mmQnPreDequantResOffset], 4443, baseParams_->dimHeadSizeQc);

        // printf("CrossCoreSetFlag FINISH_VEC_DEQUANT_QC_SPLIT_N %d\n", colOffsetVec / (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope));
        // CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);

            colOffsetVec += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            mmQcQrOffset += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            mmQnPreDequantResOffset += baseParams_->dimHeadSizeQc;
            // break;
        }

        while (colOffsetRope + baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope <= colOffsetVecEnd) {   // 循环singleNumHeadSize次

            GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrResOffset];

            LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
            LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
            LocalTensor<uint8_t> ropeShareTmpUb = outputLocalRope[curVecToken * baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();

            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

            RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType>(outputLocalRope, mmQcQrResGm_[ropeQrOffset],
                cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams, ropeStride,  deqScaleQcQrW_[colOffsetRope + baseParams_->dimHeadSizeQc],
                dequantTool_.deQuantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM]);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

            DataCopy(qrOutGm_[ropeQrResOffset], outputLocalRope, outputRopeParams);
            // DumpTensor(outputLocalRope, 4466, curVecToken * baseParams_->dimHeadRope);

            colOffsetRope += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            ropeQrOffset += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            ropeQrResOffset += baseParams_->dimHeadRope;
        }
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
    // DumpTensor(mmQcQrResGm_[resdump1], 4441, 128 * baseParams_->dimHeadSizeQc);
    // DumpTensor(deqScaleQcQrW_, 4442, 256);
    // DumpTensor(mmQcQrResDequantGm_[resdump], 4444, curVecToken * baseParams_->dimHeadSizeQc);
}

// 用于算力切分dequant切N场景
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset,
                                                                            int64_t mmQnPreDequantResOffset,
                                                                            int64_t qcQrScaleOffset)
{
    if (blockIdx_ >= QC_CORE_NUM * 2) {
        return;
    }
    uint32_t subBlockIdx_ = blockIdx_ & 1;
    uint32_t oriCol = (baseParams_->headSizeQc) / QC_CORE_NUM;
    uint32_t curCol = (baseParams_->headSizeQc) / (QC_CORE_NUM * 2);
    uint32_t srcStride = baseParams_->headSizeQc - curCol;
    uint32_t dstStride = baseParams_->headSizeQc - curCol;
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    Rectangle rectangleParams {
        (uint32_t)mmQcQrParam_.m,    // row
        (uint32_t)curCol,// col
        (uint32_t)baseParams_->headSizeQc // columnStride
    };
    DequantSplitNQc(mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset], 
                deqScaleQcQrW_[qcQrScaleOffset], dequantTool_.deQuantScaleCqLocal_, shareTmpUb, 
                rectangleParams, oriCol, dstStride, subBlockIdx_);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::DequantQc(int64_t mmQnPreDequantOffset,
                                                                      int64_t mmQnPreDequantResOffset,
                                                                      int64_t curVecToken,
                                                                      int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    
    Rectangle rectangleParams {
        (uint32_t)baseParams_->stepNumHeadDequant,    // row
        (uint32_t)baseParams_->dimHeadSizeQc,// col
        (uint32_t)baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        DequantPerTokenQc(  mmQcQrResDequantGm_[mmQnPreDequantResOffset],
                            mmQcQrResGm_[mmQnPreDequantOffset],
                            deqScaleQcQrW_, 
                            dequantTool_.deQuantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM],
                            shareTmpUb,
                            rectangleParams,
                            baseParams_->numHeadSize
                            );
        mmQnPreDequantOffset += baseParams_->headSizeQc + baseParams_->headSizeQr;
        mmQnPreDequantResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset,
                                                int64_t curStepBatchSize, int64_t curBlockTokenOffset, int64_t numHeadOffset, int64_t mmQnLoops)
{
    // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
    // >> 1 是将curStepBatchSize分到每个vec核上；
    int64_t curStepBatchSizeVec = (curStepBatchSize + (blockIdx_ & 1u)) >> 1;
    if (blockIdx_ >= baseParams_->mm4BlockNum * 2 || curStepBatchSizeVec == 0) {
        return;
    }

    // 等待前面的Qr部分完成
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY = EVENT_ID0;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;

    int64_t blockBatchOffset = (blockIdx_ & 1u) * (curStepBatchSize >> 1);
    int64_t totalSizeCkv = static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->headSizeCkv); 
    // 由于两个vec核分curStepBatchSize，各处理curStepBatchSize/2，blockIdx_ & 1 表示是否为第二个vec核
    int64_t dynamicQuantQueryOffset = baseParams_->stepBatchSize * baseParams_->numHeadSize * baseParams_->headSizeCkv * cubeBlockIdx_ +
                        blockBatchOffset * totalSizeCkv + numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryResOffset = batchOffset * totalSizeCkv + blockBatchOffset * totalSizeCkv +
        numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t scaleQueryNopeOffset = batchOffset * static_cast<int64_t>(baseParams_->numHeadSize) + 
        blockBatchOffset * static_cast<int64_t>(baseParams_->numHeadSize) + numHeadOffset;
    int64_t queryOutStride = totalSizeCkv;
    int64_t qrOutputStride = baseParams_->numHeadSize * baseParams_->dimHeadRope;
    int64_t qrPostProcessResOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeQr) + 
                                    numHeadOffset * static_cast<int64_t>(baseParams_->dimHeadRope) +
                                    blockBatchOffset * static_cast<int64_t>(baseParams_->headSizeQr);
    

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    
    float quantScaleCkv = quantScaleCkvGm_.GetValue(0);
    
    // Dynamic Quant
    SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
    SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);

    // Rope Post Process
    SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);

    CrossCoreWaitFlag(FINISH_MM_QN_SPLIT_N);
    // per-head循环
    for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
        DynamicQuantQnWithMulQr(
                            dequantScaleQNopeGm_[scaleQueryNopeOffset],
                            queryOutGm_[dynamicQuantQueryResOffset],
                            qrOutGm_[qrPostProcessResOffset],
                            mmQnResGm_[dynamicQuantQueryOffset], shareTmpUb, 
                            curStepBatchSizeVec, 
                            baseParams_->headSizeCkv, 
                            baseParams_->numHeadSize, queryOutStride,
                            // Rope Post Process
                            qrOutGm_[qrPostProcessResOffset],
                            quantScaleCkv, baseParams_->dimHeadRope, qrOutputStride);
        // DumpTensor(mmQnResGm_[dynamicQuantQueryOffset], 8881, baseParams_->headSizeCkv);
        // DumpTensor(queryOutGm_[dynamicQuantQueryResOffset], 8887, baseParams_->headSizeCkv);
        dynamicQuantQueryOffset += baseParams_->headSizeCkv;
        scaleQueryNopeOffset += 1;
        dynamicQuantQueryResOffset += baseParams_->headSizeCkv;
        qrPostProcessResOffset += baseParams_->dimHeadRope;
    }
    // Rope Post Process
    WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    // Dynamic Quant
    WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
    WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
}



} // namespace MlaPrologV3

#endif // MLA_PROLOG_V3_VEC_S1_CUB_S2_H