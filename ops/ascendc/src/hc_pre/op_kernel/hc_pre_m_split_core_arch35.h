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
 * \file hc_pre_m_k_split_core.h
 * \brief
 */

/*
  * y-register-window-accumulate（r2-tk1001-yfuse-register-window 契约特性 F2，仅本文件）：
  * round1 c1（窗口化 UB 累加器 yfusion，3.0-3.7x 回退）的修正重试——保留其已被差分证明
  * 无罪/正确的部分，替换其被证伪的累加结构。
  * (a) phase-1 k-chunk 迭代序沿用 c1 的段分组交错形态：外层窗口 chunk 索引 c ∈ [0, ceil(d/W))，
  *     内层段 j ∈ [0, hcMult)，chunk 的 k 偏移 = j*d + c*W（W=融合分支 kUbSize / legacy 分支
  *     kL1Size，见下"粒度裁决"）。交错仅在它构成原 chunk 集合纯重排（每个 k 元素恰好覆盖
  *     一次、无越界）时启用，充要条件为 d % kL1Size == 0 且 multCoreSplitKSize == hcMult*d；
  *     host tiling 恒有 kUbSize = kL1Size/2，故该条件与契约 (b) 的 d % (2*kUbSize) == 0
  *     等价。接口合法但 d 非 128 对齐（如 d=4160）时不满足该条件，保持原顺序，tk1001 整条
  *     路径与基线逐位一致。交错序无罪（c1/c2 差分：aic_mte1 下降、fixpipe_ratio ≤0.02）。
  * 粒度裁决（契约内一致性，实施时机械验证）：契约 F2 字面结构（kL1-chunk + cvLoopKSize=2 +
  * "4 cast 缓冲"驻留）在其自身 F1 预算公式 4*mUbSize*(RoundUp(kUbSize)*4+32) 下不可行——
  * 每个 kL1-chunk 含 2 个 kUb 窗口，c 组共 8 个 (段,窗口) cast 单元，而 4 缓冲仅容纳 4 个
  * kUb 宽驻留（kL1 宽驻留需 4*mUbSize*(kL1Size+8)*4B，mL1Size≤144 才适配预算，违反 F1
  * 目标 160/160/208/208/208）；实测验证：kL1-chunk 下同一 slice 被同 chunk 的两个 cvLoop
  * 先后覆写，恰好偶数窗口（flushIdx=0）全错、奇数窗口全对（bs3585 探针 bad 64-windows =
  * 0,2,4,...,62）。跨 chunk 的 cvLoop 重排（(c,w,j) 序）则破坏 AIC 的 kL1-chunk 原子 L1
  * 填充/信用同步（AIV chunk 完成标志延迟 → AIC Process 等待 → L1 信用死锁，已在实施中
  * 推演排除）。故融合分支取 kUb-chunk 粒度（W=kUbSize、cvLoop 恒 1）：段分组交错结构、
  * 4 缓冲驻留、预算公式、F1 目标、事件集合（0/1 nd2Nz ping-pong、id 2 yOut 排空）全部
  * 原样成立——每个窗口 c 的 4 个段 chunk 处理完后 4 缓冲恰好全驻留（下一窗口段 0 的
  * cast store 在本窗口 y pass load 之后，VEC 管内按序程序序覆盖复用冒险）；AIC 侧
  * CopyInB1Nd2Nz/Process 以 kL1RealSize=kUbSize 泛化调用（Process 内部按 kL1_=currentK
  * 以 baseK 步进循环，Mmad 总步数守恒），L1 A/B 区域减半使用，chunk 数×2（B tile 同
  * 字节量、CrossCore 往返×2，AIC mac_ratio 0.11-0.32 有充足 slack）。legacy 分支（含
  * 交错有效的 nopm 形态）保持 kL1-chunk 粒度与 c1 逐字节同构。
  * (b) 融合启用条件与 c1 逐字相同：kInterleave && hasPreMix_ && d % (2*kUbSize) == 0
  *     时走融合分支。y 累积结构（寄存器窗口，替代 c1 窗口化 UB 累加器）：
  *     ① c 组（=单个 kUb 宽窗口）内 hcMult(=4) 个段 chunk 的 cast 输出驻留 4 个独立等尺寸
  *        slice（单 TBuf castSegBuf 四等分：段 j 偏移 = j*mUbSize*(RoundUp(kUbSize)+8)，
  *        公共接口 TORCH_CHECK 强制 hc 维=4，故段数恒 4；nd2NzBuf 的 DOUBLE_BUFFER
  *        同款"单 TBuf 多 slice"记账模式）；ND2NZ/CopyToL1 仍即时消费各 chunk 的 slice，
  *        配对 AIC 流水零延迟；
  *     ② 第 4 段 chunk 完成后，每行做 j 内层寄存器链累加（VFYRegisterWindowPass）：
  *        对该 kUb 宽窗口 sum=Duplicate(0)；for j: brc pre_mix[row*pmRowStride+j] →
  *        LoadInputData cast slice j 的 (row, 窗口内 64-lane 子块) → Mul → Add(sum,sum,x)；
  *        然后一次 cast BF16（castTraitB322B16Even，StoreOutputData 同 VFProcessY 路径）
  *        写入 yQue，按窗口偏移 c*kUbSize CopyOut 到 yGm（与契约 chunkInSeg*kL1Size +
  *        window*kUbSize 的线性偏移等价：kUb-chunk 下 chunkInSeg*kUbSize 覆盖同一 [0,d)）；
  *     ③ 契约明令禁止的结构（相对 c1 的病理结构）：y 累加器不得在段间经 UB
  *        load/store 往返（sum 寄存器驻留）；per-row/per-window 循环内不得 UpdateMask
  *        （kUbSize 为 64 的倍数且融合条件 d%kL1Size==0 保证每窗口满块，全 lane 常量
  *        mask CreateMask 提升出全部循环；调优域 kUbSize=64=VL_FP32）；pre_mix 广播
  *        不得按 64-lane 块重复（brc 每行每段每窗口至多一次，与 VFProcessY 的 j 内层
  *        LoadInputDataWithBrc 同构）；
  *     ④ 寄存器预算：每 __VEC_SCOPE__ ≤32 RegTensor（本 pass 实际 3：x/mix/sum）；
  *     ⑤ pre_mix 每轮一次经既有 CopyInWithUbStride 模式加载（hcMixAlign 行距）；
  *     ⑥ y 逐元素运算序与 VFProcessY 完全一致（j 升序、Duplicate(0)+Add 首项，
  *        0+a 位级等于直写 a 含 -0 边界语义）→ 同输入 y 逐位一致；
  *     ⑦ 跨 c 组（窗口）的 cast slice 复用冒险由 VEC 管内按序程序序覆盖（同管 store→load
  *        经 UB 的既定可靠序，基线 cast→ND2NZ、Pre→Y 同款依赖）——不新增 c1 集合
  *        （event id 0/1 nd2Nz ping-pong、id 2 yOut 排空）之外的跨管事件。
  * (c) legacy 路径（!hasPreMix_ 或 d % (2*kUbSize) != 0）phase-2 经 VFProcessY + x
  *     重读的 y 计算逐行保留（单 castBuf 记账与基线逐字节同构，kL1-chunk 粒度）；其 k
  *     顺序在交错有效时同样为交错序（kL1 粒度，与 c1 相同）。
  * 数值注记：仅 mixes（Mmad k 块 FP32 累加分组）因交错重排变化（c1 实测 ≤1.07e-6 vs
  * 阈值 1.2207e-4；kUb 粒度 chunk 化不改变 Mmad 总步数与 k 覆盖，仅重排分组）；y/pre/post/
  * comb 的逐元素运算顺序不变。
  */

#ifndef HC_PRE_M_SPLIT_CORE_H
#define HC_PRE_M_SPLIT_CORE_H

#include "kernel_operator.h"
#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"

namespace HcPreNs {
using namespace AscendC;

// y-register-window-accumulate（F2）：融合分支 cast 缓冲 slice 数 = hcMult。公开接口
// check_hc_pre_shape_and_dtype 以 TORCH_CHECK 强制 x 的 hc 维 == 4（HC_LIMIT），故合法
// 调用域内恒为 4；hc_mult attr 与 x hc 维不一致的调用在基线即为未定义行为
// （multCoreSplitKSize/hc_fn 尺寸校验互相矛盾、GM 越界读），本结构不扩大该边界。
constexpr uint32_t YFUSE_CAST_SEG_NUM = 4;

// ===== y-register-window-accumulate（F2）phase-1 y 寄存器窗口 pass =====
// c 组第 hcMult 段 chunk 完成后，对单个 kUb 宽 y 窗口按行做 j 内层寄存器链累加并一次
// cast 写出：每行 sum=Duplicate(0)；for j∈[0,hcMult)：brc preMix[row*pmRowStride+j]
// （broadcast，每行每段每窗口至多一次）→ LoadInputData 段 j cast slice 的 (row, window)
// → Mul → Add(sum,sum,x)；StoreOutputData 单次 cast BF16（castTraitB322B16Even）落
// yOut。逐元素运算序与 phase-2 VFProcessY 的 j 内层循环完全一致（j 升序、j==0 为
// Duplicate(0)+Add），同输入下 y 逐位一致。
// pmRowStride=RoundUp(hcMix)：pre_mix 经 CopyInWithUbStride 落 UB 的 hcMixAlign 行距
// （16B 行的 DataCopyPad 块足迹按 32B 对齐，须显式行距，与 phase-2 preMix 布局一致）。
// xCastSegLocal 为 4 段 slice 的基址，段 j 偏移 = j*segRows*xCastRowStride（float 计），
// xCastRowStride 与 VFProcessCastAndInvRmsPart1 的 dstCurColNumAlign 一致
// （RoundUp<float>(colNum)+8）。
// mask：colNum=kUbSize 为 64 的倍数（kL1Size 恒为 128 的倍数、kUb=kL1/2）且融合条件
// d%kL1Size==0 保证 chunk 满块，每个 64-lane 子块全满 → 全 lane 常量 mask
// CreateMask 提升出全部循环（调优域 kUbSize=64=VL_FP32，winBlocks=1）。
template <typename T>
__aicore__ inline void VFYRegisterWindowPass(const LocalTensor<T> &yOutLocal,
                                              const LocalTensor<float> &preMixLocal,
                                              const LocalTensor<float> &xCastSegLocal,
                                              const uint16_t rowNum, const uint16_t colNum,
                                              const uint16_t hcMult, const uint16_t pmRowStride,
                                              const uint16_t segRows)
{
    __local_mem__ T *yOutAddr = (__local_mem__ T *)yOutLocal.GetPhyAddr();
    __local_mem__ float *pmAddr = (__local_mem__ float *)preMixLocal.GetPhyAddr();
    __local_mem__ float *xcSegBase = (__local_mem__ float *)xCastSegLocal.GetPhyAddr();
    uint16_t yOutRowStride = RoundUp<T>(colNum);
    uint16_t winBlocks = CeilDiv(colNum, VL_FP32);
    uint32_t xCastRowStride = RoundUp<float>(colNum) + BLOCK_SIZE / sizeof(float);
    uint32_t segStride = static_cast<uint32_t>(segRows) * xCastRowStride;
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> mix;
        RegTensor<float> sum;
        // 全 lane 常量 mask：提升出 per-row/per-window/per-segment 全部循环（契约 ③）
        MaskReg pregFull = CreateMask<float>();
        for (uint16_t i = 0; i < rowNum; i++) {
            for (uint16_t w = 0; w < winBlocks; w++) {
                Duplicate(sum, static_cast<float>(0), pregFull);
                for (uint16_t j = 0; j < hcMult; j++) {
                    LoadInputDataWithBrc<float>(mix, pmAddr, pregFull, i * pmRowStride + j);
                    LoadInputData<float>(x, xcSegBase, pregFull,
                                         j * segStride + i * xCastRowStride + w * VL_FP32);
                    Mul(x, mix, x, pregFull);
                    Add(sum, sum, x, pregFull);
                }
                StoreOutputData<T>(yOutAddr, sum, pregFull, i * yOutRowStride + w * VL_FP32);
            }
        }
    }
}

template <typename T>
class HcPreMSplitCorePart1 {
public:
    __aicore__ inline HcPreMSplitCorePart1()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcFn, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR preMix,
        GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, GM_ADDR pre, const HcPreTilingData* tilingDataPtr,
        TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcFnGm.SetGlobalBuffer((__gm__ float*)hcFn);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        hasPreMix_ = (preMix != nullptr);
        hasPreOut_ = (pre != nullptr);
        if (hasPreMix_) {
            preMixGm.SetGlobalBuffer((__gm__ float*)preMix);
        }
        if (hasPreOut_) {
            preGm.SetGlobalBuffer((__gm__ float*)pre);
        }
        ubRowGapBlocks_ = UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];
        
        pipe->InitBufPool(tbufPool0, tilingData->bufferPool0Size);
        tbufPool0.InitBuffer(mmXBuf, CeilDiv(tilingData->mL1Size, 2) * RoundUp<float>(tilingData->hcMix) * sizeof(float));
        mmXLocal = mmXBuf.Get<float>();

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        } else {
            tbufPool0.InitBuffer(rmsNormBuf, RoundUp<float>(CeilDiv(tilingData->mL1Size, 2)) * sizeof(float));
            tbufPool0.InitBufPool(tbufPool1, tilingData->bufferPool1Size);

            tbufPool0.InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

            hcBase0Local = hcBaseBuf0.Get<float>();
            hcBase1Local = hcBaseBuf1.Get<float>();
            hcBase2Local = hcBaseBuf2.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t logicalBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            logicalBlockIdx = curBlockIdx / 2;
        }
        if (logicalBlockIdx >= tilingData->cubeBlockDimM) {
            if ASCEND_IS_AIV {
                // Drain the two double-buffer credits seeded by the paired AIC.
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            } else {
                mmService_.End();
            }
            return;
        }

        if ASCEND_IS_AIV {
          CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, 1, tilingData->hcMult * tilingData->hcMult);
          event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
          SetFlag<HardEvent::MTE2_V>(eventId);
          WaitFlag<HardEvent::MTE2_V>(eventId);
        }

        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimM;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        // todo 移到tiling计算
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound = (mainCoreCount == 0 || logicalBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;
        if ASCEND_IS_AIC {
            if (mainCoreCount == 0 || curBlockIdx <= mainCoreCount) {
                mGmOffset = curBlockIdx * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset = (mainCoreCount * singleCoreMaxRound + (curBlockIdx - mainCoreCount) * (singleCoreMaxRound - 1)) * tilingData->mL1Size;
            }
        } else {
            if (mainCoreCount == 0 || (curBlockIdx / 2) <= mainCoreCount) {
                mGmOffset = curBlockIdx / 2 * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset = (mainCoreCount * singleCoreMaxRound + (curBlockIdx / 2 - mainCoreCount) * (singleCoreMaxRound - 1)) * tilingData->mL1Size;
            }
        }
        int64_t xGmBaseOffset = 0;
        int64_t yGmBaseOffset = 0;
        int64_t postGmBaseOffset = 0;
        int64_t combFragGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            xGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->d;
            yGmBaseOffset = mGmOffset * tilingData->d;
            postGmBaseOffset = mGmOffset * tilingData->hcMult;
            combFragGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->hcMult;
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        }
  
        uint64_t cvLoopKSize = tilingData->kL1Size / tilingData->kUbSize;  
        int64_t xSplitOffset = 0;
        int64_t ySplitOffset = 0;
        int64_t postSplitOffset = 0;
        int64_t combFragSplitOffset = 0;
        int64_t xOutSplitOffset = 0;
        // y-register-window-accumulate（F2）路由标志（详见文件头注释）：
        // kInterleave——段分组交错构成原 kL1-chunk 集合纯重排（每 k 元素恰好一次、
        //   无越界）的充要条件：d % kL1Size == 0 且 multCoreSplitKSize == hcMult*d
        //   （kGmEndOffset 即 multCoreSplitKSize）；不满足时 k 迭代保持原顺序。
        // fusedYRound——融合条件 kInterleave && hasPreMix_ && d % (2*kUbSize) == 0，
        //   且 2*kUbSize == kL1Size 使其与 d % kL1Size == 0 等价；此时 phase-1 寄存器
        //   窗口累积 y、phase-2 跳过 x 重读与 VFProcessY，其余走 legacy（原路径）。
        const bool kInterleave =
            (tilingData->d % tilingData->kL1Size == 0) &&
            (tilingData->multCoreSplitKSize == tilingData->hcMult * tilingData->d);
        const bool fusedYRound =
            kInterleave && hasPreMix_ && (tilingData->d % (2 * tilingData->kUbSize) == 0);
        // F2 粒度裁决（详见文件头注释）：融合分支 k-chunk 粒度 = kUbSize（每 chunk 恰一个
        // kUb 宽 y 窗口、cvLoop 恒 1，4 段 chunk 后 4 cast slice 全驻留）；legacy 分支保持
        // kL1Size 粒度（cvLoopKSize 个 cvLoop，与基线/c1 逐字节同构）。
        const uint64_t kChunkWidth = fusedYRound ? tilingData->kUbSize : tilingData->kL1Size;
        const uint64_t cvLoopCount = fusedYRound ? 1 : cvLoopKSize;
        // m轴切分 按照0 0 1 1..分核
        for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx)
        {
            uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t kGmStartOffset = 0;
            uint64_t kGmEndOffset = tilingData->multCoreSplitKSize;
            uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
            if ASCEND_IS_AIV {
                tbufPool1.Reset();
                tbufPool1.InitBuffer(xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
                if (fusedYRound) {
                    // F2(b)-①: 4 段 cast 缓冲驻留——单 TBuf 四等分 slice（段 j 偏移 =
                    // j*mUbSize*(RoundUp(kUbSize)+8) float，nd2NzBuf 的 DOUBLE_BUFFER
                    // 同款记账模式）；段 j 的 chunk cast 输出驻留至本 c 组 y 寄存器窗口
                    // pass 读取，ND2NZ/CopyToL1 仍即时消费各 chunk 的 slice。
                    tbufPool1.InitBuffer(castSegBuf, YFUSE_CAST_SEG_NUM * tilingData->mUbSize *
                        (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
                    xCastSegLocal = castSegBuf.Get<float>();
                    // yOut 双 buffer TQue（V→MTE3 异步写出）、pre_mix 单 buffer TQue
                    // （每轮加载一次，F2(b)-⑤）
                    tbufPool1.InitBuffer(yQue, 2,
                        tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
                    tbufPool1.InitBuffer(preMixQue, 1,
                        tilingData->mUbSize * RoundUp<float>(tilingData->hcMix) * sizeof(float));
                } else {
                    // legacy：单 castBuf 记账与基线逐字节同构（F2(c)）
                    tbufPool1.InitBuffer(castBuf, tilingData->mUbSize * (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
                    xCastLocal = castBuf.Get<float>();
                }
                tbufPool1.InitBuffer(nd2NzBuf, nd2NzBufSize * sizeof(float) * DOUBLE_BUFFER);

                xNd2NzLocal = nd2NzBuf.Get<float>();
                rmsNormLocal = rmsNormBuf.Get<float>();
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                if (GetBlockIdx() % 2 != 0) {
                    xSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->d;
                    xOutSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->d;
                    ySplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->d;
                    postSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult;
                    combFragSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->hcMult;
                }
                if (fusedYRound) {
                    // F2(b)-⑤: pre_mix 每行 hcMult 个 float，phase-1 每轮加载一次
                    // （TQue 管理 MTE2→V 同步）；行划分与 phase-2 legacy 的 preMix
                    // 加载一致：偶数核前半 CeilDiv(mL1RealSize,2) 行，奇数核后半。
                    int64_t preMixRowFactor = CeilDiv(mL1RealSize, 2);
                    if (curBlockIdx % 2 == 1) {
                        preMixRowFactor = mL1RealSize - preMixRowFactor;
                    }
                    preMixLocal = preMixQue.AllocTensor<float>();
                    // 16B 行经 DataCopyPad 落 UB 时块足迹按 32B 对齐，须用与 phase-2 相同的
                    // ubRowGapBlocks_ 显式行距（hcMixAlign=24 float 行距），读取侧按
                    // pm[i*RoundUp(hcMix)+j] 索引（与 VFProcessY 的 mix 索引同构）
                    CopyInWithUbStride(
                        preMixGm[postGmBaseOffset + postSplitOffset +
                                 roundIdx * tilingData->mL1Size * tilingData->hcMult],
                        preMixLocal, preMixRowFactor, tilingData->hcMult, 0, ubRowGapBlocks_);
                    preMixQue.EnQue(preMixLocal);
                    preMixLocal = preMixQue.DeQue<float>();
                }
            }
            // k轴切分（kCoreDim=1）
            int64_t bufferIdx = 0;
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
            // F2(a): k-chunk 迭代序。kInterleave 时段分组交错（外层窗口 chunk 索引
            // c ∈ [0, ceil(d/W))，内层段 j ∈ [0, hcMult)，chunk k 偏移 = j*d + c*W，
            // W=kChunkWidth：融合分支 kUbSize / legacy 分支 kL1Size），为原顺序 chunk
            // 集合的纯重排，循环体零改动；否则保持原顺序（kGmOffset 等差递增）。
            // 两种顺序 chunk 总数一致。
            const uint64_t kChunkTotal = CeilDiv(kGmEndOffset - kGmStartOffset, kChunkWidth);
            for (uint64_t kChunkIdx = 0; kChunkIdx < kChunkTotal; kChunkIdx++) {
                uint64_t segIdx = 0;
                uint64_t chunkInSeg = 0;
                int64_t kGmOffset = kGmStartOffset;
                if (kInterleave) {
                    segIdx = kChunkIdx % (uint64_t)tilingData->hcMult;
                    chunkInSeg = kChunkIdx / (uint64_t)tilingData->hcMult;
                    kGmOffset = kGmStartOffset +
                        (int64_t)(segIdx * (uint64_t)tilingData->d + chunkInSeg * kChunkWidth);
                } else {
                    kGmOffset = kGmStartOffset + (int64_t)(kChunkIdx * kChunkWidth);
                }
                if ASCEND_IS_AIC {
                    bool isFirstKL1 = kGmOffset == kGmStartOffset;
                    bool isLastKL1 = (kGmOffset + kChunkWidth) >= kGmEndOffset;
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, kChunkWidth);
                    mmService_.CopyInB1Nd2Nz(tilingData->multCoreSplitKSize, kL1RealSize, 
                                             tilingData->hcMix, hcFnGm[kGmOffset], 
                                             wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                    uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                    uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                    mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize, (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32, 
                                       isFirstKL1, isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET], wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if (isLastKL1) {
                        mmService_.CopyOut(mmXLocal);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG + FLAG_ID_MAX);
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    // 偶数核取前半段(CeilDiv，多处理一行)、奇数核取后半段，须与 sinkhorn/输出阶段的行划分一致，
                    // 否则 mL1RealSize 为奇数时后半段错位一行。
                    int64_t rowFactor = CeilDiv(mL1RealSize, 2);
                    int64_t tailRowFactor = mL1RealSize - rowFactor;
                    int64_t curRowFactor = rowFactor;
                    int64_t mL1SizeAlign = CeilAlign(mL1RealSize, AscendC::BLOCK_CUBE);
                    if (curBlockIdx % 2 == 1) {
                        curRowFactor = tailRowFactor;
                    }
                    float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                    // F2(b)-①: 段 j chunk 的 cast 落点——融合分支为 castSegBuf 的段 j
                    // slice（偏移 j*mUbSize*(RoundUp(kUbSize)+8) float），legacy 为单
                    // castBuf（基线原样）；三元只对选中分支求值，另一侧 handle 不被解引用
                    int64_t castSegStride = static_cast<int64_t>(tilingData->mUbSize) *
                        (RoundUp<float>(tilingData->kUbSize) + BLOCK_SIZE / sizeof(float));
                    LocalTensor<float> xCastDst = fusedYRound
                        ? xCastSegLocal[static_cast<uint32_t>(segIdx) * static_cast<uint32_t>(castSegStride)]
                        : xCastLocal;
                    for (int64_t cvLoopIdx = 0; cvLoopIdx < (int64_t)cvLoopCount; cvLoopIdx++) {
                        uint64_t kRealSize = kGmOffset + tilingData->kUbSize >= kGmEndOffset ? kGmEndOffset - kGmOffset : tilingData->kUbSize;

                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(xGm[xGmBaseOffset + xSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d + kGmOffset + cvLoopIdx * tilingData->kUbSize],
                               xLocal, curRowFactor, tilingData->kUbSize, tilingData->hcMult * tilingData->d - tilingData->kUbSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                            VFProcessCastAndInvRmsPart1<T, false>(rmsNormLocal, xCastDst, xLocal, coeff, curRowFactor, tilingData->kUbSize);
                        } else {
                            VFProcessCastAndInvRmsPart1<T, true>(rmsNormLocal, xCastDst, xLocal, coeff, curRowFactor, tilingData->kUbSize);
                        }
                        xQue.template FreeTensor(xLocal);
                        
                        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastDst, curRowFactor, tilingData->kUbSize);
                        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                        if (curBlockIdx % 2 == 0) {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                            CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                        } else {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) -  curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                            CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + rowFactor * (BLOCK_SIZE / sizeof(float)) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                        }
                        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        bufferIdx++;
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    if (fusedYRound && segIdx == (uint64_t)tilingData->hcMult - 1) {
                        // F2(b)-②: 本窗口 c 组全部 hcMult 段 chunk 完成 → 4 cast slice 恰好
                        // 全驻留，每行做 j 内层寄存器链累加（sum 寄存器驻留、全 lane 常量
                        // mask、brc 每行每段每窗口一次）+ 单次 cast BF16 写出；置于
                        // CrossCoreSetFlag 之后，不阻塞配对 AIC 取下一 chunk。跨窗口的
                        // slice 复用冒险由 VEC 管内按序程序序覆盖（F2(b)-⑦：下一窗口段 0
                        // 的 cast store 在本 pass 的 load 之后，同管 store→load 既定可靠序）。
                        // 窗口 GM 偏移 = chunkInSeg*kUbSize（kUb-chunk 粒度下与契约
                        // chunkInSeg*kL1Size + window*kUbSize 的线性偏移等价）。
                        yLocal = yQue.template AllocTensor<T>();
                        VFYRegisterWindowPass<T>(yLocal, preMixLocal, xCastSegLocal,
                                                 (uint16_t)curRowFactor, (uint16_t)tilingData->kUbSize,
                                                 (uint16_t)tilingData->hcMult,
                                                 (uint16_t)RoundUp<float>(tilingData->hcMix),
                                                 (uint16_t)tilingData->mUbSize);
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<T>();
                        CopyOut(yLocal,
                                yGm[yGmBaseOffset + ySplitOffset +
                                    roundIdx * tilingData->mL1Size * tilingData->d +
                                    (int64_t)chunkInSeg * tilingData->kUbSize],
                                curRowFactor, tilingData->kUbSize, tilingData->d - tilingData->kUbSize);
                        yQue.template FreeTensor(yLocal);
                    }
                }
                aL1BufferID_ ^= 1;
            }

            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
                if (fusedYRound) {
                    // F2: 排空融合分支全部 yOut 的 MTE3 写出后再 Reset/复用 tbufPool1：
                    // SetFlag 落 MTE3 流（位于全部 yOut 拷贝之后才触发），WaitFlag 落
                    // V 流（位于 phase-2 首个 pool 写之前）。event id 2：MTE3_V 硬件
                    // 共 8 个 id（QUE_MAX_EVENT=8, arch35），本文件手动仅用 0/1
                    // （ND2NZ ping-pong），AIV 侧 TQue 走 bufId 互斥不占 event id。
                    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
                    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
                    preMixQue.FreeTensor(preMixLocal);
                }
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_PRE_POST_FLAG);
                // mm计算结果存入mmXLocal，mmXLocal每轮循环需要累加;
                tbufPool1.Reset();
                if (!fusedYRound) {
                    // F2/F3: 融合分支 phase-2 不再重读 x / 计算 y，跳过 xQue/yQue 分配
                    // （legacy 原样）
                    tbufPool1.InitBuffer(xQue, 2, tilingData->rowInnerFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
                    tbufPool1.InitBuffer(
                        yQue, 2, tilingData->rowInnerFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
                }
                tbufPool1.InitBuffer(postQue, 2, tilingData->rowInnerFactor * tilingData->hcMultAlign * sizeof(float));
                tbufPool1.InitBuffer(combFragQue, DOUBLE_BUFFER,
                    tilingData->rowInnerFactor * tilingData->hcMult * tilingData->hcMult * sizeof(float));

                // TBuf
                tbufPool1.InitBuffer(mixesBuf, tilingData->rowInnerFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

                // 可选输入pre_mix使用独立的UB空间，由TQue管理MTE2->V同步；未传入时跳过分配
                // F2: 融合分支 pre_mix 已在 phase-1 加载，phase-2 不再使用
                if (hasPreMix_ && !fusedYRound) {
                    tbufPool1.InitBuffer(preMixQue, DOUBLE_BUFFER,
                        tilingData->rowInnerFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));
                }
                // 可选输出pre使用独立UB空间（hcMultAlign行距紧凑布局），TQue double buffer
                // 管理V->MTE3同步，MTE3搬出与下一轮V计算重叠；未请求输出时跳过分配
                if (hasPreOut_) {
                    tbufPool1.InitBuffer(preQue, DOUBLE_BUFFER,
                        tilingData->rowInnerFactor * tilingData->hcMultAlign * sizeof(float));
                }

                mixesLocal = mixesBuf.Get<float>();

                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

                // m内层循环
                int64_t currentRow = mL1RealSize / 2;
                if (mL1RealSize % 2 == 1 && curBlockIdx % 2 == 0) {
                    // m不整除时偶数核多处理一行
                    currentRow += 1;
                }
                for (int64_t innerRowIdx = 0; innerRowIdx < currentRow; innerRowIdx += tilingData->rowInnerFactor) {
                    int64_t currentInnerRowFactor = innerRowIdx + tilingData->rowInnerFactor >= currentRow ? currentRow - innerRowIdx : 
                                                    tilingData->rowInnerFactor;
                    VFProcessInvRmsPart3(mixesLocal, mmXLocal[innerRowIdx * tilingData->hcMix], rmsNormLocal[innerRowIdx],
                                         tilingData->normEps, currentInnerRowFactor, tilingData->hcMix);

                    if (hasPreMix_ && !fusedYRound) {
                        preMixLocal = preMixQue.AllocTensor<float>();
                        CopyInWithUbStride(
                            preMixGm[postGmBaseOffset + postSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult +
                                     innerRowIdx * tilingData->hcMult],
                            preMixLocal, currentInnerRowFactor, tilingData->hcMult, 0, ubRowGapBlocks_);
                        preMixQue.EnQue(preMixLocal);
                    }

                    // 内部pre仅在需要输出或未传入pre_mix(用于y计算)时计算
                    if (hasPreOut_ || !hasPreMix_) {
                        VFProcessPre(
                            mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                            currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);
                    }
                    if (hasPreOut_) {
                        // pre与post同为[bs, hcMult]，复用post的GM偏移；先将mixesLocal行首的hcMult个
                        // 元素按hcMixAlign行距聚拢到preLocal(hcMultAlign行距)，再经TQue异步搬出
                        preLocal = preQue.AllocTensor<float>();
                        CopyOut(mixesLocal, preLocal, currentInnerRowFactor, tilingData->hcMult, 0,
                                ubRowGapBlocks_);
                        preQue.EnQue(preLocal);
                        preLocal = preQue.DeQue<float>();
                        CopyOut(preLocal,
                                preGm[postGmBaseOffset + postSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult +
                                        innerRowIdx * tilingData->hcMult],
                                currentInnerRowFactor, tilingData->hcMult);
                        preQue.FreeTensor(preLocal);
                    }
                    if (hasPreMix_ && !fusedYRound) {
                        preMixLocal = preMixQue.DeQue<float>();
                    }
                    // F2/F3: 融合分支 y 已在 phase-1 完成写出，跳过 x 重读与 VFProcessY
                    // （循环体 legacy 原样保留，dLoopTrip=0 时零迭代）
                    const int64_t dLoopTrip = fusedYRound ? 0 : tilingData->dLoop;
                    for (int64_t dLoopIdx = 0; dLoopIdx < dLoopTrip; dLoopIdx++)
                    {
                        int64_t curDFactor =
                            (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(
                            xGm[xGmBaseOffset + xOutSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d +
                                innerRowIdx * tilingData->hcMult * tilingData->d + dLoopIdx * tilingData->dFactor],
                            xLocal, currentInnerRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();

                        yLocal = yQue.template AllocTensor<T>();
                        // pre_mix传入时y的加权求和使用pre_mix(布局与mixesLocal一致)，否则使用本轮计算的pre
                        VFProcessY(yLocal, hasPreMix_ ? preMixLocal : mixesLocal, xLocal, currentInnerRowFactor,
                                   tilingData->hcMult, curDFactor, tilingData->hcMix);
                        xQue.template FreeTensor(xLocal);
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<T>();
                        CopyOut(yLocal, yGm[yGmBaseOffset + ySplitOffset + roundIdx * tilingData->mL1Size * tilingData->d + innerRowIdx * tilingData->d + dLoopIdx * tilingData->dFactor],
                                currentInnerRowFactor, curDFactor, tilingData->d - curDFactor);
                        yQue.template FreeTensor(yLocal);
                    }
                    if (hasPreMix_ && !fusedYRound) {
                        preMixQue.FreeTensor(preMixLocal);
                    }

                    // post
                    postLocal = postQue.AllocTensor<float>();
                    VFProcessPost(
                        postLocal, mixesLocal[tilingData->hcMult], hcBase1Local,
                        hcScaleGm.GetValue(1), tilingData->hcEps, currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);

                    postQue.EnQue(postLocal);
                    postLocal = postQue.DeQue<float>();
                    CopyOut(postLocal, postGm[postGmBaseOffset + postSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult + innerRowIdx * tilingData->hcMult], currentInnerRowFactor, tilingData->hcMult);
                    postQue.FreeTensor(postLocal);

                    // combFrag
                    combFragLocal = combFragQue.AllocTensor<float>();
                    VFProcessCombFragPacked(
                        combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local, hcScaleGm.GetValue(2), tilingData->hcEps,
                        tilingData->iterTimes - 1, currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);

                    combFragQue.EnQue(combFragLocal);
                    combFragLocal = combFragQue.DeQue<float>();
                    CopyOut(combFragLocal, combFragGm[combFragGmBaseOffset + combFragSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->hcMult + innerRowIdx * tilingData->hcMult * tilingData->hcMult],
                            currentInnerRowFactor, tilingData->hcMult * tilingData->hcMult);
                    combFragQue.FreeTensor(combFragLocal);
                }
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            }
        }
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            // Drain the two double-buffer credits seeded by the paired AIC.
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        } else {
            mmService_.End();
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> invRmsGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> preMixGm;
    GlobalTensor<float> preGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;
    TQue<QuePosition::VECIN, 1> preMixQue;
    TQue<QuePosition::VECOUT, 1> preQue;

    // legacy 单 cast 缓冲（!fusedYRound，与基线逐字节同构，F2(c)）
    TBuf<QuePosition::VECCALC> castBuf;
    // y-register-window-accumulate（F2）：融合分支 4 段 cast 缓冲（单 TBuf 四等分
    // slice，段 j 偏移 = j*mUbSize*(RoundUp(kUbSize)+8) float）
    TBuf<QuePosition::VECCALC> castSegBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;

    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> rmsNormBuf;
    TBuf<QuePosition::VECCALC> mmXBuf;

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    // legacy 单 cast 落点（!fusedYRound）
    LocalTensor<float> xCastLocal;
    // y-register-window-accumulate（F2）：4 段 cast 缓冲基址（段 j slice 偏移由
    // segIdx*mUbSize*(RoundUp(kUbSize)+8) 运行期计算）
    LocalTensor<float> xCastSegLocal;
    LocalTensor<float> xNd2NzLocal;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> preMixLocal;
    LocalTensor<float> preLocal;
    bool hasPreMix_ = false;
    bool hasPreOut_ = false;
    uint32_t ubRowGapBlocks_ = 0;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 10;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_{0};

    // TBufPool第二模板参数为描述符表长度：TQue的每个buffer占1槽位，嵌套子池占子池表长个槽位。
    // tbufPool1 phase1 融合分支峰值 = xQue2+castSegBuf1+nd2NzBuf1+yQue2+preMixQue1 = 7；
    // phase2 legacy 峰值 = xQue2+yQue2+postQue2+combFragQue2+mixes1+preMixQue2+preQue2 = 13 > 12
    // 会越界写坏pool元数据（基线既有注释），16 槽位两分支均适配。
    // 父子池模板参数必须一致：InitBufPool需跨实例访问protected/private成员，类型不同无法编译
    TBufPool<QuePosition::VECCALC, 16> tbufPool0;
    TBufPool<QuePosition::VECCALC, 16> tbufPool1;
};

} // namespace HCPreSinkhorn

#endif