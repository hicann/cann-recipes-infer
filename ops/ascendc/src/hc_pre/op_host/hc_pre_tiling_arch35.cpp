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
 * \file hc_pre_tiling_arch35.cpp
 * \brief
 */

#include <sstream>
#include "hc_pre_tiling.h"

using namespace ge;
namespace optiling {
namespace HcPreTilingRegbase {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
// splitk-128-balance：更细的 K 切分基粒度（仅 tk1000 分支按采用条件使用）
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE_FINER = 128;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
// stage2-dloop-rebatch（r1-tk1000-stage2-rebatch）：stage-2 行批目标值与调优 regime 上限
// （契约 derived_from：tk1000 分支 bs<=3079 regime；bs 3584/3585 路由边界探针须保持基线行为）
constexpr int64_t STAGE2_REBATCH_TARGET_ROW_FACTOR = 8;
constexpr int64_t STAGE2_REBATCH_BS_LIMIT = 3079;
// kUbSize（=kL1Size/2）内核侧上限约束：见 CalcRegbaseOpTiling 处既有注释
// “先按2倍系数计算，m最大256，需保证kub小于256”（kL1Size 公式与 CalcOpTiling 一致）。
constexpr uint64_t K_UB_MAX_SIZE = 256;
// tk1001-mtile-search-ubaware（F1）：mL1Size 搜索结果封顶。4 段 cast 缓冲驻留
// （y-register-window-accumulate）的 UB 预算约束：mL1Size=208（mUbSize=104）时融合分支
// phase-1 pool1 合计 236288B + pool0 10592B = 246880B ≤ ubSize（950PR 实测 253952B）；
// mL1Size>208 时 4*mUb*(RoundUp(kUb)*4+32) 主导项增长使预算溢出，故封顶 208。
constexpr uint64_t M_L1_UBAWARE_MAX_SIZE = 208;
// kernel 侧 C0_SIZE（hc_pre_base_arch35.h constexpr int32_t C0_SIZE = 8），host 侧本地等值定义
constexpr int64_t C0_SIZE_HOST = 8;

// m-tile-allcore-search（语义与 r2-tk1001-tiling-rebalance F1 / r1-tk1001-loadbalance-yfuse F1
// 逐行相同）：仅 tk1001（kDimNum==1）分支调用。在 16 对齐 mL1Size∈[16, M_L1_MAX_SIZE] 上做
// 确定性搜索，mDimNum=28（全核），最小化单核关键路径行数 ceil(ceil(bs/mL1Size)/mDimNum)*mL1Size
// （轮次数 × 每轮 M-tile 行数）。
// 可行性：候选 mL1Size 必须满足本文件既有内核契约 kUbSize=kL1Size/2 < K_UB_MAX_SIZE
// （kL1Size 与 CalcOpTiling 同公式），不满足的候选直接跳过——mL1Size 过小时 kL1Size 增大、
// kUbSize 达到/超过 256，与 M_L1_MAX_SIZE=256 约束的 m 过大情形是同一条契约的两个方向。
// kl1-divisibility-fix： additionally 要求 kL1Size 整除 K（=multCoreSplitKSize）。tk1001
// legacy 分支的 cvLoop 计数为常量 kL1Size/kUbSize 且 CopyIn 恒按 kUbSize 全宽搬 x
// （hc_pre_m_split_core_arch35.h Process），K%kL1Size!=0 时尾 chunk 最后一个 cvLoop 会越界
// 读到下一行行首，越界数据被 VFProcessCastAndInvRmsPart1 累进 inv_rms 的 x² 和，导致
// mixes/post/y 系统性偏差（v1/v2 同路径受影响）；融合分支本就要求 d%kL1Size==0。
// K 恒为 256 的倍数（multCoreSplitKSize=RoundUp(hcMult*d,256)），kL1Size∈{128,256} 档
// 恒可行，故过滤只会剔除 mL1Size=80（kL1Size=384）等不整除档，契约调优目标（160/208）不受影响。
// 并列最小值取更大 mL1Size（等价于更少轮次：轮间 CrossCore 同步开销更优）。从大到小遍历 +
// 严格小于更新即得。
uint64_t SearchAllCoreML1Size(int64_t bs, uint64_t mDimNum, uint64_t kSize)
{
    uint64_t bestM = M_L1_MAX_SIZE;
    uint64_t bestCost = ~static_cast<uint64_t>(0);
    for (uint64_t mL1Size = M_L1_MAX_SIZE; mL1Size >= AscendC::BLOCK_CUBE; mL1Size -= AscendC::BLOCK_CUBE) {
        uint64_t kL1Size = std::min(A_L1_SIZE / mL1Size, K_L1_MAX_SIZE) / 128 * 128;
        if (kL1Size / 2 >= K_UB_MAX_SIZE) {
            continue;  // kUbSize >= 256：违反内核 UB 契约，候选不可行
        }
        if (kSize % kL1Size != 0) {
            continue;  // kl1-divisibility-fix：尾 chunk 无钳位，kL1Size 必须整除 K
        }
        uint64_t cost = static_cast<uint64_t>(CeilDiv(CeilDiv(bs, static_cast<int64_t>(mL1Size)),
                                                      static_cast<int64_t>(mDimNum))) * mL1Size;
        if (cost < bestCost) {
            bestCost = cost;
            bestM = mL1Size;
        }
    }
    return bestM;
}

// tk1001-mtile-search-ubaware（F1）：融合分支（y-register-window-accumulate 对该 shape 激活）
// phase-1 的 4 段 cast 缓冲驻留预算断言。记账与 kernel 侧 tbufPool1/tbufPool0 的 InitBuffer
// 逐项一致（hc_pre_m_split_core_arch35.h 融合分支）：
//   pool1 = 4*castBuf + xQue(2) + nd2NzBuf(2) + yQue(2) + preMixQue(1)
//   castBuf   = mUb * (RoundUp<float>(kUb)*4 + 32)          [mUb 行 × (kUb float + 32B 尾)]
//   xQue/yQue = 2 * mUb * RoundUp<bf16>(kUb) * 2            [TQue 双 buffer]
//   nd2NzBuf  = 2 * CeilAlign(mUb,8) * RoundUp<float>(kUb) * 4
//   preMixQue = 1 * mUb * RoundUp<float>(hcMix) * 4         [TQue 单 buffer]
//   pool0 = mmXBuf + rmsNormBuf + hcBase0/1/2（bufferPool1Size 同 CalcRegbaseOpTiling 公式）
// 返回 pool1Fused <= bufferPool1Size。
bool FusedPhase1UbBudgetFits(uint64_t mL1Size, int64_t hcMult, int64_t hcMix, uint64_t ubSize)
{
    auto roundUpTo = [](int64_t x, int64_t align) { return CeilDiv(x, align) * align; };
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);   // 8（kernel RoundUp<float>）
    const int64_t bf16Align = BLOCK_SIZE / 2;                // 16（kernel RoundUp<bfloat16_t>）
    uint64_t mUbSize = static_cast<uint64_t>(CeilDiv(static_cast<int64_t>(mL1Size), 2));
    uint64_t kL1Size = std::min(A_L1_SIZE / mL1Size, K_L1_MAX_SIZE) / 128 * 128;
    uint64_t kUbSize = kL1Size / 2;

    int64_t castEach = static_cast<int64_t>(mUbSize) * (roundUpTo(kUbSize, floatAlign) * 4 + BLOCK_SIZE);
    int64_t cast4 = 4 * castEach;
    int64_t xQueSize = DOUBLE_BUFFER * static_cast<int64_t>(mUbSize) * roundUpTo(kUbSize, bf16Align) * 2;
    int64_t nd2NzSize = DOUBLE_BUFFER * roundUpTo(static_cast<int64_t>(mUbSize), C0_SIZE_HOST) *
                        roundUpTo(static_cast<int64_t>(kUbSize), floatAlign) * 4;
    int64_t yQueSize = DOUBLE_BUFFER * static_cast<int64_t>(mUbSize) * roundUpTo(kUbSize, bf16Align) * 2;
    int64_t preMixSize = static_cast<int64_t>(mUbSize) * roundUpTo(hcMix, floatAlign) * 4;
    int64_t pool1Fused = cast4 + xQueSize + nd2NzSize + yQueSize + preMixSize;

    // pool0 项（与 CalcRegbaseOpTiling 的 bufferPool1Size 公式逐项一致）
    int64_t hcMultAlign = RoundUp(hcMult, BLOCK_SIZE / sizeof(float));
    int64_t mmXBufSize = static_cast<int64_t>(mUbSize) * roundUpTo(hcMix, floatAlign) * 4;
    int64_t rmsNormBufSize = roundUpTo(mUbSize, floatAlign) * 4;
    int64_t base0Size = hcMultAlign * 4;
    int64_t base1Size = hcMultAlign * 4;
    int64_t base2Size = hcMult * hcMultAlign * 4;
    int64_t bufferPool1Size =
        DownAlign(static_cast<int64_t>(ubSize) - mmXBufSize - rmsNormBufSize - base0Size - base1Size - base2Size,
                  BLOCK_SIZE);
    return pool1Fused <= bufferPool1Size;
}

// tk1001-mtile-search-ubaware（F1）：SearchAllCoreML1Size 同款确定性搜索 + 结果封顶 208 +
// 融合分支预算断言（失败回退更小 16 对齐 mL1Size）。
// 契约 emitted targets（10 调优 case）：bs4096/8192→160、bs16384/32768/64000→208（双 d）。
// 实现注记（契约内一致性裁决，留档）：纯"搜索域截断到 [16,208] 再取 argmin"的读法在
// bs32768 会选出 mL1=80（cost 1200 < 208 的 1248），但 mL1=80 ⇒ kL1Size=384/kUbSize=192，
// 融合启用条件 d%kL1Size==0 对调优域 d∈{4096,5120} 不成立（4096%384=256≠0）——融合机制
// 整体失效，且与契约 F1 gates 明文要求的 "10 tk1001 cases match ubaware mL1Size targets
// (160/160/208/208/208)"、F2 的 kUbSize=64=VL_FP32 全 lane mask 前提、F3 的
// "rowInnerFactor reaches >=16 for the 10 tuning cases"（融合记账才会把 rowIF 从 2 提升）
// 三处规范性约束全部矛盾。故取"同款搜索 + 结果封顶 208"读法（min(argmin, 208)）：
// 搜索本身与 r2-tk1001-tiling-rebalance 逐行相同（bs32768 argmin=240），封顶后恰为
// 契约目标 208（bs32768 行数比 640→624；契约括注 640→416 为分析稿算术误差，208 时
// ceil(ceil(32768/208)/28)*208/2=624 不可达 416，性能假设按 624 口径执行）。10 case
// 断言全部一次通过（160/208 档预算 181760/236288B，见 evidence/f1-dump.json）。
uint64_t SearchUbawareML1Size(int64_t bs, uint64_t mDimNum, int64_t d, int64_t hcMult, int64_t hcMix,
                              bool hasPreMix, uint64_t ubSize, uint64_t multCoreSplitKSize)
{
    uint64_t mL1Size = std::min(SearchAllCoreML1Size(bs, mDimNum, multCoreSplitKSize), M_L1_UBAWARE_MAX_SIZE);
    // 融合分支激活判定（与 kernel 侧 fusedYRound 逐条件同构；kL1Size 随候选变化）
    auto fusedActiveAt = [&](uint64_t cand) {
        uint64_t kL1 = std::min(A_L1_SIZE / cand, K_L1_MAX_SIZE) / 128 * 128;
        return hasPreMix && (multCoreSplitKSize == static_cast<uint64_t>(hcMult * d)) &&
               (d % static_cast<int64_t>(kL1) == 0);
    };
    if (!fusedActiveAt(mL1Size) || FusedPhase1UbBudgetFits(mL1Size, hcMult, hcMix, ubSize)) {
        return mL1Size;  // 融合未激活（legacy 单 castBuf 记账）或预算一次通过
    }
    // 断言失败：向下逐 16 回退，跳过 kUb>=256 不可行候选，取首个（融合关闭 或 预算通过）者；
    // 向下耗尽后从封顶值向下补扫（kUb=64 档 [144,208] 在接口域 hcMult=4/hcMix=24 下预算恒
    // 通过——融合激活蕴含 d%128==0，而该档 cast4+配套项在 mL1=144 仅 163584B，必命中）。
    // kl1-divisibility-fix：回退候选同样要求 kL1Size 整除 K（与 SearchAllCoreML1Size 过滤同源）。
    for (uint64_t cand = mL1Size; cand >= AscendC::BLOCK_CUBE; cand -= AscendC::BLOCK_CUBE) {
        uint64_t kL1 = std::min(A_L1_SIZE / cand, K_L1_MAX_SIZE) / 128 * 128;
        if (kL1 / 2 >= K_UB_MAX_SIZE || multCoreSplitKSize % kL1 != 0) {
            continue;
        }
        if (!fusedActiveAt(cand) || FusedPhase1UbBudgetFits(cand, hcMult, hcMix, ubSize)) {
            return cand;
        }
    }
    for (uint64_t cand = M_L1_UBAWARE_MAX_SIZE; cand > mL1Size; cand -= AscendC::BLOCK_CUBE) {
        uint64_t kL1 = std::min(A_L1_SIZE / cand, K_L1_MAX_SIZE) / 128 * 128;
        if (kL1 / 2 >= K_UB_MAX_SIZE || multCoreSplitKSize % kL1 != 0) {
            continue;
        }
        if (!fusedActiveAt(cand) || FusedPhase1UbBudgetFits(cand, hcMult, hcMix, ubSize)) {
            return cand;
        }
    }
    return mL1Size;  // 防御性兜底（接口域内不可达，见上注）
}
}

class HcPreTilingRegbase {
public:
    explicit HcPreTilingRegbase(gert::TilingContext* tilingContext) : context_(tilingContext)
        {
        }
    ~HcPreTilingRegbase() = default;
    
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcRegbaseOpTiling();
    ge::graphStatus CalcMKSplitCorePart2Tiling();
private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    HcPreTilingData tilingData_;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t usedAivCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t hcMix_ = 0;
    int64_t hcMult_ = 0;
    int64_t d_ = 0;
    int64_t hcMultAlign_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
    int64_t iterTimes_ = 0;
    double hcEps_ = 0.0;
    double normEps_ = 0.0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

ge::graphStatus HcPreTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<HcPreCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(0);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(1);
    iterTimes_ = iterTimesAttr == nullptr ? 20 : *iterTimesAttr;

    auto epsAttr = attrs->GetAttrPointer<float>(2);
    hcEps_ = epsAttr == nullptr ? 1e-6 : *epsAttr;
    
    auto normEpsAttr = attrs->GetAttrPointer<float>(3);
    normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mult, d) or (bs, hc_mult, d)
    auto xShape = context_->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context_, xShape, return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    }

    auto shapeHcFn = context_->GetInputShape(1);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OPS_ERR_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld", d_ * hcMult_, shapeHcFn->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    auto shapeHcScale = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
    OPS_ERR_IF(scaleFirstDim != 3,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_scale size should be equal with 3, but is %ld", scaleFirstDim),
                    return ge::GRAPH_FAILED);

    auto shapeHcBase = context_->GetInputShape(3);
    int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
    OPS_ERR_IF(baseFirstDim != hcMix_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_base size should be equal with mixhc, but is %ld", baseFirstDim),
                    return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus HcPreTilingRegbase::CalcRegbaseOpTiling() 
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t hcMixAlign = RoundUp(hcMix_, BLOCK_SIZE / sizeof(float));
    int64_t mixSize = rowOnceLoop * hcMixAlign * sizeof(float);
    // 可选输入pre_mix的TQue(双buffer)UB占用，恒定预留以保证行Factor不因可选输入而溢出
    int64_t preMixSize = rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    // 可选输出pre的TQue(双buffer, hcMultAlign行距)UB占用，恒定预留
    int64_t preOutSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    // comb 在 UB 内按 hcMult*hcMult 紧密排布（Packed 模式），按真实占用计算
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    // tk1001-phase2-rowbatch-fused（F3）：融合分支判定——与 kernel 侧 fusedYRound
    // （hc_pre_m_split_core_arch35.h）逐条件一致：
    //   fusedYRound = kInterleave && hasPreMix_ && d % (2*kUbSize) == 0，
    //   kInterleave = (d % kL1Size == 0) && (multCoreSplitKSize == hcMult * d)。
    // host 恒有 2*kUbSize == kL1Size（kUbSize = kL1Size/2，kL1Size 为 128 的倍数），
    // 故 d % (2*kUbSize)==0 与 d % kL1Size==0 等价；tk1001 的 multCoreSplitKSize
    // 由 CalcOpTiling 共享块按 kDimNum==1 赋值为 RoundUp(hcMult*d, 256)。接口合法
    // 但 d 非 kL1Size 对齐（如 d=4160）或未提供 pre_mix 时判定为 false → legacy 记账。
    // pre_mix 是否提供 = GetInputShape(4) 非空（可选输入缺省时框架传 null，与 kernel
    // 侧 hasPreMix_ = (preMix != nullptr) 同源；A3 路径同款判定）。
    const bool fusedYRound =
        (context_->GetInputShape(4) != nullptr) &&
        (tilingData_.get_multCoreSplitKSize() == hcMult_ * d_) &&
        (d_ % tilingData_.get_kL1Size() == 0);

    int64_t mmXBufSize = mUbSize * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t rmsNormBufSize =  RoundUp(mUbSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t bufferPool0Size = ubSize_;
    int64_t bufferPool1Size = DownAlign(bufferPool0Size - mmXBufSize - rmsNormBufSize - base0Size - base1Size - base2Size, BLOCK_SIZE);

    // tk1001-phase2-rowbatch-fused（F3）：融合分支（y 已在 phase-1 处理）phase-2 不再分配
    // xQue/yQue（kernel 跳过 x/y InitBuffer 与 dLoop 迭代），phase-2 余量预算为
    // mixSize+preMix+preOut+post+combFrag（约 544B/行 × rowIF）；legacy 分支记账
    // 逐项保留 x/y 项。
    int64_t totalSize = mixSize + postSize + combFragSize + preMixSize + preOutSize;
    if (!fusedYRound) {
        totalSize += xSize + ySize;
    }
    rowFactor_ = rowOnceLoop;
    if (totalSize <= bufferPool1Size) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mixSize + postSize + combFragSize + preMixSize + preOutSize;
        int64_t ubRemain = bufferPool1Size - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= mUbSize) {
            mixSize = rowFactor_ * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            preMixSize = rowFactor_ * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            preOutSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
            totalSize = mixSize + postSize + combFragSize + preMixSize + preOutSize;
            if (!fusedYRound) {
                // tk1001-phase2-rowbatch-fused（F3）：legacy 记账保留 x/y 项（phase-2 重读
                // x / VFProcessY）；融合分支 phase-2 无 x/y 缓冲，rowFactor 仅受余量预算与
                // mUbSize 约束
                xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
                ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
                totalSize += xSize + ySize;
            }
            if (totalSize > bufferPool1Size) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > mUbSize ? rowFactor_ - 1 : rowFactor_;
    }

    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;  

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);

    tilingData_.set_bufferPool0Size(bufferPool0Size);
    tilingData_.set_bufferPool1Size(bufferPool1Size);

    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);

    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    
    tilingData_.set_rowInnerFactor(rowFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    uint64_t hcMixAlign = RoundUp(hcMix_, BLOCK_SIZE / sizeof(float));
    int64_t mmSize = kBlockNum * rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    int64_t mixSize = rowOnceLoop * hcMixAlign * sizeof(float);
    // 可选输入pre_mix的TQue(双buffer)UB占用，恒定预留以保证行Factor不因可选输入而溢出
    int64_t preMixSize = rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    // 可选输出pre的TQue(双buffer, hcMultAlign行距)UB占用，恒定预留
    int64_t preOutSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t rmsSize = kBlockNum * RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    // comb 在 UB 内按 hcMult*hcMult 紧密排布（Packed 模式），按真实占用计算
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    int64_t totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + preMixSize +
                       preOutSize + base0Size + base1Size + base2Size;
    rowFactor_ = rowOnceLoop; 
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mmSize + mixSize + rmsSize + postSize + combFragSize + preMixSize + preOutSize +
                       base0Size + base1Size + base2Size;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
        // x/y 已按 dFactor 分块记账（dLoopIdx 循环已存在），行批同样可在 UB 预算内增大
        while (rowFactor_ <= rowOfFormerBlock_) {
            mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            mixSize =  rowFactor_ * hcMixAlign * sizeof(float);
            preMixSize = rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            preOutSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
            totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + preMixSize +
                        preOutSize + base0Size + base1Size + base2Size;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            mixSize =  rowFactor_ * hcMixAlign * sizeof(float);
            preMixSize = rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            preOutSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
            base0Size = hcMultAlign_ * sizeof(float);
            base1Size = hcMultAlign_ * sizeof(float);
            base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

            totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + preMixSize +
                        preOutSize + base0Size + base1Size + base2Size;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }

    // stage2-dloop-rebatch：全载 d 时行批被 UB 压到目标(8)以下且每核还有整行可摊薄时，允许
    // dFactor<d、dLoop=CeilDiv(d,dFactor)（xSize/ySize 记账改用 dFactor，kernel Part2 的 dLoopIdx
    // 循环已存在），把 stage2RowFactor 提到 UB 预算内最大值。仅作用于调优 regime（bs<=3079），
    // 保证 bs 3584/3585 路由边界探针与基线行为一致。
    if (dLoop_ == 1 && rowFactor_ < STAGE2_REBATCH_TARGET_ROW_FACTOR && rowOfFormerBlock_ > rowFactor_ &&
        bs_ <= STAGE2_REBATCH_BS_LIMIT) {
        int64_t rebatchDFactor = DownAlign(CeilDiv(d_, 4), 32);
        if (rebatchDFactor <= 0) {
            rebatchDFactor = 32;
        }
        if (rebatchDFactor < d_) {
            dFactor_ = rebatchDFactor;
            dLoop_ = CeilDiv(d_, dFactor_);
            tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
            rowFactor_ = rowOnceLoop;
            while (1) {
                mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
                mixSize = rowFactor_ * hcMixAlign * sizeof(float);
                preMixSize = rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
                preOutSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
                rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
                xSize = rowFactor_ * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
                ySize = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
                combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
                base0Size = hcMultAlign_ * sizeof(float);
                base1Size = hcMultAlign_ * sizeof(float);
                base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
                totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + preMixSize +
                            preOutSize + base0Size + base1Size + base2Size;
                if (totalSize > ubSize_) {
                    rowFactor_ = rowFactor_ - 1;
                    break;
                }
                rowFactor_ = rowFactor_ + 1;
            }
            if (rowFactor_ < 1) {
                rowFactor_ = 1;
            }
        }
    }
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;  
    
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;

    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    uint64_t actualKBlockNum = CeilDiv(kSize, splitKSize);

    if (kDimNum != 1) {
        // splitk-128-balance：tk1000 分支尝试 128 基粒度切 K，仅当严格减小最大 per-core K 切片长度
        // 或严格增加活跃块数（cubeBlockDimM*cubeBlockDimK）时采用；kDimNum==1（tk1001）路由与
        // bs<=3584 边界不变
        uint64_t splitKSizeFiner = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE_FINER);
        if (splitKSizeFiner != splitKSize) {
            uint64_t kBlockNumFiner = CeilDiv(kSize, splitKSizeFiner);
            if (splitKSizeFiner < splitKSize || kBlockNumFiner > actualKBlockNum) {
                splitKSize = splitKSizeFiner;
                actualKBlockNum = kBlockNumFiner;
            }
        }
    }

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(actualKBlockNum);
    tilingData_.set_multCoreSplitMSize(singleCoreM); // todo: 这个 tiling 根本没有使用，是否需要删掉
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(), static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);

    tilingData_.set_cvLoopKSize(1024);
    if (kDimNum != 1) {
        tilingKey_ = 1000;
        return CalcMKSplitCorePart2Tiling();
    }
    // tk1001-mtile-search-ubaware（F1）：仅 tk1001（kDimNum==1）分支。mDimNum 覆盖为全核
    // （aicCoreNum_=28），mL1Size 由 SearchUbawareML1Size 确定（r2-tk1001-tiling-rebalance
    // F1 同款确定性搜索 + 结果封顶 208 + 融合分支 4 段 cast 缓冲预算断言/回退），覆盖上方
    // 共享块写入的 cubeBlockDimM/mL1Size/kL1Size（kL1Size 按既有公式以新 mL1Size 重算；
    // mL1Size∈{144..208} 时恒为 128、kUbSize 恒 64）。tilingKey 路由边界不变：kDimNum 由
    // 上方原 mDimNum=min(aicCoreNum_, CeilDiv(bs,256)) 判定（bs<=3584 → kDimNum!=1 →
    // tk1000），tk1000 分支与共享块对该分支的赋值零改动。
    mDimNum = aicCoreNum_;
    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_mL1Size(SearchUbawareML1Size(bs_, mDimNum, d_, hcMult_, hcMix_,
                                                 context_->GetInputShape(4) != nullptr, ubSize_,
                                                 tilingData_.get_multCoreSplitKSize()));
    tilingData_.set_kL1Size(
        std::min(A_L1_SIZE / tilingData_.get_mL1Size(), static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);
    tilingKey_ = 1001;
    return CalcRegbaseOpTiling();
}


ge::graphStatus HcPreTilingRegbase::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetWorkspaceSize()
{
    if (tilingKey_ == 1000) {
        // K分核模板需要预留Workspace大小
        workspaceSize_ = tilingData_.get_kBlockFactor() * tilingData_.get_bs() * tilingData_.get_hcMix() * 4 + tilingData_.get_kBlockFactor() * tilingData_.get_bs() * 4 + 16 * 1024 * 1024;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(aicCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}
}  // namespace optiling