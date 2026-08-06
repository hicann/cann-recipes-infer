# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K3 Phase-1 ``block_attn_res_prepare`` CANNDSL implementation.

整网接口沿有效历史 block 轴做多 Query 注意力残差聚合：

    qe_s     = effective_queries[s]                     整网预先融合 q ⊙ g
    logit_sn = (qe_s · V_n) / sqrt(mean_d V_n^2 + eps)  RMSNorm 后点积（rms 折进打分,不物化 K）
    max      = max_n(logit_n)                            每个 token 的稳定 softmax 最大值
    sum      = Σ_n exp(logit_n - max)                   每个 token 的稳定 softmax 指数和
    numerator = Σ_n exp(logit_n - max) · V_n             未归一化加权和

纯 AIV（Vector）核，D 进 lane（段循环）。Prefill 按 token 分核，每个 token 把固定
容量 [max_blocks,D] 搬入 UB 并复用给全部 slots；decode 的小 token shape 按
token×slot 分核，用少量 V 重读换取多核并行。valid_blocks 和 eps 从设备标量
Tensor 读取，不产生 host 同步；无效 block lane 不参与
max/sum/numerator。核内 fp32 计算，numerator/max/sum 均固定写回 fp32。
三个输出直接作为 ``block_attn_res_update`` 的 Phase-1 输入。
"""

import cannbotdsl
import torch

from cannbotdsl import dtypes
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.runtime import from_torch_npu
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.control_flow import range as dsl_range
from cannbotdsl.core.frontend.compiler import compile_function
from cannbotdsl.buffer import Buffer
from cannbotdsl.dtypes import (
    bfloat16 as BFloat16,
    float16 as Float16,
    float32 as Float32,
)
from cannbotdsl.integer import Int64
from cannbotdsl.tensor import local_slice, tile_view, mem_copy
from cannbotdsl.typing.types import MemLoc, Tensor
from cannbotdsl.vf import vf
from cannbotdsl.raw_reg import (
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp_sub,
    vload,
    vload_brc,
    vload_unpack,
    vmem_bar,
    vmerge,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
)

VL = 64
DEFAULT_BLOCK_NUM = 64
UB_BYTES = 240 * 1024  # Arch35 UB 256KB,留 ~16KB 给对齐/保留
# K3 decode 网络 shape 为 T=1/2。仅对这两类 shape 使用 token×slot
# 并行，避免改变 T>=4 的 prefill 路径及其 V 复用特性。
DECODE_SLOT_PARALLEL_MAX_TOKENS = 2

_TORCH_TO_DSL = {torch.bfloat16: BFloat16, torch.float16: Float16, torch.float32: Float32}
_COMPILED_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _use_decode_slot_parallel(
    tokens: int,
    slots: int,
    required_ub: int,
) -> bool:
    """Return whether a hold-capable shape should parallelize token×slot.

    The stream kernel already uses depth-2 D tiling and is left unchanged.  The
    decode path deliberately targets only K3's T=1/2 shapes so prefill keeps
    loading each token's V once and reusing it across all query slots.
    """
    return (
        0 < int(tokens) <= DECODE_SLOT_PARALLEL_MAX_TOKENS
        and int(slots) > 1
        and int(required_ub) <= UB_BYTES
    )


def _launch_block_num(mode: str, tokens: int, slots: int) -> int:
    """Launch only as many blocks as the selected path can use."""
    work_items = int(tokens) * int(slots) if mode == "decode" else int(tokens)
    return max(1, min(DEFAULT_BLOCK_NUM, work_items))


# --- 打分 body：seg 4 路 unroll(4 部分累加器打破依赖链)。logit_n = (qe·V_n)/sqrt(mean V_n^2 + eps) --
def _score_body_16bit(v_ub, qe_ub, logits_ub, nplus1, num_seg, w, avg, eps, num_col):
    ng = (num_col // VL) // 4          # 4 路 unroll 只覆盖完整 VL 段;尾段（含非对齐 D 的偏段）走掩码
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            d0 = vdup_scalar(0.0, Float32, mask=full); d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full); d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full); s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full); s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vcast(vload_unpack(v_ub, vb, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_ub, vb + VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_ub, vb + 2 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_ub, vb + 3 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                q0 = vload(qe_ub, qb); q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL); q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full); d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full); d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full); s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full); s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(Int64(num_col) - seg * VL, elem_bits=32)   # 尾段 partial(非对齐 D)/满(对齐)
                off = base_n + seg * VL
                xf = vcast(vload_unpack(v_ub, off, mode=UnpackMode.B16_TO_B32), Float32, mask=mask)
                qf = vload(qe_ub, seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=mask), mask=full); ssq = vadd(ssq, vmul(xf, xf, mask=mask), mask=full)
            denom = vsqrt(vadds(vmuls(vreduce_sum(ssq, mask=full), avg, mask=full), eps, mask=full), mask=full)
            vstore_first(logits_ub, n, vdiv(vreduce_sum(dot, mask=full), denom, mask=full))


def _score_body_fp32(v_ub, qe_ub, logits_ub, nplus1, num_seg, w, avg, eps, num_col):
    ng = (num_col // VL) // 4          # 4 路 unroll 只覆盖完整 VL 段;尾段（含非对齐 D 的偏段）走掩码
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            d0 = vdup_scalar(0.0, Float32, mask=full); d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full); d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full); s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full); s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vload(v_ub, vb); x1 = vload(v_ub, vb + VL)
                x2 = vload(v_ub, vb + 2 * VL); x3 = vload(v_ub, vb + 3 * VL)
                q0 = vload(qe_ub, qb); q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL); q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full); d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full); d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full); s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full); s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(Int64(num_col) - seg * VL, elem_bits=32)   # 尾段 partial(非对齐 D)/满(对齐)
                off = base_n + seg * VL
                xf = vload(v_ub, off); qf = vload(qe_ub, seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=mask), mask=full); ssq = vadd(ssq, vmul(xf, xf, mask=mask), mask=full)
            denom = vsqrt(vadds(vmuls(vreduce_sum(ssq, mask=full), avg, mask=full), eps, mask=full), mask=full)
            vstore_first(logits_ub, n, vdiv(vreduce_sum(dot, mask=full), denom, mask=full))


# --- 聚合 body：候选 n 4 路 unroll(4 部分累加器)。h[seg] = Σ_n w_n · V_n[seg] --------------------------------
def _agg_body_16bit(v_ub, logits_ub, h_ub, nplus1, num_seg, w):
    ng = nplus1 // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            a0 = vdup_scalar(0.0, Float32, mask=full); a1 = vdup_scalar(0.0, Float32, mask=full)
            a2 = vdup_scalar(0.0, Float32, mask=full); a3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                nb = it * 4
                w0 = vload_brc(logits_ub, nb); w1 = vload_brc(logits_ub, nb + 1)
                w2 = vload_brc(logits_ub, nb + 2); w3 = vload_brc(logits_ub, nb + 3)
                x0 = vcast(vload_unpack(v_ub, nb * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_ub, (nb + 1) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_ub, (nb + 2) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_ub, (nb + 3) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                a0 = vadd(a0, vmul(x0, w0, mask=full), mask=full); a1 = vadd(a1, vmul(x1, w1, mask=full), mask=full)
                a2 = vadd(a2, vmul(x2, w2, mask=full), mask=full); a3 = vadd(a3, vmul(x3, w3, mask=full), mask=full)
            acc = vadd(vadd(a0, a1, mask=full), vadd(a2, a3, mask=full), mask=full)
            for n in range(Int64(4 * ng), Int64(nplus1)):
                wn = vload_brc(logits_ub, n)
                xf = vcast(vload_unpack(v_ub, n * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                acc = vadd(acc, vmul(xf, wn, mask=full), mask=full)
            vstore(h_ub, off, acc, full)


def _agg_body_fp32(v_ub, logits_ub, h_ub, nplus1, num_seg, w):
    ng = nplus1 // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            a0 = vdup_scalar(0.0, Float32, mask=full); a1 = vdup_scalar(0.0, Float32, mask=full)
            a2 = vdup_scalar(0.0, Float32, mask=full); a3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                nb = it * 4
                w0 = vload_brc(logits_ub, nb); w1 = vload_brc(logits_ub, nb + 1)
                w2 = vload_brc(logits_ub, nb + 2); w3 = vload_brc(logits_ub, nb + 3)
                x0 = vload(v_ub, nb * w + off); x1 = vload(v_ub, (nb + 1) * w + off)
                x2 = vload(v_ub, (nb + 2) * w + off); x3 = vload(v_ub, (nb + 3) * w + off)
                a0 = vadd(a0, vmul(x0, w0, mask=full), mask=full); a1 = vadd(a1, vmul(x1, w1, mask=full), mask=full)
                a2 = vadd(a2, vmul(x2, w2, mask=full), mask=full); a3 = vadd(a3, vmul(x3, w3, mask=full), mask=full)
            acc = vadd(vadd(a0, a1, mask=full), vadd(a2, a3, mask=full), mask=full)
            for n in range(Int64(4 * ng), Int64(nplus1)):
                wn = vload_brc(logits_ub, n)
                xf = vload(v_ub, n * w + off)
                acc = vadd(acc, vmul(xf, wn, mask=full), mask=full)
            vstore(h_ub, off, acc, full)


# --- 流式打分 body：只算「当前 Dt 块」每候选的部分 dot/ssq,写 pdot/pssq 的 lane n（跨块累加见 kernel）----
#   大 D 装不下 hold-all 时用:分 Dt 块流式过 D,dot/ssq 是标量(每候选)可跨块累加。qe 常驻[D],读偏移 qe_off。
#   Dt 为 VL 整数倍(host 保证),块内无非对齐尾;不除 rms(要全 D 的 ssq),留到 finalize。
def _score_stream_body_16bit(v_dt, qe_ub, pdot, pssq, nplus1, num_seg_dt, w_dt, qe_off):
    ng = num_seg_dt // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d0 = vdup_scalar(0.0, Float32, mask=full); d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full); d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full); s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full); s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL); qb = qe_off + it * (4 * VL)
                x0 = vcast(vload_unpack(v_dt, vb, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_dt, vb + VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_dt, vb + 2 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_dt, vb + 3 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                q0 = vload(qe_ub, qb); q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL); q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full); d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full); d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full); s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full); s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg_dt)):
                off = base_n + seg * VL
                xf = vcast(vload_unpack(v_dt, off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                qf = vload(qe_ub, qe_off + seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=full), mask=full); ssq = vadd(ssq, vmul(xf, xf, mask=full), mask=full)
            vstore_first(pdot, n, vreduce_sum(dot, mask=full))
            vstore_first(pssq, n, vreduce_sum(ssq, mask=full))


def _score_stream_body_fp32(v_dt, qe_ub, pdot, pssq, nplus1, num_seg_dt, w_dt, qe_off):
    ng = num_seg_dt // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d0 = vdup_scalar(0.0, Float32, mask=full); d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full); d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full); s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full); s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL); qb = qe_off + it * (4 * VL)
                x0 = vload(v_dt, vb); x1 = vload(v_dt, vb + VL)
                x2 = vload(v_dt, vb + 2 * VL); x3 = vload(v_dt, vb + 3 * VL)
                q0 = vload(qe_ub, qb); q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL); q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full); d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full); d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full); s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full); s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg_dt)):
                off = base_n + seg * VL
                xf = vload(v_dt, off); qf = vload(qe_ub, qe_off + seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=full), mask=full); ssq = vadd(ssq, vmul(xf, xf, mask=full), mask=full)
            vstore_first(pdot, n, vreduce_sum(dot, mask=full))
            vstore_first(pssq, n, vreduce_sum(ssq, mask=full))


_score_body_16bit = compile_function(
    _score_body_16bit, enable_preprocessor=True,
).function
_score_body_fp32 = compile_function(
    _score_body_fp32, enable_preprocessor=True,
).function
_score_stream_body_16bit = compile_function(
    _score_stream_body_16bit, enable_preprocessor=True,
).function
_score_stream_body_fp32 = compile_function(
    _score_stream_body_fp32, enable_preprocessor=True,
).function
_agg_body_16bit = compile_function(
    _agg_body_16bit, enable_preprocessor=True,
).function
_agg_body_fp32 = compile_function(
    _agg_body_fp32, enable_preprocessor=True,
).function


class BlockAttnResPrepareVector:
    """Whole-network multi-query AIV buffers and per-token primitives."""

    def __init__(self, nplus1: int, d: int, eps, dtype=BFloat16,
                 dt: int = 0):
        self.nplus1 = int(nplus1)
        self.d = int(d)
        self.eps = eps
        self.avg = 1.0 / self.d
        self.is16 = dtype in (BFloat16, Float16)
        self.num_seg = _ceil_div(self.d, VL)
        w = self.num_seg * VL
        self.w = w
        self.dt = int(dt)                 # 流式 Dt 分块宽（>0 时走大 D 流式:候选不全驻留,分块过 D）
        state = dtype if self.is16 else Float32
        # 每次只驻留 multi-query 输入中的当前 fp32 effective query。
        self.effective_q_ub = Channel(
            MemLoc.UB, shape=(1, w), dtype=Float32, depth=1,
        )
        if self.dt > 0:
            # 大 D 流式:只驻留一个 Dt 块的 N+1 候选（(N+1)·Dt,与 D 解耦）。dot/ssq 标量跨块累加。
            self.num_dt = self.d // self.dt          # 块数（host 保证 dt|D 且 dt 为 VL 整数倍）
            self.num_seg_dt = self.dt // VL
            # 两槽 FIFO：MTE2 预取下一 D 块时，Vector 消费当前块。
            self.v_dt = Channel(MemLoc.UB, shape=(1, self.nplus1 * self.dt), dtype=state, depth=2)
            self.dot_acc_ub = Buffer(MemLoc.UB, (1, VL), Float32)   # 候选 n 的 dot 累加(lane n)
            self.ssq_acc_ub = Buffer(MemLoc.UB, (1, VL), Float32)
            self.pdot_ub = Buffer(MemLoc.UB, (1, VL), Float32)      # 本块 partial(lane n)
            self.pssq_ub = Buffer(MemLoc.UB, (1, VL), Float32)
            self.h_dt_ub = Channel(MemLoc.UB, shape=(1, self.dt), dtype=Float32, depth=2)
        else:
            # 一个 token 的全部历史 block 常驻 UB，供所有 query slot 复用。
            self.v_ub = Channel(
                MemLoc.UB, shape=(1, self.nplus1 * w), dtype=state, depth=1,
            )
        self.logits_ub = Buffer(MemLoc.UB, (1, VL), Float32)        # VF 内部 logit -> stable exp
        # softmax 原本就会计算 m/s；用独立 Channel 直接写回，不重算也不与 logits 复用冲突。
        self.stats_ub = Channel(
            MemLoc.UB, shape=(1, VL), dtype=Float32, depth=1,
        )
        if self.dt == 0:
            self.h_ub = Channel(
                MemLoc.UB, shape=(1, w), dtype=Float32, depth=1,
            )

    def load_effective_query(self, gm_effective_queries, slot_idx):
        """Load one fp32 prefused query [D] into its UB Channel."""
        q_slot = self.effective_q_ub.acquire()
        mem_copy(
            local_slice(q_slot, (1, self.d), offset=0),
            tile_view(gm_effective_queries, (1, self.d), (slot_idx, Int64(0))),
        )
        self.effective_q_ub.commit(q_slot)

    # --- GM -> UB：载入当前 token 的 N+1 个候选到指定槽（vf 外）------------------------------------
    @jit
    def load_row(self, gm_v2d, row, valid_blocks):
        d, w = self.d, self.w
        slot = self.v_ub.acquire()
        # The physical buffer has max_blocks rows, but later score/aggregate
        # only read [0, valid_blocks).  Do not spend GM bandwidth on inactive
        # history slots, especially in early AttnRes blocks.
        for n in dsl_range(Int64(0), valid_blocks, Int64(1)):
            # 唯一公开布局为 [*M,N+1,D]，展平 GM 行号为 row*(N+1)+n。
            gm_row = row * Int64(self.nplus1) + Int64(n)
            src = tile_view(gm_v2d, (1, d), (gm_row, Int64(0)))
            # ``local_slice`` requires a compile-time byte offset.  Use a
            # dynamic tile coordinate for the block row, then trim the padded
            # UB width ``w`` back to the logical ``d`` columns.
            dst_block = tile_view(slot, (1, w), (Int64(0), n))
            mem_copy(tile_view(dst_block, (1, d), (Int64(0), Int64(0))), src)
        self.v_ub.commit(slot)

    # --- Stage 1：使用预融合 effective query 打分 -----------------------------------------------
    def score_with_query(self, v_ub, effective_q_ub, valid_blocks):
        if self.is16:
            _score_body_16bit(
                v_ub, effective_q_ub, self.logits_ub, valid_blocks,
                self.num_seg, self.w, self.avg, self.eps, self.d,
            )
        else:
            _score_body_fp32(
                v_ub, effective_q_ub, self.logits_ub, valid_blocks,
                self.num_seg, self.w, self.avg, self.eps, self.d,
            )
        # score body 写 logits_ub，后续 VF 读取前显式建立 VST→VLD 顺序。
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    # --- Stage 2：softmax over N+1（masked reduce/exp；无效 lane 精确置 0）--------------------------
    def softmax(self, valid_blocks):
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            mask_n, _ = update_mask(Int64(valid_blocks), elem_bits=32)
            v = vload(self.logits_ub, 0)
            # 用第 0 个有效 logit 填充无效 lane；重复有效元素不改变 max，因此归约可用 full mask。
            # exp 的无效 lane 再动态填成 m-80，避开极端输入，最后精确 merge 为 0。
            lane0 = vload_brc(self.logits_ub, 0)
            max_input = vmerge(mask_n, v, lane0)
            m = vdup_lane0(vreduce_max(max_input, mask=full), mask=full)
            exp_floor = vadds(m, -80.0, mask=full)
            exp_input = vmerge(mask_n, v, exp_floor)
            e_all = vexp_sub(exp_input, m, mask=full)
            zero = vdup_scalar(0.0, Float32, mask=full)
            e = vmerge(mask_n, e_all, zero)
            s = vdup_lane0(vreduce_sum(e, mask=full), mask=full)
            vstore_first(stats_slot, 0, m)
            # MTE 小块搬运的 UB 源地址需 32B 对齐；s 放 lane 8（fp32 偏移 32B）。
            vstore_first(stats_slot, 8, s)
            # 保留未归一化的 stable exp，Stage 3 聚合得到
            # o_tilde = Σ exp(logit-max) * V，供 online-softmax merge 直接使用。
            vstore(self.logits_ub, 0, e, full)
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)

    # --- Stage 3：聚合（运行时 seg×候选 循环 body）-------------------------------------------------
    def aggregate(self, v_ub, valid_blocks):
        h_slot = self.h_ub.acquire()
        if self.is16:
            _agg_body_16bit(
                v_ub, self.logits_ub, h_slot, valid_blocks,
                self.num_seg, self.w,
            )
        else:
            _agg_body_fp32(
                v_ub, self.logits_ub, h_slot, valid_blocks,
                self.num_seg, self.w,
            )
        self.h_ub.commit(h_slot)

    def zero_result(self):
        """Materialize the empty-attention identity: h=0, max=-FLT_MAX, sum=0."""
        h_slot = self.h_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            for seg in range(self.num_seg):
                off = seg * VL
                vstore(h_slot, off, zero, full)
        self.h_ub.commit(h_slot)
        self.zero_stats()

    def zero_stats(self):
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            neg_flt_max = vdup_scalar(-3.4028234663852886e38, Float32, mask=full)
            vstore_first(stats_slot, 0, neg_flt_max)
            vstore_first(stats_slot, 8, zero)
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)

    def zero_stream_block(self, gm_h, row, db_idx):
        h_slot = self.h_dt_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            for seg in range(self.num_seg_dt):
                off = seg * VL
                vstore(h_slot, off, zero, full)
        self.h_dt_ub.commit(h_slot)
        h_cur = self.h_dt_ub.wait()
        mem_copy(
            tile_view(gm_h, (1, self.dt), (row, Int64(db_idx))),
            local_slice(h_cur, (1, self.dt), offset=0),
        )
        self.h_dt_ub.release(h_cur)

    def store_row(self, gm_h, row):
        d = self.d
        h_cur = self.h_ub.wait()
        mem_copy(tile_view(gm_h, (1, d), (row, Int64(0))), local_slice(h_cur, (1, d), offset=0))
        self.h_ub.release(h_cur)

    def store_stats(self, gm_max, gm_sum, row):
        """Write per-token softmax stats: max(logits), sum(exp(logits-max))."""
        stats_cur = self.stats_ub.wait()
        mem_copy(tile_view(gm_max, (1, 1), (row, Int64(0))),
                 local_slice(stats_cur, (1, 1), offset=0))
        mem_copy(tile_view(gm_sum, (1, 1), (row, Int64(0))),
                 local_slice(stats_cur, (1, 1), offset=32))
        self.stats_ub.release(stats_cur)

    # === 大 D 流式（候选不全驻留,分 Dt 块过 D;V 读两遍：score 一遍、agg 一遍）====================
    # 载入 token row 的 N+1 候选在 D 块 db_idx 的那段 [db_idx·Dt : +Dt] → v_dt（候选步长 Dt）。
    @jit
    def load_dt(self, gm_v2d, row, db_idx, valid_blocks):
        # tile_view 第 2 维坐标是「瓦片单位」(瓦片宽=dt)→ 传块索引 db_idx,元素列自动 = db_idx·dt。
        slot = self.v_dt.acquire()
        for n in dsl_range(Int64(0), valid_blocks, Int64(1)):
            gm_row = row * Int64(self.nplus1) + Int64(n)
            src = tile_view(gm_v2d, (1, self.dt), (gm_row, Int64(db_idx)))
            mem_copy(
                tile_view(slot, (1, self.dt), (Int64(0), n)),
                src,
            )
        self.v_dt.commit(slot)

    def score_stream_init(self):
        with vf(mode="raw"):
            full = full_mask(32)
            z = vdup_scalar(0.0, Float32, mask=full)
            vstore(self.dot_acc_ub, 0, z, full)
            vstore(self.ssq_acc_ub, 0, z, full)
            vmem_bar(mode="vst_vld")

    # 打分:算本块每候选 partial dot/ssq(lane n)→ 累加到常驻 dot/ssq_acc。qe_off 编译期(块循环 Python 展开)。
    def score_stream_block(self, v_dt, db_idx, effective_q_ub, valid_blocks):
        qe_off = db_idx * self.dt
        if self.is16:
            _score_stream_body_16bit(
                v_dt, effective_q_ub, self.pdot_ub, self.pssq_ub,
                valid_blocks, self.num_seg_dt, self.dt, qe_off,
            )
        else:
            _score_stream_body_fp32(
                v_dt, effective_q_ub, self.pdot_ub, self.pssq_ub,
                valid_blocks, self.num_seg_dt, self.dt, qe_off,
            )
        with vf(mode="raw"):
            full = full_mask(32)
            vmem_bar(mode="vst_vld")
            vstore(self.dot_acc_ub, 0, vadd(vload(self.dot_acc_ub, 0), vload(self.pdot_ub, 0), mask=full), full)
            vstore(self.ssq_acc_ub, 0, vadd(vload(self.ssq_acc_ub, 0), vload(self.pssq_ub, 0), mask=full), full)
            vmem_bar(mode="vst_vld")

    # 全块累加完:lane n = 候选 n 的完整 dot/ssq → logit_n = dot/sqrt(ssq·avg+eps)。
    def score_stream_finalize(self):
        with vf(mode="raw"):
            full = full_mask(32)
            dot = vload(self.dot_acc_ub, 0)
            ssq = vload(self.ssq_acc_ub, 0)
            denom = vsqrt(vadds(vmuls(ssq, self.avg, mask=full), self.eps, mask=full), mask=full)
            vstore(self.logits_ub, 0, vdiv(dot, denom, mask=full), full)
            vmem_bar(mode="vst_vld")

    # 聚合:本块 v_dt 复用 agg body 算 h_dt[Dt],写回 gm_h[row, db_idx·Dt : +Dt]。
    def agg_stream_block(self, v_dt, gm_h, row, db_idx, valid_blocks):
        h_slot = self.h_dt_ub.acquire()
        if self.is16:
            _agg_body_16bit(
                v_dt, self.logits_ub, h_slot, valid_blocks,
                self.num_seg_dt, self.dt,
            )
        else:
            _agg_body_fp32(
                v_dt, self.logits_ub, h_slot, valid_blocks,
                self.num_seg_dt, self.dt,
            )
        self.h_dt_ub.commit(h_slot)
        # 输出块 db_idx → gm_h[row, db_idx·dt : +dt]。列坐标同样是瓦片单位,传 db_idx。
        h_cur = self.h_dt_ub.wait()
        mem_copy(tile_view(gm_h, (1, self.dt), (row, Int64(db_idx))),
                 local_slice(h_cur, (1, self.dt), offset=0))
        self.h_dt_ub.release(h_cur)


@kernel
class block_attn_res_prepare_multi_query_kernel:
    """Whole-network single-launch multi-query kernel.

    Each core owns token rows.  A token's [max_blocks,D] values are loaded once and
    retained in UB while every effective query slot computes its own score,
    softmax statistics and unnormalized numerator.  valid_blocks and eps are read
    from device scalar tensors so the launch does not introduce host sync.
    """

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor, gm_eps: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        eps = gm_eps[(Int64(0),)]
        # K3 guarantees 0 <= valid_blocks <= max_blocks. Clamp in-kernel as a
        # final OOB guard without copying the scalar back to the host.
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)
        vector = BlockAttnResPrepareVector(
            self.max_blocks, self.d, eps, dtype=self.dtype,
        )
        num_row = Int64(self.num_row)
        logical = block_num
        if logical > num_row:
            logical = num_row
        rows_per_core = (num_row + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_row:
            row_end = num_row
        if block_idx < logical:
            for row in range(row_start, row_end):
                if valid_blocks > Int64(0):
                    vector.load_row(gm_v2d, row, valid_blocks)
                    v_cur = vector.v_ub.wait()
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        vector.load_effective_query(gm_effective_queries, slot_idx)
                        q_cur = vector.effective_q_ub.wait()
                        vector.score_with_query(
                            v_cur, q_cur, valid_blocks=valid_blocks,
                        )
                        vector.effective_q_ub.release(q_cur)
                        vector.softmax(valid_blocks)
                        vector.aggregate(v_cur, valid_blocks=valid_blocks)
                        out_row = slot_idx * num_row + row
                        vector.store_row(gm_h, out_row)
                        vector.store_stats(gm_max, gm_sum, out_row)
                    vector.v_ub.release(v_cur)
                else:
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        vector.zero_result()
                        out_row = slot_idx * num_row + row
                        vector.store_row(gm_h, out_row)
                        vector.store_stats(gm_max, gm_sum, out_row)


@kernel
class block_attn_res_prepare_multi_query_kernel_decode:
    """Small-token kernel: distribute flattened ``[slot, token]`` rows.

    The regular hold kernel owns a token row and serially reuses its resident V
    across every query.  That is efficient for prefill, but T=1/2 activates only
    one or two vector cores.  Decode instead exposes ``slots * tokens`` work
    items.  Each work item reloads one token's small ``[max_blocks,D]`` buffer,
    while up to 64 cores compute different query slots concurrently.
    """

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor, gm_eps: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        eps = gm_eps[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)

        vector = BlockAttnResPrepareVector(
            self.max_blocks, self.d, eps, dtype=self.dtype,
        )
        num_row = Int64(self.num_row)
        num_work = Int64(self.num_row * self.num_query)
        logical = block_num
        if logical > num_work:
            logical = num_work
        work_per_core = (num_work + logical - Int64(1)) // logical
        work_start = block_idx * work_per_core
        work_end = work_start + work_per_core
        if work_end > num_work:
            work_end = num_work

        if block_idx < logical:
            for out_row in range(work_start, work_end):
                # Output is slot-major: out_row = slot_idx * num_row + row.
                slot_idx = out_row // num_row
                row = out_row - slot_idx * num_row
                if valid_blocks > Int64(0):
                    vector.load_row(gm_v2d, row, valid_blocks)
                    v_cur = vector.v_ub.wait()
                    vector.load_effective_query(
                        gm_effective_queries, slot_idx,
                    )
                    q_cur = vector.effective_q_ub.wait()
                    vector.score_with_query(
                        v_cur, q_cur, valid_blocks=valid_blocks,
                    )
                    vector.effective_q_ub.release(q_cur)
                    vector.softmax(valid_blocks)
                    vector.aggregate(v_cur, valid_blocks=valid_blocks)
                    vector.v_ub.release(v_cur)
                    vector.store_row(gm_h, out_row)
                    vector.store_stats(gm_max, gm_sum, out_row)
                else:
                    vector.zero_result()
                    vector.store_row(gm_h, out_row)
                    vector.store_stats(gm_max, gm_sum, out_row)


@kernel
class block_attn_res_prepare_multi_query_kernel_stream:
    """D-tiled whole-network fallback when [max_blocks,D] cannot fit in UB."""

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dt: int, dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dt = int(dt)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor, gm_eps: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        eps = gm_eps[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)
        vector = BlockAttnResPrepareVector(
            self.max_blocks, self.d, eps, dtype=self.dtype, dt=self.dt,
        )
        num_row = Int64(self.num_row)
        num_dt = self.d // self.dt
        logical = block_num
        if logical > num_row:
            logical = num_row
        rows_per_core = (num_row + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_row:
            row_end = num_row
        if block_idx < logical:
            for row in range(row_start, row_end):
                if valid_blocks > Int64(0):
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        vector.load_effective_query(gm_effective_queries, slot_idx)
                        q_cur = vector.effective_q_ub.wait()
                        vector.score_stream_init()
                        vector.load_dt(
                            gm_v2d, row, Int64(0), valid_blocks,
                        )
                        for db in range(Int64(0), Int64(num_dt)):
                            nxt = db + Int64(1)
                            if nxt < Int64(num_dt):
                                vector.load_dt(
                                    gm_v2d, row, nxt, valid_blocks,
                                )
                            v_cur = vector.v_dt.wait()
                            vector.score_stream_block(
                                v_cur, db, q_cur, valid_blocks,
                            )
                            vector.v_dt.release(v_cur)
                        vector.score_stream_finalize()
                        vector.effective_q_ub.release(q_cur)
                        vector.softmax(valid_blocks)

                        out_row = slot_idx * num_row + row
                        vector.load_dt(
                            gm_v2d, row, Int64(0), valid_blocks,
                        )
                        for db in range(Int64(0), Int64(num_dt)):
                            nxt = db + Int64(1)
                            if nxt < Int64(num_dt):
                                vector.load_dt(
                                    gm_v2d, row, nxt, valid_blocks,
                                )
                            v_cur = vector.v_dt.wait()
                            vector.agg_stream_block(
                                v_cur, gm_h, out_row, db, valid_blocks,
                            )
                            vector.v_dt.release(v_cur)
                        vector.store_stats(gm_max, gm_sum, out_row)
                else:
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        out_row = slot_idx * num_row + row
                        for db in range(Int64(0), Int64(num_dt)):
                            vector.zero_stream_block(gm_h, out_row, db)
                        vector.zero_stats()
                        vector.store_stats(gm_max, gm_sum, out_row)


class BlockAttnResPrepare:
    """Host launcher for the whole-network multi-query path."""

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dtype=BFloat16, mode: str = "hold", dt: int = 0,
                 block_num: int = DEFAULT_BLOCK_NUM):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dtype = dtype
        self.block_num = int(block_num)
        if mode == "stream":
            self._kernel_cls = block_attn_res_prepare_multi_query_kernel_stream
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                int(dt), self.dtype,
            )
        elif mode == "decode":
            self._kernel_cls = block_attn_res_prepare_multi_query_kernel_decode
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                self.dtype,
            )
        else:
            self._kernel_cls = block_attn_res_prepare_multi_query_kernel
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                self.dtype,
            )

    @jit
    def run(self, gm_v2d, gm_effective_queries, gm_valid_blocks, gm_eps,
            gm_h, gm_max, gm_sum):
        op = self._kernel_cls(*self._kernel_args)
        op[self.block_num](
            gm_v2d, gm_effective_queries, gm_valid_blocks, gm_eps,
            gm_h, gm_max, gm_sum,
        )


def _compiled_prepare_kernel(
    max_blocks: int,
    tokens: int,
    slots: int,
    d: int,
    input_dtype: torch.dtype,
    mode: str,
    dt: int,
    block_num: int,
):
    """Compile once per static kernel configuration and reuse the callable."""
    key = (
        int(max_blocks), int(tokens), int(slots), int(d), input_dtype,
        str(mode), int(dt), int(block_num),
    )
    compiled = _COMPILED_KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled

    op = BlockAttnResPrepare(
        max_blocks, tokens, slots, d,
        dtype=_TORCH_TO_DSL[input_dtype], mode=mode, dt=dt,
        block_num=block_num,
    )
    fake = cannbotdsl.TensorSpec
    compiled = op.run.compile(
        fake((tokens * max_blocks, d), _TORCH_TO_DSL[input_dtype]),
        fake((slots, d), dtypes.float32),
        fake((1,), dtypes.int64),
        fake((1,), dtypes.float32),
        fake((slots * tokens, d), dtypes.float32),
        fake((slots * tokens, 1), dtypes.float32),
        fake((slots * tokens, 1), dtypes.float32),
    )
    _COMPILED_KERNEL_CACHE[key] = compiled
    return compiled


def _block_attn_res_prepare_eager(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    *,
    eps: float | torch.Tensor = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K3 Phase 1 block attention residual.

    Args:
        v: [tokens,max_blocks,D] fp32 resident block buffer.
        effective_queries: [slots,D] fp32, pre-fused q * RMSNorm gain.
        valid_blocks: scalar int64 tensor on the same device. Only
            v[:, :valid_blocks, :] participates in attention.
        eps: Python float or scalar fp32 tensor on the same device.

    Returns:
        numerator/max/sum are all fp32. numerator has shape
        [slots,tokens,D], while max/sum have shape [slots,tokens].
        numerator is Σ exp(logit-max)*V and is intentionally not divided by sum;
        all three outputs can be passed directly to the update operator.
    """
    assert v.dtype == torch.float32, \
        f"v 必须为 fp32,实际 {v.dtype}"
    assert v.dim() == 3, "v 必须为 [tokens,max_blocks,D]"
    assert effective_queries.dtype == torch.float32, \
        f"effective_queries 必须为 fp32,实际 {effective_queries.dtype}"
    assert effective_queries.dim() == 2, \
        "effective_queries 必须为 [slots,D]"
    assert isinstance(valid_blocks, torch.Tensor), \
        "valid_blocks 必须为标量 int64 Tensor"
    assert valid_blocks.dtype == torch.int64 and valid_blocks.numel() == 1, \
        "valid_blocks 必须为标量 int64 Tensor"
    assert v.is_contiguous() and effective_queries.is_contiguous() \
        and valid_blocks.is_contiguous(), \
        "v / effective_queries / valid_blocks 必须 contiguous"
    assert v.device == effective_queries.device == valid_blocks.device, \
        "v / effective_queries / valid_blocks 必须同 device"

    tokens, max_blocks, d = (int(size) for size in v.shape)
    slots = int(effective_queries.shape[0])
    assert int(effective_queries.shape[1]) == d, \
        f"effective_queries hidden 必须为 D={d},实际 {effective_queries.shape[1]}"
    assert d >= 1, f"D={d} 非法"
    assert 1 <= max_blocks <= VL, \
        f"max_blocks={max_blocks} 当前实现须在 [1,{VL}]"

    if isinstance(eps, torch.Tensor):
        assert eps.dtype == torch.float32 and eps.numel() == 1, \
            "Tensor eps 必须为标量 fp32 Tensor"
        assert eps.is_contiguous() and eps.device == v.device, \
            "Tensor eps 必须与 v 同 device 且 contiguous"
        eps_tensor = eps.reshape(1)
    else:
        assert isinstance(eps, (float, int)), "eps 必须为 float 或标量 fp32 Tensor"
        assert float(eps) > 0.0, "eps 必须大于 0"
        eps_tensor = None

    out_shape = (slots, tokens, d)
    stats_shape = (slots, tokens)
    out = torch.empty(out_shape, dtype=torch.float32, device=v.device)
    max_out = torch.empty(stats_shape, dtype=torch.float32, device=v.device)
    sum_out = torch.empty(stats_shape, dtype=torch.float32, device=v.device)

    # Empty dimensions are valid host-level shapes and need no kernel launch.
    if tokens == 0 or slots == 0:
        return out, max_out, sum_out

    if eps_tensor is None:
        eps_tensor = torch.tensor(
            float(eps), dtype=torch.float32, device=v.device,
        ).reshape(1)

    ebytes = 2 if v.dtype in (torch.bfloat16, torch.float16) else 4
    w_pad = _ceil_div(d, VL) * VL
    # One token's physical block buffer, one fp32 effective query, one output,
    # and softmax/stat scratch are resident in UB. slots do not multiply UB.
    required_ub = (
        max_blocks * w_pad * ebytes
        + w_pad * 4
        + w_pad * 4
        + 2 * VL * 4
    )
    mode = "hold"
    dt = 0
    if required_ub > UB_BYTES:
        assert d % VL == 0, (
            f"整网多 query D 分块暂要求 D 是 {VL} 的整数倍；D={d}"
        )
        # effective query stays resident; V and h use depth=2 D-tile Channels.
        fixed_stream_bytes = w_pad * 4 + 6 * VL * 4
        # v_dt is depth=2 in V dtype; h_dt is depth=2 and always fp32.
        stream_bytes_per_d = 2 * max_blocks * ebytes + 2 * 4
        dt_max = (
            (UB_BYTES - fixed_stream_bytes)
            // stream_bytes_per_d
            // VL
        ) * VL
        candidate = min(dt_max, d)
        while candidate >= VL:
            if d % candidate == 0:
                dt = candidate
                break
            candidate -= VL
        assert dt >= VL, (
            f"整网多 query D 分块仍超 UB: max_blocks={max_blocks}, D={d}"
        )
        mode = "stream"
    elif _use_decode_slot_parallel(tokens, slots, required_ub):
        mode = "decode"

    block_num = _launch_block_num(mode, tokens, slots)
    compiled = _compiled_prepare_kernel(
        max_blocks, tokens, slots, d, v.dtype, mode, dt, block_num,
    )
    compiled(
        from_torch_npu(v.reshape(tokens * max_blocks, d)),
        from_torch_npu(effective_queries.reshape(slots, d)),
        from_torch_npu(valid_blocks.reshape(1)),
        from_torch_npu(eps_tensor),
        from_torch_npu(out.reshape(slots * tokens, d)),
        from_torch_npu(max_out.reshape(slots * tokens, 1)),
        from_torch_npu(sum_out.reshape(slots * tokens, 1)),
    )
    return out, max_out, sum_out


# ``from_torch_npu`` extracts a raw data pointer and therefore must never be
# traced by TorchDynamo.  Publish the functional whole-network entry as an
# opaque dispatcher op; its PrivateUse1 implementation executes the existing
# eager launcher only after graph capture has finished.
_PREPARE_TORCH_LIBRARY = torch.library.Library("cannbot_attn_res", "FRAGMENT")
_PREPARE_TORCH_LIBRARY.define(
    "block_attn_res_prepare(Tensor v, Tensor effective_queries, "
    "Tensor valid_blocks, Tensor eps) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_PREPARE_TORCH_LIBRARY, "block_attn_res_prepare", "Meta")
def _block_attn_res_prepare_meta(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del valid_blocks, eps
    tokens, _, d = v.shape
    slots = effective_queries.shape[0]
    numerator = v.new_empty((slots, tokens, d), dtype=torch.float32)
    max_out = v.new_empty((slots, tokens), dtype=torch.float32)
    sum_out = v.new_empty((slots, tokens), dtype=torch.float32)
    return numerator, max_out, sum_out


@torch.library.impl(
    _PREPARE_TORCH_LIBRARY, "block_attn_res_prepare", "PrivateUse1",
)
def _block_attn_res_prepare_npu(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _block_attn_res_prepare_eager(
        v, effective_queries, valid_blocks, eps=eps,
    )


def block_attn_res_prepare(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    *,
    eps: float | torch.Tensor = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-safe public entry for K3 Phase-1 block attention residual.

    The four-input functional form is emitted as one opaque Torch operator, so
    Dynamo never enters ``from_torch_npu``.
    """
    if v.device.type != "npu":
        return _block_attn_res_prepare_eager(
            v,
            effective_queries,
            valid_blocks,
            eps=eps,
        )

    if isinstance(eps, torch.Tensor):
        eps_tensor = eps.reshape(1)
    else:
        eps_tensor = torch.scalar_tensor(
            float(eps), dtype=torch.float32, device=v.device,
        ).reshape(1)
    return torch.ops.cannbot_attn_res.block_attn_res_prepare.default(
        v, effective_queries, valid_blocks, eps_tensor,
    )
