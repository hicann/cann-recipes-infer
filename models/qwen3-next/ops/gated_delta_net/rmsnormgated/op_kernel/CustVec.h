/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "tensorutils.h"
#include "paramutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2Rmsnormgated {

constexpr int BASED = 2048;

struct CustVecShapeInfo {
    int group_size;
    float eps;
    int B;
    int S;
    int D;
    int G;
    float E;
    int ngroups;
    int BASED;
    int BASEG;
    int loopnum;
    int vec_d;
};

__aicore__ inline void tilingShapeCustVec(int B, int S, int D, int G, float E,
                                          CustVecShapeInfo &shape)
{
    shape.group_size = (D / G);
    shape.eps = E;
    shape.B = B;
    shape.S = S;
    shape.D = D;
    shape.G = G;
    shape.E = E;
    shape.ngroups = G;
    shape.BASED = BASED;
    shape.BASEG = (shape.BASED / shape.group_size);
    shape.loopnum = CeilDiv((B * S), (GetBlockNum() * TWO));
    shape.vec_d = (shape.loopnum * shape.D);
}

class CustVec {
public:
    __aicore__ inline CustVec() {}
    __aicore__ inline void Init(GM_ADDR xmtx_, GM_ADDR zmtx_, GM_ADDR wmtx_,
                                GM_ADDR outmtx_, CustVecShapeInfo shape_)
    {
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();

        // Global Tensors (bfloat16_t input/output)
        xmtx.SetGlobalBuffer((__gm__ bfloat16_t*) xmtx_);
        zmtx.SetGlobalBuffer((__gm__ bfloat16_t*) zmtx_);
        wmtx.SetGlobalBuffer((__gm__ bfloat16_t*) wmtx_);
        outmtx.SetGlobalBuffer((__gm__ bfloat16_t*) outmtx_);

        // Local buffers
        outbuf.Init(shape.BASED);
        fp32_xbuf.Init(shape.BASED);           // 原始 x 数据（norm 前）
        fp32_normbuf.Init(shape.BASED);        // norm 后的 x 数据 [NEW]
        fp32_zbuf.Init(shape.BASED);
        fp32_wbuf.Init(shape.BASED / shape.G);
        fp32_squarebuf.Init(shape.BASED);
        fp32_meanbuf.Init(BLK_SIZE);
        fp32_meantempbuf.Init(BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(fp32_dupbuf, BLK_SIZE);
        fp32_scaletempbuf.Init((shape.BASEG * NUM_DBLK_FLOAT));
        fp32_normscalebuf.Init((shape.BASEG * BLK_SIZE));
        fp32_silubuf.Init(shape.BASED);        // silu(z) 结果（重命名，语义更清晰）
        // bfloat16_t buffers for GM-UB transfer
        bf16_xbuf.Init(shape.BASED);
        bf16_zbuf.Init(shape.BASED);
        bf16_wbuf.Init(shape.BASED / shape.G);
        bf16_outbuf.Init(shape.BASED);

        in_ready.Init();
        in_empty.Init();
        out_ready.Init();
        out_empty.Init();
    }

    __aicore__ inline void Compute()
    {
        in_empty.setall();
        out_empty.setall();

        float scale = shape.group_size;
        scale = ((float)((float)1.0 / scale));

        int remainElements = (shape.B * shape.S) - (GetBlockIdx() * shape.loopnum);
        int ub_d = (shape.vec_d) < (remainElements * shape.D) ?
                   (shape.vec_d) : (remainElements * shape.D);

        cnt = 0;
        mte2_base_d = 0;
        mte2_kernel_d = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();
        w_mul_params = MakeDefaultBinaryRepeatParams();
        w_mul_params.dstBlkStride = 1;
        w_mul_params.src0BlkStride = 1;
        w_mul_params.src1BlkStride = 1;
        w_mul_params.dstRepStride = 16;
        w_mul_params.src0RepStride = 16;
        w_mul_params.src1RepStride = 0;

        for (int kernel_d = 0; kernel_d < ub_d; kernel_d += shape.D) {
            for (int base_d = 0; base_d < shape.D; base_d += shape.BASED) {
                in_empty.wait();
                int64_t blockIdxOffset = GetBlockIdx() * shape.vec_d;
                if (kernel_d == 0 && base_d == 0) {
                    // GM -> UB: 搬运 bfloat16_t 数据
                    GM2UB(bf16_xbuf.get(cnt),
                          xmtx[blockIdxOffset + kernel_d + base_d],
                          1, (shape.BASED / MTE_HALF), 0, 0);
                    GM2UB(bf16_zbuf.get(cnt),
                          zmtx[blockIdxOffset + kernel_d + base_d],
                          1, (shape.BASED / MTE_HALF), 0, 0);
                    GM2UB(bf16_wbuf.get(cnt),
                          wmtx[base_d],
                          1, ((shape.BASED / shape.G) / MTE_HALF), 0, 0);
                }
                if (kernel_d + base_d < ub_d - shape.BASED) {
                    mte2_base_d = (base_d + shape.BASED) % shape.D;
                    int step = (base_d + shape.BASED) / shape.D;
                    mte2_kernel_d = kernel_d + step * shape.D;

                    int64_t nextOffset = blockIdxOffset + mte2_kernel_d + mte2_base_d;

                    // GM -> UB: 搬运 bfloat16_t 数据（预取）
                    GM2UB(bf16_xbuf.get(cnt + 1),
                          xmtx[nextOffset],
                          1, (shape.BASED / MTE_HALF), 0, 0);
                    GM2UB(bf16_zbuf.get(cnt + 1),
                          zmtx[nextOffset],
                          1, (shape.BASED / MTE_HALF), 0, 0);
                }
                in_ready.set();

                out_empty.wait();
                in_ready.wait();
                // bfloat16_t -> float Cast (使用 CAST_NONE，相同指数位无需舍入)
                Cast<float, bfloat16_t>(fp32_xbuf.get(cnt), bf16_xbuf.get(cnt),
                                        RoundMode::CAST_NONE, shape.BASED);
                Cast<float, bfloat16_t>(fp32_zbuf.get(cnt), bf16_zbuf.get(cnt),
                                        RoundMode::CAST_NONE, shape.BASED);

                if (kernel_d == 0 && base_d == 0) {
                    Cast<float, bfloat16_t>(fp32_wbuf.get(cnt), bf16_wbuf.get(cnt),
                                            RoundMode::CAST_NONE, (shape.BASED / shape.G));
                }
                PipeBarrier<PIPE_V>();
                Process_calc(scale);

                // float -> bfloat16_t Cast (输出)
                Cast<bfloat16_t, float>(bf16_outbuf.get(cnt), outbuf.get(cnt),
                                        RoundMode::CAST_RINT, shape.BASED);
                PipeBarrier<PIPE_V>();
                out_ready.set();
                in_empty.set();

                out_ready.wait();
                // UB -> GM: 搬运 bfloat16_t 数据
                UB2GM(outmtx[blockIdxOffset + kernel_d + base_d],
                      bf16_outbuf.get(cnt), 1, (shape.BASED / MTE_HALF), 0, 0);
                out_empty.set();

                cnt = (cnt + 1);
            }
        }

        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_calc(float scale)
    {
        // === Step 1: 计算 RMSNorm(x) ===
        // 1.1 计算 x²
        Mul<float, false>(fp32_squarebuf.get(cnt), fp32_xbuf.get(cnt), fp32_xbuf.get(cnt),
                          MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();

        // 1.2 计算均值（按组）
        Duplicate<float, false>(fp32_meanbuf.get(cnt), 0.000000f,
                                MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();

        for (int base_g = 0; base_g < shape.BASEG; ++base_g) {
            ReduceSum<float>(fp32_meanbuf.get(cnt)[base_g],
                             fp32_squarebuf.get(cnt)[(base_g * shape.group_size)],
                             fp32_meantempbuf.get(cnt), shape.group_size);
            PipeBarrier<PIPE_V>();
        }

        // 1.3 计算 rstd = 1 / sqrt(mean(x²) + eps)
        Muls<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), scale, shape.BASEG);
        PipeBarrier<PIPE_V>();
        Adds<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), shape.eps, shape.BASEG);
        PipeBarrier<PIPE_V>();
        Sqrt<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), shape.BASEG);
        PipeBarrier<PIPE_V>();

        // 1.4 广播 rstd
        Duplicate<float, false>(fp32_dupbuf, 1.000000f,
                                MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        auto custparam = MakeDefaultBinaryRepeatParams();
        custparam.src0RepStride = 0;
        Div<float, false>(fp32_meanbuf.get(cnt), fp32_dupbuf, fp32_meanbuf.get(cnt),
                          MASK_PLACEHOLDER, 1, custparam);
        PipeBarrier<PIPE_V>();
        Brcb(fp32_scaletempbuf.get(cnt), fp32_meanbuf.get(cnt),
             (shape.BASEG / 8), {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        Brcb(fp32_normscalebuf.get(cnt), fp32_scaletempbuf.get(cnt),
             shape.BASEG, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();

        // 1.5 norm_x = x * rstd
        for (int base_g = 0; base_g < shape.BASEG; ++base_g) {
            int index = base_g * shape.group_size;
            Mul<float, false>(fp32_normbuf.get(cnt)[index],
                              fp32_normscalebuf.get(cnt)[(base_g * BLK_SIZE)],
                              fp32_xbuf.get(cnt)[index],
                              MASK_PLACEHOLDER, (shape.group_size / VEC_FLOAT), custparam);
            PipeBarrier<PIPE_V>();
        }

        // === Step 2: 计算 SiLU(z) ===
        int vecOpsBase = shape.BASED / VEC_FLOAT;
        Muls<float, false>(fp32_silubuf.get(cnt), fp32_zbuf.get(cnt), -1.0f,
                           MASK_PLACEHOLDER, vecOpsBase, unary_params);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(fp32_silubuf.get(cnt), fp32_silubuf.get(cnt),
                          MASK_PLACEHOLDER, vecOpsBase, unary_params);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(fp32_silubuf.get(cnt), fp32_silubuf.get(cnt), 1.0f,
                           MASK_PLACEHOLDER, vecOpsBase, unary_params);
        PipeBarrier<PIPE_V>();
        Div<float, false>(fp32_silubuf.get(cnt), fp32_dupbuf, fp32_silubuf.get(cnt),
                          MASK_PLACEHOLDER, vecOpsBase, custparam);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(fp32_silubuf.get(cnt), fp32_silubuf.get(cnt), fp32_zbuf.get(cnt),
                          MASK_PLACEHOLDER, vecOpsBase, binary_params);
        PipeBarrier<PIPE_V>();

        // === Step 3: out = norm_x * w * silu(z) ===
        int halfVecOps = (shape.BASED / VEC_FLOAT) / 2;

        Mul<float, false>(fp32_normbuf.get(cnt)[0],
                          fp32_normbuf.get(cnt)[0],
                          fp32_wbuf.get(0)[0],
                          MASK_PLACEHOLDER, halfVecOps, w_mul_params);

        Mul<float, false>(fp32_normbuf.get(cnt)[VEC_FLOAT],
                          fp32_normbuf.get(cnt)[VEC_FLOAT],
                          fp32_wbuf.get(0)[VEC_FLOAT],
                          MASK_PLACEHOLDER, halfVecOps, w_mul_params);
        PipeBarrier<PIPE_V>();

        Mul<float, false>(outbuf.get(cnt), fp32_normbuf.get(cnt), fp32_silubuf.get(cnt),
                          MASK_PLACEHOLDER, vecOpsBase, binary_params);
        PipeBarrier<PIPE_V>();
    }

private:
    CustVecShapeInfo shape;
    // Global Params
    int cnt;
    int mte2_base_d;
    int mte2_kernel_d;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;
    BinaryRepeatParams w_mul_params;
    // Global Tensors (bfloat16_t input/output)
    GlobalTensor<bfloat16_t> xmtx;
    GlobalTensor<bfloat16_t> zmtx;
    GlobalTensor<bfloat16_t> wmtx;
    GlobalTensor<bfloat16_t> outmtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> outbuf;
    DBuff<float, TPosition::VECCALC> fp32_xbuf;
    DBuff<float, TPosition::VECCALC> fp32_normbuf;    // norm 后的 x 数据 [NEW]
    DBuff<float, TPosition::VECCALC> fp32_zbuf;
    DBuff<float, TPosition::VECCALC> fp32_wbuf;
    DBuff<float, TPosition::VECCALC> fp32_squarebuf;
    DBuff<float, TPosition::VECCALC> fp32_meanbuf;
    DBuff<float, TPosition::VECCALC> fp32_meantempbuf;
    LocalTensor<float> fp32_dupbuf;
    DBuff<float, TPosition::VECCALC> fp32_scaletempbuf;
    DBuff<float, TPosition::VECCALC> fp32_normscalebuf;
    DBuff<float, TPosition::VECCALC> fp32_silubuf;     // silu(z) 结果（重命名）
    // bfloat16_t buffers for GM-UB transfer
    DBuff<bfloat16_t, TPosition::VECCALC> bf16_xbuf;
    DBuff<bfloat16_t, TPosition::VECCALC> bf16_zbuf;
    DBuff<bfloat16_t, TPosition::VECCALC> bf16_wbuf;
    DBuff<bfloat16_t, TPosition::VECCALC> bf16_outbuf;

    // Double events
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    // User-defined events
};
// Auto-generated code. Readability is not guaranteed
}
}