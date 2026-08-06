/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// AscendC FUSED MXFP4 -> W8A8: int8 weight + per-output-channel oscale in ONE pass (reads MXFP4
// once). Block-partitioned: each core owns a contiguous, ACC-aligned row range; per row it does
// decode/scale/reduce once, emits int8 (two contiguous planes [lo|hi]) AND accumulates oscale into
// a UB block; each full block is flushed as ONE large contiguous DataCopy (the idiom that makes
// the small per-channel scale store survive alongside loads + int8 stores).
//
// The kernels delegate to plain __aicore__ device functions that share three descriptors: FusedGm
// (global tensors), FusedBufs (owns the UB allocations) and FusedUb (the bound views plus this
// layer's dimensions). The sequence of AscendC calls is exactly the same as when each kernel was
// one flat function -- the helpers only give the phases names and keep every parameter list short.
#include "kernel_operator.h"
using namespace AscendC;

// CANN 9.0.0 derives a kernel's core type from what its body contains, so vector work sitting
// behind calls that were not inlined defeats the derivation and the kernel then fails to
// resolve at run time. `static` gives the helpers internal linkage (no out-of-line copy to
// fall back on) and always_inline folds every call site back into the kernel body. Verified
// on 910B: this emits a .so byte-identical to the `inline __attribute__((always_inline))`
// spelling, so it buys the same code without the `inline` keyword G.FUD.06 objects to.

constexpr int32_t HALF_MAX = 2048;
constexpr int32_t IN_MAX = 4096;
constexpr int32_t NB_MAX = 128;
// oscale flush block (floats), 8-aligned. CONTRACT: the host allocates oscaleg with
// ceil(R/ACC)*ACC floats, not R (see mxfp4_fused_op.py: `rows_pad = (rows + _ACC - 1) // _ACC *
// _ACC`, then it slices osc[:rows]). A tail block is therefore flushed as a full ACC-element
// DataCopy into the padding -- never out of bounds. Flushing only `bend - base` instead would
// break DataCopy's 32-byte (8-float) granularity whenever the tail is not a multiple of 8, so keep
// the padded allocation on the host side rather than shortening the copy here.
constexpr int32_t ACC = 512;

constexpr int32_t LUT_ELEMS = 256;          // e8m0 / fp4 lookup tables are indexed by a byte
constexpr int32_t COMB_PLANES = 2;          // the comb buffer holds the lo and hi halves
constexpr int32_t BLOCK_CODES = 16;         // code bytes per GGUF block_mxfp4 block
constexpr int32_t BLOCK_BYTES = 17;         // ... plus the leading e8m0 scale byte
constexpr int32_t ALIGN_ELEMS = 32;         // DataCopy granularity, in bytes
constexpr int32_t TAIL_LANES = 8;           // width the vector max-tree reduces down to
constexpr float INT8_MAX_F = 127.0f;        // symmetric int8 range used by the quantizer
constexpr float AMAX_FLOOR = 1e-8f;         // keeps 1/amax finite for an all-zero row
constexpr half FP4_INDEX_SCALE = (half)4.0; // byte -> float LUT offset (4 bytes per entry)
constexpr int32_t BLK_MAX = (HALF_MAX / BLOCK_CODES) * BLOCK_BYTES;   // 2176

// ---- launch descriptors -----------------------------------------------------------------------
// The launchers take one descriptor instead of a flat argument list. Pointers come first and the
// uint32 fields last so the layout has no interior padding; mxfp4_fused_op.py mirrors it with a
// ctypes.Structure and checks sizeof() against Mxfp4FusedArgsSize() before the first launch.
// `stream` is an aclrtStream handle, which CANN itself defines as an opaque pointer.
struct Mxfp4FusedArgs {
    void *stream;
    uint8_t *codes;
    uint8_t *scale;
    uint8_t *out;
    uint8_t *oscale;
    uint8_t *lutLo;
    uint8_t *lutHi;
    uint8_t *lutE8;
    uint8_t *scOff;
    uint32_t blockdim;
    uint32_t rows;
    uint32_t halfLen;
    uint32_t nbCount;
    uint32_t inDim;
};

// Launch descriptor of the block-input variant; same contract as Mxfp4FusedArgs above.
struct Mxfp4FusedBlkArgs {
    void *stream;
    uint8_t *blocks;
    uint8_t *out;
    uint8_t *oscale;
    uint8_t *lutLo;
    uint8_t *lutHi;
    uint8_t *lutE8;
    uint8_t *scOff;
    uint8_t *codeOff;
    uint8_t *scaleOff;
    uint32_t blockdim;
    uint32_t rows;
    uint32_t halfLen;
    uint32_t nbCount;
    uint32_t inDim;
};

// What each kernel receives, by value, in place of a dozen positional parameters. GM_ADDR members
// are plain device addresses, so the struct stays POD and marshals into the launch packet as the
// flat argument list did.
struct FusedKernelArgs {
    GM_ADDR codes;
    GM_ADDR scaleg;
    GM_ADDR outg;
    GM_ADDR oscaleg;
    GM_ADDR lutLoG;
    GM_ADDR lutHiG;
    GM_ADDR lutE8G;
    GM_ADDR scOffG;
    uint32_t rows;
    uint32_t hlen;
    uint32_t nb;
    uint32_t inDim;
};

struct FusedBlkKernelArgs {
    GM_ADDR blocks;
    GM_ADDR outg;
    GM_ADDR oscaleg;
    GM_ADDR lutLoG;
    GM_ADDR lutHiG;
    GM_ADDR lutE8G;
    GM_ADDR scOffG;
    GM_ADDR codeOffG;
    GM_ADDR scaleOffG;
    uint32_t rows;
    uint32_t hlen;
    uint32_t nb;
    uint32_t inDim;
};

// ---- device-side descriptors -------------------------------------------------------------------
struct FusedGm {
    GlobalTensor<uint8_t> codes;
    GlobalTensor<uint8_t> scale;
    GlobalTensor<uint8_t> blocks;
    GlobalTensor<uint8_t> out;
    GlobalTensor<float> oscale;
    GlobalTensor<float> lutLo;
    GlobalTensor<float> lutHi;
    GlobalTensor<float> lutE8;
    GlobalTensor<uint32_t> scOff;
    GlobalTensor<uint32_t> codeOff;
    GlobalTensor<uint32_t> scaleOff;
};

// Owns the UB allocations, so it has to live in the kernel frame alongside its TPipe.
struct FusedBufs {
    TQue<QuePosition::VECIN, 1> qCodes;
    TQue<QuePosition::VECIN, 1> qScale;
    TQue<QuePosition::VECOUT, 1> qOut;
    TBuf<TPosition::VECCALC> tLutLo;
    TBuf<TPosition::VECCALC> tLutHi;
    TBuf<TPosition::VECCALC> tLutE8;
    TBuf<TPosition::VECCALC> tScOff;
    TBuf<TPosition::VECCALC> tCodeOff;
    TBuf<TPosition::VECCALC> tScaleOff;
    TBuf<TPosition::VECCALC> tBlkH;
    TBuf<TPosition::VECCALC> tComb;
    TBuf<TPosition::VECCALC> tOff;
    TBuf<TPosition::VECCALC> tOffH;
    TBuf<TPosition::VECCALC> tScI;
    TBuf<TPosition::VECCALC> tScF;
    TBuf<TPosition::VECCALC> tScHalf;
    TBuf<TPosition::VECCALC> tAbs;
    TBuf<TPosition::VECCALC> tWork;
    TBuf<TPosition::VECCALC> tAcc;
};

// The bound views every helper works through, plus this layer's dimensions.
struct FusedUb {
    LocalTensor<float> lutLo;
    LocalTensor<float> lutHi;
    LocalTensor<float> lutE8;
    LocalTensor<float> comb;
    LocalTensor<float> scF;
    LocalTensor<float> scHalf;
    LocalTensor<float> absb;
    LocalTensor<float> work;
    LocalTensor<float> acc;
    LocalTensor<uint32_t> scOff;
    LocalTensor<uint32_t> codeOff;
    LocalTensor<uint32_t> scaleOff;
    LocalTensor<int32_t> off;
    LocalTensor<int32_t> scI;
    LocalTensor<half> offH;
    LocalTensor<half> blkH;
    uint32_t hlen;
    uint32_t nb;
    uint32_t inDim;
};

static __attribute__((always_inline)) __aicore__
void BindLutGlobals(FusedGm &g, GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G,
                    GM_ADDR scOffG)
{
    g.lutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    g.lutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    g.lutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    g.scOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);
}

static __attribute__((always_inline)) __aicore__
void BindSplitIo(FusedGm &g, GM_ADDR codes, GM_ADDR scaleg, GM_ADDR outg,
                 GM_ADDR oscaleg)
{
    g.codes.SetGlobalBuffer((__gm__ uint8_t *)codes);
    g.scale.SetGlobalBuffer((__gm__ uint8_t *)scaleg);
    g.out.SetGlobalBuffer((__gm__ uint8_t *)outg);
    g.oscale.SetGlobalBuffer((__gm__ float *)oscaleg);
}

static __attribute__((always_inline)) __aicore__
void BindBlkIo(FusedGm &g, GM_ADDR blocks, GM_ADDR outg, GM_ADDR oscaleg)
{
    g.blocks.SetGlobalBuffer((__gm__ uint8_t *)blocks);
    g.out.SetGlobalBuffer((__gm__ uint8_t *)outg);
    g.oscale.SetGlobalBuffer((__gm__ float *)oscaleg);
}

static __attribute__((always_inline)) __aicore__
void BindBlkOffsets(FusedGm &g, GM_ADDR codeOffG, GM_ADDR scaleOffG)
{
    g.codeOff.SetGlobalBuffer((__gm__ uint32_t *)codeOffG);
    g.scaleOff.SetGlobalBuffer((__gm__ uint32_t *)scaleOffG);
}

// Buffers every row needs, in both kernels.
static __attribute__((always_inline)) __aicore__
void InitCommonUb(TPipe &pipe, FusedBufs &b, FusedUb &u)
{
    pipe.InitBuffer(b.qOut, 1, IN_MAX * sizeof(uint8_t));
    pipe.InitBuffer(b.tLutLo, LUT_ELEMS * sizeof(float));
    pipe.InitBuffer(b.tLutHi, LUT_ELEMS * sizeof(float));
    pipe.InitBuffer(b.tLutE8, LUT_ELEMS * sizeof(float));
    pipe.InitBuffer(b.tScOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(b.tComb, COMB_PLANES * HALF_MAX * sizeof(float));
    pipe.InitBuffer(b.tOff, HALF_MAX * sizeof(int32_t));
    pipe.InitBuffer(b.tOffH, HALF_MAX * sizeof(half));
    pipe.InitBuffer(b.tScI, NB_MAX * sizeof(int32_t));
    pipe.InitBuffer(b.tScF, NB_MAX * sizeof(float));
    pipe.InitBuffer(b.tScHalf, HALF_MAX * sizeof(float));
    pipe.InitBuffer(b.tAbs, HALF_MAX * sizeof(float));
    pipe.InitBuffer(b.tWork, HALF_MAX * sizeof(float));
    pipe.InitBuffer(b.tAcc, ACC * sizeof(float));

    u.lutLo = b.tLutLo.Get<float>();
    u.lutHi = b.tLutHi.Get<float>();
    u.lutE8 = b.tLutE8.Get<float>();
    u.scOff = b.tScOff.Get<uint32_t>();
    u.comb = b.tComb.Get<float>();
    u.off = b.tOff.Get<int32_t>();
    u.offH = b.tOffH.Get<half>();
    u.scI = b.tScI.Get<int32_t>();
    u.scF = b.tScF.Get<float>();
    u.scHalf = b.tScHalf.Get<float>();
    u.absb = b.tAbs.Get<float>();
    u.work = b.tWork.Get<float>();
    u.acc = b.tAcc.Get<float>();
}

// Extra inputs of the codes+scale kernel.
static __attribute__((always_inline)) __aicore__
void InitSplitUb(TPipe &pipe, FusedBufs &b, FusedUb &u)
{
    InitCommonUb(pipe, b, u);
    pipe.InitBuffer(b.qCodes, 1, HALF_MAX * sizeof(uint8_t));
    pipe.InitBuffer(b.qScale, 1, (NB_MAX + ALIGN_ELEMS) * sizeof(uint8_t));
}

// Extra inputs of the raw-GGUF-block kernel: de-interleave offsets + the half-cast block buffer.
static __attribute__((always_inline)) __aicore__
void InitBlkUb(TPipe &pipe, FusedBufs &b, FusedUb &u)
{
    InitCommonUb(pipe, b, u);
    pipe.InitBuffer(b.qCodes, 1, BLK_MAX * sizeof(uint8_t));
    pipe.InitBuffer(b.tCodeOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(b.tScaleOff, NB_MAX * sizeof(uint32_t));
    pipe.InitBuffer(b.tBlkH, BLK_MAX * sizeof(half));
    u.codeOff = b.tCodeOff.Get<uint32_t>();
    u.scaleOff = b.tScaleOff.Get<uint32_t>();
    u.blkH = b.tBlkH.Get<half>();
}

// The three LUTs and the code->scale-block map. The caller issues the single PipeBarrier after its
// own remaining loads, exactly as the flat version did.
static __attribute__((always_inline)) __aicore__
void LoadLuts(FusedUb &u, const FusedGm &g)
{
    DataCopy(u.lutLo, g.lutLo, LUT_ELEMS);
    DataCopy(u.lutHi, g.lutHi, LUT_ELEMS);
    DataCopy(u.lutE8, g.lutE8, LUT_ELEMS);
    DataCopy(u.scOff, g.scOff, u.hlen);
}

// Row range this core owns: contiguous and ACC-aligned, so every oscale flush stays block-aligned.
static __attribute__((always_inline)) __aicore__
void CoreRowRange(uint32_t rows, uint32_t &rStart, uint32_t &rEnd)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();
    // The fallback cannot be reached on device; it makes the divisor provably non-zero in place.
    const uint32_t cores = (nblk > 0) ? (uint32_t)nblk : 1U;
    const uint32_t chunk = ((rows + cores - 1) / cores + (ACC - 1)) / ACC * ACC;
    rStart = (uint32_t)blkid * chunk;
    rEnd = rStart + chunk;
    if (rEnd > rows) {
        rEnd = rows;
    }
}

// codes/scale -> dequantized lo/hi halves in u.comb: load one row, gather through the LUTs, apply
// the per-32-element e8m0 scale.
static __attribute__((always_inline)) __aicore__
void DecodeRowSplit(FusedUb &u, FusedBufs &b, const FusedGm &g, uint32_t r)
{
    const uint32_t hlen = u.hlen;
    const uint32_t nb = u.nb;
    const uint32_t scLoad = (nb + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
    LocalTensor<float> vlo = u.comb;
    LocalTensor<float> vhi = u.comb[hlen];

    LocalTensor<uint8_t> cu = b.qCodes.AllocTensor<uint8_t>();
    DataCopy(cu, g.codes[(uint64_t)r * hlen], hlen);
    b.qCodes.EnQue(cu);
    LocalTensor<uint8_t> cuU = b.qCodes.DeQue<uint8_t>();
    LocalTensor<uint8_t> su = b.qScale.AllocTensor<uint8_t>();
    DataCopy(su, g.scale[(uint64_t)r * nb], scLoad);
    b.qScale.EnQue(su);
    LocalTensor<uint8_t> suU = b.qScale.DeQue<uint8_t>();

    Cast(u.offH, cuU, RoundMode::CAST_NONE, hlen);
    Muls(u.offH, u.offH, FP4_INDEX_SCALE, hlen);
    Cast(u.off, u.offH, RoundMode::CAST_RINT, hlen);
    LocalTensor<uint32_t> offU = u.off.ReinterpretCast<uint32_t>();
    Gather(vlo, u.lutLo, offU, (uint32_t)0, hlen);
    Gather(vhi, u.lutHi, offU, (uint32_t)0, hlen);
    b.qCodes.FreeTensor(cuU);

    Cast(u.offH, suU, RoundMode::CAST_NONE, nb);
    Muls(u.offH, u.offH, FP4_INDEX_SCALE, nb);
    Cast(u.scI, u.offH, RoundMode::CAST_RINT, nb);
    Gather(u.scF, u.lutE8, u.scI.ReinterpretCast<uint32_t>(), (uint32_t)0, nb);
    b.qScale.FreeTensor(suU);
    PipeBarrier<PIPE_V>();
    Gather(u.scHalf, u.scF, u.scOff, (uint32_t)0, hlen);
    Mul(vlo, vlo, u.scHalf, hlen);
    Mul(vhi, vhi, u.scHalf, hlen);
    PipeBarrier<PIPE_V>();
}

// Same result as DecodeRowSplit, but the row arrives as raw GGUF blocks ([nb,17] = 1 e8m0 scale +
// 16 codes each) and the de-interleave happens in UB via Gather -- the host does none.
static __attribute__((always_inline)) __aicore__
void DecodeRowBlk(FusedUb &u, FusedBufs &b, const FusedGm &g, uint32_t r)
{
    const uint32_t hlen = u.hlen;
    const uint32_t nb = u.nb;
    const uint32_t nb17 = (hlen / BLOCK_CODES) * BLOCK_BYTES;
    const uint32_t blkLoad = (nb17 + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
    LocalTensor<float> vlo = u.comb;
    LocalTensor<float> vhi = u.comb[hlen];

    // blkLoad rounds up to DataCopy's granularity, so it over-reads when nb17 is not 32-aligned,
    // and on the last row it reads past the end of `blocks`. Only the first nb17 bytes are used,
    // and every production in_dim keeps nb17 aligned (2048 -> 1088, 7168 -> 3808).
    LocalTensor<uint8_t> bu = b.qCodes.AllocTensor<uint8_t>();
    DataCopy(bu, g.blocks[(uint64_t)r * nb17], blkLoad);
    b.qCodes.EnQue(bu);
    LocalTensor<uint8_t> buU = b.qCodes.DeQue<uint8_t>();
    Cast(u.blkH, buU, RoundMode::CAST_NONE, nb17);   // blocks -> half
    b.qCodes.FreeTensor(buU);
    PipeBarrier<PIPE_V>();

    Gather(u.offH, u.blkH, u.codeOff, (uint32_t)0, hlen);   // de-interleave codes -> half
    Muls(u.offH, u.offH, FP4_INDEX_SCALE, hlen);
    Cast(u.off, u.offH, RoundMode::CAST_RINT, hlen);
    LocalTensor<uint32_t> offU = u.off.ReinterpretCast<uint32_t>();
    Gather(vlo, u.lutLo, offU, (uint32_t)0, hlen);
    Gather(vhi, u.lutHi, offU, (uint32_t)0, hlen);

    Gather(u.offH, u.blkH, u.scaleOff, (uint32_t)0, nb);    // de-interleave scale -> half
    Muls(u.offH, u.offH, FP4_INDEX_SCALE, nb);
    Cast(u.scI, u.offH, RoundMode::CAST_RINT, nb);
    Gather(u.scF, u.lutE8, u.scI.ReinterpretCast<uint32_t>(), (uint32_t)0, nb);
    PipeBarrier<PIPE_V>();
    Gather(u.scHalf, u.scF, u.scOff, (uint32_t)0, hlen);
    Mul(vlo, vlo, u.scHalf, hlen);
    Mul(vhi, vhi, u.scHalf, hlen);
    PipeBarrier<PIPE_V>();
}

// max(|lo|,|hi|) over the row: a vector tree down to TAIL_LANES, then a scalar tail.
static __attribute__((always_inline)) __aicore__
float RowAmax(FusedUb &u)
{
    const uint32_t hlen = u.hlen;
    LocalTensor<float> vlo = u.comb;
    LocalTensor<float> vhi = u.comb[hlen];

    Abs(u.absb, vlo, hlen);
    Abs(u.work, vhi, hlen);
    Max(u.scHalf, u.absb, u.work, hlen);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> fa = u.scHalf;
    LocalTensor<float> fb = u.absb;
    for (uint32_t h = hlen >> 1; h >= TAIL_LANES; h >>= 1) {
        Max(fb, fa, fa[h], h);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> tmp = fa;
        fa = fb;
        fb = tmp;
    }
    PipeBarrier<PIPE_ALL>();
    float amax = fa.GetValue(0);
    for (int32_t i = 1; i < TAIL_LANES; i++) {
        const float v = fa.GetValue(i);
        if (v > amax) {
            amax = v;
        }
    }
    if (amax < AMAX_FLOOR) {
        amax = AMAX_FLOOR;
    }
    return amax;
}

// Scale by 127/amax, clamp to int8, store the row as two contiguous planes [lo|hi]. RowAmax has
// already floored amax; re-applying the floor here keeps the divisor provably non-zero in place.
static __attribute__((always_inline)) __aicore__
void QuantStoreRow(FusedUb &u, FusedBufs &b, const FusedGm &g, uint32_t r, float amax)
{
    const uint32_t hlen = u.hlen;
    const float safeAmax = (amax > AMAX_FLOOR) ? amax : AMAX_FLOOR;
    const float inv = INT8_MAX_F / safeAmax;
    LocalTensor<float> vlo = u.comb;
    LocalTensor<float> vhi = u.comb[hlen];

    PipeBarrier<PIPE_ALL>();                    // scalar inv -> vector
    Muls(vlo, vlo, inv, hlen);
    Muls(vhi, vhi, inv, hlen);
    PipeBarrier<PIPE_V>();
    Mins(vlo, vlo, INT8_MAX_F, hlen);
    Maxs(vlo, vlo, -INT8_MAX_F, hlen);
    Mins(vhi, vhi, INT8_MAX_F, hlen);
    Maxs(vhi, vhi, -INT8_MAX_F, hlen);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> outrow = b.qOut.AllocTensor<uint8_t>();
    LocalTensor<int8_t> outI = outrow.ReinterpretCast<int8_t>();
    Cast(u.offH, vlo, RoundMode::CAST_NONE, hlen);
    PipeBarrier<PIPE_V>();
    Cast(outI, u.offH, RoundMode::CAST_RINT, hlen);
    PipeBarrier<PIPE_V>();
    Cast(u.offH, vhi, RoundMode::CAST_NONE, hlen);
    PipeBarrier<PIPE_V>();
    Cast(outI[hlen], u.offH, RoundMode::CAST_RINT, hlen);
    PipeBarrier<PIPE_V>();
    b.qOut.EnQue(outrow);
    LocalTensor<uint8_t> outU = b.qOut.DeQue<uint8_t>();
    DataCopy(g.out[(uint64_t)r * u.inDim], outU, u.inDim);
    b.qOut.FreeTensor(outU);
}

// Flush the oscale block as one large contiguous DataCopy (8-aligned base; the buffer is
// ACC-padded on the host, see the ACC comment above).
static __attribute__((always_inline)) __aicore__
void FlushOscale(const FusedUb &u, const FusedGm &g, uint32_t base)
{
    PipeBarrier<PIPE_ALL>();
    DataCopy(g.oscale[base], u.acc, ACC);
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void Mxfp4Fused(FusedKernelArgs a)
{
    FusedGm gm;
    BindSplitIo(gm, a.codes, a.scaleg, a.outg, a.oscaleg);
    BindLutGlobals(gm, a.lutLoG, a.lutHiG, a.lutE8G, a.scOffG);

    TPipe pipe;
    FusedBufs bufs;
    FusedUb ub;
    ub.hlen = a.hlen;
    ub.nb = a.nb;
    ub.inDim = a.inDim;
    InitSplitUb(pipe, bufs, ub);
    LoadLuts(ub, gm);
    PipeBarrier<PIPE_ALL>();

    uint32_t rStart = 0;
    uint32_t rEnd = 0;
    CoreRowRange(a.rows, rStart, rEnd);
    for (uint32_t base = rStart; base < rEnd; base += ACC) {
        uint32_t bend = base + ACC;
        if (bend > rEnd) {
            bend = rEnd;
        }
        for (uint32_t r = base; r < bend; r++) {
            DecodeRowSplit(ub, bufs, gm, r);
            const float amax = RowAmax(ub);
            ub.acc.SetValue(r - base, amax / INT8_MAX_F);   // oscale, flushed per block
            QuantStoreRow(ub, bufs, gm, r, amax);
        }
        FlushOscale(ub, gm, base);
    }
}

extern "C" void LaunchMxfp4Fused(const Mxfp4FusedArgs *args)
{
    FusedKernelArgs a;
    a.codes = (GM_ADDR)args->codes;
    a.scaleg = (GM_ADDR)args->scale;
    a.outg = (GM_ADDR)args->out;
    a.oscaleg = (GM_ADDR)args->oscale;
    a.lutLoG = (GM_ADDR)args->lutLo;
    a.lutHiG = (GM_ADDR)args->lutHi;
    a.lutE8G = (GM_ADDR)args->lutE8;
    a.scOffG = (GM_ADDR)args->scOff;
    a.rows = args->rows;
    a.hlen = args->halfLen;
    a.nb = args->nbCount;
    a.inDim = args->inDim;
    Mxfp4Fused<<<args->blockdim, nullptr, args->stream>>>(a);
}

extern "C" uint32_t Mxfp4FusedArgsSize(void)
{
    return (uint32_t)sizeof(Mxfp4FusedArgs);
}

// ---- block-input variant: reads raw GGUF block_mxfp4 ([nb*17] per row = nb x (1 e8m0 scale + 16
// codes)) and de-interleaves IN UB via Gather (same gather-from-UB-by-offset op the base kernel
// already uses for scHalf), so the host does NO de-interleave (the slow 16-of-17 strided int8 copy).
// codeOff[j] = byte offset of code j in the half-cast block buffer = ((j/16)*17 + 1 + j%16)*2;
// scaleOff[b] = (b*17)*2. Everything after the input load is byte-identical to Mxfp4Fused.
extern "C" __global__ __aicore__ void Mxfp4FusedBlk(FusedBlkKernelArgs a)
{
    FusedGm gm;
    BindBlkIo(gm, a.blocks, a.outg, a.oscaleg);
    BindLutGlobals(gm, a.lutLoG, a.lutHiG, a.lutE8G, a.scOffG);
    BindBlkOffsets(gm, a.codeOffG, a.scaleOffG);

    TPipe pipe;
    FusedBufs bufs;
    FusedUb ub;
    ub.hlen = a.hlen;
    ub.nb = a.nb;
    ub.inDim = a.inDim;
    InitBlkUb(pipe, bufs, ub);
    LoadLuts(ub, gm);
    DataCopy(ub.codeOff, gm.codeOff, a.hlen);
    DataCopy(ub.scaleOff, gm.scaleOff, a.nb);
    PipeBarrier<PIPE_ALL>();

    uint32_t rStart = 0;
    uint32_t rEnd = 0;
    CoreRowRange(a.rows, rStart, rEnd);
    for (uint32_t base = rStart; base < rEnd; base += ACC) {
        uint32_t bend = base + ACC;
        if (bend > rEnd) {
            bend = rEnd;
        }
        for (uint32_t r = base; r < bend; r++) {
            DecodeRowBlk(ub, bufs, gm, r);
            const float amax = RowAmax(ub);
            ub.acc.SetValue(r - base, amax / INT8_MAX_F);
            QuantStoreRow(ub, bufs, gm, r, amax);
        }
        FlushOscale(ub, gm, base);
    }
}

extern "C" void LaunchMxfp4FusedBlk(const Mxfp4FusedBlkArgs *args)
{
    FusedBlkKernelArgs a;
    a.blocks = (GM_ADDR)args->blocks;
    a.outg = (GM_ADDR)args->out;
    a.oscaleg = (GM_ADDR)args->oscale;
    a.lutLoG = (GM_ADDR)args->lutLo;
    a.lutHiG = (GM_ADDR)args->lutHi;
    a.lutE8G = (GM_ADDR)args->lutE8;
    a.scOffG = (GM_ADDR)args->scOff;
    a.codeOffG = (GM_ADDR)args->codeOff;
    a.scaleOffG = (GM_ADDR)args->scaleOff;
    a.rows = args->rows;
    a.hlen = args->halfLen;
    a.nb = args->nbCount;
    a.inDim = args->inDim;
    Mxfp4FusedBlk<<<args->blockdim, nullptr, args->stream>>>(a);
}

extern "C" uint32_t Mxfp4FusedBlkArgsSize(void)
{
    return (uint32_t)sizeof(Mxfp4FusedBlkArgs);
}
