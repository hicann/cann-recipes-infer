// SituAndMulSparse kernel for the registry (custom ops) project.
//
// Integration provenance (ascendc-perf-integrate, faithful port):
// The kernel class below is the round-1 final-selected scheme
// r1-restore-full-blockdim-dbuf-rowpipeline-tilecols3072, ported line-by-line
// from the accepted <<<>>> direct-invoke implementation
// op_kernel/situ_and_mul_sparse_kernel.asc (sha256
//  4e85b8750e38d8e1c7bdb5b0796cadd04f10771bedcf9169f7426b07f83bd36b,
// verified binary libsitu_and_mul_sparse_direct_ops.so sha256
//  69aae450d60c10b087ed6acce32f66f2e05aed0105e9c72224407a69406677af),
// which itself derives from the frozen registry kernel body (source tree
// sha256 bb3eb21cb4978d0e02a2f25b42236bde51f94deaf0b7a6156027dfca3b041829)
// with only the contract-mandated modifications of schemes r2c and r1-C.
//
// regbase run round-1 integration (second ascendc-perf-integrate faithful
// port): the accepted a-regbase-vf-compute-single-tile VF Compute change is
// layered on top, ported line-by-line from the regbase run final_optimized
// op_kernel/situ_and_mul_sparse_kernel.asc (sha256
//  62a882d502fb1967c53e176a9a5c34cb167ac7c17126fab97d13dcbd1a910485,
// verified binary libsitu_and_mul_sparse_direct_ops.so sha256
//  d00d072d681d0273e7002e23334e52169152391b4ec72802e87a22d8815fe37f).
// The change surface is confined to the Compute chain (register-resident VF
// dispatch via SituAndMulSparseComputeVF), the deletion of the
// floatBuffer_/roundBuffer_ TBufs with their InitBuffer calls, and the
// deletion of the RoundToInput member; every other element of the r1-C port
// below (registered entry, runner layout, pipeline shell, guards, branch
// boundaries) is unchanged from the capexplore integration.
//
// Registry adaptation (bucket B, mechanical and bounded):
//   - the two direct-invoke entries situ_and_mul_sparse_kernel_fp16/_bf16
//     (extern "C" __global__ __vector__, 4 GM_ADDR params, own tiling read)
//     converge into the single registered entry situ_and_mul_sparse below
//     (extern "C" __global__ __aicore__, 5 GM_ADDR params incl. workspace),
//     with the framework-injected DTYPE_X template argument replacing the
//     per-dtype entry pair and GET_TILING_DATA(tilingData, tiling) replacing
//     the raw SituAndMulSparseLoadTiling field copy;
//   - the TilingData struct comes from kernel_tiling/kernel_tiling.h
//     (generated from the unchanged 7-field op_host/situ_and_mul_sparse_tiling.h
//     TILING_DATA_DEF: totalElements/outputElementsPerRow/expertTokenCount/
//     chunkElements/beta/alpha/highPrecision), so the class body's
//     `tiling_->field` accesses stay byte-for-byte unchanged.
// The mechanical layout of the accepted implementation is preserved: the
// runner template is declared BEFORE the entry and defined after the class,
// so the expert_tokens[0] load statement of the entry stays textually BEFORE
// every SetGlobalBuffer/InitBuffer call site of this translation unit (f1
// layout), while its use point (Process -> LoadValidRows) is after all of
// them; within the entry body the issue->use overlap window (SetGlobalBuffer
// x3 + InitBuffer x2 after the regbase candidate A buffer deletion; the
// invariant is unaffected, the load statement still precedes every
// SetGlobalBuffer/InitBuffer call site textually) is identical to the
// direct-invoke entries.
//
// Preserved scheme features (all guards/branch boundaries verbatim):
//   f1-early-issue-tokens0-overlap-init: the expert_tokens[0] GM scalar read
//      is issued immediately after the tiling read (raw __gm__ int64 pointer
//      form - the dav-3510 device-verified lowering equivalent of
//      GlobalTensor::GetValue, no SetGlobalBuffer needed; guarded by
//      expertTokenCount > 0 so a zero-length tensor never emits the load and
//      never goes out of bounds). The value is consumed only inside
//      Process -> LoadValidRows, i.e. AFTER the three SetGlobalBuffer calls
//      and every InitBuffer call (x4 in r2c; x2 after the regbase round-1
//      candidate A deleted the floatBuffer_/roundBuffer_ staging buffers -
//      the invariant is unaffected, the load statement still precedes every
//      SetGlobalBuffer/InitBuffer call site textually), immediately before
//      the row loop; expert_tokens[i >= 1] are read after that use point in
//      baseline ascending order, and the read set / summation / clamp
//      semantics are identical to the baseline LoadValidRows.
//   f2-copyin-merge-aligned-singlerow: CopyIn takes a merged branch when
//      (elements == tiling->outputElementsPerRow) AND
//      (2*elements*sizeof(T)) % 32 == 0: single-row single-chunk with a
//      32B-aligned total. The gate/up half rows are adjacent in GM, so the
//      two DataCopy calls become ONE DataCopy of 2*elements contiguous
//      elements from xGm_[inputRowOffset]. elements == outputElementsPerRow
//      implies column == 0 and chunkElements == elements (chunkElements =
//      min(EPR, 3072) <= EPR, while a single chunk covering a whole row
//      requires chunkElements >= EPR), so the merged copy lands gate at
//      both[0] and up at both[chunkElements] exactly where the Compute
//      views expect them. Every other case (multi-chunk rows with
//      elements < outputElementsPerRow, and single rows whose
//      2*EPR*sizeof(T) is not a 32B multiple) takes the split branch, whose
//      DataCopy GM offsets and element counts are byte-identical to the
//      baseline two-copy path.
//   f3-single-vecin-queue: gateQueue_ + upQueue_ (two TQue<VECIN, 1>) are
//      merged into one TQue<VECIN, 1> gateupQueue_ whose buffer is
//      2*chunkElements*sizeof(T): one AllocTensor/EnQue pair in CopyIn and
//      one DeQue/FreeTensor pair in Compute per chunk (2 -> 1 instances
//      each); TPipe::InitBuffer calls go 5 -> 4 (gateupQueue / outQueue /
//      floatBuffer / roundBuffer). The floatBuffer 4-region view and the
//      roundBuffer structure keep the baseline form; the depth-1 TQue
//      synchronization semantics are fully preserved - no SetFlag/WaitFlag
//      or any manual synchronization primitive is introduced.
//   fc2-gateup-out-queue-double-buffer: InitBuffer num 1 -> 2 for
//      gateupQueue_ (per-block size unchanged: chunkElements*sizeof(T)*2)
//      and outQueue_ (chunkElements*sizeof(T)), enabling the hardware
//      double buffer; the TQue template depth stays 1, floatBuffer_ /
//      roundBuffer_ stay single TBuf, total buffer count 6 <= 64. Actual
//      per-chunk-element UB usage becomes 2*2*2 + 2*2 + 4*4 + 2 = 30 B/elem,
//      exactly the budget the frozen tiling reserves
//      (BYTES_PER_TILE_ELEMENT = 30), so 30 B/elem * chunkElements
//      <= 92160 B @ chunkElements = 3072 <= ubSize*7/8 = 229376 B.
//   fc3-rowchunk-prefetch-pipeline: the r2c Process row loop + ProcessRange
//      while-loop are flattened into this block's row x chunk sequence and
//      driven as a cross-row prefetch pipeline
//          CopyIn(first chunk);
//          while (chunk in flight) {
//              DeQue(cur); if (next chunk) CopyIn(next);
//              Compute(cur); CopyOut(cur); }
//      so the MTE2 of chunk k+1 overlaps the VEC of chunk k (also across
//      row boundaries - with MAX_TILE_COLS = 3072 every EPR = 3072 row is a
//      single chunk, and the pipeline then prefetches the next row). The
//      chunk arithmetic (column / remaining / rowRemaining / chunk /
//      current) is the r2c ProcessRange arithmetic verbatim (NextChunk),
//      the CopyIn merged/split branches, the 26-call vector chain, the
//      CopyOut path and the RoundToInput timing are byte-identical to r2c.
//      EnQue run stays <= 1 (every CopyIn EnQue is preceded by the DeQue of
//      the in-flight chunk), preserving the TQue depth-1 synchronization
//      semantics - no SetFlag/WaitFlag or manual sync introduced.
//   fc4-single-chunk-sparse-path-invariant: a block with no chunk
//      (block >= validRows) performs no queue operation at all, and a block
//      with exactly one chunk performs exactly the r2c sequence
//      CopyIn -> Compute -> CopyOut (the hasNext guard suppresses every
//      extra queue operation), so the sparse small-token cases keep the r2c
//      call sequence identically.
//
// blockDim note (fc1, host-side feature of the r1-C scheme): the row loop
// `row = block; row < validRows; row += blockCount` adapts to ANY launched
// block count (GetBlockNum() returns the actual launch grid size), and the
// registered host tiling keeps blockDim = min(GetCoreNumAiv(), totalRows)
// (the frozen registry formula), so full-load correctness (validRows ==
// totalRows, e.g. tokens=tmax) never depends on a block-count assumption.
// The 26-call vector computation chain, the CopyOut path and the
// RoundToInput timing (highPrecision == 0) are identical to the frozen
// registry kernel; the validRows clamp semantics and the launch ABI are
// unchanged.
//
// regbase run round-1 candidate a-regbase-vf-compute-single-tile (relative to
// the r2c/fc2/fc3/fc4 kernel above; launch ABI, public contract, tiling
// schema and the pipeline shell are unchanged - contract
// implementation-contract-regbase-vf-compute-single-tile.json):
//   vf-body-vecscope-single-tile: the 26-call LocalTensor vector chain of
//      Compute (r2c lines 312-354) and the RoundToInput UB staging are
//      replaced by a register-resident chain in the standalone
//      __aicore__ inline helper SituAndMulSparseComputeVF<T, kRoundToInput>
//      below (defined between the entry and the kernel class). The helper
//      opens one __VEC_SCOPE__ VF compute domain, receives only __ubuf__ T*
//      gate/up/out pointers (GetPhyAddr-derived, up = gate + chunkElements),
//      the element count and the six hoisted tiling-derived scalars, and
//      processes one 64-fp32-element tile per iteration (uint16_t loop
//      counter from 0, step 1, no break/continue, no runtime if inside the
//      VF body). asc_vf_call is NOT used (PRC-VF-006: it blocks MTE2 during
//      the call and would break the fc3 cross-row prefetch MTE2/VEC
//      overlap); the __VEC_SCOPE__ direct form keeps the fc3 pipeline
//      intact.
//   vf-io-unpack-pack-mask: UB<->register traffic uses
//      Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16> (no mask) on input and
//      Reg::StoreAlign<T, StoreDist::DIST_PACK_B32> (carrying the tile
//      MaskReg) on output; every tile narrows its mask via
//      Reg::UpdateMask<float>(uint32_t& remaining) (mask generated on the
//      wide fp32 type, count decremented), mirroring the dav-3510
//      kernel-side Cast lowering (GenLoadL2/GenStoreL2 +
//      CastIntrinsicsImplVF, impl/basic_api/dav_3510/
//      kernel_operator_vec_vconv_impl.h). Tile offsets are uint32_t
//      (i*64 element stepping); no int64_t/uint64_t offset variable appears
//      inside the VF body; no UnalignReg/LoadUnAlign/StoreUnAlign and no
//      CreateAddrReg (the three pointers advance linearly).
//   vf-casttrait-mirror-kernel-side: both Reg::Cast CastTrait instances
//      mirror the kernel-side CastIntrinsicsImplVF values exactly
//      (static constexpr at function scope): fp32 -> T uses
//      {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
//      RoundMode::CAST_RINT} and T -> fp32 uses
//      {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
//      RoundMode::CAST_NONE}. The ascendc-performance-best-practices
//      skeleton trait castTraitB322B16 (SatMode::NO_SAT) is deliberately
//      NOT copied - the baseline saturating behavior is SAT.
//   vf-roundtoinput-in-register: RoundToInput (r2c lines 363-368, two Casts
//      staged through the roundBuffer_ UB tensor) becomes two in-register
//      Reg::Cast calls (fp32 RegTensor -> b16 RegTensor -> fp32 RegTensor)
//      guarded by if constexpr(kRoundToInput); no UB staging remains.
//   vf-no-intermediate-ub: every intermediate of the chain lives in
//      RegTensors; the floatBuffer_ TBuf (4-region gate/up/tmp/branch work
//      view) and the roundBuffer_ TBuf, together with their InitBuffer
//      calls, are deleted (InitBuffer count 4 -> 2, both queues only);
//      UB usage drops from 30 to 12 B/elem. The gateupQueue_/outQueue_
//      AllocTensor/EnQue/DeQue/FreeTensor semantics and call positions are
//      unchanged.
//   vf-highprecision-branch-hoisted: the tiling_->highPrecision == 0
//      runtime branch is hoisted out of the VF body - Compute dispatches to
//      the <T, true> / <T, false> helper instances and the only branch
//      inside __VEC_SCOPE__ is if constexpr(kRoundToInput).
//   vf-default-algo-bit-identity: all arithmetic keeps the baseline
//      algorithm modes - Reg::Exp and Reg::Div use their default
//      (INTRINSIC) modes, matching the kernel-side DEFAULT_EXP_CONFIG /
//      DEFAULT_DIV_CONFIG lowering branches the baseline 26-call chain
//      lowers through; Reg::Muls/Adds/Duplicate use the scalar forms and
//      Reg::Mul the binary form for the final branch*up product. Per
//      element the intrinsic sequence (op order, algo, round, sat, merge
//      modes) is identical to the baseline lowering; only the intermediate
//      UB load/store instructions are removed.

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

// Kernel-template runner: constructs the TPipe and drives Init/Process. Only
// declared here (before the registered entry and before the kernel class) and
// defined after the class, so that the expert_tokens[0] load statement of the
// entry stays textually BEFORE every SetGlobalBuffer/InitBuffer call site of
// this translation unit (f1 mechanical layout, same pattern as the accepted
// direct-invoke implementation).
template <typename T>
__aicore__ inline void SituAndMulSparseRun(GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y,
                                            const SituAndMulSparseTilingData* tiling,
                                            int64_t token0);

// ---------------------------------------------------------------------------
// Registered entry (launch ABI unchanged from the frozen registry kernel:
// extern "C" __global__ __aicore__, OpType-matched name, 5 GM_ADDR params).
// f1: right after the GET_TILING_DATA tiling read, the expert_tokens[0] GM
// scalar load is issued; its value is passed down to the runner and is first
// consumed after Init (SetGlobalBuffer x3 + InitBuffer x2 after the regbase
// candidate A buffer deletion), immediately before the row loop. DTYPE_X is
// injected by the build system per dtype instance (float16 / bfloat16),
// replacing the per-dtype direct entries.
// ---------------------------------------------------------------------------

extern "C" __global__ __aicore__ void situ_and_mul_sparse(
    GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    int64_t token0 = 0;
    if (tilingData.expertTokenCount > 0) {
        const __gm__ int64_t* expertTokensPtr =
            reinterpret_cast<const __gm__ int64_t*>(expertTokens);
        token0 = expertTokensPtr[0];
    }
    SituAndMulSparseRun<DTYPE_X>(x, expertTokens, y, &tilingData, token0);
}

// ---------------------------------------------------------------------------
// Register-resident VF compute helper (regbase round-1 candidate A).
//
// One __VEC_SCOPE__ domain; one 64-fp32-element tile per iteration. The
// per-element intrinsic sequence is identical to the baseline 26-call
// LocalTensor chain lowered through the dav_3510 kernel-side impls (see the
// candidate block in the file header for the feature-by-feature mapping);
// only the intermediate UB load/store instructions are removed.
//
// Boundary notes (function-scope static constexpr CastTrait is the
// FAQ-standard positive form for a combined host/device ASC translation
// unit; the signatures use the __ubuf__ spelling, the __local_mem__ alias
// resolves to it on CANN 9.1.0).
// ---------------------------------------------------------------------------
namespace situ_vf {
// dav-3510 vector register width is 256B; a tile is one full fp32 register:
// 256 / sizeof(float) = 64 lanes.
constexpr uint32_t VF_TILE_FLOATS = 64U;
}

template <typename T, bool kRoundToInput>
__aicore__ inline void SituAndMulSparseComputeVF(__ubuf__ T* gate, __ubuf__ T* up, __ubuf__ T* out,
                                                 uint32_t elements, float neg2OverBeta, float negBeta,
                                                 float neg2OverAlpha, float negAlpha, float beta,
                                                 float alpha)
{
    // CastTrait instances mirror the kernel-side Cast lowering on dav-3510
    // (CastIntrinsicsImplVF, kernel_operator_vec_vconv_impl.h:250:
    // {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, roundMode}).
    // SAT, not the example skeleton's NO_SAT.
    static constexpr Reg::CastTrait kF32ToTCastTrait = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    const uint16_t tileCount = static_cast<uint16_t>(
        (elements + situ_vf::VF_TILE_FLOATS - 1U) / situ_vf::VF_TILE_FLOATS);
    uint32_t remaining = elements;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> gateF;
        Reg::RegTensor<float> upF;
        Reg::RegTensor<float> tmpF;
        Reg::RegTensor<float> branchF;
        Reg::RegTensor<T> packedF;
        Reg::MaskReg preg;
        for (uint16_t i = 0; i < tileCount; ++i) {
            preg = Reg::UpdateMask<float>(remaining);
            const uint32_t offset = static_cast<uint32_t>(i) * situ_vf::VF_TILE_FLOATS;
            // Entry casts (baseline Cast(gate/up, gateIn/upIn, CAST_NONE)):
            // unpack load + Reg::Cast; the fp32 value stays in registers.
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, gate + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(gateF, packedF, preg);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, up + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(upF, packedF, preg);

            // gate subchain (11 Reg ops; default INTRINSIC Exp/Div modes,
            // scalar Muls/Adds/Duplicate forms - same lowering as baseline).
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
                // In-register RoundToInput: fp32 -> T (CAST_RINT) -> fp32
                // (CAST_NONE); the roundBuffer_ UB round-trip is gone.
                Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
                Reg::Cast<float, T, kTToF32CastTrait>(branchF, packedF, preg);
            }

            // up subchain (7 Reg ops).
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

            // Exit cast (baseline Cast(out, branch, CAST_RINT)): fp32 -> T
            // then packed store with the tile mask.
            Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(out + offset, packedF, preg);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel template class. The CopyIn merged/split branches, the CopyOut path,
// the fc2/fc3/fc4 pipeline shell and the f1/f2/f3 features are byte-identical
// to the accepted r1-C kernel of the capexplore integration; the regbase
// round-1 candidate A change is confined to Compute (VF dispatch to
// SituAndMulSparseComputeVF above), the deletion of the floatBuffer_/
// roundBuffer_ TBufs with their InitBuffer calls, and the deletion of the
// RoundToInput member (replaced by the in-register round-trip inside the VF
// helper).
// ---------------------------------------------------------------------------

template <typename T>
class SituAndMulSparseKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y,
                                const SituAndMulSparseTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        expertTokensGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokens);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        // fc2-gateup-out-queue-double-buffer: num 1 -> 2 on both queues
        // (per-block sizes unchanged); TQue template depth stays 1.
        // vf-no-intermediate-ub: the floatBuffer_/roundBuffer_ InitBuffer
        // calls of the baseline are deleted (4 -> 2); every intermediate of
        // the compute chain now lives in registers inside the VF helper.
        pipe->InitBuffer(gateupQueue_, 2, tiling_->chunkElements * sizeof(T) * 2);
        pipe->InitBuffer(outQueue_, 2, tiling_->chunkElements * sizeof(T));
    }

    __aicore__ inline void Process(int64_t token0)
    {
        const int64_t totalRows = tiling_->totalElements / tiling_->outputElementsPerRow;
        const int64_t validRows = LoadValidRows(totalRows, token0);

        const int64_t block = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();

        // fc3-rowchunk-prefetch-pipeline: this block's row x chunk sequence
        // (the r2c `for (row = block; row < validRows; row += blockCount)`
        // row loop, each row walked by the ProcessRange while-loop) is
        // flattened and driven as a cross-row prefetch pipeline:
        //     CopyIn(first chunk);
        //     while (chunk in flight) {
        //         DeQue(cur);              // gateupQueue_ EnQue run <= 1
        //         if (next chunk) CopyIn(next);
        //         Compute(cur);
        //         CopyOut(cur); }
        // fc4-single-chunk-sparse-path-invariant: a block with no chunk
        // (block >= validRows) performs no queue operation at all; a block
        // with exactly one chunk performs exactly the r2c sequence
        // CopyIn -> Compute -> CopyOut with no extra queue operation.
        int64_t row = block;
        int64_t processed = 0;
        int64_t offset = 0;
        int64_t current = 0;
        if (NextChunk(row, processed, validRows, blockCount, offset, current)) {
            CopyIn(offset, current);
            int64_t nextOffset = 0;
            int64_t nextCurrent = 0;
            while (true) {
                LocalTensor<T> both = gateupQueue_.DeQue<T>();
                const bool hasNext =
                    NextChunk(row, processed, validRows, blockCount, nextOffset, nextCurrent);
                if (hasNext) {
                    CopyIn(nextOffset, nextCurrent);
                }
                Compute(both, current);
                CopyOut(offset, current);
                if (!hasNext) {
                    break;
                }
                offset = nextOffset;
                current = nextCurrent;
            }
        }
    }

    __aicore__ inline int64_t LoadValidRows(int64_t totalRows, int64_t token0)
    {
        int64_t validRows = token0;
        for (int64_t i = 1; i < tiling_->expertTokenCount; ++i) {
            validRows += expertTokensGm_.GetValue(i);
        }
        return validRows > totalRows ? totalRows : (validRows > 0 ? validRows : 0);
    }

private:
    // fc3-rowchunk-prefetch-pipeline: flat chunk cursor over this block's
    // row x chunk sequence, visiting chunks in exactly the r2c order
    // (row = block; row < validRows; row += blockCount; within each row
    // processed walks 0 -> outputElementsPerRow). Computes the chunk at
    // (row, processed) with the r2c ProcessRange chunk arithmetic verbatim
    // (elements is always a full row, exactly as the r2c Process row loop
    // passed outputElementsPerRow to ProcessRange), advances the cursor
    // past that chunk and returns true; returns false once the sequence is
    // exhausted (row >= validRows, i.e. this block has no further chunk).
    __aicore__ inline bool NextChunk(int64_t& row, int64_t& processed,
                                     int64_t validRows, int64_t blockCount,
                                     int64_t& offset, int64_t& current)
    {
        if (row >= validRows) {
            return false;
        }
        const int64_t globalOffset = row * tiling_->outputElementsPerRow;
        const int64_t elements = tiling_->outputElementsPerRow;
        offset = globalOffset + processed;
        const int64_t column = offset % tiling_->outputElementsPerRow;
        const int64_t remaining = elements - processed;
        const int64_t rowRemaining = tiling_->outputElementsPerRow - column;
        const int64_t chunk = remaining < tiling_->chunkElements
            ? remaining : tiling_->chunkElements;
        current = chunk < rowRemaining ? chunk : rowRemaining;
        processed += current;
        if (processed >= tiling_->outputElementsPerRow) {
            row += blockCount;
            processed = 0;
        }
        return true;
    }

    __aicore__ inline void CopyIn(int64_t globalOffset, int64_t elements)
    {
        LocalTensor<T> both = gateupQueue_.AllocTensor<T>();
        const int64_t row = globalOffset / tiling_->outputElementsPerRow;
        const int64_t inputRowOffset = row * tiling_->outputElementsPerRow * 2;
        const int64_t column = globalOffset % tiling_->outputElementsPerRow;
        if (elements == tiling_->outputElementsPerRow
            && (2 * elements * sizeof(T)) % 32 == 0) {
            DataCopy(both, xGm_[inputRowOffset], elements * 2);
        } else {
            DataCopy(both, xGm_[inputRowOffset + column], elements);
            DataCopy(both[tiling_->chunkElements],
                     xGm_[inputRowOffset + tiling_->outputElementsPerRow + column], elements);
        }
        gateupQueue_.EnQue(both);
    }

    // fc3: the gateupQueue_ DeQue is issued by the Process prefetch loop
    // (DeQue(cur) must precede CopyIn(next) so the TQue depth-1 EnQue run
    // stays <= 1); the pre-dequeued tensor is passed in. Below the signature
    // only the outQueue_ alloc/enque and the gateupQueue_ free remain of the
    // r2c body: the 26-call vector chain and the RoundToInput UB staging are
    // replaced by the register-resident VF helper
    // (vf-body-vecscope-single-tile / vf-roundtoinput-in-register /
    // vf-no-intermediate-ub); the queue call positions are unchanged.
    __aicore__ inline void Compute(LocalTensor<T> both, int64_t elements)
    {
        LocalTensor<T> out = outQueue_.AllocTensor<T>();
        // VF helper inputs: __ubuf__ pointers derived from GetPhyAddr
        // (up = gate + chunkElements, matching the CopyIn split-branch
        // layout that lands the up half at both[chunkElements]) and the six
        // tiling-derived scalars hoisted out of the per-tile loop (computed
        // once per Compute, no in-loop division).
        __ubuf__ T* gate = (__ubuf__ T*)both.GetPhyAddr();
        __ubuf__ T* up = gate + tiling_->chunkElements;
        __ubuf__ T* outPtr = (__ubuf__ T*)out.GetPhyAddr();
        const uint32_t count = static_cast<uint32_t>(elements);
        const float neg2OverBeta = -2.0f / tiling_->beta;
        const float negBeta = -tiling_->beta;
        const float neg2OverAlpha = -2.0f / tiling_->alpha;
        const float negAlpha = -tiling_->alpha;
        // vf-highprecision-branch-hoisted: the highPrecision runtime branch
        // stays outside the VF body; the two template instances differ only
        // in if constexpr(kRoundToInput) inside __VEC_SCOPE__.
        if (tiling_->highPrecision == 0) {
            SituAndMulSparseComputeVF<T, true>(gate, up, outPtr, count, neg2OverBeta, negBeta,
                                               neg2OverAlpha, negAlpha, tiling_->beta,
                                               tiling_->alpha);
        } else {
            SituAndMulSparseComputeVF<T, false>(gate, up, outPtr, count, neg2OverBeta, negBeta,
                                                neg2OverAlpha, negAlpha, tiling_->beta,
                                                tiling_->alpha);
        }
        outQueue_.EnQue(out);
        gateupQueue_.FreeTensor(both);
    }

    __aicore__ inline void CopyOut(int64_t globalOffset, int64_t elements)
    {
        LocalTensor<T> out = outQueue_.DeQue<T>();
        DataCopy(yGm_[globalOffset], out, elements);
        outQueue_.FreeTensor(out);
    }

    const SituAndMulSparseTilingData* tiling_;
    GlobalTensor<T> xGm_, yGm_;
    GlobalTensor<int64_t> expertTokensGm_;
    TQue<QuePosition::VECIN, 1> gateupQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
};

// Definition of the runner declared above the entry. TPipe lifetime is the
// runner function scope, exactly the entry scope of the frozen registry
// kernel (constructed after the expert_tokens[0] load, destroyed after
// Process returns).
template <typename T>
__aicore__ inline void SituAndMulSparseRun(GM_ADDR x, GM_ADDR expertTokens, GM_ADDR y,
                                            const SituAndMulSparseTilingData* tiling,
                                            int64_t token0)
{
    TPipe pipe;
    SituAndMulSparseKernel<T> kernel;
    kernel.Init(x, expertTokens, y, tiling, &pipe);
    kernel.Process(token0);
}
