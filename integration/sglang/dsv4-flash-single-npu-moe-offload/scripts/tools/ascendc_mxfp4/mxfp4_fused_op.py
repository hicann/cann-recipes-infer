# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime wrapper for the fused AscendC MXFP4->W8A8 kernel, for kt_stream_prefill depool.

Builds libmxfp4fused.so on first use (bisheng), loads it via ctypes, and exposes:

  mxfp4_layer_to_nz_slots(c13, s13, c2, s2, hidden, inter, blockdim=40)
      -> (w13_nz, s13b, w2_nz, s2b)   # exactly the slot tensors npu_fused_experts consumes
                                       # w*_nz: FRACTAL_NZ int8 [E,IN,OUT];  s*b: bf16 [E,OUT]

Inputs are this layer's combined MXFP4 (device uint8):
  c13/s13: w13 = cat(w1,w3) codes [E,2I,H/2] + e8m0 scale [E,2I,H/32]
  c2/s2  : w2  codes [E,H,I/2] + e8m0 scale [E,H,I/32]

Reads MXFP4 once; one kernel pass per projection. Validated end-to-end (cos 0.99999976 vs fp32
golden through npu_fused_experts). See SPEC in tools/mxfp4_w8a8_op/.
"""
import collections
import ctypes
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mxfp4_fused_kernel.cpp"
_SO = _HERE / "libmxfp4fused.so"
_NZ = 29
_ACC = 512
_FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

# The fused kernel is launched via a raw ctypes <<<stream>>> call. With TASK_QUEUE_ENABLE=1 torch
# dispatches its ops through an async queue that is NOT ordered against that direct launch, so the
# post-step racing `out`/`osc` needs an explicit per-chunk host sync (correctness; but it stalls the
# host and serializes the convert with the rest of the forward -> slow prefill). With
# TASK_QUEUE_ENABLE=0 torch ops go straight to the stream, so kernel+post-step are FIFO-ordered and
# NO sync is needed (validated: deterministic + byte-equal, runs async -> fast). So only sync when
# the task queue is on.
_TQ_SYNC = os.environ.get("TASK_QUEUE_ENABLE", "1") != "0"

_lib = None
_lock = threading.Lock()
_consts_cache = {}


def _cann_home():
    """Resolve the CANN root the same way launch_ds4flash_npu.sh does: environment first
    (set_env.sh exports both ASCEND_TOOLKIT_HOME and ASCEND_HOME_PATH), then the standard
    install layouts, and hard-fail with guidance if none is usable. Never fall back to a
    hardcoded version-specific path: on a machine whose CANN sits elsewhere that silently
    compiles against the wrong toolkit instead of telling the user to source set_env.sh.
    """
    for env in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        home = os.environ.get(env)
        if home and os.path.isfile(os.path.join(home, "set_env.sh")):
            return home
    for cand in ("/usr/local/Ascend/ascend-toolkit/latest",
                 os.path.expanduser("~/Ascend/ascend-toolkit/latest")):
        if os.path.isfile(os.path.join(cand, "set_env.sh")):
            return cand
    raise RuntimeError(
        "无法确定 CANN 根目录(ASCEND_TOOLKIT_HOME)。请先 source CANN 的 set_env.sh,"
        " 或显式设置 export ASCEND_TOOLKIT_HOME=/path/to/Ascend/cann-9.0.0")


def _build():
    cann = _cann_home()
    tk = f"{cann}/aarch64-linux/tikcpp"
    inc = [f"{tk}/tikcfw", f"{tk}/tikcfw/impl", f"{tk}/tikcfw/interface", f"{tk}/tikcfw/lib",
           f"{cann}/aarch64-linux/include"]
    cmd = ["bisheng", "-x", "asc", "--cce-aicore-arch=dav-c220", "-O2", "-std=c++17", "-fPIC",
           "-shared", *[f"-I{p}" for p in inc], str(_SRC), "-o", str(_SO),
           f"-L{cann}/aarch64-linux/lib64", "-lruntime", "-lascendcl"]
    # capture_output swallows bisheng's diagnostics: a CalledProcessError alone just says
    # "returned non-zero", which is unusable for a kernel compile. Re-raise with the compiler
    # output attached so a build break is diagnosable from the server log.
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"找不到 bisheng 编译器(CANN 根 {cann})。请先 source CANN 的 set_env.sh 再启动。") from exc
    except subprocess.CalledProcessError as exc:
        out = (exc.stderr or b"").decode("utf-8", "replace") or (exc.stdout or b"").decode("utf-8", "replace")
        raise RuntimeError(
            f"编译 {_SRC.name} 失败(bisheng 退出码 {exc.returncode}),编译器输出:\n{out}") from exc


# Mirrors of the launch descriptors in mxfp4_fused_kernel.cpp. Field ORDER (not name) defines the
# layout: pointers first, uint32 fields last, so neither side has interior padding. get_lib()
# compares ctypes.sizeof() against the C sizeof() before the first launch -- a silent layout drift
# would turn into wild pointers rather than a clean error.
class _FusedArgs(ctypes.Structure):
    _fields_ = [("stream", ctypes.c_void_p),
                ("codes", ctypes.c_void_p),
                ("scale", ctypes.c_void_p),
                ("out", ctypes.c_void_p),
                ("oscale", ctypes.c_void_p),
                ("lut_lo", ctypes.c_void_p),
                ("lut_hi", ctypes.c_void_p),
                ("lut_e8", ctypes.c_void_p),
                ("sc_off", ctypes.c_void_p),
                ("blockdim", ctypes.c_uint32),
                ("rows", ctypes.c_uint32),
                ("half_len", ctypes.c_uint32),
                ("nb_count", ctypes.c_uint32),
                ("in_dim", ctypes.c_uint32)]


class _FusedBlkArgs(ctypes.Structure):
    _fields_ = [("stream", ctypes.c_void_p),
                ("blocks", ctypes.c_void_p),
                ("out", ctypes.c_void_p),
                ("oscale", ctypes.c_void_p),
                ("lut_lo", ctypes.c_void_p),
                ("lut_hi", ctypes.c_void_p),
                ("lut_e8", ctypes.c_void_p),
                ("sc_off", ctypes.c_void_p),
                ("code_off", ctypes.c_void_p),
                ("scale_off", ctypes.c_void_p),
                ("blockdim", ctypes.c_uint32),
                ("rows", ctypes.c_uint32),
                ("half_len", ctypes.c_uint32),
                ("nb_count", ctypes.c_uint32),
                ("in_dim", ctypes.c_uint32)]


def get_lib():
    """Build (if needed) and load the fused kernel .so. Thread-safe, idempotent."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            if not _SO.exists() or _SO.stat().st_mtime < _SRC.stat().st_mtime:
                _build()
            lib = ctypes.CDLL(str(_SO))
            lib.LaunchMxfp4Fused.restype = None
            lib.LaunchMxfp4Fused.argtypes = [ctypes.POINTER(_FusedArgs)]
            lib.LaunchMxfp4FusedBlk.restype = None
            lib.LaunchMxfp4FusedBlk.argtypes = [ctypes.POINTER(_FusedBlkArgs)]
            for fn, cls in ((lib.Mxfp4FusedArgsSize, _FusedArgs),
                            (lib.Mxfp4FusedBlkArgsSize, _FusedBlkArgs)):
                fn.restype = ctypes.c_uint32
                fn.argtypes = []
                if fn() != ctypes.sizeof(cls):
                    raise RuntimeError(
                        f"{cls.__name__} 与 {_SRC.name} 中的结构体布局不一致"
                        f"(C {fn()} 字节 vs ctypes {ctypes.sizeof(cls)} 字节);"
                        f" 请删除 {_SO.name} 让其重新编译")
            _lib = lib
    return _lib


def _consts(half, nb, dev):
    key = (half, nb, str(dev))
    if key in _consts_cache:
        return _consts_cache[key]
    b = np.arange(256, dtype=np.int64)
    lut_lo = _FP4[b & 0xF].astype(np.float32)
    lut_hi = _FP4[(b >> 4) & 0xF].astype(np.float32)
    lut_e8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(half, dtype=np.int64)
    sc_off = ((j >> 4) * 4).astype(np.int32)
    out = tuple(torch.from_numpy(a).to(dev) for a in (lut_lo, lut_hi, lut_e8, sc_off))
    _consts_cache[key] = out
    return out


_blk_consts_cache = {}


def _blk_consts(half, nb, dev):
    """code_off/scale_off: byte offsets of code j / scale block b in the half-cast GGUF block buffer
    ([nb,17] per row). Used by Mxfp4FusedBlk to de-interleave in UB via Gather."""
    key = (half, nb, str(dev))
    if key in _blk_consts_cache:
        return _blk_consts_cache[key]
    j = np.arange(half, dtype=np.int64)
    code_off = (((j // 16) * 17 + 1 + (j % 16)) * 2).astype(np.uint32)
    b = np.arange(nb, dtype=np.int64)
    scale_off = ((b * 17) * 2).astype(np.uint32)
    out = (torch.from_numpy(code_off).to(dev), torch.from_numpy(scale_off).to(dev))
    _blk_consts_cache[key] = out
    return out


_NZ_CHUNK = int(os.environ.get("KT_MXFP4_NZ_CHUNK", "32"))  # experts/chunk -> bounds HBM transient

# One projection's MXFP4 input, kept together so convert_proj takes a named group rather than a
# growing positional list (codes [E,OUT,half] uint8, scale [E,OUT,nb] e8m0, both on device).
Mxfp4Proj = collections.namedtuple("Mxfp4Proj", ["codes", "scale"])


def convert_proj(proj, in_dim, blockdim=40, packing="consecutive", out_nz=None):
    """
    One projection: MXFP4 codes/scale [E,OUT,*] -> (q_nz [E,in_dim,OUT] FRACTAL_NZ, oscale bf16).

    proj: an Mxfp4Proj(codes, scale) pair for this projection, both already on device.

    Chunked over experts so the transient (int8 planes + de-interleave + NZ cast) stays small —
    only the final [E,in_dim,OUT] NZ output is full-size (HBM-bounded like the W8A8 slot).

    out_nz: optional pre-allocated FRACTAL_NZ [E,in_dim,OUT] int8 buffer to write into (the
      reserved streaming slot). When given, no per-call ~GBs output allocation happens — the
      layer's NZ is produced straight into the reused slot (HBM budgeted once at load). Must
      already be NZ-format with matching shape. When None, a fresh buffer is allocated.

    packing: nibble layout of the code bytes — how the kernel's lo/hi planes map back to
      K-positions.
      "consecutive" (native safetensors): byte j -> Kpos 2j (lo), 2j+1 (hi)  -> interleave.
      "halfblock"   (GGUF block_mxfp4):   byte j -> Kpos g*32+jl (lo), +16 (hi) within its
                                           32-group -> per-group [lo0..15 | hi0..15] concat.
    Both decode the SAME K-ordered weights bit-for-bit; the kernel and scale->block mapping
    (sc_off) are packing-agnostic, so only this post-step rearrange differs (no .so change).
    """
    import torch_npu
    lib = get_lib()
    codes_dev, scale_dev = proj.codes, proj.scale
    dev = codes_dev.device
    num_experts, out_dim, half = codes_dev.shape
    nb = scale_dev.shape[2]
    half_p = in_dim // 2
    lut_lo, lut_hi, lut_e8, sc_off = _consts(half, nb, dev)
    st = torch.npu.current_stream().npu_stream

    oscale = torch.empty((num_experts, out_dim), dtype=torch.bfloat16, device=dev)
    for c in range(0, num_experts, _NZ_CHUNK):
        ce = min(c + _NZ_CHUNK, num_experts)
        exp_c = ce - c
        rows = exp_c * out_dim
        cd = codes_dev[c:ce].reshape(rows, half).contiguous()
        sd = scale_dev[c:ce].reshape(rows, nb).contiguous()
        out = torch.empty((rows, in_dim), dtype=torch.int8, device=dev)   # two planes [lo|hi]
        # Pad the oscale buffer to a whole number of ACC blocks: the kernel flushes each block
        # (including the tail one) as a full ACC-element DataCopy, which needs the padding to
        # land in-bounds. Only osc[:rows] is ever read back.
        rows_pad = (rows + _ACC - 1) // _ACC * _ACC
        osc = torch.empty((rows_pad,), dtype=torch.float32, device=dev)
        args = _FusedArgs(stream=st, codes=cd.data_ptr(), scale=sd.data_ptr(),
                          out=out.data_ptr(), oscale=osc.data_ptr(),
                          lut_lo=lut_lo.data_ptr(), lut_hi=lut_hi.data_ptr(),
                          lut_e8=lut_e8.data_ptr(), sc_off=sc_off.data_ptr(),
                          blockdim=blockdim, rows=rows, half_len=half, nb_count=nb, in_dim=in_dim)
        lib.LaunchMxfp4Fused(ctypes.byref(args))
        # See _TQ_SYNC: only needed when the task queue is on (then the raw ctypes launch is not
        # ordered against the torch post-step reading `out`/`osc`). With it off, stream FIFO orders
        # them and we skip the host stall -> async convert, fast prefill.
        if _TQ_SYNC:
            torch.npu.synchronize()
        # De-interleave the [lo|hi] planes (contiguous stack) then transpose OUT<->IN. The old depool
        # hot spot was (a) a strided 1-byte de-interleave scatter (~2.4s/layer) and (b) an int8
        # transpose that degenerates to a 1-byte gather (~0.6s, ~20GB/s). (a) is gone via the
        # contiguous stack; (b) is killed by transposing in fp16 (vectorized) and round-tripping
        # int8->fp16->int8 — exact because |q|<=127. Net post-step ~3s -> ~0.13s. The .contiguous()
        # is mandatory: feeding a transposed view to format_cast lays down WRONG NZ bytes on device
        # (looks fine via .cpu() which de-formats, but grouped_matmul reads garbage).
        lo, hi = out[:, :half_p], out[:, half_p:]
        if packing == "halfblock":
            nb_q = half_p // 16
            q = torch.cat([lo.reshape(rows, nb_q, 16), hi.reshape(rows, nb_q, 16)],
                          dim=2).reshape(exp_c, out_dim, in_dim)
        else:
            # consecutive interleave [E,OUT,in_dim]
            q = torch.stack([lo, hi], dim=2).reshape(exp_c, out_dim, in_dim)
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)      # [E,in_dim,OUT]
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((num_experts,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:rows].reshape(exp_c, out_dim).to(torch.bfloat16)
        # Second sync (task-queue-on only): let the osc read finish before the next chunk reuses it.
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


# pylint: disable=huawei-too-many-arguments
# The two mxfp4_layer_to_nz_slots* entry points keep a flat signature on purpose: they are the ABI
# the patched sglang streaming-prefill path calls (see 0003-sglang-streaming-prefill-depool.patch).
# Grouping the arguments here would have to change that call site inside the third-party patch, so
# the grouping stops at convert_proj (which takes an Mxfp4Proj) and these stay positional.
def mxfp4_layer_to_nz_slots(c13, s13, c2, s2, hidden, inter, blockdim=40, packing="consecutive",
                            out_w13=None, out_w2=None):
    """
    Full layer depool conversion -> (w13_nz, s13b, w2_nz, s2b), the exact tensors the streaming
    slot + npu_fused_experts consume (replacing the resident W8A8 pool). packing: see convert_proj
    ("consecutive" for native safetensors codes, "halfblock" for GGUF block_mxfp4 codes).
    out_w13/out_w2: optional pre-reserved NZ slots to convert into (no per-layer output alloc).
    """
    w13_nz, s13b = convert_proj(Mxfp4Proj(c13, s13), hidden, blockdim, packing, out_nz=out_w13)
    w2_nz, s2b = convert_proj(Mxfp4Proj(c2, s2), inter, blockdim, packing, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b


def convert_proj_blk(blocks_dev, in_dim, blockdim=40, out_nz=None):
    """
    One projection from RAW GGUF block_mxfp4 [E,OUT,nb*17] -> (q_nz [E,in_dim,OUT] FRACTAL_NZ,
    oscale bf16 [E,OUT]). The de-interleave (scale|codes per 17B block) is done IN-KERNEL
    (Mxfp4FusedBlk, UB Gather) -- no host/device de-interleave (the slow 16-of-17 strided int8
    copy). The kernel output `out` (two [lo|hi] planes) is byte-identical to the de-interleaved
    path, so the post-step is the same half-block rearrange.
    out_nz: optional pre-reserved NZ buffer.
    """
    import torch_npu
    lib = get_lib()
    dev = blocks_dev.device
    num_experts, out_dim, nb17 = blocks_dev.shape
    nb = nb17 // 17
    half = nb * 16
    half_p = in_dim // 2
    lut_lo, lut_hi, lut_e8, sc_off = _consts(half, nb, dev)
    code_off, scale_off = _blk_consts(half, nb, dev)
    st = torch.npu.current_stream().npu_stream
    oscale = torch.empty((num_experts, out_dim), dtype=torch.bfloat16, device=dev)
    for c in range(0, num_experts, _NZ_CHUNK):
        ce = min(c + _NZ_CHUNK, num_experts)
        exp_c = ce - c
        rows = exp_c * out_dim
        bd = blocks_dev[c:ce].reshape(rows, nb17).contiguous()
        out = torch.empty((rows, in_dim), dtype=torch.int8, device=dev)
        rows_pad = (rows + _ACC - 1) // _ACC * _ACC     # ACC-padded: see convert_proj
        osc = torch.empty((rows_pad,), dtype=torch.float32, device=dev)
        args = _FusedBlkArgs(stream=st, blocks=bd.data_ptr(), out=out.data_ptr(),
                             oscale=osc.data_ptr(), lut_lo=lut_lo.data_ptr(),
                             lut_hi=lut_hi.data_ptr(), lut_e8=lut_e8.data_ptr(),
                             sc_off=sc_off.data_ptr(), code_off=code_off.data_ptr(),
                             scale_off=scale_off.data_ptr(),
                             blockdim=blockdim, rows=rows, half_len=half, nb_count=nb,
                             in_dim=in_dim)
        lib.LaunchMxfp4FusedBlk(ctypes.byref(args))
        if _TQ_SYNC:
            torch.npu.synchronize()
        lo, hi = out[:, :half_p], out[:, half_p:]
        nb_q = half_p // 16
        q = torch.cat([lo.reshape(rows, nb_q, 16), hi.reshape(rows, nb_q, 16)],
                      dim=2).reshape(exp_c, out_dim, in_dim)
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((num_experts,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:rows].reshape(exp_c, out_dim).to(torch.bfloat16)
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, bd
    return out_nz, oscale


# pylint: disable=huawei-too-many-arguments   # same patched-call-site ABI as mxfp4_layer_to_nz_slots
def mxfp4_layer_to_nz_slots_blk(blk13, blk2, hidden, inter, blockdim=40, out_w13=None,
                                out_w2=None):
    """
    Full layer conversion from RAW GGUF blocks (in-kernel de-interleave) -> slot tensors.
    blk13 = cat(gate,up) blocks [E,2*inter,nb_hidden*17]; blk2 = down blocks [E,hidden,nb_inter*17].
    """
    w13_nz, s13b = convert_proj_blk(blk13, hidden, blockdim, out_nz=out_w13)
    w2_nz, s2b = convert_proj_blk(blk2, inter, blockdim, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b
