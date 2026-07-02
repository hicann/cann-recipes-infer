#!/usr/bin/env python3
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
"""
P2 acceptance for the MXFP4 converter: prove the GGUF repack is LOSSLESS.

Two checks:
  1. unit: random [N,K/2] native bytes -> repack -> dequant(GGUF semantics) must
     equal dequant(native semantics), element-wise. Pure logic, no files.
  2. layer: for a real converted layer GGUF, dequant each proj tensor and compare
     element-wise to the native checkpoint dequant (a subset of experts).

Native semantics  : value = FP4_TABLE[nibble] * 2^(e-127), byte i -> Kpos 2i(lo),2i+1(hi)
GGUF  semantics   : value = kvalues_mxfp4[nibble] * 2^(e-128), qs[j] -> Kpos j(lo),j+16(hi)
These are algebraically identical (kvalues_mxfp4 = 2*FP4_TABLE), so equality is bit-exact.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("kt.tools.verify_mxfp4_layer")

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The two inserts below use insert(0) on purpose: the vendored gguf-py (pinned to
# llama.cpp b3173) MUST shadow any pip-installed `gguf` (one IS installed).
_GGUF_PY = str(_REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py")
_TOOLS = str(_REPO_ROOT / "tools")
sys.path.insert(0, _GGUF_PY)  # pylint: disable=no-use-sys-path-insert  # vendored gguf-py must shadow installed gguf
sys.path.insert(0, _TOOLS)  # pylint: disable=no-use-sys-path-insert  # vendored gguf-py must shadow installed gguf


def _setup_logging() -> None:
    """INFO -> stdout, WARNING+ -> stderr (preserves the original print stream split)."""
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[out, err])

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)


def e8m0_to_fp32(e: np.ndarray) -> np.ndarray:
    """2^(e-127) exactly (matches loader _ue8m0_to_bf16 semantics)."""
    bits = (e.astype(np.uint32)) << 23
    return bits.view(np.float32)


def e8m0_to_fp32_half(e: np.ndarray) -> np.ndarray:
    """2^(e-128) == 2^(e-127)*0.5 (matches ggml_e8m0_to_fp32_half)."""
    e = e.astype(np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - 1) << 23).astype(np.uint32)
    return bits.view(np.float32)


def dequant_native(w_u8: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """[N,K/2] bytes + [N,K/32] e8m0 -> [N,K] float32 (native consecutive nibble order)."""
    n_rows, kh = w_u8.shape
    k_dim = kh * 2
    lo = FP4_TABLE[(w_u8 & 0x0F)]   # [N, K/2] -> Kpos 0,2,...
    hi = FP4_TABLE[(w_u8 >> 4)]     # [N, K/2] -> Kpos 1,3,...
    vals = np.empty((n_rows, k_dim), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    scale = e8m0_to_fp32(s_u8)               # [N, nb]
    scale = np.repeat(scale, 32, axis=1)     # [N, K]
    return vals * scale


def dequant_gguf_blocks(packed_row: np.ndarray, k_dim: int) -> np.ndarray:
    """[N, nb*17] uint8 -> [N, K] float32 (GGUF half-block order)."""
    n_rows = packed_row.shape[0]
    nb = k_dim // 32
    blk = packed_row.reshape(n_rows, nb, 17)
    e = blk[..., 0]                  # [N, nb]
    qs = blk[..., 1:]               # [N, nb, 16]
    lo = KVALUES_MXFP4[(qs & 0x0F)].astype(np.float32)   # Kpos j (0..15)
    hi = KVALUES_MXFP4[(qs >> 4)].astype(np.float32)     # Kpos j+16
    half = e8m0_to_fp32_half(e)[..., None]               # [N, nb, 1]
    g = np.concatenate([lo * half, hi * half], axis=-1)   # [N, nb, 32]
    return g.reshape(n_rows, k_dim)


def unit_test(seed: int = 0) -> int:
    from convert_mxfp4_layer_to_gguf import _repack_consecutive_to_halfblock
    rng = np.random.default_rng(seed)
    n_rows, k_dim = 7, 128
    w = rng.integers(0, 256, size=(n_rows, k_dim // 2), dtype=np.uint8)
    s = rng.integers(100, 140, size=(n_rows, k_dim // 32), dtype=np.uint8)
    qs = _repack_consecutive_to_halfblock(w)                # [N, K/2]
    nb = k_dim // 32
    packed = np.concatenate([s.reshape(n_rows, nb, 1), qs.reshape(n_rows, nb, 16)], axis=-1).reshape(n_rows, nb * 17)
    a = dequant_native(w, s)
    b = dequant_gguf_blocks(packed, k_dim)
    if not np.array_equal(a, b):
        bad = int((a != b).sum())
        logger.error(f"[unit] FAIL: {bad}/{a.size} elements differ (max abs {np.abs(a-b).max()})")
        return 1
    logger.info(f"[unit] PASS: repack lossless, {a.size} elements bit-exact (N={n_rows},K={k_dim})")
    return 0


def layer_test(gguf_path: Path, model_dir: Path, layer_idx: int, n_experts_check: int) -> int:
    import gguf
    from convert_mxfp4_layer_to_gguf import _load_weight_map, _detect_experts_prefix, _open_shard, _as_u8

    reader = gguf.GGUFReader(str(gguf_path))
    tmap = {t.name: t for t in reader.tensors}
    weight_map = _load_weight_map(model_dir)
    prefix = _detect_experts_prefix(weight_map, layer_idx)
    cache: dict = {}

    # (gguf tensor name, native proj, K)
    projs = [(f"blk.{layer_idx}.ffn_gate_exps.weight", "w1", 4096),
             (f"blk.{layer_idx}.ffn_up_exps.weight", "w3", 4096),
             (f"blk.{layer_idx}.ffn_down_exps.weight", "w2", 2048)]

    for gname, proj, k_dim in projs:
        t = tmap[gname]
        # GGUF reader gives ne order [K, N, E] (fastest dim first); data is C-order [E, N, K].
        k_reduced, n_rows, num_experts = (int(x) for x in t.shape)
        if k_reduced != k_dim:
            raise ValueError(f"{gname} K {k_reduced} != {k_dim}")
        nb = k_dim // 32
        packed = np.asarray(t.data).reshape(num_experts, n_rows, nb * 17)
        ok = True
        for e in range(min(n_experts_check, num_experts)):
            wk = f"{prefix}.{e}.{proj}.weight"
            sk = f"{prefix}.{e}.{proj}.scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            w_u8 = _as_u8(h.get_tensor(wk))
            s_u8 = _as_u8(h.get_tensor(sk))
            a = dequant_native(w_u8, s_u8)
            b = dequant_gguf_blocks(packed[e], k_dim)
            if not np.array_equal(a, b):
                bad = int((a != b).sum())
                logger.error(f"  [{gname}] expert {e}: FAIL {bad}/{a.size} differ (max abs {np.abs(a-b).max()})")
                ok = False
        logger.info(
            f"[layer] {gname}: {'PASS' if ok else 'FAIL'} "
            f"(checked {min(n_experts_check, num_experts)} experts, "
            f"shape E={num_experts} N={n_rows} K={k_dim})")
        if not ok:
            return 1
    return 0


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=None, help="Converted layer GGUF (layer_test)")
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--n-experts-check", type=int, default=8)
    args = ap.parse_args()

    rc = unit_test()
    if rc != 0:
        return rc
    if args.gguf is not None:
        rc = layer_test(args.gguf.expanduser().resolve(), args.model_dir.expanduser().resolve(),
                        args.layer_idx, args.n_experts_check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
