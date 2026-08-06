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
Validate the full MXFP4 GGUF set. Three levels, each stronger than the last:

  L1 completeness (seconds, no dependencies): all 43 layer files present and each exactly the
     expected size. An interrupted batch/parallel conversion leaves truncated files (one run
     left layer9 at 576 B), so a wrong size means that layer must be reconverted.
  L2 fingerprint (minutes, no dependencies): compare each layer's sha256 against the published
     manifest. Conversion is byte-deterministic (reconverting the same checkpoint with this
     repo's gguf-py is byte-identical, verified), so a mismatch means the conversion input or
     environment differed and the set should not be deployed.
  L3 numerical (requires the native checkpoint): calls verify_mxfp4_layer.py on a sample of
     layers to compare GGUF dequantisation against native dequantisation element by element,
     bit-exact -- the strongest evidence that the repack is lossless.

Usage::

  # L1+L2, the recommended pre-deployment check; the manifest ships as tools/mxfp4_gguf_sha256.txt
  python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache \
      --sha256-manifest tools/mxfp4_gguf_sha256.txt

  # L1 only (in a hurry, or slow disk)
  python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache --skip-sha256

  # add the L3 deep sample (--model-dir must point at the native MXFP4 checkpoint)
  python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache \
      --sha256-manifest tools/mxfp4_gguf_sha256.txt \
      --deep 3 --model-dir /path/to/DeepSeek-V4-Flash
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("kt.tools.verify_mxfp4_gguf_set")

# Exact byte size of one GGUF layer at DSv4-Flash shapes (256 experts x (2x[2048,4096] +
# [4096,2048]) MXFP4, plus the header)
EXPECTED_SIZE = 3_422_552_640
LAYERS = list(range(43))


def _setup_logging() -> None:
    """INFO -> stdout, WARNING+ -> stderr (preserves the original print stream split)."""
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[out, err])


def _sha256_file(path: Path) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path.name, h.hexdigest()


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="directory holding the dsv4_layer{L}_mxfp4.gguf files")
    ap.add_argument("--name-tpl", type=str, default="dsv4_layer{L}_mxfp4.gguf")
    ap.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    ap.add_argument("--sha256-manifest", type=Path, default=None,
                    help="published sha256 manifest (`sha256sum` format); without it, and "
                         "without --skip-sha256, L2 only warns")
    ap.add_argument("--skip-sha256", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="sha256 parallelism")
    ap.add_argument("--deep", type=int, default=0,
                    help="number of layers for the L3 element-wise sample (0 disables; sampled evenly)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="required by L3: the native MXFP4 checkpoint directory")
    args = ap.parse_args()

    d = args.dir.expanduser().resolve()
    fail = False

    # ---- L1: presence + exact size ----
    logger.info(f"[L1] checking 43 files in {d} (expected size {args.expected_size}) ...")
    missing, badsize = [], []
    for layer_idx in LAYERS:
        p = d / args.name_tpl.format(L=layer_idx)
        if not p.is_file():
            missing.append(layer_idx)
        elif p.stat().st_size != args.expected_size:
            badsize.append((layer_idx, p.stat().st_size))
    if missing:
        logger.info(f"[L1] FAIL missing layers: {missing}")
        fail = True
    if badsize:
        logger.info(f"[L1] FAIL wrong-size layers (truncated convert? re-convert these): {badsize}")
        fail = True
    if not missing and not badsize:
        logger.info("[L1] PASS — 43/43 present, all sizes exact")

    # ---- L2: sha256 vs manifest ----
    if not args.skip_sha256 and not fail:
        if args.sha256_manifest is None:
            logger.info("[L2] SKIP — no manifest given (pass --sha256-manifest or --skip-sha256)")
        else:
            ref = {}
            for line in args.sha256_manifest.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    ref[Path(parts[-1]).name] = parts[0]
            logger.info(f"[L2] hashing 43 files with {args.jobs} workers (~138GiB, takes a few minutes) ...")
            mismatch = []
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {
                    ex.submit(_sha256_file, d / args.name_tpl.format(L=layer_idx)): layer_idx
                    for layer_idx in LAYERS
                }
                for fu in as_completed(futs):
                    name, hx = fu.result()
                    want = ref.get(name)
                    if want is None:
                        mismatch.append((name, "NOT-IN-MANIFEST"))
                    elif want != hx:
                        mismatch.append((name, f"got {hx[:16]}.. want {want[:16]}.."))
            if mismatch:
                logger.info(f"[L2] FAIL {len(mismatch)} mismatches: {mismatch[:5]}")
                fail = True
            else:
                logger.info("[L2] PASS — all 43 sha256 match the manifest")

    # ---- L3: element-wise vs native checkpoint ----
    if args.deep > 0 and not fail:
        if args.model_dir is None:
            logger.info("[L3] FAIL — --deep needs --model-dir")
            fail = True
        else:
            here = Path(__file__).resolve().parent
            step = max(1, len(LAYERS) // args.deep)
            sample = LAYERS[::step][: args.deep]
            logger.info(f"[L3] element-wise check on layers {sample} (lossless => bit-exact required)")
            for layer_idx in sample:
                r = subprocess.run(
                    [sys.executable, str(here / "verify_mxfp4_layer.py"),
                     "--gguf", str(d / args.name_tpl.format(L=layer_idx)),
                     "--model-dir", str(args.model_dir), "--layer-idx", str(layer_idx),
                     "--n-experts-check", "4"],
                    capture_output=True, text=True,
                )
                ok = r.returncode == 0 and "FAIL" not in (r.stdout + r.stderr)
                logger.info(f"[L3] layer {layer_idx}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    logger.info((r.stdout + r.stderr)[-800:])
                    fail = True

    verdict = "FAIL - see the details above, fix and rerun" if fail else "PASS - the weight set is deployable"
    logger.info(f"\nRESULT: {verdict}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
