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
MXFP4 GGUF 全集校验（开源交付的验收入口）。三级校验，逐级加强：

  L1 完整性（秒级，无依赖）：43 层文件齐全 + 每层大小精确等于期望值。
     批量/并发转换被中断会留下截断文件（实测踩过：layer9 被写成 576B），
     大小不对 = 必须重转该层。
  L2 指纹（分钟级，无依赖）：sha256 与发布清单逐层比对。
     转换是字节级确定性的（同一 checkpoint + 本仓库 gguf-py 重转 → byte-identical，
     已验证），所以指纹不匹配 = 转换环境/输入有差异，不要带病上线。
  L3 数值（需原生 checkpoint 在场）：调 verify_mxfp4_layer.py 抽样若干层做
     GGUF 反量化 vs 原生反量化的逐元素 bit-exact 对账（无损 repack 的最强证明）。

用法::

  # L1+L2（推荐的部署前标准动作；清单随仓库发布在 tools/mxfp4_gguf_sha256.txt）
  python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache \
      --sha256-manifest tools/mxfp4_gguf_sha256.txt

  # 只做 L1（赶时间/磁盘慢）
  python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache --skip-sha256

  # 加 L3 深度抽查（需 --model-dir 指向原生 MXFP4 checkpoint）
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

# DSv4-Flash 形状下单层 GGUF 的精确字节数（256 专家 × (2×[2048,4096]+[4096,2048]) MXFP4 + header）
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
    ap.add_argument("--dir", type=Path, required=True, help="存放 dsv4_layer{L}_mxfp4.gguf 的目录")
    ap.add_argument("--name-tpl", type=str, default="dsv4_layer{L}_mxfp4.gguf")
    ap.add_argument("--expected-size", type=int, default=EXPECTED_SIZE)
    ap.add_argument("--sha256-manifest", type=Path, default=None,
                    help="发布的 sha256 清单（`sha256sum` 格式）；不传且未 --skip-sha256 时仅告警")
    ap.add_argument("--skip-sha256", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="sha256 并行度")
    ap.add_argument("--deep", type=int, default=0, help="L3 逐元素抽查层数（0 关闭；均匀抽样）")
    ap.add_argument("--model-dir", type=Path, default=None, help="L3 需要：原生 MXFP4 checkpoint 目录")
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

    logger.info(f"\nRESULT: {'FAIL — 见上方明细，修复后重跑' if fail else 'PASS — 权重集可部署'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
