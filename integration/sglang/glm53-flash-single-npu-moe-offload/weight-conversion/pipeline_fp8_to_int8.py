#!/usr/bin/env python
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
"""FP8 -> INT8 in one pass per shard, without ever writing the BF16 intermediate.

Why not the two documented steps back to back: the full BF16 checkpoint is 598.5 GiB,
and 305.8 (FP8) + 598.5 (BF16) + 306.1 (INT8) = 1210 GiB does not fit the 983 GiB
volume. fp8_to_bf16.py --delete-source would fit, but it unlinks the vendor FP8
shards, and those are the only copy on either A3 machine -- the 8-card box kept
nothing but the index, and the MXFP4 leg still needs them.

Staging BF16 per shard and unlinking it immediately does fit, but measurement showed
it is also the bottleneck: with reads fully served from page cache, the volume writes
~350 MB/s, so the 598.5 GiB of BF16 that gets deleted moments later costs ~30 min of
pure waste. Composing the two transforms in memory drops total writes to the 306.1
GiB of INT8 that we actually keep.

The arithmetic is not reimplemented. Each tensor goes through the upstream kernels
unchanged -- fp8_to_bf16.dequantize then bf16_to_int8_ct.quantize_channel -- in the
same order and on the same selection the two-step flow would apply: a tensor is
dequantized iff the FP8 checkpoint stored a weight_scale_inv beside it, and that same
index is what names the tensors to quantize. The index/config tail below is the logic
from bf16_to_int8_ct.main().

Shards are independent (every weight_scale_inv lives in the same shard as its weight),
which is what makes per-shard streaming safe.

Output shards are written .partial and renamed, so a killed run leaves no truncated
shard for the resume path to mistake for a finished one.

Example::

    <python> pipeline_fp8_to_int8.py \\
        --src /path/to/GLM-5.3-Flash-FP8 \\
        --dst /path/to/GLM-5.3-Flash-W8A8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TOOLS = Path(__file__).resolve().parent / "tools"
f2b = load("fp8_to_bf16", TOOLS / "fp8_to_bf16.py")
b2i = load("bf16_to_int8_ct", TOOLS / "bf16_to_int8_ct.py")

ESZ = {"I8": 1, "BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "BOOL": 1, "U8": 1}


def free_gib(p: Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1024**3


def one_shard(job):
    src_dir, shard, block, to_quantize, dst_dir = job
    t0 = time.time()
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(Path(src_dir) / shard), framework="pt") as f:
        names = set(f.keys())
        for name in sorted(names):
            if name.endswith(f2b.SCALE_SUFFIX):
                continue  # consumed by dequantize, dropped from the output
            tensor = f.get_tensor(name)
            scale_name = name + "_scale_inv" if name.endswith(".weight") else None
            if scale_name in names:
                w = f2b.dequantize(tensor, f.get_tensor(scale_name), block)
            else:
                if tensor.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"{name} is fp8 but has no {scale_name}")
                w = tensor
            if name in to_quantize:
                q, scale = b2i.quantize_channel(w)
                out[name] = q
                out[name[: -len("weight")] + "weight_scale"] = scale
            else:
                out[name] = w
    dst = Path(dst_dir) / shard
    tmp = dst.with_suffix(dst.suffix + ".partial")
    save_file(out, str(tmp), metadata={"format": "pt"})
    tmp.rename(dst)
    nbytes = sum(v.numel() * v.element_size() for v in out.values())
    return shard, {k: shard for k in out}, nbytes, time.time() - t0


def scan_done(path: Path) -> tuple[dict[str, str], int]:
    """weight_map / byte count for a shard already on disk, read from its header."""
    with open(path, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    wmap, total = {}, 0
    for k, m in hdr.items():
        if k == "__metadata__":
            continue
        wmap[k] = path.name
        n = 1
        for d in m["shape"]:
            n *= d
        total += n * ESZ[m["dtype"]]
    return wmap, total


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", type=Path, required=True, help="source FP8 checkpoint directory"
    )
    ap.add_argument(
        "--dst", type=Path, required=True, help="output W8A8 checkpoint directory"
    )
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", action="append", default=None, help="convert just this shard")
    ap.add_argument("--min-free-gib", type=float, default=40.0)
    args = ap.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    block = json.loads((args.src / "config.json").read_text())["quantization_config"]["weight_block_size"]
    to_quantize = frozenset(
        k[: -len("_scale_inv")] for k in weight_map if k.endswith("weight_scale_inv")
    )
    LOGGER.info(f"weight_block_size={block}  tensors_to_quantize={len(to_quantize)}")

    all_shards = sorted(set(weight_map.values()))
    args.dst.mkdir(parents=True, exist_ok=True)
    for stale in args.dst.glob("*.partial"):
        stale.unlink()

    if args.only:
        todo = [s for s in all_shards if s in set(args.only)]
        all_shards = todo
    else:
        todo = [s for s in all_shards if not (args.dst / s).exists()]

    LOGGER.info(f"{len(all_shards)} shards, {len(todo)} to convert")
    if free_gib(args.dst) < args.min_free_gib:
        LOGGER.error("ABORT: not enough free space")
        return 1

    done_map: dict[str, str] = {}
    total_bytes = 0
    for s in all_shards:
        if s not in todo:
            wmap, nb = scan_done(args.dst / s)
            done_map.update(wmap)
            total_bytes += nb

    t0 = time.time()
    jobs = [(str(args.src), s, block, to_quantize, str(args.dst)) for s in todo]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (shard, wmap, nbytes, secs) in enumerate(ex.map(one_shard, jobs), 1):
            done_map.update(wmap)
            total_bytes += nbytes
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(todo) - i) / rate if rate else 0
            LOGGER.info(f"  [{i}/{len(todo)}] {shard} {nbytes/1024**3:5.1f} GiB {secs:5.0f}s  "
                        f"(elapsed {time.time()-t0:5.0f}s, eta {eta/60:4.1f}m, "
                        f"free {free_gib(args.dst):.0f} GiB)")

    if args.only:
        LOGGER.info("partial run: index/config not written")
        return 0

    # --- tail: the logic from bf16_to_int8_ct.main() ---
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": done_map}, indent=2))

    # fp8_to_bf16 copies these into the BF16 dir; bf16_to_int8_ct then carries every
    # non-safetensors file forward. Same net set, taken straight from the source.
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "chat_template.jinja", "processor_config.json", "configuration.json",
                  "LICENSE"):
        if (args.src / extra).exists():
            shutil.copy2(args.src / extra, args.dst / extra)

    cfg = json.loads((args.src / "config.json").read_text())
    fp8_quant = cfg.pop("quantization_config", {})
    quant = dict(b2i.QUANT_CONFIG)
    quant["ignore"] = fp8_quant.get("modules_to_not_convert", [])
    cfg["quantization_config"] = quant
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2))

    LOGGER.info(f"\nwrote {len(all_shards)} shards, {total_bytes/1024**3:.1f} GiB "
                f"({total_bytes} bytes), {len(done_map)} tensors, {time.time()-t0:.0f}s -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
