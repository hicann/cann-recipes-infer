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
"""Dequantize a blockwise-FP8 checkpoint to BF16, one shard at a time.

GLM-5.3-Flash ships as FP8 E4M3 with ``weight_block_size=[128, 128]``: every
quantized ``W.weight`` of shape [N, K] carries a ``W.weight_scale_inv`` of shape
[ceil(N/128), ceil(K/128)], and the BF16 value is ``fp8 * scale`` broadcast over
each 128x128 block. All 37338 scales live in the same shard as their weight, so
a shard converts independently of every other shard.

Tensors that are already BF16 (the ViT tower, norms, lm_head) are copied through
untouched; the F32 scale tensors are dropped from the output.

Source shards are never deleted -- reclaim them with --delete-source only after
the output has been verified.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)
SCALE_SUFFIX = ".weight_scale_inv"


def dequantize(weight: torch.Tensor, scale_inv: torch.Tensor, block: list[int]) -> torch.Tensor:
    """fp8 [N, K] x f32 [ceil(N/bn), ceil(K/bk)] -> bf16 [N, K]."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    bn, bk = block
    n, k = weight.shape
    expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale_inv.shape)} != expected {expected} for weight {(n, k)}")
    # Expand the per-block scale to per-element, then trim the tail block, which
    # is only partially covered when a dimension is not a multiple of the block.
    full = scale_inv.to(torch.float32).repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    return (weight.to(torch.float32) * full[:n, :k]).to(torch.bfloat16)


def convert_shard(src: Path, dst: Path, block: list[int]) -> dict[str, int]:
    stats = {"dequantized": 0, "copied": 0, "scales_dropped": 0}
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt") as f:
        names = set(f.keys())
        for name in sorted(names):
            if name.endswith(SCALE_SUFFIX):
                stats["scales_dropped"] += 1
                continue
            tensor = f.get_tensor(name)
            scale_name = name + "_scale_inv" if name.endswith(".weight") else None
            if scale_name in names:
                out[name] = dequantize(tensor, f.get_tensor(scale_name), block)
                stats["dequantized"] += 1
            else:
                if tensor.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"{name} is fp8 but has no {scale_name}")
                out[name] = tensor
                stats["copied"] += 1
    tmp = dst.with_suffix(dst.suffix + ".partial")
    save_file(out, str(tmp), metadata={"format": "pt"})
    tmp.rename(dst)
    return stats


def free_gib(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N shards")
    ap.add_argument("--only", action="append", default=None, help="convert just this shard filename (repeatable)")
    ap.add_argument("--delete-source", action="store_true",
                    help="unlink each source shard once its output is written (IRREVERSIBLE)")
    ap.add_argument("--min-free-gib", type=float, default=20.0,
                    help="abort before a shard if free space would drop below this")
    args = ap.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    block = json.loads((args.src / "config.json").read_text())["quantization_config"]["weight_block_size"]
    LOGGER.info(f"weight_block_size = {block}")

    shards = sorted(set(weight_map.values()))
    if args.only:
        shards = [s for s in shards if s in set(args.only)]
    if args.limit is not None:
        shards = shards[: args.limit]
    args.dst.mkdir(parents=True, exist_ok=True)

    totals = {"dequantized": 0, "copied": 0, "scales_dropped": 0}
    for i, shard in enumerate(shards, 1):
        out_path = args.dst / shard
        if out_path.exists():
            LOGGER.info(f"[{i}/{len(shards)}] {shard}: already present, skipping")
            continue
        if free_gib(args.dst) < args.min_free_gib:
            LOGGER.error(f"ABORT: only {free_gib(args.dst):.1f} GiB free at {args.dst}")
            return 1
        stats = convert_shard(args.src / shard, out_path, block)
        for k, v in stats.items():
            totals[k] += v
        size = out_path.stat().st_size / 1024**3
        LOGGER.info(f"[{i}/{len(shards)}] {shard}: dequant={stats['dequantized']} "
                    f"copy={stats['copied']} drop={stats['scales_dropped']} -> {size:.2f} GiB "
                    f"(free {free_gib(args.dst):.0f} GiB)")
        if args.delete_source:
            (args.src / shard).unlink()

    # The output index drops the scale entries; everything else keeps its shard.
    if not args.only and args.limit is None:
        new_map = {k: v for k, v in weight_map.items() if not k.endswith(SCALE_SUFFIX)}
        (args.dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=2)
        )
        config = json.loads((args.src / "config.json").read_text())
        config.pop("quantization_config", None)
        (args.dst / "config.json").write_text(json.dumps(config, indent=2))
        for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                      "chat_template.jinja", "processor_config.json", "configuration.json", "LICENSE"):
            if (args.src / extra).exists():
                shutil.copy2(args.src / extra, args.dst / extra)
        LOGGER.info(f"wrote index ({len(new_map)} tensors), cleaned config, copied tokenizer files")

    LOGGER.info(f"TOTALS: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
