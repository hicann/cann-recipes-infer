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
"""Quantize the vendor FP8 checkpoint to MXFP4 MoE + INT8 everything-else, shard by shard.

One lossy step per tensor. The FP8 shards dequantize to fp32 in memory and are
re-quantized straight to the target format; no BF16 checkpoint is materialized.
That intermediate only ever existed because BF16 was a serving target in its own
right, not because the conversion needed it -- every shard carries its own
``weight_scale_inv``, so a shard converts independently of every other shard.

Which tensors are quantized is not a judgement call: a weight is quantized iff
the vendor shipped a ``weight_scale_inv`` beside it (37338 tensors). Re-deriving
that set from module-name patterns would be a second source of truth.

    MoE experts (37281)   -> MXFP4, OCP MX floor rule, block 32 along in_features
    everything else (57)  -> INT8, symmetric per-output-channel, absmax/127
    unquantized tensors   -> copied through

Both quantizers are verified against a reference:

  * The MXFP4 codes are bit-identical to HiFloat4's ``QType('mxfp4')`` kernel, so
    a checkpoint written here is the same object the accuracy study measured.
  * The INT8 codes are bit-identical to the existing W8A8 checkpoint, which went
    the long way round through BF16 -- lossless here because the vendor's FP8
    block scales are all exact powers of two, so ``fp8 * scale`` keeps its four
    significant bits and BF16's eight hold it exactly.

    $ <python> fp8_to_mxfp4_moe.py \\
        --src /path/to/GLM-5.3-Flash-FP8 \\
        --dst /path/to/GLM-5.3-Flash-MXFP4
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOGGER = logging.getLogger(__name__)
SCALE_INV = ".weight_scale_inv"
BLOCK = 32  # MX block, along in_features

# E2M1: the eight magnitudes an MXFP4 element can take, and the midpoints that
# decide which one a scaled value rounds to.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# floor(log2(6.0)) -- the OCP MX shared-exponent rule is
# shared_exp = floor(log2(block_amax)) - emax_elem
EMAX_ELEM = 2.0


def is_moe(name: str) -> bool:
    return ".mlp.experts." in name or "shared_experts" in name


def dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor, block: list[int]) -> torch.Tensor:
    """fp8 [N, K] x f32 [ceil(N/bn), ceil(K/bk)] -> fp32 [N, K]."""
    bn, bk = block
    n, k = w.shape
    expected = ((n + bn - 1) // bn, (k + bk - 1) // bk)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale_inv.shape)} != expected {expected}")
    s = scale_inv.to(torch.float32).repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return w.to(torch.float32) * s


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 with the OCP MX floor rule. Returns (packed uint8 [out, in/2], E8M0 uint8 [out, in/32]).

    Two elements per byte, the even index in the low nibble. The stored scale is
    the shared exponent biased by 127, so a reader reconstructs an element as
    ``E2M1[code & 7] * (-1)**(code >> 3) * 2**(scale - 127)``.
    """
    out_f, in_f = w.shape
    if in_f % BLOCK:
        raise ValueError(f"in_features {in_f} is not a multiple of the MX block {BLOCK}")

    x = w.reshape(-1, BLOCK).float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    # An all-zero block has no exponent to derive; 0 reproduces it exactly.
    exp = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp_min(1e-38))) - EMAX_ELEM,
        torch.zeros_like(amax),
    ).clamp(-127, 127)

    scaled = x / torch.exp2(exp)
    # How many midpoints the magnitude reaches == its index into E2M1, saturating
    # at 6.0, which is what the MX format does with an out-of-range element.
    # ``>=`` not ``>``: ties round away from zero. This is not a detail -- the
    # vendor's FP8 values are dyadic (four significant bits, power-of-two block
    # scale), so after scaling they land exactly on a midpoint for 20.6% of
    # elements, and rounding those the other way would corrupt a fifth of the
    # codes. Quantizing from the INT8 checkpoint instead hides this: its values
    # almost never land on a midpoint, so both rules agree there.
    ordv = (scaled.abs().unsqueeze(-1) >= E2M1_BOUNDS.to(x.device)).sum(-1)
    codes = (ordv | ((scaled < 0).to(ordv.dtype) << 3)).to(torch.uint8)

    codes = codes.reshape(out_f, in_f)
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    e8m0 = (exp.reshape(out_f, in_f // BLOCK) + 127).to(torch.uint8).contiguous()
    return packed, e8m0


def dequantize_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_mxfp4`; the round-trip check and the reference reader."""
    out_f, half = packed.shape
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.empty((out_f, half * 2), dtype=torch.uint8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    mag = E2M1.to(codes.device)[(codes & 0x07).long()]
    sign = torch.where((codes & 0x08) > 0, -1.0, 1.0)
    scale = torch.exp2(e8m0.float() - 127).repeat_interleave(BLOCK, dim=1)
    return sign * mag * scale


def quantize_int8_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8, identical to bf16_to_int8_ct.quantize_channel."""
    f = w.to(torch.float32)
    absmax = f.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(f / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


class ShardResult(NamedTuple):
    name: str
    weight_map: dict[str, str]
    nbytes: int
    n_mx: int
    n_i8: int
    seconds: float


def convert_shard(job) -> ShardResult:
    src, dst, block, verify, moe_format = job
    t0 = time.time()
    out: dict[str, torch.Tensor] = {}
    n_mx = n_i8 = 0

    with safe_open(str(src), framework="pt") as f:
        names = set(f.keys())
        for name in sorted(names):
            if name.endswith(SCALE_INV):
                continue  # consumed with its weight
            sname = name[: -len("weight")] + "weight_scale_inv" if name.endswith("weight") else None
            if sname is None or sname not in names:
                out[name] = f.get_tensor(name)  # not quantized by the vendor
                continue

            w = dequant_fp8(f.get_tensor(name), f.get_tensor(sname), block)
            base = name[: -len("weight")]
            if is_moe(name) and moe_format == "mxfp4":
                packed, e8m0 = quantize_mxfp4(w)
                if verify:
                    back = dequantize_mxfp4(packed, e8m0)
                    repacked, rescaled = quantize_mxfp4(back)
                    if not torch.equal(packed, repacked):
                        raise ValueError("MXFP4 packing round-trip failed")
                    if not torch.equal(e8m0, rescaled):
                        raise ValueError("MXFP4 scale round-trip failed")
                out[base + "weight_packed"] = packed
                out[base + "weight_scale"] = e8m0
                n_mx += 1
            else:
                q, scale = quantize_int8_channel(w)
                out[base + "weight"] = q
                out[base + "weight_scale"] = scale
                n_i8 += 1

    save_file(out, str(dst), metadata={"format": "pt"})
    nbytes = sum(v.numel() * v.element_size() for v in out.values())
    return ShardResult(dst.name, {k: dst.name for k in out}, nbytes, n_mx, n_i8, time.time() - t0)


QUANT_CONFIG = {
    "quant_method": "mixed-mxfp4-int8",
    "format": "mxfp4-moe-int8-rest",
    "quantization_status": "compressed",
    "config_groups": {
        "moe_experts": {
            "targets": ["*.mlp.experts.*", "*.shared_experts.*"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "format": "mxfp4-e2m1",
                "symmetric": True,
                "strategy": "block",
                "block_structure": [1, BLOCK],
                "block_axis": -1,
                "scale_dtype": "e8m0-uint8-bias127",
                "shared_exponent_rule": "ocp-mx-floor",
                "packing": "2-per-uint8-low-nibble-first",
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "token", "dynamic": True, "observer": None,
            },
            "output_activations": None,
        },
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "channel", "dynamic": False, "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "token", "dynamic": True, "observer": None,
            },
            "output_activations": None,
        },
    },
}


#: What sglang's compressed-tensors selector matches on for the plain W8A8
#: checkpoint: weights per channel and static, activations per token and dynamic.
INT8_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "channel", "dynamic": False, "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8, "type": "int", "symmetric": True,
                "strategy": "token", "dynamic": True, "observer": None,
            },
            "output_activations": None,
        }
    },
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="convert only N shards (smoke)")
    ap.add_argument("--verify", action="store_true", help="round-trip check every MoE tensor")
    ap.add_argument(
        "--moe-format",
        choices=["mxfp4", "int8"],
        default="mxfp4",
        help="mxfp4: the MXFP4 MoE checkpoint. int8: plain W8A8, every quantized "
             "tensor per-channel INT8 -- the same bytes the BF16 route produces, "
             "without materializing the 599 GiB BF16 intermediate.",
    )
    args = ap.parse_args()

    cfg_src = json.loads((args.src / "config.json").read_text())
    block = cfg_src.get("quantization_config", {}).get("weight_block_size", [128, 128])
    LOGGER.info(f"FP8 weight_block_size = {block}")

    shards = sorted(args.src.glob("*.safetensors"))
    if args.limit:
        shards = shards[: args.limit]
    args.dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total = mx = i8 = 0
    t0 = time.time()
    jobs = [(s, args.dst / s.name, block, args.verify, args.moe_format) for s in shards]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, wmap, nbytes, nmx, ni8, secs) in enumerate(ex.map(convert_shard, jobs), 1):
            weight_map.update(wmap)
            total += nbytes
            mx += nmx
            i8 += ni8
            LOGGER.info(f"  [{i}/{len(shards)}] {name} {nbytes / 1024**3:5.2f} GiB  "
                        f"mxfp4={nmx:<5} int8={ni8:<3} {secs:5.0f}s  "
                        f"(elapsed {time.time() - t0:.0f}s)")

    (args.dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    for extra in args.src.glob("*"):
        if extra.suffix != ".safetensors" and extra.name != "model.safetensors.index.json":
            shutil.copy2(extra, args.dst / extra.name)

    cfg = json.loads((args.dst / "config.json").read_text())
    quant = dict(QUANT_CONFIG) if args.moe_format == "mxfp4" else dict(INT8_QUANT_CONFIG)
    quant["ignore"] = cfg_src.get("quantization_config", {}).get("modules_to_not_convert", [])
    cfg["quantization_config"] = quant
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2))

    LOGGER.info(f"\n{len(shards)} shards, {total / 1024**3:.1f} GiB, "
                f"{mx} MXFP4 + {i8} INT8 tensors, {time.time() - t0:.0f}s -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
