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

import json
import math
import os
import re
import shutil
import sys
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    import torch_npu
    _HAS_NPU = bool(getattr(torch, "npu", None)) and torch.npu.is_available()
except Exception:
    _HAS_NPU = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_config import generate_quant_config


BLOCK_K = 32
FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

# e2m1 (mxfp4) value table; must match unpack_mxfloat4_to_fp32 in module/quantization/mxfp4.py.
# index = sign_bit(<<3) | magnitude_index ; magnitudes: {0,.5,1,1.5,2,3,4,6}
_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# Routed-expert projections become the MoEGMM (w4a8) group; everything else 2-D is Linear (w8a8).
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$")


def rename_hf_to_runtime(name):
    """
    Rename an original-HF BailingMoeV2_5 checkpoint key to the NPU runtime module name.

    An un-renamed HF tensor matches no param and is silently dropped at load. Idempotent:
    runtime-named keys contain none of the source patterns.
    """
    if name.startswith("model.word_embeddings."):
        name = name.replace("model.word_embeddings.", "model.embed_tokens.", 1)
    if ".attention." in name:
        name = name.replace(".attention.", ".self_attn.")
        name = name.replace(".self_attn.dense.", ".self_attn.o_proj.")
    if name.endswith(".mlp.gate.weight"):
        name = name[: -len(".mlp.gate.weight")] + ".mlp.router.classifier.weight"
    elif name.endswith(".mlp.gate.expert_bias"):
        name = name[: -len(".mlp.gate.expert_bias")] + ".mlp.router.e_score_correction_bias"
    return name


_FORCE_BF16_SUBSTRS = (
    "kv_b_proj",
    "q_a_layernorm",
    "kv_a_layernorm",
    "input_layernorm",
    "post_attention_layernorm",
    ".router.",
    "e_score_correction_bias",
    "expert_bias",
)


def round_to_decimal(x):
    abs_x = torch.abs(x)
    exponent = torch.floor(torch.log2(abs_x))
    mantissa = abs_x / (2 ** exponent)
    exponent = torch.where(mantissa > 1.75, exponent + 1, exponent)
    return exponent


def _shared_exponents(tensor, axes, shared_exp_round_method="round2decimal"):
    shared_exp = tensor
    for axis in axes:
        shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    shared_exp = shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
    if shared_exp_round_method == "floor":
        shared_exp = torch.floor(torch.log2(shared_exp))
    elif shared_exp_round_method == "round":
        shared_exp = torch.log2(shared_exp).round()
    elif shared_exp_round_method == "round2decimal":
        shared_exp = round_to_decimal(shared_exp)
    else:
        raise ValueError("Unrecognized round method!")
    return shared_exp


def _reshape_to_blocks(tensor, axes, block_size):
    axes = [(x + len(tensor.shape) if x < 0 else x) for x in axes]
    axes = sorted(axes)
    for i, axis in enumerate(axes):
        axes[i] = axis + i
        tensor = torch.unsqueeze(tensor, dim=axes[i] + 1)

    orig_shape = tensor.size()
    pad = [0, 0] * len(orig_shape)
    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True
    if do_padding:
        pad = list(reversed(pad))
        tensor = torch.nn.functional.pad(tensor, pad, mode="constant")

    def _reshape(shape):
        for axis in axes:
            if shape[axis] >= block_size:
                assert shape[axis] % block_size == 0
                shape[axis + 1] = block_size
                shape[axis] = shape[axis] // block_size
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    padded_shape = tensor.size()
    tensor = tensor.view(_reshape(list(padded_shape)))
    return tensor, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(tensor, padded_shape, orig_shape, axes):
    tensor = tensor.view(padded_shape)
    if not list(padded_shape) == list(orig_shape):
        tensor = tensor[tuple(slice(0, x) for x in orig_shape)]
    for axis in reversed(axes):
        tensor = torch.squeeze(tensor, dim=axis + 1)
    return tensor


def _safe_lshift(x, bits, exp):
    return x * (2 ** bits) if exp is None else x / (2 ** exp) * (2 ** bits)


def _safe_rshift(x, bits, exp):
    return x / (2 ** bits) if exp is None else x / (2 ** bits) * (2 ** exp)


def _round_mantissa_nearest(tensor):
    return torch.sign(tensor) * torch.floor(torch.abs(tensor) + 0.5)


def _quantize_elemwise_core(tensor, bits, exp_bits, max_norm):
    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(torch.abs(tensor) + (tensor == 0).type(tensor.dtype)))
        min_exp = -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None
    out = _safe_lshift(tensor, bits - 2, private_exp)
    out = _round_mantissa_nearest(out)
    out = _safe_rshift(out, bits - 2, private_exp)
    out = torch.clamp(out, min=-max_norm, max=max_norm)
    return out


def quantize_mx(tensor, quant_bit, scale_bits=8, block_size=BLOCK_K, axes=(-1,), name=None):
    """
    MX* quantize along `axes`. Returns (quantized_values_fp32, shared_exp_uint8).

    For quant_bit == 8 the values land on the e4m3 grid; for quant_bit == 4 they land on the
    e2m1 grid {0,.5,1,1.5,2,3,4,6} x sign. The scale is the e8m0 shared exponent stored as uint8
    (bias 127), one per `block_size` elements along the last quantized axis.
    """
    axes = list(axes)
    axes = [x + tensor.ndim if x < 0 else x for x in axes]

    if quant_bit == 8:
        ebits, mbits, emax, max_norm = 4, 5, 8, 448.0   # e4m3
    elif quant_bit == 4:
        ebits, mbits, emax, max_norm = 2, 3, 2, 6.0     # e2m1
    else:
        raise ValueError("quant_bit must be 8 or 4")

    tensor, axes, orig_shape, padded_shape = _reshape_to_blocks(tensor, axes, block_size)
    shared_exp_axes = [x + 1 for x in axes]

    shared_exp = _shared_exponents(tensor, axes=shared_exp_axes)
    shared_exp = shared_exp - emax

    scale_emax = 2 ** (scale_bits - 1) - 1
    finite = torch.isfinite(shared_exp)
    over = finite & (shared_exp > scale_emax)
    if over.any() or not finite.all():
        where = name or "tensor"
        if over.any():
            reason = (f"{int(over.sum())} block(s) exceed the e8m0 scale range: "
                      f"max exponent {shared_exp[over].max().item():.0f} > {scale_emax}")
        else:
            reason = f"{int((~finite).sum())} block(s) contain non-finite weights"
        raise ValueError(f"{where}: {reason}; refusing to quantize")
    shared_exp[shared_exp < -scale_emax] = -scale_emax

    tensor = tensor / (2 ** shared_exp)
    tensor = _quantize_elemwise_core(tensor, mbits, ebits, max_norm)
    tensor = _undo_reshape_to_blocks(tensor, padded_shape, orig_shape, axes)

    scale_uint8 = (shared_exp + scale_emax).to(torch.uint8).squeeze(-1)
    return tensor, scale_uint8


def to_e2m1_nibbles(values):
    """Map fp32 tensor whose entries lie on the e2m1 grid to uint8 nibble codes (0-15)."""
    mag = torch.abs(values)
    idx = torch.zeros_like(values, dtype=torch.uint8)
    for code, m in enumerate(_E2M1_MAGNITUDES):
        idx = torch.where(mag == m, torch.full_like(idx, code), idx)
    neg = torch.signbit(values)
    idx = torch.where(neg, idx | 0x08, idx)
    return idx


def pack_uint4(nibbles):
    """Pack 2 nibbles per byte along the last dim: byte = low | (high << 4)."""
    shape = nibbles.shape
    assert shape[-1] % 2 == 0, f"last dim {shape[-1]} must be even to pack 4-bit"
    flat = nibbles.contiguous().view(-1)
    packed = (flat[::2] | (flat[1::2] << 4)).to(torch.uint8)
    return packed.view(*shape[:-1], shape[-1] // 2)


def _to_compute_device(weight, device):
    """weight.float() on the NPU device (fast) when available, else CPU; the math is pure torch."""
    w = weight.float()
    return w.npu() if (device == "npu" and _HAS_NPU) else w


def _scale_to_uint8_2d(scale):
    """
    Normalize an MX shared-exponent scale to a contiguous 2-D uint8 tensor [out, n_blocks].

    npu_dynamic_mx_quant returns the e8m0 exponent as a 3-D, possibly float8_e8m0fnu tensor. The
    on-disk form must be the raw e8m0 bytes viewed as uint8; a value-cast would corrupt them.
    """
    if scale.dim() == 3:
        d0, d1, d2 = scale.shape
        scale = scale.reshape(d0, d1 * d2)
    if scale.dtype != torch.uint8:
        assert scale.element_size() == 1, f"unexpected MX scale dtype {scale.dtype}"
        scale = scale.view(torch.uint8)
    return scale.contiguous()


def quant_mxfp8(weight, device="npu", impl="mxref", name=None):
    """w8a8-mxfp8 -> (float8_e4m3fn weight[out,in], uint8 e8m0 scale[out, ceil(in/32)])."""
    out_f, in_f = weight.shape
    if impl == "npu-native":
        q, s = torch_npu.npu_dynamic_mx_quant(weight.npu(), dst_type=torch_npu.float8_e4m3fn)
        q = q.cpu().contiguous()
        s = _scale_to_uint8_2d(s.cpu())
    else:
        q, s = quantize_mx(_to_compute_device(weight, device), quant_bit=8, name=name)
        q = q.cpu().to(torch.float8_e4m3fn).contiguous()
        s = s.cpu().to(torch.uint8).contiguous()
    assert q.dtype == torch.float8_e4m3fn and q.shape == (out_f, in_f), (q.dtype, q.shape)
    assert s.dtype == torch.uint8 and s.shape == (out_f, math.ceil(in_f / BLOCK_K)), (s.dtype, s.shape)
    return q, s


def quant_mxfp4(weight, device="npu", impl="mxref", name=None):
    """w4a8-mxfp4 -> (uint8 packed weight[out, in/2], uint8 e8m0 scale[out, ceil(in/32)])."""
    out_f, in_f = weight.shape
    if impl == "npu-native":
        q, s = torch_npu.npu_dynamic_mx_quant(weight.npu(), dst_type=torch_npu.float4_e2m1fn_x2)
        q = q.cpu().contiguous()
        assert q.element_size() == 1, f"unexpected mxfp4 weight dtype {q.dtype}"
        q = q.view(torch.uint8)
        s = _scale_to_uint8_2d(s.cpu())
    else:
        qf, s = quantize_mx(_to_compute_device(weight, device), quant_bit=4, name=name)
        q = pack_uint4(to_e2m1_nibbles(qf.cpu())).contiguous()
        s = s.cpu().to(torch.uint8).contiguous()
    assert q.dtype == torch.uint8 and q.shape == (out_f, in_f // 2), (q.dtype, q.shape)
    assert s.dtype == torch.uint8 and s.shape == (out_f, math.ceil(in_f / BLOCK_K)), (s.dtype, s.shape)
    return q, s


def classify(base, ignore_set):
    """
    Return 'mxfp4', 'mxfp8' or 'keep' for a weight whose base name (no '.weight') is `base`.

    Mirrors the runtime decision in CompressedTensorsConfig.get_quant_method: exact-match in
    `ignore` -> bf16; FusedMoEGMM routed experts -> mxfp4; any other LinearBase -> mxfp8.
    """
    if base in ignore_set:
        return "keep"
    if any(s in base for s in _FORCE_BF16_SUBSTRS):
        return "keep"
    if base.endswith(".mlp.gate") or base.endswith(".router.gate"):
        return "keep"
    if _ROUTED_EXPERT_RE.search(base):
        return "mxfp4"
    return "mxfp8"


def parse_shard_spec(spec, n_files):
    """
    Parse a 1-based shard selector like '1,3,5' / '1-10' / '1-10,20,25-30' into a set of
    1-based indices into the sorted shard-file list. Returns None for 'all'/empty (all shards).
    """
    if spec is None or spec.strip().lower() in ("", "all"):
        return None
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"bad shard range '{part}' (lo>hi)")
            selected.update(range(lo, hi + 1))
        else:
            selected.add(int(part))
    out_of_range = {i for i in selected if i < 1 or i > n_files}
    if out_of_range:
        raise ValueError(f"shard indices {sorted(out_of_range)} out of range 1..{n_files}")
    return selected


def build_weight_map_from_dir(path):
    """
    Rebuild weight_map by reading each shard's header keys (cheap; no tensor data loaded).

    Reading from disk rather than accumulating during conversion keeps the index faithful when
    only some shards are converted: converted shards carry runtime names + scales, the rest
    still carry original-HF names.
    """
    wm = {}
    for st in sorted(glob(os.path.join(path, "*.safetensors"))):
        fn = os.path.basename(st)
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                wm[k] = fn
    return wm


def main(input_path, output_path, device="npu", impl="mxref", shards=None):
    inplace = os.path.realpath(input_path) == os.path.realpath(output_path)

    if impl == "npu-native" and not _HAS_NPU:
        raise RuntimeError("--quant-impl npu-native needs an NPU (torch_npu.npu_dynamic_mx_quant). "
                           "Use the default --quant-impl mxref (mx_quantize.py algorithm), which "
                           "runs on CPU when no NPU is present.")
    if impl == "mxref" and device == "npu" and not _HAS_NPU:
        print("NPU unavailable; running the mx_quantize.py algorithm on CPU (bit-exact, slower).")
        device = "cpu"

    os.makedirs(output_path, exist_ok=True)

    config_file = os.path.join(input_path, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    qcfg = config.get("quantization_config")
    if qcfg is None:
        qcfg = generate_quant_config(config)
        config["quantization_config"] = qcfg
    ignore_set = set(qcfg.get("ignore", []))

    index_file = os.path.join(input_path, "model.safetensors.index.json")
    have_index = os.path.exists(index_file)

    all_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    if not all_files:
        raise FileNotFoundError(f"No .safetensors found under {input_path}")
    sel = parse_shard_spec(shards, len(all_files))
    safetensor_files = all_files if sel is None else [f for i, f in enumerate(all_files, 1) if i in sel]
    if not inplace and sel is not None:
        print(f"NOTE: converting only shards {sorted(sel)} of {len(all_files)} to a separate dir; "
              "the resulting checkpoint is PARTIAL (for inspection/testing), not loadable as a "
              "whole model until all shards are converted.")
    print(f"Processing {len(safetensor_files)}/{len(all_files)} shard(s) "
          f"with quant-impl={impl!r} on device={device!r}.")

    counts = {"mxfp4": 0, "mxfp8": 0, "keep": 0}
    for st_file in tqdm(safetensor_files, desc="shards"):
        file_name = os.path.basename(st_file)
        state_dict = load_file(st_file, device="cpu")
        new_state_dict = {}

        renamed_keys = {rename_hf_to_runtime(k) for k in state_dict}

        for name, weight in state_dict.items():
            name = rename_hf_to_runtime(name)

            if name.endswith(".weight_scale") or name.endswith(".scale"):
                new_state_dict[name] = weight
                continue

            decision = "keep"
            # element_size() == 2 skips anything already quantized to fp8/uint8, so a re-run over
            # an already-converted checkpoint never double-quantizes
            if name.endswith(".weight") and weight.dim() == 2 and weight.element_size() == 2:
                base = name[: -len(".weight")]
                # a sibling scale means this shard was converted before
                if base + ".weight_scale" in renamed_keys:
                    decision = "keep"
                else:
                    decision = classify(base, ignore_set)

            if decision == "mxfp8":
                q, scale = quant_mxfp8(weight, device, impl, name=name)
                new_state_dict[name] = q
                new_state_dict[base + ".weight_scale"] = scale
            elif decision == "mxfp4":
                q, scale = quant_mxfp4(weight, device, impl, name=name)
                new_state_dict[name] = q
                new_state_dict[base + ".weight_scale"] = scale
            else:
                new_state_dict[name] = weight
            counts[decision] += 1

        out_file = os.path.join(output_path, file_name)
        if inplace:
            tmp_file = out_file + ".tmp"
            save_file(new_state_dict, tmp_file, metadata={"format": "pt"})
            os.replace(tmp_file, out_file)
        else:
            save_file(new_state_dict, out_file, metadata={"format": "pt"})

        del state_dict, new_state_dict

    if not inplace:
        for root, _, files in os.walk(input_path):
            for fn in files:
                if fn.endswith((".json", ".py", ".jinja", ".txt", ".model")) and \
                        fn != "model.safetensors.index.json":
                    rel = os.path.relpath(root, input_path)
                    dst_dir = os.path.join(output_path, rel)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(os.path.join(root, fn), os.path.join(dst_dir, fn))

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metadata = {}
    if have_index:
        with open(index_file, "r") as f:
            metadata = json.load(f).get("metadata", {})
    new_weight_map = build_weight_map_from_dir(output_path)
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": metadata, "weight_map": new_weight_map}, f, indent=2)

    print(f"Done. quantized this run: {counts['mxfp8']} mxfp8 (w8a8) + {counts['mxfp4']} mxfp4 "
          f"(w4a8); kept bf16: {counts['keep']}. Index now lists {len(new_weight_map)} tensors. "
          f"Output: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="In-place MXFP w8a8/w4a8 converter for lingv25.")
    parser.add_argument("--input_hf_path", type=str, required=True,
                        help="Source HuggingFace checkpoint dir (bf16/fp16).")
    parser.add_argument("--output_hf_path", type=str, default=None,
                        help="Destination dir. Defaults to --input_hf_path (true in-place).")
    parser.add_argument("--quant-impl", type=str, default="mxref", choices=["mxref", "npu-native"],
                        help="'mxref' (default): the mx_quantize.py microscaling algorithm "
                             "(round2decimal scale; matches the deepseek reference). 'npu-native': "
                             "torch_npu.npu_dynamic_mx_quant (fused hardware op; faster but its "
                             "scale bytes differ from mx_quantize.py).")
    parser.add_argument("--device", type=str, default="npu", choices=["npu", "cpu"],
                        help="Where the mxref math runs: 'npu' (default, fast) or 'cpu' "
                             "(byte-exact to mx_quantize.py). Ignored for --quant-impl npu-native.")
    parser.add_argument("--shards", type=str, default=None,
                        help="1-based shard selector into the sorted *.safetensors list, e.g. "
                             "'1', '1-3', '1-10,20,25-30'. Default: all shards. Useful to test a "
                             "few shards first, or to convert in-place in batches across runs.")
    args = parser.parse_args()
    out = args.output_hf_path or args.input_hf_path
    main(args.input_hf_path, out, device=args.device, impl=args.quant_impl, shards=args.shards)
