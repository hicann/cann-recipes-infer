# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Deterministic streaming export and strict mixed-checkpoint loading."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from .int8_linear import Int8Linear, quantize_rows
from .config import TARGET, QuantizationConfig, load_conversion_ignored_layers, load_quantization_config
from .calibration import (
    METHOD,
    SCALE_MAX,
    SCALE_MIN,
    digest,
    load_calibration,
    smooth_weight,
    validate_alpha,
)

SCHEME = "davinci-w8a8-dynamic-token-channel-v1"
QUANTIZATION = {
    "weight": "symmetric_per_output_channel",
    "activation": "dynamic_per_token",
    "range": [-127, 127],
    "rounding": "nearest_even",
    "scale_dtype": "float32",
    "zero_weight_scale": 1,
    "runtime_layout": "ND",
}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def safe_file(root, name):
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ValueError(f"invalid checkpoint filename: {name}")
    return path


def source_index(root):
    from safetensors import safe_open

    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as src:
            mapping = dict.fromkeys(src.keys(), "model.safetensors")
    if not mapping or any(not name.endswith(".safetensors") for name in mapping.values()):
        raise ValueError("only uncompressed safetensors supported; explicitly decompress .zst first")
    return mapping


def convert(source, destination, ignore=(), calibration=None, alpha=None):
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("destination must not exist (incomplete exports are never overwritten)")
    if load_quantization_config(source) is not None:
        raise ValueError("source already contains quantization_config; refusing double quantization")
    source_config = source / "config.json"
    config = json.loads(source_config.read_text(encoding="utf-8")) if source_config.exists() else {}
    config["quantization_config"] = QuantizationConfig(ignored_layers=sorted(ignore)).model_dump()
    mapping = source_index(source)
    candidates = sorted(k[:-7] for k in mapping if k.endswith(".weight") and TARGET.fullmatch(k[:-7]))
    if len(set(ignore)) != len(ignore) or set(ignore) - set(candidates):
        raise ValueError("duplicate or unknown ignored layer")
    targets = sorted(set(candidates) - set(ignore))
    if not targets:
        raise ValueError("no quantized targets")
    if any(
        name + suffix in mapping
        for name in targets
        for suffix in (".weight_scale", ".smooth_scale")
    ):
        raise ValueError("source already contains target weight scales; refusing double quantization")
    source_hashes = {name: digest(safe_file(source, name)) for name in sorted(set(mapping.values()))}
    calibration_meta, statistics, calibration_digest = None, None, None
    if calibration is not None:
        alpha = 0.5 if alpha is None else alpha
        validate_alpha(alpha)
        calibration_meta, statistics, calibration_digest = load_calibration(
            calibration, source_hashes, targets
        )
    elif alpha is not None:
        raise ValueError("alpha requires observed calibration artifacts")
    destination.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "converter_version": "2.0.0",
        "scheme": SCHEME,
        "quantization": QUANTIZATION,
        "targets": targets,
        "modules": {},
        "source_sha256": source_hashes,
        "files": {},
        "tensors": {},
    }
    if calibration_meta is not None:
        calibration_copy = destination / "calibration"
        calibration_copy.mkdir()
        shutil.copyfile(Path(calibration) / "calibration.json", calibration_copy / "calibration.json")
        shutil.copyfile(
            Path(calibration) / calibration_meta["statistics_file"],
            calibration_copy / calibration_meta["statistics_file"],
        )
        manifest["smoothing"] = {
            "method": METHOD,
            "alpha": alpha,
            "scale_min": SCALE_MIN,
            "scale_max": SCALE_MAX,
            "activation_amax_min": SCALE_MIN,
            "weight_amax_min": SCALE_MIN,
            "zero_channel_policy": "floor_before_power",
            "calibration_sha256": calibration_digest,
            "calibration": calibration_meta,
            "calibration_directory": "calibration",
        }
    output_map = {}
    seen = set()
    for i, filename in enumerate(sorted(set(mapping.values()))):
        path = safe_file(source, filename)
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as src:
            if set(src.keys()) != {k for k, v in mapping.items() if v == filename}:
                raise ValueError(f"source shard/index coverage mismatch: {filename}")
            for key in sorted(src.keys()):
                value = src.get_tensor(key)
                if key.endswith(".weight") and key[:-7] in targets:
                    if value.dtype != torch.bfloat16:
                        raise ValueError(f"target must be BF16: {key}")
                    zero_rows = (value == 0).all(-1).nonzero().flatten().tolist()
                    manifest["modules"][key[:-7]] = {
                        "logical_layout": "[E*N,K]; expert-major; original channel order",
                        "expert_output_rows": value.shape[0],
                        "K": value.shape[1],
                        "expert_resolution": "validated against original model module at load",
                        "zero_weight_rows": zero_rows,
                    }
                    if calibration_meta is not None:
                        name = key[:-7]
                        info = calibration_meta["modules"][name]
                        if list(value.shape) != [info["experts"] * info["out_features"], info["in_features"]]:
                            raise ValueError(f"calibration/source module shape mismatch: {name}")
                        value, smooth_scale = smooth_weight(value, statistics[name], alpha)
                        tensors[name + ".smooth_scale"] = smooth_scale
                        manifest["modules"][name]["experts"] = info["experts"]
                        manifest["modules"][name]["out_features"] = info["out_features"]
                    value, scale = quantize_rows(value)
                    tensors[key[:-7] + ".weight_scale"] = scale.contiguous()
                tensors[key] = value.contiguous()
        out_name = f"model-{i + 1:05d}.safetensors"
        for key, value in tensors.items():
            if key in seen:
                raise ValueError(f"duplicate/generated scale collision: {key}")
            seen.add(key)
            output_map[key] = out_name
            manifest["tensors"][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        save_file(tensors, destination / out_name)
        manifest["files"][out_name] = digest(destination / out_name)
    write_json(destination / "model.safetensors.index.json", {"weight_map": output_map})
    write_json(destination / "config.json", config)
    manifest["config_sha256"] = digest(destination / "config.json")
    # Manifest is written last: its presence marks a completed export.
    write_json(destination / "quantization_manifest.json", manifest)
    return manifest


def load_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "quantization_manifest.json").read_text())
    if manifest.get("scheme") != SCHEME or manifest.get("schema_version") != 2:
        raise ValueError("unsupported quantization scheme/schema; reconvert weights")
    config = load_quantization_config(root)
    if config is None:
        raise ValueError("missing checkpoint quantization_config")
    if digest(root / "config.json") != manifest.get("config_sha256"):
        raise ValueError("checkpoint config.json checksum mismatch")
    targets, ignore = manifest["targets"], config.ignored_layers
    if (
        not targets
        or len(targets) != len(set(targets))
        or len(ignore) != len(set(ignore))
        or set(targets) & set(ignore)
        or any(not TARGET.fullmatch(x) for x in targets + ignore)
    ):
        raise ValueError("invalid targets/ignore")
    if (
        manifest.get("quantization") != QUANTIZATION
        or set(manifest["modules"]) != set(targets)
    ):
        raise ValueError("quantization metadata/target mapping mismatch")
    if "smoothing" in manifest:
        smoothing = manifest["smoothing"]
        validate_alpha(smoothing["alpha"])
        if (
            smoothing.get("method") != METHOD
            or smoothing.get("scale_min") != SCALE_MIN
            or smoothing.get("scale_max") != SCALE_MAX
            or smoothing.get("zero_channel_policy") != "floor_before_power"
            or smoothing.get("activation_amax_min") != SCALE_MIN
            or smoothing.get("weight_amax_min") != SCALE_MIN
        ):
            raise ValueError("unsupported smoothing metadata")
        calibration = smoothing["calibration"]
        if smoothing.get("calibration_directory") != "calibration":
            raise ValueError("invalid embedded calibration directory")
        observed, _, observed_digest = load_calibration(
            root / "calibration", manifest["source_sha256"], targets
        )
        if observed != calibration or observed_digest != smoothing["calibration_sha256"]:
            raise ValueError("embedded calibration metadata checksum mismatch")
    for name, expected_digest in manifest["files"].items():
        if digest(safe_file(root, name)) != expected_digest:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    return manifest


def load_quantized(model, root):
    from safetensors import safe_open

    root = Path(root)
    manifest = load_manifest(root)
    config = load_quantization_config(root)
    modules = dict(model.named_modules())
    candidates = {name for name in modules if TARGET.fullmatch(name)}
    if candidates != set(manifest["targets"]) | set(config.ignored_layers):
        raise ValueError("manifest/model candidate coverage mismatch")
    # Exact original module dimensions define expert splits, never artifact guesses.
    for name in manifest["targets"]:
        old = modules[name]
        replacement = Int8Linear(
            old.in_features,
            old.out_features,
            old.num_experts,
            old.bias is not None,
            device=old.weight.device,
            smoothing="smoothing" in manifest,
        )
        if old.bias is not None:
            replacement.bias = torch.empty_like(old.bias)
        layout = manifest["modules"][name]
        if (
            layout["K"] != old.in_features
            or layout["expert_output_rows"] != old.num_experts * old.out_features
        ):
            raise ValueError(f"module layout mismatch: {name}")
        if "smoothing" in manifest and (
            layout["experts"] != old.num_experts or layout["out_features"] != old.out_features
        ):
            raise ValueError(f"smoothing expert layout mismatch: {name}")
        if "smoothing" in manifest:
            observed = manifest["smoothing"]["calibration"]["modules"][name]
            if (observed["experts"], observed["out_features"], observed["in_features"]) != (
                old.num_experts,
                old.out_features,
                old.in_features,
            ):
                raise ValueError(f"calibration/model expert layout mismatch: {name}")
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, replacement)
    expected = model.state_dict()
    if set(expected) != set(manifest["tensors"]):
        raise ValueError("full model tensor coverage mismatch")
    mapping = source_index(root)
    if set(mapping) != set(expected) or set(mapping.values()) != set(manifest["files"]):
        raise ValueError("export index/manifest coverage mismatch")
    seen = set()
    for filename in sorted(manifest["files"]):
        with safe_open(safe_file(root, filename), framework="pt", device="cpu") as src:
            if set(src.keys()) != {k for k, v in mapping.items() if v == filename}:
                raise ValueError("export shard/index mismatch")
            tensors = {}
            for key in src.keys():
                value = src.get_tensor(key)
                meta = manifest["tensors"][key]
                target = expected[key]
                if (
                    list(value.shape) != meta["shape"]
                    or str(value.dtype) != meta["dtype"]
                    or value.shape != target.shape
                    or value.dtype != target.dtype
                ):
                    raise ValueError(f"tensor shape/dtype mismatch: {key}")
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite tensor: {key}")
                if key.endswith(".weight_scale") and not (value > 0).all():
                    raise ValueError(f"nonpositive weight scale: {key}")
                if key.endswith(".smooth_scale") and not (value >= SCALE_MIN).all():
                    raise ValueError(f"invalid smoothing scale: {key}")
                if key.endswith(".weight") and key[:-7] in manifest["targets"] and (value < -127).any():
                    raise ValueError(f"static weight outside symmetric [-127,127]: {key}")
                tensors[key] = value
                seen.add(key)
            # Full coverage validated separately; assign preserves INT8 and releases meta allocation.
            model.load_state_dict(tensors, strict=False, assign=True)
    if seen != set(expected):
        raise ValueError("missing checkpoint tensors")
    for key, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            raise ValueError(f"unmaterialized model tensor after load: {key}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", help="Conversion recipe containing quantization_config.ignored_layers")
    selection.add_argument("--ignore", action="append", help="Projection to retain in BF16; repeat as needed")
    parser.add_argument(
        "--calibration", help="Directory containing observed calibration.json and amax safetensors"
    )
    parser.add_argument(
        "--alpha", type=float, help="Smoothing exponent [0,1]; requires --calibration; default 0.5"
    )
    args = parser.parse_args()
    ignored_layers = load_conversion_ignored_layers(args.config) if args.config else (args.ignore or [])
    result = convert(args.source, args.output, ignored_layers, args.calibration, args.alpha)
    print(f"Exported {len(result['targets'])} INT8 modules; ignored {len(ignored_layers)}")


if __name__ == "__main__":
    main()
