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

"""Validated, observed input-channel statistics for optional static smoothing."""

import hashlib
import json
import math
from pathlib import Path

import torch

KIND = "davinci-input-channel-amax-v1"
METHOD = "input-channel-equivalent-rescale-v1"
SCALE_MIN, SCALE_MAX = 1e-5, None


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_alpha(alpha):
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("smoothing alpha must be finite and in [0,1]")


def load_calibration(directory, source_hashes, targets):
    from safetensors.torch import load_file

    directory = Path(directory)
    path = directory / "calibration.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("kind") != KIND:
        raise ValueError("unsupported calibration schema/kind")
    if metadata.get("source_sha256") != source_hashes:
        raise ValueError("calibration source checkpoint hash mismatch")
    # Only the capture location affects the statistics used for smoothing.
    # Other provenance fields are retained for reproducibility, not interpreted.
    if metadata["provenance"]["capture"] != "forward_pre_hook_before_linear":
        raise ValueError("calibration must observe original pre-linear inputs")
    stats_path = (directory / metadata["statistics_file"]).resolve()
    if stats_path.parent != directory.resolve() or not stats_path.is_file():
        raise ValueError("invalid calibration statistics path")
    if digest(stats_path) != metadata["statistics_sha256"]:
        raise ValueError("calibration statistics checksum mismatch")
    stats = load_file(stats_path, device="cpu")
    if set(stats) != set(targets) or set(metadata["modules"]) != set(targets):
        raise ValueError("calibration must exactly cover quantized targets")
    for name in targets:
        item, value = metadata["modules"][name], stats[name]
        if any(
            type(item.get(key)) is not int or item[key] <= 0
            for key in ("experts", "in_features", "out_features")
        ):
            raise ValueError(f"invalid calibration module dimensions: {name}")
        observations = item.get("observations")
        if (
            not isinstance(observations, list)
            or len(observations) != item["experts"]
            or any(type(v) is not int or v <= 0 for v in observations)
        ):
            raise ValueError(f"every calibration expert needs observed tokens: {name}")
        if (
            value.dtype != torch.float32
            or list(value.shape) != [item["experts"], item["in_features"]]
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError(f"invalid FP32 channel amax: {name}")
    return metadata, stats, digest(path)


def smooth_weight(weight, activation_amax, alpha):
    """X/s and W*s are algebraically equivalent before quantization/rounding.

    Each expert computes W channel maxima over its own output rows. Activation
    and weight maxima are floored at 1e-5 before powers, and the resulting scale
    is floored at 1e-5. This is safe for observed zero channels; absent expert
    observations are rejected by load_calibration rather than fabricated.
    """
    validate_alpha(alpha)
    experts, features = activation_amax.shape
    if (
        weight.ndim != 2
        or weight.shape[1] != features
        or weight.shape[0] % experts
        or not torch.isfinite(weight).all()
        or not torch.isfinite(activation_amax).all()
        or (activation_amax < 0).any()
    ):
        raise ValueError("invalid smoothing weight/statistics contract")
    w = weight.float().reshape(experts, -1, features)
    maxima = w.abs().amax(dim=1)
    scale = (
        activation_amax.clamp_min(SCALE_MIN).pow(alpha) / maxima.clamp_min(SCALE_MIN).pow(1 - alpha)
    ).clamp_min(SCALE_MIN)
    scaled = (w * scale[:, None, :]).reshape_as(weight)
    if not torch.isfinite(scale).all() or not torch.isfinite(scaled).all():
        raise ValueError("smoothed weight overflow")
    return scaled, scale.contiguous()
