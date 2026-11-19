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

"""Strict configuration for the single-rank eager INT8 release."""

import json
import os
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TARGET = re.compile(
    r"block\.layers\.\d+\.(?:attention\.(?:linear_qkv|linear_proj)|mlp\.(?:up_gate_proj|down_proj))$"
)


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    mode: Literal["w8a8"] = "w8a8"
    activation_method: Literal["dynamic"] = "dynamic"
    weight_granularity: Literal["per_channel"] = "per_channel"
    activation_granularity: Literal["per_token"] = "per_token"
    symmetric: Literal[True] = True
    ignored_layers: list[str] = Field(default_factory=list)
    fallback_dtype: Literal["bfloat16"] = "bfloat16"
    invalid_metadata_policy: Literal["error"] = "error"

    @field_validator("symmetric", mode="before")
    @classmethod
    def validate_symmetric_type(cls, value):
        """Prevent Literal[True] from accepting numerically equal non-booleans."""
        if type(value) is not bool:
            raise ValueError("symmetric must be a boolean")
        return value

    @model_validator(mode="after")
    def check_contract(self):
        if len(self.ignored_layers) != len(set(self.ignored_layers)):
            raise ValueError("duplicate ignored modules")
        if any(not TARGET.fullmatch(name) for name in self.ignored_layers):
            raise ValueError("ignored_layers must contain DiT projection names")
        return self


def load_conversion_ignored_layers(path):
    """Read the conversion recipe; the exporter embeds it in the weight directory."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("quantization_config"), dict):
        raise ValueError("conversion config requires a quantization_config object")
    config = QuantizationConfig.model_validate(data["quantization_config"])
    return config.ignored_layers


def load_quantization_config(root):
    """Identify quantization from the checkpoint directory, or return None for BF16."""
    if root is None:
        return None
    root = Path(root)
    path = root / "config.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError("checkpoint config.json must contain an object")
    if "quantization_config" not in data:
        if (root / "quantization_manifest.json").exists():
            raise ValueError("quantized checkpoint requires config.json with quantization_config; reconvert weights")
        return None
    return QuantizationConfig.model_validate(data["quantization_config"])


def validate_execution(engine_config):
    if (int(os.getenv("WORLD_SIZE", "1")) != 1 or any(
            getattr(engine_config, key, 1) != 1 for key in ("tp_size", "pp_size", "cp_size", "dp_size"))):
        raise ValueError("INT8 v1 requires single rank tp=pp=cp=dp=1")
    if os.getenv("MAGI_COMPILE_COMPILE_MODE", "NONE") != "NONE":
        raise ValueError("INT8 v1 requires MAGI_COMPILE_COMPILE_MODE=NONE")
