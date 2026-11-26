# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Wan-only Taylor configuration, explicit scheduling and borrowed-factor math.

The public TaylorSeer algorithm is reused without its layer counter or offload
implementation. Tensor ownership and asynchronous transfers belong to the Wan
transfer manager. Importing or constructing this module does not allocate NPU
streams; real derivative evaluation still requires the public NPU backend.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import yaml

from module.dit_cache.cache_method import (
    DEFAULT_CACHE_CONFIG,
    TaylorSeer,
    load_cache_config,
)

from .cache_transfer import MAX_CACHE_BYTES, cache_byte_requirements

_METHODS = frozenset(("NoCache", "FBCache", "TeaCache", "TaylorSeer"))


def _integer(value, key, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        allowed = f"integer >= {minimum}" if maximum is None else f"integer in [{minimum}, {maximum}]"
        raise ValueError(f"{key}={value!r}: expected {allowed}, excluding bool")
    return value


def _boolean(value, key):
    if type(value) is not bool:
        raise ValueError(f"{key}={value!r}: expected a boolean")
    return value


@dataclass(frozen=True)
class CacheBudget:
    """Resolved byte caps and their minimum requirements, excluding the model."""

    num_groups: int
    feature_bytes: int
    bundle_bytes: int
    resident_bytes: int
    host_bytes: int
    pinned_bytes: int
    device_bytes: int
    required_host_bytes: int
    required_pinned_bytes: int
    required_device_bytes: int


@dataclass(frozen=True)
class TaylorConfig:
    n_derivatives: int = 2
    skip_interval_steps: int = 3
    warmup: int = 2
    cutoff_steps: int = 2
    offload: bool = False
    prefetch: bool = False
    prefetch_distance: int = 1
    cache_group_size: int = 4
    max_inflight: int = 2
    host_cache_budget_bytes: Optional[int] = None
    pinned_budget_bytes: Optional[int] = None
    device_cache_budget_bytes: Optional[int] = None

    def __post_init__(self):
        _integer(self.n_derivatives, "n_derivatives", 0, 3)
        _integer(self.skip_interval_steps, "skip_interval_steps", 1)
        _integer(self.warmup, "warmup", 1)
        _integer(self.cutoff_steps, "cutoff_steps", 0)
        _boolean(self.offload, "offload")
        _boolean(self.prefetch, "prefetch")
        _integer(self.prefetch_distance, "prefetch_distance", 1, 1)
        _integer(self.cache_group_size, "cache_group_size", 1)
        _integer(self.max_inflight, "max_inflight", 2, 2)
        if self.prefetch and not self.offload:
            raise ValueError("prefetch=True requires offload=True")
        for name in (
            "host_cache_budget_bytes", "pinned_budget_bytes", "device_cache_budget_bytes"
        ):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, 1, MAX_CACHE_BYTES)

    @classmethod
    def from_params(cls, params):
        if not isinstance(params, Mapping):
            raise ValueError(f"dit_cache.params={params!r}: expected a mapping")
        unknown = set(params) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown TaylorSeer parameter(s): {sorted(unknown, key=str)!r}")
        return cls(**dict(params))

    def resolve_budget(self, batch, seq_len, hidden_dim, num_layers):
        """Resolve finite caps before allocation; actual free memory is checked separately."""
        _integer(batch, "batch", 1, 1)
        _integer(seq_len, "seq_len", 1)
        _integer(hidden_dim, "hidden_dim", 1)
        _integer(num_layers, "num_layers", 1)
        if self.cache_group_size > num_layers:
            raise ValueError(
                f"cache_group_size={self.cache_group_size}: expected <= num_layers={num_layers}"
            )
        groups = (num_layers + self.cache_group_size - 1) // self.cache_group_size
        feature = 4 * batch * seq_len * hidden_dim
        bundle = (self.n_derivatives + 1) * feature
        resident = 2 * groups * bundle
        required_host, required_pinned, required_device = cache_byte_requirements(
            feature, bundle, resident, self.offload
        )

        def resolve(name, required):
            configured = getattr(self, name)
            if configured is not None and configured < required:
                raise ValueError(f"{name}={configured}: requires at least {required} bytes")
            return required if configured is None else configured

        return CacheBudget(
            num_groups=groups,
            feature_bytes=feature,
            bundle_bytes=bundle,
            resident_bytes=resident,
            host_bytes=resolve("host_cache_budget_bytes", required_host),
            pinned_bytes=resolve("pinned_budget_bytes", required_pinned),
            device_bytes=resolve("device_cache_budget_bytes", required_device),
            required_host_bytes=required_host,
            required_pinned_bytes=required_pinned,
            required_device_bytes=required_device,
        )

    def public_config(self):
        """Return an independent config; the shared class never owns transfer streams."""
        params = asdict(self)
        params.update(cache_name="TaylorSeer", offload=False)
        return {"TaylorSeer": params}


class _UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently replacing user choices."""


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError(f"Configuration mapping key {key!r} is not hashable") from exc
        if duplicate:
            raise ValueError(f"Duplicate configuration key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}")
        result[key] = value
    return result


def load_wan_cache_config(config_path):
    """Return (public loader config, optional TaylorConfig) after strict selection checks.

    Documented legacy method parameters are left unchanged for the public loader.
    Its legacy JSON envelope remains accepted; new Wan transfer fields belong to
    the selected TaylorSeer parameters only.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Cache configuration file not found: {path}")
    if path.suffix.lower() in (".yaml", ".yml") and path.suffix not in (".yaml", ".yml"):
        raise ValueError("YAML cache configuration requires a lowercase .yaml or .yml suffix")
    try:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            root = yaml.load(raw, Loader=_UniqueSafeLoader)
            if not isinstance(root, Mapping):
                raise ValueError("Configuration root must be a mapping")
            selected = root.get("dit_cache", {})
            if not isinstance(selected, Mapping):
                raise ValueError("dit_cache must be a mapping")
            unknown = set(selected) - {"method", "enable_separate_cfg", "params"}
            if unknown:
                raise ValueError(f"Unknown dit_cache key(s): {sorted(unknown, key=str)!r}")
            method = selected.get("method", "NoCache")
            separate = selected.get("enable_separate_cfg", True)
            params = selected.get("params", {})
        elif path.suffix.lower() == ".json":
            root = json.loads(raw, object_pairs_hook=_unique_json_pairs)
            if not isinstance(root, Mapping):
                raise ValueError("Configuration root must be a mapping")
            unknown = set(root) - {"cache_forward", "enable_separate_cfg", *_METHODS}
            if unknown:
                raise ValueError(f"Unknown cache configuration key(s): {sorted(unknown, key=str)!r}")
            method = root.get("cache_forward")
            separate = root.get("enable_separate_cfg")
            params = root.get(method, {}) if isinstance(method, str) else {}
            if not isinstance(params, Mapping):
                raise ValueError(f"{method} parameters must be a mapping")
            params = dict(params)
            cache_name = params.pop("cache_name", method)
            if cache_name != method:
                raise ValueError(f"cache_name={cache_name!r} disagrees with selected method {method!r}")
        else:
            raise ValueError(f"Unsupported cache configuration suffix: {path.suffix!r}")
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cache configuration {path}: {exc}") from exc

    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"dit_cache.method={method!r}: expected one of {sorted(_METHODS)}")
    _boolean(separate, "dit_cache.enable_separate_cfg")
    if not isinstance(params, Mapping):
        raise ValueError(f"dit_cache.params={params!r}: expected a mapping")
    if method == "TaylorSeer":
        if not separate:
            raise ValueError("TaylorSeer requires dit_cache.enable_separate_cfg=True")
        config = TaylorConfig.from_params(params)
    else:
        allowed = set(DEFAULT_CACHE_CONFIG[method]) - {"cache_name"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"Unsupported {method} parameter(s): {sorted(unknown, key=str)!r}")
        config = None
    # Keep the public loader's representation and all existing method parameters.
    normalized = load_cache_config(str(path))
    if config is not None:
        normalized["TaylorSeer"] = {"cache_name": "TaylorSeer", **asdict(config)}
    return normalized, config


@dataclass(frozen=True)
class StepPlan:
    global_step: int
    local_step: int
    total_steps: int
    full: bool
    activated_steps: Tuple[int, ...]
    reasons: Tuple[str, ...]


class TaylorSchedule:
    """One expert epoch schedule shared by both independent CFG histories."""

    def __init__(self, config, total_steps):
        if not isinstance(config, TaylorConfig):
            raise ValueError("config must be a TaylorConfig")
        _integer(total_steps, "total_steps", 1)
        self.config = config
        self.total_steps = total_steps
        self.activated_steps = ()
        self.full_steps = 0
        self.predict_steps = 0
        self._next_local = 0
        self._last_global = None
        self._pending = None

    def begin_step(self, global_step, local_step, *, force_full=False):
        _integer(global_step, "global_step", 0, self.total_steps - 1)
        _integer(local_step, "local_step", 0, global_step)
        _boolean(force_full, "force_full")
        if self._pending is not None:
            raise RuntimeError("Previous Taylor step has not been finished")
        if local_step != self._next_local:
            raise ValueError(f"local_step={local_step}: expected {self._next_local}")
        if self._last_global is not None and global_step != self._last_global + 1:
            raise ValueError("An expert epoch must use consecutive global steps")
        cfg = self.config
        reasons = []
        if local_step < cfg.warmup:
            reasons.append("warmup")
        if (local_step - cfg.warmup) % cfg.skip_interval_steps == 0:
            reasons.append("interval")
        if global_step >= self.total_steps - min(cfg.cutoff_steps, self.total_steps):
            reasons.append("cutoff")
        if local_step == 0:
            reasons.append("expert_start")
        if force_full or not self.activated_steps:
            reasons.append("missing_history")
        full = bool(reasons)
        activated = self.activated_steps + (local_step,) if full else self.activated_steps
        plan = StepPlan(
            global_step, local_step, self.total_steps, full, activated, tuple(reasons)
        )
        self._pending = plan
        return plan

    def finish_step(self, plan):
        if self._pending is None or plan is not self._pending:
            raise ValueError("finish_step requires this schedule's pending StepPlan")
        self.activated_steps = plan.activated_steps
        self.full_steps += int(plan.full)
        self.predict_steps += int(not plan.full)
        self._next_local += 1
        self._last_global = plan.global_step
        self._pending = None


class WanTaylorMath(TaylorSeer):
    """Reuse public Taylor math with explicit local steps and borrowed NPU factors."""

    def __init__(self, config, num_groups, total_steps):
        if not isinstance(config, TaylorConfig):
            raise ValueError("config must be a TaylorConfig")
        _integer(num_groups, "num_groups", 1)
        _integer(total_steps, "total_steps", 1)
        self.wan_config = config
        super().__init__(
            config.public_config(),
            {
                "double_stream_layers": 0,
                "single_stream_layers": num_groups,
                "num_steps": total_steps,
            },
        )

    def streams_init(self):
        self.copy_stream = None
        self.cal_stream = None

    def _get_step_type(self, step):
        # TaylorSeer.__init__ dispatches here before an actual request step exists.
        return "full"

    def update_layer_counter(self):
        raise RuntimeError("Wan Taylor steps must be bound by the denoising loop")

    def _bind(self, group_id, plan, expect_full):
        _integer(group_id, "group_id", 0, self.single_stream_layers - 1)
        if not isinstance(plan, StepPlan) or plan.total_steps != self.total_steps:
            raise ValueError("StepPlan does not match this request")
        if plan.full is not expect_full:
            raise ValueError("StepPlan full/predict mode does not match the operation")
        _integer(plan.global_step, "global_step", 0, self.total_steps - 1)
        _integer(plan.local_step, "local_step", 0, plan.global_step)
        activated = plan.activated_steps
        if not activated or any(type(step) is not int or step < 0 for step in activated):
            raise ValueError("StepPlan must contain nonnegative local full steps")
        if any(left >= right for left, right in zip(activated, activated[1:])):
            raise ValueError("Local activated_steps must be strictly increasing")
        if (expect_full and activated[-1] != plan.local_step) or (
            not expect_full and activated[-1] >= plan.local_step
        ):
            raise ValueError("Local activated_steps do not match the current operation")
        self.num_steps = plan.local_step
        self.activated_steps = list(activated)
        self.current.update(
            step=plan.local_step,
            activated_steps=self.activated_steps,
            stream="single_stream",
            layer=group_id,
            module="total",
            type="full" if expect_full else "taylor_cache",
        )
        return self.cache_dic["cache"][-1]["single_stream"][group_id]

    @staticmethod
    def _check_tensor(tensor, name):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if tensor.device.type != "npu" or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be an FP32 NPU tensor")
        if tensor.ndim != 3 or tensor.shape[0] != 1 or any(size <= 0 for size in tensor.shape):
            raise ValueError(f"{name} must have nonempty shape [1, S, C]")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device.index != torch.npu.current_device():
            raise ValueError(f"{name} must be on the current NPU device")

    def _check_factors(self, factors, reference=None, allow_empty=False):
        if not isinstance(factors, Mapping):
            raise ValueError("Taylor factors must be an integer-keyed mapping")
        if not factors:
            if allow_empty:
                return
            raise ValueError("Taylor prediction requires a zero-order history")
        if any(type(key) is not int for key in factors):
            raise ValueError("Taylor factor orders must be integers, excluding bool")
        if set(factors) != set(range(len(factors))) or len(factors) > self.n_derivatives + 1:
            raise ValueError("Taylor factor orders must be consecutive from zero and <= configured order")
        reference = factors[0] if reference is None else reference
        for order, tensor in factors.items():
            self._check_tensor(tensor, f"factor[{order}]")
            if tensor.shape != reference.shape or tensor.device != reference.device:
                raise ValueError("Taylor factors must match residual shape and device")

    def update(self, group_id, plan, old_factors, residual) -> Dict[int, torch.Tensor]:
        """Return owned new factors; old factors are borrowed only during this call."""
        slot = self._bind(group_id, plan, expect_full=True)
        self._check_tensor(residual, "residual")
        self._check_factors(old_factors, residual, allow_empty=True)
        feature = residual.detach()
        if not old_factors:
            return {0: feature}
        if len(plan.activated_steps) < 2:
            raise ValueError("Updating existing history requires two activated steps")
        slot["total"] = old_factors
        try:
            self.derivative_approximation(self.cache_dic, self.current, feature)
            return slot["total"]
        finally:
            slot["total"] = {}

    def predict(self, group_id, plan, factors) -> torch.Tensor:
        """Return an FP32 residual prediction, without mutating the history."""
        slot = self._bind(group_id, plan, expect_full=False)
        self._check_factors(factors)
        slot["total"] = factors
        try:
            return self.taylor_formula(self.cache_dic, self.current)
        finally:
            slot["total"] = {}
