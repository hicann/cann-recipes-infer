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
"""Wan request/expert/CFG lifecycle and grouped residual Taylor adaptation.

The shared cache manager remains the same object. Only its method is replaced;
Taylor never asks WanModel to break its block loop after the first block.
"""

import functools
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict

import torch

from module.dit_cache.cache_method import NoCache, cache_manager
from .cache_block import first_block_forward
from .cache_transfer import WanCacheTransfer
from .taylorseer import TaylorSchedule, WanTaylorMath, load_wan_cache_config


_REQUEST_LOCK = threading.Lock()
# Keep a failed manager and its storage alive until process exit. No subsequent
# pipeline may reuse a runtime whose DMA completion could not be established.
_FATAL_CACHE_TRANSFER = None
_LOG = logging.getLogger(__name__)


def _require_healthy_cache_runtime():
    failed_transfer = _FATAL_CACHE_TRANSFER
    if failed_transfer is not None:
        raise RuntimeError("cache drain failed; restart the inference process") from failed_transfer._drain_failure


def validate_wan_cache_options(args, world_size, *, model_config=None):
    """Validate unsupported combinations before allocating the model."""
    _, config = load_wan_cache_config(args.cache_config)
    if config is None:
        return
    for name in ("cfg_size", "ulysses_size", "ring_size", "tp_size"):
        if getattr(args, name) != 1:
            raise ValueError(f"TaylorSeer requires {name}=1")
    for name in ("t5_fsdp", "dit_fsdp", "vae_parallel"):
        if getattr(args, name):
            raise ValueError(f"TaylorSeer does not support {name}=true")
    if world_size != 1 or args.task != "i2v-A14B":
        raise ValueError("TaylorSeer requires single-process i2v-A14B")
    quant_mode = args.quant_mode if model_config is None else model_config.get("quant_mode", 0)
    quant_data_dir = None if model_config is None else model_config.get("quant_data_dir")
    # WanI2V also enables W8A8 from an artifact directory when quant_mode is zero.
    has_quant_artifact = quant_data_dir and os.path.isfile(os.path.join(quant_data_dir, "config.json"))
    if quant_mode != 0 or has_quant_artifact:
        raise ValueError("TaylorSeer does not support quantization")


def wan_cache_request(function):
    """Acquire the singleton's process lock before any generation device work."""
    @functools.wraps(function)
    def wrapped(pipeline, *args, **kwargs):
        if not _REQUEST_LOCK.acquire(blocking=False):
            raise RuntimeError("Wan generation is already active in this process")
        context = getattr(pipeline, "_wan_cache", None)
        error = None
        try:
            _require_healthy_cache_runtime()
            if context is not None:
                context.open_request()
            return function(pipeline, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if context is not None:
                    try:
                        context.close(error=error)
                    except BaseException as close_error:
                        if error is None:
                            error = close_error
                            raise
                        _LOG.error("Cache close failed after %s: %s", type(error).__name__, close_error)
            finally:
                try:
                    if error is not None and context is not None and _FATAL_CACHE_TRANSFER is None:
                        # Include close-time failures; keep the original exception.
                        try:
                            torch.npu.synchronize(pipeline.device)
                        except BaseException as cleanup_error:
                            _LOG.error("Device sync failed after %s: %s", type(error).__name__, cleanup_error)
                        models = list(context.models.values())
                        encoder = getattr(pipeline, "text_encoder", None)
                        if encoder is not None:
                            models.append(encoder.model)
                        for model in models:
                            try:
                                model.cpu()
                            except BaseException as cleanup_error:
                                _LOG.error("Model cleanup failed after %s: %s", type(error).__name__, cleanup_error)
                finally:
                    _REQUEST_LOCK.release()
    return wrapped


def reset_wan_shape(pipeline, signature):
    """Invalidate grid-dependent RoPE for all cache modes on a new shape."""
    signature = tuple(signature)
    if getattr(pipeline, "_wan_shape_signature", None) != signature:
        for model in (pipeline.high_noise_model, pipeline.low_noise_model):
            model.freqs_list = None
        pipeline._wan_shape_signature = signature


def configure_wan_cache(config_path, pipeline, *, num_steps, world_size=1):
    """Install wrappers without racing the process-wide generation state."""
    if not _REQUEST_LOCK.acquire(blocking=False):
        raise RuntimeError("Wan generation or cache configuration is already active in this process")
    try:
        _require_healthy_cache_runtime()
        return _configure_wan_cache(config_path, pipeline, num_steps=num_steps, world_size=world_size)
    finally:
        _REQUEST_LOCK.release()


def _configure_wan_cache(config_path, pipeline, *, num_steps, world_size):
    if getattr(pipeline, "_wan_cache", None) is not None:
        raise RuntimeError("This Wan pipeline already has a cache controller")
    normalized, config = load_wan_cache_config(config_path)
    if config is not None and world_size != 1:
        raise ValueError("TaylorSeer supports world_size=1 only")
    if config is not None and getattr(pipeline, "quant_mode", 0) != 0:
        raise ValueError("TaylorSeer does not support quantization")
    context = WanCacheContext(config_path, pipeline, config, num_steps, world_size)
    context.method_name = normalized["cache_forward"]
    previous_manager = vars(cache_manager).copy()
    replaced = []
    try:
        if config is None:
            # Existing methods may retain the old dict by reference. Mutate a
            # fresh dict so rollback preserves both its identity and contents.
            if isinstance(previous_manager.get("cache_params"), dict):
                cache_manager.cache_params = previous_manager["cache_params"].copy()
            cache_manager.from_config(config_path, {"num_steps": num_steps})
            if world_size > 1:
                cache_manager.cache_method = NoCache()
            else:
                for model in context.models.values():
                    block = model.blocks[0]
                    replaced.append((block, block.forward))
                    block.forward = first_block_forward.__get__(block, type(block))
        else:
            cache_manager.config = normalized
            cache_manager.enable_separate_cfg = True
            cache_manager.cache_method = context
            for expert_id, model in context.models.items():
                for block_id, block in enumerate(model.blocks):
                    original = block.forward
                    replaced.append((block, original))
                    block.forward = context.wrap_block(expert_id, block_id, original)
        pipeline._wan_cache = context
    except BaseException:
        for block, original in reversed(replaced):
            block.forward = original
        vars(cache_manager).clear()
        vars(cache_manager).update(previous_manager)
        raise
    return context


class WanCacheContext:
    cache_name = "TaylorSeer"
    should_skip = False

    def __init__(self, config_path, pipeline, config, num_steps, world_size):
        self.config_path = config_path
        self.pipeline = pipeline
        self.config = config
        self.method_name = "TaylorSeer" if config is not None else "legacy"
        self.num_steps = num_steps
        self.world_size = world_size
        self.models = {"high": pipeline.high_noise_model, "low": pipeline.low_noise_model}
        self.num_layers = {name: len(model.blocks) for name, model in self.models.items()}
        self.groups = {}
        if config is not None:
            for name, layers in self.num_layers.items():
                if not 1 <= config.cache_group_size <= layers:
                    raise ValueError(f"cache_group_size exceeds {name} block count {layers}")
                self.groups[name] = (layers + config.cache_group_size - 1) // config.cache_group_size
            dims = {model.dim for model in self.models.values()}
            if len(dims) != 1:
                raise ValueError("TaylorSeer requires equal high/low expert hidden dimensions")
            if pipeline.param_dtype != torch.bfloat16:
                raise ValueError("TaylorSeer currently supports BF16 Wan weights only")
        self.active = False
        self.transfer = None
        self.last_statistics = {}
        self.plan = None
        self.group_input = None
        self.lease = None
        self.credit = None

    def open_request(self):
        _require_healthy_cache_runtime()
        if self.active:
            raise RuntimeError("Wan cache request cannot be reentered")
        self.active = True
        self.request_id = uuid.uuid4().hex
        self.feature_shape = None
        self.expert = None
        self.epoch = 0
        self.local_step = 0
        self.next_step = 0
        self.branch = None
        self.finished_branches = set()
        self.next_block = 0
        self.plan = None
        self.math = {}
        self.schedule = None
        self.transfer = None
        self.group_input = None
        self.lease = None
        self.credit = None
        self.finite = None
        self.records = []
        self.last_statistics = {}
        if self.config is not None:
            cache_manager.cache_method = self
        for model in self.models.values():
            model.blocks[0].cfg_step = 0

    def prepare_shape(self, feature_shape):
        if not self.active or self.feature_shape is not None:
            raise RuntimeError("prepare_shape requires a fresh cache request")
        self.feature_shape = tuple(feature_shape)
        if self.config is not None:
            self.budget = self.config.resolve_budget(
                batch=self.feature_shape[0], seq_len=self.feature_shape[1],
                hidden_dim=self.feature_shape[2], num_layers=max(self.num_layers.values()))
            _LOG.info("WAN_CACHE_BUDGET %s", json.dumps(asdict(self.budget), sort_keys=True))

    def bind_timesteps(self, timesteps):
        if not self.active or self.feature_shape is None:
            raise RuntimeError("Cache shape must be validated before scheduler binding")
        self.total_steps = len(timesteps)
        if self.total_steps < 1:
            raise ValueError("Scheduler must have at least one timestep")
        if self.config is None:
            cache_manager.from_config(self.config_path, {"num_steps": self.total_steps})
            if self.world_size > 1:
                cache_manager.cache_method = NoCache()
            return
        if self.transfer is not None:
            raise RuntimeError("Scheduler is already bound to this request")
        self.transfer = WanCacheTransfer(
            request_id=self.request_id, device=self.pipeline.device,
            feature_shape=self.feature_shape, num_groups=max(self.groups.values()),
            max_order=self.config.n_derivatives, offload=self.config.offload,
            prefetch=self.config.prefetch, max_inflight=self.config.max_inflight,
            host_cache_budget_bytes=self.config.host_cache_budget_bytes,
            pinned_budget_bytes=self.config.pinned_budget_bytes,
            device_cache_budget_bytes=self.config.device_cache_budget_bytes)
        self.finite = torch.ones((), dtype=torch.bool, device=self.pipeline.device)

    def _key(self, group_id, branch=None):
        return (self.request_id, self.expert, self.epoch, branch or self.branch, group_id)

    def begin_step(self, global_step, expert_id):
        if self.config is None:
            return
        if self.transfer is None or self.plan is not None or global_step != self.next_step:
            raise RuntimeError("Cache step is out of order or scheduler is not bound")
        if expert_id not in self.models:
            raise ValueError(f"Unknown Wan expert {expert_id}")
        if expert_id != self.expert:
            if self.expert is not None:
                self.transfer.retire((self.expert, self.epoch))
            self.expert = expert_id
            self.epoch += 1
            self.local_step = 0
            self.schedule = TaylorSchedule(self.config, self.total_steps)
            self.math = {branch: WanTaylorMath(self.config, self.groups[expert_id], self.total_steps)
                         for branch in ("cond", "uncond")}
        missing = any(not self.transfer.has_history(self._key(group, branch))
                      for branch in ("cond", "uncond") for group in range(self.groups[expert_id]))
        self.plan = self.schedule.begin_step(global_step, self.local_step, force_full=missing)
        self.finished_branches = set()
        self.branch = None

    def begin_branch(self, is_cond):
        if self.config is None:
            return
        if type(is_cond) is not bool:
            raise ValueError("is_cond must be bool")
        branch = "cond" if is_cond else "uncond"
        expected = "cond" if not self.finished_branches else "uncond"
        if self.plan is None or self.branch is not None or branch != expected or branch in self.finished_branches:
            raise RuntimeError("CFG branches must run cond then uncond once per step")
        self.branch = branch
        self.next_block = 0
        self.branch_hits = 0
        self.branch_orders = []

    def wrap_block(self, expert_id, block_id, original):
        @functools.wraps(original)
        def forward(x, *args, **kwargs):
            return self.block_forward(expert_id, block_id, original, x, args, kwargs)
        return forward

    def check_tensor(self, tensor):
        if self.config is not None:
            self._check_finite(tensor)

    def _check_finite(self, tensor):
        # Reductions stay on the compute stream; inspect the one scalar at close.
        self.finite.logical_and_(torch.isfinite(tensor).all())

    def block_forward(self, expert_id, block_id, original, x, args, kwargs):
        if not self.active or self.plan is None or self.branch is None:
            raise RuntimeError("Taylor blocks require an active step and CFG branch")
        if expert_id != self.expert or block_id != self.next_block:
            raise RuntimeError("Expert/block traversal does not match the active cache plan")
        if tuple(x.shape) != self.feature_shape or x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Wan block feature shape or dtype changed during a request")
        group_id, offset = divmod(block_id, self.config.cache_group_size)
        group_end = min((group_id + 1) * self.config.cache_group_size, self.num_layers[expert_id]) - 1
        key = self._key(group_id)
        if offset == 0:
            self.lease = self.transfer.acquire(key) if self.transfer.has_history(key) else None
            self.branch_hits += int(self.lease is not None)
            if self.plan.full:
                # One producer credit precedes even the group-input snapshot.
                self.credit = self.transfer.reserve_update()
                self.group_input = x.detach().to(dtype=torch.float32, copy=True).contiguous()
            else:
                if self.lease is None:
                    raise RuntimeError("Predict group has no full-step history")
                self.branch_orders.append(max(self.lease.factors))
                residual = self.math[self.branch].predict(group_id, self.plan, self.lease.factors)
                self._check_finite(residual)
                x = (x.float() + residual).to(dtype=x.dtype)
                del residual
        if self.plan.full:
            x = original(x, *args, **kwargs)
        if offset == 0 and self.lease is not None and group_id + 1 < self.groups[expert_id]:
            # Enqueue this group before staging the next group on the Host.
            self.transfer.prefetch(self._key(group_id + 1))
        if block_id == group_end:
            if self.plan.full:
                residual = x.detach().float() - self.group_input
                self.group_input = None
                self._check_finite(residual)
                old = self.lease.factors if self.lease is not None else {}
                factors = self.math[self.branch].update(group_id, self.plan, old, residual)
                self.branch_orders.append(max(factors))
                for factor in factors.values():
                    self._check_finite(factor)
                if self.lease is not None:
                    self.transfer.release(self.lease)
                    self.lease = None
                del old
                self.transfer.publish(key, factors, self.credit)
                self.credit = None
                del factors, residual
            elif self.lease is not None:
                self.transfer.release(self.lease)
                self.lease = None
            self._check_finite(x)
        self.next_block += 1
        if self.next_block == self.num_layers[expert_id]:
            self.transfer.drain()
            record = {"request_id": self.request_id, "step": self.plan.global_step,
                      "local_step": self.plan.local_step, "expert": self.expert, "epoch": self.epoch,
                      "cfg": self.branch, "type": "full" if self.plan.full else "predict",
                      "groups": self.groups[expert_id], "cache_hits": self.branch_hits,
                      "derivative_updates": self.groups[expert_id] if self.plan.full else 0,
                      "factor_orders": self.branch_orders, "reasons": list(self.plan.reasons),
                      "activated_steps": list(self.plan.activated_steps)}
            self.records.append(record)
            _LOG.info("WAN_CACHE_STEP %s", json.dumps(record, sort_keys=True))
            self.finished_branches.add(self.branch)
            self.branch = None
        return x

    def finish_step(self):
        if self.config is None:
            self.next_step += 1
            return
        if self.plan is None or self.finished_branches != {"cond", "uncond"}:
            raise RuntimeError("Cannot finish an incomplete Taylor step")
        self.schedule.finish_step(self.plan)
        self.plan = None
        self.local_step += 1
        self.next_step += 1

    def post_cache_update(self, latent):
        # Group exits own updates; WanModel must never reuse a whole-DiT residual.
        return None

    def close(self, error=None):
        global _FATAL_CACHE_TRANSFER
        if not self.active:
            return self.last_statistics
        cleanup_error = None
        stats = {}
        try:
            if self.transfer is not None:
                try:
                    if self.lease is not None:
                        self.transfer.release(self.lease)
                        self.lease = None
                    if self.credit is not None:
                        self.transfer.cancel_update(self.credit)
                        self.credit = None
                finally:
                    stats = self.transfer.close(error=error)
                if error is None and (self.plan is not None or self.next_step != self.total_steps):
                    raise RuntimeError("Taylor request ended before all scheduler steps completed")
                if error is None and not bool(self.finite.item()):
                    raise FloatingPointError("Taylor request produced NaN/Inf")
        except BaseException as exc:
            cleanup_error = exc
        finally:
            self.last_statistics = {
                "request_id": self.request_id,
                "method": self.method_name,
                "completed_steps": self.next_step,
                "status": "aborted" if error is not None or cleanup_error is not None else "completed",
                "full_steps": self.next_step if self.method_name == "NoCache" else sum(
                    r["type"] == "full" and r["cfg"] == "cond" and r["step"] < self.next_step
                    for r in self.records
                ),
                "predict_steps": sum(
                    r["type"] == "predict" and r["cfg"] == "cond" and r["step"] < self.next_step
                    for r in self.records
                ),
                "transfer": stats,
            }
            if self.transfer is not None and self.transfer._drain_failure is not None:
                _FATAL_CACHE_TRANSFER = self.transfer
            else:
                self.transfer = None
            self.math = {}
            self.schedule = None
            self.plan = None
            self.group_input = None
            self.lease = None
            self.credit = None
            self.finite = None
            self.active = False
            if self.config is not None:
                self.print_statistics()
        if cleanup_error is not None:
            if error is None:
                raise cleanup_error
            _LOG.error("Cache cleanup also failed while propagating %s: %s", type(error).__name__, cleanup_error)
        return self.last_statistics

    def print_statistics(self):
        _LOG.info("WAN_CACHE_REQUEST %s", json.dumps(self.last_statistics, sort_keys=True))
