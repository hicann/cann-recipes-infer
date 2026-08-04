# coding=utf-8
# Adapted from
# https://github.com/GAIR-NLP/daVinci-MagiHuman,
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Copyright (c) 2026 SandAI. All Rights Reserved.
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

from typing import Any, Callable, Dict, Tuple

import torch


class CPUOffloadWrapper:
    def __init__(self, model: Any, is_cpu_offload: bool = False, is_running_on_gpu: bool = True):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "is_cpu_offload", is_cpu_offload)
        object.__setattr__(self, "is_running_on_gpu", is_running_on_gpu)

        cpu_device = torch.device("cpu")
        cuda_device = torch.device("cuda")

        object.__setattr__(self, "cpu_device", cpu_device)
        object.__setattr__(self, "cuda_device", cuda_device)

        object.__setattr__(self, "_resident_backup", None)

        # Initialize placement location
        if is_cpu_offload:
            self.model.to(cpu_device)
        else:
            self.model.to(cuda_device)

        # Whitelist non-compute methods that shouldn't trigger device hops (pass-through only; no device switch)
        object.__setattr__(
            self,
            "_non_compute_methods",
            {
                "to",
                "cpu",
                "cuda",
                "eval",
                "train",
                "state_dict",
                "load_state_dict",
                "parameters",
                "named_parameters",
                "buffers",
                "named_buffers",
                "modules",
                "named_modules",
                "children",
                "named_children",
                "register_forward_hook",
                "register_forward_pre_hook",
                "register_full_backward_hook",
                "zero_grad",
                "share_memory",
                "half",
                "float",
                "bfloat16",
            },
        )

        # nn.Module methods that return self — the wrapper must also return self
        # so callers like `self.model = self.model.to(...)` don't lose the wrapper.
        object.__setattr__(
            self,
            "_self_returning_methods",
            {
                "to",
                "cpu",
                "cuda",
                "eval",
                "train",
                "half",
                "float",
                "bfloat16",
                "share_memory",
            },
        )

    # Get current primary device (for external reads)
    @property
    def device(self) -> torch.device:
        if isinstance(self.model, torch.nn.Module):
            return next(self.model.parameters()).device
        else:
            for k, v in self.model.__dict__.items():
                if isinstance(v, torch.Tensor):
                    return v.device
                elif isinstance(v, torch.nn.Module):
                    return next(v.parameters()).device
            return self.cuda_device

    def _backup_cpu_state(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        # Backup module parameters and buffers
        module_param_backup = {}
        module_buffer_backup = {}
        other_backup = {}

        def save_module_state(mod: torch.nn.Module, prefix: str):
            for name, param in mod.named_parameters():
                if param is not None:
                    full_key = prefix + name
                    module_param_backup[full_key] = param.data
            for name, buffer in mod.named_buffers():
                if buffer is not None:
                    full_key = prefix + name
                    module_buffer_backup[full_key] = buffer.data

        if isinstance(self.model, torch.nn.Module):
            save_module_state(self.model, "")
        else:
            for name, attr_val in self.model.__dict__.items():
                if isinstance(attr_val, torch.nn.Module):
                    save_module_state(attr_val, name + ".")
                elif isinstance(attr_val, torch.Tensor):
                    other_backup[name] = attr_val

        return module_param_backup, module_buffer_backup, other_backup

    def _restore_cpu_state(self, backups: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]):
        # Restore module parameters and buffers
        module_param_backup, module_buffer_backup, other_backup = backups

        def restore_module_state(mod: torch.nn.Module, prefix: str):
            for name, param in mod.named_parameters():
                full_key = prefix + name
                if full_key in module_param_backup:
                    param.data = module_param_backup[full_key]

            for name, buffer in mod.named_buffers():
                full_key = prefix + name
                if full_key in module_buffer_backup:
                    buffer.data = module_buffer_backup[full_key]

        if isinstance(self.model, torch.nn.Module):
            restore_module_state(self.model, "")
        else:
            for name, attr_val in self.model.__dict__.items():
                if isinstance(attr_val, torch.nn.Module):
                    restore_module_state(attr_val, name + ".")

        if not isinstance(self.model, torch.nn.Module):
            for name, val in other_backup.items():
                setattr(self.model, name, val)

    def _save_cpu_copies(self):
        """Snapshot CPU copies of params/buffers (model may be on NPU). Used to later
        restore to CPU without an NPU->CPU copy, since weights are read-only in inference."""
        module_param_backup = {}
        module_buffer_backup = {}
        other_backup = {}

        def save(mod: torch.nn.Module, prefix: str):
            for name, param in mod.named_parameters():
                if param is not None:
                    module_param_backup[prefix + name] = param.detach().to(self.cpu_device)
            for name, buffer in mod.named_buffers():
                if buffer is not None:
                    module_buffer_backup[prefix + name] = buffer.detach().to(self.cpu_device)

        if isinstance(self.model, torch.nn.Module):
            save(self.model, "")
        else:
            for name, attr_val in self.model.__dict__.items():
                if isinstance(attr_val, torch.nn.Module):
                    save(attr_val, name + ".")
                elif isinstance(attr_val, torch.Tensor):
                    other_backup[name] = attr_val.detach().to(self.cpu_device)

        return module_param_backup, module_buffer_backup, other_backup

    def prepare_resident_backup(self):
        """Model is resident on NPU. Create CPU copies once (outside the timed Base stage)
        so enable_offload can reassign them later without an NPU->CPU copy."""
        object.__setattr__(self, "_resident_backup", self._save_cpu_copies())

    def disable_offload(self):
        # Idempotent: move model to NPU and run resident. No-op if already resident.
        if self.is_cpu_offload:
            object.__setattr__(self, "_resident_backup", self._backup_cpu_state())
            self.model.to(self.cuda_device)
            object.__setattr__(self, "is_cpu_offload", False)

    def enable_offload(self):
        # Idempotent: restore CPU tensors (no D2H copy) and re-enable per-call offload.
        if not self.is_cpu_offload:
            if self._resident_backup is None:
                raise RuntimeError(
                    "enable_offload() called before prepare_resident_backup()/disable_offload(); "
                    "no CPU backup available to restore."
                )
            torch.cuda.synchronize()
            self._restore_cpu_state(self._resident_backup)
            object.__setattr__(self, "_resident_backup", None)
            object.__setattr__(self, "is_cpu_offload", True)

    # Unified on/offload executor
    def _run_with_optional_offload(self, func: Callable[..., Any], *args, **kwargs):
        if self.is_cpu_offload and self.is_running_on_gpu:
            backups = self._backup_cpu_state()
            self.model.to(self.cuda_device)
            try:
                return func(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                self._restore_cpu_state(backups)
        else:
            # Make sure model and args are on the same device
            args = [
                arg.to(self.device) if isinstance(arg, torch.Tensor) and arg.device != self.device else arg
                for arg in args
            ]
            kwargs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) and v.device != self.device else v
                for k, v in kwargs.items()
            }
            return func(*args, **kwargs)

    # Direct call (equivalent to forward)
    def __call__(self, *args, **kwargs):
        return self._run_with_optional_offload(self.model.__call__, *args, **kwargs)

    # Explicit forward; some code calls model.forward(...)
    def forward(self, *args, **kwargs):
        return self._run_with_optional_offload(self.model.forward, *args, **kwargs)

    # Key: passthrough all attrs/methods. For callables, wrap with on/offload;
    # for non-compute methods, pass-through only with no device switch.
    def __getattr__(self, name: str):
        # Fetch attribute from the wrapped model first
        attr = getattr(self.model, name)

        if callable(attr):
            if name in self._self_returning_methods:
                def _self_wrapped(*args, **kwargs):
                    attr(*args, **kwargs)
                    return self

                return _self_wrapped

            # Wrap methods (except in whitelist)
            if name not in self._non_compute_methods:
                def _wrapped(*args, **kwargs):
                    return self._run_with_optional_offload(attr, *args, **kwargs)

                return _wrapped

        return attr

    def __dir__(self):
        return sorted(set(list(super().__dir__()) + dir(self.model)))

    def __setattr__(self, name: str, value: Any):
        raise AttributeError("CPUOffloadWrapper is immutable")

    def __repr__(self) -> str:
        return (
            f"CPUOffloadWrapper(is_cpu_offload={self.is_cpu_offload}, "
            f"is_running_on_gpu={self.is_running_on_gpu}, model={repr(self.model)})"
        )
