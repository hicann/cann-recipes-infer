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
"""Loaders for custom AscendC KV offload ops.

The repo custom ops must be compiled and installed (see README "KV Offload").
For eager and npugraph_ex we only load the installed custom_ops .so. For
ge_graph we additionally register the torchair converters by file path. This
keeps the normal model path away from `import custom_ops`, whose __init__
blanket-mounts custom ops onto torch_npu.
"""

import glob
import importlib.util
import os
import sysconfig

import torch
import torch_npu

_REGISTERED_CONVERTERS = set()
_LOADED_HELPER_MODULES = {}
_DSA_NPUGRAPH_PATCHED_MODULES = None


def _custom_ops_dir():
    return os.path.join(sysconfig.get_paths()["purelib"], "custom_ops")


def _candidate_custom_ops_dirs():
    dirs = []
    spec = importlib.util.find_spec("custom_ops")
    if spec is not None and spec.submodule_search_locations:
        dirs.extend(spec.submodule_search_locations)
    dirs.append(_custom_ops_dir())
    deduped = []
    for path in dirs:
        if path and path not in deduped:
            deduped.append(path)
    return deduped


def _custom_op_registered(op_name):
    custom_ns = getattr(torch.ops, "custom", None)
    return custom_ns is not None and hasattr(custom_ns, op_name)


def _custom_op_has_privateuse1(op_name):
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(f"custom::{op_name}", "PrivateUse1")
    except RuntimeError:
        return False


def _load_custom_ops_library(required_op=None):
    if required_op is not None and _custom_op_registered(required_op):
        return True
    for custom_ops_dir in _candidate_custom_ops_dirs():
        so = glob.glob(os.path.join(custom_ops_dir, "custom_ops_lib*.so"))
        if not so:
            continue
        torch.ops.load_library(so[0])
        if required_op is None or _custom_op_registered(required_op):
            return True
    return required_op is None or _custom_op_registered(required_op)


def _register_converter(converter_name):
    if converter_name in _REGISTERED_CONVERTERS:
        return
    conv = []
    for custom_ops_dir in _candidate_custom_ops_dirs():
        conv = glob.glob(os.path.join(custom_ops_dir, "converter", f"{converter_name}.py"))
        if conv:
            break
    if not conv:
        return
    spec = importlib.util.spec_from_file_location(f"_glm52_{converter_name}_ge_converter", conv[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED_HELPER_MODULES[converter_name] = mod
    _REGISTERED_CONVERTERS.add(converter_name)


def _load_dsa_functionalization():
    module_name = "dsa_functionalization"
    if module_name in _LOADED_HELPER_MODULES:
        return
    candidates = []
    for custom_ops_dir in _candidate_custom_ops_dirs():
        candidates = glob.glob(os.path.join(custom_ops_dir, "converter", f"{module_name}.py"))
        if candidates:
            break
    if not candidates:
        raise RuntimeError("dsa_functionalization.py is missing from the installed custom ops")
    spec = importlib.util.spec_from_file_location(f"_glm52_{module_name}", candidates[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _LOADED_HELPER_MODULES[module_name] = mod


def load_offload_gather_op():
    """Mount npu_gather_selection_kv_cache onto torch_npu from the installed custom_ops .so.

    No-op if the op is already present or the .so is absent (offload not installed). Covers
    eager and npugraph_ex.
    """
    if hasattr(torch_npu, "npu_gather_selection_kv_cache"):
        return
    if _load_custom_ops_library("npu_gather_selection_kv_cache"):
        torch_npu.npu_gather_selection_kv_cache = torch.ops.custom.npu_gather_selection_kv_cache


def register_offload_ge_converter():
    """Register the torchair GE/AscendIR converter for the gather op (ge_graph only).

    Loaded by file path so `import custom_ops` is not triggered. No-op if the converter is absent.
    """
    _register_converter("npu_gather_selection_kv_cache")


def register_dsa_shared_ge_converters():
    """Register GE converters for the DsaPlan/DsaServe/DsaInstall route."""
    _load_dsa_functionalization()
    for converter_name in ("dsa_plan", "dsa_serve", "dsa_install"):
        _register_converter(converter_name)


def register_dsa_shared_npugraph_reinplace():
    """Install DSA-only TorchAir reinplace checks without importing custom_ops."""
    global _DSA_NPUGRAPH_PATCHED_MODULES
    if _DSA_NPUGRAPH_PATCHED_MODULES is not None:
        return _DSA_NPUGRAPH_PATCHED_MODULES
    registration_files = []
    for custom_ops_dir in _candidate_custom_ops_dirs():
        registration_files = glob.glob(
            os.path.join(custom_ops_dir, "converter", "dsa_inplace_registration.py")
        )
        if registration_files:
            break
    if not registration_files:
        raise RuntimeError("dsa_inplace_registration.py is missing from the installed custom ops")
    spec = importlib.util.spec_from_file_location(
        "_glm52_dsa_inplace_registration", registration_files[0]
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    serve_count = mod.register_dsa_inplace_pair(
        torch.ops.custom.dsa_serve_functional.default,
        torch.ops.custom.dsa_serve.default,
        (5, 6),
    )
    install_count = mod.register_dsa_inplace_pair(
        torch.ops.custom.dsa_install_functional.default,
        torch.ops.custom.dsa_install.default,
        (4, 5, 6, 7, 8),
    )
    if serve_count < 1 or install_count < 1:
        raise RuntimeError(
            "DSA reinplace registration found no compatible TorchAir module: "
            f"serve={serve_count}, install={install_count}"
        )
    _LOADED_HELPER_MODULES["dsa_inplace_registration"] = mod
    _DSA_NPUGRAPH_PATCHED_MODULES = min(serve_count, install_count)
    return _DSA_NPUGRAPH_PATCHED_MODULES


def load_dsa_shared_custom_ops(register_ge_converters=False, register_npugraph_reinplace=False):
    """Load DSA shared custom ops for eager, npugraph_ex, and ge_graph.

    GE graph can run with schema/Meta registration plus converters. Eager and
    npugraph_ex need real PrivateUse1 kernels; otherwise torch will CPU-fallback
    inside graph capture and fail on host-device copies.
    """
    required_ops = (
        "dsa_plan",
        "dsa_serve",
        "dsa_serve_functional",
        "dsa_install",
        "dsa_install_functional",
    )
    if not _load_custom_ops_library("dsa_plan"):
        return False
    if register_ge_converters:
        register_dsa_shared_ge_converters()
        return all(_custom_op_registered(op_name) for op_name in required_ops)
    if register_npugraph_reinplace:
        register_dsa_shared_npugraph_reinplace()
    return all(
        _custom_op_registered(op_name) and _custom_op_has_privateuse1(op_name)
        for op_name in required_ops
    )
