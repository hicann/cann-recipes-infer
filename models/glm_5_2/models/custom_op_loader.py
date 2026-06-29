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
"""Loader for the custom AscendC op npu_gather_selection_kv_cache used by KV offload.

The op is a repo custom op (ops/ascendc/src/gather_selection_kv_cache) and must be compiled
and installed (see README "KV Offload"). We mount ONLY this op onto torch_npu instead of
`import custom_ops`, whose __init__ overrides working builtin torch_npu ops (e.g.
npu_moe_gating_top_k).
"""

import glob
import os
import sysconfig

import torch
import torch_npu


def _custom_ops_dir():
    return os.path.join(sysconfig.get_paths()["purelib"], "custom_ops")


def load_offload_gather_op():
    """Mount npu_gather_selection_kv_cache onto torch_npu from the installed custom_ops .so.

    No-op if the op is already present or the .so is absent (offload not installed). Covers
    eager and npugraph_ex.
    """
    if hasattr(torch_npu, "npu_gather_selection_kv_cache"):
        return
    so = glob.glob(os.path.join(_custom_ops_dir(), "custom_ops_lib*.so"))
    if so:
        torch.ops.load_library(so[0])
        torch_npu.npu_gather_selection_kv_cache = torch.ops.custom.npu_gather_selection_kv_cache


def register_offload_ge_converter():
    """Register the torchair GE/AscendIR converter for the gather op (ge_graph only).

    Loaded by file path so `import custom_ops` is not triggered. No-op if the converter is absent.
    """
    import importlib.util
    conv = glob.glob(os.path.join(_custom_ops_dir(), "converter", "npu_gather_selection_kv_cache.py"))
    if conv:
        spec = importlib.util.spec_from_file_location("_glm52_gather_ge_converter", conv[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
