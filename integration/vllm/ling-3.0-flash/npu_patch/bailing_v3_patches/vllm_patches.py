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
"""Apply vLLM-side source diffs for BailingMoeV3 at apply time.

Each modified vLLM Python file has a corresponding ``.diff`` file in
``vllm_diffs/`` (extracted from ``bailing_v3_vllm.patch``, excluding new
files which are copied separately by :mod:`new_files_adapter`).

The diffs are applied to the installed source files *in place* (persistently)
via textual replacement.  This is equivalent to ``git apply`` of the
corresponding sections of ``bailing_v3_vllm.patch``, but does not require a
git repository and does not import the target modules.

This replaces the previous runtime monkey-patch approach, which could not
replicate source-level changes such as Triton kernel modifications
(``chunk_gla_fwd_kernel_o`` / ``kda_gate_fwd_kernel``) and missed references
imported via ``from ... import ...``.
"""

from __future__ import annotations

from pathlib import Path

from .patch_core import apply_diffs_from_dir

_DIFFS_DIR = Path(__file__).parent / "vllm_diffs"


def apply_vllm_diffs() -> int:
    """Apply all vLLM diff files to installed source files.

    Returns the number of successfully applied diffs.
    """
    return apply_diffs_from_dir(_DIFFS_DIR, label="vllm")
