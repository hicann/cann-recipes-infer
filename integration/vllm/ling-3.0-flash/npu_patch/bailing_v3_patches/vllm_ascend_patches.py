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
"""Apply vllm-ascend Python source diffs at apply time.

Each modified vllm-ascend Python file has a corresponding ``.diff`` file in
``vllm_ascend_diffs/``.  This module applies every diff to the installed
source file via textual replacement (no ``git apply`` required, no import of
the target module).

The source files are modified *in place* (persistently), which is equivalent
to ``git apply`` of the corresponding sections of
``bailing_v3_vllm_ascend.patch``.
"""

from __future__ import annotations

from pathlib import Path

from .patch_core import apply_diffs_from_dir

_DIFFS_DIR = Path(__file__).parent / "vllm_ascend_diffs"


def apply_ascend_diffs() -> int:
    """Apply all vllm-ascend diff files to installed source files.

    Returns the number of successfully applied diffs.
    """
    return apply_diffs_from_dir(_DIFFS_DIR, label="vllm-ascend")
