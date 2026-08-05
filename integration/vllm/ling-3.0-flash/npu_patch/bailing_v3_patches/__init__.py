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
"""BailingMoeV3 patch package.

Applies all BailingMoeV3 patches at apply time:
  1. Copy new files to installed vllm / vllm-ascend package directories
  2. Apply vllm-ascend source diffs (modify installed .py files in place)
  3. Apply vllm source diffs (modify installed .py files in place)

All changes are persistent (written to the installed source files), which is
equivalent to ``git apply`` of ``bailing_v3_vllm.patch`` and
``bailing_v3_vllm_ascend.patch`` (the C++ parts of the latter are applied
separately via ``git apply`` by ``patch_bailing_v3.sh --monkey``).
"""

from __future__ import annotations

from .new_files_adapter import install_new_files
from .vllm_ascend_patches import apply_ascend_diffs
from .vllm_patches import apply_vllm_diffs


def apply_all() -> bool:
    """Apply all patches in the correct order.

    Order matters:
      1. New files copied first (so later source diffs that import them, e.g.
         ``registry.py`` importing ``bailing_moe_v3``, resolve correctly).
      2. vllm-ascend source diffs (modify .py files in place).
      3. vllm source diffs (modify .py files in place).
    """
    print("[INFO] === BailingMoeV3 patches: applying ===")

    print("[INFO] Step 1: Installing new files...")
    install_new_files()

    print("[INFO] Step 2: Applying vllm-ascend source diffs...")
    apply_ascend_diffs()

    print("[INFO] Step 3: Applying vllm source diffs...")
    apply_vllm_diffs()

    print("[INFO] === BailingMoeV3 patches: done ===")
    return True
