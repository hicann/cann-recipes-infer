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
"""Copy new BailingMoeV3 files to installed vllm / vllm-ascend packages.

These files are *completely new* (not modifications of existing files), so
they are copied to the target package directories at patch-apply time.
"""

from __future__ import annotations

from .patch_core import copy_new_file

# (relative_path_in_Incremental_adapter_files, target_top_level_package)
_NEW_FILES: list[tuple[str, str]] = [
    # vllm model files
    ("vllm/model_executor/models/bailing_moe_v3.py", "vllm"),
    ("vllm/model_executor/models/bailing_moe_v3_mtp.py", "vllm"),
    # vllm-ascend ops
    ("vllm_ascend/ops/bailing_moe_v3_kda.py", "vllm_ascend"),
    # vllm-ascend pypto/kda
    ("vllm_ascend/ops/pypto/kda/chunk_kda_impl.py", "vllm_ascend"),
    ("vllm_ascend/ops/pypto/kda/fused_recurrent_kda_impl.py", "vllm_ascend"),
    # vllm-ascend triton/fla
    ("vllm_ascend/ops/triton/fla/op_kda.py", "vllm_ascend"),
    # vllm-ascend triton/kda package
    ("vllm_ascend/ops/triton/kda/__init__.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/chunk_delta_h.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/cumsum.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/fused_recurrent_kda.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/kda.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/l2norm.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/solve_tril.py", "vllm_ascend"),
    ("vllm_ascend/ops/triton/kda/utils.py", "vllm_ascend"),
]


def install_new_files() -> None:
    """Copy all new files to their target package locations."""
    for rel_path, pkg in _NEW_FILES:
        try:
            copy_new_file(rel_path, pkg)
        except Exception as e:
            print(f"[WARNING] Failed to copy {rel_path}: {e}")
