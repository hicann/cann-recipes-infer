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
# Apply entry point for BailingMoeV3 on Ascend NPU
# (vllm 0.20.2 + vllm-ascend 0.20.2rc1)
#
# This module applies all BailingMoeV3 Python patches (new file copies +
# in-place source diffs for vllm and vllm-ascend) when invoked.  It is run
# directly by ``patch_bailing_v3.sh --monkey`` via:
#
#     PYTHONPATH=<npu_patch> python -c \
#         "from bailing_v3_patches import apply_all; apply_all()"
#
# The vllm-ascend C++ kernel patches are applied separately via ``git apply``
# by the shell script (bailing_v3_vllm_ascend_cpp.patch).

from bailing_v3_patches import apply_all

apply_all()
