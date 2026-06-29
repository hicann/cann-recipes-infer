#!/usr/bin/env bash
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
# Reproducible generator for the parent-repo (ktransformers-AK) clean patch series.
# Base = pristine d7b5b49 (0.6.2.post1). Run from repo root after clean-code is in the working tree.
set -euo pipefail
BASE=d7b5b49
# Output dir = the main_repo/ next to this script (override with OUT=...); avoids
# hardcoding any release-dir name and works wherever this script is checked out.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/main_repo}"
mkdir -p "$OUT"

gen() { # gen <outfile> <pathspec...>
  local out="$1"; shift
  git diff "$BASE" -- "$@" > "$OUT/$out"
  printf "  %-52s %5s lines, %3s files\n" "$out" "$(wc -l < "$OUT/$out")" "$(git diff --name-only "$BASE" -- "$@" | wc -l)"
}

echo "Generating main_repo patches (base=$BASE):"

# 0001 — kt-kernel Ascend NPU backend + build system (NPU adaptation)
gen 0001-kt-kernel-ascend-npu-backend.patch \
  kt-kernel/CMakeLists.txt kt-kernel/setup.py kt-kernel/pyproject.toml \
  kt-kernel/requirements.txt kt-kernel/install.sh kt-kernel/ext_bindings.cpp \
  kt-kernel/cpu_backend/ascend_callback_worker.cpp kt-kernel/cpu_backend/ascend_callback_worker.h \
  kt-kernel/cpu_backend/cpuinfer.h kt-kernel/cpu_backend/vendors/ascend_npu.h \
  kt-kernel/cpu_backend/vendors/vendor.h

# 0002 — kt-kernel CPU MoE native MXFP4 kernel + GGUF loaders (clean: no debug env)
gen 0002-kt-kernel-cpu-moe-mxfp4-kernel.patch \
  kt-kernel/operators/llamafile/moe.hpp \
  kt-kernel/python/experts_base.py kt-kernel/python/utils/llamafile.py kt-kernel/python/utils/loader.py

echo "Done. (tools/docs/launch-scripts are delivered standalone, NOT via patch.)"
