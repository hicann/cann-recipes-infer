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
# One-key apply for the DeepSeek-V4-Flash single-NPU clean patch series.
#
# Prereq: clone the three repos at the EXACT pristine SHAs (see README §1):
#   <repo_root>                         ktransformers-AK @ d7b5b49
#   <repo_root>/third_party/sglang      sglang           @ 298193eb3
#   <repo_root>/third_party/llama.cpp   llama.cpp        @ a94e6ff (b3173)
#
# Usage:  bash apply_all.sh <repo_root>
#
# Run ONCE on the pristine SHAs above. This is not idempotent and has no rollback:
# if it fails partway (e.g. wrong base SHA), some repos may already be patched —
# revert each with `git -C <repo> checkout .` (or `git -C <repo> reset --hard <SHA>`)
# before retrying.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(pwd)}"
SGLANG="$ROOT/third_party/sglang"
LLAMA="$ROOT/third_party/llama.cpp"

die() { echo "ERROR: $*" >&2; exit 1; }
[ -d "$ROOT/kt-kernel" ]   || die "$ROOT is not the ktransformers-AK root (no kt-kernel/)"
[ -d "$SGLANG/python" ]    || die "missing $SGLANG (clone sglang @ 298193eb3 there)"
[ -d "$LLAMA" ]            || die "missing $LLAMA (clone llama.cpp @ a94e6ff there)"

# apply_set <target_dir> <patch_dir> [extra git-apply args...]
apply_set() {
  local target="$1" dir="$2"; shift 2
  local extra=("$@") p
  echo "== $dir -> $target =="
  for p in "$HERE/$dir"/*.patch; do
    [ -e "$p" ] || continue
    printf "  checking %-50s " "$(basename "$p")"
    git -C "$target" apply --check "${extra[@]}" "$p" \
      || die "patch does not apply cleanly (is the base SHA pinned correctly?): $p
       revert any partially-applied repos with 'git -C <repo> checkout .' before retrying."
    git -C "$target" apply "${extra[@]}" "$p"
    echo "applied"
  done
}

apply_set "$ROOT"    "main_repo"
apply_set "$SGLANG"  "sglang"
apply_set "$LLAMA"   "llama_cpp"  -p1

echo
echo "All code patches applied. Next: docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md (build kt-kernel, convert MXFP4 GGUF, launch, coherence smoke)."
echo "Reminder: convert/launch/verify scripts in scripts/ are standalone — copy them into tools/ , script/ , and the repo root as needed."
