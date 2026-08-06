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
# Reproducible generator for the sglang clean patch series.
# Base = public DSv4 sglang 298193eb3. Run from repo root.
set -euo pipefail
BASE=298193eb3
SG=third_party/sglang
# Output dir = the sglang/ next to this script (override with OUT=...); avoids
# hardcoding any release-dir name and works wherever this script is checked out.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/sglang}"
mkdir -p "$OUT"
gen() { local out="$1"; shift
  git -C "$SG" diff "$BASE" -- "$@" > "$OUT/$out"
  printf "  %-52s %5s lines\n" "$out" "$(wc -l < "$OUT/$out")"; }

echo "Generating sglang patches (base=$BASE):"
# 0001 — NPU KV / triton-ascend fallback to torch-equivalent (auto-detect, no env switch)
gen 0001-sglang-npu-kv-triton-fallback.patch \
  python/sglang/srt/hardware_backend/npu/allocator_npu.py \
  python/sglang/srt/mem_cache/common.py \
  python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py
# 0002 — KT EP wrapper (CPU MoE offload) + expert placement + scheduler/accel/args
gen 0002-sglang-kt-ep-cpu-moe-offload.patch \
  python/sglang/srt/layers/moe/kt_ep_wrapper.py \
  python/sglang/srt/layers/moe/kt_expert_masks.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/utils/kt_accel.py \
  python/sglang/srt/environ.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/models/deepseek_v2.py \
  test/manual/ascend/test_kt_cleanup_equiv.py
# 0003 — streaming prefill + depool (AscendC MXFP4 op) + dynamic-resident + GGUF dedup
gen 0003-sglang-streaming-prefill-depool.patch \
  python/sglang/srt/layers/moe/kt_stream_prefill.py
# 0004 — NSA compressor mode (CANN 8.5.0 / 9.0.0 compat) + NPU memory pools
gen 0004-sglang-nsa-compressor-and-mem-pools.patch \
  python/sglang/srt/hardware_backend/npu/nsa_compressor_mode.py \
  python/sglang/srt/layers/attention/nsa/nsa_indexer.py \
  python/sglang/srt/hardware_backend/npu/memory_pool_npu.py \
  python/sglang/srt/hardware_backend/npu/hybrid_swa_c4_c128_memory_pool.py
# 0005 — packaging (Ascend/NPU build config)
gen 0005-sglang-packaging.patch \
  python/pyproject.toml python/pyproject_npu.toml
echo "Done."
