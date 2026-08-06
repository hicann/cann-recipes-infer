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
# Pre-flight check before starting the full pipeline (per-layer GGUF files + kt_kernel_ext path).
#
# Usage:
#   bash tools/e2e_preflight.sh
#   GGUF_DIR=/path/cache GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh
#
# Exit code: 0 = pass, 1 = something is missing or points at the wrong path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/ensure_kt_kernel.sh
source "${SCRIPT_DIR}/ensure_kt_kernel.sh"
ensure_kt_kernel "$REPO"
# GGUF_DIR is required: environment-specific paths are never hardcoded. Point it at the directory
# holding the MXFP4 GGUF files.
if [[ -z "${GGUF_DIR:-}" ]]; then
  echo "[preflight] ERROR: GGUF_DIR is not set (directory holding the MXFP4 GGUF files), e.g." >&2
  echo "[preflight]   GGUF_DIR=/your/cache bash $0" >&2
  exit 1
fi
GGUF_PREFIX="${GGUF_PREFIX:-dsv4_layer}"
# The batch converter writes the _mxfp4 suffix by default -> dsv4_layer3_mxfp4.gguf, matching
# batch_convert / verify / the sha256 manifest.
GGUF_SUFFIX="${GGUF_SUFFIX:-_mxfp4}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-42}"
MIN_GIB="${MIN_GIB:-3}"  # one MXFP4 layer is ~3.42 GiB (truncated ones are smaller); use 6 for Q8_0, 12 for BF16

# Shares resolve_python_bin with launch_ds4flash_npu.sh and asks for the same module list, so the
# preflight validates the very interpreter the server will use. kt_kernel is deliberately NOT in
# that list: it is what the preflight reports on, and including it would turn a load failure
# (a missing libhwloc, say) into "no interpreter found" and hide the real cause.
# shellcheck source=tools/python_env.sh
source "${SCRIPT_DIR}/python_env.sh"
PYBIN="${PYBIN:-$(resolve_python_bin preflight numpy torch torch_npu sglang)}" || exit 1
echo "[preflight] PYBIN=${PYBIN}"

echo "[preflight] REPO=$REPO"
echo "[preflight] GGUF: ${GGUF_DIR}/${GGUF_PREFIX}{L}${GGUF_SUFFIX}.gguf  layers ${LAYER_START}-${LAYER_END}"

missing=0
for L in $(seq "$LAYER_START" "$LAYER_END"); do
  f="${GGUF_DIR}/${GGUF_PREFIX}${L}${GGUF_SUFFIX}.gguf"
  if [[ ! -f "$f" ]]; then
    echo "[preflight] MISSING $f"
    missing=$((missing + 1))
    continue
  fi
  sz=$(stat -c%s "$f")
  min_bytes=$((MIN_GIB * 1024 * 1024 * 1024))
  if (( sz < min_bytes )); then
    echo "[preflight] TOO_SMALL $f ($(numfmt --to=iec "$sz" 2>/dev/null || echo "${sz}B") < ${MIN_GIB}GiB)"
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "[preflight] FAIL: ${missing} layer file(s) missing or too small."
  echo "[preflight] Batch convert with: $PYBIN $REPO/tools/batch_convert_mxfp4_layers_mp.py --input ... --output-dir $GGUF_DIR --layer-start $LAYER_START --layer-end $LAYER_END --skip-existing"
  exit 1
fi
echo "[preflight] OK: all $((LAYER_END - LAYER_START + 1)) GGUF files present (>= ${MIN_GIB} GiB each)."

echo "[preflight] kt_kernel_ext:"
"$PYBIN" -c "from kt_kernel import kt_kernel_ext; print('  ', kt_kernel_ext.__file__)"
so_path=$("$PYBIN" -c "from kt_kernel import kt_kernel_ext; print(kt_kernel_ext.__file__)")
if [[ "$so_path" == *"${REPO}/kt-kernel/python/"* || "$so_path" == *"${REPO}/kt-kernel/kt_kernel/"* ]]; then
  echo "[preflight] OK: kt_kernel_ext resolves inside the repo's kt-kernel package"
else
  echo "[preflight] WARN: the .so is not under ${REPO}/kt-kernel/{python,kt_kernel}/ (see guide 4.1)"
fi
shopt -s nullglob
build_so_candidates=(/tmp/kt_kernel_build/kt_kernel_ext.cpython-*-linux-gnu.so)
shopt -u nullglob
if (( ${#build_so_candidates[@]} > 0 )); then
  # newest by mtime via -nt comparison (avoid parsing ls)
  bso="${build_so_candidates[0]}"
  for cand in "${build_so_candidates[@]}"; do
    [[ "$cand" -nt "$bso" ]] && bso="$cand"
  done
  if [[ -f "$bso" && "$so_path" -ef "$bso" ]]; then
    echo "[preflight] OK: the loaded .so is the same inode as /tmp/kt_kernel_build (or already copied into python/)"
  elif [[ -f "$bso" ]]; then
    echo "[preflight] HINT: the build directory has a newer .so, run:"
    echo "  cp -f $bso ${REPO}/kt-kernel/python/"
  fi
fi
echo "[preflight] PASS"
