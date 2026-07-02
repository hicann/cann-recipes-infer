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
# 整网 e2e 启动前预检（GGUF 层文件 + kt_kernel_ext 路径）。
#
# 用法：
#   bash tools/e2e_preflight.sh
#   GGUF_DIR=/path/cache GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh
#
# 退出码：0=通过，1=有缺失/路径不对。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/ensure_kt_kernel.sh
source "${SCRIPT_DIR}/ensure_kt_kernel.sh"
ensure_kt_kernel "$REPO"
# GGUF_DIR 必填：不硬编码环境特定路径（红线 R5）。指向 MXFP4 GGUF 所在目录。
if [[ -z "${GGUF_DIR:-}" ]]; then
  echo "[preflight] ERROR: 未设置 GGUF_DIR（MXFP4 GGUF 所在目录），例如：" >&2
  echo "[preflight]   GGUF_DIR=/your/cache bash $0" >&2
  exit 1
fi
GGUF_PREFIX="${GGUF_PREFIX:-dsv4_layer}"
# 批量 convert 默认输出 _mxfp4 后缀 → dsv4_layer3_mxfp4.gguf（与 batch_convert / verify / sha256 清单一致）。
GGUF_SUFFIX="${GGUF_SUFFIX:-_mxfp4}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-42}"
MIN_GIB="${MIN_GIB:-3}"  # MXFP4 单层约 3.42 GiB（截断/缺失更小）；Q8_0 请设 MIN_GIB=6，BF16 设 12

PYBIN="${PYTHON_BIN:-${PYBIN:-/usr/local/python3.11.14/bin/python3.11}}"

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
  echo "[preflight] 批量转换: $PYBIN $REPO/tools/batch_convert_mxfp4_layers_mp.py --input ... --output-dir $GGUF_DIR --layer-start $LAYER_START --layer-end $LAYER_END --skip-existing"
  exit 1
fi
echo "[preflight] OK: all $((LAYER_END - LAYER_START + 1)) GGUF files present (>= ${MIN_GIB} GiB each)."

echo "[preflight] kt_kernel_ext:"
"$PYBIN" -c "from kt_kernel import kt_kernel_ext; print('  ', kt_kernel_ext.__file__)"
so_path=$("$PYBIN" -c "from kt_kernel import kt_kernel_ext; print(kt_kernel_ext.__file__)")
if [[ "$so_path" == *"${REPO}/kt-kernel/python/"* || "$so_path" == *"${REPO}/kt-kernel/kt_kernel/"* ]]; then
  echo "[preflight] OK: kt_kernel_ext 在仓库 kt-kernel 包内"
else
  echo "[preflight] WARN: .so 不在 ${REPO}/kt-kernel/{python,kt_kernel}/ 下（见手册 §2.4）"
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
    echo "[preflight] OK: 已加载 /tmp/kt_kernel_build 同 inode（或已 cp 到 python/）"
  elif [[ -f "$bso" ]]; then
    echo "[preflight] HINT: build 目录有更新 .so，请执行:"
    echo "  cp -f $bso ${REPO}/kt-kernel/python/"
  fi
fi
echo "[preflight] PASS"
