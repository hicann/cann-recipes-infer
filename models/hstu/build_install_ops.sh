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

#!/bin/bash
set -euo pipefail

########################################
# One-click build & install RecSDK ops
# Usage:
#   bash onekey_build_install_ops.sh
#   bash onekey_build_install_ops.sh A2
#   bash onekey_build_install_ops.sh A2 /path/to/RecSDK
########################################

ARCH_TAG="${1:-A2}"
USER_RECSKD_ROOT="${2:-}"

log()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m $*" 1>&2; }

if [[ -z "${BASH_VERSION:-}" ]]; then
  err "Please run this script with bash."
  exit 1
fi

find_recsdk_root() {
  if [[ -n "$USER_RECSKD_ROOT" ]]; then
    echo "$USER_RECSKD_ROOT"
    return
  fi

  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    if [[ -d "$d/RecSDK/cust_op" ]]; then
      echo "$d/RecSDK"
      return
    fi
    if [[ -d "$d/cust_op" && -d "$d/cust_op/ascendc_op" ]]; then
      echo "$d"
      return
    fi
    d="$(dirname "$d")"
  done
  echo ""
}

RECSKD_ROOT="$(find_recsdk_root)"
if [[ -z "$RECSKD_ROOT" ]]; then
  err "Cannot locate RecSDK root. Please run from within RecSDK tree or pass RecSDK path as 2nd arg."
  err "Example: bash onekey_build_install_ops.sh A2 /path/to/RecSDK"
  exit 1
fi

if [[ ! -d "$RECSKD_ROOT/cust_op/ascendc_op" ]]; then
  err "Invalid RecSDK root: $RECSKD_ROOT (missing cust_op/ascendc_op)"
  exit 1
fi

log "RecSDK root: $RECSKD_ROOT"
log "Target arch tag: $ARCH_TAG"

# 1. Build AI Core op
BUILD_DIR="$RECSKD_ROOT/cust_op/ascendc_op/build"
BUILD_SCRIPT="$BUILD_DIR/build_ai_core_op.sh"

if [[ ! -f "$BUILD_SCRIPT" ]]; then
  err "Missing build script: $BUILD_SCRIPT"
  exit 1
fi

log "Step 1/4: Build AI Core ops"
pushd "$BUILD_DIR" >/dev/null
bash ./build_ai_core_op.sh "$ARCH_TAG"
popd >/dev/null

# 2. Extract tarball
OUTPUT_DIR="$RECSKD_ROOT/cust_op/ascendc_op/output"
if [[ ! -d "$OUTPUT_DIR" ]]; then
  err "Missing output dir: $OUTPUT_DIR"
  exit 1
fi

TARBALL="$OUTPUT_DIR/Ascend-recsdk-npu-ops-${ARCH_TAG}-linux-aarch64.tar.gz"
if [[ ! -f "$TARBALL" ]]; then
  err "Cannot find tarball: $TARBALL"
  err "Please check build output under: $OUTPUT_DIR"
  exit 1
fi

log "Step 2/4: Extract tarball"
pushd "$OUTPUT_DIR" >/dev/null
tar zxvf "$(basename "$TARBALL")"
popd >/dev/null

OPS_DIR="$OUTPUT_DIR/recsdk-npu-ops/recsdk_ops"
if [[ ! -d "$OPS_DIR" ]]; then
  err "Missing ops dir after extract: $OPS_DIR"
  exit 1
fi

# 3. Install required ops + in_linear_silu
log "Step 3/4: Install required operator packages"

REQUIRED_RUNS=(
  mxrec_opp_asynchronous_complete_cumsum.run
  mxrec_opp_concat_2d_jagged.run
  mxrec_opp_concat_2d_jagged_grad.run
  mxrec_opp_dense_to_jagged.run
  mxrec_opp_index_select_for_rank1_backward.run
  mxrec_opp_jagged_to_padded_dense.run
  mxrec_opp_gather_for_rank1.run
  mxrec_opp_hstu_dense_forward.run
  mxrec_opp_hstu_dense_backward.run
  mxrec_opp_split_embedding_codegen_forward_unweighted.run
)

pushd "$OPS_DIR" >/dev/null
for f in "${REQUIRED_RUNS[@]}"; do
  if [[ -f "$f" ]]; then
    log "Installing: $f"
    bash "$f"
  else
    warn "Not found, skip: $OPS_DIR/$f"
  fi
done
popd >/dev/null

# install in_linear_silu
IN_LINEAR_DIR="$RECSKD_ROOT/cust_op/ascendc_op/ai_core_op/in_linear_silu/v220"
IN_LINEAR_RUN="$IN_LINEAR_DIR/run.sh"
if [[ -f "$IN_LINEAR_RUN" ]]; then
  log "Installing in_linear_silu: $IN_LINEAR_RUN"
  pushd "$IN_LINEAR_DIR" >/dev/null
  bash run.sh
  popd >/dev/null
else
  warn "in_linear_silu run.sh not found, skip: $IN_LINEAR_RUN"
fi

# 4. Build torch adaptation library
TORCH_ADAPTER_DIR="$RECSKD_ROOT/cust_op/framework/torch_plugin/torch_library/common"
TORCH_ADAPTER_SCRIPT="$TORCH_ADAPTER_DIR/build_ops.sh"

if [[ ! -f "$TORCH_ADAPTER_SCRIPT" ]]; then
  err "Missing torch adapter build script: $TORCH_ADAPTER_SCRIPT"
  exit 1
fi

log "Step 4/4: Build torch adaptation library"
pushd "$TORCH_ADAPTER_DIR" >/dev/null
bash build_ops.sh
popd >/dev/null

log "All done"
log "Tarball: $TARBALL"
log "Ops installed from: $OPS_DIR"
log "Torch adapter built in: $TORCH_ADAPTER_DIR"
