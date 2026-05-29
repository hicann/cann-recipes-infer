#!/bin/bash
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# infer_platform.sh — end-to-end one-stop-platform quickstart for Wan2.2-I2V.
# Usage: edit the two paths in "User configuration", then run `bash infer_platform.sh`.
# The script is idempotent: completed source, dependency and weight steps are skipped.

die() { echo "[platform] ERROR: $*" >&2; exit 1; }

verify_sha256() {
    local file_path="$1"
    local expected_sha256="$2"
    local actual_sha256

    [ -f "$file_path" ] || return 1
    actual_sha256=$(sha256sum "$file_path" | awk '{print $1}')
    [ "$actual_sha256" = "$expected_sha256" ]
}

download_with_sha256() {
    local url="$1"
    local file_path="$2"
    local expected_sha256="$3"

    if [ -f "$file_path" ] && verify_sha256 "$file_path" "$expected_sha256"; then
        return 0
    fi
    if [ -f "$file_path" ]; then
        echo "[platform] checksum mismatch for cached $(basename "$file_path"), downloading again"
        rm -f "$file_path"
    fi

    wget -O "$file_path" "$url" || die "download $(basename "$file_path") failed"
    verify_sha256 "$file_path" "$expected_sha256" || die "checksum verification failed for $(basename "$file_path")"
}

ensure_torch_npu_import_deps() {
    "$PYTHON_BIN" -m pip install \
        "PyYAML" \
        "numpy>=1.23.5,<2" \
        "attrs>=23.0.0" \
        "decorator>=5.1.0" \
        "psutil>=5.9" \
        "scipy>=1.7.3" || die "pip install torch_npu import dependencies failed"
}

check_torch_runtime() {
    "$PYTHON_BIN" - <<'PYEOF'
import torch
import torch_npu
PYEOF
}

check_torch_version() {
    "$PYTHON_BIN" - <<'PYEOF'
import sys
import torch
expected = "2.7.1"
actual = torch.__version__.split("+", 1)[0]
if actual != expected:
    print(f"torch version mismatch: expected {expected}, got {torch.__version__}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
}

# ========== User configuration ==========
WEIGHTS_DIR="/mnt/workspace/gitCode/cann/models"
CANN_SET_ENV="/home/developer/Ascend/ascend-toolkit/set_env.sh"
SKIP_MODEL_DOWNLOAD=${SKIP_MODEL_DOWNLOAD:-0}
# ========== End of user configuration ==========

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
WORK_DIR=$(cd "$REPO_ROOT/.." &>/dev/null && pwd)
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
PYTHON_BIN_DIR=$(dirname "$PYTHON_BIN")

WAN_DIR="$SCRIPT_PATH"
WAN_COMMIT="42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN_MODEL_ID="Wan-AI/Wan2.2-I2V-A14B-BF16"
WAN_CKPT_DIR="$WEIGHTS_DIR/Wan2.2-I2V-A14B-BF16"

echo "[platform] source CANN env: $CANN_SET_ENV"
[ -f "$CANN_SET_ENV" ] || die "CANN set_env.sh not found at $CANN_SET_ENV"
source "$CANN_SET_ENV"
export PATH="$PYTHON_BIN_DIR:$PATH"
export PYTHONNOUSERSITE=1

for _glx in /lib/aarch64-linux-gnu/libGLdispatch.so.0 /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0; do
    if [ -f "$_glx" ]; then
        export LD_PRELOAD="$_glx${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "[platform] LD_PRELOAD += $_glx (TLS workaround)"
        break
    fi
done

export MODELSCOPE_CACHE="$WEIGHTS_DIR/modelscope_cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WEIGHTS_DIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HUB_CACHE" "$WAN_CKPT_DIR"

"$PYTHON_BIN" - <<'PYEOF' || die "Python 3.11 is required for the Wan2.2 torch / torch_npu wheels"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PYEOF

ensure_torch_npu_import_deps

if check_torch_runtime &>/dev/null; then
    echo "[platform] torch and torch_npu already installed, skipping runtime installation"
else
    echo "[platform] installing torch + torch_npu"
    ARCH=$(uname -m)
    local_torch_sha256=""
    local_torch_npu_sha256=""
    case "$ARCH" in
        aarch64)
            local_torch_sha256="5fe6045b8f426bf2d0426e4fe009f1667a954ec2aeb82f1bd0bf60c6d7a85445"
            local_torch_npu_sha256="ac352efd7897701ac9290e09d2b9c4977907c3988198aad183ed0c6133263a38"
            ;;
        x86_64)
            local_torch_sha256="a1684793e352f03fa14f78857e55d65de4ada8405ded1da2bf4f452179c4b779"
            local_torch_npu_sha256="e3a463946b8ae13edb4a2fe035e652602c7c9eff86be8a57772d5e7f35432022"
            ;;
        *) die "unsupported CPU architecture: $ARCH" ;;
    esac

    WHEEL_DIR="$WEIGHTS_DIR/platform_wheels"
    TORCH_WHL="torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_${ARCH}.whl"
    TORCH_NPU_WHL="torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_${ARCH}.whl"
    TORCH_URL="https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_${ARCH}.whl"
    TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/$TORCH_NPU_WHL"

    mkdir -p "$WHEEL_DIR"
    echo "[platform] checking PyTorch wheel for $ARCH"
    download_with_sha256 "$TORCH_URL" "$WHEEL_DIR/$TORCH_WHL" "$local_torch_sha256"
    echo "[platform] checking torch_npu wheel for $ARCH"
    download_with_sha256 "$TORCH_NPU_URL" "$WHEEL_DIR/$TORCH_NPU_WHL" "$local_torch_npu_sha256"

    "$PYTHON_BIN" -m pip install "$WHEEL_DIR/$TORCH_WHL" || die "pip install torch failed"
    "$PYTHON_BIN" -m pip install "$WHEEL_DIR/$TORCH_NPU_WHL" || die "pip install torch_npu failed"
    check_torch_version || die "torch version changed unexpectedly after torch_npu installation"
    check_torch_runtime || die "torch / torch_npu is still not importable after installation"
fi

if [ -f "$WAN_DIR/examples/i2v_input.JPG" ] && [ -f "$WAN_DIR/wan/utils/utils.py" ]; then
    echo "[platform] upstream Wan2.2 source already merged, skipping"
else
    echo "[platform] merging upstream Wan2.2 source"
    cd "$WORK_DIR" || die "cd $WORK_DIR failed"
    [ -d Wan2.2 ] || git clone https://github.com/Wan-Video/Wan2.2.git || die "git clone Wan2.2 failed"
    if ! git -C Wan2.2 cat-file -e "$WAN_COMMIT^{commit}" 2>/dev/null; then
        git -C Wan2.2 fetch origin main || die "fetch Wan2.2 main failed"
    fi
    git -C Wan2.2 checkout --detach "$WAN_COMMIT" || die "checkout Wan2.2 commit failed"
    cp -rn Wan2.2/* "$WAN_DIR/" || die "cp -rn Wan2.2/* failed"
fi

echo "[platform] ensuring Python runtime dependencies"
cd "$WAN_DIR" || die "cd $WAN_DIR failed"
"$PYTHON_BIN" -m pip install -r requirements.txt || die "pip install requirements failed"
"$PYTHON_BIN" -m pip install "modelscope" "PyYAML" || die "pip install modelscope / PyYAML failed"
check_torch_version || die "torch version was changed by dependency installation; recreate the env or reinstall the platform torch wheel"
check_torch_runtime || die "torch_npu is not importable in the active Python environment"

echo "[platform] checking model weights"
if [ "$SKIP_MODEL_DOWNLOAD" = "1" ]; then
    echo "[platform] SKIP_MODEL_DOWNLOAD=1, skipping ModelScope weight download"
elif [ ! -f "$WAN_CKPT_DIR/configuration.json" ] || [ ! -f "$WAN_CKPT_DIR/Wan2.1_VAE.pth" ]; then
    "$PYTHON_BIN" - <<PYEOF || die "download $WAN_MODEL_ID failed"
from modelscope import snapshot_download
snapshot_download("$WAN_MODEL_ID", local_dir="$WAN_CKPT_DIR")
PYEOF
fi

TMP_YAML=$(mktemp -t wan22_i2v_platform.XXXXXX.yaml) || die "mktemp failed"
trap 'rm -f "$TMP_YAML"' EXIT
"$PYTHON_BIN" - <<PYEOF || die "generate temp yaml failed"
import yaml
with open("$WAN_DIR/config/14b_single_platform.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model_args", {})["ckpt_dir"] = "$WAN_CKPT_DIR"
with open("$TMP_YAML", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF

echo "[platform] launching inference"
source "$REPO_ROOT/executor/scripts/mm_function.sh" || die "source mm_function.sh failed"

export MODEL_DIR=$(basename "$WAN_DIR")
export YAML_PARENT_PATH="$(dirname "$TMP_YAML")"
export YAML_FILE_NAME="$(basename "$TMP_YAML")"
export YAML="$TMP_YAML"

mm_launch
