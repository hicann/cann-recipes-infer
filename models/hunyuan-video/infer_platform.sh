#!/bin/bash
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# infer_platform.sh — end-to-end one-stop-platform quickstart for HunyuanVideo.
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

hf_download() {
    local repo="$1"
    local local_dir="$2"

    mkdir -p "$local_dir"
    command -v hf >/dev/null 2>&1 || die "hf CLI not found. Install with: python -m pip install 'huggingface_hub[cli]'"
    HF_ENDPOINT=https://hf-mirror.com hf download "$repo" --local-dir "$local_dir"
}

ensure_torch_runtime() {
    "$PYTHON_BIN" -m pip install "PyYAML" "numpy==1.26.4" > /dev/null || die "pip install torch_npu import dependencies failed"

    if "$PYTHON_BIN" -c "import torch, torch_npu" &>/dev/null; then
        echo "[platform] torch and torch_npu already installed, skipping runtime installation"
        return 0
    fi

    "$PYTHON_BIN" - <<'PYEOF' || die "Python 3.11 is required for the bundled torch / torch_npu wheels"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PYEOF

    command -v wget >/dev/null 2>&1 || die "wget not found, cannot download torch / torch_npu wheels"

    local arch
    local wheel_arch
    local torch_sha256
    local torch_npu_sha256
    arch=$(uname -m)
    case "$arch" in
        aarch64)
            wheel_arch="aarch64"
            torch_sha256="5fe6045b8f426bf2d0426e4fe009f1667a954ec2aeb82f1bd0bf60c6d7a85445"
            torch_npu_sha256="ac352efd7897701ac9290e09d2b9c4977907c3988198aad183ed0c6133263a38"
            ;;
        x86_64)
            wheel_arch="x86_64"
            torch_sha256="a1684793e352f03fa14f78857e55d65de4ada8405ded1da2bf4f452179c4b779"
            torch_npu_sha256="e3a463946b8ae13edb4a2fe035e652602c7c9eff86be8a57772d5e7f35432022"
            ;;
        *) die "unsupported CPU architecture: $arch" ;;
    esac

    local wheel_dir="$WEIGHTS_DIR/platform_wheels"
    local torch_wheel="torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_${wheel_arch}.whl"
    local torch_npu_wheel="torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_${wheel_arch}.whl"
    local torch_url="https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_${wheel_arch}.whl"
    local torch_npu_url="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/${torch_npu_wheel}"

    mkdir -p "$wheel_dir"
    echo "[platform] checking PyTorch wheel for $wheel_arch"
    download_with_sha256 "$torch_url" "$wheel_dir/$torch_wheel" "$torch_sha256"
    echo "[platform] checking torch_npu wheel for $wheel_arch"
    download_with_sha256 "$torch_npu_url" "$wheel_dir/$torch_npu_wheel" "$torch_npu_sha256"

    echo "[platform] installing PyTorch runtime"
    "$PYTHON_BIN" -m pip install "$wheel_dir/$torch_wheel" || die "install PyTorch wheel failed"
    "$PYTHON_BIN" -m pip install "$wheel_dir/$torch_npu_wheel" || die "install torch_npu wheel failed"
    "$PYTHON_BIN" -c "import torch, torch_npu" &>/dev/null || die "torch / torch_npu is still not importable after installation"
}

# ========== User configuration ==========
WEIGHTS_DIR="/mnt/workspace/gitCode/cann/models"
CANN_SET_ENV="/home/developer/Ascend/ascend-toolkit/set_env.sh"
# ========== End of user configuration ==========

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
WORK_DIR=$(cd "$REPO_ROOT/.." &>/dev/null && pwd)
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
PYTHON_BIN_DIR=$(dirname "$PYTHON_BIN")

HUNYUAN_DIR="$SCRIPT_PATH"
HUNYUAN_COMMIT="e260ed40c88d104801a8b1de05d2ab81e965a9ef"
export MODEL_BASE="$WEIGHTS_DIR/HunyuanVideo"
HUNYUAN_DIT="$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
LLAVA_RAW="$MODEL_BASE/llava-llama-3-8b-v1_1-transformers"
TEXT_ENCODER="$MODEL_BASE/text_encoder"
TEXT_ENCODER_2="$MODEL_BASE/text_encoder_2"

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

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WEIGHTS_DIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE" "$MODEL_BASE"

ensure_torch_runtime

if [ -f "$HUNYUAN_DIR/hyvideo/constants.py" ] && [ -f "$HUNYUAN_DIR/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py" ]; then
    echo "[platform] upstream HunyuanVideo source already merged, skipping"
else
    echo "[platform] merging upstream HunyuanVideo source"
    cd "$WORK_DIR" || die "cd $WORK_DIR failed"
    [ -d HunyuanVideo ] || git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git || die "git clone HunyuanVideo failed"
    if ! git -C HunyuanVideo cat-file -e "$HUNYUAN_COMMIT^{commit}" 2>/dev/null; then
        git -C HunyuanVideo fetch origin main || die "fetch HunyuanVideo main failed"
    fi
    git -C HunyuanVideo checkout --detach "$HUNYUAN_COMMIT" || die "checkout HunyuanVideo commit failed"
    cp -rn HunyuanVideo/* "$HUNYUAN_DIR/" || die "cp -rn HunyuanVideo/* failed"
fi

echo "[platform] ensuring Python runtime dependencies"
cd "$HUNYUAN_DIR" || die "cd $HUNYUAN_DIR failed"
"$PYTHON_BIN" -m pip install -r requirements.txt > /dev/null || die "pip install requirements failed"
"$PYTHON_BIN" -m pip install "huggingface_hub[cli]<1.0" > /dev/null || die "pip install huggingface_hub failed"

"$PYTHON_BIN" -c "import torch_npu" &>/dev/null || die "torch_npu is not importable in the active Python environment"

echo "[platform] checking model weights"
if [ ! -f "$HUNYUAN_DIT" ]; then
    hf_download tencent/HunyuanVideo "$MODEL_BASE" || die "download tencent/HunyuanVideo failed"
fi

if [ ! -d "$LLAVA_RAW" ]; then
    hf_download xtuner/llava-llama-3-8b-v1_1-transformers "$LLAVA_RAW" \
        || die "download llava text encoder failed"
fi

if [ ! -d "$TEXT_ENCODER" ]; then
    "$PYTHON_BIN" hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
        --input_dir "$LLAVA_RAW" \
        --output_dir "$TEXT_ENCODER" || die "preprocess text_encoder failed"
fi

if [ ! -d "$TEXT_ENCODER_2" ]; then
    hf_download openai/clip-vit-large-patch14 "$TEXT_ENCODER_2" \
        || die "download CLIP text_encoder_2 failed"
fi

TMP_YAML=$(mktemp -t hunyuan_platform.XXXXXX.yaml) || die "mktemp failed"
trap 'rm -f "$TMP_YAML"' EXIT
"$PYTHON_BIN" - <<PYEOF || die "generate temp yaml failed"
import yaml
with open("$HUNYUAN_DIR/config/single_platform.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model_args", {})["model-base"] = "$MODEL_BASE"
with open("$TMP_YAML", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF

echo "[platform] launching inference"
source "$REPO_ROOT/executor/scripts/mm_function.sh" || die "source mm_function.sh failed"

export MODEL_DIR=$(basename "$HUNYUAN_DIR")
export YAML_PARENT_PATH="$(dirname "$TMP_YAML")"
export YAML_FILE_NAME="$(basename "$TMP_YAML")"
export YAML="$TMP_YAML"

mm_launch
