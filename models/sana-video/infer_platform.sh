# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
# infer_platform.sh — end-to-end one-stop-platform quickstart for SANA-Video.
# Activates CANN, installs torch/torch_npu, merges upstream Sana source,
# installs project deps + NPU mmcv, downloads weights via hf-mirror, patches
# a temporary YAML to point at local weights, and launches single-card
# python inference.
#
# Prerequisite: an active conda env with Python 3.10 (see README).
# Usage: edit the 2 paths in "User configuration" below, then `bash infer_platform.sh`.
# All steps are idempotent: rerunning the script safely skips completed steps.

# NOTE: do NOT `set -e` — mm_function.sh's mm_generate_cache_config runs a
# `python3 -c` whose exit status is then inspected with `if [ $? -ne 0 ]`,
# which would be killed prematurely by `set -e`. Use explicit checks instead.

die() { echo "[platform] ERROR: $*" >&2; exit 1; }

# ========== User configuration ==========
# Writable directory that will hold model weights and the HF cache.
WEIGHTS_DIR="/mnt/workspace/gitCode/cann/models"

# Path to the CANN toolkit's set_env.sh on this platform.
CANN_SET_ENV="/home/developer/Ascend/ascend-toolkit/set_env.sh"
# ========== End of user configuration ==========

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd "$SCRIPT_PATH/../.." &>/dev/null && pwd)
WORK_DIR=$(cd "$REPO_ROOT/.." &>/dev/null && pwd)

SANA_DIR="$SCRIPT_PATH"
DIT_PATH="$WEIGHTS_DIR/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"
VAE_PATH="$WEIGHTS_DIR/SANA-Video_2B_480p/vae/Wan2.1_VAE.pth"

echo "[platform] source CANN env: $CANN_SET_ENV"
[ -f "$CANN_SET_ENV" ] || die "CANN set_env.sh not found at $CANN_SET_ENV"
source "$CANN_SET_ENV"

# Work around glibc's static-TLS exhaustion that surfaces as
# "ImportError: libGLdispatch.so.0: cannot allocate memory in static TLS block"
# when CANN's shared libraries consume the static-TLS pool before cv2 tries to
# dlopen its native bindings. Pre-loading libGLdispatch through LD_PRELOAD
# puts it into the loader's initial TLS allotment instead of the static pool.
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
export DISABLE_XFORMERS=1
mkdir -p "$HF_HUB_CACHE"

GEMMA_CACHE="$HF_HUB_CACHE/models--Efficient-Large-Model--gemma-2-2b-it"

# 1. torch / torch_npu
if python -c "import torch_npu" &>/dev/null; then
    echo "[platform] torch_npu already installed, skipping"
else
    echo "[platform] installing torch + torch_npu"
    cd "$WORK_DIR" || die "cd $WORK_DIR failed"
    TORCH_WHL=torch-2.6.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
    TORCH_NPU_WHL=torch_npu-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl
    [ -f "$TORCH_WHL" ] || wget "https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl" || die "wget torch failed"
    python -m pip install "$TORCH_WHL" || die "pip install torch failed"
    [ -f "$TORCH_NPU_WHL" ] || wget "https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0-pytorch2.6.0/$TORCH_NPU_WHL" || die "wget torch_npu failed"
    python -m pip install "$TORCH_NPU_WHL" || die "pip install torch_npu failed"
fi

# 2. upstream Sana source merge
if [ -d "$SANA_DIR/diffusion" ]; then
    echo "[platform] upstream Sana source already merged, skipping"
else
    echo "[platform] merging upstream Sana source"
    cd "$WORK_DIR" || die "cd $WORK_DIR failed"
    [ -d Sana ] || git clone https://github.com/NVlabs/Sana.git || die "git clone Sana failed"
    (cd Sana && git checkout 08c656c3) || die "git checkout 08c656c3 failed"
    cp -rn Sana/* "$SANA_DIR/" || die "cp -rn Sana/* failed"
fi

# 3. project dependencies
# Use `pip install -e . --no-deps` so we don't hang on the
# `clip@git+https://github.com/openai/CLIP.git` entry in pyproject.toml
# (CLIP is only needed by offline metric tools under tools/metrics/, not by
# the inference entry). Then install a curated list of runtime deps that the
# inference path actually touches; skipping CLIP and dev-only packages like
# gradio / wandb / spaces / yapf / pre-commit / image-reward / pytorch-fid.
if python -m pip show sana &>/dev/null || [ -d "$SANA_DIR/sana.egg-info" ]; then
    echo "[platform] sana package already installed, skipping pip install -e ."
else
    echo "[platform] installing sana (pip install -e . --no-deps, skips CLIP git dep)"
    cd "$SANA_DIR" || die "cd $SANA_DIR failed"
    python -m pip install -e . --no-deps > /dev/null || die "pip install -e . --no-deps failed"
fi

RUNTIME_DEPS=(
    "setuptools<82,>=61.0"            # sana/pyproject pin; mmcv 1.x setup.py still imports pkg_resources
    "numpy<2"                         # sana/pyproject pins numpy<2; also needed for cv2 4.8 ABI
    # CANN TBE init requires these at first NPU call; unmet ones surface as
    # "Environment_Error_Import_Python_Module_Failed: No module named 'scipy'" etc.
    "scipy"
    "attrs"
    "decorator"
    "cloudpickle"
    "ml-dtypes"
    "tornado"
    "absl-py"
    "pyyaml"                          # used by mm_function.sh + various loaders
    "pytz"                            # diffusion/utils/logger.py
    "huggingface-hub==0.36.0"
    "accelerate==1.0.1"
    "diffusers==0.36.0"
    "peft==0.17.0"
    "transformers==4.57.0"
    "timm==0.6.13"
    "torchvision==0.21.0"
    "torchaudio==2.6.0"
    "mmengine"
    "addict"                          # imported by mmcv at top level
    "Pillow"                          # imported by mmcv.image
    "yapf"                            # mmcv Config uses yapf for formatting
    "packaging"
    "pyrallis"
    "fire"
    "termcolor"
    "einops"
    "ftfy"
    "ninja"
    "protobuf"
    "sentencepiece"
    "omegaconf"
    "imageio[pyav,ffmpeg]"
    "psutil"
    "regex"
    "tensorboardX"
    "came-pytorch"
    "moviepy"
    "patch_conv"
    "qwen-vl-utils"
)
echo "[platform] ensuring runtime dependencies"
python -m pip install "${RUNTIME_DEPS[@]}" > /dev/null || die "install runtime deps failed"

# ensure opencv-python-headless (mmcv imports cv2 at runtime; force-install to
# avoid `import mmcv` failing and being misdiagnosed as "mmcv not installed").
# If cv2 is importable, skip. Otherwise install — and if pip short-circuits
# on a "ghost" install where dist-info remains but the cv2/ dir is gone, fall
# back to --force-reinstall to re-lay the files.
if python -c "import cv2" &>/dev/null; then
    echo "[platform] opencv-python (cv2) already importable, skipping"
else
    echo "[platform] installing opencv-python-headless"
    python -m pip uninstall -y opencv-python 2>/dev/null
    python -m pip install opencv-python-headless==4.8.0.76 > /dev/null || die "install opencv-python-headless failed"
    if ! python -c "import cv2" &>/dev/null; then
        echo "[platform] cv2 still missing after install (likely stale dist-info), force-reinstalling"
        python -m pip install --force-reinstall --no-deps opencv-python-headless==4.8.0.76 > /dev/null \
            || die "force-reinstall opencv-python-headless failed"
        python -c "import cv2" &>/dev/null \
            || die "cv2 still not importable after force-reinstall"
    fi
fi

# 4. mmcv (1.x branch, NPU build). cv2 has already been ensured above, so if
# `import mmcv` fails here it really means the package is missing.
if python -c "import mmcv" &>/dev/null; then
    echo "[platform] mmcv already installed, skipping"
else
    echo "[platform] building NPU mmcv (1.x)"
    cd "$WORK_DIR" || die "cd $WORK_DIR failed"
    [ -d mmcv ] || git clone -b 1.x https://github.com/open-mmlab/mmcv.git || die "git clone mmcv failed"
    (cd mmcv && MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install) || die "mmcv build failed"
    # surface the real error if post-build import still fails
    python -c "import mmcv" || die "mmcv build completed but still not importable (see error above)"
fi

# 5. model weights (hf_download_or_fpath returns existing local path; only download missing files)
echo "[platform] checking model weights"
if [ ! -f "$DIT_PATH" ]; then
    hf download Efficient-Large-Model/SANA-Video_2B_480p checkpoints/SANA_Video_2B_480p.pth \
        --local-dir "$WEIGHTS_DIR/SANA-Video_2B_480p" || die "DiT weights download failed"
fi
if [ ! -f "$VAE_PATH" ]; then
    hf download Efficient-Large-Model/SANA-Video_2B_480p vae/Wan2.1_VAE.pth \
        --local-dir "$WEIGHTS_DIR/SANA-Video_2B_480p" || die "VAE weights download failed"
fi
if [ ! -d "$GEMMA_CACHE" ]; then
    hf download Efficient-Large-Model/gemma-2-2b-it || die "Gemma download failed"
fi

# 6. generate a temporary YAML with local paths (upstream yaml stays pristine)
TMP_YAML=$(mktemp -t sana_platform.XXXXXX.yaml) || die "mktemp failed"
trap 'rm -f "$TMP_YAML"' EXIT
python - <<PYEOF || die "generate temp yaml failed"
import yaml
with open("$SANA_DIR/config/2b_480p_single_platform.yaml") as f:
    cfg = yaml.safe_load(f)
ma = cfg.setdefault("model_args", {})
ma["model_path"] = "$DIT_PATH"
ma["vae.vae_pretrained"] = "$VAE_PATH"
ma["sample_nums"] = 1
with open("$TMP_YAML", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF

# 7. launch inference via mm_function.sh (python direct, no accelerate wrapper)
echo "[platform] launching inference"
source "$REPO_ROOT/executor/scripts/mm_function.sh" || die "source mm_function.sh failed"

export MODEL_DIR=$(basename "$SANA_DIR")
export YAML_PARENT_PATH="$(dirname "$TMP_YAML")"
export YAML_FILE_NAME="$(basename "$TMP_YAML")"
export YAML="$TMP_YAML"

mm_init_env
mm_check_launch
mm_parse_yaml
mm_setup_env
mm_generate_cache_config

MODEL_ARGS=$(mm_build_args)
cd "$MM_MODEL_DIR" || die "cd $MM_MODEL_DIR failed"
LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"

set -o pipefail
# CANN TBE compile-subprocesses are grandchildren of the Python process and
# cannot be reached via the in-process `pgrep -P` cleanup in the entry script.
# Filter that shutdown chatter from the terminal view only; the full log file
# is preserved by `tee` upstream of `grep`.
NOISE_PATTERN='TBE Subprocess\[task_distribute\].*main process disappeared'

eval PYTHONPATH="$RECIPES_ROOT:\$PYTHONPATH" \
     python -u "$ENTRY_SCRIPT_PATH" $MODEL_ARGS 2>&1 \
     | tee "$LOG_FILE" \
     | grep --line-buffered -v -E "$NOISE_PATTERN"

exit "${PIPESTATUS[0]}"
