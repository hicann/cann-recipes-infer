# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash

# Compute project root path (do NOT export to PYTHONPATH globally)
function mm_init_env()
{
    MM_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    export RECIPES_ROOT=$(dirname "$(dirname "$MM_SCRIPT_DIR")")
}

function mm_launch()
{
    mm_init_env
    mm_check_launch
    mm_validate_yaml
    mm_parse_yaml
    mm_setup_env
    mm_generate_cache_config
    mm_generate_sparse_config
    mm_launch_task
}

# Check YAML file is set and exists
function mm_check_launch()
{
    if [ -z "${YAML}" ]; then
        echo "[ERROR] YAML is not set. Please export YAML before calling mm_launch."
        exit 1
    fi
    if [ ! -f "${YAML}" ]; then
        echo "[ERROR] YAML file not found: ${YAML}"
        exit 1
    fi
}

# Validate YAML structure: required top-level keys, allowed-keys whitelist, and enum
# values for nested dit_cache.method / sparse.method. Fails fast with a clear message
# before any subprocess is launched. model_args sub-keys are intentionally NOT checked
# here: argparse in the entry script already rejects unknown flags.
function mm_validate_yaml()
{
    python3 - <<PY_VALIDATE
import sys, yaml
try:
    with open("${YAML}", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"[ERROR] Failed to parse YAML ${YAML}: {e}")
    sys.exit(1)

if not isinstance(cfg, dict):
    print("[ERROR] YAML root must be a mapping.")
    sys.exit(1)

ALLOWED_TOP = {
    "model_name", "world_size", "master_port", "entry_script",
    "launcher", "launcher_args", "env_vars",
    "dit_cache", "sparse", "model_args",
}
REQUIRED_TOP = {"model_name", "world_size", "entry_script"}
CACHE_METHODS = {"NoCache", "FBCache", "TeaCache", "TaylorSeer"}
SPARSE_METHODS = {"no_sparse", "TopK", "SVG"}

unknown = set(cfg.keys()) - ALLOWED_TOP
if unknown:
    print(f"[ERROR] Unknown top-level YAML key(s): {sorted(unknown)}. Allowed: {sorted(ALLOWED_TOP)}")
    sys.exit(1)

missing = REQUIRED_TOP - set(cfg.keys())
if missing:
    print(f"[ERROR] Missing required top-level YAML key(s): {sorted(missing)}")
    sys.exit(1)

ws = cfg.get("world_size")
if not isinstance(ws, int) or ws <= 0:
    print(f"[ERROR] world_size must be a positive int, got {ws!r}")
    sys.exit(1)
mp = cfg.get("master_port", 29600)
if not isinstance(mp, int):
    print(f"[ERROR] master_port must be int, got {mp!r}")
    sys.exit(1)

dit_cache = cfg.get("dit_cache")
if dit_cache is not None:
    if not isinstance(dit_cache, dict):
        print("[ERROR] dit_cache must be a mapping.")
        sys.exit(1)
    m = dit_cache.get("method", "NoCache")
    if m not in CACHE_METHODS:
        print(f"[ERROR] dit_cache.method must be one of {sorted(CACHE_METHODS)}, got {m!r}")
        sys.exit(1)

sparse = cfg.get("sparse")
if sparse is not None:
    if not isinstance(sparse, dict):
        print("[ERROR] sparse must be a mapping.")
        sys.exit(1)
    m = sparse.get("method", "no_sparse")
    if m not in SPARSE_METHODS:
        print(f"[ERROR] sparse.method must be one of {sorted(SPARSE_METHODS)}, got {m!r}")
        sys.exit(1)

print("[INFO] YAML validation passed.")
PY_VALIDATE
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

# Parse common fields from YAML config
function mm_parse_yaml()
{
    export MODEL_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_name'])")
    export WORLD_SIZE=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['world_size'])")
    export MASTER_PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML')).get('master_port', 29600))")
    export ENTRY_SCRIPT=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['entry_script'])")
    export LAUNCHER=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML')).get('launcher', 'torchrun'))")

    echo "[INFO] model_name: ${MODEL_NAME}"
    echo "[INFO] world_size: ${WORLD_SIZE}"
    echo "[INFO] master_port: ${MASTER_PORT}"
    echo "[INFO] entry_script: ${ENTRY_SCRIPT}"
    echo "[INFO] launcher: ${LAUNCHER}"
}

# Setup environment variables, HCCL config, and log directories
function mm_setup_env()
{
    # Common NPU optimization environment variables (defaults, can be overridden by YAML env_vars)
    export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-"expandable_segments:True"}
    export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-"2"}
    export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-"1"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}

    # Work around glibc static-TLS exhaustion when CANN libraries are loaded
    # before cv2's libGLdispatch dependency. Preloading libGLdispatch puts it
    # into the loader's initial TLS allocation.
    for _glx in /lib/aarch64-linux-gnu/libGLdispatch.so.0 /usr/lib/aarch64-linux-gnu/libGLdispatch.so.0; do
        if [ -f "$_glx" ] && [[ ":${LD_PRELOAD:-}:" != *":$_glx:"* ]]; then
            export LD_PRELOAD="$_glx${LD_PRELOAD:+:$LD_PRELOAD}"
            echo "[INFO] LD_PRELOAD += $_glx (TLS workaround)"
            break
        fi
    done

    # Export model-specific environment variables from YAML (overrides defaults above)
    eval "$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML'))
env_vars = cfg.get('env_vars', {})
for k, v in env_vars.items():
    print(f'export {k}=\"{v}\"')
")"

    # Auto-derive MODEL_BASE env from model_args.model-base if user didn't set it explicitly.
    # HunyuanVideo's hyvideo/constants.py reads MODEL_BASE at import time and bakes it into
    # VAE_PATH / TEXT_ENCODER_PATH / TOKENIZER_PATH; the argparse --model-base only controls
    # DiT weight resolution. Without this auto-export, users must specify model-base twice
    # (once in model_args, once in env_vars) for VAE / text encoders to find local weights.
    # Only triggers when MODEL_BASE is not already exported (yaml env_vars takes precedence).
    if [ -z "${MODEL_BASE}" ]; then
        AUTO_MODEL_BASE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML')) or {}
ma = cfg.get('model_args') or {}
val = ma.get('model-base') or ma.get('model_base')
print(val if val is not None else '')
")
        if [ -n "${AUTO_MODEL_BASE}" ]; then
            export MODEL_BASE="${AUTO_MODEL_BASE}"
            echo "[INFO] Auto-derived MODEL_BASE from model_args.model-base: ${MODEL_BASE}"
        fi
    fi

    # HCCL distributed communication settings
    LOCAL_HOST=$(hostname -I | awk -F " " '{print$1}')
    export HCCL_IF_IP=${LOCAL_HOST}
    export HCCL_IF_BASE_PORT=23456
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200

    # Create log directory
    DATE=$(date +%Y%m%d)
    SCRIPT_PATH_FUNC=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    PARENT_PARENT_DIR=$(cd "$SCRIPT_PATH_FUNC/../.." &>/dev/null && pwd)
    export MM_MODEL_DIR="${PARENT_PARENT_DIR}/models/${MODEL_DIR}"
    export LOG_DIR="${MM_MODEL_DIR}/res/${DATE}/${MODEL_NAME}"
    mkdir -p "${LOG_DIR}"
    echo "[INFO] Log directory: ${LOG_DIR}"

    # Resolve entry script path (relative to model directory)
    export ENTRY_SCRIPT_PATH="${MM_MODEL_DIR}/${ENTRY_SCRIPT}"
    if [ ! -f "${ENTRY_SCRIPT_PATH}" ]; then
        echo "[ERROR] Entry script not found: ${ENTRY_SCRIPT_PATH}"
        exit 1
    fi
}

# If dit_cache is present in YAML, pass the YAML file directly as cache config.
# The Python cache_manager.load_cache_config() reads dit_cache from YAML natively.
# If dit_cache is absent, do nothing (model uses its own default or NoCache).
function mm_generate_cache_config()
{
    export CACHE_CONFIG_PATH=""
    export CACHE_CONFIG_ARG_NAME=""

    python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$YAML'))
sys.exit(0 if 'dit_cache' in cfg else 1)
" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "[INFO] No dit_cache section in YAML, skipping cache config."
        return
    fi

    CACHE_CONFIG_PATH="${YAML}"

    CACHE_CONFIG_ARG_NAME=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML'))
model = cfg.get('model_name', '')
print('cache-config' if 'hunyuan-video' in model else 'cache_config')
")
    echo "[INFO] Cache config: --${CACHE_CONFIG_ARG_NAME} ${CACHE_CONFIG_PATH}"
}

# If sparse is present in YAML, pass the YAML file directly as sparse config and
# propagate sparse.method via --sparse-method. load_sparse_config_from_file reads
# the `sparse` section natively. If sparse is absent, do nothing (argparse default
# --sparse-method=no_sparse keeps the sparse branch disabled).
function mm_generate_sparse_config()
{
    export SPARSE_METHOD_VALUE=""
    export SPARSE_CONFIG_PATH=""

    HAS_SPARSE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML')) or {}
print('1' if 'sparse' in cfg else '0')
")
    if [ "${HAS_SPARSE}" != "1" ]; then
        echo "[INFO] No sparse section in YAML, skipping sparse config."
        return
    fi

    SPARSE_METHOD_VALUE=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML'))
print(cfg['sparse'].get('method', 'no_sparse'))
")
    SPARSE_CONFIG_PATH="${YAML}"
    echo "[INFO] Sparse config: --sparse-method ${SPARSE_METHOD_VALUE} --sparse-attention-config ${SPARSE_CONFIG_PATH}"
}

# Build command-line arguments from YAML model_args section
# Rules:
#   list value           -> --key val1 val2 ... (expanded, for nargs="+")
#   string/number value  -> --key value
#   boolean true         -> --key (flag only)
#   boolean false        -> (skip, argparse uses its default)
#
# To force passing a boolean as an explicit value (e.g. for pyrallis/Hydra),
# use a string in YAML:  key: "False"  → --key False
#                         key: "True"   → --key True
function mm_build_args()
{
    MODEL_ARGS=$(python3 -c "
import shlex
import yaml
cfg = yaml.safe_load(open('$YAML'))
model_args = cfg.get('model_args', {})
parts = []
for k, v in model_args.items():
    if isinstance(v, bool):
        if v:
            parts.append(f'--{k}')
    elif isinstance(v, list):
        vals = ' '.join(shlex.quote(str(x)) for x in v)
        parts.append(f'--{k} {vals}')
    else:
        parts.append(f'--{k} {shlex.quote(str(v))}')
print(' '.join(parts))
")

    # Append cache config arg if dit_cache was present in YAML
    if [ -n "${CACHE_CONFIG_PATH}" ] && [ -n "${CACHE_CONFIG_ARG_NAME}" ]; then
        MODEL_ARGS="${MODEL_ARGS} --${CACHE_CONFIG_ARG_NAME} ${CACHE_CONFIG_PATH}"
    fi

    # Append sparse args if sparse section was present in YAML
    if [ -n "${SPARSE_METHOD_VALUE}" ] && [ -n "${SPARSE_CONFIG_PATH}" ]; then
        MODEL_ARGS="${MODEL_ARGS} --sparse-method ${SPARSE_METHOD_VALUE} --sparse-attention-config ${SPARSE_CONFIG_PATH}"
    fi

    echo "${MODEL_ARGS}"
}

# Build accelerate launcher arguments from YAML launcher_args section
function mm_build_accelerate_args()
{
    ACCELERATE_ARGS=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML'))
launcher_args = cfg.get('launcher_args', {})
parts = []
for k, v in launcher_args.items():
    parts.append(f'--{k}={v}')
print(' '.join(parts))
")
    echo "${ACCELERATE_ARGS}"
}

# Resolve accelerate from the active Python environment instead of relying on
# PATH, which may pick up a user-site script bound to a different interpreter.
function mm_resolve_accelerate_bin()
{
    ACCELERATE_BIN=$(python -c "
import os
import sys
print(os.path.join(os.path.dirname(sys.executable), 'accelerate'))
")
    if [ -x "${ACCELERATE_BIN}" ]; then
        echo "${ACCELERATE_BIN}"
        return
    fi

    command -v accelerate
}

# Launch inference task using the configured launcher
function mm_launch_task()
{
    MODEL_ARGS=$(mm_build_args)
    echo "[INFO] Entry script: ${ENTRY_SCRIPT_PATH}"
    echo "[INFO] Model args: ${MODEL_ARGS}"

    cd "${MM_MODEL_DIR}"

    if [ "${LAUNCHER}" == "accelerate" ]; then
        ACCELERATE_EXTRA=$(mm_build_accelerate_args)
        ACCELERATE_BIN=$(mm_resolve_accelerate_bin)
        if [ -z "${ACCELERATE_BIN}" ]; then
            echo "[ERROR] accelerate not found in current Python env or PATH."
            exit 1
        fi
        echo "[INFO] Launching with accelerate (num_processes=${WORLD_SIZE}, main_process_port=${MASTER_PORT})"
        echo "[INFO] Accelerate command: ${ACCELERATE_BIN}"
        echo "[INFO] Accelerate extra args: ${ACCELERATE_EXTRA}"
        echo "==================================>"

        eval PYTHONPATH=${RECIPES_ROOT}:\$PYTHONPATH \
                 ${ACCELERATE_BIN} launch \
                 --num_processes=${WORLD_SIZE} \
                 --num_machines=1 \
                 --main_process_port=${MASTER_PORT} \
                 ${ACCELERATE_EXTRA} \
                 ${ENTRY_SCRIPT_PATH} \
                 ${MODEL_ARGS} \
                 '2>&1 | tee "${LOG_DIR}/log_$(date +%Y%m%d_%H%M%S).log"'
    else
        echo "[INFO] Launching with torchrun (nproc_per_node=${WORLD_SIZE}, master_port=${MASTER_PORT})"
        echo "==================================>"

        eval PYTHONPATH=${RECIPES_ROOT}:\$PYTHONPATH \
                 torchrun --master_port=${MASTER_PORT} \
                 --nproc_per_node=${WORLD_SIZE} \
                 ${ENTRY_SCRIPT_PATH} \
                 ${MODEL_ARGS} \
                 '2>&1 | tee "${LOG_DIR}/log_$(date +%Y%m%d_%H%M%S).log"'
    fi
}
