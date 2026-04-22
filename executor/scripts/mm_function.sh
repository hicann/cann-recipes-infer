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
    mm_parse_yaml
    mm_setup_env
    mm_generate_cache_config
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

    # Export model-specific environment variables from YAML (overrides defaults above)
    eval "$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$YAML'))
env_vars = cfg.get('env_vars', {})
for k, v in env_vars.items():
    print(f'export {k}=\"{v}\"')
")"

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

# Launch inference task using the configured launcher
function mm_launch_task()
{
    MODEL_ARGS=$(mm_build_args)
    echo "[INFO] Entry script: ${ENTRY_SCRIPT_PATH}"
    echo "[INFO] Model args: ${MODEL_ARGS}"

    cd "${MM_MODEL_DIR}"

    if [ "${LAUNCHER}" == "accelerate" ]; then
        ACCELERATE_EXTRA=$(mm_build_accelerate_args)
        echo "[INFO] Launching with accelerate (num_processes=${WORLD_SIZE}, main_process_port=${MASTER_PORT})"
        echo "[INFO] Accelerate extra args: ${ACCELERATE_EXTRA}"
        echo "==================================>"

        eval PYTHONPATH=${RECIPES_ROOT}:\$PYTHONPATH \
                 accelerate launch \
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
