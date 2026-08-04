# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
# daVinci-MagiHuman unified inference script
# Usage: bash infer.sh <mode>
#   bash infer.sh base              # T2V base
#   bash infer.sh base_ti2v         # TI2V base
#   bash infer.sh distill           # T2V distill
#   bash infer.sh distill_ti2v      # TI2V distill
#   bash infer.sh sr_540p           # T2V super-resolution 540p
#   bash infer.sh sr_540p_ti2v      # TI2V super-resolution 540p
#   bash infer.sh sr_1080p          # T2V super-resolution 1080p
#   bash infer.sh sr_1080p_ti2v     # TI2V super-resolution 1080p

set -euo pipefail

MODE="${1:-base}"

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MM_FUNCTION_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/mm_function.sh"
MM_FUNCTION_ABS_PATH=$(realpath "${MM_FUNCTION_ABS_PATH}")

source "${MM_FUNCTION_ABS_PATH}"

export MODEL_DIR="$(basename "$SCRIPT_PATH")"
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"

case "${MODE}" in
    base|base_ti2v|distill|distill_ti2v|sr_540p|sr_540p_ti2v|sr_1080p|sr_1080p_ti2v)
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Usage: bash infer.sh <base|base_ti2v|distill|distill_ti2v|sr_540p|sr_540p_ti2v|sr_1080p|sr_1080p_ti2v>"
        exit 1
        ;;
esac

case "${MODE}" in
    sr_540p|sr_540p_ti2v|sr_1080p|sr_1080p_ti2v)
        export CPU_OFFLOAD="${CPU_OFFLOAD:-true}"
        export MAGI_COMPILE_OFFLOAD_CONFIG='{"model_cpu_offload":true,"gpu_resident_weight_ratio":0.35,"offload_policy":"HEURISTIC"}'
        ;;
esac

case "${MODE}" in
    sr_1080p|sr_1080p_ti2v)
        export SR2_1080="${SR2_1080:-true}"
        export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
        ;;
esac

PROMPT_FILE="${SCRIPT_PATH}/example/assets/prompt.txt"
RUNTIME_YAML="${YAML_PARENT_PATH}/${MODE}_runtime.yaml"

python3 <<GEN_RUNTIME_YAML
import yaml, sys, os

yaml_path = '${YAML_PARENT_PATH}/${MODE}.yaml'
runtime_path = '${RUNTIME_YAML}'
prompt_file = '${PROMPT_FILE}'

try:
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(prompt_file, 'r') as pf:
        prompt = pf.read().strip()
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]
    cfg['model_args']['prompt'] = prompt
    if os.path.exists(runtime_path):
        os.remove(runtime_path)
    with open(runtime_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("[INFO] Generated runtime YAML: " + runtime_path)
except Exception as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)
GEN_RUNTIME_YAML

if [ $? -ne 0 ]; then
    exit 1
fi

export YAML_FILE_NAME="${MODE}_runtime.yaml"
export YAML="${YAML_PARENT_PATH}/${YAML_FILE_NAME}"

export ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
export PROF_DIR="${PROF_DIR:-./prof/${MODE}_prof/}"

mm_launch
