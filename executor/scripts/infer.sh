# coding=utf-8
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

#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/set_env.sh"
FUNCTION_ABS_PATH="${SCRIPT_PATH}/function.sh"
VALIDATE_ABS_PATH="${SCRIPT_PATH}/validate_infer_args.py"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")
VALIDATE_ABS_PATH=$(realpath "${VALIDATE_ABS_PATH}")

source "${FUNCTION_ABS_PATH}"

# Set defaults here (edit directly) or override via command-line args (--model / --mode / --pd-role / --yaml).
MODEL="deepseek_r1"                           # model directory name under models/ (e.g. deepseek_r1, qwen3_moe, gpt_oss)
MODE="offline"                                 # inference mode: "online" (prefill-decode disaggregation), or offline
PD_ROLE=""                                     # required when MODE=online: prefill / decode
YAML_FILE="decode_r1_rank_16_16ep_a8w8.yaml"   # required when MODE!=online: yaml filename under models/<MODEL>/config/
P_YAML_NAME=""                                 # optional, overrides prefill yaml name (default: ${MODEL}_pd/prefill.yaml)
D_YAML_NAME=""                                 # optional, overrides decode yaml name (default: ${MODEL}_pd/decode.yaml)
ENGINE_BACKEND=""                              # optional, KV-transfer backend: memfabric | mooncake. Empty -> server.py default

# Parse --key value arguments (overrides defaults above).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2";     shift 2 ;;
        --mode)    MODE="$2";      shift 2 ;;
        --pd-role) PD_ROLE="$2";   shift 2 ;;
        --yaml)         YAML_FILE="$2";      shift 2 ;;
        --p-yaml-name) P_YAML_NAME="$2";    shift 2 ;;
        --d-yaml-name) D_YAML_NAME="$2";    shift 2 ;;
        --engine-backend) ENGINE_BACKEND="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model <name> --mode <online|offline> [--pd-role <role>] [--yaml <file>] [--p-yaml-name <name>] [--d-yaml-name <name>] [--engine-backend <backend>]"
            echo ""
            echo "  offline: $0 --model qwen3_moe --yaml qwen3_235b_16tp.yaml"
            echo "  online:  $0 --model deepseek_r1 --mode online --pd-role prefill"
            echo "  online (custom yaml): $0 --model deepseek_r1 --mode online --pd-role prefill --p-yaml-name my_prefill.yaml --d-yaml-name my_decode.yaml"
            echo "  online (mooncake KV transfer): $0 --model gpt_oss --mode online --pd-role prefill --engine-backend mooncake"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODEL_SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../models/${MODEL}/set_env.sh"
if [ -f "${MODEL_SET_ENV_ABS_PATH}" ]; then
    SET_ENV_ABS_PATH=$(realpath "${MODEL_SET_ENV_ABS_PATH}")
fi
source "${SET_ENV_ABS_PATH}"

export ENGINE_BACKEND

export MODEL_DIR="$MODEL"
export YAML_PARENT_PATH="${SCRIPT_PATH}/../../models/${MODEL}/config"

if [ "$MODE" = "online" ]; then
    export PD_ROLE="${PD_ROLE}"
    export P_YAML="${YAML_PARENT_PATH}/${P_YAML_NAME:-${MODEL}_pd/prefill.yaml}"
    export D_YAML="${YAML_PARENT_PATH}/${D_YAML_NAME:-${MODEL}_pd/decode.yaml}"
else
    export YAML="${YAML_PARENT_PATH}/${YAML_FILE}"
fi

python3 "${VALIDATE_ABS_PATH}" \
    --models-root "${SCRIPT_PATH}/../../models" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --pd-role "${PD_ROLE}" \
    --yaml "${YAML}" \
    --p-yaml "${P_YAML}" \
    --d-yaml "${D_YAML}" || exit 1

if [ "$MODE" = "online" ]; then
    echo "====================> launch online inference (model=${MODEL}, role=${PD_ROLE})"
else
    echo "====================> launch offline inference (model=${MODEL}, yaml=${YAML_FILE})"
fi

launch "$MODE"
