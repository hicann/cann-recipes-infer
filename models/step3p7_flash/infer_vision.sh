#!/bin/bash
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

# Image+text inference launcher for Step-3.7-Flash.
#
# Reuses the SAME distributed env-setup as the standard offline launcher
# (executor/scripts/{set_env.sh,function.sh}: get_rank / check_env_vars /
# set_hccl) but runs models/step3p7_flash/infer_vision.py
# instead of the framework infer.py. The executor source is NOT modified.
#
# Usage:
#   bash infer_vision.sh                                          # default yaml + test image
#   bash infer_vision.sh step3p7_flash_rank16_attndp16_ep16.yaml
#   bash infer_vision.sh <yaml> "<prompt>" /path/to/image.jpg
#

set -uo pipefail
# NOTE: the framework's check_env_vars (executor/scripts/function.sh) REASSIGNS
# the bash var `SCRIPT_PATH` to executor/scripts. So we keep our own model-dir
# path under a private name (VISION_MODEL_DIR) that the sourced functions never
# touch, and use it for all model-local paths (YAML / infer_vision.py / image).
VISION_MODEL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(realpath "${VISION_MODEL_DIR}/../..")
EXEC_SCRIPTS="${REPO_ROOT}/executor/scripts"

YAML_FILE_NAME="${1:-step3p7_flash_rank16_attndp16_ep16.yaml}"
PROMPT="${2:-Describe this image in detail.}"
IMAGE="${3:-${VISION_MODEL_DIR}/dataset/test_image.jpg}"

# The default test image is NOT committed (licensing). Fail early with a hint
# rather than deep inside the distributed launch.
if [ ! -f "${IMAGE}" ]; then
  echo "[infer_vision] test image not found: ${IMAGE}"
  echo "  - fetch a public sample : bash ${VISION_MODEL_DIR}/dataset/fetch_test_image.sh"
  echo "  - or use your own image : bash infer_vision.sh ${YAML_FILE_NAME} \"<prompt>\" /path/to/image.jpg"
  exit 1
fi

export MODEL_DIR="step3p7_flash"
export MODE="offline"
export PD_ROLE=""
export YAML="${VISION_MODEL_DIR}/config/${YAML_FILE_NAME}"

# Reuse the framework's verified env setup (CANN, PYTHONPATH, ranks, HCCL, dirs).
source "${EXEC_SCRIPTS}/set_env.sh"
source "${EXEC_SCRIPTS}/function.sh"

check_launch
get_rank "${MODE}"
check_env_vars "${MODE}"
set_hccl

# Custom per-rank loop (mirrors function.sh:launch_infer_task offline branch),
# but pointing at infer_vision.py. Use VISION_MODEL_DIR
# (NOT SCRIPT_PATH, which check_env_vars overwrote to executor/scripts).
INFER_PATH="${VISION_MODEL_DIR}/infer_vision.py"
EXTRA_ARGS=(--prompt "${PROMPT}" --image "${IMAGE}")

cores=$(grep -c "processor" /proc/cpuinfo)
avg_core_per_rank=$(expr "${cores}" / "${MA_NUM_GPUS}")
core_gap=$(expr "${avg_core_per_rank}" - 1)
for ((i = 0; i < MA_NUM_GPUS; i++)); do
    start=$(expr "${i}" \* "${avg_core_per_rank}")
    end=$(expr "${start}" + "${core_gap}")
    cmdopt="${start}-${end}"
    export LOCAL_RANK="${i}"
    export RANK_ID=$(expr "${i}" + "${RANK_OFFSET}")
    cmd=(taskset -c "${cmdopt}" python3 "${INFER_PATH}" --yaml_file_path="${YAML}" "${EXTRA_ARGS[@]}")
    if [ "${i}" -eq 0 ]; then
        "${cmd[@]}" 2>&1 | tee "${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log" &
    else
        "${cmd[@]}" &>"${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log" &
    fi
done
wait
