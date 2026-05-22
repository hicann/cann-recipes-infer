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

set -eo pipefail

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
QWEN_ROOT=$(cd "${SCRIPT_PATH}/.." &>/dev/null && pwd)
SET_ENV_ABS_PATH="${QWEN_ROOT}/../../executor/scripts/set_env.sh"
FUNCTION_ABS_PATH="${QWEN_ROOT}/../../executor/scripts/function.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")

YAML_PATH="${1:-${SCRIPT_PATH}/qwen3_5_35b_mmlu_test.yaml}"
ENTRY_PATH="${BENCHMARK_ENTRY:-}"

source "${SET_ENV_ABS_PATH}"
source "${FUNCTION_ABS_PATH}"

# Override the shared launcher implementation so benchmark startup does not
# trip over `expr` returning exit code 1 when the first rank starts at core 0.
function launch_infer_task()
{
    local cores avg_core_per_rank core_gap i start end cmdopt
    cores=$(grep -c "processor" /proc/cpuinfo)
    avg_core_per_rank=$(( cores / MA_NUM_GPUS ))
    core_gap=$(( avg_core_per_rank - 1 ))
    for ((i = 0; i < MA_NUM_GPUS; i++)); do
        echo "${i}"
        start=$(( i * avg_core_per_rank ))
        end=$(( start + core_gap ))
        cmdopt="${start}-${end}"
        export LOCAL_RANK=${i}
        export RANK_ID=$(( i + RANK_OFFSET ))
        local infer_entry
        infer_entry="${INFER_PATH}"
        if [ -n "${ENTRY_PATH}" ]; then
            infer_entry="${ENTRY_PATH}"
        fi
        if [ "${i}" -eq 0 ] && [[ ${LAUNCH_MODE:-0} -ne 1 ]]; then
            taskset -c "${cmdopt}" python3 "${infer_entry}" \
                --yaml_file_path="${YAML}" 2>&1 | tee "${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log" &
        else
            taskset -c "${cmdopt}" python3 "${infer_entry}" \
                --yaml_file_path="${YAML}" &> "${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log" &
        fi
    done
    wait
}

export MODEL_DIR=$(basename "${QWEN_ROOT}")
export YAML="${YAML_PATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

launch
