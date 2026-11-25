# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

# User Guide:
# 1. To execute this script, please enter executor/scripts (cd executor/scripts).
# 2. Execute test_all_case.sh to run CI tests for all models, currently support deepseek_r1, qwen3-moe.
#    a. test one specific model: please provide the model folder name, eg. bash test_all_case.sh --models "deepseek_r1"
#    b. test multiple models: separate model folder names with blank space,
#       eg. bash test_all_case.sh --models "deepseek_r1 qwen3-moe"
#    c. test all models: no need to provide specific names of models, eg. bash test_all_case.sh

#!/bin/bash
source set_env.sh
source function.sh

function extract_result()
{
    # Extract inference result from log_0.log,
    # If multiple queries are input, only show the result of the last query.
    find res -name "log_0.log" -exec sh -c '
        for file do
            echo "=== The following contents are extracted from the [$file] ==="
            tac "$file" | sed -n "/run success/,/decode result/{p; /decode result/q}" | tac
            echo ""
            echo ""
        done
    ' sh {} +
}

models=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            if [ -n "$2" ]; then
                IFS=' ' read -ra model_array <<< "$2"
                shift 2
            else
                echo "ERROR: the parameter '--models' requires specifying model name. "
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown parameter $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
MODELS_DIR="${PROJECT_ROOT}/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: the directory $MODELS_DIR doesn't exist. "
    exit 1
fi

export LAUNCH_MODE=1
export IPs_ORI=("${IPs[@]}")

if [ ${#model_array[@]} -eq 0 ]; then
    # 从MODELS_DIR目录下获取所有文件夹名称
    echo "If the '--models' parameter is not specified, all models under $MODELS_DIR will be tested."
    while IFS= read -r dir; do
        dir_name=$(basename "$dir")
        model_array+=("$dir_name")
    done < <(find "$MODELS_DIR" -maxdepth 1 -type d ! -name "$(basename "$MODELS_DIR")")
fi

for model in "${model_array[@]}"; do
    model_dir=${MODELS_DIR}/${model}
    if [ ! -d "$model_dir" ]; then
        echo "ERROR: the directory $model_dir doesn't exist."
        exit 1
    fi
    echo '================start test model' $model '================'
    find "$model_dir" -type f -path "*/config/ci/*.yaml" | sort | while read -r yaml_file; do
        echo '----------------start test case' $yaml_file '----------------'
        IPs=("${IPs_ORI[@]}")
        export YAML=$yaml_file
        export MODEL_DIR=${model}
        launch
        wait
        check_result
        echo '----------------finish test case' $yaml_file '----------------'
    done
done
extract_result > all_case_rank0_result.log 2>&1
echo '----------------all case pass----------------'
