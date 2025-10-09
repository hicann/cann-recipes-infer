# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# User Guide:
# 1. To execute this script, please enter executor/scripts (cd executor/scripts).
# 2. Execute test_all_case.sh to run CI tests for all models, currently support deepseek, qwen3_moe and gpt_oss.
#    a. test one specific model: please provide the model folder name, eg. bash test_all_case.sh --models "deepseek"
#    b. test multiple models: seperate model folder names with blank space, eg. bash test_all_case.sh --models "deepseek qwen3_moe"
#    c. test all models: no need to provide specific names of models, eg. bash test_all_case.sh

#!/bin/bash
source set_env.sh
source function.sh

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
echo '----------------all case pass----------------'