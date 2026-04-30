# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/set_env.sh"
FUNCTION_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/function.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")

source ${SET_ENV_ABS_PATH}
source ${FUNCTION_ABS_PATH}

export MODEL_DIR=$(basename "$SCRIPT_PATH")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"

mode="$1"
pd_role="$2"

if [ "$mode" = "online" ]; then
    export PD_ROLE="$pd_role"
    export P_YAML="${YAML_PARENT_PATH}/qwen3_235b_pd/prefill.yaml"
    export D_YAML="${YAML_PARENT_PATH}/qwen3_235b_pd/decode.yaml"
    echo "====================> launch online inference"
else
    export YAML_FILE_NAME="${2:-qwen3_235b_16tp}" # override via 2nd arg, e.g. qwen3_235b_128ep
    export YAML="${YAML_PARENT_PATH}/${YAML_FILE_NAME}.yaml"
    echo "====================> launch offline inference (${YAML_FILE_NAME})"
fi

launch "$mode"
