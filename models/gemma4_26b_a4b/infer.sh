# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
# Gemma-4-26B-A4B 推理入口
# 用法（与仓库其他模型保持一致：$1=mode, $2=yaml）:
#   bash infer.sh                                                  # offline + 默认 yaml (8卡 ge_graph + 8ep)
#   bash infer.sh offline gemma4_rank_8_8ep_decode.yaml            # 8卡 eager + 8ep
#   bash infer.sh offline gemma4_rank_8_8ep_npugraph_decode.yaml   # 8卡 npugraph + 8ep
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH=$(realpath "${SCRIPT_PATH}/../../executor/scripts/set_env.sh")
FUNCTION_ABS_PATH=$(realpath "${SCRIPT_PATH}/../../executor/scripts/function.sh")

source "${SET_ENV_ABS_PATH}"
source "${FUNCTION_ABS_PATH}"

export MODEL_DIR=$(basename "${SCRIPT_PATH}")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"

mode="${1:-offline}"

if [ "$mode" = "online" ]; then
    echo "[ERROR] gemma-4 does not currently support online inference."
    exit 1
else
    export YAML_FILE_NAME="${2:-gemma4_rank_8_8ep_gegraph_decode.yaml}"
    export YAML="${YAML_PARENT_PATH}/${YAML_FILE_NAME}"
    if [[ ! -f "${YAML}" ]]; then
        echo "[ERROR] YAML not found: ${YAML}"
        echo "Available yamls in config/:"
        ls "${YAML_PARENT_PATH}" | sed 's/^/  /'
        exit 1
    fi
    echo "====================> launch offline inference (${YAML_FILE_NAME})"
fi

launch "$mode"
