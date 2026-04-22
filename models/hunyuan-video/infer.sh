# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MM_FUNCTION_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/mm_function.sh"
MM_FUNCTION_ABS_PATH=$(realpath "${MM_FUNCTION_ABS_PATH}")

source ${MM_FUNCTION_ABS_PATH}

export MODEL_DIR=$(basename "$SCRIPT_PATH")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"
export YAML_FILE_NAME=sp8.yaml
export YAML=${YAML_PARENT_PATH}/${YAML_FILE_NAME}

mm_launch
