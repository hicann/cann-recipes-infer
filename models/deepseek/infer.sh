# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
export YAML_FILE_NAME=ci/decode_r1_rank_16_16ep_a8w8.yaml # modify to your yaml file name
export YAML=${YAML_PARENT_PATH}/${YAML_FILE_NAME}

launch
