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
export YAML_FILE_NAME=gpt_oss_20b.yaml  # modify to your yaml file name
export YAML=${YAML_PARENT_PATH}/${YAML_FILE_NAME}

launch
