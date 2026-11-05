#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH=$(realpath "${SCRIPT_PATH}/set_env.sh")
FUNCTION_ABS_PATH=$(realpath "${SCRIPT_PATH}/../../executor/scripts/function.sh")

source "${SET_ENV_ABS_PATH}"
source "${FUNCTION_ABS_PATH}"

export MODEL_DIR=$(basename "${SCRIPT_PATH}")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"
export YAML_FILE_NAME="${YAML_FILE_NAME:-kimi_k3_rank_32P_100k_1batch.yaml}"
export YAML="${YAML_PARENT_PATH}/${YAML_FILE_NAME}"

cp -- "${SCRIPT_PATH}/examples/dspark_rulerv2_100k/default_prompt.json" \
    "${SCRIPT_PATH}/../../dataset/default_prompt.json" || exit 1

launch
