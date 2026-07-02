#!/bin/bash
# Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export TM_LOG_LEVEL=DEBUG
# get abs path of script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
rm -rf $SCRIPT_DIR/../../outputIds.txt
LAUNCH_PY="$SCRIPT_DIR/../../launch_inference.py"
python "$LAUNCH_PY" --config "$SCRIPT_DIR/inference_run_config_tp8_ep32_dp1.yaml" --data_path "$SCRIPT_DIR/../../HumanEval.jsonl" --batchSize 2
