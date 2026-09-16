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

#!/bin/bash
# mHC AscendC ops (npu_hc_pre/post), built for deepseek_v4 under ops/ascendc
source "${ASCEND_HOME_PATH}/opp/vendors/customize/bin/set_env.bash"
# cann_ops_transformer causal_conv1d kernels (KDA short conv)
source "${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash"

# flash_kda / fused_recurrent_kda are a python DSL under ops/cannbot_dsl, so the
# repo root has to be importable (kimi_k3 does the same).
CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
RECIPES_PATH=$(dirname "$(dirname "$CURRENT_PATH")")
export PYTHONPATH=$PYTHONPATH:$RECIPES_PATH
