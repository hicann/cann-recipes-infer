# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# prefill and decode master server ip required to connect between pd disaggregation
Prefill_Master_Server_IP="xx.xx.xx.xx"
Decode_Master_Server_IP="xx.xx.xx.xx"
# add sglang code path to PYTHONPATH
PATH_TO_SGLANG=/PATH/TO/SGLANG

export PYTHONPATH=${PATH_TO_SGLANG}/python:$PYTHONPATH

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 通信buffer
export HCCL_BUFFSIZE=1024

export ASCEND_MF_STORE_URL=tcp://${Prefill_Master_Server_IP}:24667

python3 -m sglang_router.launch_router \
--decode http://${Decode_Master_Server_IP}:30001 \
--prefill http://${Prefill_Master_Server_IP}:30001 \
--pd-disaggregation \
--policy cache_aware \
--host 0.0.0.0 \
--port 30002 \