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

# only offline need
# IPs of all servers. Please separate multiple servers with blank space in between.
# The first one is the master server.
export IPs=('xx.xx.xx.xx' 'xx.xx.xx.xx')

# only online need
# When prefill and decode share the same host, set ASCEND_RT_VISIBLE_DEVICES
# before calling infer.sh to isolate NPUs per role.
PREFILL_IPS=('xx.xx.xx.xx')
DECODE_IPS=('xx.xx.xx.xx')

rm -rf /root/atc_data/

CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
RECIPES_PATH=$(dirname "$(dirname "$CURRENT_PATH")")
export PYTHONPATH=$PYTHONPATH:$RECIPES_PATH

cann_path="your_cann_pkgs_path"
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path
