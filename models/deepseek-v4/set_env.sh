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
export ON_CLOUD=0 # 0: local deployment, 1: for internal use on cloud servers
export IPs=('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx') # IPs of all servers. Please seperate multiple servers with blank space in between. The first one is the master server.

rm -rf /root/atc_data/

CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
RECIPES_PATH=$(dirname "$(dirname "$CURRENT_PATH")")
export PYTHONPATH=$PYTHONPATH:$RECIPES_PATH

cann_path="your_cann_pkgs_path"
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path
source $cann_path/opp/vendors/customize/bin/set_env.bash
source $cann_path/opp/vendors/custom_transformer/bin/set_env.bash

filename=$(basename "$YAML")
enable_core_num=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_config'].get('enable_limit_core', False))")
enable_multi_streams=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['model_config'].get('enable_multi_streams', False))")
if [ "$enable_core_num" = "True" ]; then
    export SC_BLOCK_OP="Compressor:InplacePartialRotaryMul"
    export FORCE_ENABLE_STATIC_SHAPE_KERNEL=True
fi