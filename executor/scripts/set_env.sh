# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
export ON_CLOUD=0 # 0: local deployment, 1: for internal use on cloud servers
export IPs=('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx') # IPs of all servers. Please seperate multiple servers with blank space in between. The first one is the master server.

rm -rf /root/atc_data/

recipes_path="your_cann_recipes_path"
export PYTHONPATH=$PYTHONPATH:$recipes_path

cann_path="your_cann_pkgs_path"
source $cann_path/bin/setenv.bash
export ASCEND_HOME_PATH=$cann_path

driver_path="your_driver_path" # default is in /usr/local/Ascend/driver
source $driver_path/bin/setenv.bash
