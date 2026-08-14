#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# only offline need
export IPs=('xx.xx.xx.xx' 'xx.xx.xx.xx') # IPs of all servers. Please seperate multiple servers with blank space in between. The first one is the master server.

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