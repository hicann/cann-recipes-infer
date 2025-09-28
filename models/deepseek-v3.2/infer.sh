# coding=utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
source set_env.sh
source function.sh

check_launch

export YAML_PARENT_PATH=config
export YAML_FILE_NAME=deepseek_v3.2_rank_64_64ep_prefill.yaml
export YAML=${YAML_PARENT_PATH}/${YAML_FILE_NAME}

get_rank
check_env_vars
launch_infer_task
save_key_info