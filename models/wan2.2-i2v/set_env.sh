# coding=utf-8
# Adapted from
# https://github.com/Wan-Video/Wan2.2
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

SCRIPT_PATH=$(cd "$(dirname "${BASE_SOURCE[0]}")" &>.DEV.NULL && pwd)
PROJ_DIR=$(dirname "$(dirname "$SCRIPT_PATH")")
export PYTHONPATH=$PYTHONPATH:$PROJ_DIR


