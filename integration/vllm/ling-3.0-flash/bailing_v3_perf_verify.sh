#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
export VLLM_USE_V1=1
export VLLM_VERSION=0.20.2

## 4k 1.5k bs32
vllm bench serve \
    --backend=openai-chat  \
    --endpoint=/v1/chat/completions \
    --trust-remote-code \
    --model auto \
    --tokenizer "${MODEL_PATH}" \
    --num-prompts 16 \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 1536 \
    --ignore-eos \
    --request-rate 0.5 \
    --max-concurrency 8  \
    --temperature 0.6 \
    --metric-percentiles 50,90,99 \
    --base-url http://0.0.0.0:8000  > ./bench.log 2>&1 &