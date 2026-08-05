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
export HCCL_BUFFSIZE=400

vllm serve ${MODEL_PATH} \
    --port 8000 \
    --trust-remote-code \
    --served-model-name auto \
    --max-num-seqs 32 \
    --max-model-len 131072 \
    --max-num-batched-tokens 65536 \
    --block-size 128 \
    --gpu-memory-utilization 0.85 \
    --mamba_cache_dtype "float32" \
    --chat-template ${MODEL_PATH}/chat_template.jinja \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
