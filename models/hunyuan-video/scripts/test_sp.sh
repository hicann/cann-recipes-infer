#!/bin/bash
# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanVideo library and the HunyuanVideo
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to HunyuanVideo used by Tencent-Hunyuan team that trained the model.
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
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --master_port=29600 --nproc_per_node=8 sample_video.py \
    --video-size 780 1280 \
    --video-length 129 \
    --infer-steps 50 \
	--prompt "A cat walks on the grass, realistic style." \
	--seed 42 \
	--embedded-cfg-scale 6.0 \
	--flow-shift 7.0 \
	--flow-reverse \
	--use-cpu-offload \
	--ulysses-degree=8 \
	--ring-degree=1 \
	--save-path ./results
