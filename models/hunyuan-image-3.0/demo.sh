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
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CFG_PARALLEL=1
export USE_VAE_PARALLEL=1

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/set_env.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
source ${SET_ENV_ABS_PATH}

torchrun --master_port=10086 --nproc_per_node 16 run_image_gen.py --reproduce --model-id ./ckpts/weight_tp8 --verbose 0 --image-size 1024x1024 --moe-impl npu_grouped_matmul \
    --prompt "A cinematic medium shot captures a single Asian woman seated on a chair within a dimly lit room, creating an intimate and theatrical atmosphere. The composition is focused on the subject, rendered with rich colors and intricate textures that evoke a nostalgic and moody feeling.\n\nThe primary subject is a young Asian woman with a thoughtful and expressive countenance, her gaze directed slightly away from the camera. She is seated in a relaxed yet elegant posture on an ornate, vintage armchair. The chair is upholstered in a deep red velvet, its fabric showing detailed, intricate textures and slight signs of wear. She wears a simple, elegant dress in a dark teal hue, the material catching the light in a way that reveals its fine-woven texture. Her skin has a soft, matte quality, and the light delicately models the contours of her face and arms.\n\nThe surrounding room is characterized by its vintage decor, which contributes to the historic and evocative mood. In the immediate background, partially blurred due to a shallow depth of field consistent with a f/2.8 aperture, the wall is covered with wallpaper featuring a subtle, damask pattern. The overall color palette is a carefully balanced interplay of deep teal and rich red hues, creating a visually compelling and cohesive environment. The entire scene is detailed, from the fibers of the upholstery to the subtle patterns on the wall.\n\nThe lighting is highly dramatic and artistic, defined by high contrast and pronounced shadow play. A single key light source, positioned off-camera, projects gobo lighting patterns onto the scene, casting intricate shapes of light and shadow across the woman and the back wall. These dramatic shadows create a strong scense of depth and a theatrical quality. While some shadows are deep and defined, others remain soft, gently wrapping around the subject and preventing the loss of detail in darker areas. The soft focus on the background enhances the intimate feeling, drawing all attention to the expressive subject. The overall image presents a cinematic, photorealistic photography style."