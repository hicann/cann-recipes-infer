# coding=utf-8
# Adapted from
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/__init__.py
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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

import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_s2v_14B import s2v_14B
from .wan_t2v_A14B import t2v_A14B
from .wan_ti2v_5B import ti2v_5B
from .wan_animate_14B import animate_14B

WAN_CONFIGS = {
    't2v-A14B': t2v_A14B,
    'i2v-A14B': i2v_A14B,
    'ti2v-5B': ti2v_5B,
    'animate-14B': animate_14B,
    's2v-14B': s2v_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '640*360': (640, 360),
    '360*640': (360, 640),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    '1024*704': (1024, 704),
    '704*1024': (704, 1024),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
    '1024*704': 1024 * 704,
    '704*1024': 704 * 1024,
    '640*360': 640 * 360,
    '360*640': 360 * 640,

}

SUPPORTED_SIZES = {
    't2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '640*360', '360*640'),
    'ti2v-5B': ('704*1280', '1280*704'),
    's2v-14B': ('720*1280', '1280*720', '480*832', '832*480', '1024*704',
                '704*1024', '704*1280', '1280*704'),
    'animate-14B': ('720*1280', '1280*720')
}
