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

__all__ = ["get_window_topk_idxs", "get_compress_topk_idxs",
           "_prepare_4d_causal_attention_mask", "one_hot",
           "yarn_get_mscale", "DeepseekV3RMSNorm", "_init_rope", "apply_rotary_pos_emb",
           "DEEPSEEKV3_START_DOCSTRING", "DEEPSEEKV3_INPUTS_DOCSTRING", "DeepseekV3PreTrainedModel",
           "apply_rotary_emb", "rotate_activation",
           "Compressor", "Indexer", "CacheData", "AttnMetaData"]

from .common_modules import (get_window_topk_idxs, get_compress_topk_idxs,
                        _prepare_4d_causal_attention_mask, one_hot, yarn_get_mscale,
                        DeepseekV3RMSNorm, _init_rope, apply_rotary_pos_emb, DEEPSEEKV3_START_DOCSTRING,
                        DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel, apply_rotary_emb, rotate_activation,
                        )
from .compressor import Compressor
from .indexer import Indexer
from .attention_data import CacheData, AttnMetaData