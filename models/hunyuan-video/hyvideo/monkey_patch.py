# Copyright (c) 2025, HUAWEI CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import sys
import importlib
import types
import os
import math
import torch
import torch_npu
from loguru import logger

import yunchang.kernels.attention as attn_mod


def npu_fia(q, k, v, 
        dropout_p=0.0, 
        softmax_scale=None, 
        causal=False, 
        window_size=(-1, -1), 
        softcap=None, 
        alibi_slopes=None, 
        return_softmax=False):
    b, s, n, d = q.shape
    head_num = n
    scale = 1.0 / math.sqrt(d)
    block_out, block_lse = torch_npu.npu_fused_infer_attention_score(
                                                q.transpose(1, 2), 
                                                k.transpose(1, 2), 
                                                v.transpose(1, 2), 
                                                num_heads=head_num, 
                                                input_layout="BNSD",  
                                                scale=scale, 
                                                softmax_lse_flag=True,
                                                pre_tokens=65535, 
                                                next_tokens=65535)
    return block_out.transpose(1, 2), block_lse.squeeze(dim=-1)


attn_mod.flash_attn_forward = npu_fia

for mod in list(sys.modules.values()):
    if not isinstance(mod, types.ModuleType):
        continue
    if hasattr(mod, 'flash_attn_forward'):
        if mod.flash_attn_forward is not npu_fia:
            mod.flash_attn_forward = npu_fia

if int(os.getenv('LOCAL_RANK', 0)) == 0:
    logger.info('Use monkey patch replace yunchang.kernels.attention')