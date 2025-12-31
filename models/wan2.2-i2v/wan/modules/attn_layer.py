# coding=utf-8
# Adapted from
# https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/modules/attn_layer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError as e: 
    raise ImportError("Please install yunchang 0.6.0 or later") from e

from module.unified_sp.core import UnifiedSPAttention
from ..distributed.parallel_mgr import get_sp_group

logger = logging.getLogger(__name__)


class xFuserLongContextAttention(LongContextAttention):
    
    def __init__(
        self,
        args: Any,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type=attn_type,
        )
        
        self.args = args
        self.video_size = ['480*832', '832*480', '480*720', '720*480', '1280*720']
        self.use_all_head = args.size in self.video_size if hasattr(args, 'size') else False
        
        self.sp_attn = UnifiedSPAttention(
            ulysses_group=get_sp_group().ulysses_group,
            ring_group=get_sp_group().ring_group,
            use_ring_overlap=True,
        )
    
    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_lens: int = None,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None,
        t_idx=-1,
    ) -> Tensor:
        
        output = self.sp_attn.forward(
            q=query,
            k=key,
            v=value,
            causal=causal,
            softmax_scale=softmax_scale or scale,
            seq_lens=seq_lens,
            per_head_compute=not self.use_all_head,
            joint_tensor_query=joint_tensor_query,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        
        return output