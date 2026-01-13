# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# https://github.com/ali-vilab/TeaCache,
# https://github.com/chengzeyi/ParaAttention.
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
# Copyright (C) 2025 ali-vilab.
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
from typing import Any, List, Tuple, Optional, Union, Dict
import os
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import numpy as np

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from ..modules.activation_layers import get_activation_layer
from ..modules.norm_layers import get_norm_layer
from ..modules.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from ..modules.attention import attention, parallel_attention, get_cu_seqlens
from ..modules.posemb_layers import apply_rotary_emb
from ..modules.mlp_layers import MLP, MLPEmbedder, FinalLayer
from ..modules.modulate_layers import ModulateDiT, modulate, apply_gate
from ..modules.token_refiner import SingleTokenRefiner
from module.dit_cache_step.cache_step import cache_manager


def first_block_forward(
    self,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    freqs_cis: tuple = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        img_mod1_shift,
        img_mod1_scale,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ) = self.img_mod(vec).chunk(6, dim=-1)
    (
        txt_mod1_shift,
        txt_mod1_scale,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ) = self.txt_mod(vec).chunk(6, dim=-1)

    # Prepare image for attention.
    img_modulated = self.img_norm1(img)
    img_modulated = modulate(
        img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
    )
    enable_separate_cfg = cache_manager.enable_separate_cfg
    if not hasattr(self, 'cfg_step'):
        self.cfg_step = 0
    is_cond = True
    if enable_separate_cfg:
        is_cond = (self.cfg_step == 0)
        self.cfg_step = 1 - self.cfg_step
    # TeaCache
    if cache_manager.cache_step.cache_name == "TeaCache":
        judge_input = img_modulated
        args = {
            "latent": img,
            "judge_input": judge_input,
            "is_cond": is_cond
        }
        should_calc, img = cache_manager.cache_step.pre_cache_process(args)

        if not should_calc:
            cache_manager.cache_step.should_skip = True
            cache_manager.cache_step.last_is_cond = is_cond
            cache_manager.cache_step.post_cache_update(img)
            return img, txt


    img_qkv = self.img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(
        img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
    )
    # Apply QK-Norm if needed
    img_q = self.img_attn_q_norm(img_q).to(img_v.dtype)
    img_k = self.img_attn_k_norm(img_k).to(img_v.dtype)

    # Apply RoPE if needed.
    if freqs_cis is not None:
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        if img_qq.shape != img_q.shape or img_kk.shape != img_k.shape:
            raise ValueError(
                f"Shape mismatch: img_qq: {img_qq.shape}, img_q: {img_q.shape}, "
                f"img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
        img_q, img_k = img_qq, img_kk

    # Prepare txt for attention.
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = modulate(
        txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
    )
    txt_qkv = self.txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(
        txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
    )
    # Apply QK-Norm if needed.
    txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
    txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

    # Run actual attention.
    q = torch.cat((img_q, txt_q), dim=1)
    k = torch.cat((img_k, txt_k), dim=1)
    v = torch.cat((img_v, txt_v), dim=1)
    if cu_seqlens_q.shape[0] != 2 * img.shape[0] + 1:
        raise ValueError(
            f"Sequence length mismatch: cu_seqlens_q.shape:{cu_seqlens_q.shape}, "
            f"img.shape[0]:{img.shape[0]}, expected length:{2 * img.shape[0] + 1}"
        )
    
    # attention computation start
    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q,
            k,
            v,
            mode='flash',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q,
            k,
            v,
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv
        )
        
    # attention computation end

    img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

    # Calculate the img bloks.
    img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
    img = img + apply_gate(
        self.img_mlp(
            modulate(
                self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
            )
        ),
        gate=img_mod2_gate,
    )

    # Calculate the txt bloks.
    txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
    txt = txt + apply_gate(
        self.txt_mlp(
            modulate(
                self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
            )
        ),
        gate=txt_mod2_gate,
    )
    if cache_manager.cache_step.cache_name == "FBCache":
        args = {
            "latent": img,
            "judge_input": img.clone(),
            "is_cond": is_cond
        }
        should_calc, img = cache_manager.cache_step.pre_cache_process(args)

    return img, txt