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

from module.dit_cache.cache_method import cache_manager
from ..modules.activation_layers import get_activation_layer
from ..modules.norm_layers import get_norm_layer
from ..modules.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from ..modules.attention import attention, parallel_attention, get_cu_seqlens
from ..modules.posemb_layers import apply_rotary_emb
from ..modules.mlp_layers import MLP, MLPEmbedder, FinalLayer
from ..modules.modulate_layers import ModulateDiT, modulate, apply_gate
from ..modules.token_refiner import SingleTokenRefiner


def _double_block_full_compute(
    self, 
    block_args: dict,
    cache_dic: dict, 
    current: dict
):
    img = block_args['img']
    txt = block_args['txt']
    freqs_cis = block_args['freqs_cis']

    # Image Attention
    current['module'] = 'img_attn'
    img_modulated = self.img_norm1(img)
    img_modulated = modulate(img_modulated, shift=block_args['img_mod1_shift'], scale=block_args['img_mod1_scale'])
    img_qkv = self.img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    img_q = self.img_attn_q_norm(img_q).to(img_v.dtype)
    img_k = self.img_attn_k_norm(img_k).to(img_v.dtype)
    
    if freqs_cis is not None:
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
    
    # Text Attention
    current['module'] = 'txt_attn'
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = modulate(txt_modulated, shift=block_args['txt_mod1_shift'], scale=block_args['txt_mod1_scale'])
    txt_qkv = self.txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    txt_q = self.txt_attn_q_norm(txt_q).to(txt_v.dtype)
    txt_k = self.txt_attn_k_norm(txt_k).to(txt_v.dtype)
    # Combined Attention
    q = torch.cat((img_q, txt_q), dim=1)
    k = torch.cat((img_k, txt_k), dim=1)
    v = torch.cat((img_v, txt_v), dim=1)

    # attention computation start
    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q,
            k,
            v,
            mode='flash',
            cu_seqlens_q=block_args['cu_seqlens_q'], 
            cu_seqlens_kv=block_args['cu_seqlens_kv'], 
            max_seqlen_q=block_args['max_seqlen_q'], 
            max_seqlen_kv=block_args['max_seqlen_kv'], 
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
            cu_seqlens_q=block_args['cu_seqlens_q'], 
            cu_seqlens_kv=block_args['cu_seqlens_kv'], 
        )
    
    img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]
    # Image Attn Update
    current['module'] = 'img_attn'
    img_attn_out = self.img_attn_proj(img_attn)
    img = img + apply_gate(img_attn_out, gate=block_args['img_mod1_gate'])
    cache_manager.cache_method.derivative_approximation(cache_dic, current, img_attn_out)

    current['module'] = 'img_mlp'
    img_mlp_in = modulate(self.img_norm2(img), shift=block_args['img_mod2_shift'], scale=block_args['img_mod2_scale'])
    img_mlp_out = self.img_mlp(img_mlp_in)
    img = img + apply_gate(img_mlp_out, gate=block_args['img_mod2_gate'])
    cache_manager.cache_method.derivative_approximation(cache_dic, current, img_mlp_out)

    # Text Attn Update
    current['module'] = 'txt_attn'
    txt_attn_out = self.txt_attn_proj(txt_attn)
    txt = txt + apply_gate(txt_attn_out, gate=block_args['txt_mod1_gate'])
    cache_manager.cache_method.derivative_approximation(cache_dic, current, txt_attn_out)

    # Text MLP
    current['module'] = 'txt_mlp'
    txt_mlp_in = modulate(self.txt_norm2(txt), shift=block_args['txt_mod2_shift'], scale=block_args['txt_mod2_scale'])
    txt_mlp_out = self.txt_mlp(txt_mlp_in)
    txt = txt + apply_gate(txt_mlp_out, gate=block_args['txt_mod2_gate'])
    cache_manager.cache_method.derivative_approximation(cache_dic, current, txt_mlp_out)
    
    return img, txt


def _double_block_taylor_compute(
    self,
    block_args: dict,
    cache_dic: dict, 
    current: dict
):
    img = block_args['img']
    txt = block_args['txt']

    # Taylor approximation for each module
    current['module'] = 'img_attn'
    img_attn_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    img = img + apply_gate(img_attn_out, gate=block_args['img_mod1_gate'])

    current['module'] = 'img_mlp'
    img_mlp_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    img = img + apply_gate(img_mlp_out, gate=block_args['img_mod2_gate'])

    current['module'] = 'txt_attn'
    txt_attn_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    txt = txt + apply_gate(txt_attn_out, gate=block_args['txt_mod1_gate'])

    current['module'] = 'txt_mlp'
    txt_mlp_out = cache_manager.cache_method.taylor_formula(cache_dic, current)
    txt = txt + apply_gate(txt_mlp_out, gate=block_args['txt_mod2_gate'])

    return img, txt


def double_block_forward(
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
    cache_dic = cache_manager.cache_method.cache_dic
    current = cache_manager.cache_method.current
    double_stream_layers = cache_manager.cache_method.double_stream_layers
    current_layer_idx = cache_manager.cache_method.layer_counter - 1
    current_layer_idx = current_layer_idx % double_stream_layers
    current.update({
        "stream": "double_stream",
        "layer": current_layer_idx,
        "step": cache_manager.cache_method.num_steps
    })
    cache_manager.cache_method.update_layer_counter()
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

    block_args = {
        'img': img,
        'txt': txt,
        'img_mod1_shift': img_mod1_shift,
        'img_mod1_scale': img_mod1_scale,
        'freqs_cis': freqs_cis,
        'txt_mod1_shift': txt_mod1_shift,
        'txt_mod1_scale': txt_mod1_scale,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_kv': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_kv': max_seqlen_kv,
        'img_mod1_gate': img_mod1_gate,
        'img_mod2_shift': img_mod2_shift,
        'img_mod2_scale': img_mod2_scale,
        'img_mod2_gate': img_mod2_gate,
        'txt_mod1_gate': txt_mod1_gate,
        'txt_mod2_shift': txt_mod2_shift,
        'txt_mod2_scale': txt_mod2_scale,
        'txt_mod2_gate': txt_mod2_gate
    }

    if current.get('type') == 'full':
        img, txt = _double_block_full_compute(
            self, block_args, cache_dic, current
        )

    elif current.get('type') == 'taylor_cache':
        img, txt = _double_block_taylor_compute(
            self, block_args, cache_dic, current
        )

    return img, txt


def _single_block_full_compute(
    self, 
    block_args: dict,
    cache_dic: dict, 
    current: dict
):
    x = block_args['x']
    txt_len = block_args['txt_len']
    freqs_cis = block_args['freqs_cis']

    current['module'] = 'total'
    x_mod = modulate(self.pre_norm(x), shift=block_args['mod_shift'], scale=block_args['mod_scale'])
    qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
    q = self.q_norm(q).to(v.dtype)
    k = self.k_norm(k).to(v.dtype)
    
    if freqs_cis is not None:
        img_q, txt_q = q[:, :-txt_len], q[:, -txt_len:]
        img_k, txt_k = k[:, :-txt_len], k[:, -txt_len:]
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

    # attention computation start
    if not self.hybrid_seq_parallel_attn:
        attn = attention(
            q,
            k,
            v,
            mode='flash',
            cu_seqlens_q=block_args['cu_seqlens_q'], 
            cu_seqlens_kv=block_args['cu_seqlens_kv'], 
            max_seqlen_q=block_args['max_seqlen_q'], 
            max_seqlen_kv=block_args['max_seqlen_kv'], 
            batch_size=k.shape[0],
        )
    else:
        attn = parallel_attention(
            self.hybrid_seq_parallel_attn,
            q,
            k,
            v,
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            cu_seqlens_q=block_args['cu_seqlens_q'], 
            cu_seqlens_kv=block_args['cu_seqlens_kv'], 
        )
    
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))
    cache_manager.cache_method.derivative_approximation(cache_dic, current, output)
    x = x + apply_gate(output, gate=block_args['mod_gate'])

    return x


def _single_block_taylor_compute(
    self, 
    block_args: dict,
    cache_dic: dict, 
    current: dict
):
    x = block_args['x']

    current['module'] = 'total'
    output = cache_manager.cache_method.taylor_formula(cache_dic, current)
    x = x + apply_gate(output, gate=block_args['mod_gate'])

    return x


def single_block_forward(
    self,
    x: torch.Tensor,
    vec: torch.Tensor,
    txt_len: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
) -> torch.Tensor:
    cache_dic = cache_manager.cache_method.cache_dic
    current = cache_manager.cache_method.current
    double_stream_layers = cache_manager.cache_method.double_stream_layers
    current_layer_idx = cache_manager.cache_method.layer_counter - 1
    single_layer_idx = (current_layer_idx - double_stream_layers) % cache_manager.cache_method.single_stream_layers
    current.update({
        "stream": "single_stream",
        "layer": single_layer_idx,
        "step": cache_manager.cache_method.num_steps
    })
    cache_manager.cache_method.update_layer_counter()
    mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
    block_args = {
        'x': x,
        'mod_shift': mod_shift,
        'mod_scale': mod_scale,
        'txt_len': txt_len,
        'mod_gate': mod_gate,
        'freqs_cis': freqs_cis,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_kv': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_kv': max_seqlen_kv
    }
    if current.get('type') == 'full':
        x = _single_block_full_compute(
            self, block_args, cache_dic, current
        )
        
    elif current.get('type') == 'taylor_cache':
        x = _single_block_taylor_compute(
        self, block_args, cache_dic, current
    )
        
    return x


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

    # TeaCache
    if cache_manager.cache_method.cache_name == "TeaCache":
        judge_input = img_modulated
        args = {
            "latent": img,
            "judge_input": judge_input
        }
        should_calc, img = cache_manager.cache_method.pre_cache_process(args)

        if not should_calc:
            cache_manager.cache_method.post_cache_update(img)
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
    if cache_manager.cache_method.cache_name == "FBCache":
        args = {
            "latent": img,
            "judge_input": img.clone()
        }
        should_calc, img = cache_manager.cache_method.pre_cache_process(args)

    return img, txt