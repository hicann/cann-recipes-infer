from typing import Tuple, Optional, Dict
import os
import time
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from module.blockwise_sparse.sparse_method import sparse_predictor_manager, sync_and_get_time
from ..modules.attention import attention
from ..modules.posemb_layers import apply_rotary_emb
from ..modules.mlp_layers import MLP
from ..modules.modulate_layers import modulate, apply_gate


def sparse_double_block_forward(
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
        
    current = sparse_predictor_manager.sparse_attn_mode.current
    sparse_time_step = sparse_predictor_manager.sparse_attn_mode.sparse_time_step
    if current["step"] not in sparse_time_step:
        start_time = sync_and_get_time()
        attn = attention(q, k, v, mode='flash', cu_seqlens_q=cu_seqlens_q, 
                        cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, 
                        max_seqlen_kv=max_seqlen_kv, batch_size=img.shape[0])
        exec_time = sync_and_get_time(start_time)
        logger.info(f"Full Double Attention time: {exec_time * 1000:.2f} ms")
    else:
        start_time = sync_and_get_time()
        attn = sparse_predictor_manager.sparse_attn_mode.attention(q, k, v, cu_seqlens_q=cu_seqlens_q, 
                        cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, 
                        max_seqlen_kv=max_seqlen_kv, batch_size=img.shape[0])
        exec_time = sync_and_get_time(start_time)
        logger.info(f"Sparse Double Attention time: {exec_time * 1000:.2f} ms")

    sparse_predictor_manager.sparse_attn_mode.update_layer_counter()
    img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

    # Calculate the img blocks.
    img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
    img = img + apply_gate(
        self.img_mlp(
            modulate(
                self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
            )
        ),
        gate=img_mod2_gate,
    )

    # Calculate the txt blocks.
    txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
    txt = txt + apply_gate(
        self.txt_mlp(
            modulate(
                self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
            )
        ),
        gate=txt_mod2_gate,
    )

    return img, txt


def sparse_single_block_forward(
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
    mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
    x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
    qkv, mlp = torch.split(
        self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
    )

    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

    # Apply QK-Norm if needed.
    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)

    # Apply RoPE if needed.
    if freqs_cis is not None:
        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        
        img_q, img_k = img_qq, img_kk
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

    # Compute attention.
    
    current = sparse_predictor_manager.sparse_attn_mode.current
    sparse_time_step = sparse_predictor_manager.sparse_attn_mode.sparse_time_step
    if current["step"] not in sparse_time_step:
        start_time = sync_and_get_time()
        attn = attention(q, k, v, mode='flash', cu_seqlens_q=cu_seqlens_q, 
                        cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, 
                        max_seqlen_kv=max_seqlen_kv, batch_size=x.shape[0])
        exec_time = sync_and_get_time(start_time)
        logger.info(f"Full Single Attention time: {exec_time * 1000:.2f} ms")
    else:
        start_time = sync_and_get_time()
        attn = sparse_predictor_manager.sparse_attn_mode.attention(q, k, v, cu_seqlens_q=cu_seqlens_q, 
                        cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, 
                        max_seqlen_kv=max_seqlen_kv, batch_size=x.shape[0])
        exec_time = sync_and_get_time(start_time)
        logger.info(f"Sparse Single Attention time: {exec_time * 1000:.2f} ms")

    sparse_predictor_manager.sparse_attn_mode.update_layer_counter()
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    return x + apply_gate(output, gate=mod_gate)
        