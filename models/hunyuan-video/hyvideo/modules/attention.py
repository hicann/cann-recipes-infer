# coding=utf-8
# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
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
import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from module.fa_quant import npu_fp8_attn

MEMORY_LAYOUT = {
    "TND": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]), 
        lambda x: x,
    ),
    "BNSD": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "BSND": (
        lambda x: x,
        lambda x: x,
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    if mode == "torch":
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
    elif mode == "flash":
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BNSD"]
    elif mode == 'perblock_fp8':
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BSND"]
    elif mode == "vanilla":
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT["BSND"]
    
    b, s, n, d = q.shape
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        if cu_seqlens_q is None:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
        else:
            attn1 = F.scaled_dot_product_attention(
                q[:, :, :cu_seqlens_q[1]],
                k[:, :, :cu_seqlens_kv[1]],
                v[:, :, :cu_seqlens_kv[1]],
                attn_mask=attn_mask,
                dropout_p=drop_rate,
                is_causal=causal
            )
            attn2 = F.scaled_dot_product_attention(
                q[:, :, cu_seqlens_q[1]:],
                k[:, :, cu_seqlens_kv[1]:],
                v[:, :, cu_seqlens_kv[1]:],
                attn_mask=None,
                dropout_p=drop_rate,
                is_causal=False
            )
            x = torch.cat([attn1, attn2], dim=2)
    elif mode == "flash":
        scale = 1.0 / math.sqrt(d)
        if cu_seqlens_q is None:
            x = torch_npu.npu_fused_infer_attention_score(
                q, k, v,
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
        else:
            attn1 = torch_npu.npu_fused_infer_attention_score(
                q[:, :, :cu_seqlens_q[1], :],
                k[:, :, :cu_seqlens_kv[1], :],
                v[:, :, :cu_seqlens_kv[1], :],
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            attn2 = torch_npu.npu_fused_infer_attention_score(
                q[:, :, cu_seqlens_q[1]:, :],
                k[:, :, cu_seqlens_kv[1]:, :],
                v[:, :, cu_seqlens_kv[1]:, :],
                num_heads=n,
                input_layout="BNSD",
                scale=scale,
            )[0]
            x = torch.cat([attn1, attn2], dim=2)
    elif mode == "perblock_fp8":
        scale = 1.0 / math.sqrt(d)
        attn1 = npu_fp8_attn(
            q[:, :cu_seqlens_q[1], ...],
            k[:, :cu_seqlens_kv[1], ...],
            v[:, :cu_seqlens_kv[1], ...],
            dst_type=torch.float8_e4m3fn,
            softmax_scale=scale
        )

        attn2 = torch_npu.npu_fused_infer_attention_score(
            q[:, cu_seqlens_q[1]:, ...],
            k[:, cu_seqlens_kv[1]:, ...],
            v[:, cu_seqlens_kv[1]:, ...],
            num_heads=n,
            input_layout="BSND",
            scale=scale,
        )[0]

        x = torch.cat([attn1, attn2], dim=1)


    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(	 
    hybrid_seq_parallel_attn,	 
    q,	 
    k, 
    v, 
    img_q_len,	 
    img_kv_len,	 
    cu_seqlens_q,	 
    cu_seqlens_kv	 
):	 
    b, s, n, d = q.shape	 

    attn1 = hybrid_seq_parallel_attn(	 
        q=q[:, :img_q_len, :, :],
        k=k[:, :img_kv_len, :, :],
        v=v[:, :img_kv_len, :, :],
        dropout_p=0.0, 
        causal=False,	 
        joint_tensor_query=q[:, img_q_len:cu_seqlens_q[1]],	 
        joint_tensor_key=k[:, img_kv_len:cu_seqlens_kv[1]],	 
        joint_tensor_value=v[:, img_kv_len:cu_seqlens_kv[1]],	 
        joint_strategy="rear",	 
    )	 

    attn2 = torch_npu.npu_fused_infer_attention_score(	 
        q[:, cu_seqlens_q[1]:], 	 
        k[:, cu_seqlens_kv[1]:], 	 
        v[:, cu_seqlens_kv[1]:],	 
        num_heads=n,	 
        input_layout="BSND",	 
        scale=q.shape[-1] ** (-0.5),	 
    )[0]	 

    attn = torch.cat([attn1, attn2], dim=1)
    attn = attn.reshape(b, s, -1)
    
    return attn


def parallel_sparse_attention(
    hybrid_seq_parallel_attn,
    block_args: dict,
):
    q, k, v = block_args["q"], block_args["k"], block_args["v"]
    img_q_len, img_kv_len = block_args["img_q_len"], block_args["img_kv_len"]
    cu_seqlens_q, cu_seqlens_kv = block_args["cu_seqlens_q"], block_args["cu_seqlens_kv"]
    joint_q_local_bnsd = block_args.get("joint_q_local_bnsd")
    b, s, n, d = q.shape
    from module.blockwise_sparse.sparse_method import sparse_predictor_manager

    q_img, k_img, v_img = q[:, :img_q_len, :, :], k[:, :img_kv_len, :, :], v[:, :img_kv_len, :, :]
    txt_q = (
        joint_q_local_bnsd
        if joint_q_local_bnsd is not None
        else q[:, img_q_len:cu_seqlens_q[1], :, :]
    )
    txt_k, txt_v = k[:, img_kv_len:cu_seqlens_kv[1], :, :], v[:, img_kv_len:cu_seqlens_kv[1], :, :]

    sparse_mode = getattr(sparse_predictor_manager, "sparse_attn_mode", None)
    sp_attn = getattr(hybrid_seq_parallel_attn, "__self__", None)
    attn_prefix = sparse_mode.forward_ulysses_sparse(
        runtime_attn=sp_attn,
        block_args={
            "q_img_local": q_img,
            "k_img_local": k_img,
            "v_img_local": v_img,
            "txt_q": txt_q,
            "txt_k": txt_k,
            "txt_v": txt_v,
        },
        softmax_scale=q.shape[-1] ** (-0.5),
    )

    attn = attn_prefix
    if int(attn_prefix.shape[1]) < s:
        attn2 = torch_npu.npu_fused_infer_attention_score(
            q[:, cu_seqlens_q[1]:],
            k[:, cu_seqlens_kv[1]:],
            v[:, cu_seqlens_kv[1]:],
            num_heads=n,
            input_layout="BSND",
            scale=q.shape[-1] ** (-0.5),
        )[0]
        attn = torch.cat([attn_prefix, attn2], dim=1)
    attn = attn.reshape(b, s, -1)
    return attn
