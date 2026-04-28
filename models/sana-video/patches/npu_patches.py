# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

_npu_available = torch.npu.is_available() and torch_npu.__version__ >= "2.6.0"


def patch_rms_norm() -> None:
    norms = importlib.import_module("diffusion.model.norms")
    original_forward = norms.RMSNorm.forward

    def forward(self, x):
        if _npu_available:
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        return original_forward(self, x)

    norms.RMSNorm.forward = forward


def patch_conv1x1_matmul() -> None:
    basic_modules = importlib.import_module("diffusion.model.nets.basic_modules")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        if self.conv.kernel_size == (1, 1):
            batch_size, in_channels, height, width = x.shape
            out_channels, *_ = self.conv.weight.shape
            x_trans = x.view(batch_size, in_channels, -1).permute(0, 2, 1).contiguous()
            kernel_trans = self.conv.weight.view(out_channels, in_channels)
            x = x_trans @ kernel_trans.T
            if self.conv.bias is not None:
                x += self.conv.bias
            x = x.permute(0, 2, 1).contiguous().view(batch_size, out_channels, height, width)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    basic_modules.ConvLayer.forward = forward


def patch_temporal_conv_swap() -> None:
    basic_modules = importlib.import_module("diffusion.model.nets.basic_modules")
    original_init = basic_modules.GLUMBConvTemp.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        t_kernel_size = self.t_conv.kernel_size[0]
        t_padding = self.t_conv.padding[0]
        out_feature = self.t_conv.in_channels
        self.t_conv_swap = nn.Conv2d(
            out_feature,
            out_feature,
            kernel_size=(1, t_kernel_size),
            stride=1,
            padding=(0, t_padding),
            bias=False,
        )
        nn.init.zeros_(self.t_conv_swap.weight)

    def forward(self, x: torch.Tensor, hw=None, **kwargs) -> torch.Tensor:
        if hw is None:
            hw = kwargs.pop("HW", None)
        batch_size, num_tokens, channels = x.shape

        if hw is None or len(hw) != 3:
            raise ValueError("HW must be a tuple of (T, H, W)")
        time_steps, height, width = hw
        x = x.reshape(batch_size * time_steps, height, width, channels).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        x = self.point_conv(x)

        x_reshaped = x.view(batch_size, time_steps, channels, height * width).permute(0, 2, 1, 3)
        x_out = x_reshaped + self.t_conv_swap(x_reshaped.transpose(-1, -2)).transpose(-1, -2)

        x_out = x_out.permute(0, 2, 3, 1).reshape(batch_size, num_tokens, channels)
        return x_out

    basic_modules.GLUMBConvTemp.__init__ = __init__
    basic_modules.GLUMBConvTemp.forward = forward


def patch_fusion_attention() -> None:
    sana_blocks = importlib.import_module("diffusion.model.nets.sana_blocks")

    def forward(self, x, cond, mask=None):
        batch_size, num_tokens, channels = x.shape
        first_dim = batch_size

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(first_dim, -1, 2, channels)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(q.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, num_tokens, 1)

        if _npu_available:
            bool_mask = None if mask is None else mask < -1000
            x = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                q.shape[1],
                input_layout="BNSD",
                pse=None,
                atten_mask=bool_mask,
                scale=1.0 / math.sqrt(q.shape[-1]),
                pre_tockens=2147483647,
                next_tockens=2147483647,
                keep_prob=1,
            )[0]
        else:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    sana_blocks.MultiHeadCrossAttention.forward = forward


def patch_rotary_mul() -> None:
    sana_blocks = importlib.import_module("diffusion.model.nets.sana_blocks")

    def forward(self, x: torch.Tensor, mask=None, hw=None, rotary_emb=None, **kwargs) -> torch.Tensor:
        if hw is None:
            hw = kwargs.pop("HW", None)
        batch_size, num_tokens, channels = x.shape

        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, channels)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = q.reshape(batch_size, channels // self.dim, self.dim, num_tokens)
        k = k.reshape(batch_size, channels // self.dim, self.dim, num_tokens)
        v = v.reshape(batch_size, channels // self.dim, self.dim, num_tokens)

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
            if _npu_available:
                cos = freqs.real.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
                sin = freqs.imag.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
                x_out = torch_npu.npu_rotary_mul(hidden_states.permute(0, 1, 3, 2), cos, sin, "interleave")
                return x_out.permute(0, 1, 3, 2)

            x_rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
            x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
            return x_out.type_as(hidden_states)

        q_rotated = apply_rotary_emb(q, rotary_emb)
        k_rotated = apply_rotary_emb(k, rotary_emb)

        if self.qkv_store_buffer is not None:
            self.qkv_store_buffer["q"] = q_rotated.permute(0, 3, 1, 2)[0].cpu()
            self.qkv_store_buffer["k"] = k_rotated.permute(0, 3, 1, 2)[0].cpu()
            self.qkv_store_buffer["v"] = v.permute(0, 3, 1, 2)[0].cpu()

        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q_rotated, k_rotated, v = q_rotated.float(), k_rotated.float(), v.float()

        z = 1 / (k.sum(dim=-1, keepdim=True).transpose(-2, -1) @ q + self.eps)
        vk = torch.matmul(v, k_rotated.transpose(-1, -2))
        out = torch.matmul(vk, q_rotated)

        out = (out * z).to(dtype)
        out = out.view(batch_size, channels, num_tokens).permute(0, 2, 1).contiguous()
        out = self.proj(out)
        return out

    sana_blocks.LiteLAReLURope.forward = forward


def patch_local_text_encoder_path() -> None:
    builder = importlib.import_module("diffusion.model.builder")
    original_get_tokenizer_and_text_encoder = builder.get_tokenizer_and_text_encoder

    def get_tokenizer_and_text_encoder(name="T5", device="cuda"):
        if not os.path.exists(name):
            return original_get_tokenizer_and_text_encoder(name=name, device=device)

        tokenizer = builder.AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = "right"
        text_encoder = (
            builder.AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
        return tokenizer, text_encoder

    builder.get_tokenizer_and_text_encoder = get_tokenizer_and_text_encoder


def apply_npu_optimization_patches() -> None:
    patch_conv1x1_matmul()
    patch_temporal_conv_swap()
    patch_rms_norm()
    patch_fusion_attention()
    patch_rotary_mul()
    patch_local_text_encoder_path()
