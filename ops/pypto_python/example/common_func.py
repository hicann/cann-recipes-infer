#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """
import torch


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成, 与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(
            low=min_value, high=max_value, size=data_shape, dtype=dtype
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    """
    q: (t, n_q, rope_dim), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    """
    input_dtype = q.dtype
    q_new = q.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=1)  # [t, 1, rope_dim]
    sin = torch.unsqueeze(sin, dim=1)  # [t, 1, rope_dim]

    t, n, d = q_new.shape
    q_re = q_new.reshape(t, n, d // 2, 2).permute(0, 1, 3, 2).reshape(t, n, d)

    q_rotary = (
        rotate_half(q_re)
        .reshape((t, n, 2, d // 2))
        .permute(0, 1, 3, 2)
        .reshape((t, n, d))
    )

    # (t, n_q, rope_dim), (t, 1, rope_dim) = (t, n_q, rope_dim)
    q_embed = (q_new * cos) + (q_rotary * -sin)

    if input_dtype != torch.float32:
        q_embed = q_embed.to(input_dtype)

    return q_embed


def quant_golden(x: torch.Tensor):
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_int32 = torch.round(y_fp32).to(torch.int32)  # rint mode
    y_int8 = torch.trunc(y_int32.to(x_dtype)).to(torch.int8)
    scale_dequant = 1.0 / scale_quant  # fp32
    return y_int8, scale_dequant
