# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import torch
import torch.nn.functional as F
from ..registry import register_op_impl

def hc_split_sinkhorn_torch(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6):
    # mixes: [b, s, mix_hc], hc_scale: [3], hc_base: [mix_hc]
    # mix_hc = (hc + 2) * hc
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


# hc_pre currently support Native, AscendC and PyPTO version
@register_op_impl(op_type="hc_pre", func_key="hc_pre_ascendc")
def hc_pre_ascendc(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    y, post, comb = torch.ops.custom.npu_hc_pre(
                x, hc_fn, hc_scale, hc_base, hc_mult=hc_mult,
                hc_sinkhorn_iters=hc_sinkhorn_iters,
                norm_eps=norm_eps,
                hc_eps=hc_eps)
    return y, post, comb


@register_op_impl(op_type="hc_pre", func_key="hc_pre_pypto_a3")
def hc_pre_pypto_a3(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    # x: [b * s, hc, d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b * s, hc, d]
    from ops.pypto_python.impl.hc_pre_pypto import hc_pre_pypto
    b, s, hc, d = x.shape
    x = x.reshape(b * s, hc, d)
    y, post, comb = hc_pre_pypto(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    y = y.reshape(b, s, d)
    post = post.reshape(b, s, hc)
    comb = comb.reshape(b, s, hc, hc)
    return y, post, comb


@register_op_impl(op_type="hc_pre")
def hc_pre_native(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt

    pre, post, comb = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    y = y.to(dtype)
    return y, post, comb


# hc_post currently support Native and AscendC version
@register_op_impl(op_type="hc_post", func_key="hc_post_ascendc")
def hc_post_ascendc(x, residual, post, comb):
    y = torch.ops.custom.npu_hc_post(x, residual, post, comb)
    return y


@register_op_impl(op_type="hc_post")
def hc_post_native(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    y = y.type_as(x)
    return y