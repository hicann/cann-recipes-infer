# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
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

from dataclasses import dataclass

import torch
import custom_ops
from ..registry import register_op_impl


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    # x in shape [T, H], pre_mix in shape [T, hc_mult]
    pre_mix = x.new_zeros(x.size(0), hc_mult, dtype=torch.float32)
    pre_mix[:, 0] = 1.0
    return pre_mix


def hc_pre_mix(x: torch.Tensor, pre_mix: torch.Tensor):
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1)
    return y.to(x.dtype)


@dataclass(frozen=True)
class _HCPreParams:
    """Parameters shared by the hc_pre preprocessing implementations."""

    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    hc_mult: int
    hc_sinkhorn_iters: int
    norm_eps: float
    hc_eps: float


@register_op_impl(op_type="hc_pre", func_key="hc_pre_ascendc")
def hc_pre_ascendc(x, pre_mix, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    # x: [T, hc_mult, H], pre_mix: [T, hc_mult], y: [T, H]
    y, post, comb, pre = torch.ops.custom.npu_hc_pre_v2(
        x=x,
        hc_fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        pre_mix=pre_mix,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps
    )
    return y, post, comb, pre


@register_op_impl(op_type="hc_post", func_key="hc_post_ascendc")
def hc_post_ascendc(x, residual, post, comb, is_prefill=False):
    y = torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)
    return y
