# coding=utf-8
# Shared building blocks for GLM-5.3: RMSNorm variants and the mHC
# (Manifold-Constrained Hyper-Connections) stream ops.
# Adapted from
# https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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
from torch import nn

import torch_npu
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import custom_ops


class Glm53RMSNorm(nn.Module):
    """
    RMSNorm backed by npu_rms_norm. GLM-5.3 mHC layers use the plain
    (non-fused-residual) form: the residual path is owned by hc_pre/hc_post,
    so unlike glm_5_2 there is no add_rms_norm variant on the main layers.
    The MTP layer (no mHC) uses the fused (hidden, residual) form.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
        self.variance_epsilon = eps

    def forward(self, hidden_states, *args):
        if len(args) == 0:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        if len(args) == 1 and args[0] is None:  # first layer of a fused-residual stack
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return result, hidden_states
        if len(args) == 1:  # fused residual-add + norm (MTP layer path)
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(
                residual, hidden_states, self.weight, self.variance_epsilon)
            return y, x
        raise NotImplementedError(
            f"unsupported Glm53RMSNorm arity: {len(args) + 1}")


ALL_LAYERNORM_LAYERS.append(Glm53RMSNorm)


class Glm53RMSNormGated(nn.Module):
    """
    Gated RMSNorm used by the KDA output path (checkpoint name: self_attn.o_norm).

    Matches the HF reference `Glm5NextTextRMSNormGated` exactly:
      - strict FP32 normalization (weight is NOT downcast before the multiply);
      - sigmoid gate applied in FP32;
      - result cast back to the input dtype.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ============================================================================
# mHC (Manifold-Constrained Hyper-Connections)
# ----------------------------------------------------------------------------
# Hidden streams are [..., hc_mult, D] throughout the model.
#   hc_pre : streams -> (collapsed [..., D], post [..., hc], comb [..., hc, hc])
#   hc_post: out[..., h, :] = post[h] * x + sum_j comb[j, h] * residual[..., j, :]
# ============================================================================

def hc_pre(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    """
    x: [..., hc, D] streams (packed TND [T, hc, D] or batched [B, S, hc, D])
    -> (collapsed [..., D] in x.dtype, post [..., hc] fp32,
    comb [..., hc, hc] fp32).
    """
    # the AscendC op consumes the [B, S, hc, D] layout; lift packed TND
    squeeze = x.dim() == 3
    x_op = x.unsqueeze(0) if squeeze else x
    y, post, comb = torch.ops.custom.npu_hc_pre(
        x_op, hc_fn, hc_scale, hc_base, hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps)
    if squeeze:
        y, post, comb = y.squeeze(0), post.squeeze(0), comb.squeeze(0)
    return y, post, comb


def hc_post(x, residual, post, comb):
    """
    x: [..., D] sublayer output; residual: [..., hc, D] streams;
    post: [..., hc]; comb: [..., hc, hc]. Returns new streams [..., hc, D].

    out[..., h, :] = post[h] * x + sum_j comb[j, h] * residual[..., j, :]
    (comb is consumed transposed, matching HF's matmul(comb.T, residual)).
    """
    if x.dim() == 2:  # packed TND
        out = torch.ops.custom.npu_hc_post(
            x.unsqueeze(0), residual.unsqueeze(0),
            post.unsqueeze(0), comb.unsqueeze(0))
        return out.squeeze(0)
    return torch.ops.custom.npu_hc_post(x, residual, post, comb)


def hc_head_mean(x):
    """
    Final GLM-5.3 stream collapse: unweighted mean over the hc dim
    ([..., hc, D] -> [..., D]). (DeepSeek-V4 uses a learned weighted head
    instead; GLM-5.3 does not.)
    """
    return x.mean(dim=-2)
