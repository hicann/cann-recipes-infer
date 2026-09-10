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
import torch_npu
from executor.core.config import PlatformVersion
from ..registry import register_op_impl


def _routing_scores(module, logits):
    if module.scoring_func == "sigmoid":
        return logits.sigmoid()
    if module.scoring_func == "softmax":
        return logits.softmax(dim=-1, dtype=torch.float32)
    if module.scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    raise NotImplementedError(
        f"insupportable scoring function for MoE gating: {module.scoring_func}"
    )


def _apply_vision_routing(module, scores, topk_idx, topk_weight, image_mask):
    if image_mask is None or getattr(module.gate, "bias_vl", None) is None:
        return topk_idx, topk_weight
    image_mask = image_mask.view(-1)
    vision_scores = scores[image_mask]
    _, vision_idx = torch.topk(
        vision_scores + module.gate.bias_vl,
        k=module.top_k,
        dim=-1,
        sorted=False,
    )
    vision_weight = vision_scores.gather(1, vision_idx)
    if module.top_k > 1 and module.norm_topk_prob:
        vision_weight = vision_weight / vision_weight.sum(dim=-1, keepdim=True)
    vision_weight = vision_weight * module.routed_scaling_factor
    vision_idx = vision_idx.to(topk_idx.dtype)
    topk_idx = topk_idx.clone()
    topk_weight = topk_weight.clone()
    topk_idx[image_mask] = vision_idx
    topk_weight[image_mask] = vision_weight
    return topk_idx, topk_weight


@register_op_impl(op_type="gate_topk")
def gate_topk_ascendc(module, logits, input_ids, image_mask=None):
    # select top-k experts
    if module.topk_method == "greedy":
        scores = _routing_scores(module, logits)
        topk_weight, topk_idx = torch.topk(
            scores, k=module.top_k, dim=-1, sorted=False
        )
    elif module.topk_method == "noaux_tc":
        # A3 gating_top_k supports norm_type 0/1 only; sqrtsoftplus requires mode 2.
        if module.use_native_gate_topk:
            return gate_topk_native(module, logits, input_ids, image_mask)
        scoring_func_mapping = {
            "softmax": 0,
            "sigmoid": 1,
            "sqrtsoftplus": 2
        }
        op = torch_npu.npu_moe_gating_top_k
        if module.platform_version == PlatformVersion.A3:
            import custom_ops  # Registers the A3 custom operator binding.
            op = torch.ops.custom.npu_moe_gating_top_k
        topk_weight, topk_idx, _ = op(
            logits,
            k=module.top_k,
            bias=module.gate.e_score_correction_bias,
            k_group=1,
            group_count=1,
            group_select_mode=1,  # 0: max value in group; 1: sum of top 2 expert scores in group
            renorm=0,  # 0: softmax->topk; 1: topk->softmax
            norm_type=scoring_func_mapping[module.scoring_func],  # 0: softmax; 1: sigmoid; 2: sqrtsoftplus
            routed_scaling_factor=module.routed_scaling_factor,
            eps=float(1e-20),
            out_flag=False
        )
        if image_mask is None:
            return topk_idx, topk_weight, None
        scores = _routing_scores(module, logits)
        topk_idx, topk_weight = _apply_vision_routing(
            module, scores, topk_idx, topk_weight, image_mask
        )
        return topk_idx, topk_weight, None
    else:
        raise NotImplementedError(
            f"insupportable TopK function for MoE gating: {module.topk_method}"
        )

    # norm gate to sum 1
    if module.top_k > 1 and module.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * module.routed_scaling_factor # must multiply the scaling factor
    topk_idx, topk_weight = _apply_vision_routing(
        module, scores, topk_idx, topk_weight, image_mask
    )
    return topk_idx, topk_weight, None


@register_op_impl(op_type="gate_topk")
def gate_topk_native(module, logits, input_ids, image_mask=None):
    token_num = logits.shape[0]
    scores = _routing_scores(module, logits)

    # select top-k experts
    if module.topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(
            scores, k=module.top_k, dim=-1, sorted=False
        )
    elif module.topk_method == "noaux_tc":
        tmp_scores = scores.view(token_num, -1) + module.gate.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(
            tmp_scores, k=module.top_k, dim=-1, sorted=False
        )
        topk_weight = scores.gather(1, topk_idx)
    else:
        raise NotImplementedError(
            f"insupportable TopK function for MoE gating: {module.topk_method}"
        )

    # norm gate to sum 1
    if module.top_k > 1 and module.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight / denominator
    topk_weight = topk_weight * module.routed_scaling_factor # must multiply the scaling factor
    topk_idx, topk_weight = _apply_vision_routing(
        module, scores, topk_idx, topk_weight, image_mask
    )
    return topk_idx, topk_weight, None
