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

import torch
from ..registry import register_op_impl


@register_op_impl(op_type="gate_topk")
def gate_topk_ascendc(module, logits, input_ids, image_mask=None, is_prefill=False):
    if module.topk_method != "noaux_tc":
        raise NotImplementedError(
            f"unsupported TopK function for MoE gating: {module.topk_method}"
        )

    scoring_func_mapping = {
        "softmax": 0,
        "sigmoid": 1,
        "sqrtsoftplus": 2,
    }
    import custom_ops

    bias_vl = getattr(module.gate, "bias_vl", None)
    use_vision_bias = image_mask is not None and bias_vl is not None
    topk_weight, topk_idx, _ = torch.ops.custom.npu_moe_gating_top_k(
        logits.float(),
        k=module.top_k,
        bias=module.gate.e_score_correction_bias,
        additional_bias=bias_vl if use_vision_bias else None,
        additional_token_mask=image_mask.view(-1) if use_vision_bias else None,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=0,
        norm_type=scoring_func_mapping[module.scoring_func],
        routed_scaling_factor=module.routed_scaling_factor,
        eps=float(1e-20),
        out_flag=False,
    )
    return topk_idx, topk_weight, None
