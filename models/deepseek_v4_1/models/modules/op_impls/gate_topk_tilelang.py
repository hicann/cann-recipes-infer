# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from tile_kernels.moe import moe_topk_gate

from ..registry import register_op_impl
from .gate_topk import gate_topk_ascendc


@register_op_impl(op_type="gate_topk", func_key="gate_topk_tilelang")
def gate_topk_tilelang(module, logits, input_ids, image_mask=None, is_prefill=False):
    if not is_prefill:
        return gate_topk_ascendc(module, logits, input_ids, image_mask)

    bias_vl = getattr(module.gate, "bias_vl", None)
    use_vision_bias = image_mask is not None and bias_vl is not None
    ep_rank = (
        module.comm_manager.get_rank("moe_ep_group") if module.moe_ep_size > 1 else 0
    )
    topk_idx, topk_weight = moe_topk_gate(
        logits.float().contiguous(),
        num_topk=module.top_k,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=module.routed_scaling_factor,
        ep_rank=ep_rank,
        bias=module.gate.e_score_correction_bias.float().contiguous(),
        image_bias=bias_vl.float().contiguous() if use_vision_bias else None,
        image_token_mask=(
            image_mask.view(-1).bool().contiguous() if use_vision_bias else None
        ),
    )
    return topk_idx, topk_weight, None
