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

import torch
import torch_npu

from cann_ops_transformer.ops.ds41 import indexer_prologue_qw
from ..registry import register_op_impl

MXFP8_SCALE_PAIR_GROUPS = 2
MXFP8_SCALE_PAIR_SIZE = 64


@register_op_impl(op_type="indexer_prolog_qw")
def indexer_prolog_qw_native(module, x, qr, cos, sin):
    weights = module.weights_proj(x) * (
        module.softmax_scale * module.n_heads ** -0.5
    )
    q = module.wq_b(qr).view(-1, module.n_heads, module.head_dim)
    torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
        q.unsqueeze(2),
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=module.partial_slice,
    )
    q, q_scale = torch_npu.npu_dynamic_mx_quant(
        q, dst_type=torch_npu.float4_e2m1fn_x2
    )
    return q.view(torch.uint8), q_scale, weights


@register_op_impl(op_type="indexer_prolog_qw", func_key="indexer_prolog_qw_ascendc")
def indexer_prolog_qw_ascendc(module, x, qr, cos, sin):
    qr, qr_scale = torch_npu.npu_dynamic_mx_quant(
        qr.view(-1, module.q_lora_rank),
        dst_type=torch.float8_e4m3fn,
    )
    qr = qr.view(torch.uint8)
    scale_pair_dim = (
        module.q_lora_rank + MXFP8_SCALE_PAIR_SIZE - 1
    ) // MXFP8_SCALE_PAIR_SIZE
    qr_scale = qr_scale.view(torch.uint8).view(
        qr.shape[0], scale_pair_dim, MXFP8_SCALE_PAIR_GROUPS,
    )
    wq_b_scale = module.wq_b.weight_scale.view(torch.uint8).view(
        module.n_heads * module.head_dim,
        scale_pair_dim,
        MXFP8_SCALE_PAIR_GROUPS,
    )
    return indexer_prologue_qw(
        x.view(-1, module.dim),
        qr,
        module.wq_b.weight.view(torch.uint8),
        module.weights_proj.weight,
        qr_scale,
        wq_b_scale,
        sin.view(-1, module.rope_head_dim),
        cos.view(-1, module.rope_head_dim),
        softmax_scale=module.softmax_scale * module.n_heads ** -0.5,
    )
