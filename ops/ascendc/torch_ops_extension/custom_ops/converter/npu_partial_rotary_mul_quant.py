# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Any
import torch
import torch_npu
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, DataType
from torchair.ge import attr


# 为 torch.compile 场景注册 converter
@register_fx_node_ge_converter(torch.ops.custom.partial_rotary_mul_quant.default)
def convert_partial_rotary_mul_quant(
    x: Tensor,
    r1: Tensor,
    r2: Tensor,
    *,
    rotary_mode: str = "half",
    partial_slice: list = None,
    scale: float = 1.0,
    meta_outputs: Any = None):
    if partial_slice is None:
        partial_slice = [0, 0]
    rotary_mode_map = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
    mode_val = rotary_mode_map.get(rotary_mode, 0)
    return torchair.ge.custom_op(
        "PartialRotaryMulQuant",
        inputs={
            "x": x,
            "cos": r1,
            "sin": r2,
        },
        attrs={
            "mode": attr.Int(mode_val),
            "partial_slice": attr.ListInt(partial_slice),
            "scale": attr.Float(scale),
        },
        outputs=['y']
    )