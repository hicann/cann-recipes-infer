# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图

# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_swiglu_group_quant.default)
def convert_npu_swiglu_group_quant(
    x: Tensor,
    *,
    weight: Optional[Tensor] = None,
    group_index: Optional[Tensor] = None,
    dst_type: int = 24,
    quant_mode: int = 0,
    block_size: int = 0,
    round_scale: bool = False,
    clamp_limit: Optional[float] = None,
    output_origin: bool = False,
    meta_outputs: Any = None):
    dst_type_code = 35
    if dst_type == 23:
        dst_type_code = 35
    elif dst_type == 24:
        dst_type_code = 36
    elif dst_type == 5:
        dst_type_code = 1
    elif dst_type == 15:
        dst_type_code = 27

    actual_clamp_limit = clamp_limit if clamp_limit is not None else -float("inf")

    return torchair.ge.custom_op(
        "SwigluGroupQuant",
        inputs={"x": x,
                "weight": weight,
                "group_index": group_index,
                },
        attrs={
               "dst_type": attr.Int(dst_type_code),
               "quant_mode": attr.Int(quant_mode),
               "block_size": attr.Int(block_size),
               "round_scale": attr.Bool(round_scale),
               "output_origin": attr.Bool(output_origin),
               "clamp_limit": attr.Float(actual_clamp_limit),
               },
        outputs=['y', 'scale_out', 'y_origin']
    )
