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
@register_fx_node_ge_converter(torch.ops.custom.npu_moe_init_routing_group_quant.default)
def convert_npu_moe_init_routing_group_quant(
        x: Tensor,
        expert_idx: Tensor,
        scale: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        active_num: int = -1,
        expert_capacity: int = -1,
        expert_num: int = -1,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 0,
        expert_tokens_num_flag: bool = False,
        quant_mode: int = -1,
        active_expert_range: List[int] = None,
        row_idx_type: int = 0,):
    return torchair.ge.custom_op(
        "MoeInitRoutingGroupQuant",
        inputs={"x": x,
                "expert_idx": expert_idx,
                "scale": scale,
                "offset": offset,
                },
        attrs={"active_num": attr.Int(active_num),
               "expert_capacity": attr.Int(expert_capacity),
               "expert_num": attr.Int(expert_num),
               "drop_pad_mode": attr.Int(drop_pad_mode),
               "expert_tokens_num_type": attr.Int(expert_tokens_num_type),
               "expert_tokens_num_flag": attr.Bool(expert_tokens_num_flag),
               "quant_mode": attr.Int(quant_mode),
               "active_expert_range": attr.Int(active_expert_range),
               "row_idx_type": attr.Int(row_idx_type),
               },
        outputs=["expanded_x",
                 "expanded_row_idx",
                 "expert_tokens_count_or_cumsum",
                 "expanded_scale"]
    )