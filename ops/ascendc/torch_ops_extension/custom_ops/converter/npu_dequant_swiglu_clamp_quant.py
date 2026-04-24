from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload, Optional
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, torch_dtype_value_to_ge_type, \
    torch_dtype_value_to_ge_proto_type
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote


@register_fx_node_ge_converter(torch.ops.custom.npu_dequant_swiglu_clamp_quant.default)
def conveter_npu_dequant_swiglu_clamp_quant_default(
        x: Tensor,
        weight_scale: Tensor = None,
        activation_scale: Tensor = None,
        bias: Tensor = None,
        quant_scale: Tensor = None,
        quant_offset: Tensor = None,
        group_index: Tensor = None,
        activate_left: bool = False,
        quant_mode: int = 0,
        dst_type: int = None,
        round_mode: int = None,
        activate_dim: int = None,
        swiglu_mode: int = 0,
        clamp_limit: float = 7.0,
        glu_alpha: float = 1.702,
        glu_bias: float = 1.0):
    dst_type = dst_type if dst_type is not None else 1
    round_mode = round_mode if round_mode is not None else 0
    activate_dim = activate_dim if activate_dim is not None else -1
    quant_mode_str = 'static'
    if quant_mode == 1:
        quant_mode_str = 'dynamic'

    round_mode_str = "rint"
    if round_mode == 1:
        round_mode_str = "round"
    elif round_mode == 2:
        round_mode_str = "floor"
    elif round_mode == 3:
        round_mode_str = "ceil"
    elif round_mode == 4:
        round_mode_str = "trunc"

    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    return torchair.ge.custom_op(
        "DequantSwigluClampQuant",
        inputs={"x": x,
                "weight_scale": weight_scale,
                "activation_scale": activation_scale,
                "bias": bias,
                "quant_scale": quant_scale,
                "quant_offset": quant_offset,
                "group_index": group_index,
                },
        attrs={
            "activate_left": attr.Bool(activate_left),
            "quant_mode": attr.Str(quant_mode_str),
            "dst_type": attr.Int(acl_dst_type),
            "round_mode": attr.Str(round_mode_str),
            "activate_dim": attr.Int(activate_dim),
            "swiglu_mode": attr.Int(swiglu_mode),
            "clamp_limit": attr.Float(clamp_limit),
            "glu_alpha": attr.Float(glu_alpha),
            "glu_bias": attr.Float(glu_bias),
        },
        outputs=['y','scale']
    )