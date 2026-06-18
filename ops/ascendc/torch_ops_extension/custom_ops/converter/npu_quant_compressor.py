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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.custom.quant_compressor.default)
def convert_quant_compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        quant_mode,
        cmp_ratio,
        *,
        x_descale=None,
        wkv_descale=None,
        wgate_descale=None,
        state_block_table=None,
        cu_seqlens=None,
        seqused=None,
        start_pos=None,
        coff=1,
        cache_mode=1
):
    state_cache_stride_dim0 = int(state_cache.symsize[-2]) * int(state_cache.symsize[-1])
    out = torchair.ge.custom_op(
        "QuantCompressor", x, wkv, wgate, state_cache, ape,
        x_descale, wkv_descale, wgate_descale, state_block_table, cu_seqlens, seqused, start_pos,
        quant_mode, cmp_ratio, coff, cache_mode, state_cache_stride_dim0)
    return (out[0])