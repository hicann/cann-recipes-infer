# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
@register_fx_node_ge_converter(torch.ops.custom_pypto.npu_lightning_indexer_pto.default)
def convert_npu_lightning_indexer_pto(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    *,
    actual_seq_lengths_query: Tensor = None,
    actual_seq_lengths_key: Tensor = None,
    block_table: Tensor = None,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    meta_outputs: Any = None
    ):

    '''NB: npu_lightning_indexer_pto(Tensor query, Tensor key, Tensor weights, Tensor actual_seq_lengths_key, *, Tensor? block_table=None) -> Tensor'''

    input_list = [query, key, weights, actual_seq_lengths_key, block_table]
    
    return torchair.ge.custom_op(
        "LightningIndexerPto",
        inputs={"x0": query,
                "x1": key,
                "x2": weights,
                "x3": actual_seq_lengths_key,
                "x4": block_table,
                },
        outputs=['y0']
    )