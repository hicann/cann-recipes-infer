# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
@register_fx_node_ge_converter(torch.ops.custom.npu_lightning_indexer_quant.default)
def convert_npu_lightning_indexer_quant(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    query_dequant_scale: Tensor,
    key_dequant_scale: Tensor,
    *,
    actual_seq_lengths_query: Tensor = None,
    actual_seq_lengths_key: Tensor = None,
    block_table: Tensor = None,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "LightningIndexerQuant",
        inputs={"query": query,
                "key": key,
                "weights": weights,
                "query_dequant_scale": query_dequant_scale,
                "key_dequant_scale": key_dequant_scale,
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_key": actual_seq_lengths_key,
                "block_table": block_table,
                },
        attrs={"query_quant_mode": attr.Int(query_quant_mode),
               "key_quant_mode": attr.Int(key_quant_mode),
               "layout_query": attr.Str(layout_query),
               "layout_key": attr.Str(layout_key),
               "sparse_count": attr.Int(sparse_count),
               "sparse_mode": attr.Int(sparse_mode),
               },
        outputs=['selected_indices']
    )
