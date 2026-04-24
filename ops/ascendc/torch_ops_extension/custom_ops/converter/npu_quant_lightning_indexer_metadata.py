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
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair.ge import attr


# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_quant_lightning_indexer_metadata.default)
def convert_npu_quant_lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    query_quant_mode: int,
    key_quant_mode: int,
    *,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_key: Optional[Tensor] = None,
    batch_size: int = 0,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    cmp_ratio: int = 1,
    device: str = "npu:0",
    meta_outputs: TensorSpec = None,
):
    stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
    aic_core_num = stream_info.get("cube_core_num")
    aiv_core_num = stream_info.get("vector_core_num")
    soc_version = torch.npu.get_device_properties().name
    
    return torchair.ge.custom_op(
        "QuantLightningIndexerMetadata",
        inputs={
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_key": actual_seq_lengths_key,
               },
        attrs={
               "aic_core_num": attr.Int(aic_core_num),
               "aiv_core_num": attr.Int(aiv_core_num),
               "soc_version": attr.Str(soc_version),
               "num_heads_q": attr.Int(num_heads_q),
               "num_heads_k": attr.Int(num_heads_k),
               "head_dim": attr.Int(head_dim),
               "query_quant_mode": attr.Int(query_quant_mode),
               "key_quant_mode": attr.Int(key_quant_mode),
               "batch_size": attr.Int(batch_size),
               "max_seqlen_q": attr.Int(max_seqlen_q),
               "max_seqlen_k": attr.Int(max_seqlen_k),
               "layout_query": attr.Str(layout_query),
               "layout_key": attr.Str(layout_key),
               "sparse_count": attr.Int(sparse_count),
               "sparse_mode": attr.Int(sparse_mode),
               "pre_tokens": attr.Int(pre_tokens),
               "next_tokens": attr.Int(next_tokens),
               "cmp_ratio": attr.Int(cmp_ratio),
               },
        outputs=['metadata']
    )
