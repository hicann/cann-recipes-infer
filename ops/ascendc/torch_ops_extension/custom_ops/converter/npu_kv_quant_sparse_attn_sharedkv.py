# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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
@register_fx_node_ge_converter(torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv.default)
def convert_npu_kv_quant_sparse_attn_sharedkv(
    q: Tensor,
    kv_quant_mode: int,
    *,
    ori_kv: Optional[Tensor] = None,
    cmp_kv: Optional[Tensor] = None,
    ori_sparse_indices: Optional[Tensor] = None,
    cmp_sparse_indices: Optional[Tensor] = None,
    ori_block_table: Optional[Tensor] = None,
    cmp_block_table: Optional[Tensor] = None,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_ori_kv: Optional[Tensor] = None,
    cu_seqlens_cmp_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    sinks: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    tile_size: int = None,
    rope_head_dim: int = 0,
    softmax_scale: float = None,
    cmp_ratio: int = None,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
    meta_outputs: TensorSpec = None,
    softmax_lse: TensorSpec = None,
    ):
    return torchair.ge.custom_op(
    "KvQuantSparseAttnSharedkv",
    inputs={"q": q,
            "ori_kv": ori_kv,
            "cmp_kv": cmp_kv,
            "cmp_sparse_indices": cmp_sparse_indices,
            "ori_sparse_indices": ori_sparse_indices,
            "ori_block_table": ori_block_table,
            "cmp_block_table": cmp_block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
            "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "sinks": sinks,
            "metadata": metadata,
            },
    attrs={"kv_quant_mode": attr.Int(kv_quant_mode),
            "tile_size": attr.Int(tile_size),
            "rope_head_dim": attr.Int(rope_head_dim),
            "softmax_scale": attr.Float(softmax_scale),
            "cmp_ratio": attr.Int(cmp_ratio),
            "ori_mask_mode": attr.Int(ori_mask_mode),
            "cmp_mask_mode": attr.Int(cmp_mask_mode),
            "ori_win_left": attr.Int(ori_win_left),
            "ori_win_right": attr.Int(ori_win_right),
            "layout_q": attr.Str(layout_q),
            "layout_kv": attr.Str(layout_kv),
            "return_softmax_lse": attr.Bool(return_softmax_lse),
            },
    outputs=['attn_out', 'softmax_lse']
)