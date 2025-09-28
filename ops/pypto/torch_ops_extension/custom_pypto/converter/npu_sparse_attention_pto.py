# This program is free software, you can redistribute it and/or modify it.
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
@register_fx_node_ge_converter(torch.ops.custom_pypto.npu_sparse_attention_pto.default)
def convert_npu_sparse_attention_pto(
    x: Tensor,
    w_dq: Tensor,
    w_uq_qr: Tensor,
    w_uk: Tensor,
    w_dkv_kr: Tensor,
    gamma_cq: Tensor,
    gamma_ckv: Tensor,
    sin: Tensor,
    cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    block_table: Tensor,
    act_seqs: Tensor,
    w_idx_qb: Tensor,
    w_idx_k: Tensor,
    w_idx_proj: Tensor,
    in_gamma_k: Tensor,
    in_beta_k: Tensor,
    index_k_cache: Tensor,
    meta_outputs: Any = None
    ):

    '''NB: npu_sparse_attention_pto(Tensor x, Tensor w_dq, Tensor w_uq_qr, Tensor w_uk, Tensor w_dkv_kr, Tensor gamma_cq, Tensor gamma_ckv,
                                    Tensor sin, Tensor cos, Tensor cache_index, Tensor kv_cache, Tensor kr_cache, Tensor block_table,
                                    Tensor act_seqs, Tensor w_idx_qb, Tensor w_idx_k, Tensor w_idx_proj, Tensor in_gamma_k, Tensor in_beta_k, Tensor index_k_cache) -> Tensor'''

    input_list = [
        x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
        sin, cos, cache_index, kv_cache, kr_cache, block_table,
        act_seqs, w_idx_qb, w_idx_k, w_idx_proj, in_gamma_k, in_beta_k, index_k_cache
    ]
    
    return torchair.ge.custom_op(
        "SparseAttentionPto",
        inputs={
            "x0": x,
            "x1": w_dq,
            "x2": w_uq_qr,
            "x3": w_uk,
            "x4": w_dkv_kr,
            "x5": gamma_cq,
            "x6": gamma_ckv,
            "x7": sin,
            "x8": cos,
            "x9": cache_index,
            "x10": kv_cache,
            "x11": kr_cache,
            "x12": block_table,
            "x13": act_seqs,
            "x14": w_idx_qb,
            "x15": w_idx_k,
            "x16": w_idx_proj,
            "x17": in_gamma_k,
            "x18": in_beta_k,
            "x19": index_k_cache,
        },
        outputs=['y0']
    )


