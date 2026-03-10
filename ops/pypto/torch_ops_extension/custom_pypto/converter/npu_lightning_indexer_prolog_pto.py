# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr

# 为自定义算子注册converter，用于torch.compile 场景成图
# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom_pypto.npu_lightning_indexer_prolog_pto.default)
def convert_npu_lightning_indexer_prolog_pto(
    token_x: Tensor,
    q_norm: Tensor,
    q_norm_scale: Tensor,
    wq_b: Tensor,
    wq_b_scale: Tensor,
    wk: Tensor,
    weights_proj: Tensor,
    ln_gamma_k: Tensor,
    ln_beta_k: Tensor,
    cos_idx_rope: Tensor,
    sin_idx_rope: Tensor,
    hadamard_q: Tensor,
    hadamard_k: Tensor,
    idx_k_cache: Tensor,
    idx_k_scale_cache: Tensor,
    idx_k_cache_index: Tensor,
    layernorm_epsilon_k: float,
    *,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    meta_outputs: Any = None
):
    return torchair.ge.custom_op(
        "LightningIndexerPrologPto",
        inputs={
            "x0": token_x,
            "x1": q_norm,
            "x2": q_norm_scale,
            "x3": wq_b,
            "x4": wq_b_scale,
            "x5": wk,
            "x6": weights_proj,
            "x7": ln_gamma_k,
            "x8": ln_beta_k,
            "x9": cos_idx_rope,
            "x10": sin_idx_rope,
            "x11": hadamard_q,
            "x12": hadamard_k,
            "x13": idx_k_cache,
            "x14": idx_k_scale_cache,
            "x15": idx_k_cache_index,
            },
        outputs=['y0','y1','y2']
    )