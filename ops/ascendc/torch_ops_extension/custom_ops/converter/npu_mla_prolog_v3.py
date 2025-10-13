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
from torch.library import Library, impl

m = Library("custom", "FRAGMENT")
@impl(m, "npu_mla_prolog_v3", "Functionalize")
def custom_npu_mla_prolog_v3_func(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False
):
    query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm, kv_cache_out, kr_cache_out = torch.ops.custom.npu_mla_prolog_v3_functional(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        dequant_scale_x=dequant_scale_x, dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv, quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq, actual_seq_len=actual_seq_len,
        cache_mode=cache_mode, query_norm_flag=query_norm_flag)
    kv_cache.copy_(kv_cache_out)
    kr_cache.copy_(kr_cache_out)
    return query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm

# 为自定义算子注册converter，用于torch.compile 场景成图

# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_mla_prolog_v3.default)
def convert_npu_npu_mla_prolog_v3(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    meta_outputs: TensorSpec = None,
):
    return torchair.ge.custom_op(
        "MlaPrologV3",
        inputs={"token_x": token_x,
                "weight_dq": weight_dq,
                "weight_uq_qr": weight_uq_qr,
                "weight_uk": weight_uk,
                "weight_dkv_kr": weight_dkv_kr,
                "rmsnorm_gamma_cq": rmsnorm_gamma_cq,
                "rmsnorm_gamma_ckv": rmsnorm_gamma_ckv,
                "rope_sin": rope_sin,
                "rope_cos": rope_cos,
                "cache_index": cache_index,
                "kv_cache": kv_cache,
                "kr_cache": kr_cache,
                "dequant_scale_x": dequant_scale_x,
                "dequant_scale_w_dq": dequant_scale_w_dq,
                "dequant_scale_w_uq_qr": dequant_scale_w_uq_qr,
                "dequant_scale_w_dkv_kr": dequant_scale_w_dkv_kr,
                "quant_scale_ckv": quant_scale_ckv,
                "quant_scale_ckr": quant_scale_ckr,
                "smooth_scales_cq": smooth_scales_cq,
                "actual_seq_len": actual_seq_len,
                },
        attrs={"rmsnorm_epsilon_cq": attr.Float(rmsnorm_epsilon_cq),
               "rmsnorm_epsilon_ckv": attr.Float(rmsnorm_epsilon_ckv),
               "cache_mode": attr.Str(cache_mode),
               "query_norm_flag": attr.Bool(query_norm_flag),
               },
        outputs=['query', 'query_rope', 'dequant_scale_q_nope', 'query_norm', 'dequant_scale_q_norm']
    )

@register_fx_node_ge_converter(torch.ops.custom.npu_mla_prolog_v3_functional.default)
def convert_npu_npu_mla_prolog_v3_functional(
    token_x: Tensor,
    weight_dq: Tensor,
    weight_uq_qr: Tensor,
    weight_uk: Tensor,
    weight_dkv_kr: Tensor,
    rmsnorm_gamma_cq: Tensor,
    rmsnorm_gamma_ckv: Tensor,
    rope_sin: Tensor,
    rope_cos: Tensor,
    cache_index: Tensor,
    kv_cache: Tensor,
    kr_cache: Tensor,
    *,
    dequant_scale_x: Optional[Tensor] = None,
    dequant_scale_w_dq: Optional[Tensor] = None,
    dequant_scale_w_uq_qr: Optional[Tensor] = None,
    dequant_scale_w_dkv_kr: Optional[Tensor] = None,
    quant_scale_ckv: Optional[Tensor] = None,
    quant_scale_ckr: Optional[Tensor] = None,
    smooth_scales_cq: Optional[Tensor] = None,
    actual_seq_len: Optional[Tensor] = None,
    rmsnorm_epsilon_cq: float = 1e-5,
    rmsnorm_epsilon_ckv: float = 1e-5,
    cache_mode: str = "PA_BSND",
    query_norm_flag: bool = False,
    meta_outputs: TensorSpec = None
):
    kv_cache_copy = ge.TensorMove(kv_cache)
    kr_cache_copy = ge.TensorMove(kr_cache)
    (
        query,
        query_rope,
        kv_cache_out,
        kr_cache_out,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
    ) = ge.MlaPrologV3(
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        cache_index,
        kv_cache_copy,
        kr_cache_copy,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_w_dq=dequant_scale_w_dq,
        dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr=dequant_scale_w_dkv_kr,
        quant_scale_ckv=quant_scale_ckv,
        quant_scale_ckr=quant_scale_ckr,
        smooth_scales_cq=smooth_scales_cq,
        actual_seq_kvlen=actual_seq_len,
        rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
        cache_mode=cache_mode,
        query_norm_flag=query_norm_flag,
    )
    return (
        query,
        query_rope,
        dequant_scale_q_nope,
        query_norm,
        dequant_scale_q_norm,
        kv_cache_out,
        kr_cache_out,
    )