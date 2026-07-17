# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

from typing import Any

import torch
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, auto_convert_to_tensor


@auto_convert_to_tensor(
    [False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False])
def DsaServe(plan: Tensor,
             full_kv_cache: Tensor,
             full_k_rope: Tensor,
             pool_kv_cache: Tensor,
             pool_k_rope: Tensor,
             selection_kv_cache: Tensor,
             selection_k_rope: Tensor,
             *,
             raw_seq: int = 1,
             topk: int = 2048,
             selection_block_size: int = 128,
             compact_layout: int = 1):
    inputs = {
        "plan": plan,
        "full_kv_cache": full_kv_cache,
        "full_k_rope": full_k_rope,
        "pool_kv_cache": pool_kv_cache,
        "pool_k_rope": pool_k_rope,
        "selection_kv_cache": selection_kv_cache,
        "selection_k_rope": selection_k_rope,
    }
    attrs = {
        "raw_seq": attr.Int(raw_seq),
        "topk": attr.Int(topk),
        "selection_block_size": attr.Int(selection_block_size),
        "compact_layout": attr.Int(compact_layout),
    }
    outputs = [
        "selection_kv_cache",
        "selection_k_rope",
    ]
    return torchair.ge.custom_op("DsaServe", inputs=inputs, attrs=attrs, outputs=outputs)


@register_fx_node_ge_converter(torch.ops.custom.dsa_serve.default)
def convert_dsa_serve(plan: Tensor,
                      full_kv_cache: Tensor,
                      full_k_rope: Tensor,
                      pool_kv_cache: Tensor,
                      pool_k_rope: Tensor,
                      selection_kv_cache: Tensor,
                      selection_k_rope: Tensor,
                      *,
                      raw_seq: int = 1,
                      topk: int = 2048,
                      selection_block_size: int = 128,
                      compact_layout: int = 1,
                      meta_outputs: Any = None):
    return DsaServe(
        plan,
        full_kv_cache,
        full_k_rope,
        pool_kv_cache,
        pool_k_rope,
        selection_kv_cache,
        selection_k_rope,
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=selection_block_size,
        compact_layout=compact_layout,
    )


@register_fx_node_ge_converter(torch.ops.custom.dsa_serve_functional.default)
def convert_dsa_serve_functional(plan: Tensor,
                                 full_kv_cache: Tensor,
                                 full_k_rope: Tensor,
                                 pool_kv_cache: Tensor,
                                 pool_k_rope: Tensor,
                                 selection_kv_cache: Tensor,
                                 selection_k_rope: Tensor,
                                 *,
                                 raw_seq: int = 1,
                                 topk: int = 2048,
                                 selection_block_size: int = 128,
                                 compact_layout: int = 1,
                                 meta_outputs: Any = None):
    selection_kv_copy = ge.TensorMove(selection_kv_cache)
    selection_rope_copy = ge.TensorMove(selection_k_rope)
    return DsaServe(
        plan,
        full_kv_cache,
        full_k_rope,
        pool_kv_cache,
        pool_k_rope,
        selection_kv_copy,
        selection_rope_copy,
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=selection_block_size,
        compact_layout=compact_layout,
    )
