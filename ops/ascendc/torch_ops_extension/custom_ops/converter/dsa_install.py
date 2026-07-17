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
    [False, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, False])
def DsaInstall(install_records: Tensor,
               selection_kv_cache: Tensor,
               selection_k_rope: Tensor,
               selection_kv_block_table: Tensor,
               pool_kv_cache: Tensor,
               pool_k_rope: Tensor,
               pool_ids: Tensor,
               id_to_slot: Tensor,
               lru_counter: Tensor,
               *,
               raw_seq: int = 1,
               topk: int = 2048,
               selection_block_size: int = 128,
               metadata_update: int = 1):
    return torchair.ge.custom_op(
        "DsaInstall",
        inputs={
            "install_records": install_records,
            "selection_kv_cache": selection_kv_cache,
            "selection_k_rope": selection_k_rope,
            "selection_kv_block_table": selection_kv_block_table,
            "pool_kv_cache": pool_kv_cache,
            "pool_k_rope": pool_k_rope,
            "pool_ids": pool_ids,
            "id_to_slot": id_to_slot,
            "lru_counter": lru_counter,
        },
        attrs={
            "raw_seq": attr.Int(raw_seq),
            "topk": attr.Int(topk),
            "selection_block_size": attr.Int(selection_block_size),
            "metadata_update": attr.Int(metadata_update),
        },
        outputs=[
            "pool_kv_cache",
            "pool_k_rope",
            "pool_ids",
            "id_to_slot",
            "lru_counter",
        ],
    )


@register_fx_node_ge_converter(torch.ops.custom.dsa_install.default)
def convert_dsa_install(install_records: Tensor,
                        selection_kv_cache: Tensor,
                        selection_k_rope: Tensor,
                        selection_kv_block_table: Tensor,
                        pool_kv_cache: Tensor,
                        pool_k_rope: Tensor,
                        pool_ids: Tensor,
                        id_to_slot: Tensor,
                        lru_counter: Tensor,
                        *,
                        raw_seq: int = 1,
                        topk: int = 2048,
                        selection_block_size: int = 128,
                        metadata_update: int = 1,
                        meta_outputs: Any = None):
    return DsaInstall(
        install_records,
        selection_kv_cache,
        selection_k_rope,
        selection_kv_block_table,
        pool_kv_cache,
        pool_k_rope,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=selection_block_size,
        metadata_update=metadata_update,
    )


@register_fx_node_ge_converter(torch.ops.custom.dsa_install_functional.default)
def convert_dsa_install_functional(install_records: Tensor,
                                   selection_kv_cache: Tensor,
                                   selection_k_rope: Tensor,
                                   selection_kv_block_table: Tensor,
                                   pool_kv_cache: Tensor,
                                   pool_k_rope: Tensor,
                                   pool_ids: Tensor,
                                   id_to_slot: Tensor,
                                   lru_counter: Tensor,
                                   *,
                                   raw_seq: int = 1,
                                   topk: int = 2048,
                                   selection_block_size: int = 128,
                                   metadata_update: int = 1,
                                   meta_outputs: Any = None):
    resident_copies = (
        ge.TensorMove(pool_kv_cache),
        ge.TensorMove(pool_k_rope),
        ge.TensorMove(pool_ids),
        ge.TensorMove(id_to_slot),
        ge.TensorMove(lru_counter),
    )
    return DsaInstall(
        install_records,
        selection_kv_cache,
        selection_k_rope,
        selection_kv_block_table,
        *resident_copies,
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=selection_block_size,
        metadata_update=metadata_update,
    )
