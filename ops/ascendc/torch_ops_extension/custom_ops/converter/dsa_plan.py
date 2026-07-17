# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

from typing import Any

import torch
import torch_npu
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge import attr
from torchair.ge._ge_graph import Tensor, auto_convert_to_tensor


@auto_convert_to_tensor(
    [False, False, False, False, False],
    [False, False, False, False, False])
def DsaPlan(selection_topk_indices: Tensor,
            full_kv_actual_seq: Tensor,
            pool_ids: Tensor,
            id_to_slot: Tensor,
            lru_counter: Tensor,
            *,
            raw_seq: int = 1,
            group_id: int = 0,
            owner_layer: int = 0,
            group_kind: int = 0):
    inputs = {
        "selection_topk_indices": selection_topk_indices,
        "full_kv_actual_seq": full_kv_actual_seq,
        "pool_ids": pool_ids,
        "id_to_slot": id_to_slot,
        "lru_counter": lru_counter,
    }
    attrs = {
        "raw_seq": attr.Int(raw_seq),
        "group_id": attr.Int(group_id),
        "owner_layer": attr.Int(owner_layer),
        "group_kind": attr.Int(group_kind),
    }
    outputs = [
        "plan",
        "install_records",
        "selection_kv_actual_seq",
    ]
    return torchair.ge.custom_op("DsaPlan", inputs=inputs, attrs=attrs, outputs=outputs)


@register_fx_node_ge_converter(torch.ops.custom.dsa_plan.default)
def convert_dsa_plan(selection_topk_indices: Tensor,
                     full_kv_actual_seq: Tensor,
                     pool_ids: Tensor,
                     id_to_slot: Tensor,
                     lru_counter: Tensor,
                     *,
                     raw_seq: int = 1,
                     group_id: int = 0,
                     owner_layer: int = 0,
                     group_kind: int = 0,
                     meta_outputs: Any = None):
    return DsaPlan(
        selection_topk_indices,
        full_kv_actual_seq,
        pool_ids,
        id_to_slot,
        lru_counter,
        raw_seq=raw_seq,
        group_id=group_id,
        owner_layer=owner_layer,
        group_kind=group_kind,
    )
