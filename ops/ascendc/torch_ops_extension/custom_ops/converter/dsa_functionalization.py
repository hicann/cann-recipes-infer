# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

import torch
from torch.library import Library, impl


m = Library("custom", "FRAGMENT")


@impl(m, "dsa_serve", "Functionalize")
def custom_dsa_serve_func(plan,
                          full_kv_cache,
                          full_k_rope,
                          pool_kv_cache,
                          pool_k_rope,
                          selection_kv_cache,
                          selection_k_rope,
                          *,
                          raw_seq=1,
                          topk=2048,
                          selection_block_size=128,
                          compact_layout=1):
    selection_kv_next, selection_rope_next = torch.ops.custom.dsa_serve_functional(
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
    selection_kv_cache.copy_(selection_kv_next)
    selection_k_rope.copy_(selection_rope_next)
    return None


@impl(m, "dsa_install", "Functionalize")
def custom_dsa_install_func(install_records,
                            selection_kv_cache,
                            selection_k_rope,
                            selection_kv_block_table,
                            pool_kv_cache,
                            pool_k_rope,
                            pool_ids,
                            id_to_slot,
                            lru_counter,
                            *,
                            raw_seq=1,
                            topk=2048,
                            selection_block_size=128,
                            metadata_update=1):
    outputs = torch.ops.custom.dsa_install_functional(
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
    for destination, source in zip(
        (pool_kv_cache, pool_k_rope, pool_ids, id_to_slot, lru_counter),
        outputs,
    ):
        destination.copy_(source)
    return None
