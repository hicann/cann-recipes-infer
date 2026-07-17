# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared-indexer offload helpers for GLM-5.2.

The production model path is selected by ``shared_indexer_offload: True`` in
the model config. This module keeps only model-route orchestration; operator
shape and tiling contracts are validated by the custom operators.
"""

from __future__ import annotations

import torch

from executor.utils.stream_utils import (
    npu_stream_switch,
    record_event,
    record_stream,
    wait_event,
)


def shared_indexer_offload_enabled(model_config: dict | None = None) -> bool:
    if model_config is None:
        return False
    return bool(model_config.get("enable_offload", False)) and bool(model_config.get("shared_indexer_offload", False))


def run_shared_indexer_offload(
    *,
    layer_idx: int,
    topk_indices: torch.Tensor,
    full_kv_actual_seq: torch.Tensor,
    full_kv_cache: torch.Tensor,
    full_k_rope: torch.Tensor,
    selection_kv_cache: torch.Tensor,
    selection_k_rope: torch.Tensor,
    pool_kv_cache: torch.Tensor,
    pool_k_rope: torch.Tensor,
    offload_cache,
    raw_seq: int,
    topk: int,
    selection_block_size: int,
    enable_install_stream: bool = False,
    install_stream=None,
    install_events=(),
    exe_mode: str = "ge_graph",
):
    group_id, pool_ids, id_to_slot, lru_counter = offload_cache.get_dsa_shared_group_metadata(layer_idx)
    group_spec = offload_cache.get_dsa_shared_group_spec(group_id)
    pool_size = offload_cache.dsa_pool_size
    batch = topk_indices.size(0)
    kv_dim = pool_kv_cache.size(-1)
    rope_dim = pool_k_rope.size(-1)
    pool_kv_view = pool_kv_cache.view(batch, pool_size, kv_dim)
    pool_rope_view = pool_k_rope.view(batch, pool_size, rope_dim)

    if offload_cache.is_dsa_shared_group_owner(layer_idx):
        plan, install_records, actual_seq = torch.ops.custom.dsa_plan(
            topk_indices,
            full_kv_actual_seq,
            pool_ids,
            id_to_slot,
            lru_counter,
            raw_seq=raw_seq,
            group_id=int(group_id),
            owner_layer=int(group_spec["owner_layer"]),
            group_kind=int(group_spec["group_kind"]),
        )
        offload_cache.set_dsa_shared_group_plan(
            group_id,
            plan,
            install_records,
            actual_seq,
        )
    else:
        plan, install_records, actual_seq = offload_cache.get_dsa_shared_group_plan(group_id)

    torch.ops.custom.dsa_serve(
        plan,
        full_kv_cache,
        full_k_rope,
        pool_kv_view,
        pool_rope_view,
        selection_kv_cache,
        selection_k_rope,
        raw_seq=raw_seq,
        topk=topk,
        selection_block_size=selection_block_size,
        compact_layout=1,  # KV/rope pools use compact [batch, pool_size, dim] layout.
    )
    offload_cache.update_selected_key_values(layer_idx, selection_kv_cache, selection_k_rope)
    is_owner = offload_cache.is_dsa_shared_group_owner(layer_idx)
    is_last = offload_cache.is_dsa_shared_group_last_layer(layer_idx)
    has_install = not is_owner or is_last
    install_stream_enabled = enable_install_stream and install_stream is not None and has_install
    npugraph_install_stream = install_stream_enabled and exe_mode == "npugraph_ex"
    if npugraph_install_stream and (not install_events or install_events[0] is None):
        raise RuntimeError("npugraph_ex DsaInstall side stream requires a capture event")

    def record_install_inputs(target_layer_idx: int) -> None:
        target_selection_kv, target_selection_rope = offload_cache.selected_key_values[target_layer_idx]
        target_pool_kv, target_pool_rope = offload_cache.get_dsa_layer_pool_values(target_layer_idx)
        tensors = (
            install_records,
            target_selection_kv,
            target_selection_rope,
            offload_cache.selection_kv_block_table[target_layer_idx],
            target_pool_kv,
            target_pool_rope,
            pool_ids,
            id_to_slot,
            lru_counter,
        )
        for tensor in tensors:
            record_stream(True, tensor, install_stream, exe_mode=exe_mode)

    def install_layer(target_layer_idx: int, metadata_update: int) -> None:
        target_selection_kv, target_selection_rope = offload_cache.selected_key_values[target_layer_idx]
        target_pool_kv, target_pool_rope = offload_cache.get_dsa_layer_pool_values(target_layer_idx)
        torch.ops.custom.dsa_install(
            install_records,
            target_selection_kv,
            target_selection_rope,
            offload_cache.selection_kv_block_table[target_layer_idx],
            target_pool_kv.view(batch, pool_size, kv_dim),
            target_pool_rope.view(batch, pool_size, rope_dim),
            pool_ids,
            id_to_slot,
            lru_counter,
            raw_seq=raw_seq,
            topk=topk,
            selection_block_size=selection_block_size,
            metadata_update=metadata_update,
        )
        offload_cache.update_dsa_layer_pool_values(target_layer_idx, target_pool_kv, target_pool_rope)
        if metadata_update != 0:
            offload_cache.update_dsa_shared_group_metadata(
                group_id,
                pool_ids,
                id_to_slot,
                lru_counter,
            )

    if npugraph_install_stream:
        if not is_owner:
            record_install_inputs(layer_idx)
        if is_last:
            record_install_inputs(int(group_spec["owner_layer"]))
        record_event(True, install_events, 0, exe_mode=exe_mode)
    elif install_stream_enabled and exe_mode != "ge_graph":
        install_stream.wait_stream(torch.npu.current_stream())
    with npu_stream_switch(install_stream_enabled, install_stream, exe_mode=exe_mode):
        wait_event(npugraph_install_stream, install_events, 0, exe_mode=exe_mode)
        if not is_owner:
            install_layer(layer_idx, 0)
        if is_last:
            install_layer(int(group_spec["owner_layer"]), 1)

    if is_last:
        offload_cache.clear_dsa_shared_group_plan(group_id)

    return selection_kv_cache, selection_k_rope, actual_seq, pool_kv_view, pool_rope_view
