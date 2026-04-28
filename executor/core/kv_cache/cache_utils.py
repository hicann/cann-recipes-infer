# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Utilities for paged-attention cache sizing."""

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from .cache_info import CacheEntry, ModelCacheInfo
from .kv_cache_manager import KVCacheManager
from .single_type_kv_cache_manager import ATTN_TYPE_MANAGER_MAP


# Cache types with fixed block count (independent of seq_len),
# e.g., sliding window attention. Distinguished from paged attention types
# where block count scales with sequence length.
FIXED_BLOCK_ATTN_TYPES = {"SlidingWindow"}


def dtype_itemsize(dtype: torch.dtype) -> int:
    """Return item size in bytes for a torch dtype."""
    return torch.empty((), dtype=dtype).element_size()


def validate_cache_info(cache_info: ModelCacheInfo) -> None:
    """Validate cache metadata before block sizing and manager creation."""
    if cache_info.num_layers != len(cache_info.layer_infos):
        raise ValueError(
            f"ModelCacheInfo.num_layers mismatch: num_layers={cache_info.num_layers}, "
            f"len(layer_infos)={len(cache_info.layer_infos)}"
        )
    if cache_info.block_size <= 0:
        raise ValueError(f"block_size must be positive, but got {cache_info.block_size}")

    supported_attn_types = set(ATTN_TYPE_MANAGER_MAP.keys())

    for layer_info in cache_info.layer_infos:
        if not layer_info.caches:
            raise ValueError(f"Layer {layer_info.layer_idx} must define at least one cache entry")

        for cache in layer_info.caches:
            if not cache.needs_block:
                continue

            if cache.attn_type not in supported_attn_types:
                raise ValueError(
                    f"Unsupported attn_type '{cache.attn_type}' found in layer {layer_info.layer_idx}, "
                    f"cache {cache.cache_name}. Supported attn types are: {sorted(supported_attn_types)}"
                )
            if cache.num_head <= 0:
                raise ValueError(
                    f"cache {cache.cache_name} in layer {layer_info.layer_idx} must have positive num_head, "
                    f"but got {cache.num_head}"
                )
            dims = cache.dim if isinstance(cache.dim, list) else [cache.dim]
            if any(d <= 0 for d in dims):
                raise ValueError(
                    f"cache {cache.cache_name} in layer {layer_info.layer_idx} must have positive dim, "
                    f"but got {cache.dim}"
                )


def allocate_cache_tensors(device, cache_info: ModelCacheInfo, block_num_by_type: Dict[str, int]) -> None:
    """Allocate per-layer cache tensors according to cache metadata."""
    for layer_info in cache_info.layer_infos:
        for cache in layer_info.caches:
            if not cache.needs_block:
                continue
            if cache.tensor_setter is None:
                raise ValueError(
                    f"CacheEntry {cache.cache_name} in layer {layer_info.layer_idx} has no tensor_setter"
                )
            if cache.attn_type not in block_num_by_type:
                raise KeyError(
                    f"Missing block_num for attn_type={cache.attn_type} when allocating {cache.cache_name}"
                )

            block_num = block_num_by_type[cache.attn_type]
            if cache.attn_type in ["FullAttention", "SlidingWindow"]:
                cache_tensor = torch.empty(
                    (block_num, cache_info.block_size, cache.num_head, cache.dim),
                    dtype=cache.dtype,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Creating cache tensor for attn_type='{cache.attn_type}' is not supported. "
                    f"Please add support in allocate_cache_tensors function."
                )

            cache.tensor_setter(cache_tensor)


def calculate_fixed_block_memory_bytes(infer_config, cache_info: ModelCacheInfo) -> Tuple[Dict[str, int], int]:
    """Estimate reserved fixed-block cache memory by type and total footprint."""
    max_concurrency = infer_config.scheduler_config.batch_size_per_dp_rank
    per_type_block_num: Dict[str, int] = {}
    total_fixed_block_memory_bytes = 0

    for layer_info in cache_info.layer_infos:
        for cache in layer_info.caches:
            if cache.attn_type not in FIXED_BLOCK_ATTN_TYPES:
                continue

            if "SlidingWindow" in cache.attn_type:
                # In MTP (Multi-Token Prediction) scenario, ensure the cache can store all tokens
                # in the valid sliding window for both main model verification and MTP speculation.
                # Main model verification requires next_n + 1 positions, while MTP further speculates
                # next_n - 1 positions based on the main model's output, totaling 2 * next_n positions.
                # Pre-reserving 2 * next_n positions avoids frequent allocation failures during decode.
                fixed_block_num_per_batch = \
                    (2 * infer_config.model_config.next_n + cache.sliding_window + cache_info.block_size - 1) \
                    // cache_info.block_size
                # The additional blocks are designed for sliding window spanning across block boundaries,
                # ensuring all blocks within the window are preserved;
                fixed_block_num = max_concurrency * (fixed_block_num_per_batch + 1)
                # The additional one block is allocated for the null block.
                fixed_block_num += 1

                tmp_memory_bytes = fixed_block_num * cache_info.block_size * cache.num_head \
                    * cache.dim * dtype_itemsize(cache.dtype)
            else:
                raise AttributeError(
                        f"If other attention types {cache.attn_type} are added to FIXED_BLOCK_ATTN_TYPES, "
                        " please compute the corresponding fixed_block_num."
                    )

            per_type_block_num[cache.attn_type] = fixed_block_num
            total_fixed_block_memory_bytes += tmp_memory_bytes

    return per_type_block_num, total_fixed_block_memory_bytes


def calculate_block_num(
    infer_config,
    cache_info: ModelCacheInfo,
    offline_max_len=None,
    tp_group=None,
) -> Dict[str, int]:
    """Calculate block count keyed by attention type."""
    block_num_by_type: Dict[str, int] = {}
    paged_attn_types = set()
    has_fixed_block_cache = False
    per_token_bytes = 0

    # First collect two kinds of metadata:
    # 1. fixed-block attention types, whose block count is determined by attention semantics
    #    rather than by available memory;
    # 2. non fixed-block types, whose block count is derived from the remaining memory budget.
    for layer_info in cache_info.layer_infos:
        for cache in layer_info.caches:
            if cache.attn_type in FIXED_BLOCK_ATTN_TYPES:
                has_fixed_block_cache = True
                continue
            per_token_bytes += cache.dim * cache.num_head * dtype_itemsize(cache.dtype)
            paged_attn_types.add(cache.attn_type)

    if has_fixed_block_cache:
        # Reserve memory for fixed-block caches first, so non fixed-block sizing only uses
        # the memory that remains after these mandatory allocations.
        fixed_block_num_by_type, fixed_block_memory_bytes = calculate_fixed_block_memory_bytes(
            infer_config=infer_config,
            cache_info=cache_info,
        )
        block_num_by_type.update(fixed_block_num_by_type)
    else:
        fixed_block_memory_bytes = 0

    if not paged_attn_types:
        return block_num_by_type

    # In offline mode, block_num can be computed directly from the fixed token length.
    if offline_max_len:
        block_num = int((offline_max_len + cache_info.block_size - 1) / cache_info.block_size)
        block_num = block_num * infer_config.scheduler_config.batch_size_per_dp_rank
        for attn_type in paged_attn_types:
            # The extra one block is allocated for the null block.
            block_num_by_type[attn_type] = block_num + 1

        # Validate that current free_memory can support all requested block memory
        paged_attention_memory_bytes = (block_num + 1) * cache_info.block_size * per_token_bytes
        required_memory_bytes = paged_attention_memory_bytes + fixed_block_memory_bytes
        free_memory, total_memory = torch.npu.mem_get_info()
        if required_memory_bytes > free_memory:
            raise MemoryError(
                f"Insufficient memory for offline mode cache allocation. "
                f"Please reduce the length of requests or the total batch size."
            )
        return block_num_by_type

    # Estimate how many paged-attention tokens can fit into the configured
    # memory utilization budget after subtracting fixed-block cache memory.
    free_memory, total_memory = torch.npu.mem_get_info()
    used_memory = total_memory - free_memory

    mem_fraction_static = infer_config.scheduler_config.mem_fraction_static
    available_memory = total_memory * mem_fraction_static - used_memory - fixed_block_memory_bytes
    if available_memory <= 0:
        raise MemoryError(
            "No available memory for paged attention after fixed-block cache reservation. "
            f"used={used_memory}, fixed_block={fixed_block_memory_bytes}, total={total_memory}, "
            f"mem_fraction_static={mem_fraction_static}, Please boost mem_fraction_static in yaml."
        )

    # Convert the remaining memory budget to token capacity, then to block capacity.
    max_tokens = int(available_memory // per_token_bytes)
    non_fixed_block_num = max_tokens // cache_info.block_size
    if tp_group is not None and dist.is_available() and dist.is_initialized():
        # Different TP ranks may observe different free memory at startup, which can lead to
        # inconsistent paged-attention cache sizing inside the same TP domain. Synchronize the
        # locally computed non_fixed_block_num with an all-reduce MIN so every rank in the TP
        # group allocates the same block count using the most conservative memory budget.
        min_block_num_tensor = torch.tensor(
            [non_fixed_block_num],
            dtype=torch.int64,
            device=torch.device("npu", torch.npu.current_device()),
        )
        dist.all_reduce(min_block_num_tensor, op=dist.ReduceOp.MIN, group=tp_group)
        synced_non_fixed_block_num = int(min_block_num_tensor.item())
        if synced_non_fixed_block_num != non_fixed_block_num:
            logging.info(
                "Sync non_fixed_block_num across attn_tp_group: local=%s, synced_min=%s",
                non_fixed_block_num,
                synced_non_fixed_block_num,
            )
        non_fixed_block_num = synced_non_fixed_block_num

    # Ensure the computed paged-attention capacity can still satisfy the configured
    # maximum input length requirement.
    # Note: Although non_fixed_block_num blocks are allocated, the first block is reserved
    # as the null_block placeholder and does not participate in actual cache storage,
    # so the first block should be subtracted when calculating the supported token count.
    supported_tokens = (non_fixed_block_num - 1) * cache_info.block_size
    required_tokens = infer_config.scheduler_config.max_prefill_tokens
    if supported_tokens - 1 < required_tokens:
        raise MemoryError(
            "Current memory cannot satisfy max input length requirement. "
            f"supported max tokens={supported_tokens}, required max tokens={required_tokens}, "
            f"fixed_block_memory_gb={fixed_block_memory_bytes / 1024**3:.2f}"
        )

    for attn_type in paged_attn_types:
        block_num_by_type[attn_type] = non_fixed_block_num
    return block_num_by_type


def prepare_block_tables(
    requests: Sequence,
    kv_cache_manager: KVCacheManager,
    max_block_num: int,
    device: torch.device,
    batch_size: int = 0,
) -> Dict[str, torch.Tensor]:
    """Prepare block tables for all requests across all KV cache types."""
    block_tables_by_type: Dict[str, torch.Tensor] = {}

    for manager in kv_cache_manager.single_type_managers:
        attn_type = manager.attn_type
        null_block_id = manager.block_pool.get_null_block()

        if requests is None and batch_size > 0:
            # dummy
            block_table_tensor = torch.zeros([batch_size, max_block_num], dtype=torch.int32, device=device)
        else:
            block_table_list: List[List[int]] = []
            for request in requests:
                request_id = request.request_id
                blocks = manager.req_to_blocks.get(request_id, [])

                # Pad with null_block_id to max_block_num
                padded_blocks = list(blocks)
                if len(padded_blocks) < max_block_num:
                    padded_blocks.extend([null_block_id] * (max_block_num - len(padded_blocks)))

                block_table_list.append(padded_blocks)

            # Convert to tensor
            block_table_tensor = torch.tensor(block_table_list, dtype=torch.int32, device=device).view(batch_size, -1)
        block_tables_by_type[attn_type] = block_table_tensor

    return block_tables_by_type


def prepare_slot_mapping(
    position_ids: torch.Tensor,
    actual_seq_lengths_cu_q: torch.Tensor,
    block_tables: Dict[str, torch.Tensor],
    block_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute slot mapping for each cache type from position_ids and block_tables."""
    slot_mapping_by_type: Dict[str, torch.Tensor] = {}

    for attn_type, block_table in block_tables.items():
        if block_table.shape[1] == 0:
            raise ValueError(f"block_table for attn_type={attn_type} must have non-zero width")
        slot_mappings = []
        for idx in range(actual_seq_lengths_cu_q.shape[0]):
            # Split position_ids for single batch
            start_idx = 0 if idx == 0 else actual_seq_lengths_cu_q[idx - 1].item()
            end_idx = actual_seq_lengths_cu_q[idx].item()
            tmp_position_ids = position_ids[start_idx: end_idx]

            # Compute block indices and offsets from position_ids
            block_indices = tmp_position_ids // block_size
            position_offsets = tmp_position_ids % block_size
            max_block_index = int(block_indices.max().item()) if block_indices.numel() > 0 else -1
            if max_block_index >= block_table.shape[1]:
                raise ValueError(
                    f"block_indices out of range for attn_type={attn_type}: "
                    f"max_index={max_block_index}, block_table_width={block_table.shape[1]}"
                )

            # block_table[idx] shape: (max_block_num)
            # Gather block IDs using block_indices
            # For each position, get the corresponding block_id from block_table
            block_ids = torch.gather(block_table[idx], dim=0, index=block_indices)

            # Compute slot mapping: block_id * block_size + offset
            temp_slot_mapping = block_ids * block_size + position_offsets
            slot_mappings.append(temp_slot_mapping)

        total_slot_mapping = torch.cat(slot_mappings)
        slot_mapping_by_type[attn_type] = total_slot_mapping

    return slot_mapping_by_type
