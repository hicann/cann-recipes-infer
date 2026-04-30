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

"""Top-level coordinator for paged-attention KV cache managers."""

from typing import List, Optional, Sequence, Tuple, Dict

from .cache_info import ModelCacheInfo
from .single_type_kv_cache_manager import SingleTypeKVCacheManager


class KVCacheManager:
    """Coordinates all single-type KV cache managers for one worker."""

    def __init__(
        self,
        max_model_len: int,
        single_type_managers: Sequence[SingleTypeKVCacheManager],
        cache_info: ModelCacheInfo
    ):
        self.max_model_len = max_model_len
        self.single_type_managers = tuple(single_type_managers)
        self.num_attn_types = len(self.single_type_managers)
        self.cache_info = cache_info
        self.is_mla_backend = self.cache_info.is_mla_backend

    def allocate_slots(
        self,
        request_id: int,
        computed_tokens: int,
        num_new_tokens: int,
        lookahead_tokens: int = 0,
    ) -> bool:
        """Coordinate slot allocation across all KV cache types."""
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative")
        total_computed_tokens = min(computed_tokens, self.max_model_len)
        num_tokens_need_slot = min(
            computed_tokens + num_new_tokens + lookahead_tokens,
            self.max_model_len,
        )

        block_num_to_allocate_per_manager: List[int] = []
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id=request_id,
                total_computed_tokens=total_computed_tokens,
            )
            block_num_to_allocate = manager.get_num_blocks_to_allocate(
                request_id=request_id,
                num_tokens=num_tokens_need_slot,
            )
            if block_num_to_allocate > manager.get_num_free_blocks():
                return False
            block_num_to_allocate_per_manager.append(block_num_to_allocate)

        for manager, block_num_to_allocate in zip(
            self.single_type_managers,
            block_num_to_allocate_per_manager,
        ):
            manager.allocate_new_blocks(
                request_id=request_id,
                block_num_to_allocate=block_num_to_allocate,
                num_tokens=num_tokens_need_slot,
            )
        return True

    def get_block_ids(self, request_id: int) -> Dict:
        """Return all blocks associated with a request."""
        return {
            manager.attn_type: manager.get_blocks(request_id)
            for manager in self.single_type_managers
        }

    def free(self, request_id: int) -> None:
        """Release all cache blocks associated with a request."""
        for manager in self.single_type_managers:
            manager.free(request_id)

    def format_usage(self) -> str:
        """Per-attn-type used/total summary, e.g. ``full:12/100,sliding:8/100``.

        Used by scheduler logs to surface per-type pressure separately
        """
        if not self.single_type_managers:
            return "0/0"
        return ",".join(
            f"{m.attn_type}:{m.block_pool.num_blocks - m.get_num_free_blocks()}"
            f"/{m.block_pool.num_blocks}"
            for m in self.single_type_managers
        )

    def get_contiguous_buf_infos(
        self,
    ) -> Tuple[List[int], List[int], List[int], List[str]]:
        """Flatten all block-managed physical cache tensors for PD KV transfer.

        prefill and decode each call this independently and the returned
        lists are matched positionally (same model config ⇒ same order).

        Order contract (deterministic):
            for layer in cache_info.layer_infos (ascending ``layer_idx``):
                for cache in layer.caches (original definition order):
                    if cache.needs_block: emit one entry

        ``needs_block=False`` entries opt out of the paged-attention pipeline
        — ``allocate_cache_tensors`` does not allocate a tensor for them, they
        have no ``SingleTypeKVCacheManager`` and are absent from
        ``get_block_ids``. PD has nothing to transfer for them and they are
        intentionally skipped here.

        Returns:
            data_ptrs: data_ptr() of each physical cache tensor
            data_lens: total bytes of each cache tensor
            item_lens: bytes per block (block_size × num_head × dim × element_size)
            attn_types: attn_type of each entry (PD side uses this to look up
                the correct per-attn_type block id list)
        """

        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        attn_types: List[str] = []
        block_size = self.cache_info.block_size
        # Keeps a stable, deterministic iteration order without re-sorting per call.
        sorted_layer_infos = sorted(self.cache_info.layer_infos, key=lambda li: li.layer_idx)

        for layer_info in sorted_layer_infos:
            for cache in layer_info.caches:
                if not cache.needs_block:
                    continue
                if cache.tensor is None:
                    raise RuntimeError(
                        f"CacheEntry {cache.cache_name} in layer {layer_info.layer_idx} "
                        f"has no tensor — allocate_cache_tensors must run first."
                    )
                t = cache.tensor
                data_ptrs.append(t.data_ptr())
                data_lens.append(t.numel() * t.element_size())
                item_lens.append(
                    block_size * cache.num_head * cache.dim * t.element_size()
                )
                attn_types.append(cache.attn_type)

        return data_ptrs, data_lens, item_lens, attn_types
