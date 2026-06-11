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

"""Per-KV-cache-type manager abstractions."""

import logging
from math import ceil
from typing import Dict, List, Optional, Type

from .cache_info import CacheEntry, ModelCacheInfo

from .block_pool import BlockPool

logger = logging.getLogger(__name__)


class SingleTypeKVCacheManager:
    """Base class for a single KV cache type.

    The default implementation follows the simplest full-attention block
    management policy: blocks are appended as the sequence grows, and all
    blocks owned by a request are released together when the request ends.
    """

    def __init__(
        self,
        attn_type: str,
        block_num: int,
        block_size: int,
        manager_key: Optional[str] = None,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        self.attn_type = attn_type
        self.manager_key = manager_key if manager_key is not None else attn_type
        self.block_pool = BlockPool(num_blocks=block_num, block_size=block_size)
        self.block_size = block_size
        self.max_model_len = max_model_len
        self._null_block = self.block_pool.get_null_block()
        self.req_to_blocks: Dict[int, List[int]] = {}

    def pre_allocate_blocks(
        self,
        request_id: int,
        num_tokens: int,
        reserved_tokens: int = 0,
    ) -> tuple:
        """Return required blocks and whether the current pool can satisfy it."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        # Admission control inputs on online mode: each request already in the pipeline
        # (running + transfer-queue) reserves num_reserved_decode_tokens of
        # future decode growth.  We only admit a new request if its
        # input + reservation fits in the leftover budget.  Without this gate
        # high-QPS bursts greedily fill KV cache, then no running req can grow
        # to its next token → deadlock.
        required_block_num = ceil(num_tokens / self.block_size)
        existing_block_num = len(self.req_to_blocks.get(request_id, []))
        need_block_num = max(0, required_block_num - existing_block_num)

        reserved_block_per_request = (reserved_tokens + self.block_size - 1) // self.block_size
        reserved_block = reserved_block_per_request * len(self.req_to_blocks)
        free_block_num = self.get_num_free_blocks()
        status = need_block_num <= free_block_num - reserved_block
        if not status:
            logger.warning(
                "preallocate skip request_id=%s: admission rejected "
                "(req=%d blocks, free=%d, reserved_for_pipeline=%d)",
                request_id, need_block_num, free_block_num, reserved_block,
            )
        return need_block_num, status

    def allocate_new_blocks(
        self,
        request_id: int,
        block_num_to_allocate: int,
        num_tokens: Optional[int] = None,
    ):
        """Allocate blocks for a request."""
        new_blocks = self.block_pool.get_new_blocks(block_num_to_allocate)
        request_blocks = self.req_to_blocks.setdefault(request_id, [])
        request_blocks.extend(new_blocks)

    def free(self, request_id: int) -> None:
        """Release all blocks owned by a request."""
        blocks = self.req_to_blocks.pop(request_id, [])
        if not blocks:
            return
        self.block_pool.free_blocks(blocks)

    def get_blocks(self, request_id: int) -> List[int]:
        """Return blocks owned by a request."""
        return self.req_to_blocks.get(request_id, [])

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool."""
        return self.block_pool.get_num_free_blocks()

    def remove_skipped_blocks(self, request_id: int, total_computed_tokens: int) -> None:
        """Remove stale blocks for attention patterns with skipped regions."""
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return
        blocks = self.req_to_blocks.get(request_id, [])
        num_skipped_blocks = num_skipped_tokens // self.block_size
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks = []
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Return how many computed tokens can be skipped."""
        raise NotImplementedError(
            "Subclasses must implement get_num_skipped_tokens"
        )

    @staticmethod
    def validate_and_build_kwargs(group_entries: List[CacheEntry]) -> Dict[str, object]:
        """Validate cache entries and return manager-specific kwargs.

        Subclasses can override this method to extract and validate
        attention-type-specific parameters from cache entries.
        """
        raise NotImplementedError(
            "Subclasses must implement validate_and_build_kwargs"
        )


class FullAttentionManager(SingleTypeKVCacheManager):
    """Manager for full attention KV cache layout."""

    def get_num_skipped_tokens(self, _num_computed_tokens: int) -> int:
        """Full attention never skips tokens."""
        return 0

    @staticmethod
    def validate_and_build_kwargs(_group_entries: List[CacheEntry]) -> Dict[str, object]:
        """Full attention requires no additional kwargs."""
        return {}


class SlidingWindowManager(SingleTypeKVCacheManager):
    """Manager for sliding-window attention KV cache layout."""

    def __init__(
        self,
        attn_type: str,
        block_num: int,
        block_size: int,
        sliding_window: int,
        manager_key: Optional[str] = None,
        max_model_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            attn_type=attn_type,
            block_num=block_num,
            block_size=block_size,
            manager_key=manager_key,
            max_model_len=max_model_len,
        )
        self.sliding_window = sliding_window

    def _get_logical_block_layout(self, num_tokens: int) -> tuple[int, int, int]:
        """Return total logical blocks, first valid block, and valid block count."""
        total_block_num = ceil(num_tokens / self.block_size)
        last_token_idx = num_tokens - 1
        last_block_idx = last_token_idx // self.block_size
        first_valid_token_idx = max(0, num_tokens - self.sliding_window)
        first_valid_block_idx = first_valid_token_idx // self.block_size
        valid_block_num = last_block_idx - first_valid_block_idx + 1
        return total_block_num, first_valid_block_idx, valid_block_num

    def pre_allocate_blocks(
        self,
        request_id: int,
        num_tokens: int,
        reserved_tokens: int = 0,
    ) -> tuple:
        """Return required blocks and whether the current pool can satisfy it."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        required_block_num = ceil(num_tokens / self.block_size)
        existing_block_num = len(self.req_to_blocks.get(request_id, []))
        if existing_block_num > 0:
            # decode
            need_block_num = max(0, required_block_num - existing_block_num)
            return need_block_num, need_block_num <= self.get_num_free_blocks()

        # prefill returns real physical block count only.
        _, _, need_block_num = self._get_logical_block_layout(num_tokens)
        return need_block_num, need_block_num <= self.get_num_free_blocks()

    def allocate_new_blocks(
        self,
        request_id: int,
        block_num_to_allocate: int,
        num_tokens: Optional[int] = None,
    ):
        """Allocate SWA blocks while keeping logical block positions aligned to kv_len."""
        new_blocks = self.block_pool.get_new_blocks(block_num_to_allocate)
        request_blocks = self.req_to_blocks.setdefault(request_id, [])

        if not request_blocks:
            # prefill
            total_block_num, first_valid_block_idx, valid_block_num = self._get_logical_block_layout(num_tokens)

            request_blocks.extend([self._null_block] * first_valid_block_idx)
            request_blocks.extend(new_blocks)

            if len(request_blocks) != total_block_num:
                raise ValueError(
                    "Sliding-window block table length mismatch after prefill allocation: "
                    f"got={len(request_blocks)}, expected={total_block_num}"
                )
        else:
            request_blocks.extend(new_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Return how many computed tokens can be skipped."""
        return max(0, num_computed_tokens - self.sliding_window + 1)

    @staticmethod
    def validate_and_build_kwargs(group_entries: List[CacheEntry]) -> Dict[str, object]:
        """Validate sliding-window metadata and return required manager kwargs."""
        sliding_windows = {cache_entry.sliding_window for cache_entry in group_entries}
        if None in sliding_windows:
            raise ValueError("SlidingWindow cache must define sliding_window.")
        if len(sliding_windows) != 1:
            raise ValueError(
                "All SlidingWindow caches grouped into one manager must share the same "
                f"sliding_window, but got {sorted(sliding_windows)}."
            )
        return {"sliding_window": next(iter(sliding_windows))}


ATTN_TYPE_MANAGER_MAP: Dict[str, Type[SingleTypeKVCacheManager]] = {
    "FullAttention": FullAttentionManager,
    "SlidingWindow": SlidingWindowManager,
}


def create_single_type_managers(
    cache_info: ModelCacheInfo,
    block_num_by_type: Dict[str, int],
    max_model_len: int,
) -> List[SingleTypeKVCacheManager]:
    """Create grouped single-type cache managers from cache metadata."""
    grouped_caches: Dict[str, List] = {}

    # Group cache entries by manager key. Caches with the same attn_type can
    # still need independent block tables and slot mappings.
    for layer_info in cache_info.layer_infos:
        for cache in layer_info.caches:
            if not cache.needs_block:
                continue
            group_key = cache.group_key
            if group_key not in block_num_by_type:
                raise ValueError(
                    f"Unknown manager_key '{group_key}' found in layer {layer_info.layer_idx}, "
                    f"cache {cache.cache_name}, attn_type={cache.attn_type}. "
                    f"Supported cache types are: {list(block_num_by_type.keys())}"
                )
            grouped_caches.setdefault(group_key, []).append(cache)

    single_type_managers: List[SingleTypeKVCacheManager] = []
    manager_infos = []
    # Build one manager per manager_key using shared base kwargs plus any
    # attn-type-specific validated kwargs derived from the grouped entries.
    for group_key, group_entries in grouped_caches.items():
        attn_types = {cache.attn_type for cache in group_entries}
        if len(attn_types) != 1:
            raise ValueError(
                f"Caches grouped into group_key={group_key} must share one attn_type, "
                f"but got {sorted(attn_types)}."
            )
        attn_type = next(iter(attn_types))
        manager_class = ATTN_TYPE_MANAGER_MAP.get(attn_type)
        if manager_class is None:
            raise ValueError(
                f"Unknown attn_type: {attn_type}, "
                f"Please add a SingleTypeKVCacheManager class corresponding to {attn_type}."
            )

        block_sizes = {cache.block_size for cache in group_entries}
        if len(block_sizes) != 1:
            raise ValueError(
                "Caches grouped into one SingleTypeKVCacheManager must share the same block_size. "
                f"group_key={group_key}, attn_type={attn_type}, block_sizes={sorted(block_sizes)}."
            )

        manager_kwargs = {
            "block_num": block_num_by_type[group_key],
            "block_size": next(iter(block_sizes)),
            "manager_key": group_key,
            "max_model_len": max_model_len,
        }
        # Call the class's static method to validate and build type-specific kwargs.
        manager_kwargs.update(manager_class.validate_and_build_kwargs(group_entries))

        # Instantiate the registered manager for this attention type.
        manager = manager_class(attn_type=attn_type, **manager_kwargs)
        single_type_managers.append(manager)
        manager_infos.append((group_key, attn_type, manager_class.__name__, manager_kwargs))

    for manager_key, attn_type, manager_name, manager_kwargs in manager_infos:
        logger.info(
            "[KVCacheManager] manager_key=%s, attn_type=%s -> manager=%s, manager_kwargs=%s",
            manager_key,
            attn_type,
            manager_name,
            manager_kwargs,
        )
    return single_type_managers
