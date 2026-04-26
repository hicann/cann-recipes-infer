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

from typing import List, Sequence, Dict

from .single_type_kv_cache_manager import SingleTypeKVCacheManager


class KVCacheManager:
    """Coordinates all single-type KV cache managers for one worker."""

    def __init__(
        self,
        max_model_len: int,
        single_type_managers: Sequence[SingleTypeKVCacheManager],
    ):
        self.max_model_len = max_model_len
        self.single_type_managers = tuple(single_type_managers)
        self.num_attn_types = len(self.single_type_managers)

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
        blocks_by_type = {
            manager.attn_type: manager.get_blocks(request_id)
            for manager in self.single_type_managers
        }
        return blocks_by_type

    def free(self, request_id: int) -> None:
        """Release all cache blocks associated with a request."""
        for manager in self.single_type_managers:
            manager.free(request_id)
