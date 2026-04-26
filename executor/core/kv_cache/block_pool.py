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

"""Block pool abstractions for paged-attention KV cache management."""

from collections import deque
from typing import Deque, Iterable, List, Optional


class BlockPool:
    """Owns the global pool of KV cache blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_block_queue: Deque[int] = deque(range(num_blocks))
        self._null_block = self._free_block_queue.popleft()
        self._free_block_set = set(self._free_block_queue)

    def get_num_free_blocks(self) -> int:
        """Return the number of currently available blocks."""
        return len(self._free_block_queue)

    def get_block(self, block_id: int) -> int:
        """Return block metadata by block id."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(f"block_id {block_id} is out of range [0, {self.num_blocks})")
        return block_id

    def get_new_blocks(self, num_blocks: int) -> List[int]:
        """Allocate a number of blocks from the pool."""
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if num_blocks > len(self._free_block_queue):
            raise ValueError(
                f"requested {num_blocks} blocks, but only {len(self._free_block_queue)} are available"
            )

        allocated_blocks: List[int] = []
        for _ in range(num_blocks):
            block_id = self._free_block_queue.popleft()
            self._free_block_set.remove(block_id)
            allocated_blocks.append(block_id)
        return allocated_blocks

    def free_blocks(self, blocks: List[int]) -> None:
        """Recycle blocks back to the pool."""
        for block_id in blocks:
            self.get_block(block_id)

            if block_id == self._null_block:
                continue
            if block_id in self._free_block_set:
                raise ValueError(f"block_id {block_id} is already free")

            self._free_block_queue.append(block_id)
            self._free_block_set.add(block_id)

    def get_null_block(self) -> Optional[int]:
        """Return the optional null block used by sparse managers."""
        return self._null_block
