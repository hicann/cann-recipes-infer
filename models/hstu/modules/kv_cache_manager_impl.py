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

import math
from typing import Dict, List, Optional, Set, Any
import torch


class KVCacheManagerImpl:
    """
    - add_sequence_with_eviction: only support continuous append (start_pos must be continuous)
    - eviction: LRU
    - self-evict: when newBlocks > maxBlocksPerSeq, evict its own prefix blocks
    - primary_pool: shape [total_blocks, num_layers, 2, block_size]
    """

    def __init__(
        self,
        num_kv_heads_per_layer,
        size_per_head,
        tokens_per_block,
        blocks_in_primary_pool,
        blocks_in_secondary_pool,
        max_num_sequences,
        sink_token_length,
        max_sequence_length,
        reserved_blocks_in_primary_pool: int = 0,
        **unused_kwargs,
    ) -> None:
        self.num_layers = len(num_kv_heads_per_layer)
        self.num_kv_heads_per_layer = list(num_kv_heads_per_layer)
        self.size_per_head = int(size_per_head)
        self.tokens_per_block = int(tokens_per_block)

        self.max_num_sequences = int(max_num_sequences)
        self.max_sequence_length = int(max_sequence_length)
        self.max_blocks_per_seq = math.ceil(self.max_sequence_length / self.tokens_per_block)

        self.sink_token_length = int(sink_token_length or 0)

        # Primary pool and reserved blocks
        self.num_primary_blocks = int(blocks_in_primary_pool)
        self.num_reserved_blocks = int(reserved_blocks_in_primary_pool or 0)
        self.total_blocks = self.num_primary_blocks + self.num_reserved_blocks

        # KV primary pool
        self.primary_pool: Optional[torch.Tensor] = None
        self.secondary_pool: Optional[torch.Tensor] = None

        # request_id -> {"num_tokens": int, "beam_width": int, "block_ids": [int]}
        self.sequences: Dict[int, Dict[str, Any]] = {}
        self.seq_cache_start_pos: Dict[int, int] = {}
        self.cache_block_ids: Dict[int, List[List[int]]] = {}

        # free blocks: only contain primary pool block (0..num_primary_blocks-1)
        self.free_blocks: List[int] = list(range(self.num_primary_blocks))
        self._seq_lru_list: List[int] = []

    @staticmethod
    def _map_dtype(dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        name = getattr(dtype, "name", None)
        if name is None:
            name = str(dtype)
        name = str(name).upper()
        if "BF16" in name:
            return torch.bfloat16
        if "HALF" in name or "FP16" in name:
            return torch.float16
        if "FLOAT" in name or "FP32" in name:
            return torch.float32
        return torch.float16

    @staticmethod
    def _pick_device() -> torch.device:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
        return torch.device("cpu")

    def _lru_remove(self, rid: int) -> None:
        try:
            self._seq_lru_list.remove(rid)
        except ValueError:
            pass

    def _lru_touch(self, rid: int) -> None:
        """move rid to the head of LRU (most recent)"""
        self._lru_remove(rid)
        self._seq_lru_list.insert(0, rid)

    def _lru_insert_new(self, rid: int) -> None:
        self._lru_touch(rid)

    def _get_request_id_to_evict(self, freezed: Set[int]) -> int:
        if not self._seq_lru_list:
            raise RuntimeError("LRU list empty, cannot evict")
        if len(freezed) == 0:
            return self._seq_lru_list[-1]
        for rid in reversed(self._seq_lru_list):
            if rid in freezed:
                continue
            return rid
        raise RuntimeError("Cannot find sequence to evict for more cache space")
    
    def allocate_pools(self, dtype, use_uvm: bool = False):
        torch_dtype = self._map_dtype(dtype)
        num_heads = int(self.num_kv_heads_per_layer[0])
        if not all(int(head) == num_heads for head in self.num_kv_heads_per_layer):
            raise ValueError(
                "allocate_pools currently only supports the same num_kv_heads across all layers"
            )
        block_size = self.tokens_per_block * num_heads * self.size_per_head
        kv_factor = 2

        device = self._pick_device()
        self.primary_pool = torch.empty(
            (self.num_layers, kv_factor, self.total_blocks, block_size),
            dtype=torch_dtype,
            device=device,
        )

        num_sec_blocks = int(getattr(self, 'blocks_in_secondary_pool', 0))
        
        if num_sec_blocks > 0:
            self.secondary_pool = torch.empty(
                (num_sec_blocks, self.num_layers, kv_factor, block_size),
                dtype=torch_dtype,
                device="cpu",
            ).pin_memory()

        return self.primary_pool

    def get_primary_pool(self, pool_idx: int = 0) -> torch.Tensor:
        if self.primary_pool is None:
            raise RuntimeError("KV cache pools have not been allocated. Call allocate_pools() first")
        return self.primary_pool

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)
    
    def _evict_sequence(self, request_id: int) -> None:
        """Release all blocks of a sequence and clean up metadata + LRU"""
        seq = self.sequences.pop(request_id, None)
        if seq is not None:
            for bid in seq["block_ids"]:
                if 0 <= bid < self.num_primary_blocks:
                    self.free_blocks.append(bid)
        self.seq_cache_start_pos.pop(request_id, None)
        self.cache_block_ids.pop(request_id, None)
        self._lru_remove(request_id)

    def _evict_blocks_prefix(self, request_id: int, num_blocks_to_evict: int) -> None:
        """Evict numBlocksToEvict from the prefix of allocatedBlocks"""
        seq = self.sequences[request_id]
        blocks = seq["block_ids"]
        n = min(int(num_blocks_to_evict), len(blocks))
        if n <= 0:
            return

        evicted = blocks[:n]
        del blocks[:n]

        for bid in evicted:
            if 0 <= bid < self.num_primary_blocks:
                self.free_blocks.append(bid)

        # Approximately reduce the number of tokens using block granularity
        dec_tokens = n * self.tokens_per_block
        seq["num_tokens"] = max(0, int(seq["num_tokens"]) - dec_tokens)

        # After the prefix blocks are evicted, the cache start pos must be shifted to the right
        if request_id in self.seq_cache_start_pos:
            self.seq_cache_start_pos[request_id] = int(self.seq_cache_start_pos[request_id]) + dec_tokens

        self.cache_block_ids[request_id] = [blocks]

    def add_sequence_with_eviction(
        self,
        request_id: int,
        start_pos: int,
        length: int,
        beam_width: int,
        freezed_id_group=None,
    ) -> None:
        request_id = int(request_id)
        start_pos = int(start_pos)
        length = int(length)
        beam_width = int(beam_width)
        freezed = set(freezed_id_group or [])

        if length > self.max_blocks_per_seq * self.tokens_per_block:
            raise ValueError(
                f"Do not accept delta length {length}, "
                f"{self.max_blocks_per_seq} * {self.tokens_per_block}"
            )
        
        old_exists = request_id in self.sequences
        if old_exists:
            self._lru_touch(request_id)
            cur_len = int(self.sequences[request_id]["num_tokens"])
            cache_start = int(self.seq_cache_start_pos[request_id])
            expected_start = cache_start + cur_len
            # continuous check
            if start_pos != expected_start:
                raise ValueError(
                    "Currently, KV Cache only supports continuous input sequence. "
                    f"Last seen position {cache_start} to {cache_start + cur_len} -- "
                    f"current input position {start_pos} to {start_pos + length}"
                )
        else:
            cur_len = 0
        
        if (not old_exists) and (len(self.sequences) == self.max_num_sequences):
            victim = self._get_request_id_to_evict(freezed)
            self._evict_sequence(victim)

        new_len = cur_len + length
        cur_blocks = (cur_len + self.tokens_per_block - 1) // self.tokens_per_block if cur_len > 0 else 0
        new_blocks = (new_len + self.tokens_per_block - 1) // self.tokens_per_block if new_len > 0 else 0

        if new_blocks > self.max_blocks_per_seq:
            num_blocks_self_evict = new_blocks - self.max_blocks_per_seq

            if num_blocks_self_evict >= cur_blocks:
                # evict whole sequence
                if old_exists:
                    self._evict_sequence(request_id)
                old_exists = False
                cur_len = 0
            else:
                # evict prefix blocks
                self._evict_blocks_prefix(request_id, num_blocks_self_evict)
                cur_len = int(self.sequences[request_id]["num_tokens"])
            
            # Recalculated and adjusted for the start_pos variable
            new_len = cur_len + length
            cur_blocks = (cur_len + self.tokens_per_block - 1) // self.tokens_per_block if cur_len > 0 else 0
            new_blocks = (new_len + self.tokens_per_block - 1) // self.tokens_per_block if new_len > 0 else 0
            start_pos = start_pos + length - new_len
        
        # ensure enough free blocks: evict others
        num_blocks_required = new_blocks - cur_blocks
        while num_blocks_required > self.get_num_free_blocks():
            victim = self._get_request_id_to_evict(freezed)
            if victim == request_id:
                raise RuntimeError("Cannot evict self to satisfy allocation; check LRU/freezed logic")
            self._evict_sequence(victim)

        # allocate new blocks
        new_block_ids = [self.free_blocks.pop() for _ in range(num_blocks_required)]

        if not old_exists:
            # create new seq
            self.seq_cache_start_pos[request_id] = start_pos
            self.sequences[request_id] = {
                "num_tokens": new_len,
                "beam_width": beam_width,
                "block_ids": [],
            }
            self._lru_insert_new(request_id)
            self.sequences[request_id]["block_ids"].extend(new_block_ids)
            self.cache_block_ids[request_id] = [self.sequences[request_id]["block_ids"]]
        else:
            # append tokens, increase num_tokens, and append block_ids
            seq = self.sequences[request_id]
            seq["num_tokens"] = new_len
            seq["beam_width"] = beam_width
            seq["block_ids"].extend(new_block_ids)
            self.cache_block_ids[request_id] = [seq["block_ids"]]

    def remove_sequence(self, request_id: int, llm_request=None) -> None:
        self._evict_sequence(int(request_id))

    def evict_all_sequences(self) -> None:
        for rid in list(self.sequences.keys()):
            self._evict_sequence(rid)

    def get_cached_start_position(self, request_id: int) -> int:
        return int(self.seq_cache_start_pos.get(int(request_id), -1))

    def get_num_tokens_cached(self, request_id: int) -> int:
        seq = self.sequences.get(int(request_id))
        return int(seq["num_tokens"]) if seq is not None else 0

    def get_cache_block_ids(self, request_id: int):
        return self.cache_block_ids.get(int(request_id), [[]])