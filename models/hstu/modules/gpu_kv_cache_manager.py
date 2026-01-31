# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/modules/gpu_kv_cache_manager.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from enum import Enum
from typing import Tuple
import torch
import torch_npu
from configs import InferenceHSTUConfig, KVCacheConfig, KVCacheMetadata
from modules.kv_cache_manager_impl import KVCacheManagerImpl


class DataType(Enum):
    BF16 = "bf16"
    HALF = "fp16"
    FLOAT = "fp32"


@torch.no_grad()
def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if append_indptr.dim() != 1:
        raise ValueError(f"append_indptr must be 1D, got shape {append_indptr.shape}")
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1D, got shape {seq_lens.shape}")
    
    if append_indptr.dtype != torch.int32:
        append_indptr = append_indptr.to(torch.int32)
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    
    device = append_indptr.device
    batch_size = append_indptr.numel() - 1
    if batch_size != seq_lens.numel():
        raise ValueError(
            f"seq_lens.shape[0] ({seq_lens.numel()}) != append_indptr.shape[0] - 1 ({batch_size})"
        )
    
    total_nnz = int(append_indptr[-1].item())
    if nnz != total_nnz:
        raise ValueError(
            f"nnz ({nnz}) != append_indptr[-1] ({total_nnz}), data inconsistent"
        )
    
    batch_indices = torch.empty(nnz, dtype=torch.int32, device=device)
    positions = torch.empty(nnz, dtype=torch.int32, device=device)

    for b in range(batch_size):
        start = int(append_indptr[b].item())
        end = int(append_indptr[b + 1].item())
        if end <= start:
            continue

        seq_len_b = int(seq_lens[b].item())
        offset = seq_len_b - end

        idx_range = torch.arange(start, end, dtype=torch.int32, device=device)
        batch_indices[start:end] = b
        positions[start:end] = idx_range + offset
    return batch_indices, positions


class HSTUGpuKVCacheManager:
    def __init__(
        self, hstu_config: InferenceHSTUConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        self.num_layers = hstu_config.num_layers
        self.head_dim = hstu_config.head_dim
        self.page_size = kv_cache_config.page_size
        self.num_cache_pages = kv_cache_config.blocks_in_primary_pool
        self.max_batch_size = kv_cache_config.max_batch_size
        self.max_seq_len = kv_cache_config.max_seq_len
        if kv_cache_config.max_attention_window is None:
            self.max_attention_window = kv_cache_config.max_seq_len
        else:
            self.max_attention_window = min(
                self.max_seq_len, kv_cache_config.max_attention_window
            )

        if self.page_size not in (32, 64):
            raise ValueError(
                f"Unsupported NPU KV-cache page size: {self.page_size}. "
                "Current paged HSTU attention kernel only support page size = 32 or 64"
            )

        max_pages_per_batch = math.ceil(
            self.max_batch_size * self.max_seq_len / self.page_size
        )
        if not (max_pages_per_batch < self.num_cache_pages):
            raise ValueError(
                "The number of pages in NPU KVCache is {0}, smaller than the potential "
                "maximum number of pages required by a single batch: #MAX_PAGES = {1} "
                "x {2} x 2 / {3} = {4}.".format(
                    self.num_cache_pages,
                    self.max_batch_size,
                    self.max_seq_len,
                    self.page_size,
                    max_pages_per_batch,
                )
            )

        self.num_heads_per_layer = [
            hstu_config.num_heads for _ in range(self.num_layers)
        ]
        self.max_attention_window_vec = [
            self.max_attention_window for _ in range(self.num_layers)
        ]
        self.num_reserved_cache_pages = math.ceil(
            self.max_batch_size * self.max_seq_len / self.page_size
        )
        self.offload_chunksize = kv_cache_config.offload_chunksize

        self._onload_stream = torch_npu.npu.Stream()
        self._offload_stream = torch_npu.npu.Stream()
        self._offload_start_event = torch_npu.npu.Event()
        self._offload_end_event = torch_npu.npu.Event()

        kwargs = {
            "num_kv_heads_per_layer": self.num_heads_per_layer,
            "size_per_head": self.head_dim,
            "tokens_per_block": self.page_size,
            "blocks_in_primary_pool": self.num_cache_pages,
            "blocks_in_secondary_pool": 0,
            "max_num_sequences": 1024,  # not to be confused with self.max_batch_size
            "max_beam_width": 1,
            "max_attention_window_vec": self.max_attention_window_vec,
            "temporary_attention_window": 0,
            "sink_token_length": 0,
            "stream": self._offload_stream,
            "max_sequence_length": self.max_seq_len,
            "enable_block_reuse": False,
            "onboard_blocks": True,
            "cache_type": None,
            "secondary_offload_min_priority": None,
            "event_manager": None,
            "enable_partial_reuse": True,
            "copy_on_partial_reuse": True,
            "reserved_blocks_in_primary_pool": self.num_reserved_cache_pages,
        }
        self.impl = KVCacheManagerImpl(**kwargs)

        self.dtype = (
            torch.bfloat16
            if hstu_config.bf16
            else torch.float16
            if hstu_config.fp16
            else torch.float32
        )
        kv_cache_dtype = (
            DataType.BF16
            if hstu_config.bf16
            else DataType.HALF
            if hstu_config.fp16
            else DataType.FLOAT
        )
        self.impl.allocate_pools(kv_cache_dtype, False)
        self._offload_kvdata_host_buffers = [
            torch.empty(
                (
                    self.num_reserved_cache_pages,
                    2,
                    self.page_size,
                    self.num_heads_per_layer[i],
                    self.head_dim,
                ),
                dtype=self.dtype,
                pin_memory=True,
            )
            for i in range(self.num_layers)
        ]

    def allocate(
        self,
        user_ids: torch.Tensor,
        user_start_pos: torch.Tensor,
        new_history_lengths: torch.Tensor,
    ):
        self._offload_end_event.wait()
        batch_size = len(user_ids)
        user_ids_set = set(user_ids.int().tolist())
        for idx in range(batch_size):
            user_id = user_ids[idx].item()
            start_pos = user_start_pos[idx].item()
            new_history_length = new_history_lengths[idx].item()
            self.impl.add_sequence_with_eviction(
                user_id, start_pos, new_history_length, 1, user_ids_set
            )

    def evict(self, user_ids: torch.Tensor):
        for user_id in user_ids:
            self.impl.remove_sequence(user_id.item(), None)

    def evict_all(self):
        self.impl.evict_all_sequences()

    def get_user_kvdata_info(self, user_id: int) -> Tuple[int, int]:
        cached_start_pos = self.impl.get_cached_start_position(
            user_id
        )  # (-1) for not in cache
        cached_length = self.impl.get_num_tokens_cached(user_id)
        return (cached_start_pos, cached_length)

    def get_batch_kvdata_info(
        self, user_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = user_ids.shape[0]
        cached_start_pos = torch.tensor(
            [
                self.impl.get_cached_start_position(user_ids[idx].item())
                for idx in range(batch_size)
            ],
            dtype=torch.int32,
        )
        cached_lengths = torch.tensor(
            [
                self.impl.get_num_tokens_cached(user_ids[idx].item())
                for idx in range(batch_size)
            ],
            dtype=torch.int32,
        )
        return (cached_start_pos, cached_lengths)

    def offload(
        self,
        user_ids: torch.Tensor,
        host_start_pos: torch.Tensor,
        host_lengths: torch.Tensor,
        kvcache_metadata: KVCacheMetadata,
    ):
        offload_results = self.offload_async(
            user_ids, host_start_pos, host_lengths, kvcache_metadata
        )
        self.offload_wait()
        return offload_results

    def offload_wait(self):
        self._offload_end_event.wait()

    def offload_async(
        self,
        user_ids: torch.Tensor,
        host_start_pos: torch.Tensor,
        host_lengths: torch.Tensor,
        kvcache_metadata: KVCacheMetadata,
    ):
        batch_size = len(user_ids)
        pages_per_chunk = self.offload_chunksize // self.page_size

        offload_user_ids = []
        offload_start_pos = []
        offload_page_indptr = [0]

        page_ids_to_offload = []
        for idx in range(batch_size):
            uid = user_ids[idx].item()
            cur_offloaded_start_pos, cur_offloaded_length = (
                host_start_pos[idx],
                host_lengths[idx],
            )
            cached_start_pos, cached_length = self.get_user_kvdata_info(uid)

            new_offload_start_pos = cur_offloaded_start_pos + cur_offloaded_length

            new_offload_length = max(
                0, (cached_start_pos + cached_length) - new_offload_start_pos
            )
            new_offload_chunks = new_offload_length // self.offload_chunksize
            if new_offload_chunks == 0:
                continue
            new_offload_start_page_idx = (
                kvcache_metadata.kv_indptr[idx]
                + new_offload_start_pos // self.page_size
            )
            new_offload_end_page_idx = (
                new_offload_start_page_idx + new_offload_chunks * pages_per_chunk
            )
            new_offload_length = new_offload_chunks * self.offload_chunksize

            offload_page_ids = kvcache_metadata.kv_indices[
                new_offload_start_page_idx:new_offload_end_page_idx
            ]

            num_pages = len(offload_page_ids)
            if num_pages > 0:
                offload_user_ids.append(uid)
                offload_start_pos.append(new_offload_start_pos)
                page_ids_to_offload.append(offload_page_ids)
                offload_page_indptr.append(offload_page_indptr[-1] + num_pages)

        offload_user_ids = torch.tensor(offload_user_ids).long()
        offload_start_pos = torch.tensor(offload_start_pos, dtype=torch.int32)
        offload_page_indptr = torch.tensor(offload_page_indptr, dtype=torch.int32)

        num_offload_user = len(offload_user_ids)
        if num_offload_user == 0:
            return None

        device = torch_npu.npu.current_device()

        page_ids_offload = (
            page_ids_to_offload[0]
            if num_offload_user == 1
            else torch.cat(page_ids_to_offload, dim=0)
        )
        num_pages = page_ids_offload.shape[0]
        with torch_npu.npu.stream(self._offload_stream):
            self._offload_start_event.wait(self._offload_stream)
            if page_ids_offload.device != device:
                page_ids = page_ids_offload.to(device, non_blocking=True)
            else:
                page_ids = page_ids_offload
            if page_ids.dtype != torch.long:
                page_ids = page_ids.to(torch.long)
            primary_pool = self.impl.get_primary_pool()
            for layer_idx in range(self.num_layers):
                layer_pool = primary_pool[layer_idx]
                bs = layer_pool.shape[1]
                num_heads = self.num_heads_per_layer[layer_idx]

                base = layer_pool.view(2 * bs, -1)
                k_idx = page_ids
                v_idx = page_ids + bs
                all_idx = torch.cat((k_idx, v_idx), dim=0)

                selected = base.index_select(0, all_idx)
                selected = selected.view(
                    2,
                    num_pages,
                    self.page_size,
                    num_heads,
                    self.head_dim,
                )

                selected = selected.permute(1, 0, 2, 3, 4)
                self._offload_kvdata_host_buffers[layer_idx][:num_pages, ...].copy_(
                    selected, non_blocking=True
                )
            self._offload_end_event.record(self._offload_stream)
        return (
            self._offload_kvdata_host_buffers,
            offload_user_ids,
            offload_start_pos,
            offload_page_indptr,
        )

    def onload(self, host_kv_data: torch.Tensor, onload_length: int, kv_cache_metadata):
        if onload_length == 0:
            return
        onload_num_pages = onload_length // self.page_size
        with torch_npu.npu.stream(self._onload_stream):
            for layer_idx in range(self.num_layers):
                dst = kv_cache_metadata.onload_history_kv_buffer[layer_idx]
                dst[0, :onload_num_pages].copy_(host_kv_data[layer_idx, :onload_num_pages, 0], non_blocking=True)
                dst[1, :onload_num_pages].copy_(host_kv_data[layer_idx, :onload_num_pages, 1], non_blocking=True)
                kv_cache_metadata.onload_history_kv_events[layer_idx].record(
                    self._onload_stream
                )

    def get_cache_metadata(self, user_ids: torch.Tensor) -> "KVCacheMetadata":
        batch_size = len(user_ids)

        kv_cache_page_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=torch.device("cpu")
        )
        kv_cache_last_page_lengths = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cpu")
        )
        cached_history_lengths = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cpu")
        )

        user_ids_list = user_ids.tolist()
        total_history_lengths = torch.tensor(
            [
                self.impl.get_num_tokens_cached(uid)
                + self.impl.get_cached_start_position(uid)
                for uid in user_ids_list
            ],
            dtype=torch.int32,
        )
        kv_page_ids = [
            torch.tensor(self.impl.get_cache_block_ids(uid)[0], dtype=torch.int32)
            for uid in user_ids_list
        ]
        kv_num_pages = torch.tensor(
            [page_ids.shape[0] for page_ids in kv_page_ids], dtype=torch.int32
        )

        kv_page_indices = torch.cat(kv_page_ids, dim=0)
        kv_page_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32)
        torch.cumsum(kv_num_pages, 0, out=kv_page_indptr[1:])
        kv_last_page_len = torch.remainder(total_history_lengths, self.page_size)
        kv_last_page_len[kv_last_page_len == 0] = self.page_size

        total_history_offsets = torch.zeros((batch_size + 1,), dtype=torch.int32)
        torch.cumsum(total_history_lengths, 0, out=total_history_offsets[1:])

        return KVCacheMetadata(
            kv_indices=kv_page_indices.npu(),
            kv_indptr=kv_page_indptr.npu(),
            kv_last_page_len=kv_last_page_len.npu(),
            total_history_lengths=total_history_lengths.npu(),
            total_history_offsets=total_history_offsets.npu(),
        )

    @staticmethod
    def get_append_metadata(
        new_history_lengths: torch.Tensor, total_history_lengths: torch.Tensor
    ) -> "KVCacheMetadata":
        batch_size = total_history_lengths.shape[0]

        new_history_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=new_history_lengths.device
        )
        torch.cumsum(new_history_lengths, 0, out=new_history_offsets[1:])

        new_history_token_nnz = new_history_offsets[-1].item()

        (
            history_batch_indices,
            history_positions,
        ) = get_batch_indices_positions(
            new_history_offsets.npu(),
            total_history_lengths.npu(),
            new_history_token_nnz,
        )
        new_history_nnz_cuda = torch.tensor(
            [new_history_token_nnz], dtype=torch.int32
        ).npu()

        return KVCacheMetadata(
            batch_indices=history_batch_indices,
            position=history_positions,
            new_history_nnz=new_history_token_nnz,
            new_history_nnz_cuda=new_history_nnz_cuda,
        )

    def get_cached_length(self, user_id: int) -> int:
        return self.impl.get_num_tokens_cached(user_id)

    def get_page_size(self) -> int:
        return self.page_size

    def get_kvcache_table(self, layer_idx: int) -> torch.Tensor:
        pool = self.impl.get_primary_pool()

        layer_pool = pool[layer_idx]

        total_page = layer_pool.shape[1]

        num_heads = self.num_heads_per_layer[layer_idx]
        kv = layer_pool.view(
            2, total_page, self.page_size, num_heads, self.head_dim
        )
        return kv

    def get_onload_buffers(self, layer_idx: int) -> torch.Tensor:
        pool = self.impl.get_primary_pool()
        layer_pool = pool[layer_idx]

        reserved = layer_pool[:, self.num_cache_pages:, :]
        return reserved.view(
            2, reserved.shape[1], self.page_size,
            self.num_heads_per_layer[layer_idx], self.head_dim
        )
