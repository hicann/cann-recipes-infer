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

"""Cache metadata structures for paged-attention initialization."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Union

import torch


class CacheAllocator(str, Enum):
    """Physical allocator used for a cache entry."""

    HBM = "hbm"
    SWAPPED_MEMORY = "swapped_memory"


@dataclass
class MemoryBudgetItem:
    """Memory item reported by a model for cache/workspace budgeting."""

    name: str
    bytes: int
    location: str = "npu"


@dataclass
class OffloadWorkspaceMemoryInfo:
    """Structured offload workspace memory report."""

    items: List[MemoryBudgetItem]

    @property
    def npu_bytes(self) -> int:
        return sum(item.bytes for item in self.items if item.location == "npu")


@dataclass
class CacheEntry:
    """Single cache entry metadata."""

    cache_name: str
    attn_type: str
    dim: Union[int, List[int]]
    num_head: int
    dtype: torch.dtype
    needs_block: bool
    block_size: Optional[int] = None
    manager_key: Optional[str] = None
    tensor_setter: Optional[Callable[[torch.Tensor], None]] = None
    sliding_window: Optional[int] = None
    allocator: CacheAllocator = CacheAllocator.HBM
    compress_ratio: int = 1
    tensor: Optional[torch.Tensor] = None
    cache_layout: str = "BnBsND"

    @property
    def group_key(self) -> str:
        """Manager grouping key for cache allocation and metadata tables."""
        return self.manager_key if self.manager_key is not None else self.attn_type

    def cache_dim_numel(self) -> int:
        """Return flattened element count for a cache entry's trailing dim."""
        dims = self.dim if isinstance(self.dim, list) else [self.dim]
        numel = 1
        for cur_dim in dims:
            numel *= cur_dim
        return numel

    @property
    def storage_block_size(self) -> int:
        """Calculate the physical block size when creating cache tensors."""
        if self.block_size % self.compress_ratio > 0:
            raise ValueError(
                "block_size must be divisible by compress_ratio when calculating "
                f"storage_block_size, but got cache_name={self.cache_name}, "
                f"block_size={self.block_size}, compress_ratio={self.compress_ratio}."
            )
        storage_block_size = self.block_size // self.compress_ratio
        return storage_block_size


@dataclass
class MambaCacheEntry(CacheEntry):
    """Cache entry for a fixed-size Mamba-style recurrent state.

    Paged attention entries describe ``num_head x dim`` per token; a recurrent
    state is instead one whole fixed-shape tensor per request, so ``shape``
    carries the complete trailing shape and ``dim``/``num_head`` are unused.

    Construct with keyword arguments only: the overridden fields are keyword-only,
    which shifts the positional order away from the base class.
    """

    attn_type: str = field(default="Mamba", kw_only=True)
    dim: Optional[Union[int, List[int]]] = field(default=None, kw_only=True)
    num_head: Optional[int] = field(default=None, kw_only=True)
    block_size: int = field(default=1, kw_only=True)
    shape: List[int] = field(default_factory=list, kw_only=True)
    # Speculative decoding depth. The manager reserves 1 + next_n state blocks
    # per request; models that support speculative state advancement override it.
    next_n: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        if self.attn_type != "Mamba":
            raise ValueError(
                f"Mamba cache '{self.cache_name}' must keep attn_type='Mamba', "
                f"but got '{self.attn_type}'."
            )
        if self.block_size != 1:
            raise ValueError(
                f"Mamba cache '{self.cache_name}' block_size must be 1, "
                f"but got {self.block_size}."
            )
        if not isinstance(self.shape, list) or len(self.shape) == 0:
            raise ValueError(
                f"Mamba cache '{self.cache_name}' must define a "
                f"non-empty shape list, but got {self.shape}."
            )
        if any(not isinstance(cur_dim, int) or cur_dim <= 0 for cur_dim in self.shape):
            raise ValueError(
                f"Mamba cache '{self.cache_name}' shape dimensions "
                f"must all be positive integers, but got {self.shape}."
            )
        if self.next_n < 0:
            raise ValueError(
                f"Mamba cache '{self.cache_name}' next_n must be "
                f"non-negative, but got {self.next_n}."
            )

    def cache_dim_numel(self) -> int:
        """Return flattened element count of one state block."""
        numel = 1
        for cur_dim in self.shape:
            numel *= cur_dim
        return numel


@dataclass
class LayerCacheInfo:
    """Cache metadata for one transformer layer."""

    layer_idx: int
    caches: List[CacheEntry]


@dataclass
class ModelCacheInfo:
    """Whole-model cache metadata."""

    num_layers: int
    layer_infos: List[LayerCacheInfo]
    # True for MLA backends (latent KV replicated across TP ranks). Set
    # explicitly by the model's get_cache_info(); PD transfer uses it to
    # pick a single target TP rank and mark the rest as dummy.
    # Do NOT infer from num_head==1 — GQA with num_kv_heads<=tp_size also
    # yields per-rank num_head==1 but is not MLA.
    is_mla_backend: bool = False

    def merge(self, other: "ModelCacheInfo") -> "ModelCacheInfo":
        """Merge two cache-info objects into one complete model description."""
        if self.is_mla_backend != other.is_mla_backend:
            raise ValueError(
                "is_mla_backend mismatch across merged cache infos: "
                f"{self.is_mla_backend} vs {other.is_mla_backend}"
            )

        merged_layer_infos = list(self.layer_infos)
        layer_idx_offset = len(merged_layer_infos)
        for layer_info in other.layer_infos:
            layer_info.layer_idx += layer_idx_offset
        merged_layer_infos.extend(other.layer_infos)
        return ModelCacheInfo(
            num_layers=len(merged_layer_infos),
            layer_infos=merged_layer_infos,
            is_mla_backend=self.is_mla_backend,
        )
