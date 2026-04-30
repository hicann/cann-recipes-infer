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

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch


@dataclass
class CacheEntry:
    """Single cache entry metadata."""

    cache_name: str
    attn_type: str
    dim: Union[int, List[int]]
    num_head: int
    dtype: torch.dtype
    needs_block: bool
    tensor_setter: Optional[Callable[[torch.Tensor], None]] = None
    sliding_window: Optional[int] = None
    tensor: Optional[torch.Tensor] = None


@dataclass
class LayerCacheInfo:
    """Cache metadata for one transformer layer."""

    layer_idx: int
    caches: List[CacheEntry]


@dataclass
class ModelCacheInfo:
    """Whole-model cache metadata."""

    num_layers: int
    block_size: int
    layer_infos: List[LayerCacheInfo]
    # True for MLA backends (latent KV replicated across TP ranks). Set
    # explicitly by the model's get_cache_info(); PD transfer uses it to
    # pick a single target TP rank and mark the rest as dummy.
    # Do NOT infer from num_head==1 — GQA with num_kv_heads<=tp_size also
    # yields per-rank num_head==1 but is not MLA.
    is_mla_backend: bool = False

    def merge(self, other: "ModelCacheInfo") -> "ModelCacheInfo":
        """Merge two cache-info objects into one complete model description."""
        if self.block_size != other.block_size:
            raise ValueError(
                f"block_size mismatch: {self.block_size} != {other.block_size}"
            )
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
            block_size=self.block_size,
            layer_infos=merged_layer_infos,
            is_mla_backend=self.is_mla_backend,
        )
