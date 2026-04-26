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

"""Paged-attention memory cache module."""

from .block_pool import BlockPool
from .cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from .cache_utils import (
    allocate_cache_tensors,
    calculate_block_num,
    dtype_itemsize,
    prepare_block_tables,
    prepare_slot_mapping,
    validate_cache_info,
)
from .kv_cache_manager import KVCacheManager
from .single_type_kv_cache_manager import (
    ATTN_TYPE_MANAGER_MAP,
    FullAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
    create_single_type_managers,
)

__all__ = [
    "ATTN_TYPE_MANAGER_MAP",
    "BlockPool",
    "CacheEntry",
    "FullAttentionManager",
    "KVCacheManager",
    "LayerCacheInfo",
    "ModelCacheInfo",
    "allocate_cache_tensors",
    "calculate_block_num",
    "dtype_itemsize",
    "prepare_block_tables",
    "prepare_slot_mapping",
    "validate_cache_info",
    "SingleTypeKVCacheManager",
    "SlidingWindowManager",
    "create_single_type_managers",
]
