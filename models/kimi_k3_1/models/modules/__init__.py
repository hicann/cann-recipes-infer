# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from .attention_data import (
    AttnMetaData,
    CacheData,
    build_dsa_indexer_metadata,
    build_dspark_mla_slot_mapping,
    build_paged_slot_mapping,
    gather_cp_sp_shards_to_owner,
    gather_sp_shards_to_owner,
    validate_mega_kda_replayssm_switch,
)
from .parallel_utils import (
    all_gather_first_dim,
    distributed_argmax,
    dp_to_tp_all_to_all,
    reduce_scatter_first_dim,
    vocab_tp_to_owner,
)
from .parallel_layout import request_owner_coordinates
from .zigzag_cp import get_zigzag_layout

__all__ = [
    "AttnMetaData",
    "CacheData",
    "build_dsa_indexer_metadata",
    "build_dspark_mla_slot_mapping",
    "build_paged_slot_mapping",
    "gather_cp_sp_shards_to_owner",
    "gather_sp_shards_to_owner",
    "validate_mega_kda_replayssm_switch",
    "all_gather_first_dim",
    "distributed_argmax",
    "dp_to_tp_all_to_all",
    "reduce_scatter_first_dim",
    "vocab_tp_to_owner",
    "request_owner_coordinates",
    "get_zigzag_layout",
]
