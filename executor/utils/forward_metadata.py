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

from dataclasses import dataclass, replace
from typing import Optional, Dict
import torch


@dataclass
class PrefillCPMetaData:
    """Context parallel metadata for prefill-only packed TND inputs.

    The framework pads prefill requests for CP and prepares both temporary
    compute-cache metadata and persistent-cache metadata. The model selects this
    rank's local padded tokens before embedding, then restores gathered outputs
    back to global padded order and drops CP padding.
    """
    # Basic CP topology.
    enabled: bool = False
    cp_size: int = 1

    # Total number of tokens after request-level CP padding.
    global_padded_token_num: int = 0
    # Number of local tokens entering model compute, including CP padding tokens.
    local_token_num: int = 0

    # Select this rank's local padded compute tokens from padded global packed
    # TND tensors.
    local_indices: Optional[torch.Tensor] = None

    # Restore concatenated [rank0_padded, rank1_padded, ...] all-gather output
    # into global zigzag padded token order.
    restore_indices: Optional[torch.Tensor] = None

    # Real-token indices in restored padded CP order. Used to drop padding
    # after output restore, and to select real tokens from padded compute KV.
    global_valid_indices: Optional[torch.Tensor] = None

    # Temporary CP compute slot mapping. It is local to prefill CP and must not
    # be treated as persistent request ownership.
    global_slot_mapping: Optional[Dict[str, torch.Tensor]] = None

    # Temporary CP compute block table used by prefill attention/indexer after
    # CP all-gather. Persistent KV ownership still follows local owner slots.
    global_block_table: Optional[Dict[str, torch.Tensor]] = None

    # Real-token indices in restored padded CP order that should be persisted
    # into KV/indexer cache for the current mode.
    persistent_valid_indices: Optional[torch.Tensor] = None

    # Persistent slot mapping paired with persistent_valid_indices.
    persistent_slot_mapping: Optional[Dict[str, torch.Tensor]] = None

    # Request rows owned by this rank for logits, next_tokens, request updates
    # and MTP next-token restoration.
    output_request_indices: Optional[torch.Tensor] = None

    # Cumulative query lengths for the two zigzag attention halves on this rank.
    actual_seq_q_prev: Optional[torch.Tensor] = None
    actual_seq_q_next: Optional[torch.Tensor] = None

    # KV lengths visible to the prev/next attention calls.
    kv_len_prev: Optional[torch.Tensor] = None
    kv_len_next: Optional[torch.Tensor] = None

    # Local padded compute token counts in the prev/next halves.
    local_prev_token_num: int = 0
    local_next_token_num: int = 0


@dataclass
class ForwardMetaData:
    """Metadata passed during model forward pass"""
    is_prefill: bool = False
    attention_mask: Optional[torch.Tensor] = None
    kv_len: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_cu_kv: Optional[torch.Tensor] = None
    actual_seq_lengths_cu_q: Optional[torch.Tensor] = None
    actual_seq_lengths_cu_list_kv: Optional[list] = None
    actual_seq_lengths_cu_list_q: Optional[list] = None
    actual_seq_lengths_list_kv: Optional[list] = None
    actual_seq_lengths_list_q: Optional[list] = None
    prompt_tokens: int = 0
    block_table: Optional[Dict[str, torch.Tensor]] = None
    slot_mapping: Optional[Dict[str, torch.Tensor]] = None
    cp_metadata: Optional[PrefillCPMetaData] = None

_forward_metadata = ForwardMetaData()


def get_forward_metadata():
    return _forward_metadata


def set_forward_metadata(**kwargs):
    global _forward_metadata
    _forward_metadata = replace(_forward_metadata, **kwargs)


def reset_forward_metadata():
    global _forward_metadata
    _forward_metadata = ForwardMetaData()
