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

"""Small Torch communication helpers for K3 Prefill context parallelism."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist


def _all_gather_first_dim(
    tensor: torch.Tensor, group, cp_size: int
) -> torch.Tensor:
    gathered = tensor.new_empty(
        tensor.shape[0] * cp_size, *tensor.shape[1:]
    )
    dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=group)
    return gathered


def gather_cp_padded_tokens(
    local_tokens: torch.Tensor,
    group,
    cp_size: int,
    restore_indices: torch.Tensor,
    global_valid_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather equal-size contiguous CP shards and remove global tail padding."""
    gathered = _all_gather_first_dim(local_tokens, group, cp_size)
    restored = torch.index_select(gathered, 0, restore_indices)
    return torch.index_select(restored, 0, global_valid_indices)


def gather_kcp_summaries(
    transition: torch.Tensor,
    state_ext: torch.Tensor,
    group,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AllGather rank-local ``(M, S_ext)`` summaries in CP rank order."""
    value_dim = state_ext.shape[-1]
    packed = torch.cat((state_ext, transition), dim=-1).contiguous()
    gathered = _all_gather_first_dim(packed, group, cp_size)
    gathered = gathered.reshape(cp_size, *packed.shape)
    return gathered[..., value_dim:], gathered[..., :value_dim]


def kcp_prefix_scan(
    transitions: torch.Tensor,
    state_exts: torch.Tensor,
    initial_state: torch.Tensor,
    cp_rank: int,
) -> torch.Tensor:
    """Restore one rank's initial state from all preceding KCP summaries."""
    if not 0 <= cp_rank < transitions.shape[0]:
        raise ValueError(
            f"cp_rank must be in [0, {transitions.shape[0]}), got {cp_rank}"
        )
    state = initial_state.float()
    for rank in range(cp_rank):
        state = transitions[rank].float() @ state + state_exts[rank].float()
    return state


def sync_last_cp_state(
    local_state: torch.Tensor,
    group,
    cp_size: int,
    last_nonempty_rank: int | None = None,
) -> torch.Tensor:
    """Replicate the completed state from the final nonempty CP rank."""
    batch = local_state.shape[0]
    gathered = _all_gather_first_dim(local_state, group, cp_size)
    rank = cp_size - 1 if last_nonempty_rank is None else last_nonempty_rank
    return gathered.reshape(cp_size, batch, *local_state.shape[1:])[rank].contiguous()


def temporary_conv_cache_ids(batch: int, device: torch.device) -> torch.Tensor:
    """Return valid temporary slots; causal_conv1d reserves slot zero as null."""
    return torch.arange(1, batch + 1, dtype=torch.int32, device=device)


def previous_conv_state(
    local_qkv: torch.Tensor,
    query_boundaries: Sequence[int],
    state_width: int,
    group,
    cp_size: int,
    cp_rank: int,
    shard_lengths: Sequence[int],
) -> torch.Tensor:
    """Collect the latest convolution history from all preceding CP ranks."""
    batch = len(query_boundaries) - 1
    tails = local_qkv.new_zeros(batch, state_width, local_qkv.shape[-1])
    for request, (start, end) in enumerate(
        zip(query_boundaries, query_boundaries[1:])
    ):
        length = min(state_width, end - start)
        if length:
            tails[request, -length:].copy_(local_qkv[end - length:end])
    gathered = _all_gather_first_dim(tails, group, cp_size)
    gathered = gathered.reshape(cp_size, batch, state_width, local_qkv.shape[-1])
    if cp_rank == 0:
        return torch.zeros_like(tails)
    if len(shard_lengths) != cp_size:
        raise ValueError(
            f"expected {cp_size} CP shard lengths, got {len(shard_lengths)}"
        )

    initial_state = local_qkv.new_zeros(batch, state_width, local_qkv.shape[-1])
    for request in range(batch):
        write_end = state_width
        for source_rank in range(cp_rank - 1, -1, -1):
            available = min(state_width, max(0, int(shard_lengths[source_rank])))
            if not available:
                continue
            take = min(available, write_end)
            initial_state[request, write_end - take:write_end].copy_(
                gathered[
                    source_rank,
                    request,
                    state_width - take:state_width,
                ]
            )
            write_end -= take
            if write_end == 0:
                break
    return initial_state


__all__ = [
    "gather_cp_padded_tokens",
    "gather_kcp_summaries",
    "kcp_prefix_scan",
    "previous_conv_state",
    "sync_last_cp_state",
    "temporary_conv_cache_ids",
]
