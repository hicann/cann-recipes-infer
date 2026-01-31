# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/modules/position_encoder.py
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict
from math import sqrt
from typing import Optional
import torch
import torch.nn.functional as F
from torch.fx._symbolic_trace import is_fx_tracing


@torch.fx.wrap
def _get_high_inds(
    high_inds: torch.Tensor,
    position_embeddings_weight: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    max_pos_ind = position_embeddings_weight.size(0)
    if num_targets is not None:
        if interleave_targets:
            high_inds = high_inds - num_targets * 2
        else:
            high_inds = high_inds - num_targets
    high_inds = torch.clamp(high_inds, max=max_pos_ind - 1)
    return high_inds


def _ensure_1d_long(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1D, git shape={tuple(x.shape)}")
    if x.dtype != torch.long:
        x = x.to(dtype=torch.long)
    return x


@torch.fx.wrap
def torch_add_position_embeddings(
    jagged: torch.Tensor,
    jagged_offsets: torch.Tensor,
    high_inds: torch.Tensor,
    max_seq_len: int,
    dense: torch.Tensor,
    scale: float = 1.0,
    ind_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if jagged.dim() != 2 or dense.dim() != 2:
        raise ValueError(f"jagged and dense must be 2D. got jagged={jagged.dim()}D, dense={dense.dim()}D")
    
    jagged = jagged.contiguous()
    dense = dense.contiguous()
    device = jagged.device
    
    if jagged_offsets.device != device or high_inds.device != device or dense.device != device:
        raise ValueError("All tensors must be on the same device")
    offsets = _ensure_1d_long(jagged_offsets, "jagged_offsets")
    high_inds = _ensure_1d_long(high_inds, "high_inds")
    
    if ind_offsets is not None:
        if ind_offsets.device != device:
            raise ValueError("ind_offsets must be on the same device as jagged")
        ind_offsets = _ensure_1d_long(ind_offsets, "ind_offsets")
    
    total_tokens, dim = jagged.shape
    num_positions, dense_dim = dense.shape
    if dense_dim != dim:
        raise ValueError(f"dense.shape[1] must match jagged.shape[1]. got {dense_dim} vs {dim}")
    
    batch_size = high_inds.numel()
    if offsets.numel() != batch_size + 1:
        raise ValueError(f"jagged_offsets must have length B+1. got {offsets.numel()} vs {batch_size+1}")
    lengths = offsets[1:] - offsets[:-1]
    if lengths.sum().item() != total_tokens:
        raise ValueError(f"sum{lengths} must equal L. got {lengths.sum().item()} vs {total_tokens}")
    if total_tokens == 0:
        return jagged.clone()
    
    # Map each token row -> its batch id (shape: total_tokens,)
    batch_ids = torch.repeat_interleave(torch.arange(batch_size, device=device, dtype=torch.long), lengths)
    row_ids = torch.arange(total_tokens, device=device, dtype=torch.long)

    rel_pos = row_ids - offsets[batch_ids]
    # Per-token max allowed position index (clamped to be non-negative)
    max_ind_row = high_inds[batch_ids].clamp_min(0)
    
    if ind_offsets is None:
        pos_idx = rel_pos
    else:
        pos_idx = rel_pos + ind_offsets[batch_ids]
    
    # Clamp to [0, max_pos] first, then to the embedding table range [0, num_positions-1]
    pos_idx = torch.minimum(pos_idx, max_ind_row).clamp_min(0)
    pos_idx = pos_idx.clamp_max(num_positions - 1)

    # Lookup positional embeddings and add to jagged tokens
    dense_add = F.embedding(pos_idx, dense)

    return jagged * scale + dense_add


@torch.fx.wrap
def torch_add_timestamp_positional_embeddings(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
    num_time_buckets: int = 2048,
    time_bucket_increments: float = 60.0,
    time_bucket_scale: float = 1.0,
    time_delta: float = 0.0,
) -> torch.Tensor:
    if seq_embeddings.dim() != 2:
        raise ValueError("seq_embeddings must be 2D(L, D)")
    if pos_embeddings.dim() != 2 or ts_embeddings.dim() != 2:
        raise ValueError("pos_embeddings and ts_embeddings must be 2D")
    if timestamps.dim() != 1:
        raise ValueError("timestamps must be 1D(L, )")

    seq_embeddings = seq_embeddings.contiguous()
    pos_embeddings = pos_embeddings.contiguous()
    ts_embeddings = ts_embeddings.contiguous()
    timestamps = timestamps.contiguous()

    device = seq_embeddings.device
    if any(
        t.device != device
        for t in (seq_offsets, pos_embeddings, ts_embeddings, timestamps, seq_lengths)
    ):
        raise ValueError("All tensors must be on the same device")
    
    offsets = _ensure_1d_long(seq_offsets, "seq_offsets")
    seq_lengths = _ensure_1d_long(seq_lengths, "seq_lengths")

    if num_targets is not None:
        if num_targets.device != device:
            raise ValueError("num_targets must be on the same device")
        num_targets = _ensure_1d_long(num_targets, "num_targets")
    
    total_tokens, dim = seq_embeddings.shape
    batch_size = seq_lengths.numel()

    if offsets.numel() != batch_size + 1:
        raise ValueError(f"seq_offsets must have length B+1. got {offsets.numel()} vs {batch_size + 1}")
    if timestamps.numel() != total_tokens:
        raise ValueError(f"timestamp must have length L. got {timestamps.numel()} vs {total_tokens}")
    if pos_embeddings.shape[1] != dim or ts_embeddings.shape[1] != dim:
        raise ValueError("pos_embeddings/ts_embeddings last dim must match seq_embeddings")
    
    lengths_from_offsets = offsets[1:] - offsets[:-1]

    if lengths_from_offsets.sum().item() != total_tokens:
        raise ValueError(f"sum(offsets diff) must equal L. got {lengths_from_offsets.sum().item()} vs {total_tokens}")
    if total_tokens == 0:
        return seq_embeddings.clone()

    # For each token row, compute its batch id and its relative position within that batch
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long),
        lengths_from_offsets,
    )

    row_ids = torch.arange(total_tokens, device=device, dtype=torch.long)
    rel_pos = row_ids - offsets[batch_ids]

    seq_len_pre_batch = seq_lengths.clamp_min(0)
    if num_targets is not None:
        targets_per_batch = num_targets.clamp_min(0)
        high_ind_per_batch = seq_len_pre_batch - (targets_per_batch * 2 if interleave_targets else targets_per_batch)
    else:
        high_ind_per_batch = seq_len_pre_batch
    
    high_ind_per_batch = high_ind_per_batch.clamp_min(0)
    high_ind = high_ind_per_batch[batch_ids]

    # Build positional index into pos_embeddings.
    # Logic preserved from original:
    #  - cap n by high_ind
    #  - transform to "distance-to-boundary" indexing with max_contextual_seq_len offset
    #  - for early tokens (n < max_contextual_seq_len), use n directly
    max_pos_ind = pos_embeddings.shape[0]
    pos_ind = torch.where(rel_pos < high_ind, rel_pos, high_ind)
    pos_ind = high_ind - pos_ind + int(max_contextual_seq_len)
    pos_ind = pos_ind.clamp_max(max_pos_ind - 1)
    pos_ind = torch.where(rel_pos < int(max_contextual_seq_len), rel_pos, pos_ind)
    pos_ind = pos_ind.clamp_min(0).clamp_max(max_pos_ind - 1)

    # Query time for each batch: timestamp of the last token in that batch
    seq_end = offsets[1:]
    query_time_per_batch = timestamps[seq_end - 1]
    query_time = query_time_per_batch[batch_ids]

    token_time = timestamps.to(dtype=torch.float32)
    qt = query_time.to(dtype=torch.float32)
    dt = qt - token_time + float(time_delta)
    dt = torch.clamp_min(dt, 1e-6) / float(time_bucket_increments)

    if time_bucket_fn == "log":
        dt = torch.log(dt)
    else:
        dt = torch.sqrt(dt)
    
    dt = dt * float(time_bucket_scale)
    ts_ind = dt.to(dtype=torch.long)
    ts_ind = ts_ind.clamp_min(0).clamp_max(int(num_time_buckets))

    # Clamp to ts embedding table size
    max_ts_ind = int(ts_embeddings.shape[0]) - 1
    ts_ind = ts_ind.clamp_max(max_ts_ind)

    pos_add = F.embedding(pos_ind, pos_embeddings)
    ts_add = F.embedding(ts_ind, ts_embeddings)

    return seq_embeddings + (pos_add + ts_add).to(dtype=seq_embeddings.dtype)


class HSTUPositionalEncoder(torch.nn.Module):
    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        training_dtype: torch.dtype,
        is_inference: bool = True,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._training_dtype = training_dtype
        self._use_time_encoding: bool = use_time_encoding
        self._embedding_dim: int = embedding_dim
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        if self._use_time_encoding:
            self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                    -sqrt(1.0 / num_time_buckets),
                    sqrt(1.0 / num_time_buckets),
                ),
            )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        seq_timestamps: Optional[torch.Tensor] = None,
        seq_start_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self._embedding_dim ** 0.5
        if self._use_time_encoding:
            seq_embeddings = seq_embeddings * alpha
            seq_embeddings = torch_add_timestamp_positional_embeddings(
                seq_embeddings=seq_embeddings,
                seq_offsets=seq_offsets,
                pos_embeddings=self._position_embeddings_weight,
                ts_embeddings=self._timestamp_embeddings_weight,
                timestamps=seq_timestamps,
                max_seq_len=max_seq_len,
                max_contextual_seq_len=0,
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                interleave_targets=False,
                time_bucket_fn="sqrt",
            )
        elif not self._is_inference or seq_start_position is None:
            high_inds = _get_high_inds(
                seq_lengths, self._position_embeddings_weight, num_targets, False
            )
            if not is_fx_tracing():
                _, hidden_dim = seq_embeddings.shape
                torch._assert(
                    seq_offsets.size(0) - 1 == high_inds.size(0),
                    "wrong jagged_offsets shape[0]",
                )
                _, hidden_dim_weight = self._position_embeddings_weight.shape
                torch._assert(hidden_dim_weight == hidden_dim, "wrong dense shape[1]")

            seq_embeddings = torch_add_position_embeddings(
                jagged=seq_embeddings,
                jagged_offsets=seq_offsets,
                high_inds=high_inds,
                max_seq_len=max_seq_len,
                dense=self._position_embeddings_weight,
                scale=alpha,
            )
        else:  # use position embeddings and inference
            ind_offsets = seq_start_position
            high_inds = _get_high_inds(
                seq_lengths + seq_start_position,
                self._position_embeddings_weight,
                num_targets,
                False,
            )
            if not is_fx_tracing():
                _, hidden_dim = seq_embeddings.shape
                torch._assert(
                    seq_offsets.size(0) - 1 == high_inds.size(0),
                    "wrong jagged_offsets shape[0]",
                )
                _, hidden_dim_weight = self._position_embeddings_weight.shape
                torch._assert(hidden_dim_weight == hidden_dim, "wrong dense shape[1]")
            seq_embeddings = torch_add_position_embeddings(
                jagged=seq_embeddings,
                jagged_offsets=seq_offsets,
                high_inds=high_inds,
                max_seq_len=max_seq_len,
                dense=self._position_embeddings_weight,
                scale=alpha,
                ind_offsets=ind_offsets,
            )
        return seq_embeddings