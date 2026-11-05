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

from __future__ import annotations


def vocab_tp_group_spec(
    world_size: int,
    attn_tp_size: int,
    cp_size: int,
    vocab_tp_size: int,
) -> tuple[int, int]:
    """Return ``(group_num, group_stride)`` for Embedding or LMHead TP.

    Kimi K3 lays out Attention-TP ranks as the minor physical-rank dimension,
    so each CP lane is one contiguous rank domain. During Decode those lanes
    become independent Attention-DP request owners. A vocabulary-TP group
    smaller than the world must therefore stay inside one lane; multiple
    groups in a lane are weight replicas.

    World-sized vocabulary TP remains supported as the legacy layout that
    spans all Decode DP lanes and uses explicit activation adapters.
    """
    for label, value in (
        ("world_size", world_size),
        ("attn_tp_size", attn_tp_size),
        ("cp_size", cp_size),
        ("vocab_tp_size", vocab_tp_size),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    if world_size % attn_tp_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by "
            f"attn_tp_size={attn_tp_size}"
        )
    if world_size % cp_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by cp_size={cp_size}"
        )
    if world_size % vocab_tp_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by "
            f"vocab_tp_size={vocab_tp_size}"
        )
    if cp_size > 1:
        if attn_tp_size * cp_size != world_size:
            raise ValueError(
                "CP requires attn_tp_size * cp_size == world_size"
            )
        if vocab_tp_size != world_size and attn_tp_size % vocab_tp_size:
            raise ValueError(
                f"vocab_tp_size={vocab_tp_size} must equal world_size="
                f"{world_size} or divide attn_tp_size={attn_tp_size}"
            )
    elif attn_tp_size % vocab_tp_size:
        raise ValueError(
            f"vocab_tp_size={vocab_tp_size} must divide "
            f"attn_tp_size={attn_tp_size}"
        )

    return world_size // vocab_tp_size, 1


def vocab_tp_group_ranks(
    global_rank: int,
    world_size: int,
    attn_tp_size: int,
    cp_size: int,
    vocab_tp_size: int,
) -> tuple[int, ...]:
    """Return the ordered global ranks in this rank's vocabulary-TP group."""
    if global_rank < 0 or global_rank >= world_size:
        raise ValueError(
            f"global_rank must be in [0, {world_size}), got {global_rank}"
        )
    vocab_tp_group_spec(
        world_size,
        attn_tp_size,
        cp_size,
        vocab_tp_size,
    )
    start_rank = global_rank // vocab_tp_size * vocab_tp_size
    return tuple(range(start_rank, start_rank + vocab_tp_size))


def rank_coordinates(
    global_rank: int, attn_tp_size: int, cp_size: int
) -> tuple[int, int]:
    """Split a physical rank using the CP-major, Attention-TP-minor layout."""
    if attn_tp_size <= 0 or cp_size <= 0:
        raise ValueError("attn_tp_size and cp_size must be positive")
    if cp_size == 1:
        return global_rank % attn_tp_size, 0
    return global_rank % attn_tp_size, global_rank // attn_tp_size


def request_owner_coordinates(
    request_id: int,
    attn_tp_size: int,
    cp_size: int,
    local_batch: int,
) -> tuple[int, int, int, int]:
    """Return ``(cp_rank, attn_tp_rank, owner_row, global_rank)``.

    CP Prefill assigns requests round-robin to the CP lanes. Decode then treats
    each lane as an independent request-DP replica and partitions that lane's
    rows over Attention TP. Kimi's physical layout is CP-major, TP-minor, so
    the owner rank is reconstructed from the two coordinates.
    """
    for label, value in (
        ("attn_tp_size", attn_tp_size),
        ("cp_size", cp_size),
        ("local_batch", local_batch),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    global_batch = attn_tp_size * cp_size * local_batch
    if request_id < 0 or request_id >= global_batch:
        raise ValueError(
            f"request_id must be in [0, {global_batch}), got {request_id}"
        )

    cp_rank = request_id % cp_size
    lane_request_id = request_id // cp_size
    attn_tp_rank, owner_row = divmod(lane_request_id, local_batch)
    global_rank = cp_rank * attn_tp_size + attn_tp_rank
    return cp_rank, attn_tp_rank, owner_row, global_rank


def decode_global_request_id(
    decode_row: int, cp_rank: int, cp_size: int
) -> int:
    """Map one CP lane's Decode row back to the global Prefill request ID."""
    if decode_row < 0:
        raise ValueError(f"decode_row must be non-negative, got {decode_row}")
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    return decode_row * cp_size + cp_rank


def cp_sp_source_rank_order(
    attn_tp_size: int, cp_size: int
) -> tuple[int, ...]:
    """Return physical source ranks in contiguous sequence-shard order.

    A world collective receives shards in physical CP-major/TP-minor order,
    which is also the padded prompt's contiguous sequence-shard order.
    """
    if attn_tp_size <= 0 or cp_size <= 0:
        raise ValueError("attn_tp_size and cp_size must be positive")
    return tuple(
        cp_rank * attn_tp_size + attn_tp_rank
        for cp_rank in range(cp_size)
        for attn_tp_rank in range(attn_tp_size)
    )


def cp_last_token_location(
    valid_token_num: int,
    cp_local_token_num: int,
    attn_tp_size: int,
) -> tuple[int, int, int]:
    """Return the CP rank, Attention-TP rank and local row of the last token."""
    if valid_token_num <= 0:
        raise ValueError("valid_token_num must be positive")
    if cp_local_token_num <= 0:
        raise ValueError("cp_local_token_num must be positive")
    if attn_tp_size <= 0 or cp_local_token_num % attn_tp_size:
        raise ValueError(
            "cp_local_token_num must be divisible by attn_tp_size"
        )

    tp_local_token_num = cp_local_token_num // attn_tp_size
    last_token = valid_token_num - 1
    cp_rank, cp_offset = divmod(last_token, cp_local_token_num)
    tp_rank, local_offset = divmod(cp_offset, tp_local_token_num)
    return cp_rank, tp_rank, local_offset
