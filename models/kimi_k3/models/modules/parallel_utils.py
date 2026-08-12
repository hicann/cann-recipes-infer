# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tensor-layout conversions shared by the Kimi K3 target and draft models."""

import torch
import torch.distributed as dist


def all_gather_first_dim(
    tensor: torch.Tensor, group, tp_size: int
) -> torch.Tensor:
    """Gather rank-local rows in rank order."""
    if tp_size <= 1:
        return tensor
    gathered = tensor.new_empty(
        tensor.shape[0] * tp_size, *tensor.shape[1:]
    )
    dist.all_gather_into_tensor(
        gathered, tensor.contiguous(), group=group
    )
    return gathered


def reduce_scatter_first_dim(
    tensor: torch.Tensor, group, tp_size: int
) -> torch.Tensor:
    """Sum row-parallel partials and return this rank's row shard."""
    if tp_size <= 1:
        return tensor
    output = tensor.new_empty(
        tensor.shape[0] // tp_size, *tensor.shape[1:]
    )
    dist.reduce_scatter_tensor(
        output, tensor.contiguous(), group=group
    )
    return output


def dp_to_tp_all_to_all(
    tensor: torch.Tensor,
    group,
    tp_size: int,
    output_rows: int,
    channel_width: int,
) -> torch.Tensor:
    """Exchange owner-local tokens for precomputed row-TP channel shards."""
    if tp_size <= 1:
        return tensor
    send = tensor.reshape(-1, tp_size, channel_width)
    send = send.transpose(0, 1).contiguous().view(-1)
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=group)
    return received.view(output_rows, channel_width)


def vocab_tp_to_owner(
    logits: torch.Tensor, group, tp_size: int
) -> torch.Tensor:
    """Route vocab shards to the rank that owns each request's token rows."""
    if tp_size <= 1:
        return logits
    local_rows = logits.shape[0] // tp_size
    shard_width = logits.shape[-1]
    received = torch.empty_like(logits).view(-1)
    dist.all_to_all_single(received, logits.contiguous().view(-1), group=group)
    received = received.view(tp_size, local_rows, *logits.shape[1:])
    order = [1, *range(2, received.ndim - 1), 0, received.ndim - 1]
    return received.permute(order).contiguous().view(
        local_rows, *logits.shape[1:-1], shard_width * tp_size
    )


def distributed_argmax(
    logits: torch.Tensor,
    group,
    tp_rank: int,
    tp_size: int,
    owner_local: bool = False,
) -> torch.Tensor:
    """Select the global-vocab argmax from each rank's local candidate."""
    values, token_ids = logits.float().max(dim=-1)
    token_ids = token_ids + tp_rank * logits.shape[-1]
    if tp_size > 1:
        value_shards = [torch.empty_like(values) for _ in range(tp_size)]
        id_shards = [torch.empty_like(token_ids) for _ in range(tp_size)]
        dist.all_gather(value_shards, values.contiguous(), group=group)
        dist.all_gather(id_shards, token_ids.contiguous(), group=group)
        candidates = torch.stack(value_shards, dim=0)
        candidate_ids = torch.stack(id_shards, dim=0)
        source = candidates.argmax(dim=0, keepdim=True)
        token_ids = candidate_ids.gather(0, source).squeeze(0)
    if owner_local and tp_size > 1:
        rows = token_ids.shape[0] // tp_size
        token_ids = token_ids.narrow(0, tp_rank * rows, rows)
    return token_ids
