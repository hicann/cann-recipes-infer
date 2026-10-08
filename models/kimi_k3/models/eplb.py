# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch


def build_rank_local_eplb_topk(
    ep_size: int,
    ep_rank: int,
    local_tokens: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Distribute one source rank's assignments across all global experts."""
    if ep_size <= 0 or not 0 <= ep_rank < ep_size:
        raise ValueError(f"invalid EP rank {ep_rank} for EP size {ep_size}")
    if local_tokens < 0 or top_k <= 0 or top_k > num_experts:
        raise ValueError(
            "local_tokens must be non-negative and top_k must be in [1, num_experts]"
        )
    if num_experts % ep_size:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")

    experts_per_rank = num_experts // ep_size
    local_assignments = local_tokens * top_k
    rank_prefix = ep_rank * local_assignments
    local_slots = torch.arange(local_assignments, dtype=torch.int64, device=device)
    global_slots = local_slots + rank_prefix
    owner_rank = global_slots.remainder(ep_size)
    local_expert = global_slots.div(ep_size, rounding_mode="floor").remainder(experts_per_rank)
    expert_ids = owner_rank * experts_per_rank + local_expert
    return expert_ids.to(torch.int32).view(local_tokens, top_k)
