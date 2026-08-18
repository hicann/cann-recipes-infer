# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/distributed/utils.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch_npu


def get_moe_num_chunks(
    hidden_states: torch.Tensor,
    chunk_size: int,
    moe_ep_group: Optional[dist.ProcessGroup] = None,
) -> int:
    """Compute the MoE chunk count and optionally synchronize it across EP ranks.

    The local chunk count is derived from the leading token dimension and is at
    least one, including for an empty input. When ``moe_ep_group`` is provided,
    an all-reduce MAX aligns the number of chunked MoE collective rounds across
    ranks with different token counts.

    The synchronized result is copied to the host via ``item()``. Call this
    helper once before the MoE layer loop rather than once per layer.

    Args:
        hidden_states: Local hidden states whose leading dimension is the token count.
        chunk_size: Maximum number of local tokens processed in one MoE chunk.
        moe_ep_group: MoE EP communication group. ``None`` disables synchronization.

    Returns:
        The local chunk count when synchronization is disabled, otherwise the
        maximum chunk count across the provided group.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    local_token_num = hidden_states.shape[0]
    local_num_chunks = max(1, (local_token_num + chunk_size - 1) // chunk_size)
    if moe_ep_group is None:
        return local_num_chunks

    global_num_chunks = torch.tensor(
        [local_num_chunks], dtype=torch.int32, device=hidden_states.device
    )
    dist.all_reduce(
        global_num_chunks,
        op=dist.ReduceOp.MAX,
        group=moe_ep_group,
    )
    return int(global_num_chunks.item())


def split_moe_tensors(
    *tensors: Optional[torch.Tensor],
    num_chunks: Optional[int] = None,
) -> tuple[list[Optional[torch.Tensor]], ...]:
    """Split aligned MoE inputs into exactly ``num_chunks`` token chunks.

    All non-``None`` tensors must have the same leading token dimension and
    token order. ``torch.tensor_split`` always returns exactly ``num_chunks``
    slices, including empty tensors when the token count is smaller than the
    chunk count. This keeps the number and order of per-chunk collectives
    identical across ranks after their chunk counts have been synchronized.

    A ``None`` input is expanded to ``num_chunks`` ``None`` placeholders so
    callers can zip optional and non-optional MoE inputs without special cases.

    Args:
        *tensors: Token-aligned MoE inputs to split along dimension 0.
        num_chunks: Positive number of chunks; ``None`` is treated as one chunk.

    Returns:
        One chunk list per input tensor, preserving the input order.
    """
    num_chunks = 1 if num_chunks is None else num_chunks
    return tuple(
        [None] * num_chunks
        if tensor is None
        else list(torch.tensor_split(tensor, num_chunks, dim=0))
        for tensor in tensors
    )


def to_transpose_nz(tensor, transpose_contigous: bool = False):
    if transpose_contigous:
        tensor.data = tensor.data.transpose(-2, -1).contiguous()
    return torch_npu.npu_format_cast(tensor.data, 29)  # 29: to NZ format


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError("{} is not divisible by {}".format(
                         numerator, denominator))


def divide(numerator, denominator):
    """
    Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# Adapted from vllm.model_executor.utils.set_weight_attrs
def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        if hasattr(weight, key):
            raise RuntimeError(f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)
