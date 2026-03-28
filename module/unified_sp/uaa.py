# Adapted from 
# https://github.com/vipshop/cache-dit.
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Copyright (c) 2025 Cache-DiT Authors. All Rights Reserved
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

from typing import Tuple, List, Callable, Optional
import copy
import functools

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed._functional_collectives as fc


def split_func(tensor, num_splits, dim=-1, patch_size=2):
    """
    Split a tensor along the specified dimension with sizes that align with patchification.

    This function splits a tensor such that each split, when patchified (reduced by patch_size),
    has the same sizes as if the tensor were first patchified and then split evenly.
    This ensures consistency between "split-then-patchify" and "patchify-then-split" approaches.

    Args:
        tensor: Input tensor to split
        num_splits: Number of splits to create
        dim: Dimension along which to split (default: -1, last dimension)
        patch_size: Size of patches for patchification (default: 2)

    Returns:
        Tuple of tensor splits with sizes aligned to patchification

    Raises:
        ValueError: If the last dimension is not divisible by patch_size
        ValueError: If num_splits is less than 1
        ValueError: If the patched dimension is smaller than num_splits
    """
    # Validate inputs
    if num_splits < 1:
        raise ValueError(f"num_splits must be at least 1, got {num_splits}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be at least 1, got {patch_size}")
    if dim < -len(tensor.shape) or dim >= len(tensor.shape):
        raise ValueError(f"dim {dim} out of range for tensor with {len(tensor.shape)} dimensions")
    last_dim = tensor.shape[-1]
    patched_dim = last_dim // patch_size

    # Compute split sizes for the patched dimension (even distribution with remainder)
    # This mimics "patchify-then-split" approach: distribute patched_dim as evenly as possible
    split_sizes_patched = []
    base_size = patched_dim // num_splits
    remainder = patched_dim % num_splits

    for i in range(num_splits):
        # First 'remainder' splits get one extra element to handle uneven division
        if i < remainder:
            split_sizes_patched.append(base_size + 1)
        else:
            split_sizes_patched.append(base_size)

    # Convert patched split sizes back to original dimension sizes
    # Each original split size = corresponding patched size * patch_size
    split_sizes_original = [s * patch_size for s in split_sizes_patched]

    # Perform the actual split along the specified dimension
    splits = torch.split(tensor, split_sizes_original, dim=dim)

    return splits


def _maybe_pad_qkv_head(
    x: torch.Tensor,
    h: int,
    world_size: int,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (b, s_local, h, d)
    h: int, original global head num
    return: Tuple[torch.Tensor, int], padded tensor (b, s_local, h + h_pad, d) and h_pad
    """
    h_pad = 0
    if h % world_size != 0:
        h_pad = world_size - (h % world_size)
        new_h_local = (h + h_pad) // world_size
        # e.g., Allow: h=30, world_size=8 -> new_h_local=4, h_pad=2.
        # NOT ALLOW: h=30, world_size=16 -> new_h_local=2, h_pad=14.
        if h_pad >= new_h_local:
            raise ValueError(
                f"Padding head num {h_pad} should be less than new local head num {new_h_local}"
            )
        x = F.pad(x, (0, 0, 0, h_pad)).contiguous()
    return x, h_pad


def _maybe_unpad_qkv_head(
    x: torch.Tensor,
    h_pad: int,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (b, S_GLOBAL, H_LOCAL + h_pad, d)
    h_pad: int, head padding num
    return: torch.Tensor, unpadded tensor (b, S_GLOBAL, H_LOCAL, d)
    """
    # Only the last rank may have padding
    if h_pad > 0 and local_rank == world_size - 1:
        x = x[:, :, :-h_pad, :]
    return x.contiguous()


def _maybe_pad_o_head(
    x: torch.Tensor,
    h: int,
    world_size: int,
    rank: int,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (b, S_GLOBAL, h_LOCAL, d)
    h: int, original global head num
    return: Tuple[torch.Tensor, int], padded tensor (b, S_GLOBAL, h_LOCAL + h_pad, d) and h_pad
    """
    if h is None:
        return x, 0
    h_pad = 0
    # Only the last rank may need padding
    if h % world_size != 0:
        # We need to broadcast h_pad to all ranks to keep consistency
        # in unpadding step later for all ranks.
        h_pad = world_size - (h % world_size)
        new_h_local = (h + h_pad) // world_size
        if h_pad >= new_h_local:
            raise ValueError(
                f"Padding head num {h_pad} should be less than new local head num {new_h_local}"
            )
        if rank == world_size - 1:
            x = F.pad(x, (0, 0, 0, h_pad)).contiguous()
    return x, h_pad


def _maybe_unpad_o_head(
    x: torch.Tensor,
    h_pad: int,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (b, s_local, h_GLOBAL + h_pad, d)
    h_pad: int, head padding num
    return: torch.Tensor, unpadded tensor (b, s_local, H_GLOBAL, d)
    """
    if h_pad > 0:
        x = x[:, :, :-h_pad, :]
    return x.contiguous()


def _gather_size_by_comm(size: int, group: dist.ProcessGroup) -> List[int]:
    r"""Gather the local size from all ranks.
    size: int, local size
    return: List[int], list of size from all ranks
    """
    # NOTE(Serving/CP Safety):
    # Do NOT cache this collective result.
    #
    # In "Ulysses Anything" mode, `size` (e.g. per-rank local seq_len / s_local)
    # may legitimately differ across ranks. If we cache based on the *local* `size`,
    # different ranks can have different cache hit/miss patterns across time.
    #
    # That can lead to a catastrophic distributed hang:
    # - some ranks hit cache and *skip* dist.all_gather()
    # - other ranks miss cache and *enter* dist.all_gather()
    # This mismatched collective participation will stall the process group and
    # eventually trigger NCCL watchdog timeouts (often surfacing later as ALLTOALL
    # timeouts in Ulysses attention).
    world_size = dist.get_world_size(group=group)
    # HACK: Use Gloo backend for all_gather to avoid H2D and D2H overhead
    comm_backends = str(dist.get_backend(group=group))
    # NOTE: e.g., dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    gather_device = "npu"
    gathered_sizes = [
        torch.empty((1,), device=gather_device, dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(
        gathered_sizes,
        torch.tensor([size], device=gather_device, dtype=torch.int64),
        group=group,
    )

    gathered_sizes = [s[0].item() for s in gathered_sizes]
    # NOTE: DON'T use tolist here due to graph break - Explanation:
    # Backend compiler `inductor` failed with aten._local_scalar_dense.default
    return gathered_sizes


def _wait_tensor(tensor) -> torch.Tensor:
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()

    return tensor


@functools.lru_cache(maxsize=64)
def _fill_gather_shapes(
    shape: Tuple[int], gather_dims: Tuple[int], dim: int, world_size: int
) -> List[List[int]]:
    gather_shapes = []
    for i in range(world_size):
        # WARN: deepcopy to avoid modifying the original shape
        rank_shape = list(copy.deepcopy(shape))
        rank_shape[dim] = gather_dims[i]
        gather_shapes.append(rank_shape)
    return gather_shapes


def _maybe_pad_o_head(
    x: torch.Tensor,
    h: int,
    local_rank: int,
    world_size: int,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (b, S_GLOBAL, H_LOCAL, d)
    h: int, original global head num
    return: Tuple[torch.Tensor, int], padded tensor (b, S_GLOBAL, H_LOCAL + h_pad, d) and h_pad
    """
    if h is None:
        return x, 0

    h_pad = 0
    # Only the last rank may need padding
    if h % world_size != 0:
        # We need to broadcast h_pad to all ranks to keep consistency
        # in unpadding step later for all ranks.
        h_pad = world_size - (h % world_size)
        new_h_local = (h + h_pad) // world_size
        if h_pad >= new_h_local:
            raise ValueError(
                f"Padding head num {h_pad} should be less than new local head num {new_h_local}"
            )
        if local_rank == world_size - 1:
            x = F.pad(x, (0, 0, 0, h_pad)).contiguous()
    return x, h_pad


def _maybe_unpad_o_head(
    x: torch.Tensor,
    h_pad: int,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (b, s_local, H_GLOBAL + h_pad, d)
    h_pad: int, head padding num
    return: torch.Tensor, unpadded tensor (b, s_local, H_GLOBAL, d)
    """
    if h_pad > 0:
        x = x[:, :, :-h_pad, :]
    return x.contiguous()


def all_gather_anything(  # noqa: F811
    tensor: torch.Tensor,
    dim: int,
    world_size: int,
    group: dist.device_mesh.DeviceMesh,
) -> torch.Tensor:

    tensor = tensor.contiguous()
    shape = tensor.shape
    rank_dim = shape[dim]
    gather_dims = _gather_size_by_comm(rank_dim, group)

    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization for now.

    gather_shapes = _fill_gather_shapes(
        tuple(shape),
        tuple(gather_dims),
        dim,
        world_size,
    )

    gathered_tensors = [
        torch.empty(
            shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        for shape in gather_shapes
    ]

    dist.all_gather(gathered_tensors, tensor, group=group)
    gathered_tensor = torch.cat(gathered_tensors, dim=dim)
    return gathered_tensor