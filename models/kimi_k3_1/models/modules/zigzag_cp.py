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

"""Zigzag token exchange for Kimi K3 MLA context-parallel prefill.

KDA consumes contiguous CP shards because its recurrent prefix scan follows
sequence order. MLA instead benefits from a balanced causal-attention layout:
split the padded sequence into ``2 * cp_size`` equal fragments and give rank
``r`` fragment ``r`` plus fragment ``2 * cp_size - 1 - r``. The exchange is
performed immediately before MLA attention and inverted immediately after it,
so the rest of the model continues to observe contiguous shards.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist


def zigzag_dst_rank(fragment: int, cp_size: int) -> int:
    """Return the CP rank that owns ``fragment`` in zigzag order."""
    fragment_count = 2 * cp_size
    if not 0 <= fragment < fragment_count:
        raise ValueError(
            f"fragment must be in [0, {fragment_count}), got {fragment}"
        )
    return (
        fragment
        if fragment < cp_size
        else fragment_count - 1 - fragment
    )


def zigzag_src_rank(fragment: int, cp_size: int) -> int:
    """Return the CP rank holding ``fragment`` in contiguous order."""
    fragment_count = 2 * cp_size
    if not 0 <= fragment < fragment_count:
        raise ValueError(
            f"fragment must be in [0, {fragment_count}), got {fragment}"
        )
    return fragment // 2


class ZigzagCPLayout:
    """Fixed permutation between contiguous and zigzag CP token order."""

    def __init__(
        self,
        cp_size: int,
        cp_rank: int,
        local_token_num: int,
        group=None,
    ) -> None:
        if cp_size < 2:
            raise ValueError(f"zigzag CP needs cp_size >= 2, got {cp_size}")
        if not 0 <= cp_rank < cp_size:
            raise ValueError(
                f"cp_rank must be in [0, {cp_size}), got {cp_rank}"
            )
        if local_token_num <= 0 or local_token_num % 2:
            raise ValueError(
                "zigzag CP needs a positive, even local token count, got "
                f"{local_token_num}"
            )

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.group = group
        self.local_token_num = local_token_num
        self.fragment_count = 2 * cp_size
        self.frag_num = self.fragment_count
        self.fragment_len = local_token_num // 2
        self.padded_token_num = self.fragment_count * self.fragment_len

        held_fragments = (2 * cp_rank, 2 * cp_rank + 1)
        # all_to_all_single consumes send chunks in ascending destination-rank
        # order, so some ranks must locally swap their two contiguous fragments.
        self.send_order = tuple(
            sorted(
                range(2),
                key=lambda slot: zigzag_dst_rank(
                    held_fragments[slot], cp_size
                ),
            )
        )
        self.head_fragment = cp_rank
        self.tail_fragment = self.fragment_count - 1 - cp_rank

        self.scatter_splits = [0] * cp_size
        for slot in self.send_order:
            destination = zigzag_dst_rank(held_fragments[slot], cp_size)
            self.scatter_splits[destination] += self.fragment_len

        self.gather_splits = [0] * cp_size
        for fragment in (self.head_fragment, self.tail_fragment):
            source = zigzag_src_rank(fragment, cp_size)
            self.gather_splits[source] += self.fragment_len

        self.head_query_offset = self.head_fragment * self.fragment_len
        self.tail_query_offset = self.tail_fragment * self.fragment_len
        self.kv_len_head = (self.head_fragment + 1) * self.fragment_len
        self.kv_len_tail = (self.tail_fragment + 1) * self.fragment_len

    @property
    def frag_len(self) -> int:
        """Compatibility alias used by the reference implementation."""
        return self.fragment_len

    def length_tensors(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build query/head-KV/tail-KV lengths on ``reference``'s device."""
        return tuple(
            reference.new_tensor([value])
            for value in (
                self.fragment_len,
                self.kv_len_head,
                self.kv_len_tail,
            )
        )

    def _check_rows(self, tensor: torch.Tensor, rows: int) -> None:
        if tensor.shape[0] != rows:
            raise ValueError(
                f"expected {rows} rows on dim 0, got {tensor.shape[0]}"
            )

    @staticmethod
    def _zero_pad(tensor: torch.Tensor, rows: int) -> torch.Tensor:
        if tensor.shape[0] == rows:
            return tensor
        if tensor.shape[0] > rows:
            raise ValueError(
                f"cannot pad {tensor.shape[0]} rows down to {rows}"
            )
        padded = tensor.new_zeros(rows, *tensor.shape[1:])
        padded[: tensor.shape[0]].copy_(tensor)
        return padded

    def pad_local(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad a contiguous local shard to ``local_token_num`` rows."""
        return self._zero_pad(tensor, self.local_token_num)

    def pad_global(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad a global-order tensor to ``padded_token_num`` rows."""
        return self._zero_pad(tensor, self.padded_token_num)

    def sp_to_cp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Exchange contiguous rows into local ``[head, tail]`` zigzag rows."""
        self._check_rows(tensor, self.local_token_num)
        if self.send_order == (0, 1):
            send = tensor.contiguous()
        else:
            fragments = tensor.split(self.fragment_len, dim=0)
            send = torch.cat(
                [fragments[slot] for slot in self.send_order], dim=0
            ).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=self.gather_splits,
            input_split_sizes=self.scatter_splits,
            group=self.group,
        )
        return recv

    def cp_to_sp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`sp_to_cp` and restore contiguous local rows."""
        self._check_rows(tensor, self.local_token_num)
        recv = torch.empty_like(tensor)
        dist.all_to_all_single(
            recv,
            tensor.contiguous(),
            output_split_sizes=self.scatter_splits,
            input_split_sizes=self.gather_splits,
            group=self.group,
        )
        if self.send_order == (0, 1):
            return recv

        fragments = recv.split(self.fragment_len, dim=0)
        inverse_order = [0, 0]
        for send_slot, local_slot in enumerate(self.send_order):
            inverse_order[local_slot] = send_slot
        return torch.cat(
            [fragments[slot] for slot in inverse_order], dim=0
        )

    def zigzag_fragments(self) -> Sequence[int]:
        """Return this rank's global fragment IDs in local row order."""
        return self.head_fragment, self.tail_fragment


_LAYOUT_CACHE: dict[tuple[int, int, int], ZigzagCPLayout] = {}
_LAYOUT_CACHE_LIMIT = 32


def get_zigzag_layout(
    cp_size: int,
    cp_rank: int,
    local_token_num: int,
    group=None,
) -> ZigzagCPLayout:
    """Return an immutable layout shared by all MLA layers in one process."""
    key = cp_size, cp_rank, local_token_num
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None and cached.group is group:
        return cached

    layout = ZigzagCPLayout(cp_size, cp_rank, local_token_num, group)
    if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_LIMIT:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = layout
    return layout


def clear_zigzag_layout_cache() -> None:
    """Drop cached layouts for tests or process-group teardown."""
    _LAYOUT_CACHE.clear()


__all__ = [
    "ZigzagCPLayout",
    "clear_zigzag_layout_cache",
    "get_zigzag_layout",
    "zigzag_dst_rank",
    "zigzag_src_rank",
]
