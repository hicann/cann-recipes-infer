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

"""Communication group manager for parallel execution."""

import logging
from typing import Optional, Dict

import torch.distributed as dist

from executor.utils.hccl_utils import init_comm_group, get_group_name, get_default_group
from .inference_config import ParallelConfig


logger = logging.getLogger()


class CommManager:
    """Manages communication groups for parallel execution.

    This class creates and caches process groups for different parallel dimensions.

    Features:
    - Communication group canonicalization: Reuses groups with same rank list
    - HCCL group name support for moe_ep_group (required by NPU dispatch operators)

    Usage:
        comm_manager = CommManager(parallel_config)
        comm_manager.initialize()
        group = comm_manager.get_group("attn_tp_group")
        group_name = comm_manager.get_group_name("moe_ep_group")  # Only moe_ep_group has name
    """

    def __init__(self, parallel_config: ParallelConfig):
        """Initialize CommManager with parallel configuration.

        Args:
            parallel_config: ParallelConfig instance
        """
        self.config = parallel_config

        # Storage for communication groups and metadata
        self._groups: Dict[str, Optional[dist.ProcessGroup]] = {}
        self._group_names: Dict[str, Optional[str]] = {}
        self._ranks: Dict[str, int] = {}

    def initialize(self):
        """Initialize all communication groups at once.

        Creates groups for: attn_tp, moe_tp, embed_tp, lmhead_tp, moe_ep, etc.
        Implements canonicalization: reuses groups when rank lists match.

        Must be called after torch.distributed.init_process_group().
        """
        cfg = self.config
        global_rank = cfg.global_rank
        world_size = cfg.world_size

        # Initialize attn_tp_group
        attn_tp_group = init_comm_group(
            global_rank=global_rank,
            group_num=cfg.attn_dp_size,
            world_size=world_size,
            group_stride=1,
            group_name="attn_tp_group",
        )
        self._groups["attn_tp_group"] = attn_tp_group
        if attn_tp_group is not None:
            self._ranks["attn_tp_group"] = dist.get_rank(attn_tp_group)
        else:
            self._ranks["attn_tp_group"] = 0

        # Initialize embed_tp_group (reuse attn_tp_group if sizes match)
        if cfg.embed_tp_size == cfg.attn_tp_size:
            embed_tp_group = attn_tp_group
        else:
            embed_tp_group = init_comm_group(
                global_rank=global_rank,
                group_num=cfg.embed_dp_size,
                world_size=world_size,
                group_stride=1,
                group_name="embed_tp_group",
            )

        self._groups["embed_tp_group"] = embed_tp_group
        if embed_tp_group is not None:
            self._ranks["embed_tp_group"] = dist.get_rank(embed_tp_group)
        else:
            self._ranks["embed_tp_group"] = 0

        # Initialize lmhead_tp_group (reuse embed_tp_group if sizes match)
        if cfg.lmhead_tp_size == cfg.embed_tp_size:
            lmhead_tp_group = embed_tp_group
        else:
            lmhead_tp_group = init_comm_group(
                global_rank=global_rank,
                group_num=world_size // cfg.lmhead_tp_size,
                world_size=world_size,
                group_stride=1,
                group_name="lmhead_tp_group",
            )

        self._groups["lmhead_tp_group"] = lmhead_tp_group
        if lmhead_tp_group is not None:
            self._ranks["lmhead_tp_group"] = dist.get_rank(lmhead_tp_group)
        else:
            self._ranks["lmhead_tp_group"] = 0

        # Initialize moe_tp_group (reuse attn_tp_group if sizes match)
        if cfg.moe_tp_size > 1:
            if cfg.moe_tp_size == cfg.attn_tp_size:
                moe_tp_group = attn_tp_group
            else:
                moe_tp_group = init_comm_group(
                    global_rank=global_rank,
                    group_num=world_size // cfg.moe_tp_size,
                    world_size=world_size,
                    group_stride=1,
                    group_name="moe_tp_group",
                )

            self._groups["moe_tp_group"] = moe_tp_group
            if moe_tp_group is not None:
                self._ranks["moe_tp_group"] = dist.get_rank(moe_tp_group)
            else:
                self._ranks["moe_tp_group"] = 0

        # Initialize moe_ep_group (needs HCCL name for NPU dispatch operators)
        if cfg.moe_ep_size > 1:
            moe_ep_group, moe_ep_group_name = init_comm_group(
                global_rank=global_rank,
                group_num=world_size // cfg.moe_ep_size,
                world_size=world_size,
                group_stride=world_size // cfg.moe_ep_size,
                group_name="moe_ep_group",
                return_name=True,
            )

            self._groups["moe_ep_group"] = moe_ep_group
            self._group_names["moe_ep_group"] = moe_ep_group_name
            if moe_ep_group is not None:
                self._ranks["moe_ep_group"] = dist.get_rank(moe_ep_group)
            else:
                self._ranks["moe_ep_group"] = 0

        # Initialize dense_tp_group if needed
        if cfg.dense_tp_size > 1:
            if cfg.dense_tp_size == cfg.attn_tp_size:
                dense_tp_group = attn_tp_group
            else:
                dense_tp_group = init_comm_group(
                    global_rank=global_rank,
                    group_num=world_size // cfg.dense_tp_size,
                    world_size=world_size,
                    group_stride=1,
                    group_name="dense_tp_group",
                )

            self._groups["dense_tp_group"] = dense_tp_group
            if dense_tp_group is not None:
                self._ranks["dense_tp_group"] = dist.get_rank(dense_tp_group)
            else:
                self._ranks["dense_tp_group"] = 0

        # Initialize o_proj_tp_group if needed
        if cfg.o_proj_tp_size > 1:
            if cfg.o_proj_tp_size == cfg.attn_tp_size:
                o_proj_tp_group = attn_tp_group
            else:
                o_proj_tp_group = init_comm_group(
                    global_rank=global_rank,
                    group_num=world_size // cfg.o_proj_tp_size,
                    world_size=world_size,
                    group_stride=1,
                    group_name="o_proj_tp_group",
                )

            self._groups["o_proj_tp_group"] = o_proj_tp_group
            if o_proj_tp_group is not None:
                self._ranks["o_proj_tp_group"] = dist.get_rank(o_proj_tp_group)
            else:
                self._ranks["o_proj_tp_group"] = 0

        # Add default process group
        self._groups["default_pg"] = get_default_group()

    def get_group(self, name: str) -> Optional[dist.ProcessGroup]:
        """Get communication group by name."""
        if name not in self._groups:
            raise KeyError(
                f"Communication group '{name}' not found. "
                f"Available groups: {list(self._groups.keys())}"
            )
        return self._groups[name]

    def get_rank(self, group_name: str) -> int:
        """Get rank within the specified communication group."""
        if group_name not in self._ranks:
            raise KeyError(
                f"Communication group '{group_name}' not found. "
                f"Available groups: {list(self._ranks.keys())}"
            )
        return self._ranks[group_name]

    def has_group(self, name: str) -> bool:
        """Check if a communication group exists."""
        return name in self._groups

    def get_group_name(self, name: str) -> str:
        """Get HCCL group name string."""
        if name not in self._group_names:
            raise KeyError(
                f"Communication group '{name}' has no stored HCCL name. "
                f"Groups with HCCL names: {list(self._group_names.keys())}"
            )
        return self._group_names[name]
