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
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist

from executor.utils.hccl_utils import (
    get_default_group,
    get_group_name,
    init_comm_group_by_ranks,
)
from .inference_config import ParallelConfig, PlatformVersion


logger = logging.getLogger(__name__)


_DEFAULT_HCCL_BUFFSIZE_MB = 200
PhysicalGroupKey = Tuple[Tuple[int, ...], Optional[int], int, PlatformVersion]


@dataclass
class CommGroupConfig:
    """Pure data describing one logical communication group's creation intent.

    Attributes:
        name: Logical group name (e.g. ``"attn_tp_group"``, ``"moe_ep_group_mc2"``).
        subgroups: The *entire* subgroup partition of this logical group, not
            only the subgroup this rank belongs to.  ``dist.new_group`` is a
            collective call — every world rank must invoke it with identical
            ``ranks`` arguments, so every rank's CommManager sees the same
            ``subgroups`` list and iterates it.  Inner ``List[int]`` MUST be
            sorted ascending; enforced by ``__post_init__``.
        hccl_buffer_size: HCCL buffer size in MB.  ``None`` means "use env
            default" and is normalized to the env value at signature time.
        group_type: ``hccl_op_expansion_mode`` (see ``init_comm_group_by_ranks``).
        platform_version: Atlas platform enum.  Participates in the signature
            to prevent cross-platform physical reuse.
        return_name: If True, also capture the HCCL comm name (required by
            ``npu_moe_distribute_dispatch_v2`` / ``combine_v2`` for mc2).
        allow_physical_reuse: When False the subgroup always builds a fresh
            HCCL communicator and bypasses the signature cache and the
            world-group shortcut.
    """
    name: str
    subgroups: List[List[int]]
    hccl_buffer_size: Optional[int] = None
    group_type: Optional[int] = None
    platform_version: PlatformVersion = PlatformVersion.A3
    world_size: Optional[int] = None
    return_name: bool = False
    allow_physical_reuse: bool = True

    def __post_init__(self):
        # Treat legacy default-expansion encodings as one internal value.
        self.group_type = None if self.group_type in (None, 0) else self.group_type
        self.platform_version = PlatformVersion.from_value(self.platform_version)
        if not self.subgroups:
            raise ValueError(f"{self.name}: subgroups must be non-empty")
        for sg in self.subgroups:
            if not sg:
                raise ValueError(f"{self.name}: subgroup ranks must be non-empty")
            if list(sg) != sorted(sg):
                raise ValueError(
                    f"{self.name}: subgroup ranks must be in ascending order, "
                    f"got {list(sg)}"
                )
            if self.world_size is not None and (sg[0] < 0 or sg[-1] >= self.world_size):
                raise ValueError(
                    f"{self.name}: subgroup ranks must be in [0, {self.world_size}), "
                    f"got {list(sg)}"
                )


class CommManager:
    """Manages communication groups for parallel execution.

    Built from ``ParallelConfig`` + ``platform_version``.  Models declare the
    business groups they need through ``register_group(...)``; the manager
    handles subgroup materialization, signature-based physical reuse, forced
    exclusive creation, and name/rank caches.

    Public APIs:
        - ``get_group(name)``
        - ``get_rank(name)``
        - ``get_group_name(name)``
        - ``has_group(name)``
        - ``register_group(...)`` for model-declared communication groups

    Usage:
        comm_manager = CommManager(parallel_config, platform_version=PlatformVersion.A3)
        model.init_parallel_comm_group()
        g = comm_manager.get_group("attn_tp_group")
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        platform_version: PlatformVersion = PlatformVersion.A3,
    ):
        self.config = parallel_config
        self.platform_version = PlatformVersion.from_value(platform_version)

        # Logical-name -> artefact
        self._groups: Dict[str, Optional[dist.ProcessGroup]] = {}
        self._group_names: Dict[str, Optional[str]] = {}
        self._ranks: Dict[str, int] = {}

        # Physical signature -> ProcessGroup.
        # Key layout: (tuple(sorted(ranks)), group_type, resolved_bufsize, platform_version)
        self._physical_cache: Dict[PhysicalGroupKey, Optional[dist.ProcessGroup]] = {}

        self._cache_default_group()

    def _cache_default_group(self):
        """Cache default_pg by physical signature for later world-group reuse."""
        default_pg = get_default_group()
        default_bufsize = int(os.environ.get("HCCL_BUFFSIZE", _DEFAULT_HCCL_BUFFSIZE_MB))
        default_key = (
            tuple(range(self.config.world_size)),
            None,
            default_bufsize,
            self.platform_version,
        )
        self._physical_cache[default_key] = default_pg
        self._groups["default_pg"] = default_pg

    def init_cpu_groups(self):
        """Initialize gloo-backend CPU groups for online mode coordination.

        Creates:
        - dp_leader_group: cross-DP synchronization among DP leaders
        - tp_cpu_group: DP leader -> TP workers broadcast of Python objects
        """
        cfg = self.config
        global_rank = cfg.global_rank
        dp_size = cfg.attn_dp_size
        tp_size = cfg.attn_tp_size
        cp_size = cfg.cp_size
        group_size = tp_size * cp_size

        # dp_leader_group: ranks where global_rank % group_size == 0
        if dp_size > 1:
            dp_leader_ranks = [i * group_size for i in range(dp_size)]
            if global_rank in dp_leader_ranks:
                dp_leader_group = dist.new_group(dp_leader_ranks, backend="gloo")
                self._ranks["dp_leader_group"] = dist.get_rank(dp_leader_group)
            else:
                dp_leader_group = None
                self._ranks["dp_leader_group"] = 0
            self._groups["dp_leader_group"] = dp_leader_group
            logger.info(f"dp_leader_group initialized: ranks={dp_leader_ranks}")
        else:
            self._groups["dp_leader_group"] = None
            self._ranks["dp_leader_group"] = 0

        # tp_cpu_group: same partitioning as a full DP partition, gloo backend
        if group_size > 1:
            for dp_idx in range(dp_size):
                tp_ranks = [dp_idx * group_size + i for i in range(group_size)]
                group = dist.new_group(tp_ranks, backend="gloo")
                if global_rank in tp_ranks:
                    self._groups["tp_cpu_group"] = group
                    self._ranks["tp_cpu_group"] = dist.get_rank(group)
            logger.info(
                f"tp_cpu_group initialized: tp_size={tp_size}, cp_size={cp_size}, dp_size={dp_size}"
            )
        else:
            self._groups["tp_cpu_group"] = None
            self._ranks["tp_cpu_group"] = 0

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

    def register_group(
        self,
        name: str,
        group_num: Optional[int] = None,
        group_size: Optional[int] = None,
        group_stride: int = 1,
        start_ranks: Optional[List[int]] = None,
        subgroups: Optional[List[List[int]]] = None,
        hccl_buffer_size: Optional[int] = None,
        group_type: Optional[int] = None,
        return_name: bool = False,
        allow_physical_reuse: bool = True,
        platform_version: Optional[PlatformVersion] = None,
    ) -> Optional[dist.ProcessGroup]:
        """Register a model-specific communication group.

        This may be called directly by model constructors.  The default process
        group signature is cached during ``CommManager`` construction so
        world-sized business groups can physically reuse it.
        ``subgroups`` may be supplied directly for custom topologies.  Otherwise
        ``group_num`` / ``group_size`` / ``group_stride`` generate subgroups.
        ``start_ranks`` may be supplied for custom regular-stride topologies
        such as DP/PP/TP rank layouts.
        """
        if name in self._groups:
            logger.info(
                f"CommManager: group '{name}' already registered, "
                "skip duplicate registration"
            )
            return self._groups[name]

        if subgroups is None:
            if group_num is None or group_size is None:
                raise ValueError(
                    f"{name}: either subgroups or both group_num and group_size must be provided"
                )
            if group_num * group_size != self.config.world_size:
                raise ValueError(
                    f"{name}: group_num * group_size must equal "
                    f"world_size={self.config.world_size}, got "
                    f"{group_num} * {group_size}"
                )
            subgroups = self._build_strided_subgroups(
                group_num=group_num,
                group_size=group_size,
                group_stride=group_stride,
                start_ranks=start_ranks,
            )

        group_config = CommGroupConfig(
            name=name,
            subgroups=subgroups,
            hccl_buffer_size=hccl_buffer_size,
            group_type=group_type,
            platform_version=platform_version or self.platform_version,
            world_size=self.config.world_size,
            return_name=return_name,
            allow_physical_reuse=allow_physical_reuse,
        )
        self._register_group(group_config)
        return self._groups[name]

    @staticmethod
    def _build_strided_subgroups(
        group_num: int,
        group_size: int,
        group_stride: int = 1,
        start_ranks: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """Build regular-stride subgroup ranks.

        ``group_num`` is the number of subgroups, and ``group_size`` is the
        number of ranks in each subgroup.  When ``start_ranks`` is omitted, this
        follows the legacy ``init_comm_group`` defaults: contiguous blocks for
        ``group_stride == 1`` and interleaved groups for larger strides. Complex
        DP/PP/TP layouts can pass explicit ``start_ranks``.
        """
        if group_num <= 0:
            raise ValueError(f"group_num must be positive, got {group_num}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if group_stride <= 0:
            raise ValueError(f"group_stride must be positive, got {group_stride}")

        if start_ranks is None:
            if group_stride == 1:
                start_ranks = [
                    group_id * group_size for group_id in range(group_num)
                ]
            else:
                start_ranks = list(range(group_num))
        if len(start_ranks) != group_num:
            raise ValueError(
                f"len(start_ranks)={len(start_ranks)} must equal group_num={group_num}"
            )

        subgroups = [
            [start_rank + i * group_stride for i in range(group_size)]
            for start_rank in start_ranks
        ]
        max_rank = group_num * group_size
        for subgroup in subgroups:
            if subgroup[0] < 0 or subgroup[-1] >= max_rank:
                raise ValueError(
                    f"subgroup ranks must be in [0, {max_rank}), got {subgroup}"
                )
        return subgroups

    @staticmethod
    def _compute_physical_key(
        subgroup_ranks: List[int], group_config: CommGroupConfig
    ) -> PhysicalGroupKey:
        """Cache key for one subgroup.

        Returns:
            Tuple of ``(sorted_ranks, group_type, hccl_buffer_size,
            platform_version)``.  ``sorted_ranks`` is a tuple of global rank
            ids, ``group_type`` is the normalized HCCL expansion mode,
            ``hccl_buffer_size`` is resolved to the effective MB value, and
            ``platform_version`` prevents cross-platform physical reuse.

        ``sorted()`` here is defensive — subgroups from ``_build_*_config()``
        are already ascending (see ``CommGroupConfig.__post_init__``), but we
        sort again so hand-written configs that skip construction helpers
        still collapse to the same key as their sorted twins.
        """
        return (
            tuple(sorted(subgroup_ranks)),
            group_config.group_type,
            group_config.hccl_buffer_size
            if group_config.hccl_buffer_size is not None
            else int(os.environ.get("HCCL_BUFFSIZE", _DEFAULT_HCCL_BUFFSIZE_MB)),
            group_config.platform_version,
        )

    def _get_or_create_group(
        self, group_config: CommGroupConfig
    ) -> Tuple[Optional[dist.ProcessGroup], Optional[str]]:
        """Resolve the ProcessGroup (and optional HCCL comm name) for this config.

        Two-stage flow:
            Stage 1  Materialization: decide whether a real HCCL group is
                    needed at all.  Single-rank subgroup -> None.  World
                    subgroup with default HCCL params -> shortcut to
                    ``default_pg``.  Otherwise fall through to stage 2.
            Stage 2  Physical reuse: ``allow_physical_reuse=False`` bypasses
                    the cache (mc2 always fresh).  Otherwise consult
                    ``_physical_cache`` by signature.

        Collective protocol: every world rank iterates ``subgroups`` in the
        same order and participates in every ``new_group`` call.  Each rank
        only keeps the ProcessGroup corresponding to the subgroup it belongs
        to.

        Returns a ``(group, hccl_name)`` pair; ``hccl_name`` is ``None``
        unless the config requested it and the group was actually built.
        """
        global_rank = self.config.global_rank

        chosen_group: Optional[dist.ProcessGroup] = None
        chosen_name: Optional[str] = None

        for ranks in group_config.subgroups:
            group = None
            hccl_name = None
            key = self._compute_physical_key(ranks, group_config)
            can_reuse = group_config.allow_physical_reuse
            # Ordinary single-rank logical groups do not need a real HCCL
            # ProcessGroup. Exclusive groups (MC2) still materialize so they
            # can own a dedicated communicator even when their ranks list is
            # degenerate.
            needs_physical_group = not can_reuse or len(ranks) > 1

            # Reusable physical groups are keyed by ranks + HCCL attributes.
            # Cache misses and exclusive groups share the same creation path.
            if needs_physical_group and can_reuse and key in self._physical_cache:
                group = self._physical_cache[key]
                hccl_name = (
                    get_group_name(group, global_rank)
                    if group_config.return_name and global_rank in ranks
                    else None
                )
            elif needs_physical_group:
                result = init_comm_group_by_ranks(
                    ranks,
                    global_rank=global_rank,
                    group_name=group_config.name,
                    hccl_buffer_size=group_config.hccl_buffer_size,
                    group_type=group_config.group_type,
                    platform_version=group_config.platform_version.value,
                    return_name=group_config.return_name,
                )
                group, hccl_name = (
                    result if group_config.return_name else (result, None)
                )
                if can_reuse:
                    self._physical_cache[key] = group

            if global_rank in ranks:
                chosen_group = group
                chosen_name = hccl_name

        return chosen_group, chosen_name

    def _register_group(self, group_config: CommGroupConfig) -> None:
        """Run the config through build/reuse and stash the result under its logical name."""
        group, hccl_name = self._get_or_create_group(group_config)
        self._groups[group_config.name] = group
        if group_config.return_name:
            self._group_names[group_config.name] = hccl_name
        if group is not None:
            self._ranks[group_config.name] = dist.get_rank(group)
        else:
            self._ranks[group_config.name] = 0
        # Only log the subgroup this rank belongs to, not the whole partition —
        # large EP configs would otherwise dump hundreds of ranks per line.
        global_rank = self.config.global_rank
        my_subgroup = next(
            (sg for sg in group_config.subgroups if global_rank in sg),
            None,
        )
        logger.info(
            f"CommManager: group '{group_config.name}' registered "
            f"(my_subgroup={my_subgroup}, reuse={group_config.allow_physical_reuse}, "
            f"rank_in_group={self._ranks[group_config.name]})"
        )
