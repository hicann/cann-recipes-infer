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

"""KV-transfer data types + transfer engine (types + backend glue).

- KVPoll: request-level transfer status
- PrefillServerInfo / TargetRankMapping / PrefillRankInfo / KVArgsRegisterInfo: topology info
- _KVTransferTask: payload submitted to KVTransferManager._transfer_executor
- TransferEngine ABC + AscendTransferEngine (memfabric_hybrid wrapper)
- all_reduce_poll: TP/CP group poll consensus helper
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def all_reduce_poll(local_status: "KVPoll", group=None) -> "KVPoll":
    """Reduce poll status across TP/CP group (min = most pessimistic).

    group=None is intentionally a no-op (not the default process group) to
    prevent accidental cross-DP-rank collectives.
    """
    if group is not None and dist.is_initialized():
        tensor = torch.tensor([int(local_status)], dtype=torch.int64, device="cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
        return KVPoll(tensor.item())
    return local_status


class KVPoll(IntEnum):
    """Request-level KV transfer status."""

    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


@dataclass
class PrefillServerInfo:
    """Cluster topology of a Prefill service, fetched from bootstrap /route."""

    attn_tp_size: int
    attn_cp_size: int
    dp_size: int
    block_size: Optional[int] = None
    kv_cache_dtype: Optional[str] = None
    ranks: dict[str, dict[str, int | str]] = field(default_factory=dict)

    def __post_init__(self):
        self.attn_tp_size = int(self.attn_tp_size)
        self.attn_cp_size = int(self.attn_cp_size)
        self.dp_size = int(self.dp_size)


@dataclass
class TargetRankMapping:
    """Decode-local derivation of which Prefill ranks this decode rank pulls KV from.

    Computed from the decode rank's own parallel shape + the Prefill topology.
    """

    target_tp_rank: int
    target_tp_ranks: list[int]
    target_cp_ranks: list[int]
    required_dst_info_num: int
    required_prefill_response_num: int


@dataclass
class PrefillRankInfo:
    rank_ip: str
    rank_port: int


@dataclass
class KVArgsRegisterInfo:
    dst_kv_ptrs: list[int]
    dst_kv_item_lens: list[int]
    metadata_buffer_index: int
    dst_block_ids: dict[str, list[int]]
    dst_tp_rank: int = 0
    kv_data_lens: Optional[list[int]] = None
    kv_head_num: Optional[int] = None
    total_kv_head_num: Optional[int] = None


@dataclass
class _KVTransferTask:
    """Task submitted by AscendKVSender to KVTransferManager._transfer_executor.

    The background transfer worker drains this queue, executes the RDMA
    transfer, then notifies the decode side.
    """
    bootstrap_room: int
    bootstrap_infos: list
    src_block_ids: dict  # attn_type -> list[block_id] (prefill's own blocks)
    metadata: dict
    sender: object  # AscendKVSender — needed for prefill_unique_rank


class TransferEngine(ABC):
    """Abstract transfer-engine interface for KV and metadata movement."""

    @property
    @abstractmethod
    def rank_port(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_register(self, ptrs: Iterable[int], lengths: Iterable[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_transfer_sync(
        self,
        decode_session_id: str,
        src_addrs: Iterable[int],
        dst_addrs: Iterable[int],
        lengths: Iterable[int],
    ) -> int:
        raise NotImplementedError


class AscendTransferEngine(TransferEngine):
    """Ascend transfer engine wrapper backed by memfabric_hybrid.

    The key mechanism: prefill uses decode's session_id to RDMA-write KV data
    directly into decode's NPU memory. session_id = "<ip>:<rpc_port>" and is
    computed from engine.get_rpc_port() after construction.
    """

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        npu_id: int = 0,
        disaggregation_mode: str = "PREFILL",
        store_url: str = "",
        is_store_creator_node: bool = False,
    ):
        self.hostname = hostname
        self.npu_id = npu_id
        self.disaggregation_mode = disaggregation_mode
        if not store_url:
            raise ValueError(
                "AscendTransferEngine requires a non-empty store_url "
                "(DisaggConfig.store_url)."
            )
        self.store_url = store_url
        self.is_store_creator_node = is_store_creator_node
        try:
            from memfabric_hybrid import TransferEngine as MemFabricTransferEngine
        except ImportError as exc:
            raise ImportError(
                "memfabric_hybrid is required for PD disaggregation"
            ) from exc
        self.engine = MemFabricTransferEngine()
        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{self.hostname}:{rpc_port}"
        self._rank_port = rpc_port

    @property
    def rank_port(self) -> int:
        return self._rank_port

    def initialize(self) -> None:
        """Initialize the transfer engine.

        Only one process in the entire service — the prefill rank-0 worker whose
        local IP matches the store_url host — creates the memfabric TCP config
        store. Every other prefill/decode rank (across all instances) connects
        via engine.initialize() with retry, since the store may not be up yet.
        """
        role = "Prefill" if self.disaggregation_mode == "PREFILL" else "Decode"

        # One worker in the whole service creates the store: the local-rank-0
        # worker on the node that server.py tagged as the store-creator node.
        # Avoids fragile hostname/NIC comparison on multi-NIC hosts.
        is_store_creator = (
            self.is_store_creator_node
            and int(os.environ.get("LOCAL_RANK", "0")) == 0
        )

        if is_store_creator:
            from memfabric_hybrid import create_config_store
            ret_cs = create_config_store(self.store_url)
            if ret_cs != 0:
                raise RuntimeError(
                    f"create_config_store returned {ret_cs} (store_url={self.store_url})."
                )
            logger.info("create_config_store ok (store_url=%s)", self.store_url)

        # Choose transfer protocol: SDMA (default) or DEVICE_RDMA via env var.
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL", "sdma").lower()
        if protocol == "device_rdma":
            trans_op_type = self.engine.TransDataOpType.DEVICE_RDMA
            # Initialize HCCL in advance via all_gather to avoid conflicts with
            # RDMA initialization.
            if dist.is_initialized():
                tmp = torch.zeros(1, device="npu")
                output = [torch.empty_like(tmp) for _ in range(dist.get_world_size())]
                dist.all_gather(output, tmp)
        else:
            trans_op_type = self.engine.TransDataOpType.SDMA

        # Non-creators (all decode ranks + non-creator prefill ranks) may start
        # before the creator finishes — retry with backoff.
        max_retries = 1 if is_store_creator else 30
        retry_interval = 2.0
        ret = -1
        for attempt in range(max_retries):
            ret = self.engine.initialize(
                self.store_url,
                self.session_id,
                role,
                self.npu_id,
                trans_op_type,
            )
            if ret == 0:
                break
            if attempt + 1 < max_retries:
                logger.warning(
                    "transfer engine init returned %d (attempt %d/%d), retrying in %.1fs...",
                    ret, attempt + 1, max_retries, retry_interval,
                )
                time.sleep(retry_interval)
        if ret != 0:
            raise RuntimeError(
                f"Ascend transfer engine init returned {ret} "
                f"(store_url={self.store_url}). Check DisaggConfig.store_url."
            )

    def batch_register(self, ptrs: Iterable[int], lengths: Iterable[int]) -> None:
        self.engine.batch_register_memory(list(ptrs), list(lengths))

    def batch_transfer_sync(
        self,
        decode_session_id: str,
        src_addrs: Iterable[int],
        dst_addrs: Iterable[int],
        lengths: Iterable[int],
    ) -> int:
        """Transfer KV data to decode's NPU memory via RDMA.

        Args:
            decode_session_id: Decode peer's session id ("<ip>:<rpc_port>").
                               Prefill uses this to locate and write to remote memory.
        """
        # memfabric_hybrid 1.0.5 exposes batch_transfer_sync_write; fall back
        # to the plain batch_transfer_sync name for forward compatibility with
        # future versions that may rename the API.
        if hasattr(self.engine, "batch_transfer_sync_write"):
            return self.engine.batch_transfer_sync_write(
                decode_session_id, list(src_addrs), list(dst_addrs), list(lengths)
            )
        return self.engine.batch_transfer_sync(
            decode_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )
