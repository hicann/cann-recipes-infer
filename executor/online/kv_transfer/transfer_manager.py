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

"""KVTransferManager: ZMQ listener + async RDMA transfer executor + bootstrap client.

Owns per-rank state for PD disaggregation KV transfer:
- ZMQ PULL socket (rank_port) — prefill receives transfer_info from decode;
  decode receives transfer-done status from prefill.
- Background transfer thread pool (PREFILL only) — submits _KVTransferTask
  for async RDMA.
- Bootstrap HTTP client — registers prefill rank / queries cluster topology.
- Session failure tracking — fast-abort after N consecutive failures per
  decode session.
- Metadata scratch buffer (PREFILL only) — 64-byte slots for RDMA-writing
  metadata into decode's pre-registered metadata buffer.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
from itertools import zip_longest  # noqa: F401 — retained for potential future use
from typing import Optional

import requests

from executor.core.config import DisaggConfig
from executor.online.kv_transfer.transfer_engine import (
    AscendTransferEngine,
    KVArgsRegisterInfo,
    KVPoll,
    PrefillServerInfo,
    TargetRankMapping,
    _KVTransferTask,
)
from executor.online.kv_transfer.buffer import MetadataBufferPool

try:
    import zmq
except ImportError:
    zmq = None


logger = logging.getLogger(__name__)

_SHUTDOWN_MSG = b"__KV_MANAGER_SHUTDOWN__"


class KVTransferManager:
    """Minimal KV manager foundation for paged-view adaptation."""

    # Decode session is blacklisted after this many consecutive transfer failures.
    SESSION_MAX_FAILURES = 3

    def __init__(
        self,
        disagg_config: DisaggConfig,
        attn_tp_size: int = 1,
        attn_tp_rank: int = 0,
        attn_cp_size: int = 1,
        attn_cp_rank: int = 0,
        attn_dp_size: int = 1,
        attn_dp_rank: int = 0,
        npu_id: int = 0,
        kv_cache_dtype: str | None = None,
        is_mla_backend: bool = False,
        metadata_pool: Optional[MetadataBufferPool] = None,
        bootstrap_timeout: float = 1800.0,
        waiting_timeout: float = 60.0,
        heartbeat_interval: float = 5.0,
        heartbeat_max_failures: int = 3,
    ):
        if not disagg_config.store_url:
            raise ValueError("KVTransferManager requires DisaggConfig.store_url to be set")
        if not disagg_config.local_ip:
            raise ValueError(
                "KVTransferManager requires DisaggConfig.local_ip to be set "
                "(server.py derives it from --ips[node_index])"
            )
        self.is_mla_backend = is_mla_backend
        self.disaggregation_mode = disagg_config.disaggregation_mode
        self.local_ip = disagg_config.local_ip
        logger.info("KVTransferManager rank_ip=%s", self.local_ip)
        self.transfer_engine = AscendTransferEngine(
            hostname=self.local_ip,
            npu_id=npu_id,
            disaggregation_mode=disagg_config.disaggregation_mode,
            store_url=disagg_config.store_url,
            is_store_creator_node=disagg_config.is_store_creator_node,
        )
        # DECODE owns the metadata_pool — it's the NPU HBM buffer that prefill
        # RDMA-writes into. PREFILL passes None: its RDMA source is the
        # per-thread _metadata_scratch pool allocated in _start_listener_thread.
        self.metadata_pool = metadata_pool
        self.state_lock = threading.RLock()
        self._running = threading.Event()
        self._running.set()
        self.request_status = {}
        self.prefill_info_table: dict[str, PrefillServerInfo] = {}
        self.target_rank_map_table: dict[str, TargetRankMapping] = {}
        self.room_to_addr: dict[int, str] = {}
        self.attn_dp_size = attn_dp_size
        self.attn_dp_rank = attn_dp_rank
        self.bootstrap_host = disagg_config.bootstrap_host
        self.bootstrap_port = disagg_config.bootstrap_port
        self.bootstrap_timeout = bootstrap_timeout
        self.waiting_timeout = waiting_timeout
        self.attn_tp_size = attn_tp_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_cp_size = attn_cp_size
        self.attn_cp_rank = attn_cp_rank
        self.block_size = 0  # set by register_memory()
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_data_ptrs: list[int] = []
        self.kv_data_lens: list[int] = []
        self.kv_item_lens: list[int] = []
        # attn_type per flat entry in kv_data_ptrs; PD side uses it to pick the
        # correct per-attn_type block id list when building transfer tuples.
        self.kv_entry_attn_types: list[str] = []
        self.kv_head_num: int | None = None
        self.total_kv_head_num: int | None = None
        self.context = None
        self.server_socket = None
        self.rank_port = self.transfer_engine.rank_port
        self.connection_pool = {}
        self.connection_lock = threading.Lock()
        # Key: bootstrap_room → {decode_session_id: transfer_info_dict}
        self.transfer_infos: dict[int, dict[str, dict]] = {}
        self.transfer_metadata: dict[int, dict] = {}
        self.prefill_response_tracker: dict[int, set[int]] = {}
        self.failure_records: dict[int, str] = {}
        self.addr_to_rooms_tracker: dict[str, set[int]] = {}
        self.listener_threads: list[threading.Thread] = []
        # Background transfer pool (PREFILL only): multiple concurrent RDMA
        # transfers to different decode sessions in parallel.
        self.transfer_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Session failure tracking: count consecutive failures per decode
        # session_id and blacklist after SESSION_MAX_FAILURES.
        self.session_failures: dict[str, int] = {}
        self.failed_sessions: set[str] = set()
        self.session_lock = threading.Lock()
        # PREFILL-side scratch buffer for metadata RDMA writes.
        self._metadata_scratch: Optional[MetadataBufferPool] = None
        # DECODE-side bootstrap liveness tracking: consecutive failure count
        # per bootstrap_addr; addrs crossing heartbeat_max_failures are evicted
        # and all their in-flight rooms marked Failed.
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_max_failures = heartbeat_max_failures
        self.heartbeat_failures: dict[str, int] = {}

        self._init_rank_socket()
        self._start_listener_thread(disagg_config.disaggregation_mode)

    def _init_rank_socket(self) -> None:
        if zmq is None:
            return
        self.context = zmq.Context()
        self.server_socket = self.context.socket(zmq.PULL)
        self.server_socket.setsockopt(zmq.RCVTIMEO, 100)
        # Bind on all interfaces; advertise self.local_ip as rank_ip. Decouples
        # accept-side (works even if local_ip resolution picked a "wrong" NIC)
        # from advertise-side (wrong NIC there yields an unreachable rank_ip,
        # which is detectable vs a silent listener on the wrong interface).
        self.rank_port = self.server_socket.bind_to_random_port("tcp://0.0.0.0")

    def _start_listener_thread(self, disaggregation_mode: str) -> None:
        if self.server_socket is None:
            return
        if disaggregation_mode == "PREFILL":
            t = threading.Thread(target=self._prefill_listener_loop, daemon=True)
            t.start()
            self.listener_threads.append(t)
            n_threads = int(os.environ.get("ASCEND_TRANSFER_THREADS", "4"))
            self.transfer_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=n_threads, thread_name_prefix="kv-transfer"
            )
            # Scratch metadata buffer: one slot per transfer thread.
            self._metadata_scratch = MetadataBufferPool(capacity=n_threads)
        elif disaggregation_mode == "DECODE":
            t = threading.Thread(target=self._decode_listener_loop, daemon=True)
            t.start()
            self.listener_threads.append(t)
            # Bootstrap liveness heartbeat (decode side): periodically pings
            # each known prefill bootstrap; N consecutive failures evict the
            # addr and abort every in-flight room bound to it.
            hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
            hb.start()
            self.listener_threads.append(hb)

    @classmethod
    def connect(cls, endpoint: str, key_suffix: tuple = ()):
        """Get or create a ZMQ PUSH socket for the given endpoint.

        Args:
            endpoint: ZMQ endpoint string (e.g. "tcp://host:port").
            key_suffix: Optional extra key dimensions (e.g. (dp, cp, tp))
                to isolate connections when multiple ranks share an endpoint.
        """
        if zmq is None:
            raise RuntimeError("zmq is required for rank-level PD handshake")
        if not hasattr(cls, "socket_cache"):
            cls.socket_cache = {}
            cls.socket_locks = {}
            cls._global_lock = threading.Lock()
            cls._ctx = zmq.Context.instance()
        cache_key = (endpoint, *key_suffix)
        with cls._global_lock:
            if cache_key not in cls.socket_cache:
                sock = cls._ctx.socket(zmq.PUSH)
                try:
                    sock.connect(endpoint)
                except Exception:
                    sock.close(linger=0)
                    raise
                cls.socket_cache[cache_key] = sock
                cls.socket_locks[cache_key] = threading.Lock()
            return cls.socket_cache[cache_key], cls.socket_locks[cache_key]

    def _prefill_listener_loop(self) -> None:
        while self._running.is_set():
            try:
                msg = self.server_socket.recv_multipart()
            except Exception:
                if not self._running.is_set():
                    break
                continue
            if len(msg) == 1 and msg[0] == _SHUTDOWN_MSG:
                break
            if len(msg) < 5:
                logger.warning("prefill listener got message with too few parts: %d", len(msg))
                continue
            try:
                room = int(msg[0].decode("ascii"))
                endpoint_ip = msg[1].decode("ascii")
                endpoint_port = int(msg[2].decode("ascii"))
                transfer_info = json.loads(msg[3].decode())
                required_dst_info_num = int(msg[4].decode("ascii"))
            except ValueError as exc:
                # ValueError covers UnicodeDecodeError and json.JSONDecodeError.
                logger.warning("prefill listener got malformed message: %s", exc)
                continue
            transfer_info["endpoint_ip"] = endpoint_ip
            transfer_info["endpoint_port"] = endpoint_port
            # Use decode_session_id as the inner key. Falls back to "ip:port"
            # for older decode peers that don't send a session_id.
            key = transfer_info.get("decode_session_id") or f"{endpoint_ip}:{endpoint_port}"
            # Re-registration from a previously-failed decode session counts
            # as recovery — clear it out of the blacklist + failure count so
            # the next transfer is allowed to retry.
            with self.session_lock:
                self.failed_sessions.discard(key)
                self.session_failures.pop(key, None)
            with self.state_lock:
                room_infos = self.transfer_infos.setdefault(room, {})
                room_infos[key] = transfer_info
                ready = len(room_infos) >= required_dst_info_num
            logger.debug(
                "PD prefill listener: room=%s session=%.8s collected=%d/%d ready=%s",
                room, key, len(room_infos), required_dst_info_num, ready,
            )
            if ready:
                self.update_status(room, KVPoll.WaitingForInput)

    def _decode_listener_loop(self) -> None:
        while self._running.is_set():
            try:
                msg = self.server_socket.recv_multipart()
            except Exception:
                if not self._running.is_set():
                    break
                continue
            if len(msg) == 1 and msg[0] == _SHUTDOWN_MSG:
                break
            if len(msg) < 3:
                continue
            try:
                bootstrap_room = int(msg[0].decode("ascii"))
                status = KVPoll(int(msg[1].decode("ascii")))
                prefill_rank = int(msg[2].decode("ascii"))
            except ValueError as exc:
                # ValueError covers UnicodeDecodeError.
                logger.warning("decode listener got malformed message: %s", exc)
                continue

            if status == KVPoll.Success:
                # Metadata is delivered via RDMA (written into metadata_pool slot
                # before this status signal is sent). Decode reads the slot locally.
                with self.state_lock:
                    tracker = self.prefill_response_tracker.setdefault(
                        bootstrap_room, set()
                    )
                    tracker.add(prefill_rank)
                    addr = self.room_to_addr.get(bootstrap_room)
                    mapping = self.target_rank_map_table.get(addr) if addr else None
                    expected = mapping.required_prefill_response_num if mapping else 1
                    all_done = len(tracker) >= expected
                if all_done:
                    self.update_status(bootstrap_room, KVPoll.Success)
            elif status == KVPoll.Failed:
                # MLA defense-in-depth: only the target_tp_rank's Failed is
                # authoritative. Non-targets must short-circuit silently on the
                # prefill side (conn.py AscendKVSender.send), but ignore any
                # stray Faileds from them here so version-skew / legacy peers
                # cannot poison a room whose target delivered Success.
                if self.is_mla_backend:
                    with self.state_lock:
                        addr = self.room_to_addr.get(bootstrap_room)
                        mapping = self.target_rank_map_table.get(addr) if addr else None
                    if mapping is not None and mapping.target_cp_ranks:
                        target_unique = (
                            mapping.target_tp_rank * max(1, self.attn_cp_size)
                            + mapping.target_cp_ranks[0]
                        )
                        if prefill_rank != target_unique:
                            logger.debug(
                                "[PD-KV] ignoring non-target Failed room=%s prefill_rank=%d target=%d",
                                bootstrap_room, prefill_rank, target_unique,
                            )
                            continue
                self.record_failure(
                    bootstrap_room,
                    "Failed to receive KV transfer done signal from prefill rank",
                )
                self.update_status(bootstrap_room, KVPoll.Failed)

    def _heartbeat_loop(self) -> None:
        """Decode-side liveness checker for known prefill bootstrap addresses.

        Polls /bootstrap/health on a fixed interval; ``heartbeat_max_failures``
        consecutive failures evict the address and mark every outstanding room
        bound to it as Failed so the scheduler doesn't wait forever on a dead
        prefill instance.
        """
        while self._running.is_set():
            # Sleep in short slices so shutdown doesn't wait a full interval.
            slept = 0.0
            while slept < self.heartbeat_interval and self._running.is_set():
                time.sleep(min(0.5, self.heartbeat_interval - slept))
                slept += 0.5
            if not self._running.is_set():
                break

            with self.state_lock:
                addrs = list(self.prefill_info_table.keys())
            for addr in addrs:
                ok = False
                try:
                    resp = requests.get(
                        f"http://{addr}/bootstrap/health", timeout=(2, 3),
                    )
                    ok = resp.status_code == 200
                except Exception as exc:
                    logger.debug("bootstrap health probe %s failed: %s", addr, exc)

                if ok:
                    self.heartbeat_failures[addr] = 0
                    continue
                n = self.heartbeat_failures.get(addr, 0) + 1
                self.heartbeat_failures[addr] = n
                if n >= self.heartbeat_max_failures:
                    logger.warning(
                        "bootstrap %s failed %d consecutive heartbeats, "
                        "evicting and failing all its in-flight rooms",
                        addr, n,
                    )
                    self._handle_node_failure(addr)

    def _handle_node_failure(self, bootstrap_addr: str) -> None:
        """Evict a dead prefill bootstrap: drop topology caches for it and
        mark every room currently tracked under that addr as Failed so decode
        callers can abort cleanly instead of waiting on waiting_timeout."""
        with self.state_lock:
            self.prefill_info_table.pop(bootstrap_addr, None)
            self.target_rank_map_table.pop(bootstrap_addr, None)
            rooms = set(self.addr_to_rooms_tracker.pop(bootstrap_addr, set()))
        self.heartbeat_failures.pop(bootstrap_addr, None)
        for room in rooms:
            self.record_failure(
                room, f"bootstrap {bootstrap_addr} unreachable",
            )
            self.update_status(room, KVPoll.Failed)

    def _is_session_dead(self, session_id: str) -> bool:
        with self.session_lock:
            return session_id in self.failed_sessions

    def _record_session_failure(self, session_id: str) -> None:
        """Bump failure count; mark session dead after SESSION_MAX_FAILURES."""
        with self.session_lock:
            n = self.session_failures.get(session_id, 0) + 1
            self.session_failures[session_id] = n
            if n >= self.SESSION_MAX_FAILURES:
                self.failed_sessions.add(session_id)
                logger.warning(
                    "decode session %s marked dead after %d consecutive failures",
                    session_id, n,
                )

    def _record_session_success(self, session_id: str) -> None:
        """Reset failure count on success."""
        with self.session_lock:
            self.session_failures.pop(session_id, None)

    def execute_transfer_task(self, task: _KVTransferTask) -> None:
        prefill_rank = task.sender.prefill_unique_rank()
        # Per-request lifecycle log fires every request × prefill rank — keep
        # at DEBUG so high-QPS production logs aren't flooded.  Failures and
        # the "complete" line at the bottom share the same level (WARNING for
        # failure, DEBUG for success).
        logger.debug(
            "KV transfer started: room=%s prefill_rank=%d targets=%d",
            task.bootstrap_room, prefill_rank, len(task.bootstrap_infos),
        )
        try:
            for target in task.bootstrap_infos:
                decode_session_id = target.get("decode_session_id")
                if not decode_session_id:
                    raise RuntimeError(
                        f"missing decode_session_id for room={task.bootstrap_room}"
                    )
                if self._is_session_dead(decode_session_id):
                    raise RuntimeError(
                        f"decode session {decode_session_id} marked dead, aborting"
                    )
                # MLA non-target TP ranks receive dummy targets: skip RDMA +
                # metadata write but still count as session success so this
                # decode session isn't marked dead. Decode's _commit_metadata
                # waits for the real target's metadata-write to arrive.
                if target.get("is_dummy", False):
                    self._record_session_success(decode_session_id)
                    continue
                dst_kv_ptrs = target.get("dst_kv_ptrs", [])
                dst_kv_item_lens = target.get("dst_kv_item_lens", [])
                dst_attn_tp_size = target.get("dst_attn_tp_size", self.attn_tp_size)
                src_block_ids = task.src_block_ids
                dst_block_ids = target.get("dst_block_ids", {})
                if dst_attn_tp_size == self.attn_tp_size:
                    src_addrs, dst_addrs, byte_lens = self.build_transfer_blocks(
                        src_block_ids, dst_block_ids, dst_kv_ptrs, dst_kv_item_lens,
                    )
                else:
                    src_addrs, dst_addrs, byte_lens = self.build_tp_slice_blocks(
                        src_block_ids, dst_block_ids, dst_kv_ptrs, dst_kv_item_lens,
                        dst_attn_tp_size, target.get("dst_tp_rank", 0),
                    )
                logger.debug(
                    "RDMA transfer: room=%s rank=%d blocks=%d session=%.8s",
                    task.bootstrap_room, prefill_rank, len(src_addrs), decode_session_id,
                )
                if src_addrs:
                    ret = self.transfer_engine.batch_transfer_sync(
                        decode_session_id, src_addrs, dst_addrs, byte_lens,
                    )
                    logger.debug(
                        "RDMA done: room=%s rank=%d ret=%s",
                        task.bootstrap_room, prefill_rank, ret,
                    )
                    if ret not in (None, 0):
                        self._record_session_failure(decode_session_id)
                        raise RuntimeError(f"batch_transfer_sync returned {ret}")
                # RDMA-write metadata into decode's pre-registered slot.
                self._write_metadata_rdma(decode_session_id, target, task.metadata)
                self._record_session_success(decode_session_id)
            self.update_status(task.bootstrap_room, KVPoll.Success)
        except Exception as exc:
            logger.warning(
                "KV transfer failed: room=%s prefill_rank=%d error=%s",
                task.bootstrap_room, prefill_rank, exc,
            )
            self.update_status(task.bootstrap_room, KVPoll.Failed)
            self.sync_status_to_decode_endpoints(
                task.bootstrap_room,
                KVPoll.Failed,
                prefill_rank,
            )
            return
        logger.debug(
            "KV transfer complete: room=%s prefill_rank=%d",
            task.bootstrap_room, prefill_rank,
        )
        # Metadata is RDMA-written into decode's slot above. ZMQ carries only
        # the status signal; decode reads metadata from its local buffer.
        self.sync_status_to_decode_endpoints(
            task.bootstrap_room,
            KVPoll.Success,
            prefill_rank,
        )

    def _write_metadata_rdma(self, decode_session_id: str, target: dict, metadata: dict) -> None:
        """Pack metadata into a scratch slot and RDMA-write it to decode.

        Uses one slot from self._metadata_scratch (prefill-side registered buffer)
        to hold the packed bytes, then calls batch_transfer_sync to write into
        the decode's pre-registered metadata buffer slot.
        """
        scratch = self._metadata_scratch
        if scratch is None:
            return  # DECODE side (no scratch); only PREFILL writes metadata.
        dst_ptr = target.get("metadata_slot_ptr")
        slot_size = target.get("metadata_slot_size", 0)
        if not dst_ptr or not slot_size:
            logger.warning(
                "transfer_info missing metadata_slot_ptr/size; skipping metadata write"
            )
            return
        scratch_idx = scratch.alloc()
        try:
            scratch.write(scratch_idx, metadata)
            ret = self.transfer_engine.batch_transfer_sync(
                decode_session_id,
                [scratch.slot_ptr(scratch_idx)],
                [int(dst_ptr)],
                [int(slot_size)],
            )
            if ret not in (None, 0):
                raise RuntimeError(f"metadata batch_transfer_sync returned {ret}")
        finally:
            scratch.free(scratch_idx)

    def update_status(self, bootstrap_room: int, status: KVPoll) -> None:
        with self.state_lock:
            old = self.request_status.get(bootstrap_room)
            if old is None:
                self.request_status[bootstrap_room] = status
            elif status == KVPoll.Bootstrapping:
                # Bootstrapping is the initial state — never overwrite an
                # existing entry (setdefault semantics).
                pass
            elif status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(old, status)

    def record_failure(self, bootstrap_room: int, reason: str) -> None:
        with self.state_lock:
            self.failure_records[bootstrap_room] = reason
            self.transfer_metadata[bootstrap_room] = {"failure_reason": reason}

    def track_room(self, bootstrap_addr: str, bootstrap_room: int) -> None:
        if not bootstrap_addr:
            return
        with self.state_lock:
            self.addr_to_rooms_tracker.setdefault(bootstrap_addr, set()).add(
                bootstrap_room
            )
            self.room_to_addr[bootstrap_room] = bootstrap_addr

    def cleanup_room(
        self, bootstrap_room: int, bootstrap_addr: str | None = None
    ) -> None:
        with self.state_lock:
            self.request_status.pop(bootstrap_room, None)
            self.transfer_infos.pop(bootstrap_room, None)
            self.transfer_metadata.pop(bootstrap_room, None)
            self.prefill_response_tracker.pop(bootstrap_room, None)
            self.room_to_addr.pop(bootstrap_room, None)
            if bootstrap_addr and bootstrap_addr in self.addr_to_rooms_tracker:
                rooms = self.addr_to_rooms_tracker[bootstrap_addr]
                rooms.discard(bootstrap_room)
                if not rooms:
                    self.addr_to_rooms_tracker.pop(bootstrap_addr, None)
            self.failure_records.pop(bootstrap_room, None)

    def get_status(self, bootstrap_room: int) -> Optional[KVPoll]:
        with self.state_lock:
            return self.request_status.get(bootstrap_room)

    def get_transfer_info(self, bootstrap_room: int):
        with self.state_lock:
            return list(self.transfer_infos.get(bootstrap_room, {}).values()) or None

    def sync_status_to_decode_endpoints(
        self,
        bootstrap_room: int,
        status: KVPoll,
        prefill_rank: int,
    ) -> None:
        """Notify decode endpoints that transfer is done (success/failed).

        Metadata is NOT carried here — it's delivered via RDMA into the
        decode's metadata buffer slot before this signal is sent.
        """
        with self.state_lock:
            transfer_infos = list(self.transfer_infos.get(bootstrap_room, {}).values())
        for info in transfer_infos:
            endpoint = f"tcp://{info['endpoint_ip']}:{info['endpoint_port']}"
            sock, lock = self.connect(endpoint)
            parts = [
                str(bootstrap_room).encode("ascii"),
                str(int(status)).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
            with lock:
                try:
                    sock.send_multipart(parts)
                except Exception as exc:
                    logger.debug("status sync send failed (room=%s): %s", bootstrap_room, exc)

    def shutdown(self) -> None:
        self._running.clear()
        if self.transfer_executor is not None:
            self.transfer_executor.shutdown(wait=False)
        if (
            self.context is not None
            and self.server_socket is not None
            and zmq is not None
        ):
            try:
                wake_sock = self.context.socket(zmq.PUSH)
                wake_sock.setsockopt(zmq.LINGER, 0)
                wake_sock.connect(f"tcp://{self.local_ip}:{self.rank_port}")
                for _ in self.listener_threads:
                    wake_sock.send_multipart([_SHUTDOWN_MSG])
                wake_sock.close(0)
            except Exception as exc:
                logger.debug("listener wake-up during shutdown failed: %s", exc)
        for thread in self.listener_threads:
            thread.join(timeout=0.5)
        if self.server_socket is not None:
            try:
                if zmq is not None:
                    self.server_socket.setsockopt(zmq.LINGER, 0)
            except Exception as exc:
                logger.debug("server_socket setsockopt(LINGER) failed: %s", exc)
            try:
                self.server_socket.close(0)
            except Exception as exc:
                logger.debug("server_socket close failed: %s", exc)
        if self.context is not None:
            try:
                self.context.term()
            except Exception as exc:
                logger.debug("zmq.Context.term failed: %s", exc)
        # Clean up class-level socket cache to avoid leaking sockets across instances.
        if hasattr(self.__class__, "socket_cache"):
            for sock in self.__class__.socket_cache.values():
                try:
                    sock.close(linger=0)
                except Exception as exc:
                    logger.debug("cached socket close failed: %s", exc)
            self.__class__.socket_cache.clear()
            self.__class__.socket_locks.clear()

    def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
        if bootstrap_addr in self.prefill_info_table:
            return True
        if not bootstrap_addr:
            return False

        url = f"http://{bootstrap_addr}/bootstrap/route"
        try:
            resp = requests.get(url, timeout=5)
        except Exception:
            return False
        # Bootstrap returns 503 + {"detail": "..."} while ranks are still
        # registering; decoding that body as PrefillServerInfo would TypeError
        # and crash the whole decode instance.
        if resp.status_code != 200:
            return False
        try:
            payload = resp.json()
        except Exception:
            return False

        try:
            info = PrefillServerInfo(**payload)
        except TypeError:
            logger.warning("bootstrap /route returned unexpected payload: %s", payload)
            return False
        self.validate_prefill_info(info)
        self.prefill_info_table[bootstrap_addr] = info
        self.target_rank_map_table[bootstrap_addr] = self._resolve_rank_mapping(info)
        return True

    def validate_prefill_info(self, info: PrefillServerInfo) -> None:
        if info.block_size is not None and self.block_size not in (0, None):
            if info.block_size != self.block_size:
                raise ValueError(
                    f"Mismatched block_size: decode={self.block_size}, prefill={info.block_size}"
                )
        if info.kv_cache_dtype is not None and self.kv_cache_dtype is not None:
            if info.kv_cache_dtype != self.kv_cache_dtype:
                raise ValueError(
                    "Mismatched kv_cache_dtype: decode={}, prefill={}".format(
                        self.kv_cache_dtype,
                        info.kv_cache_dtype,
                    )
                )
        if info.attn_cp_size < 1 or info.attn_tp_size < 1:
            raise ValueError("Invalid prefill parallel info")

    def _resolve_rank_mapping(self, info: PrefillServerInfo) -> TargetRankMapping:
        # TP rank mapping. MLA's KV is replicated across prefill TP ranks, so
        # in the TP-mismatch case we pick a single real target inside the span
        # and mark the rest dummy. In equal-TP every decode rank already pairs
        # 1-to-1 with its matching prefill rank — no dummy fan-out needed.
        if self.attn_tp_size == info.attn_tp_size:
            target_tp_rank = self.attn_tp_rank
            required_prefill_response_num = 1
            required_dst_info_num = 1
            target_tp_ranks = [target_tp_rank]
        elif self.attn_tp_size > info.attn_tp_size:
            target_tp_rank = self.attn_tp_rank // max(
                1, self.attn_tp_size // info.attn_tp_size
            )
            required_dst_info_num = self.attn_tp_size // info.attn_tp_size
            required_prefill_response_num = 1
            target_tp_ranks = [target_tp_rank]
        else:
            span = max(1, info.attn_tp_size // self.attn_tp_size)
            start = self.attn_tp_rank * span
            target_tp_ranks = list(range(start, start + span))
            target_tp_rank = target_tp_ranks[0]
            required_dst_info_num = 1
            if self.is_mla_backend:
                # MLA: only target_tp_rank does real transfer, others are dummy.
                required_prefill_response_num = 1
            else:
                required_prefill_response_num = span

        if self.attn_cp_size == info.attn_cp_size:
            target_cp_ranks = [self.attn_cp_rank]
        else:
            target_cp_ranks = list(range(info.attn_cp_size))
            required_prefill_response_num *= max(
                1, info.attn_cp_size // self.attn_cp_size
            )

        return TargetRankMapping(
            target_tp_rank=target_tp_rank,
            target_tp_ranks=target_tp_ranks,
            target_cp_ranks=target_cp_ranks,
            required_dst_info_num=required_dst_info_num,
            required_prefill_response_num=required_prefill_response_num,
        )

    def fetch_bootstrap_infos(
        self, bootstrap_addr: str, prefill_dp_rank: int
    ) -> list[dict]:
        mapping = self.target_rank_map_table.get(bootstrap_addr)
        if mapping is None:
            logger.warning("fetch_bootstrap_infos: no mapping for addr=%s", bootstrap_addr)
            return []
        try:
            payload = requests.get(f"http://{bootstrap_addr}/bootstrap/route", timeout=5).json()
        except Exception as e:
            logger.warning("fetch_bootstrap_infos: route request failed addr=%s: %s", bootstrap_addr, e)
            return []
        ranks = payload.get("ranks", {})
        results = []
        for tp in mapping.target_tp_ranks:
            for cp in mapping.target_cp_ranks:
                key = f"{prefill_dp_rank},{cp},{tp}"
                rank = ranks.get(key)
                if rank is None:
                    logger.warning(
                        "fetch_bootstrap_infos: key=%s not in ranks=%s addr=%s",
                        key, list(ranks.keys()), bootstrap_addr,
                    )
                    return []
                # Expose tp_rank so send_metadata can compute per-target
                # is_dummy for MLA without re-parsing the key string.
                results.append({"key": key, "tp_rank": tp, **rank})
        return results

    @staticmethod
    def query_prefill_dp_ranks(
        bootstrap_addr: str, bootstrap_rooms: list[int]
    ) -> dict[int, int]:
        if not bootstrap_addr:
            return {}
        try:
            payload = requests.post(
                f"http://{bootstrap_addr}/bootstrap/query_dp_ranks",
                json={"bootstrap_rooms": bootstrap_rooms}, timeout=5,
            ).json()
            return {int(k): int(v) for k, v in payload.items()}
        except Exception:
            return {}

    def _register_to_bootstrap(self) -> None:
        """Register this rank to the Prefill bootstrap server (PUT /bootstrap/register_rank).

        Retries because the bootstrap uvicorn thread may not yet be listening
        when non-leader prefill ranks call register_memory — a single-shot PUT
        would silently drop them and leave is_ready stuck at False for tp>1."""
        if self.disaggregation_mode != "PREFILL":
            return
        if not self.bootstrap_host or self.bootstrap_port < 0:
            logger.error("bootstrap_host/port not set; skipping registration")
            return
        url = f"http://{self.bootstrap_host}:{self.bootstrap_port}/bootstrap/register_rank"
        payload = {
            "attn_tp_size": self.attn_tp_size,
            "attn_tp_rank": self.attn_tp_rank,
            "attn_cp_size": self.attn_cp_size,
            "attn_cp_rank": self.attn_cp_rank,
            "attn_dp_size": self.attn_dp_size,
            "attn_dp_rank": self.attn_dp_rank,
            "rank_ip": self.local_ip,
            "rank_port": self.rank_port,
            "block_size": self.block_size,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
        max_attempts = 10
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                requests.put(url, json=payload, timeout=5).raise_for_status()
                logger.info(
                    "Registered rank (dp=%d, cp=%d, tp=%d) to bootstrap on attempt %d",
                    self.attn_dp_rank, self.attn_cp_rank, self.attn_tp_rank, attempt,
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    time.sleep(1.0)
        logger.error(
            "Failed to register rank (dp=%d, cp=%d, tp=%d) to bootstrap after %d attempts: %s",
            self.attn_dp_rank, self.attn_cp_rank, self.attn_tp_rank, max_attempts, last_exc,
        )

    def make_kv_args_register_info(
        self, metadata_buffer_index: int, dst_block_ids: dict[str, list[int]]
    ) -> KVArgsRegisterInfo:
        return KVArgsRegisterInfo(
            dst_kv_ptrs=list(self.kv_data_ptrs),
            dst_kv_item_lens=list(self.kv_item_lens),
            metadata_buffer_index=metadata_buffer_index,
            dst_block_ids={k: list(v) for k, v in dst_block_ids.items()},
            dst_tp_rank=self.attn_tp_rank,
            kv_data_lens=list(self.kv_data_lens) if self.kv_data_lens else None,
            kv_head_num=self.kv_head_num,
            total_kv_head_num=self.total_kv_head_num,
        )

    def register_memory(self, kv_cache_manager) -> None:
        """Register all physical KV cache tensors with the transfer engine.

        Queries the KV pool (KVCacheManager) directly for a flat list of
        (ptrs, data_lens, item_lens, attn_types); ModelWorker is not
        involved — the pool is the source of truth.
        """
        ptrs, lengths, item_lens, attn_types = (
            kv_cache_manager.get_contiguous_buf_infos()
        )
        # Source of truth: the same block_size that allocate_cache_tensors used.
        self.block_size = kv_cache_manager.cache_info.block_size
        if ptrs:
            self.kv_data_ptrs = ptrs
            self.kv_data_lens = lengths
            self.kv_item_lens = item_lens
            self.kv_entry_attn_types = attn_types
            self.transfer_engine.batch_register(ptrs, lengths)
        self._register_to_bootstrap()
        if self.metadata_pool is not None and self.metadata_pool.base_ptr:
            self.transfer_engine.batch_register(
                [self.metadata_pool.base_ptr], [self.metadata_pool.total_bytes],
            )
        if self._metadata_scratch is not None and self._metadata_scratch.base_ptr:
            self.transfer_engine.batch_register(
                [self._metadata_scratch.base_ptr],
                [self._metadata_scratch.total_bytes],
            )

    def get_kv_layout(self) -> dict:
        layout = {
            "dst_kv_ptrs": list(self.kv_data_ptrs),
            "dst_kv_item_lens": list(self.kv_item_lens),
        }
        if self.kv_data_lens:
            layout["kv_data_lens"] = list(self.kv_data_lens)
        if self.kv_head_num is not None:
            layout["kv_head_num"] = self.kv_head_num
        if self.total_kv_head_num is not None:
            layout["total_kv_head_num"] = self.total_kv_head_num
        return layout

    @staticmethod
    def _group_concurrent_contiguous(src_pages: list[int], dst_pages: list[int]):
        if not src_pages or not dst_pages:
            return []
        groups = []
        current_src = [src_pages[0]]
        current_dst = [dst_pages[0]]
        for src, dst, prev_src, prev_dst in zip(
            src_pages[1:], dst_pages[1:], src_pages, dst_pages
        ):
            if src == prev_src + 1 and dst == prev_dst + 1:
                current_src.append(src)
                current_dst.append(dst)
            else:
                groups.append((current_src, current_dst))
                current_src = [src]
                current_dst = [dst]
        groups.append((current_src, current_dst))
        return groups

    def build_transfer_blocks(
        self,
        src_block_ids_by_type: dict[str, list[int]],
        dst_block_ids_by_type: dict[str, list[int]],
        dst_kv_ptrs: list[int],
        dst_kv_item_lens: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Block-level RDMA triplets for equal-TP transfer.

        Consecutive block pairs (src_block_id, dst_block_id) are merged into a
        single RDMA call.  Requires src and dst item_lens to match — for TP
        mismatch, use build_tp_slice_blocks instead.
        """
        if len(self.kv_data_ptrs) != len(dst_kv_ptrs):
            raise ValueError(
                f"KV entry count mismatch: src={len(self.kv_data_ptrs)}, "
                f"dst={len(dst_kv_ptrs)}"
            )
        groups_cache: dict[str, list[tuple[list[int], list[int]]]] = {}

        def groups_for(attn_type: str):
            if attn_type in groups_cache:
                return groups_cache[attn_type]
            src_ids = src_block_ids_by_type.get(attn_type, [])
            dst_ids = dst_block_ids_by_type.get(attn_type, [])
            groups = self._group_concurrent_contiguous(src_ids, dst_ids)
            groups_cache[attn_type] = groups
            return groups

        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        byte_lens: list[int] = []
        for i, (src_ptr, dst_ptr, src_item_len, dst_item_len, attn_type) in enumerate(
            zip(self.kv_data_ptrs, dst_kv_ptrs, self.kv_item_lens, dst_kv_item_lens, self.kv_entry_attn_types)
        ):
            if src_item_len != dst_item_len:
                raise ValueError(
                    f"kv item_len mismatch at entry {i} (attn_type={attn_type}): "
                    f"src={src_item_len}, dst={dst_item_len}"
                )
            for src_group, dst_group in groups_for(attn_type):
                src_addrs.append(src_ptr + int(src_group[0]) * src_item_len)
                dst_addrs.append(dst_ptr + int(dst_group[0]) * dst_item_len)
                byte_lens.append(src_item_len * len(src_group))
        return src_addrs, dst_addrs, byte_lens

    def build_tp_slice_blocks(  # pylint: disable=too-many-arguments
        self,
        src_block_ids_by_type: dict[str, list[int]],
        dst_block_ids_by_type: dict[str, list[int]],
        dst_kv_ptrs: list[int],
        dst_kv_item_lens: list[int],
        dst_attn_tp_size: int,
        dst_tp_rank: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Per-token RDMA triplets for TP-mismatched transfer.

        Token-major within each block, head-major within each token.  The slice
        offsets below are derived from item_lens, so GQA with num_kv_heads<tp
        (which makes multiple ranks replicate the same shard) and MLA (latent
        KV fully replicated) both collapse to the correct unique-head index
        without needing to know num_kv_heads.
        """
        if len(self.kv_data_ptrs) != len(dst_kv_ptrs):
            raise ValueError(
                f"KV entry count mismatch: src={len(self.kv_data_ptrs)}, "
                f"dst={len(dst_kv_ptrs)}"
            )
        src_addrs: list[int] = []
        dst_addrs: list[int] = []
        byte_lens: list[int] = []
        block_size = self.block_size
        for src_ptr, dst_ptr, src_item_len, dst_item_len, attn_type in zip(
            self.kv_data_ptrs, dst_kv_ptrs, self.kv_item_lens, dst_kv_item_lens,
            self.kv_entry_attn_types,
        ):
            s_bpt = src_item_len // block_size
            d_bpt = dst_item_len // block_size
            src_off, dst_off, length = self._compute_slice_offsets(
                s_bpt, d_bpt, dst_attn_tp_size, dst_tp_rank
            )
            for blk_s, blk_d in zip(
                src_block_ids_by_type.get(attn_type, []),
                dst_block_ids_by_type.get(attn_type, []),
            ):
                base_s = src_ptr + blk_s * block_size * s_bpt + src_off
                base_d = dst_ptr + blk_d * block_size * d_bpt + dst_off
                for t in range(block_size):
                    src_addrs.append(base_s + t * s_bpt)
                    dst_addrs.append(base_d + t * d_bpt)
                    byte_lens.append(length)
        return src_addrs, dst_addrs, byte_lens

    def _compute_slice_offsets(
        self,
        s_bpt: int,
        d_bpt: int,
        dst_attn_tp_size: int,
        dst_tp_rank: int,
    ) -> tuple[int, int, int]:
        """Per-token byte offsets for TP-sliced transfer.

        Returns (src_offset, dst_offset, length) relative to each per-token
        record.  The ``unique_head_idx = rank // replication`` step collapses
        GQA replicas (and MLA, which is fully replicated) to the correct head
        position; the ``% other_bpt`` wraps into whichever side has the
        smaller per-token record.
        """
        if self.attn_tp_size > dst_attn_tp_size:
            # Prefill has more TP ranks → this rank's shard lands at some
            # offset inside the larger decode slot.  src_replication counts
            # how many prefill ranks hold the same data (GQA with
            # num_kv_heads < prefill_tp, or MLA's full replication).
            src_replication = max(1, (self.attn_tp_size * s_bpt) // (dst_attn_tp_size * d_bpt))
            unique_head_idx = self.attn_tp_rank // src_replication
            return 0, (unique_head_idx * s_bpt) % d_bpt, s_bpt
        # Decode has more TP ranks → mirror: this decode rank reads a slice
        # from the larger prefill slot.
        dst_replication = max(1, (dst_attn_tp_size * d_bpt) // (self.attn_tp_size * s_bpt))
        unique_head_idx = dst_tp_rank // dst_replication
        return (unique_head_idx * d_bpt) % s_bpt, 0, d_bpt

