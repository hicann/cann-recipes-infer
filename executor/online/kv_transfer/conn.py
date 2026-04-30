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

"""Per-request KV transfer connection objects.

- AscendKVSender: prefill-side; submits transfer tasks to KVTransferManager's executor
- AscendKVReceiver: decode-side; handshakes via ZMQ, reads completed metadata
  from the RDMA-written slot

One object per request per side.
"""

from __future__ import annotations

import json
import logging
import os
import time

from executor.online.kv_transfer.transfer_engine import KVPoll, _KVTransferTask, all_reduce_poll


logger = logging.getLogger(__name__)

_FAKE_BOOTSTRAP_HOST = os.environ.get("FAKE_BOOTSTRAP_HOST")


class AscendKVSender:
    def __init__(
        self,
        bootstrap_room,
        kv_transfer_manager,
        bootstrap_addr,
        dp_rank: int | None = None,
        is_dp_leader: bool = True,
        is_dummy: bool = False,
    ):
        self.bootstrap_room = bootstrap_room
        self.kv_transfer_manager = kv_transfer_manager
        self.bootstrap_addr = _FAKE_BOOTSTRAP_HOST or bootstrap_addr
        self.bootstrap_infos = []
        self.init_time = time.time()
        self.conclude_state = None
        self.dp_rank = dp_rank
        self.is_dummy = is_dummy
        self.kv_transfer_manager.track_room(self.bootstrap_addr, bootstrap_room)
        self.kv_transfer_manager.update_status(bootstrap_room, KVPoll.Bootstrapping)
        if is_dummy:
            return
        if is_dp_leader and dp_rank is not None and bootstrap_addr:
            self._register_prefill_dp_rank()

    def _register_prefill_dp_rank(self) -> bool:
        """Register dp_rank for this bootstrap_room at sender init time."""
        if not self.bootstrap_addr or self.dp_rank is None:
            return False
        try:
            import requests
            requests.post(
                f"http://{self.bootstrap_addr}/bootstrap/register_dp_rank",
                json={"bootstrap_room": self.bootstrap_room, "dp_rank": self.dp_rank},
                timeout=5,
            ).raise_for_status()
            return True
        except Exception:
            return False

    def poll(self):
        if self.conclude_state is None:
            status = self.kv_transfer_manager.get_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping and self.init_time is not None:
                elapsed = time.time() - self.init_time
                if elapsed >= self.kv_transfer_manager.bootstrap_timeout:
                    self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Failed)
                    self.conclude_state = KVPoll.Failed
                    return KVPoll.Failed
            return status

        return self.conclude_state

    def poll_and_all_reduce(self, group=None):
        """Poll with TP/CP group consensus via all_reduce (min)."""
        return all_reduce_poll(self.poll(), group=group)

    def init(self):
        if self.is_dummy:
            return
        if not self.bootstrap_infos:
            infos = self.kv_transfer_manager.get_transfer_info(self.bootstrap_room)
            if infos:
                self.bootstrap_infos = infos
        # MLA: decode tags every entry with is_dummy (True for non-target
        # prefill ranks).  If all handshakes to this rank are tagged dummy,
        # we are not the real target — flip self.is_dummy so send()
        # short-circuits without running execute_transfer_task or sending a
        # premature Success that would satisfy required_prefill_response_num
        # before the real target finishes the RDMA metadata write.
        if (self.bootstrap_infos
                and all(t.get("is_dummy", False) for t in self.bootstrap_infos)):
            self.is_dummy = True

    def abort(self):
        self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed
        if self.is_dummy:
            return
        self.kv_transfer_manager.sync_status_to_decode_endpoints(
            self.bootstrap_room,
            KVPoll.Failed,
            self.prefill_unique_rank(),
        )

    def failure_exception(self):
        return RuntimeError(f"transfer failed for bootstrap_room={self.bootstrap_room}")

    def clear(self) -> None:
        self.kv_transfer_manager.cleanup_room(self.bootstrap_room, self.bootstrap_addr)

    def prefill_unique_rank(self) -> int:
        return (
            self.kv_transfer_manager.attn_tp_rank * self.kv_transfer_manager.attn_cp_size
            + self.kv_transfer_manager.attn_cp_rank
        )

    def send(self, src_block_ids: dict[str, list[int]], metadata: dict):
        """Enqueue a KV transfer task for the background worker.

        Async pattern: this method returns immediately after enqueuing; the
        background worker calls batch_transfer_sync and notifies the decode
        side when done.

        Args:
            src_block_ids: Prefill-side block ids keyed by attn_type
                (from ``kv_cache_manager.get_block_ids(request_id)``).
            metadata: Dict with request_id, output_bootstrap_room, output_id, kv_len.
        """
        if self.is_dummy:
            # Dummy sender (e.g. non-zero CP rank without ENABLE_ALL_CP_RANKS_FOR_TRANSFER):
            # no real data to transfer; mark success immediately.
            self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Success)
            return
        if not self.bootstrap_infos:
            self.abort()
            raise RuntimeError("AscendKVSender.send missing bootstrap_infos")

        bootstrap_infos = self.bootstrap_infos
        task = _KVTransferTask(
            bootstrap_room=self.bootstrap_room,
            bootstrap_infos=list(bootstrap_infos),
            src_block_ids={k: list(v) for k, v in src_block_ids.items()},
            metadata=metadata,
            sender=self,
        )
        if self.kv_transfer_manager.transfer_executor is not None:
            # Async path: mark as Transferring, submit to the thread pool.
            self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Transferring)
            self.kv_transfer_manager.transfer_executor.submit(
                self.kv_transfer_manager.execute_transfer_task, task
            )
        else:
            # Synchronous fallback (no executor, e.g. DECODE side or test).
            self.kv_transfer_manager.execute_transfer_task(task)


class AscendKVReceiver:
    def __init__(self, bootstrap_room, kv_transfer_manager, bootstrap_addr):
        self.bootstrap_room = bootstrap_room
        self.kv_transfer_manager = kv_transfer_manager
        self.bootstrap_addr = _FAKE_BOOTSTRAP_HOST or bootstrap_addr
        self.prefill_dp_rank = -1
        self.init_time = None
        self.conclude_state = None
        self.bootstrap_infos = None
        self.kv_transfer_manager.track_room(self.bootstrap_addr, bootstrap_room)
        self.kv_transfer_manager.update_status(bootstrap_room, KVPoll.Bootstrapping)

    def init(self, prefill_dp_rank):
        self.prefill_dp_rank = prefill_dp_rank
        self.bootstrap_infos = self.kv_transfer_manager.fetch_bootstrap_infos(
            self.bootstrap_addr,
            prefill_dp_rank,
        )
        if not self.bootstrap_infos:
            logger.warning(
                "KVReceiver init failed: empty bootstrap_infos for room=%s addr=%s dp_rank=%s",
                self.bootstrap_room, self.bootstrap_addr, prefill_dp_rank,
            )
            self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Failed)
            self.conclude_state = KVPoll.Failed
            return
        self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def send_metadata(self, metadata_buffer_index, dst_block_ids: dict[str, list[int]]):
        # Start the waiting_timeout timer now: from this point the decode is
        # actively waiting for the prefill to complete the KV transfer.
        self.init_time = time.time()
        register_info = self.kv_transfer_manager.make_kv_args_register_info(
            metadata_buffer_index,
            dst_block_ids,
        )
        # Metadata buffer slot address on decode side — prefill RDMA-writes here
        # after KV transfer.
        pool = self.kv_transfer_manager.metadata_pool
        base_transfer_info = {
            "metadata_buffer_index": register_info.metadata_buffer_index,
            "metadata_slot_ptr": pool.slot_ptr(register_info.metadata_buffer_index),
            "metadata_slot_size": pool.slot_size,
            "dst_block_ids": register_info.dst_block_ids,
            "dst_kv_ptrs": register_info.dst_kv_ptrs,
            "dst_kv_item_lens": register_info.dst_kv_item_lens,
            "dst_tp_rank": register_info.dst_tp_rank,
            "prefill_dp_rank": self.prefill_dp_rank,
            "bootstrap_addr": self.bootstrap_addr,
            # Decode's session_id so prefill can RDMA-write directly into our NPU memory.
            "decode_session_id": self.kv_transfer_manager.transfer_engine.session_id,
        }
        mapping = self.kv_transfer_manager.target_rank_map_table.get(self.bootstrap_addr)
        required_dst_info_num = mapping.required_dst_info_num if mapping else 1
        for bootstrap_info in self.bootstrap_infos:
            endpoint = (
                f"tcp://{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            )
            sock, lock = self.kv_transfer_manager.connect(endpoint)
            # MLA: latent KV is identical across prefill TP ranks, so only one
            # target rank does real RDMA; the others get an is_dummy=True
            # payload and return Success without touching memory.
            this_tp = bootstrap_info.get("tp_rank")
            is_dummy = (
                self.kv_transfer_manager.is_mla_backend
                and mapping is not None
                and this_tp is not None
                and this_tp != mapping.target_tp_rank
            )
            transfer_info = {
                **base_transfer_info,
                "is_dummy": is_dummy,
                "dst_attn_tp_size": self.kv_transfer_manager.attn_tp_size,
            }
            parts = [
                str(self.bootstrap_room).encode("ascii"),
                self.kv_transfer_manager.local_ip.encode("ascii"),
                str(self.kv_transfer_manager.rank_port).encode("ascii"),
                json.dumps(transfer_info).encode(),
                str(required_dst_info_num).encode("ascii"),
            ]
            with lock:
                sock.send_multipart(parts)

    def poll(self):
        if self.conclude_state is None:
            status = self.kv_transfer_manager.get_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput and self.init_time is not None:
                elapsed = time.time() - self.init_time
                if elapsed >= self.kv_transfer_manager.waiting_timeout:
                    logger.warning(
                        "[PD-KV] receiver waiting_timeout: room=%s elapsed=%.1fs",
                        self.bootstrap_room, elapsed,
                    )
                    self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Failed)
                    self.conclude_state = KVPoll.Failed
                    return KVPoll.Failed
            return status

        return self.conclude_state

    def poll_and_all_reduce(self, group=None):
        """Poll with TP/CP group consensus via all_reduce (min)."""
        return all_reduce_poll(self.poll(), group=group)

    def read_metadata(self, metadata_buffer_index: int):
        """Read the 64-byte metadata slot RDMA-written by prefill.

        Returns None if the slot is still zeroed (transfer not finished or
        metadata not yet visible). Decode's _decode_listener_loop marks the
        request status Success AFTER the RDMA write completes, so callers
        that see Success can safely call this.
        """
        return self.kv_transfer_manager.metadata_pool.read(metadata_buffer_index)

    def abort(self):
        self.kv_transfer_manager.update_status(self.bootstrap_room, KVPoll.Failed)
        self.conclude_state = KVPoll.Failed

    def clear(self) -> None:
        self.kv_transfer_manager.cleanup_room(self.bootstrap_room, self.bootstrap_addr)
