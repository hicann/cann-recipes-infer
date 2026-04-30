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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from executor.online.kv_transfer import AscendKVReceiver, KVPoll
from executor.core.types_.types import MTPInfo

logger = logging.getLogger(__name__)


@dataclass
class DecodeRequest:
    req: object
    kv_receiver: object | None = None
    waiting_for_input: bool = False
    metadata_buffer_index: int = -1


class DecodePreallocQueue:
    MAX_RETRIES = 15
    RETRY_INTERVAL_S = 1.0

    def __init__(
        self,
        kv_transfer_manager,
        kv_cache_manager,
        metadata_pool,
        transfer_queue,
        running_requests,
        num_reserved_decode_tokens: int,
        max_prefill_tokens: int,
        tp_cpu_group=None,
    ):
        self.kv_transfer_manager = kv_transfer_manager
        self.kv_cache_manager = kv_cache_manager
        self.metadata_pool = metadata_pool
        self.transfer_queue = transfer_queue
        # Live reference to scheduler.running_requests — consulted on every
        # admission decision to gauge pipeline depth.
        self.running_requests = running_requests
        self.num_reserved_decode_tokens = num_reserved_decode_tokens
        # Per-request prompt length cap. D rejects oversized prompts at admission
        # before any KV / metadata / bootstrap state is allocated; P also rejects
        # at its own _schedule_prefill_batch (different YAML may carry different
        # values) and notifies D via abort(), so both endpoints converge on
        # KVPoll.Failed regardless of which side caught it first.
        self.max_prefill_tokens = max_prefill_tokens
        # Gloo CPU group for receiver.poll consensus (MIN) across TP ranks.
        # None for single-rank setups; all-reduce becomes a no-op.
        self.tp_cpu_group = tp_cpu_group
        self.queue: list[DecodeRequest] = []
        self.pending_reqs: list[DecodeRequest] = []
        self._ensure_retry_count: dict[str, int] = {}
        self._ensure_last_attempt_time: dict[str, float] = {}
        self._query_retry_count: dict[int, int] = {}

    def add(self, req) -> None:
        decode_req = self._create_receiver_and_enqueue(req)
        prefill_dp_rank = self._resolve_prefill_dp_rank(req)
        if prefill_dp_rank is not None:
            # try_ensure_parallel_info populates target_rank_map_table, which
            # fetch_bootstrap_infos requires. Without this, init() returns
            # immediately with KVPoll.Failed (mapping is None → empty list).
            has_info = self.kv_transfer_manager.try_ensure_parallel_info(req.bootstrap_addr)
            if has_info:
                decode_req.kv_receiver.init(prefill_dp_rank)
                return
        self.pending_reqs.append(decode_req)

    def _resolve_prefill_dp_rank(self, req):
        if req.disagg_prefill_dp_rank is not None and req.disagg_prefill_dp_rank >= 0:
            return req.disagg_prefill_dp_rank
        prefill_info = self.kv_transfer_manager.prefill_info_table.get(req.bootstrap_addr)
        if prefill_info is None:
            return None
        if prefill_info.dp_size == 1:
            return 0
        return None

    def _create_receiver_and_enqueue(self, req) -> DecodeRequest:
        kv_receiver = AscendKVReceiver(req.bootstrap_room, self.kv_transfer_manager, req.bootstrap_addr)
        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req

    def _ensure_prefill_info(self, addr_to_reqs: dict[str, list[DecodeRequest]]):
        ready: dict[str, list[DecodeRequest]] = {}
        remaining: list[DecodeRequest] = []

        now = time.monotonic()
        for bootstrap_addr, decode_reqs in addr_to_reqs.items():
            last_attempt = self._ensure_last_attempt_time.get(bootstrap_addr)
            if last_attempt is not None and now - last_attempt < self.RETRY_INTERVAL_S:
                remaining.extend(decode_reqs)
                continue

            self._ensure_last_attempt_time[bootstrap_addr] = now
            if self.kv_transfer_manager.try_ensure_parallel_info(bootstrap_addr):
                self._ensure_retry_count.pop(bootstrap_addr, None)
                self._ensure_last_attempt_time.pop(bootstrap_addr, None)
                ready[bootstrap_addr] = decode_reqs
                continue

            count = self._ensure_retry_count.get(bootstrap_addr, 0) + 1
            self._ensure_retry_count[bootstrap_addr] = count
            if count >= self.MAX_RETRIES:
                logger.warning(
                    "prefill %s parallel-info unreachable after %d retries; "
                    "aborting %d pending request(s)",
                    bootstrap_addr, self.MAX_RETRIES, len(decode_reqs),
                )
                for decode_req in decode_reqs:
                    decode_req.kv_receiver.abort()
            else:
                remaining.extend(decode_reqs)
        return ready, remaining

    def _resolve_pending_reqs(self) -> None:
        if not self.pending_reqs:
            return

        addr_to_reqs: dict[str, list[DecodeRequest]] = {}
        for decode_req in self.pending_reqs:
            addr_to_reqs.setdefault(decode_req.req.bootstrap_addr, []).append(decode_req)

        ready_addrs, remaining = self._ensure_prefill_info(addr_to_reqs)
        resolved: list[tuple[DecodeRequest, int]] = []

        for bootstrap_addr, decode_reqs in ready_addrs.items():
            need_query: list[DecodeRequest] = []
            for decode_req in decode_reqs:
                prefill_dp_rank = self._resolve_prefill_dp_rank(decode_req.req)
                if prefill_dp_rank is not None:
                    resolved.append((decode_req, prefill_dp_rank))
                else:
                    need_query.append(decode_req)

            if need_query:
                rooms = [decode_req.req.bootstrap_room for decode_req in need_query]
                room_to_rank = self.kv_transfer_manager.query_prefill_dp_ranks(bootstrap_addr, rooms)
                for decode_req in need_query:
                    prefill_dp_rank = room_to_rank.get(decode_req.req.bootstrap_room)
                    if prefill_dp_rank is not None and int(prefill_dp_rank) >= 0:
                        self._query_retry_count.pop(decode_req.req.bootstrap_room, None)
                        resolved.append((decode_req, int(prefill_dp_rank)))
                    else:
                        count = self._query_retry_count.get(decode_req.req.bootstrap_room, 0) + 1
                        self._query_retry_count[decode_req.req.bootstrap_room] = count
                        if count >= self.MAX_RETRIES:
                            logger.warning(
                                "request %s: prefill_dp_rank query failed for room=%s "
                                "after %d retries; aborting",
                                decode_req.req.request_id,
                                decode_req.req.bootstrap_room,
                                self.MAX_RETRIES,
                            )
                            decode_req.kv_receiver.abort()
                        else:
                            remaining.append(decode_req)

        self.pending_reqs = remaining
        for decode_req, prefill_dp_rank in resolved:
            decode_req.kv_receiver.init(prefill_dp_rank)

    def _update_handshake_waiters(self) -> None:
        for decode_req in self.queue:
            if decode_req.waiting_for_input:
                continue
            poll = decode_req.kv_receiver.poll_and_all_reduce(group=self.tp_cpu_group)
            if poll == KVPoll.WaitingForInput:
                decode_req.waiting_for_input = True
            elif poll == KVPoll.Failed:
                logger.warning(
                    "handshake failed for room=%s addr=%s",
                    decode_req.req.bootstrap_room, decode_req.req.bootstrap_addr,
                )
                decode_req.req.is_finished = True
                decode_req.req.finish_reason = "error"
                self.transfer_queue.terminal_failed.append(decode_req)

    def pop_preallocated(self, next_n: int = 0) -> tuple[list[DecodeRequest], list[DecodeRequest]]:
        self._resolve_pending_reqs()
        self._update_handshake_waiters()
        n_wfi = sum(1 for r in self.queue if r.waiting_for_input)
        if self.queue or self.pending_reqs:
            logger.debug(
                "pop_preallocated: queue=%d pending=%d wfi=%d",
                len(self.queue), len(self.pending_reqs), n_wfi,
            )

        preallocated: list[DecodeRequest] = []
        failed: list[DecodeRequest] = []
        remaining: list[DecodeRequest] = []

        # Admission control inputs: each request already in the pipeline
        # (running + transfer-queue) reserves num_reserved_decode_tokens of
        # future decode growth.  We only admit a new request if its
        # input + reservation fits in the leftover budget.  Without this gate
        # high-QPS bursts greedily fill KV cache, then no running req can grow
        # to its next token → deadlock.
        block_size = self.kv_transfer_manager.block_size
        reserved_blocks_per_req = (
            self.num_reserved_decode_tokens + block_size - 1
        ) // block_size
        in_flight = len(self.running_requests) + len(self.transfer_queue.waiting)
        reserved_for_pipeline = reserved_blocks_per_req * in_flight
        free_blocks = min(
            (m.get_num_free_blocks() for m in self.kv_cache_manager.single_type_managers),
            default=0,
        )
        admittable_blocks = free_blocks - reserved_for_pipeline

        for decode_req in self.queue:
            room = decode_req.req.bootstrap_room
            if decode_req.req.is_finished:
                failed.append(decode_req)
                continue
            if not decode_req.waiting_for_input:
                remaining.append(decode_req)
                continue
            num_tokens = int(decode_req.req.input_ids.numel())
            if num_tokens > self.max_prefill_tokens:
                logger.warning(
                    "Dropping room=%s from decode prealloc: prompt_tokens=%d "
                    "exceeds max_prefill_tokens=%d",
                    room, num_tokens, self.max_prefill_tokens,
                )
                decode_req.req.is_finished = True
                decode_req.req.finish_reason = "prompt_too_long"
                decode_req.req.prompt_tokens = num_tokens
                # Surface via transfer_queue.terminal_failed so
                # advance_queues_consensus picks it up (kv_receiver.clear() +
                # _cleanup_terminal_request + StepOutput dispatch). When P's
                # own max_prefill_tokens is smaller, P also catches the overflow
                # in _schedule_prefill_batch and aborts the room — both sides
                # converge on KVPoll.Failed regardless of who detects first.
                self.transfer_queue.terminal_failed.append(decode_req)
                continue
            if self.metadata_pool.available_size() <= 0:
                logger.debug("preallocate skip room=%s: no metadata slot available", room)
                remaining.append(decode_req)
                continue
            # Pre-allocate paged-attention blocks for the prompt plus the MTP
            # lookahead so the first decode step's allocate_slots is a no-op.
            # Matches _schedule_decode_batch(num_new_tokens=1+next_n, lookahead=max(next_n-1,0)).
            # Extra blocks beyond what prefill transfers are harmless — prefill
            # only writes src_block_ids count blocks (zip stops at the shorter).
            required_tokens = num_tokens + self.num_reserved_decode_tokens
            required_blocks = (required_tokens + block_size - 1) // block_size
            if required_blocks > admittable_blocks:
                logger.debug(
                    "preallocate skip room=%s: admission rejected "
                    "(req=%d blocks, free=%d, reserved_for_pipeline=%d)",
                    room, required_blocks, free_blocks, reserved_for_pipeline,
                )
                remaining.append(decode_req)
                continue
            if not self.kv_cache_manager.allocate_slots(
                request_id=decode_req.req.request_id,
                computed_tokens=num_tokens,
                num_new_tokens=1 + next_n,
                lookahead_tokens=max(next_n - 1, 0),
            ):
                logger.debug("preallocate skip room=%s: KV cache slot allocation failed", room)
                remaining.append(decode_req)
                continue
            # Account for this admission so the next iteration's budget
            # reflects both the real input allocation and the future-decode
            # reservation we just promised this request.
            admittable_blocks -= required_blocks

            decode_req.metadata_buffer_index = self.metadata_pool.alloc()
            decode_req.req.metadata_buffer_index = decode_req.metadata_buffer_index
            dst_block_ids = self.kv_cache_manager.get_block_ids(decode_req.req.request_id)
            decode_req.kv_receiver.send_metadata(decode_req.metadata_buffer_index, dst_block_ids)
            self.transfer_queue.add(decode_req)
            preallocated.append(decode_req)

        self.queue = remaining
        return preallocated, failed


class DecodeTransferQueue:
    def __init__(self, metadata_pool, tp_cpu_group=None):
        self.metadata_pool = metadata_pool
        # Gloo CPU group for receiver.poll consensus (MIN) across TP ranks.
        self.tp_cpu_group = tp_cpu_group
        self.waiting: list[DecodeRequest] = []
        self.terminal_failed: list[DecodeRequest] = []

    def add(self, decode_req: DecodeRequest) -> None:
        self.waiting.append(decode_req)

    def _read_metadata(self, decode_req: DecodeRequest) -> dict | None:
        # Metadata is RDMA-written by prefill into our pre-allocated slot.
        # Read directly from the pool; returns None if slot still zeroed.
        return self.metadata_pool.read(decode_req.metadata_buffer_index)

    def _commit_metadata(self, decode_req: DecodeRequest) -> bool:
        """True iff request is resolved (committed or errored); caller removes it."""
        meta = self._read_metadata(decode_req)
        if meta is None:
            return False

        actual_room = meta.get("output_bootstrap_room")
        if actual_room is None:
            return False  # metadata not yet fully written; retry next tick
        if actual_room != decode_req.req.bootstrap_room:
            logger.warning(
                "metadata bootstrap_room mismatch: got %s expected %s",
                actual_room, decode_req.req.bootstrap_room,
            )
            decode_req.req.is_finished = True
            decode_req.req.finish_reason = "error"
            return True

        output_id = meta.get("output_id")
        if output_id is not None:
            decode_req.req.output_id_list.append(output_id)
        kv_len = meta.get("kv_len")
        if kv_len is not None:
            decode_req.req.computed_len = int(kv_len)
        mtp_spec_tokens = meta.get("mtp_spec_tokens")
        if mtp_spec_tokens is not None:
            decode_req.req.mtp_info = MTPInfo(
                spec_tokens=torch.tensor(mtp_spec_tokens, dtype=torch.long),
            )
        decode_req.req.is_prefill_done = True
        return True

    def pop_transferred(self) -> list[DecodeRequest]:
        ready = []
        indices_to_remove = set()

        for i, decode_req in enumerate(self.waiting):
            poll = decode_req.kv_receiver.poll_and_all_reduce(group=self.tp_cpu_group)
            if poll == KVPoll.Failed:
                logger.warning(
                    "request %s: KV transfer failed (room=%s) — marking as error",
                    decode_req.req.request_id, decode_req.req.bootstrap_room,
                )
                decode_req.req.is_finished = True
                decode_req.req.finish_reason = "error"
                decode_req.kv_receiver.clear()
                self.terminal_failed.append(decode_req)
                indices_to_remove.add(i)
            elif poll == KVPoll.Success:
                if self._commit_metadata(decode_req):
                    indices_to_remove.add(i)
                    decode_req.kv_receiver.clear()
                    if decode_req.req.finish_reason == "error":
                        self.terminal_failed.append(decode_req)
                    else:
                        ready.append(decode_req)
            elif poll in (KVPoll.Bootstrapping, KVPoll.WaitingForInput, KVPoll.Transferring):
                pass
            else:
                raise ValueError(f"Unexpected poll case: {poll}")

        for i in indices_to_remove:
            idx = self.waiting[i].metadata_buffer_index
            if idx is not None and idx >= 0:
                self.metadata_pool.free(idx)
                self.waiting[i].req.metadata_buffer_index = -1

        self.waiting = [
            entry for i, entry in enumerate(self.waiting) if i not in indices_to_remove
        ]
        return ready
