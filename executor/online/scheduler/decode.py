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

from executor.core.scheduler import Scheduler
from executor.core.types_.types import StepOutput
from executor.online.scheduler.queues import (
    DecodePreallocQueue,
    DecodeTransferQueue,
)

logger = logging.getLogger(__name__)


class DecodeDisaggScheduler(Scheduler):
    def __init__(
        self,
        tokenizer,
        config,
        kv_transfer_manager,
        kv_cache_manager,
        tp_cpu_group=None,
        input_truncated_len=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            config=config,
            input_truncated_len=input_truncated_len,
        )
        self.mode = 'online'
        if getattr(kv_transfer_manager, "attn_cp_size", 1) != 1:
            raise ValueError(
                f"Decode disaggregation requires cp_size == 1, "
                f"got {kv_transfer_manager.attn_cp_size}"
            )
        self.kv_transfer_manager = kv_transfer_manager
        self.kv_cache_manager = kv_cache_manager
        # Gloo CPU group for receiver.poll_and_all_reduce consensus across TP
        # ranks — keeps prealloc_queue / transfer_queue state lockstep so the
        # forward_batch running_requests agree.  None for single-rank setups.
        self.tp_cpu_group = tp_cpu_group
        self.transfer_queue = DecodeTransferQueue(
            kv_transfer_manager.metadata_pool, tp_cpu_group=tp_cpu_group,
        )
        self.prealloc_queue = DecodePreallocQueue(
            kv_transfer_manager=kv_transfer_manager,
            kv_cache_manager=kv_cache_manager,
            metadata_pool=kv_transfer_manager.metadata_pool,
            transfer_queue=self.transfer_queue,
            running_requests=self.running_requests,
            num_reserved_decode_tokens=self.config.num_reserved_decode_tokens,
            max_prefill_tokens=self.config.max_prefill_tokens,
            tp_cpu_group=tp_cpu_group,
        )
        self._pending_pd_request = None
        # Retraction sink: ids freed by _schedule_decode_batch's pressure-relief
        # path; merged into the next run_step's StepOutput so the dispatcher
        # surfaces them to clients (instead of silently leaking).
        self._retracted_ids: list[int] = []

    def set_pd_request_context(self, request_dict):
        self._pending_pd_request = dict(request_dict)

    def add_request(
        self, prompt, request_id=None, sampling_params=None, input_ids=None
    ):
        req_id = super().add_request(prompt, request_id, sampling_params, input_ids)
        req = self.waiting_queue.pop()
        request_dict = self._pending_pd_request or {}
        req.bootstrap_room = request_dict.get("bootstrap_room", req.bootstrap_room)
        req.bootstrap_host = request_dict.get("bootstrap_host", req.bootstrap_host)
        req.bootstrap_port = request_dict.get("bootstrap_port", req.bootstrap_port)
        req.disagg_prefill_dp_rank = request_dict.get(
            "disagg_prefill_dp_rank", req.disagg_prefill_dp_rank
        )
        self.prealloc_queue.add(req)
        self.last_added_request = req
        self._pending_pd_request = None
        return req_id

    def _cleanup_terminal_request(self, request) -> None:
        self.kv_cache_manager.free(request.request_id)
        metadata_idx = getattr(request, "metadata_buffer_index", -1)
        if isinstance(metadata_idx, int) and metadata_idx >= 0:
            self.kv_transfer_manager.metadata_pool.free(request.metadata_buffer_index)
            request.metadata_buffer_index = -1

    def advance_queues_consensus(self, engine):
        """Advance prealloc_queue / transfer_queue with TP-group consensus.

        Called on ALL ranks every main-loop iteration BEFORE phase negotiation.
        Internal pop_preallocated / pop_transferred use poll_and_all_reduce
        over self.tp_cpu_group (Gloo MIN) so all TP ranks move the same
        requests in lockstep.  Cannot be gated on is_dp_leader — the all-reduce
        requires full-group participation.

        Returns StepOutput for failed requests so the leader can dispatch them
        (workers' StepOutput is silently dropped by _dispatch_results).
        """
        self.prealloc_queue.pop_preallocated(next_n=engine.next_n)
        ready = self.transfer_queue.pop_transferred()
        failed = list(self.transfer_queue.terminal_failed)
        self.transfer_queue.terminal_failed.clear()
        for decode_req in ready:
            self.running_requests[decode_req.req.request_id] = decode_req.req
        failed_ids = []
        for decode_req in failed:
            # clear() drops bootstrap/transfer_info state in KVTransferManager; must
            # run before _cleanup_terminal_request or those rooms leak.
            if decode_req.kv_receiver is not None:
                decode_req.kv_receiver.clear()
            self.finished_requests[decode_req.req.request_id] = decode_req.req
            self._cleanup_terminal_request(decode_req.req)
            failed_ids.append(decode_req.req.request_id)
        return (
            StepOutput(next_tokens={}, finished_requests=failed_ids)
            if failed_ids else None
        )

    def run_step(self, engine, phase=None):
        if phase is None:
            return None
        # Always call super().run_step to keep TP HCCL collectives aligned.
        # Status logging happens inside super() via _log_step override.
        output = super().run_step(engine, phase="decode")
        if self._retracted_ids:
            if output is None:
                output = StepOutput(next_tokens={}, finished_requests=[])
            output.finished_requests.extend(self._retracted_ids)
            self._retracted_ids.clear()
        return output

    def _log_step(self, engine, batch, output) -> None:
        # See PD-Prefill _log_step: gate only on dummy batches; queue counts
        # alone can momentarily all be zero (e.g. prefill side, mid-handoff)
        # even when a real forward just ran.
        if batch.is_dummy:
            return
        n_running = len(self.running_requests)
        n_prealloc = len(self.prealloc_queue.queue)
        n_pending = len(self.prealloc_queue.pending_reqs)
        n_transfer = len(self.transfer_queue.waiting)
        kv_str = engine.kvcache_manager.format_usage()
        infer_ms = output.get("inference_time", 0.0) * 1000
        logger.info(
            f"[PD-Decode] step={self._step} batch={len(batch.requests)} "
            f"kv={kv_str} "
            f"running={n_running} prealloc={n_prealloc} pending={n_pending} "
            f"transfer={n_transfer} retracted={len(self._retracted_ids)} "
            f"infer={infer_ms:.2f}ms"
        )

    def _schedule_decode_batch(self, engine):
        """Retract under KV pressure to keep the decode loop making progress.

        When ``running_requests`` is non-empty but the base scheduler can't fit
        any of them in the current step's allocation budget, the returned batch
        is empty and ``_schedule_batch`` would fall through to a dummy batch
        forever — KV stays full, no req finishes, no req progresses.

        Retraction strategy: pick the cheapest victim (least output progress
        + largest input → most space freed for the least wasted work), free
        its KV, retry.  The last surviving request is also aborted if it
        can't fit alone, so the loop always terminates.

        TP coordination: every TP rank runs the same ``allocate_slots`` against
        identically-sized KV pools, so the empty-batch detection and victim
        selection (deterministic by request_id ordering and stable sort key)
        are consistent across ranks without an explicit collective.
        """
        batch = super()._schedule_decode_batch(engine)
        while (
            batch is not None
            and batch.is_empty()
            and self.running_requests
        ):
            self._retract_one()
            batch = super()._schedule_decode_batch(engine)
        return batch

    def _retract_one(self) -> None:
        # Cheapest to retract: smallest output_id_list (least progress wasted),
        # largest prompt_tokens (most KV freed).
        victim = min(
            self.running_requests.values(),
            key=lambda r: (len(r.output_id_list), -r.prompt_tokens),
        )
        is_only_one = len(self.running_requests) == 1
        victim.is_finished = True
        victim.finish_reason = "abort_oom" if is_only_one else "preempted_oom"
        self.running_requests.pop(victim.request_id, None)
        self.finished_requests[victim.request_id] = victim
        self._cleanup_terminal_request(victim)
        self._retracted_ids.append(victim.request_id)
        logger.warning(
            "[PD-Decode] retracted request %s (%s): output=%d prompt=%d running_left=%d",
            victim.request_id, victim.finish_reason,
            len(victim.output_id_list), victim.prompt_tokens,
            len(self.running_requests),
        )

    def has_work(self) -> bool:
        # Only count work that requires NPU (forward_batch). prealloc/transfer
        # queues are CPU-only RDMA waits — counting them causes tight-loop
        # dummy forward_batch calls which trigger HCCL AIV all-reduce counter
        # divergence and deadlock on Ascend.
        return bool(self.waiting_queue or self.running_requests)

    def _on_request_finished(self, request) -> None:
        self._cleanup_terminal_request(request)
