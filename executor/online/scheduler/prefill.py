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
import os
from collections import deque

from executor.core.scheduler import Scheduler
from executor.core.types_.types import StepOutput
from executor.online.kv_transfer import AscendKVSender, KVPoll

logger = logging.getLogger(__name__)


class PrefillDisaggScheduler(Scheduler):
    def __init__(
        self,
        tokenizer,
        config,
        kv_transfer_manager,
        kv_cache_manager,
        dp_rank=0,
        is_dp_leader=True,
        tp_cpu_group=None,
        input_truncated_len=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            config=config,
            input_truncated_len=input_truncated_len,
        )
        self.kv_transfer_manager = kv_transfer_manager
        self.kv_cache_manager = kv_cache_manager
        self.dp_rank = dp_rank
        self.is_dp_leader = is_dp_leader
        # Gloo CPU group for sender.poll_and_all_reduce consensus across TP
        # ranks — keeps bootstrap_queue / inflight_queue advancement lockstep
        # so forward_batch running_requests agree.  None for single-rank setups.
        self.tp_cpu_group = tp_cpu_group
        self._use_dummy_sender = (
            getattr(kv_transfer_manager, "attn_cp_rank", 0) != 0
            and not os.environ.get("ENABLE_ALL_CP_RANKS_FOR_TRANSFER")
        )
        self.bootstrap_queue = deque()
        self.inflight_queue = deque()
        self._pending_pd_request = None
        self._bootstrap_failures = []
        # Last drain count from _pop_inflight + bootstrap_failures, exposed so
        # _log_step can fold it into the per-step status line.
        self._last_drained_count: int = 0

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
        req.disagg_kv_sender = self._create_sender(req)
        self.bootstrap_queue.append(req)
        self.last_added_request = req
        self._pending_pd_request = None
        return req_id

    def _create_sender(self, request):
        return AscendKVSender(
            request.bootstrap_room,
            self.kv_transfer_manager,
            request.bootstrap_addr,
            dp_rank=self.dp_rank,
            is_dp_leader=self.is_dp_leader,
            is_dummy=self._use_dummy_sender,
        )

    def _cleanup_terminal_request(self, request) -> None:
        sender = getattr(request, "disagg_kv_sender", None)
        if sender is not None:
            # If we're tearing down before the transfer concluded (e.g. the
            # request was dropped for prompt_too_long after the bootstrap
            # handshake but before send()), notify decode via KVPoll.Failed
            # so it can free the KV blocks and metadata slot it preallocated.
            # Skipped when conclude_state is already set, both to avoid
            # corrupting a Success conclusion and because Failed paths have
            # already sent the signal upstream.
            if sender.conclude_state is None:
                sender.abort()
            sender.clear()
        # Release the paged-attention blocks prefill allocated for this request.
        # KV was already RDMA'd to decode at KVPoll.Success; prefill's copy is
        # no longer needed. Skipping this would leak blocks on the prefill rank
        # until the process dies.
        self.kv_cache_manager.free(request.request_id)

    def advance_queues_consensus(self, engine):
        """Advance bootstrap_queue with TP-group consensus.

        Called on ALL ranks every main-loop iteration BEFORE phase negotiation.
        Uses poll_and_all_reduce (MIN across tp_cpu_group, Gloo-backed CPU
        collective — safe on Ascend) so a request only moves to waiting_queue
        when every TP rank's listener has received decode's transfer_info.
        Plan B guarantees decode sends to every prefill rank (with is_dummy tag
        for non-targets), so the consensus actually converges.

        Bootstrap failures accumulate in `_bootstrap_failures` and are drained
        by the next `run_step` — no dispatch-able StepOutput here.
        """
        remaining = deque()
        while self.bootstrap_queue:
            req = self.bootstrap_queue.popleft()
            poll = req.disagg_kv_sender.poll_and_all_reduce(group=self.tp_cpu_group)
            if poll == KVPoll.WaitingForInput:
                req.disagg_kv_sender.init()
                self.waiting_queue.append(req)
            elif poll == KVPoll.Failed:
                logger.warning(
                    "request %s: bootstrap failed (room=%s addr=%s) — marking as error",
                    req.request_id, req.bootstrap_room, req.bootstrap_addr,
                )
                req.is_finished = True
                req.finish_reason = "error"
                self.finished_requests[req.request_id] = req
                self._cleanup_terminal_request(req)
                self._bootstrap_failures.append(req.request_id)
            else:
                remaining.append(req)
        self.bootstrap_queue = remaining
        return None

    def _schedule_prefill_batch(self, engine):
        return super()._schedule_prefill_batch(engine)

    def _pop_inflight(self):
        """Drain inflight_queue with TP-group consensus.

        Uses poll_and_all_reduce so every TP rank agrees on KV-transfer state
        before popping — otherwise ranks' inflight_queues drift and the next
        forward's running_requests disagree (HCCL deadlock).  Must be called
        on ALL ranks in lockstep (poll_and_all_reduce is a collective).
        """
        finished = []
        remaining = deque()
        while self.inflight_queue:
            req = self.inflight_queue.popleft()
            poll = req.disagg_kv_sender.poll_and_all_reduce(group=self.tp_cpu_group)
            if poll in (KVPoll.Success, KVPoll.Failed):
                req.is_finished = True
                req.finish_reason = "stop" if poll == KVPoll.Success else "error"
                if poll == KVPoll.Failed:
                    logger.warning(
                        "request %s: KV transfer failed (room=%s) — marking as error",
                        req.request_id, req.bootstrap_room,
                    )
                self.finished_requests[req.request_id] = req
                self._cleanup_terminal_request(req)
                finished.append(req.request_id)
            else:
                remaining.append(req)
        self.inflight_queue = remaining
        return finished

    def run_step(self, engine, phase=None):
        # Drain inflight + bootstrap failures on ALL ranks. _pop_inflight is a
        # Gloo collective and must see full-group participation; running it
        # here (before the phase branch) lets both forward and idle paths
        # surface completed requests through the same code.
        finished = self._pop_inflight()
        if self._bootstrap_failures:
            finished += list(self._bootstrap_failures)
            self._bootstrap_failures.clear()
        self._last_drained_count = len(finished)

        if phase is None:
            return (
                StepOutput(next_tokens={}, finished_requests=finished)
                if finished else None
            )

        # Every TP rank must enter forward_batch every iteration — skipping on
        # one rank while others proceed desyncs HCCL collectives → deadlock.
        # Status logging happens inside super().run_step via _log_step override.
        base_output = super().run_step(engine, phase="prefill")

        if finished and base_output:
            base_output.finished_requests.extend(finished)
        elif finished:
            return StepOutput(next_tokens={}, finished_requests=finished)
        return base_output

    def _log_step(self, engine, batch, output) -> None:
        # The base only invokes _log_step after a real forward, so any reach
        # here means we did work this step.  Skip only the synthetic
        # collective-alignment batch.
        if batch.is_dummy:
            return
        n_waiting = len(self.waiting_queue)
        n_bootstrap = len(self.bootstrap_queue)
        n_inflight = len(self.inflight_queue)
        kv_str = engine.kvcache_manager.format_usage()
        infer_ms = output.get("inference_time", 0.0) * 1000
        logger.info(
            f"[PD-Prefill] step={self._step} "
            f"batch_reqs={len(batch.requests)} batch_tokens={batch.total_tokens} "
            f"kv={kv_str} "
            f"waiting={n_waiting} bootstrap={n_bootstrap} inflight={n_inflight} "
            f"drained={self._last_drained_count} infer={infer_ms:.2f}ms"
        )

    def _build_send_metadata(self, request) -> dict:
        meta = {
            "request_id": request.request_id,
            "output_bootstrap_room": request.bootstrap_room,
            "output_id": request.output_id_list[-1] if request.output_id_list else 0,
            "kv_len": request.computed_len,
        }
        if request.mtp_info is not None and request.mtp_info.spec_tokens is not None:
            meta["mtp_spec_tokens"] = request.mtp_info.spec_tokens.cpu().tolist()
        return meta

    def _on_prefill_complete(self, request):
        # Intentionally does not call super() — prevents the base hook from
        # putting this request into running_requests; PD routes it through
        # inflight_queue and waits for the KV transfer to complete instead.
        if request.disagg_kv_sender is None:
            return
        try:
            metadata = self._build_send_metadata(request)
            src_block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            request.disagg_kv_sender.send(src_block_ids, metadata)
        except Exception as exc:
            logger.warning(
                "KV send failed for room=%s, failing request: %s",
                request.bootstrap_room, exc,
            )
            request.disagg_kv_sender.abort()
            request.is_finished = True
            request.finish_reason = "error"
            self.finished_requests[request.request_id] = request
            self._cleanup_terminal_request(request)
            self._bootstrap_failures.append(request.request_id)
            return
        self.inflight_queue.append(request)

    def has_work(self) -> bool:
        # Only count work that requires NPU (forward_batch). bootstrap_queue
        # and inflight_queue are CPU-only RDMA waits — counting them causes
        # tight-loop dummy forward_batch calls which trigger HCCL AIV
        # all-reduce counter divergence and deadlock on Ascend.
        return bool(self.waiting_queue or self._bootstrap_failures)

    def _on_request_finished(self, request) -> None:
        self._cleanup_terminal_request(request)
