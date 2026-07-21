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

"""Core scheduler implementation for request lifecycle management."""

import logging
from typing import List, Dict, Optional, Union
from collections import deque
import torch
from executor.core.config import SchedulerConfig
from ..forward_data_info import Request, Batch, MTPInfo, SamplingParams
from ..engine import ExecutionEngine

logger = logging.getLogger(__name__)


class Scheduler:
    """Core scheduler managing request lifecycle with batch support.

    The Scheduler is responsible for:
    1. Managing request state transitions
    2. Assembling requests into batches
    3. Preparing batched inputs for ExecutionEngine

    Attributes:
        config: Scheduler behavior configuration.
        tokenizer: Tokenizer for encoding prompts.
        waiting_queue: Queue of pending requests.
        running_requests: Dict of requests in decode phase.
        finished_requests: Completed requests.
        _request_counter: Auto-incrementing request ID.
        _batch_counter: Auto-incrementing batch ID.
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[SchedulerConfig] = None,
        input_truncated_len: Optional[int] = None,
    ):
        """Initialize the scheduler.

        Args:
            tokenizer: Tokenizer instance for text encoding.
            config: Scheduler configuration. Uses defaults if None.
            input_truncated_len: Optional max prompt length passed to the
                tokenizer for truncation; None disables truncation.
        """
        self.config = config or SchedulerConfig()
        self.tokenizer = tokenizer
        self.input_truncated_len = input_truncated_len

        # Request state management
        self.waiting_queue: deque[Request] = deque()
        self.running_requests: Dict[int, Request] = {}
        self.finished_requests: Dict[int, Request] = {}
        self.prefilled_request_count: int = 0
        self._step: int = 0

        # ID generators
        self._request_counter = 0
        self._batch_counter = 0
        self.mode = 'offline'

    def tokenize_request(
        self,
        prompt,
        input_truncated_len=None,
        chat_template_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Tokenize a prompt and return input_ids tensor."""
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise ValueError("Empty prompt list")
            if isinstance(prompt[0], dict):
                template_kwargs = {}
                if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
                    # Keep framework-controlled apply_chat_template kwargs fixed;
                    # only pass the supported template option for now.
                    template_kwargs["enable_thinking"] = chat_template_kwargs["enable_thinking"]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            elif isinstance(prompt[0], str):
                raise ValueError(
                    "List[str] prompt not supported. Use single string prompt or "
                    "submit multiple requests."
                )
            else:
                raise ValueError(f"Unsupported prompt type: {type(prompt[0])}")
        else:
            prompt_text = prompt
        kwargs = {}
        if input_truncated_len is not None:
            kwargs = dict(
                truncation=True,
                max_length=input_truncated_len,
            )
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            **kwargs
        )
        return encoded.input_ids[0]

    def add_request(
        self,
        prompt: Union[str, List[dict]],
        request_id: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> int:
        """Add a new request to the pending queue.

        Args:
            prompt: Input text or messages list.
            request_id: Optional request ID (auto-generated if None).
            sampling_params: Per-request sampling parameters (online PD mode).
            input_ids: Pre-tokenized input IDs (skips tokenization if provided).
        """
        if request_id is None:
            request_id = self._request_counter
        self._request_counter += 1

        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params
            or SamplingParams(max_tokens=self.config.max_new_tokens),
        )
        if input_ids is not None:
            request.input_ids = input_ids
            request.prompt_tokens = input_ids.numel()
        # Rejected requests are recorded in finished_requests
        # and NOT queued, so every downstream stage (e.g. _schedule_prefill_batch)
        # can assume the queue holds only valid requests and never re-checks.
        if self.reject_if_prompt_too_long(request):
            return request_id
        self.waiting_queue.append(request)
        return request_id

    def run_step(
        self,
        engine: ExecutionEngine,
        phase: Optional[str] = None,
    ) -> bool:
        """Execute one scheduling step.

        This is the main entry point for driving the generation process.
        It assembles a batch, executes it through the engine, and
        processes results.

        The scheduling strategy:
        1. Process all pending prefill requests in batch cycles
        2. Otherwise, try to schedule prefill requests first (up to budget)
        3. If no prefill, schedule decode requests
        4. Execute batch and update states

        Args:
            engine: ExecutionEngine instance for model inference.

        Finished requests are recorded in ``finished_requests``; dispatch reads
        that dict directly. Returns True if a real (non-dummy) batch ran, False
        otherwise — offline uses this to break a has_work-but-no-batch stall.
        """
        # Assemble batch (normal mode)
        batch = self._schedule_batch(engine, phase)
        if batch is None or batch.is_empty():
            return False

        # Execute batch through engine (runs forward + TP collectives even on
        # dummy batches so other TP ranks in the DP group don't hang).
        output = engine.forward_batch(batch)

        # Dummy batch is only there to keep collectives aligned; it has a
        # fake request and no real state — skip state updates / output emit.
        if batch.is_dummy:
            return False

        # Process outputs and update request states
        self._process_batch_output(batch, engine)

        self._step += 1
        self._log_step(engine, batch, output)
        return True

    def _schedule_batch(
        self,
        engine: ExecutionEngine,
        phase: Optional[str] = None,
    ) -> Optional[Batch]:
        """Assemble the next batch for execution.

        Online PD passes an explicit ``phase`` (``"prefill"`` / ``"decode"``)
        and expects the scheduler to stay in that phase — a prefill rank must
        not silently fall through to decode its own in-flight PD requests
        queued in ``running_requests`` (and vice versa). Offline drops the
        argument and uses prefill-first-then-decode.
        """
        if phase in ("prefill", "decode"):
            if phase == "prefill":
                batch = self._schedule_prefill_batch(engine)
            else:
                batch = self._schedule_decode_batch(engine)
            if batch is not None and not batch.is_empty():
                return batch
            # Phase is forced but this rank has no local work. Return a dummy
            # batch so every TP rank in the DP group runs forward together and
            # TP collectives stay aligned; run_step discards the output.
            return self._build_dummy_batch(phase, engine)
        if phase is not None:
            raise ValueError(f"Unsupported phase: {phase!r}")

        # Offline fallback: try prefill first
        if self.waiting_queue:
            batch = self._schedule_prefill_batch(engine)
            if batch and not batch.is_empty():
                return batch

        # Try decode if no prefill
        if self.running_requests:
            batch = self._schedule_decode_batch(engine)
            if batch and not batch.is_empty():
                return batch

        return None

    def _schedule_prefill_batch(self, engine: ExecutionEngine) -> Optional[Batch]:
        """Schedule a batch of prefill requests.

        Selects requests from queue up to the packed prefill token budget
        while respecting the number of available in-flight request slots.

        Returns:
            Batch of prefill requests, or None if queue empty.
        """
        if not self.waiting_queue:
            return None

        selected: List[Request] = []
        request_offset = self.prefilled_request_count
        total_prefill_tokens = 0
        max_prefill_tokens = self.config.max_prefill_tokens
        parallel_config = engine.infer_config.parallel_config
        enable_offline_cp = self.mode == 'offline' and parallel_config.cp_size > 1

        # Strict FIFO: only consider the queue head. If the head request
        # cannot fit in the current prefill batch budget, stop scheduling
        # instead of skipping ahead to shorter requests behind it.
        while self.waiting_queue:
            request = self.waiting_queue[0]
            # Prompt length is enforced once at admission (add_request), so the
            # queue holds only within-cap requests here — no re-check needed.

            next_total_prefill_tokens = total_prefill_tokens + request.prompt_tokens
            if selected and next_total_prefill_tokens > max_prefill_tokens:
                break
            if (
                parallel_config.cp_size > 1
                and self.config.cp_mini_batch > 0
                and len(selected) == self.config.cp_mini_batch
            ):
                break

            if enable_offline_cp:
                request.cp_rank = (request_offset + len(selected)) % parallel_config.cp_size

            if (
                engine.kvcache_manager
                and self._needs_prefill_kv_allocation(engine, request)
                and not engine.kvcache_manager.allocate_slots(
                    request.request_id, request.computed_len, request.input_ids.shape[-1]
                )
            ):
                if enable_offline_cp:
                    raise RuntimeError(
                        "CP offline prefill requires all CP ranks to schedule the same request batch, "
                        "but the owner rank failed to allocate KV cache slots. "
                        f"request_id={request.request_id}, cp_rank={request.cp_rank}, "
                        f"global_rank={parallel_config.global_rank}"
                    )
                continue

            selected.append(self.waiting_queue.popleft())
            total_prefill_tokens = next_total_prefill_tokens

        if not selected:
            return None

        # Create batch
        batch = Batch(
            requests=selected,
            is_prefill=True,
            request_offset=request_offset,
        )

        return batch

    def _needs_prefill_kv_allocation(self, engine: ExecutionEngine, request: Request) -> bool:
        parallel_config = engine.infer_config.parallel_config
        if self.mode != 'offline' or parallel_config.cp_size <= 1:
            return True
        current_cp_rank = parallel_config.global_rank % parallel_config.cp_size
        return request.cp_rank == current_cp_rank

    def _prepare_request_prompt(self, request: Request) -> None:
        if request.input_ids.numel() > 0:
            return

        request.input_ids = self.tokenize_request(request.prompt, self.input_truncated_len)
        request.prompt_tokens = int(request.input_ids.numel())

    def reject_if_prompt_too_long(self, request: Request) -> bool:
        """Admission-time prompt-length guard — single source of truth.

        If the prompt exceeds ``max_prefill_tokens``, mark the request finished
        (``prompt_too_long``), record it in ``finished_requests`` and return
        True. Dispatch later picks it up by iterating ``finished_requests``.
        Returns False when within the cap. Centralizing the threshold and the
        length source here keeps prefill and decode from drifting apart.
        """
        self._prepare_request_prompt(request)
        if request.prompt_tokens <= self.config.max_prefill_tokens:
            return False
        request.is_finished = True
        request.finish_reason = "prompt_too_long"
        self.finished_requests[request.request_id] = request
        logger.warning(
            "Rejecting request %s at admission: prompt_tokens=%s exceeds max_prefill_tokens=%s",
            request.request_id, request.prompt_tokens, self.config.max_prefill_tokens,
        )
        return True

    def _schedule_decode_batch(self, engine: ExecutionEngine) -> Optional[Batch]:
        """Schedule a batch of decode requests.

        Selects all running requests up to batch size limit.

        Returns:
            Batch of decode requests, or None if none running.
        """
        if not self.running_requests:
            return None

        # Get all running requests (up to batch size)
        running_list = list(self.running_requests.values())
        running = running_list[:self.config.batch_size_per_dp_rank]

        if not running:
            return None
        selected: List[Request] = []
        for req in running:
            # Offline mode requires that all requests in the running list are used for inference.
            if engine.kvcache_manager and not engine.kvcache_manager.allocate_slots(
                req.request_id, computed_tokens=req.computed_len + 1,
                num_new_tokens=1 + engine.next_n,
                lookahead_tokens=max(engine.next_n - 1, 0)
            ):
                continue
            selected.append(req)
        # Create batch
        batch = Batch(requests=selected, is_prefill=False)

        return batch

    @staticmethod
    def _build_dummy_batch(phase: str, engine) -> Batch:
        """Synthesize a 1-token batch to keep TP collectives aligned.

        Used when online PD forces a phase but this rank has no local work.
        The fake request carries ``request_id=-1`` so it never collides with
        real state. ``kv_cache_manager`` resolves unknown request ids to an
        empty block list → ``prepare_block_tables`` pads with the null block,
        so forward runs safely without disturbing any real request's KV.
        """
        dummy = Request(
            request_id=-1,
            prompt="0",
            input_ids=torch.zeros(1, dtype=torch.long),
        )
        dummy.prompt_tokens = 1
        dummy.computed_len = 0 if phase == "prefill" else 1
        if engine.next_n > 0:
            dummy.mtp_info = MTPInfo(
                spec_tokens=torch.zeros(engine.next_n, dtype=torch.long),
            )
        batch = Batch(
            requests=[dummy],
            is_prefill=(phase == "prefill"),
            is_dummy=True,
        )
        return batch

    def _process_batch_output(
        self,
        batch: Batch,
        engine: ExecutionEngine,
    ) -> None:
        """Process batch execution output and update request states.

        Finished requests are recorded in ``finished_requests``; dispatch reads
        that dict directly, so no id list is collected or returned here.

        Args:
            batch: The batch that was executed.
            engine: Execution engine that owns KV cache and parallel config.
        """
        parallel_config = engine.infer_config.parallel_config
        enable_offline_cp = self.mode == 'offline' and batch.is_prefill and parallel_config.cp_size > 1
        current_cp_rank = (
            parallel_config.global_rank % parallel_config.cp_size
            if enable_offline_cp else 0
        )

        for request in batch.requests:
            if batch.is_prefill:
                # Prefill completion. `_on_prefill_complete` owns the
                # running_requests transition: base default moves the request
                # there for decode; PD prefill scheduler overrides to reroute
                # into its inflight_queue instead (so base _schedule_batch
                # does not later try to decode a PD pending request).
                request.is_prefill_done = True
                self.prefilled_request_count += 1
                if not enable_offline_cp or request.cp_rank == current_cp_rank:
                    self._on_prefill_complete(request)
            else:
                if self._should_finish(request):
                    """Move a decode-completed request to finished_requests and free KV."""
                    request.is_finished = True
                    self.running_requests.pop(request.request_id, None)
                    self.finished_requests[request.request_id] = request
                    if engine.kvcache_manager:
                        engine.kvcache_manager.free(request.request_id)
                    self._on_request_finished(request)

    def _should_finish(self, request: Request) -> bool:
        """Determine if a request has completed generation.

        Offline ignores EOS so every DP rank runs the same number of decode
        steps (output is truncated at return time via valid_output_len).
        Online checks per-request ``sampling_params.ignore_eos`` to allow
        independent requests to release compute on EOS.
        """
        # for offline
        if self.mode == 'offline':
            request.decode_step_count += 1
            if request.valid_output_len is not None:
                return request.decode_step_count >= self.config.max_new_tokens

        finish = False
        output_len = len(request.output_id_list)
        if not request.sampling_params.ignore_eos:
            if request.eos_output_len is not None:
                request.finish_reason = "stop"
                if self.mode == 'offline':
                    request.valid_output_len = request.eos_output_len
                else:
                    request.output_id_list = request.output_id_list[:request.eos_output_len]
                    output_len = len(request.output_id_list)
                finish = True
        if not finish and output_len >= self.config.max_new_tokens:
            request.finish_reason = "length"
            finish = True
        sp_max = request.sampling_params.max_tokens
        if not finish and sp_max is not None and output_len >= sp_max:
            request.finish_reason = "length"
            finish = True

        # for offline
        if self.mode == 'offline' and request.finish_reason is not None:
            if request.valid_output_len is None:
                request.valid_output_len = output_len
            # offline finish according to compute step
            finish = request.decode_step_count >= self.config.max_new_tokens

        return finish

    def has_work(self) -> bool:
        """Check if there is any pending or running work.

        Returns:
            True if scheduler has requests to process (not finished).
        """
        return bool(self.waiting_queue or self.running_requests)

    # ── Subclass hooks (used by online PD schedulers). ──
    def _on_prefill_complete(self, request: Request) -> None:
        """Transition a prefilled request into the decode running set.

        Default behavior: move the request into ``running_requests`` so the
        next scheduling tick picks it up for decode. PD prefill scheduler
        overrides this **without calling super()** to reroute the request
        into its ``inflight_queue`` (awaiting PD KV transfer) instead — that
        keeps ``running_requests`` empty on the prefill rank, so base
        ``_schedule_batch`` can never accidentally decode a PD pending
        request.
        """
        self.running_requests[request.request_id] = request

    def _on_request_finished(self, request: Request) -> None:
        """Called when a request is moved to finished. Override for PD cleanup."""
        return None

    # ── Per-step status logging. ──
    def _log_step(self, engine: ExecutionEngine, batch: Batch, output: dict) -> None:
        """Emit a per-component timing log line per forward step.

        Prints the main worker time and each MTP step time separately —
        speculative decoding runs the small model multiple times per main
        step, and a slowdown in any single MTP iteration is invisible if
        only the summed total is reported. Online PD schedulers override
        this to fold queue depths + kv-cache usage into a single richer line.
        """
        stage = "prefill" if batch.is_prefill else "decode"
        main_ms = output.get("inference_time_main", 0.0)
        logger.info(f"[Main] Inference time ({stage}): {main_ms * 1000:.2f} ms")
        for idx, t in enumerate(output.get("inference_times_mtp", [])):
            logger.info(f"[MTP {idx}] Inference time ({stage}): {t * 1000:.2f} ms")

    def pop_finished_request(self, request_id: int) -> Optional[Request]:
        """Remove and return a finished request by ID.

        Args:
            request_id: The request ID to look up.

        Returns:
            The finished Request, or None if not found.
        """
        return self.finished_requests.pop(request_id, None)

    def reset(self) -> None:
        """Reset scheduler state. Clears all requests."""
        self.waiting_queue.clear()
        self.running_requests.clear()
        self.finished_requests.clear()
        self.prefilled_request_count = 0
        self._request_counter = 0
        self._batch_counter = 0

    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics.

        Returns:
            Dict with queue size, running count, finished count.
        """
        return {
            "pending": len(self.waiting_queue),
            "running": len(self.running_requests),
            "finished": len(self.finished_requests),
            "total": len(self.waiting_queue) + len(self.running_requests) + len(self.finished_requests),
        }
