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
from executor.core.mm_processor import BaseMMProcessor
from executor.utils import ceil_div
from ..forward_data_info import Request, Batch, MMEncodeBatch, SamplingParams
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
        mm_waiting_queue: Queue of requests awaiting MM Encode, when enabled.
        waiting_queue: Queue of requests ready for Prefill.
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
        mm_processor: Optional[BaseMMProcessor] = None,
        enable_mm_encode: bool = False,
    ):
        """Initialize the scheduler.

        Args:
            tokenizer: Tokenizer instance for text encoding.
            config: Scheduler configuration. Uses defaults if None.
            input_truncated_len: Optional max prompt length passed to the
                tokenizer for truncation; None disables truncation.
            mm_processor: Optional model-specific multimodal processor.
            enable_mm_encode: Route admitted requests through MM Encode before Prefill.
        """
        self.config = config or SchedulerConfig()
        self.tokenizer = tokenizer
        self.input_truncated_len = input_truncated_len
        self.mm_processor = mm_processor

        # Request state management
        self.mm_waiting_queue: Optional[deque[Request]] = (
            deque() if enable_mm_encode else None
        )
        self.waiting_queue: deque[Request] = deque()
        self.running_requests: Dict[int, Request] = {}
        self.finished_requests: Dict[int, Request] = {}
        self.prefilled_request_count: int = 0
        self.mm_encode_infer_time: float = 0.0
        self.prefill_infer_time: float = 0.0
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
        if input_truncated_len is not None and not (
            self.mm_processor is not None
            and self.mm_processor.should_defer_truncation(prompt)
        ):
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
            input_ids: Pre-tokenized input IDs that skip text tokenization.
                Registered multimodal models still run MM processing.
        """
        if request_id is None:
            request_id = self._request_counter
        self._request_counter += 1

        if sampling_params is not None and sampling_params.seed is not None:
            generator = torch.Generator(device='npu')
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None
        request = Request(
            request_id=request_id,
            prompt=prompt,
            generator=generator,
            sampling_params=sampling_params
            or SamplingParams(max_tokens=self.config.max_new_tokens),
        )
        if input_ids is not None:
            request.input_ids = input_ids
        self._prepare_request_prompt(request)
        if self.mm_processor is not None:
            self.mm_processor.process(request)
        request.prompt_tokens = int(request.input_ids.numel())
        # Rejected requests are recorded in finished_requests and never enter
        # either compute queue.
        if self.reject_if_prompt_too_long(request):
            return request_id
        self._enqueue_admitted_request(request)
        return request_id

    def _enqueue_admitted_request(self, request: Request) -> None:
        if self.mm_waiting_queue is not None:
            self.mm_waiting_queue.append(request)
        else:
            self.waiting_queue.append(request)

    def run_step(
        self,
        engine: ExecutionEngine,
        phase: Optional[str] = None,
    ) -> bool:
        """Execute one scheduling step.

        This is the main entry point for driving the generation process.
        It assembles a batch, executes it through the engine, and
        processes results.

        Offline scheduling prioritizes Prefill-ready requests, then MM Encode,
        then Decode. Online supplies an explicit global phase.

        Args:
            engine: ExecutionEngine instance for model inference.

        Finished requests are recorded in ``finished_requests``; dispatch reads
        that dict directly. Returns True if a real (non-dummy) batch ran, False
        otherwise — offline uses this to break a has_work-but-no-batch stall.
        """
        if phase == "mm_encode" or (
            phase is None
            and not self.waiting_queue
            and self.has_mm_pending()
        ):
            mm_batch = self._schedule_mm_encode_batch(engine)
            if mm_batch is None:
                mm_batch = MMEncodeBatch(is_dummy=True)

            output = engine.encode_mm_batch(mm_batch)
            if mm_batch.is_dummy:
                return False

            self._process_mm_encode_output(mm_batch)
            if self.mode == "offline":
                self.mm_encode_infer_time += output["inference_time"]
            self._step += 1
            self._log_mm_encode_step(mm_batch, output)
            return True

        # Assemble a token batch.
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
        if (
            self.mode == "offline"
            and self.mm_waiting_queue is not None
            and batch.is_prefill
        ):
            self.prefill_infer_time += output["inference_time"]

        self._step += 1
        self._log_step(engine, batch, output)
        return True

    def _schedule_mm_encode_batch(
        self,
        engine: ExecutionEngine,
    ) -> Optional[MMEncodeBatch]:
        if not self.mm_waiting_queue:
            return None

        max_mm_tokens = self.config.max_mm_encode_tokens
        budget_enabled = max_mm_tokens > 0
        if budget_enabled:
            dp_size = engine.infer_config.parallel_config.cp_prefill_dp_size
            max_mm_tokens = ceil_div(max_mm_tokens, dp_size)

        requests = []
        total_mm_tokens = 0
        while self.mm_waiting_queue:
            request = self.mm_waiting_queue[0]
            next_mm_tokens = total_mm_tokens + request.mm_token_count
            if budget_enabled and requests and next_mm_tokens > max_mm_tokens:
                break
            requests.append(self.mm_waiting_queue.popleft())
            total_mm_tokens = next_mm_tokens
        return MMEncodeBatch(requests=requests)

    def _process_mm_encode_output(
        self,
        mm_batch: MMEncodeBatch,
    ) -> None:
        for request in mm_batch.requests:
            request.mm_inputs = None
            self.waiting_queue.append(request)

    @staticmethod
    def _log_mm_encode_step(mm_batch: MMEncodeBatch, output: dict) -> None:
        logger.info(
            "[MM Encode] batch_reqs=%d infer=%.2fms",
            len(mm_batch.requests),
            output["inference_time"] * 1000,
        )

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
        cache_owner_group_size = self._offline_cache_owner_group_size(engine)

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

            request_index = request_offset + len(selected)
            if enable_offline_cp:
                request.cp_rank = request_index % parallel_config.cp_size
            elif cache_owner_group_size > 1:
                # Match persistent cache ownership to contiguous Decode request shards.
                requests_per_owner = (
                    self.config.batch_size_per_dp_rank
                    // cache_owner_group_size
                )
                request.cp_rank = (
                    request_index // requests_per_owner
                ) % cache_owner_group_size

            manager_keys = self._offline_cache_manager_keys(engine, request)

            if (
                engine.kvcache_manager
                and not engine.kvcache_manager.allocate_slots(
                    request.request_id,
                    request.computed_len,
                    request.input_ids.shape[-1],
                    lookahead_tokens=(
                        engine.draft_worker.prefill_lookahead_tokens
                        if engine.draft_worker is not None
                        else 0
                    ),
                    manager_keys=manager_keys,
                )
            ):
                if enable_offline_cp:
                    raise RuntimeError(
                        "CP offline prefill requires all CP ranks to schedule the same request batch, "
                        "but one rank failed to allocate its required cache slots. "
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

    def _offline_cache_owner_group_size(self, engine: ExecutionEngine) -> int:
        if self.mode != 'offline':
            return 1

        parallel_config = engine.infer_config.parallel_config
        if parallel_config.cp_size > 1:
            return parallel_config.cp_size
        if engine.kvcache_manager is None:
            return 1
        return parallel_config.offline_prefill_dp_group_size

    def _offline_cache_manager_keys(self, engine: ExecutionEngine, request: Request):
        if self.mode != 'offline' or engine.kvcache_manager is None:
            return None

        group_size = self._offline_cache_owner_group_size(engine)
        if group_size <= 1:
            return None

        current_rank = engine.infer_config.parallel_config.global_rank % group_size
        is_owner = request.cp_rank == current_rank
        return engine.kvcache_manager.get_request_owner_manager_keys(is_owner)

    def _prepare_request_prompt(self, request: Request) -> None:
        if request.input_ids.numel() > 0:
            return

        request.input_ids = self.tokenize_request(request.prompt, self.input_truncated_len)

    def reject_if_prompt_too_long(self, request: Request) -> bool:
        """Admission-time prompt-length guard — single source of truth.

        If the prompt exceeds ``max_prefill_tokens``, mark the request finished
        (``prompt_too_long``), record it in ``finished_requests`` and return
        True. Dispatch later picks it up by iterating ``finished_requests``.
        Returns False when within the cap. Centralizing the threshold and the
        length source here keeps prefill and decode from drifting apart.
        """
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
            manager_keys = self._offline_cache_manager_keys(engine, req)
            if engine.kvcache_manager and not engine.kvcache_manager.allocate_slots(
                req.request_id, computed_tokens=req.computed_len + 1,
                num_new_tokens=1 + engine.next_n,
                lookahead_tokens=max(engine.next_n - 1, 0),
                manager_keys=manager_keys,
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
        if engine.draft_worker is not None:
            dummy.draft_info = engine.draft_worker.state_cls(
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
        return bool(
            self.has_mm_pending()
            or self.waiting_queue
            or self.running_requests
        )

    def has_mm_pending(self) -> bool:
        return bool(self.mm_waiting_queue)

    def has_prefill_ready(self) -> bool:
        return bool(self.waiting_queue)

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
        """Emit main-model and speculative-model timing per forward step."""
        stage = "prefill" if batch.is_prefill else "decode"
        main_ms = output.get("inference_time_main", 0.0)
        logger.info(f"[Main] Inference time ({stage}): {main_ms * 1000:.2f} ms")
        draft_times = output.get("inference_times_draft", [])
        if engine.draft_worker is not None and engine.draft_worker.method == "mtp":
            for index, draft_time in enumerate(draft_times):
                logger.info(
                    f"[MTP {index}] Inference time ({stage}): {draft_time * 1000:.2f} ms"
                )
        elif draft_times and engine.draft_worker is not None:
            draft_type = engine.draft_worker.method.capitalize()
            draft_ms = sum(draft_times)
            logger.info(
                f"[{draft_type}{engine.next_n}] Inference time ({stage}): {draft_ms * 1000:.2f} ms"
            )

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
        if self.mm_waiting_queue is not None:
            self.mm_waiting_queue.clear()
        self.waiting_queue.clear()
        self.running_requests.clear()
        self.finished_requests.clear()
        self.prefilled_request_count = 0
        self.mm_encode_infer_time = 0.0
        self.prefill_infer_time = 0.0
        self._request_counter = 0
        self._batch_counter = 0

    def get_stats(self) -> Dict[str, int]:
        """Get scheduler statistics.

        Returns:
            Counts for pending, running, and finished requests.
        """
        pending = len(self.mm_waiting_queue or ()) + len(self.waiting_queue)
        return {
            "pending": pending,
            "running": len(self.running_requests),
            "finished": len(self.finished_requests),
            "total": pending + len(self.running_requests) + len(self.finished_requests),
        }
