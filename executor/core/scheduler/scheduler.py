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
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from ..types_.types import Request, Batch, StepOutput, MTPInfo
from ..engine import ExecutionEngine

logger = logging.getLogger(__name__)
from ..kv_cache import KVCacheManager
from ..engine import ExecutionEngine

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

    def __init__(self, tokenizer, config: Optional[SchedulerConfig] = None):
        """Initialize the scheduler.

        Args:
            tokenizer: Tokenizer instance for text encoding.
            config: Scheduler configuration. Uses defaults if None.
        """
        self.config = config or SchedulerConfig()
        self.tokenizer = tokenizer

        # Request state management
        self.waiting_queue: deque[Request] = deque()
        self.running_requests: Dict[int, Request] = {}
        self.finished_requests: Dict[int, Request] = {}
        self.prefilled_request_count: int = 0

        # ID generators
        self._request_counter = 0
        self._batch_counter = 0

    def tokenize_request(self, prompt, input_truncated_len) -> torch.Tensor:
        """Tokenize a prompt and return input_ids tensor."""
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise ValueError("Empty prompt list")
            if isinstance(prompt[0], dict):
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
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

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=input_truncated_len,
        )
        return encoded.input_ids[0]

    def add_request(self, prompt: Union[str, List[dict]]) -> int:
        """Add a new request to the pending queue."""
        request_id = self._request_counter
        self._request_counter += 1

        request = Request(
            request_id=request_id,
            prompt=prompt,
        )
        self.waiting_queue.append(request)
        return request_id

    def run_step(self, engine: ExecutionEngine) -> Optional[StepOutput]:
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

        Returns:
            StepOutput containing batch results,
            or None if no work to do.
        """
        # Assemble batch (normal mode)
        batch = self._schedule_batch(engine)
        if batch is None or batch.is_empty():
            return None

        # Execute batch through engine
        output = engine.forward_batch(batch)

        # Process outputs and update request states
        finished = self._process_batch_output(batch, engine.kvcache_manager)

        return StepOutput(
            next_tokens=output.get("next_tokens", {}),
            finished_requests=finished,
        )

    def _schedule_batch(self, engine: ExecutionEngine) -> Optional[Batch]:
        """Assemble the next batch for execution.

        Strategy: Prioritize prefill requests.

        Returns:
            Batch object ready for execution, or None if no work.
        """
        # Try prefill first
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
        available_batches = self.config.batch_size_per_dp_rank - len(self.running_requests)
        if available_batches <= 0:
            return None

        # Strict FIFO: only consider the queue head. If the head request
        # cannot fit in the current prefill batch budget, stop scheduling
        # instead of skipping ahead to shorter requests behind it.
        while self.waiting_queue and len(selected) < available_batches:
            request = self.waiting_queue[0]
            self._prepare_request_prompt(request, engine.input_truncated_len)

            if engine.kvcache_manager and not engine.kvcache_manager.allocate_slots(request.request_id,
                                                                                    request.computed_len,
                                                                                    request.input_ids.shape[-1]):
                break

            if request.prompt_tokens > max_prefill_tokens:
                request = self.waiting_queue.popleft()
                request.is_finished = True
                request.finish_reason = "prompt_too_long"
                self.finished_requests[request.request_id] = request
                logger.warning(
                    "Dropping request %s from prefill: prompt_tokens=%s exceeds max_prefill_tokens=%s",
                    request.request_id,
                    request.prompt_tokens,
                    max_prefill_tokens,
                )
                continue

            next_total_prefill_tokens = total_prefill_tokens + request.prompt_tokens
            if selected and next_total_prefill_tokens > max_prefill_tokens:
                break

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

    def _prepare_request_prompt(self, request: Request, input_truncated_len) -> None:
        if request.input_ids.numel() > 0:
            return

        request.input_ids = self.tokenize_request(request.prompt, input_truncated_len)
        request.prompt_tokens = int(request.input_ids.numel())

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
                num_new_tokens=1+engine.next_n,
                lookahead_tokens=max(engine.next_n - 1, 0)
                ):
                break
            selected.append(req)
        # Create batch
        batch = Batch(requests=selected, is_prefill=False)

        return batch

    def _process_batch_output(
        self,
        batch: Batch,
        kv_cache_manager: KVCacheManager,
    ) -> List[int]:
        """Process batch execution output and update request states.

        Args:
            batch: The batch that was executed.
            kv_cache_manager: KV cache manager for freeing request cache blocks.

        Returns:
            List of request IDs that finished in this step.
        """
        finished = []

        for request in batch.requests:
            if batch.is_prefill:
                # Prefill completion
                request.is_prefill_done = True
                self.running_requests[request.request_id] = request
                self.prefilled_request_count += 1
            else:
                # Decode step - increment step counter
                request.decode_step_count += 1

                # Check if finished
                if self._should_finish(request):
                    request.is_finished = True
                    request.finish_reason = request.finish_reason or "length"
                    self.running_requests.pop(request.request_id, None)
                    self.finished_requests[request.request_id] = request
                    if kv_cache_manager:
                        kv_cache_manager.free(request.request_id)
                    finished.append(request.request_id)

                if request.stop_valid_generation:
                    continue

                # Check if stop valid generation
                if valid_output_len := self._should_stop_valid_generation(request):
                    request.stop_valid_generation = True
                    request.valid_output_len = valid_output_len

        return finished

    def _should_finish(self, request: Request) -> bool:
        """Determine if a request has completed generation.

        Args:
            request: The request to check.

        Returns:
            True if generation should stop.
        """
        # Check step limit - iterate max_new_tokens steps
        if request.decode_step_count >= self.config.max_new_tokens:
            request.finish_reason = "length"
            return True

        return False

    def _should_stop_valid_generation(self, request: Request):
        """Mark request as stop_valid_generation and freeze decode length when hitting EOS or max_new_tokens."""
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        valid_output_len = None

        if len(request.output_id_list) >= self.config.max_new_tokens:
            valid_output_len = self.config.max_new_tokens

        if eos_token_id and eos_token_id in request.output_id_list:
            for idx, token_id in enumerate(request.output_id_list):
                if token_id == eos_token_id:
                    valid_output_len = min(idx + 1, valid_output_len) if valid_output_len else idx + 1
                    break

        return valid_output_len

    def has_work(self) -> bool:
        """Check if there is any pending or running work.

        Returns:
            True if scheduler has requests to process (not finished).
        """
        return bool(self.waiting_queue or self.running_requests)

    def get_finished_request(self, request_id: int) -> Optional[Request]:
        """Get a finished request by ID.

        Args:
            request_id: The request ID to look up.

        Returns:
            The finished Request, or None if not found.
        """
        return self.finished_requests.get(request_id)

    def get_all_finished_requests(self) -> List[Request]:
        """Get all finished requests.

        Returns:
            List of finished requests.
        """
        return list(self.finished_requests.values())

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
