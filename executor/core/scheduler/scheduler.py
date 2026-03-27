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

from typing import List, Dict, Optional, Set, Tuple
from collections import deque
import torch
from executor.core.config import SchedulerConfig
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from ..types_.types import Request, Batch, StepOutput, MTPInfo


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

        # ID generators
        self._request_counter = 0
        self._batch_counter = 0

    def add_request(self, prompt: str) -> int:
        """Add a new request to the pending queue."""
        request_id = self._request_counter
        self._request_counter += 1

        request = Request(
            request_id=request_id,
            prompt=prompt,
        )
        self.waiting_queue.append(request)
        return request_id

    def run_step(self, engine) -> Optional[StepOutput]:
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
        batch = self._schedule_batch()
        if batch is None or batch.is_empty():
            return None

        # Execute batch through engine
        output = engine.forward_batch(batch)

        # Process outputs and update request states
        finished = self._process_batch_output(batch)

        return StepOutput(
            next_tokens=output.get("next_tokens", {}),
            finished_requests=finished,
        )

    def _schedule_batch(self) -> Optional[Batch]:
        """Assemble the next batch for execution.

        Strategy: Prioritize prefill requests.

        Returns:
            Batch object ready for execution, or None if no work.
        """
        # Try prefill first
        if self.waiting_queue:
            batch = self._schedule_prefill_batch()
            if batch and not batch.is_empty():
                return batch

        # Try decode if no prefill
        if self.running_requests:
            batch = self._schedule_decode_batch()
            if batch and not batch.is_empty():
                return batch

        return None

    def _schedule_prefill_batch(self) -> Optional[Batch]:
        """Schedule a batch of prefill requests.

        Selects requests from queue up to batch size limit.

        Returns:
            Batch of prefill requests, or None if queue empty.
        """
        if not self.waiting_queue:
            return None

        selected: List[Request] = []
        prefill_mini_batch = self.config.prefill_mini_batch
        cycle_idx = len(self.running_requests) // prefill_mini_batch
        # Iterate through queue and select requests
        while self.waiting_queue and len(selected) < prefill_mini_batch:
            request = self.waiting_queue.popleft()

            # Tokenize if not already done
            if request.input_ids.numel() == 0:
                request.input_ids = self.tokenizer(
                    request.prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.input_max_len,
                    return_attention_mask=False
                ).input_ids[0]

            selected.append(request)

        if not selected:
            return None

        # Create batch
        batch = Batch(requests=selected, is_prefill=True, cycle_idx=cycle_idx)

        return batch

    def _schedule_decode_batch(self) -> Optional[Batch]:
        """Schedule a batch of decode requests.

        Selects all running requests up to batch size limit.

        Returns:
            Batch of decode requests, or None if none running.
        """
        if not self.running_requests:
            return None

        # Get all running requests (up to batch size)
        running_list = list(self.running_requests.values())
        selected = running_list[:self.config.batch_size_per_dp_rank]

        if not selected:
            return None

        # Create batch
        batch = Batch(requests=selected, is_prefill=False)

        return batch

    def _process_batch_output(
        self,
        batch: Batch,
    ) -> List[int]:
        """Process batch execution output and update request states.

        Args:
            batch: The batch that was executed.
            output: Engine output used for scheduling state transitions.

        Returns:
            List of request IDs that finished in this step.
        """
        finished = []

        for request in batch.requests:
            if batch.is_prefill:
                # Prefill completion
                request.is_prefill_done = True
                self.running_requests[request.request_id] = request
            else:
                # Decode step - increment step counter
                request.decode_step_count += 1
                # Check if finished
                if self._should_finish(request):
                    request.is_finished = True
                    request.finish_reason = "length"  # or "eos"
                    self.running_requests.pop(request.request_id, None)
                    self.finished_requests[request.request_id] = request
                    finished.append(request.request_id)

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
