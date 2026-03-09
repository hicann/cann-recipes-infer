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

"""Core type definitions for Executor Core."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch


@dataclass
class GenerationOutput:
    """Output from batch generation.

    Attributes:
        prompt: Original input prompt string.
        output_text: Generated text output.
        finish_reason: Reason for generation completion (default: "length").
    """
    prompt: str
    output_text: str
    finish_reason: str = "length"


@dataclass
class Request:
    """Enhanced request representation for batch processing.

    This class tracks the complete lifecycle of a generation request,
    from input prompts through token generation to final output.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Input text prompt.
        input_ids: Tokenized input IDs (populated during prefill).
        output_id_list: Generated token IDs (appended during decode).
        is_prefill_done: Whether prefill phase is completed.
        is_finished: Whether generation is complete.
        finish_reason: Reason for completion ("length", "eos", or "error").
    """
    request_id: int
    prompt: str
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    output_id_list: List[int] = field(default_factory=list)
    is_prefill_done: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None

    def get_all_token_ids(self) -> List[int]:
        """Get concatenated input and output token IDs."""
        input_list = self.input_ids.tolist() if self.input_ids.numel() > 0 else []
        return input_list + self.output_id_list

    def get_last_token_id(self) -> Optional[int]:
        """Get the most recently generated token ID."""
        if self.output_id_list:
            return self.output_id_list[-1]
        if self.input_ids.numel() > 0:
            return self.input_ids.tolist()[-1]
        return None

    def get_seq_len(self) -> int:
        """Get total sequence length (input + output)."""
        input_len = self.input_ids.numel()
        output_len = len(self.output_id_list)
        return input_len + output_len


@dataclass
class Batch:
    """A batch of requests for unified processing.

    This class aggregates multiple requests (both prefill and decode)
    into a single processing unit for the ExecutionEngine.

    Attributes:
        requests: List of requests in this batch.
        is_prefill: Whether this batch is in prefill phase.
        input_ids: Padded input token IDs [batch_size, max_seq_len].
        position_ids: Padded position IDs [batch_size, max_seq_len].
        seq_lens: Original sequence lengths for each request [batch_size].
        request_indices: Mapping from request_id to batch index.
    """
    requests: List['Request'] = field(default_factory=list)
    is_prefill: bool = True

    # Tensors (populated during batch preparation)
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None

    # Metadata
    request_indices: Dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of requests in batch."""
        return len(self.requests)

    def get_max_seq_len(self) -> int:
        """Get maximum sequence length in this batch."""
        if self.seq_lens is not None:
            return int(self.seq_lens.max().item())
        return max((req.get_seq_len() for req in self.requests), default=0)

    def get_request(self, request_id: int) -> Optional['Request']:
        """Get request by ID."""
        idx = self.request_indices.get(request_id)
        if idx is not None and 0 <= idx < len(self.requests):
            return self.requests[idx]
        return None

    def is_empty(self) -> bool:
        """Check if batch has no requests."""
        return len(self.requests) == 0


@dataclass
class StepOutput:
    """Output from a single scheduler step.

    Attributes:
        next_tokens: Dict mapping request_id to generated token.
        finished_requests: List of request IDs that completed.
    """
    next_tokens: Dict[int, int] = field(default_factory=dict)
    finished_requests: List[int] = field(default_factory=list)


@dataclass
class MTPInfo:
    """MTP (Multi-step Speculative Decoding) temporary state container.

    This class encapsulates temporary state for MTP speculative decoding,
    serving as a complement to Batch and StepOutput, specifically storing
    data related to multi-step speculative decoding.

    Attributes:
        prev_hidden_states: Hidden states from previous Main or MTP module.
        spec_tokens: Predicted token sequence for main model verification.
        kv_len_cached: Cached KV length for calculating mtp kv_len in the
            next iteration.
        accepted_num: Number of accepted tokens in the current iteration.
    """
    prev_hidden_states: Optional[torch.Tensor] = None
    spec_tokens: Optional[torch.Tensor] = None
    kv_len_cached: Optional[torch.Tensor] = None
    accepted_num: Optional[torch.Tensor] = None
