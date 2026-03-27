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
from executor.utils.forward_metadata import get_forward_metadata, set_forward_metadata


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
class MTPInfo:
    """MTP (Multi-step Speculative Decoding) state container.

    This class encapsulates state information for MTP speculative decoding,
    managing the multi-token prediction workflow between the draft and main models.

    Attributes:
        is_prefill: Whether the current phase is prefill (initial token processing).
        spec_tokens: Accumulated speculative token sequence for main model verification.
        accepted_num: Number of accepted tokens per sample in current iteration.
    """
    is_prefill: Optional[bool] = False
    spec_tokens: Optional[torch.Tensor] = None
    accepted_num: Optional[torch.Tensor] = None

    def set_mtp_info(self, **kwargs):
        """Update MTPInfo instance attributes from keyword arguments.

        Args:
            **kwargs: Keyword arguments to update instance attributes.
                      Valid keys: is_prefill, spec_tokens, accepted_num, next_n
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class Request:
    """Enhanced request representation for batch processing.

    This class tracks the complete lifecycle of a generation request,
    from input prompts through token generation to final output.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Input text prompt.
        input_ids: Tokenized input IDs (populated during prefill).
        kv_len: Per-request KV cache length used to rebuild batch kv_len during decode.
        output_id_list: Generated token IDs (appended during decode).
        is_prefill_done: Whether prefill phase is completed.
        is_finished: Whether generation is complete.
        finish_reason: Reason for completion ("length", "eos", or "error").
        spec_num_forward_ct: Number of speculative forward passes for MTP acceptance statistics.
        spec_num_accepted_tokens: Number of accepted speculative tokens for MTP acceptance statistics.
        decode_step_count: Number of decode steps completed (each step generates one or more tokens).
    """
    request_id: int
    prompt: str
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    kv_len: int = 0
    output_id_list: List[int] = field(default_factory=list)
    is_prefill_done: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None
    mtp_info: Optional[MTPInfo] = None
    # Metrics
    spec_num_forward_ct: int = 0
    spec_num_accepted_tokens: int = 0
    # Step counter for decode phase
    decode_step_count: int = 0

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

    def set_mtp_info(self, accepted_num, spec_tokens):
        if not self.mtp_info:
            self.mtp_info = MTPInfo()
        # Update metrics
        self.spec_num_forward_ct += 1
        self.spec_num_accepted_tokens += accepted_num + 1
        self.mtp_info.set_mtp_info(
            accepted_num=accepted_num,
            spec_tokens=spec_tokens,
        )


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
        mtp_infos: Original MTP state containing speculative tokens and accepted num.
    """
    requests: List['Request'] = field(default_factory=list)
    is_prefill: bool = True

    # Tensors (populated during batch preparation)
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None

    # Prefill Mini Batch
    cycle_idx: int = 0

    # Metadata
    request_indices: Dict[int, int] = field(default_factory=dict)

    # MTPInfo
    mtp_infos: Optional[MTPInfo] = None

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

    def build_tensors_from_requests(self) -> None:
        """Collect request data and build batch tensors in-place."""
        if self.is_prefill:
            self.input_ids = torch.stack([request.input_ids for request in self.requests])
            return

        self.input_ids = torch.tensor(
            [request.get_last_token_id() for request in self.requests],
            dtype=torch.long,
        ).unsqueeze(1)

        kv_lens = torch.tensor([request.kv_len for request in self.requests], dtype=torch.long)
        set_forward_metadata(kv_len=kv_lens)

        spec_tokens_list = []
        accepted_num_list = []
        for request in self.requests:
            if request.mtp_info:
                spec_tokens_list.append(request.mtp_info.spec_tokens)
                accepted_num_list.append(request.mtp_info.accepted_num)
            else:
                self.mtp_infos = None
                return

        self.mtp_infos = MTPInfo(
            spec_tokens=torch.stack(spec_tokens_list, dim=0),
            accepted_num=torch.stack(accepted_num_list, dim=0),
        )

    def update_requests_from_batch(
        self,
        next_tokens: Optional[torch.Tensor],
    ) -> Dict[int, List[int]]:
        """Split batch outputs by index and update each request in-place."""
        next_tokens_by_request: Dict[int, List[int]] = {}
        kv_lens = get_forward_metadata().kv_len

        if next_tokens is not None and next_tokens.shape[0] != len(self.requests):
            raise ValueError(
                f"next_tokens batch size {next_tokens.shape[0]} does not match request count {len(self.requests)}"
            )
        if kv_lens is not None and kv_lens.shape[0] != len(self.requests):
            raise ValueError(
                f"kv_len batch size {kv_lens.shape[0]} does not match request count {len(self.requests)}"
            )

        for i, request in enumerate(self.requests):
            accepted_num = None

            if self.mtp_infos:
                accepted_num = self.mtp_infos.accepted_num[i] if self.mtp_infos.accepted_num is not None else None
                spec_tokens = self.mtp_infos.spec_tokens[i] if self.mtp_infos.spec_tokens is not None else None
                request.set_mtp_info(accepted_num, spec_tokens)

            if next_tokens is not None:
                if accepted_num is not None:
                    request_next_tokens = next_tokens[i, :accepted_num + 1].tolist()
                else:
                    request_next_tokens = next_tokens[i].tolist()
                request.output_id_list += request_next_tokens
                next_tokens_by_request[request.request_id] = request_next_tokens

            if kv_lens is not None:
                request.kv_len = kv_lens[i]

        return next_tokens_by_request


@dataclass
class StepOutput:
    """Output from a single scheduler step.

    Attributes:
        next_tokens: Dict mapping request_id to generated token IDs.
        finished_requests: List of request IDs that completed.
    """
    next_tokens: Dict[int, List[int]] = field(default_factory=dict)
    finished_requests: List[int] = field(default_factory=list)
