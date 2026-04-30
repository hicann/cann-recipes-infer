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
from typing import Any, List, Dict, Optional

import torch
from executor.utils.forward_metadata import get_forward_metadata, set_forward_metadata


@dataclass
class GenerationOutput:
    """Output from batch generation.

    Attributes:
        prompt: Original input prompt in chat message format.
        output_text: Generated text output.
        finish_reason: Reason for generation completion (default: "length").
    """
    prompt: List[dict]
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
class SamplingParams:
    """Sampling parameters for generation.

    Attributes:
        max_tokens: Maximum tokens to generate.
        ignore_eos: If True, do not stop on EOS token.
    """

    max_tokens: Optional[int] = None
    ignore_eos: bool = False


@dataclass
class Request:
    """Enhanced request representation for batch processing.

    This class tracks the complete lifecycle of a generation request,
    from input prompts through token generation to final output.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Input text prompt.
        sampling_params: Per-request sampling parameters (used in online mode).
        input_ids: Tokenized input IDs (populated during prefill).
        computed_len: Per-request computed token length.
        prompt_tokens: Actual number of prompt tokens (excluding right padding).
        output_id_list: Generated token IDs (appended during decode).
        is_prefill_done: Whether prefill phase is completed.
        is_finished: Whether generation is complete.
        finish_reason: Reason for completion ("length", "eos", or "error").
        spec_num_forward_ct: Number of speculative forward passes for MTP acceptance statistics.
        spec_num_accepted_tokens: Number of accepted speculative tokens for MTP acceptance statistics.
        decode_step_count: Number of decode steps completed (each step generates one or more tokens).
        valid_output_len: Length of valid output tokens when hitting EOS or max_new_tokens.
    """
    request_id: int
    prompt: "str | List[dict]"
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    computed_len: int = 0
    prompt_tokens: int = 0
    output_id_list: List[int] = field(default_factory=list)
    is_prefill_done: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None
    mtp_info: Optional[MTPInfo] = None
    # Metrics
    spec_num_forward_ct: int = 0
    spec_num_accepted_tokens: int = 0
    # Measure the combined inference time of the main model and the MTP model
    infer_time: List[float] = field(default_factory=list)
    # Step counter for decode phase
    decode_step_count: int = 0
    valid_output_len: Optional[int] = None
    # PD disaggregation fields (only populated in online PD mode).
    bootstrap_host: str = ""
    bootstrap_port: int = -1
    bootstrap_room: int = -1
    disagg_prefill_dp_rank: int = -1
    metadata_buffer_index: int = -1
    disagg_kv_sender: Optional[Any] = None

    @property
    def bootstrap_addr(self) -> str:
        """Derived "host:port" of the Prefill bootstrap for this request."""
        if self.bootstrap_host and self.bootstrap_port >= 0:
            return f"{self.bootstrap_host}:{self.bootstrap_port}"
        return ""

    def get_all_token_ids(self) -> List[int]:
        """Get concatenated input and output token IDs."""
        input_len = self.prompt_tokens or self.input_ids.numel()
        input_list = self.input_ids[:input_len].tolist() if self.input_ids.numel() > 0 else []
        return input_list + self.output_id_list

    def get_last_token_id(self) -> Optional[int]:
        """Get the most recently generated token ID."""
        if self.output_id_list:
            return self.output_id_list[-1]
        if self.input_ids.numel() > 0:
            input_len = self.prompt_tokens or self.input_ids.numel()
            if input_len > 0:
                return int(self.input_ids[input_len - 1].item())
        return None

    def get_seq_len(self) -> int:
        """Get total sequence length (input + output)."""
        input_len = self.prompt_tokens or self.input_ids.numel()
        output_len = len(self.output_id_list)
        return input_len + output_len

    def update_mtp_info(self, accepted_num, spec_tokens):
        if not self.mtp_info:
            self.mtp_info = MTPInfo()
        # Update metrics
        if self.valid_output_len is None:
            self.spec_num_forward_ct += 1
            self.spec_num_accepted_tokens += accepted_num
        self.mtp_info.set_mtp_info(
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
        input_ids: Prefill token stream [total_tokens] or decode query tokens [total_query_tokens].
        position_ids: Position IDs built by the execution engine.
        seq_lens: Original sequence lengths for each request [batch_size].
        total_tokens: Total number of valid prompt tokens in prefill.
        request_offset: Absolute request-slot offset for packed prefill batches.
        request_indices: Mapping from request_id to batch index.
        mtp_infos: Original MTP state containing speculative tokens and accepted num.
    """
    requests: List['Request'] = field(default_factory=list)
    is_prefill: bool = True

    # Tensors (populated during batch preparation)
    input_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    total_tokens: int = 0
    request_offset: int = 0

    # Metadata
    request_indices: Dict[int, int] = field(default_factory=dict)

    # MTPInfo
    mtp_infos: Optional[MTPInfo] = None

    # True when this batch was synthesized to keep DP+TP collectives aligned
    # on ranks with no local work (online PD). Engine runs forward normally so
    # collectives complete, but scheduler skips state updates / output emit.
    is_dummy: bool = False

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
            actual_lens = []
            prefill_tokens = []
            for req in self.requests:
                actual_lens.append(req.prompt_tokens)
                prefill_tokens.append(req.input_ids[:req.prompt_tokens])
            self.seq_lens = torch.tensor(actual_lens, dtype=torch.long)
            self.total_tokens = int(sum(actual_lens))
            self.input_ids = torch.cat(prefill_tokens) if prefill_tokens else torch.tensor([], dtype=torch.long)
            return

        self.input_ids = torch.tensor(
            [request.get_last_token_id() for request in self.requests],
            dtype=torch.long,
        )

        kv_lens = torch.tensor([request.computed_len for request in self.requests], dtype=torch.long)
        set_forward_metadata(kv_len=kv_lens)

        spec_tokens_list = []
        for request in self.requests:
            if request.mtp_info:
                spec_tokens_list.append(request.mtp_info.spec_tokens)
            else:
                self.mtp_infos = None
                return

        self.mtp_infos = MTPInfo(
            spec_tokens=torch.stack(spec_tokens_list, dim=0),
        )

    def update_requests_from_batch(
        self,
        is_prefill: bool,
        next_tokens: Optional[torch.Tensor],
        infer_time: Optional[List],
    ) -> Dict[int, List[int]]:
        """Split batch outputs by index and update each request in-place."""
        next_tokens_by_request: Dict[int, List[int]] = {}
        computed_lens = get_forward_metadata().kv_len

        if next_tokens is not None and next_tokens.shape[0] != len(self.requests):
            raise ValueError(
                f"next_tokens batch size {next_tokens.shape[0]} does not match request count {len(self.requests)}"
            )
        if computed_lens is not None and computed_lens.shape[0] != len(self.requests):
            raise ValueError(
                f"kv_len batch size {computed_lens.shape[0]} does not match request count {len(self.requests)}"
            )

        for i, request in enumerate(self.requests):
            accepted_num = None

            if self.mtp_infos:
                accepted_num = self.mtp_infos.accepted_num[i] if self.mtp_infos.accepted_num is not None else None
                spec_tokens = self.mtp_infos.spec_tokens[i] if self.mtp_infos.spec_tokens is not None else None
                request.update_mtp_info(accepted_num, spec_tokens)

            if next_tokens is not None:
                if accepted_num is not None:
                    request_next_tokens = next_tokens[i, :accepted_num + 1].tolist()
                else:
                    request_next_tokens = next_tokens[i].tolist()
                request.output_id_list += request_next_tokens
                next_tokens_by_request[request.request_id] = request_next_tokens

            if computed_lens is not None:
                request.computed_len = computed_lens[i].item()
                if accepted_num is not None:
                    request.computed_len += accepted_num.item()

            if infer_time is not None:
                request.infer_time.append(infer_time)

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
