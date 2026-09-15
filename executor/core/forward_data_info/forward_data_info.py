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

from dataclasses import dataclass, field, fields
from typing import Any, List, Dict, Optional

import torch
from executor.utils.forward_metadata import get_forward_metadata, set_forward_metadata


@dataclass
class Logprobs:
    """
        dim info:
        logprob_token_id: [max_num_logprobs + 1]
        logprobs: [max_num_logprobs + 1]
        selected_token_rank: [1]
    """
    logprob_token_id: list
    logprobs: list
    selected_token_rank: int


@dataclass
class LogprobsTensors:
    """
        dim info:
        logprob_token_ids: [num_reqs, num_tokens, max_num_logprobs + 1]
        logprobs_tensors: [num_reqs, num_tokens, max_num_logprobs + 1]
        selected_token_ranks: [num_reqs, num_tokens, 1]
    """
    logprob_token_ids: torch.Tensor
    logprobs_tensors: torch.Tensor
    selected_token_ranks: torch.Tensor

    def filter(self, req_idx: int, accepted_num: int | None = None) -> list[Logprobs]:
        output_len = self.logprob_token_ids.size(1)
        if accepted_num is not None:
            output_len = accepted_num + 1
        logprob_token_ids = self.logprob_token_ids[req_idx, :output_len]
        logprobs_tensors = self.logprobs_tensors[req_idx, :output_len]
        selected_token_ranks = self.selected_token_ranks[req_idx, :output_len]
        logprobs_list = []
        for i in range(output_len):
            logprobs = Logprobs(
                logprob_token_id=logprob_token_ids[i].tolist(),
                logprobs=logprobs_tensors[i].tolist(),
                selected_token_rank=selected_token_ranks[i].item()
            )
            logprobs_list.append(logprobs)
        return logprobs_list


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    max_num_logprobs: int | None
    logprobs: bool
    generators: dict[int, torch.Generator] = field(default_factory=dict)


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
class BaseSpeculativeInfo:
    """State shared by speculative decoding backends.

    Attributes:
        is_prefill: Whether the current phase is prefill (initial token processing).
        spec_tokens: Accumulated speculative token sequence for main model verification.
        accepted_num: Number of accepted tokens per sample in current iteration.
    """
    is_prefill: Optional[bool] = False
    spec_tokens: Optional[torch.Tensor] = None
    accepted_num: Optional[torch.Tensor] = None

    @classmethod
    def stack(cls, infos: List["BaseSpeculativeInfo"]) -> "BaseSpeculativeInfo":
        """Batch request-local state while preserving the concrete state type."""
        if not infos:
            return cls()

        values_by_field = {}
        for state_field in fields(cls):
            values = [getattr(info, state_field.name) for info in infos]
            if state_field.name == "is_prefill":
                values_by_field[state_field.name] = values[0]
            elif all(value is None for value in values):
                values_by_field[state_field.name] = None
            else:
                values_by_field[state_field.name] = torch.stack(values, dim=0)
        return cls(**values_by_field)

    def select(self, index: int) -> "BaseSpeculativeInfo":
        """Extract one request from batched speculative state."""
        values_by_field = {}
        for state_field in fields(self):
            value = getattr(self, state_field.name)
            values_by_field[state_field.name] = (
                value if state_field.name == "is_prefill" or value is None else value[index]
            )
        return type(self)(**values_by_field)


    def update_request(self, request: "Request", is_prefill: bool) -> None:
        """Persist this request-local state without dropping backend fields."""
        request.update_draft_info(self)

    @staticmethod
    def progress_base(total_lens: torch.Tensor, is_prefill: bool):
        """Return the absolute progress base, or None for committed-token increments."""
        return total_lens if is_prefill else None


@dataclass
class MTPInfo(BaseSpeculativeInfo):
    """Native MTP state with its established update interface."""

    def update_request(self, request: "Request", is_prefill: bool) -> None:
        """Keep the established native MTP request update path."""
        accepted_num = None if is_prefill or self.accepted_num is None else int(self.accepted_num.item())
        request.update_mtp_info(accepted_num, self.spec_tokens)

    def progress_base(self, total_lens: torch.Tensor, is_prefill: bool):
        """Remove native MTP draft forwards from the shared metadata cursor."""
        next_n = self.spec_tokens.shape[-1]
        return total_lens - next_n - (next_n - 1)

    def set_mtp_info(self, **kwargs):
        """Update known MTP state attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class DSparkInfo(BaseSpeculativeInfo):
    """DSpark candidates and their sampling distributions."""

    draft_probs: Optional[torch.Tensor] = None
    proposal_lens: Optional[torch.Tensor] = None


@dataclass
class SamplingParams:
    """Sampling parameters for generation.

    Attributes:
        max_tokens: Maximum tokens to generate.
        ignore_eos: If True, do not stop on EOS token.
    """

    max_tokens: Optional[int] = None
    ignore_eos: bool = False
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    seed: Optional[int] = None


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
        mm_inputs: Opaque model-specific multimodal payload consumed by MM Encode.
        mm_token_count: Number of multimodal embedding rows produced for this request.
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
        spec_num_draft_tokens: Number of valid proposal tokens eligible for
            speculative acceptance statistics.
        eos_output_len: Output length through the first EOS token. This records
            EOS inside a multi-token MTP step without making finish decisions.
        cp_rank: Group-local rank that owns this request's partitioned persistent cache.
    """
    request_id: int
    prompt: "str | List[dict]"
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    input_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    computed_len: int = 0
    prompt_tokens: int = 0
    output_id_list: List[int] = field(default_factory=list)
    output_logprobs: List[Logprobs] = field(default_factory=list)
    is_prefill_done: bool = False
    is_finished: bool = False
    finish_reason: Optional[str] = None
    # Legacy storage name retained so existing constructors and online transfer
    # code can continue passing ``mtp_info``. New runtime code should use the
    # backend-neutral ``draft_info`` property below.
    mtp_info: Optional[BaseSpeculativeInfo] = None
    # Metrics
    spec_num_forward_ct: int = 0
    spec_num_accepted_tokens: int = 0
    spec_num_draft_tokens: int = 0
    # Measure the combined inference time of the main model and the MTP model
    infer_time: List[float] = field(default_factory=list)
    # Step counter for decode phase
    decode_step_count: int = 0
    valid_output_len: Optional[int] = None
    eos_output_len: Optional[int] = None
    # PD disaggregation fields (only populated in online PD mode).
    bootstrap_host: str = ""
    bootstrap_port: int = -1
    bootstrap_room: int = -1
    disagg_prefill_dp_rank: int = -1
    metadata_buffer_index: int = -1
    disagg_kv_sender: Optional[Any] = None
    cp_rank: int = 0
    generator: torch.Generator = None
    mm_inputs: Optional[dict] = None
    mm_token_count: int = 0

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

    @property
    def draft_info(self) -> Optional[BaseSpeculativeInfo]:
        """Backend-neutral access to this request's speculative state."""
        return self.mtp_info

    @draft_info.setter
    def draft_info(self, value: Optional[BaseSpeculativeInfo]) -> None:
        self.mtp_info = value

    def update_draft_info(self, draft_info: BaseSpeculativeInfo):
        self.draft_info = draft_info

    def update_mtp_info(
        self,
        accepted_num,
        spec_tokens,
    ):
        """Update native MTP state through its established Request API."""
        if not self.mtp_info:
            self.mtp_info = MTPInfo()
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
        draft_info: Batched speculative state shared by all draft backends.
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

    # Legacy storage name retained for constructor compatibility. Runtime code
    # should use the singular, backend-neutral ``draft_info`` property.
    mtp_infos: Optional[BaseSpeculativeInfo] = None

    # True when this batch was synthesized to keep DP+TP collectives aligned
    # on ranks with no local work (online PD). Engine runs forward normally so
    # collectives complete, but scheduler skips state updates / output emit.
    is_dummy: bool = False

    @property
    def draft_info(self) -> Optional[BaseSpeculativeInfo]:
        """Backend-neutral access to the batch's speculative state."""
        return self.mtp_infos

    @draft_info.setter
    def draft_info(self, value: Optional[BaseSpeculativeInfo]) -> None:
        self.mtp_infos = value

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

        draft_infos = [request.draft_info for request in self.requests]
        if not all(draft_infos):
            self.draft_info = None
            return
        state_cls = type(draft_infos[0])
        self.draft_info = state_cls.stack(draft_infos)

    def update_requests_from_batch(
        self,
        is_prefill: bool,
        next_tokens: Optional[torch.Tensor],
        infer_time: Optional[float],
        logprobs_tensors: Optional[LogprobsTensors] = None,
        eos_token_ids: Optional[set[int]] = None,
    ) -> Dict[int, List[int]]:
        """Split batch outputs by index and update each request in-place."""
        next_tokens_by_request: Dict[int, List[int]] = {}
        forward_metadata = get_forward_metadata()
        total_lens = forward_metadata.kv_len

        request_indices = list(range(len(self.requests)))
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if is_prefill and cp_metadata is not None and cp_metadata.enabled:
            request_indices = cp_metadata.output_request_indices.detach().cpu().tolist()

        if next_tokens is not None and next_tokens.shape[0] != len(request_indices):
            raise ValueError(
                f"next_tokens batch size {next_tokens.shape[0]} does not match request count {len(request_indices)}"
            )
        if total_lens is not None and total_lens.shape[0] != len(request_indices):
            raise ValueError(
                f"kv_len batch size {total_lens.shape[0]} does not match request count {len(request_indices)}"
            )

        for output_idx, request_idx in enumerate(request_indices):
            request = self.requests[request_idx]
            accepted_count = None

            if self.draft_info:
                state = self.draft_info.select(output_idx)
                accepted_count = int(state.accepted_num.item()) if state.accepted_num is not None else None
                state.update_request(request, is_prefill)
                computed_lens = self.draft_info.progress_base(total_lens, is_prefill)
            else:
                computed_lens = total_lens

            if next_tokens is not None:
                if accepted_count is not None:
                    request_next_tokens = next_tokens[output_idx, :accepted_count + 1].tolist()
                else:
                    request_next_tokens = next_tokens[output_idx].tolist()
                old_output_len = len(request.output_id_list)
                request.output_id_list += request_next_tokens
                if eos_token_ids and request.eos_output_len is None:
                    for token_idx, token_id in enumerate(request_next_tokens):
                        if token_id in eos_token_ids:
                            request.eos_output_len = old_output_len + token_idx + 1
                            break
                next_tokens_by_request[request.request_id] = request_next_tokens

            if logprobs_tensors is not None:
                request_logprobs = logprobs_tensors.filter(output_idx, accepted_count)
                request.output_logprobs += request_logprobs

            if computed_lens is not None:
                request.computed_len = computed_lens[output_idx].item()
                if accepted_count is not None:
                    request.computed_len += accepted_count
            elif accepted_count is not None:
                request.computed_len += accepted_count + 1

            if infer_time is not None:
                request.infer_time.append(infer_time)

        return next_tokens_by_request


@dataclass
class MMEncodeBatch:
    """Requests selected for one multimodal encode step."""

    requests: List[Request] = field(default_factory=list)
    is_dummy: bool = False
