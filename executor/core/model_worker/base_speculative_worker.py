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

"""Common contract and lifecycle for speculative decoding workers."""

from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Type

import torch

from executor.core.config import InferenceConfig
from executor.core.model_worker.model_worker import ModelWorker
from executor.utils.forward_metadata import get_forward_metadata
from ..forward_data_info import BaseSpeculativeInfo, Batch


class BaseSpeculativeWorker(ABC):
    """Method-independent façade used by :class:`ExecutionEngine`.

    Concrete workers own proposal semantics and cache policy. This base class
    only owns the common model-worker lifecycle and target verification
    template, keeping the engine independent from MTP and DSpark details.
    """

    method: ClassVar[str]
    state_cls: ClassVar[Type[BaseSpeculativeInfo]]

    def __init__(
        self,
        infer_config: InferenceConfig,
        device,
    ):
        self.infer_config = infer_config
        self.next_n = infer_config.speculative_config.num_speculative_tokens
        self.exe_mode = infer_config.model_config.exe_mode
        self.device = device
        self.kvcache_manager = None

    @property
    @abstractmethod
    def model_worker(self) -> ModelWorker:
        """Return the model worker owned by this speculative method."""

    @property
    def model(self):
        return self.model_worker.model

    @property
    def prefill_lookahead_tokens(self) -> int:
        """Extra cache slots required while the target prefill is scheduled."""
        return 0

    @staticmethod
    def _pad_seq_len_to_size(
        tensor: torch.Tensor,
        size: int,
        pad_value: int | float = 0,
    ) -> torch.Tensor:
        """Pad or truncate the sequence dimension to ``size``."""
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions, where dim 1 is seq_len")
        if tensor.shape[1] >= size:
            return tensor[:, :size]
        return torch.cat([
            tensor,
            torch.full(
                (tensor.shape[0], size - tensor.shape[1], *tensor.shape[2:]),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ], dim=1)

    def _build_main_verification_inputs(
        self,
        input_ids: torch.Tensor,
        spec_tokens: torch.Tensor,
        q_len: int,
    ):
        """Build packed target-model inputs for speculative verification."""
        kv_len = get_forward_metadata().kv_len.to(self.device)
        input_ids = torch.cat([input_ids.unsqueeze(1), spec_tokens], dim=1)
        input_ids = input_ids[:, -q_len:].reshape(-1).clone()
        kv_len = kv_len + q_len
        indices = torch.arange(q_len - 1, -1, -1, device=self.device)
        position_ids = (kv_len.unsqueeze(1) - indices).clamp(min=0).reshape(-1)
        return input_ids, kv_len, position_ids, q_len

    def init(self, model_cls, config_cls, comm_manager=None):
        self.model_worker.init(model_cls, config_cls, comm_manager=comm_manager)

    def get_cache_info(self):
        return self.model_worker.get_cache_info()

    def init_kvcache(self):
        self.model_worker.init_kvcache()

    def get_offload_workspace_memory_info(self):
        return self.model_worker.get_offload_workspace_memory_info()

    def init_offload_workspace(self):
        self.model_worker.init_offload_workspace()

    def share_weights_from_main_model(self, main_model):
        """Share optional embedding and LM-head parameters with the target."""
        draft_model = self.model_worker.model
        if hasattr(draft_model, "lm_head") and draft_model.lm_head is None:
            if not hasattr(main_model, "lm_head"):
                raise ValueError(
                    f"{draft_model.__class__.__name__} requires the main model lm_head, "
                    f"but it is missing from {main_model.__class__.__name__}."
                )
            draft_model.lm_head = main_model.lm_head

        if hasattr(draft_model.model, "embed_tokens") and draft_model.model.embed_tokens is None:
            if not hasattr(main_model.model, "embed_tokens"):
                raise ValueError(
                    f"{draft_model.__class__.__name__} requires the main model embed_tokens, "
                    f"but it is missing from {main_model.__class__.__name__}."
                )
            draft_model.model.embed_tokens = main_model.model.embed_tokens

    def configure_main_model(self, main_model):
        """Configure optional generic outputs required by a speculative worker."""
        pass

    def verify_spec_tokens(
        self,
        sampler,
        batch: Batch,
        selected_logits: torch.Tensor,
        target_logits: torch.Tensor,
        eos_token_ids=None,
    ):
        """Verify draft tokens and return the tokens committed by the target."""
        next_tokens, logprobs_tensors = self.prepare_target_tokens(
            sampler,
            batch,
            selected_logits,
        )
        accepted_num, next_tokens, proposal_lens = self._verify_spec_tokens(
            batch,
            next_tokens,
            target_logits,
            eos_token_ids,
        )
        self._update_verification_metrics(batch, accepted_num, proposal_lens)
        if logprobs_tensors is None:
            logprobs_tensors = sampler.gather_logprobs_for_tokens(
                batch,
                selected_logits,
                next_tokens,
            )
        return accepted_num, next_tokens, logprobs_tensors

    @staticmethod
    def _update_verification_metrics(batch, accepted_num, proposal_lens) -> None:
        """Record metrics before the next proposal replaces the verified state."""
        if batch.is_prefill or accepted_num is None:
            return
        default_proposal_len = (
            batch.draft_info.spec_tokens.shape[-1]
            if batch.draft_info is not None and batch.draft_info.spec_tokens is not None
            else 0
        )
        for row_idx, request in enumerate(batch.requests[:accepted_num.numel()]):
            if request.valid_output_len is not None:
                continue
            request.spec_num_forward_ct += 1
            request.spec_num_accepted_tokens += int(accepted_num[row_idx].item())
            request.spec_num_draft_tokens += int(
                proposal_lens[row_idx].item() if proposal_lens is not None else default_proposal_len
            )

    @abstractmethod
    def warm_up_prefill(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        model_inputs: Dict,
        prev_hidden_states: torch.Tensor,
    ) -> None:
        """Warm the configured proposal backend's prefill path."""

    @abstractmethod
    def build_main_warmup_batch(self, input_ids: torch.Tensor) -> Batch:
        """Build the target verification batch used during warm-up."""

    @abstractmethod
    def warm_up_decode(self, model_inputs: Dict, main_output) -> None:
        """Warm the configured proposal backend's decode path."""

    def prepare_target_tokens(self, sampler, batch: Batch, logits: torch.Tensor):
        """Sample target tokens only when the selected verifier consumes them."""
        if self.requires_target_sample(batch):
            return sampler.sample_and_gather_logprobs(batch, logits)
        return torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device), None

    def requires_target_sample(self, batch: Batch) -> bool:
        """Whether verification consumes independently sampled target tokens."""
        return True

    def _verify_spec_tokens(
        self,
        batch: Batch,
        main_next_tokens: torch.Tensor,
        target_logits: Optional[torch.Tensor] = None,
        eos_token_ids=None,
    ):
        """Run phase-independent checks before selecting a verification strategy."""
        if batch.is_prefill:
            accepted_num = torch.zeros(
                main_next_tokens.shape[0], dtype=torch.int64, device=self.device,
            )
            return accepted_num, main_next_tokens, None

        batch_size = batch.input_ids.shape[0]
        spec_tokens = batch.draft_info.spec_tokens
        if spec_tokens is None or spec_tokens.shape[-1] == 0:
            accepted_num = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
            return accepted_num, main_next_tokens[:batch_size], None
        return self._verify_decode_tokens(
            batch, spec_tokens, main_next_tokens[:batch_size], target_logits, eos_token_ids,
        )

    def _verify_decode_tokens(
        self,
        batch: Batch,
        spec_tokens: torch.Tensor,
        main_next_tokens: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        eos_token_ids=None,
    ):
        """Accept the longest exact-match proposal prefix."""
        draft_len = spec_tokens.shape[-1]
        token_mask = spec_tokens == main_next_tokens[:, :draft_len]
        rejected = torch.logical_not(token_mask)
        rejected_pos = rejected.to(torch.int64).argmax(dim=-1)
        accepted_num = torch.where(
            rejected.any(dim=-1), rejected_pos, token_mask.shape[-1],
        ).to(self.device)
        return accepted_num, main_next_tokens, None

    @abstractmethod
    def get_main_model_inputs(self, input_ids: torch.Tensor, batch: Optional[Batch]):
        """Build target-model verification inputs."""

    @abstractmethod
    def inference(
        self,
        batch: Batch,
        main_next_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        model_inputs_main: Dict,
        prev_hidden_states: torch.Tensor,
    ):
        """Generate the next proposal from the completed target verification.

        Inputs:
            batch: Current requests and sampling settings; draft_info is updated in place.
            main_next_tokens: Verified output tokens; column accepted_num is the new seed.
            accepted_num: Accepted draft prefix length per request, excluding the new seed.
            model_inputs_main: Positions and lengths from the preceding target forward.
            prev_hidden_states: Target representations consumed by the backend.

        Returns:
            A list of model execution durations in seconds. New candidates are
            persisted in batch.draft_info, not returned as part of this list.
        """
