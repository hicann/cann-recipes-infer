# coding=utf-8
# Adapted from
# https://github.com/deepseek-ai/DeepSpec
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2026 The DeepSpec Authors. All rights reserved.
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

"""DSpark speculative draft-model worker."""

from typing import Dict, Optional, Tuple

import torch

from executor.core.config import InferenceConfig
from executor.core.model_worker.model_worker import ModelWorker
from executor.utils.forward_metadata import get_forward_metadata
from .base_speculative_worker import BaseSpeculativeWorker
from ..engine.sampler import Sampler, SpeculativeRejectionSampler
from ..forward_data_info import Batch, DSparkInfo


SAMPLING_EPS = 1e-5


class DSparkWorker(BaseSpeculativeWorker):
    """Generate DSpark blocks with framework-managed proposal KV cache.

    Draft models expose dspark_target_layer_ids, dspark_noise_token_id and
    dspark_block_size. prepare_proposal_inputs converts logical worker inputs
    to model arguments outside execution timing; propose consumes those arguments.
    Decode warm-up uses forward_spec_decode_graph/compiled_forward_spec_decode.
    Target models provide set_auxiliary_hidden_layers and auxiliary_hidden_states.
    Network-specific attention metadata belongs to the draft model, not this worker.
    """

    method = "dspark"
    state_cls = DSparkInfo

    def __init__(
        self,
        infer_config: InferenceConfig,
        device,
    ):
        super().__init__(infer_config, device)
        self.dspark_model_worker = ModelWorker(
            infer_config,
            device,
            is_draft_model=True,
            model_path=infer_config.speculative_config.draft_model_path,
        )
        self.sampler = Sampler(device)
        self.speculative_rejection_sampler = SpeculativeRejectionSampler(self.sampler)
        speculative_config = infer_config.speculative_config
        self.confidence_threshold = speculative_config.confidence_threshold
        self.draft_temperature = speculative_config.draft_temperature

    @property
    def model_worker(self) -> ModelWorker:
        return self.dspark_model_worker

    @property
    def prefill_lookahead_tokens(self) -> int:
        # Reserve space for the first proposal immediately after target prefill.
        return self.next_n

    def configure_main_model(self, main_model):
        configure_layers = getattr(main_model, "set_auxiliary_hidden_layers", None)
        if not callable(configure_layers):
            raise TypeError(
                f"DSpark requires {type(main_model).__name__} to implement "
                "set_auxiliary_hidden_layers(layer_ids)."
            )
        configure_layers(self.model.dspark_target_layer_ids)

    def warm_up_prefill(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        model_inputs: Dict,
        prev_hidden_states: Dict[str, torch.Tensor],
    ) -> None:
        """Warm DSpark proposal prefill using framework-managed PA cache."""
        batch_size = seq_lens.numel()
        warmup_batch = Batch(
            is_prefill=True,
            input_ids=input_ids,
            seq_lens=seq_lens,
        )
        warmup_batch.draft_info = DSparkInfo()
        # Match DeepSpec: target prefill supplies the current token and hidden
        # context, then DSpark initializes its cache and proposes immediately.
        main_next_tokens = input_ids.new_zeros((batch_size, 1))
        accepted_num = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self.inference(
            batch=warmup_batch,
            main_next_tokens=main_next_tokens,
            accepted_num=accepted_num,
            model_inputs_main=model_inputs,
            prev_hidden_states=prev_hidden_states,
        )

    def build_main_warmup_batch(self, input_ids: torch.Tensor) -> Batch:
        """Build the fixed-width proposal block used by real main-model decode."""
        batch_size = input_ids.shape[0]
        spec_tokens = input_ids.new_zeros((batch_size, self.next_n))
        batch = Batch(
            is_prefill=False,
            input_ids=input_ids,
        )
        batch.draft_info = DSparkInfo(
            spec_tokens=spec_tokens,
        )
        return batch

    def warm_up_decode(self, model_inputs: Dict, main_output) -> None:
        logits, prev_hidden_states = main_output
        if self.exe_mode in ["ge_graph", "npugraph_ex"]:
            self.dspark_model_worker.compile_model(
                decode_q_len=self.next_n,
                interface_name="forward_spec_decode_graph",
                compiled_attr="compiled_forward_spec_decode",
            )
        batch_size = logits.shape[0]
        # Match the regular verification path: every logit position predicts the
        # token that follows the corresponding main-model input. With a full
        # verification block, accepted_num=0 below selects position zero together
        # with its matching hidden state and position metadata.
        main_next_tokens = torch.argmax(logits, dim=-1)
        warmup_batch = Batch(
            is_prefill=False,
            input_ids=main_next_tokens[:, 0],
        )
        warmup_batch.draft_info = DSparkInfo(
            spec_tokens=main_next_tokens.new_empty(batch_size, 0),
        )
        accepted_num = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self.inference(
            batch=warmup_batch,
            main_next_tokens=main_next_tokens,
            accepted_num=accepted_num,
            model_inputs_main=model_inputs,
            prev_hidden_states=prev_hidden_states,
        )

    def _sampling_tensors(
        self,
        batch: Batch,
        batch_size: int,
        vocab_size: int,
        *,
        for_draft: bool = False,
    ):
        sampling_tensors = self.sampler.build_sampling_tensors(
            batch,
            batch_size,
            vocab_size,
            default_temperature=self.infer_config.data_config.temperature,
            default_top_p=self.infer_config.data_config.top_p,
            default_top_k=self.infer_config.data_config.top_k,
        )
        if for_draft and self.draft_temperature is not None:
            sampling_tensors["temperature"].fill_(self.draft_temperature)
        return sampling_tensors

    def _proposal_sample_noise(
        self,
        batch: Batch,
        batch_size: int,
        vocab_size: int,
    ) -> Optional[torch.Tensor]:
        if self.draft_temperature is not None:
            temperatures = [self.draft_temperature] * batch_size
        else:
            temperatures = [
                float(request.sampling_params.temperature)
                for request in batch.requests[:batch_size]
            ]
            temperatures.extend([
                float(self.infer_config.data_config.temperature)
                for _ in range(batch_size - len(temperatures))
            ])
        if all(temperature < SAMPLING_EPS for temperature in temperatures):
            return None
        reference = torch.empty(
            batch_size,
            self.next_n,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
        )
        return self.sampler.random_like_by_request(reference, batch, "exponential")

    def _verify_decode_tokens(
        self,
        batch: Batch,
        spec_tokens: torch.Tensor,
        main_next_tokens: torch.Tensor,
        target_logits: Optional[torch.Tensor],
        eos_token_ids=None,
    ):
        """Verify stochastic DSpark proposals while preserving the target distribution."""
        batch_size = batch.input_ids.shape[0]
        draft_len = spec_tokens.shape[-1]

        state = batch.draft_info
        proposal_lens = self._get_proposal_lens(
            state, batch_size, draft_len, spec_tokens.device,
        )
        if target_logits is None or state.draft_probs is None:
            raise RuntimeError(
                "Rejection sampling requires both target logits and draft probabilities."
            )

        target_logits = target_logits[:batch_size]
        target_sampling = self._sampling_tensors(
            batch, batch_size, target_logits.shape[-1],
        )
        draft_probs = state.draft_probs[:, :draft_len, :]
        target_probs = self.sampler.logits_to_probs(
            target_logits[:, :draft_len + 1, :],
            target_sampling,
        )
        result = self.speculative_rejection_sampler.sample(
            batch=batch,
            draft_tokens=spec_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            fallback_tokens=main_next_tokens,
            proposal_lens=proposal_lens,
            eos_token_ids=eos_token_ids,
        )
        return result.accepted_num, result.output_tokens, result.proposal_lens

    def requires_target_sample(self, batch: Batch) -> bool:
        spec_tokens = batch.draft_info.spec_tokens if batch.draft_info is not None else None
        return (
            batch.is_prefill
            or spec_tokens is None
            or spec_tokens.shape[-1] == 0
        )

    @staticmethod
    def _get_proposal_lens(
        state: DSparkInfo,
        batch_size: int,
        draft_len: int,
        device,
    ) -> torch.Tensor:
        if state.proposal_lens is None:
            return torch.full((batch_size,), draft_len, dtype=torch.int64, device=device)
        proposal_lens = state.proposal_lens.to(device=device, dtype=torch.int64).reshape(-1)
        return proposal_lens.clamp(min=0, max=draft_len)

    @staticmethod
    def _pack_prefill_to_dense(tensor: torch.Tensor, seq_lens: torch.Tensor, pad_value: int = 0):
        """Restore packed prefill tensors to a dense [batch, seq, ...] context layout."""
        batch_size = seq_lens.numel()
        max_seq_len = int(seq_lens.max().item())
        dense = tensor.new_full((batch_size, max_seq_len, *tensor.shape[1:]), pad_value)
        start = 0
        for batch_idx, seq_len_tensor in enumerate(seq_lens):
            seq_len = int(seq_len_tensor.item())
            dense[batch_idx, :seq_len] = tensor[start:start + seq_len]
            start += seq_len
        return dense

    @staticmethod
    def _select_committed_context(
        target_hidden_states: torch.Tensor,
        target_hidden_positions: torch.Tensor,
        accepted_num: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep only tokens committed by verification while preserving decode graph shape."""
        batch_size, main_len = target_hidden_positions.shape
        accepted_len = (accepted_num.to(target_hidden_positions.device).to(torch.long) + 1).clamp(
            min=1,
            max=main_len,
        )
        token_idx = torch.arange(main_len, device=target_hidden_positions.device).view(1, main_len)
        valid_mask = token_idx < accepted_len.view(batch_size, 1)
        target_hidden_states = torch.where(
            valid_mask.unsqueeze(-1),
            target_hidden_states,
            torch.zeros_like(target_hidden_states),
        )
        target_hidden_positions = torch.where(
            valid_mask,
            target_hidden_positions,
            target_hidden_positions.new_full(target_hidden_positions.shape, -1),
        )
        return target_hidden_states, target_hidden_positions

    def _confident_prefix_lengths(
        self,
        confidence: Optional[torch.Tensor],
        spec_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        # Confidence defines a request-local valid prefix inside the fixed-width
        # verification block used by the target graph.
        batch_size = spec_tokens.shape[0]
        max_tokens = min(int(self.next_n), int(spec_tokens.shape[1]))
        if max_tokens == 0:
            proposal_lens = torch.zeros(batch_size, dtype=torch.int64, device=spec_tokens.device)
        elif self.confidence_threshold <= 0.0:
            proposal_lens = torch.full(
                (batch_size,),
                max_tokens,
                dtype=torch.int64,
                device=spec_tokens.device,
            )
        elif confidence is None:
            proposal_lens = torch.zeros(batch_size, dtype=torch.int64, device=spec_tokens.device)
        else:
            confidence = confidence.float().reshape(confidence.shape[0], -1)[:, :max_tokens]
            below_threshold = confidence.sigmoid() < self.confidence_threshold
            token_indices = torch.arange(max_tokens, dtype=torch.int64, device=confidence.device).unsqueeze(0)
            proposal_lens = torch.where(
                below_threshold,
                token_indices,
                torch.full_like(token_indices, max_tokens),
            ).amin(dim=1)

        # Keep one fixed verification bucket. proposal_lens masks each request's
        # valid prefix, so confidence changes never require a device-to-host
        # synchronization or rank-dependent graph selection in the decode path.
        return proposal_lens, max_tokens

    def get_main_model_inputs(self, input_ids, batch):
        """Build main-model verification inputs when draft state is available."""
        draft_info = batch.draft_info if batch is not None else None
        if draft_info is None or draft_info.spec_tokens is None:
            return None

        draft_info.spec_tokens = draft_info.spec_tokens.to(self.device)
        spec_tokens = draft_info.spec_tokens
        q_len = spec_tokens.shape[-1] + 1
        return self._build_main_verification_inputs(input_ids, spec_tokens, q_len)

    def inference(
        self,
        batch: Batch,
        main_next_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        model_inputs_main: Dict,
        prev_hidden_states: Dict[str, torch.Tensor],
    ):
        """Generate one proposal using the BaseSpeculativeWorker input contract.

        accepted_num selects committed hidden states, the new seed and KV length.
        prev_hidden_states contains the target auxiliary_hidden_states dictionary.
        Mutates batch.draft_info with this round's acceptance and next round's
        token/probability/length state; returns one execution duration in seconds.
        """
        if not batch.draft_info:
            batch.draft_info = DSparkInfo()
        batch.draft_info.accepted_num = accepted_num
        batch.draft_info.is_prefill = batch.is_prefill
        batch.draft_info.spec_tokens = None

        target_hidden_states = None
        # A target model may return both normal and proposal-specific hidden
        # states. The latter are consumed only by the
        # proposal model, so they are unpacked here instead of leaking into MTP.
        if isinstance(prev_hidden_states, dict):
            target_hidden_states = prev_hidden_states.get("auxiliary_hidden_states")
        if target_hidden_states is None:
            raise RuntimeError("DSpark requires auxiliary_hidden_states from the main model.")
        if batch.is_prefill:
            input_ids = batch.input_ids.to(self.device)
            proposal_input_ids, target_hidden_states, target_hidden_positions = \
                self._build_prefill_proposal_inputs(batch, input_ids, model_inputs_main, target_hidden_states)
        else:
            proposal_input_ids, target_hidden_states, target_hidden_positions = \
                self._build_decode_proposal_inputs(
                    main_next_tokens,
                    accepted_num,
                    model_inputs_main,
                    target_hidden_states,
                )
        proposal_kv_len = self._get_proposal_kv_len(
            batch,
            accepted_num,
            model_inputs_main,
        )
        # The worker owns request-level state; the model receives only tensors
        # required to initialize cache or generate the next proposal block.
        proposal_inputs = {
            "input_ids": proposal_input_ids,
            "is_prefill": batch.is_prefill,
            "kv_len": proposal_kv_len,
            "target_hidden_positions": target_hidden_positions,
        }
        # DeepSpec creates the first proposal immediately after target prefill.
        # Sampling inputs are therefore required in both prefill and decode.
        draft_model = self.dspark_model_worker.model
        sampling_params = self._sampling_tensors(
            batch,
            proposal_input_ids.shape[0],
            draft_model.config.vocab_size,
            for_draft=True,
        )
        proposal_inputs["sampling_params"] = sampling_params
        proposal_inputs["sample_noise"] = self._proposal_sample_noise(
            batch,
            proposal_input_ids.shape[0],
            draft_model.config.vocab_size,
        )
        proposal_model_worker = self.dspark_model_worker
        if proposal_model_worker.force_eplb:
            if proposal_model_worker.decode_topk_list is None:
                # Force EPLB counts logical proposal positions. HC lanes are
                # handled inside the proposal MoE.
                proposal_tokens = (
                    proposal_input_ids.shape[0]
                    * int(draft_model.dspark_block_size)
                )
                proposal_model_worker.decode_topk_list = (
                    proposal_model_worker.gen_force_eplb_topk_idx(
                        is_prefill=False,
                        total_tokens=proposal_tokens,
                    )
                )
            proposal_inputs["cur_topk_list"] = proposal_model_worker.decode_topk_list
        prepared_inputs = self._prepare_proposal_inputs(
            proposal_inputs,
            proposal_input_ids if not batch.is_prefill else main_next_tokens,
            target_hidden_states,
        )
        proposal, infer_time = self.dspark_model_worker.execute_model_call(
            self.model.propose, prepared_inputs,
        )
        self._update_draft_info_from_proposal(
            batch, proposal_input_ids, proposal, sampling_params,
        )
        return [infer_time]

    def _build_prefill_proposal_inputs(
        self,
        batch: Batch,
        input_ids: torch.Tensor,
        model_inputs_main: Dict,
        target_hidden_states: torch.Tensor,
    ):
        if batch.seq_lens is None:
            raise RuntimeError("seq_lens is required for draft prefill.")
        # Main prefill uses packed tokens. Restore batch rows and let the
        # proposal model populate its framework SlidingWindow KV entries.
        seq_lens = batch.seq_lens.to(device=self.device, dtype=torch.long)
        total_tokens = int(seq_lens.sum().item())
        input_ids = input_ids.view(-1)[:total_tokens]
        position_ids = model_inputs_main["position_ids"].to(self.device).view(-1)[:total_tokens]
        target_hidden_states = target_hidden_states.to(self.device).view(-1, target_hidden_states.shape[-1])
        proposal_input_ids = self._pack_prefill_to_dense(input_ids, seq_lens)
        target_hidden_states = self._pack_prefill_to_dense(target_hidden_states[:total_tokens], seq_lens)
        target_hidden_positions = self._pack_prefill_to_dense(position_ids, seq_lens, pad_value=-1)
        return proposal_input_ids, target_hidden_states, target_hidden_positions

    def _build_decode_proposal_inputs(
        self,
        main_next_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        model_inputs_main: Dict,
        target_hidden_states: torch.Tensor,
    ):
        batch_size, main_len = main_next_tokens.shape
        target_hidden_positions = model_inputs_main["position_ids"].to(self.device).reshape(
            -1, main_len,
        )[:batch_size]
        batch_size, main_len = target_hidden_positions.shape
        # Only the seed and accepted draft inputs have committed target hidden
        # states. The new seed predicted by verification is processed next round.
        target_hidden_states = target_hidden_states.to(self.device).reshape(
            -1, main_len, target_hidden_states.shape[-1],
        )[:batch_size]
        target_hidden_states, target_hidden_positions = self._select_committed_context(
            target_hidden_states,
            target_hidden_positions,
            accepted_num,
        )
        # Prefill already produces a proposal, so regular decode verifies N+1
        # inputs. Padding also accommodates an empty-proposal warm-up state.
        target_context_len = self.next_n + 1
        target_hidden_states = self._pad_seq_len_to_size(
            target_hidden_states,
            target_context_len,
        )
        target_hidden_positions = self._pad_seq_len_to_size(
            target_hidden_positions,
            target_context_len,
            pad_value=-1,
        )
        committed_idx = accepted_num.to(device=self.device, dtype=torch.long).clamp(
            min=0,
            max=main_next_tokens.shape[1] - 1,
        )
        proposal_input_ids = main_next_tokens.gather(1, committed_idx.view(-1, 1)).to(self.device)
        return proposal_input_ids, target_hidden_states, target_hidden_positions

    def _get_proposal_kv_len(
        self,
        batch: Batch,
        accepted_num: torch.Tensor,
        model_inputs_main: Dict,
    ) -> torch.Tensor:
        """Return the committed position that starts the next proposal.

        A proposal call materializes the complete draft block in its physical
        cache. Rejected draft tokens must not advance the logical cursor, so
        derive the next length from target verification positions and the
        accepted prefix instead of continuing from a draft-cache tail.
        """
        metadata_kv_len = get_forward_metadata().kv_len
        if batch.is_prefill:
            return metadata_kv_len

        verify_positions = model_inputs_main["position_ids"].to(
            device=self.device,
            dtype=torch.long,
        ).reshape(-1, self.next_n + 1)[:accepted_num.shape[0]]
        accepted_num = accepted_num.to(device=self.device, dtype=torch.long).reshape(-1)
        committed_source_pos = verify_positions.gather(
            1,
            accepted_num.clamp(max=verify_positions.shape[1] - 1).view(-1, 1),
        ).squeeze(1)
        # The selected target logit predicts the next committed token. Convert
        # its zero-based source position to the cache length after that token.
        return committed_source_pos + 2

    def _update_draft_info_from_proposal(
        self,
        batch: Batch,
        proposal_input_ids: torch.Tensor,
        proposal: Dict,
        sampling_params: Dict[str, torch.Tensor | bool],
    ):
        """Store proposal outputs required by target verification and the next step."""
        batch_size = proposal_input_ids.shape[0]
        state = batch.draft_info
        if batch.is_prefill:
            state.accepted_num = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
            state.is_prefill = False

        proposal_lens, verify_len = self._confident_prefix_lengths(
            proposal.get("confidence"),
            proposal["spec_tokens"],
        )
        state.proposal_lens = proposal_lens
        state.spec_tokens = proposal["spec_tokens"][:, :verify_len]
        if verify_len > 0:
            state.draft_probs = self.sampler.logits_to_probs(
                proposal["logits"][:, :verify_len, :],
                sampling_params,
            )
        else:
            state.draft_probs = None

    def _prepare_proposal_inputs(self, input_dict, main_next_tokens, target_hidden_states):
        """Prepare DSpark tokens and logical positions before model-specific metadata."""
        target_hidden_positions = input_dict.get("target_hidden_positions")
        if target_hidden_positions is None:
            raise ValueError("DSpark proposal requires target hidden positions.")
        is_prefill = input_dict.get("is_prefill", False)
        model_inputs = self._prepare_model_inputs(
            input_dict["input_ids"], input_dict.get("kv_len"),
        )
        stable_main_next_tokens = torch.empty(
            (main_next_tokens.shape[0], 1),
            dtype=main_next_tokens.dtype,
            device=main_next_tokens.device,
        )
        stable_main_next_tokens.copy_(main_next_tokens[:, :1])
        draft_input_ids = stable_main_next_tokens.new_full(
            (main_next_tokens.shape[0], self.next_n), self.model.dspark_noise_token_id)
        draft_input_ids[:, :1] = stable_main_next_tokens
        target_hidden_states = target_hidden_states.contiguous()
        target_hidden_positions = target_hidden_positions.contiguous()
        last_context_pos = target_hidden_positions.to(torch.long).max(dim=1).values
        draft_offsets = torch.arange(
            self.next_n, device=target_hidden_states.device, dtype=torch.long,
        ).view(1, self.next_n)
        draft_positions = last_context_pos.view(-1, 1) + 1 + draft_offsets
        prepared = {
            "input_ids": draft_input_ids,
            "main_hidden": target_hidden_states,
            "target_hidden_positions": target_hidden_positions,
            "draft_positions": draft_positions,
            "is_prefill": is_prefill,
            "main_next_tokens": stable_main_next_tokens,
            "cur_topk_list": input_dict.get("cur_topk_list"),
            "sampling_params": input_dict.get("sampling_params"),
            "sample_noise": input_dict.get("sample_noise"),
        }
        return self.model.prepare_proposal_inputs(prepared, model_inputs)

    def _prepare_model_inputs(self, input_ids: torch.Tensor, kv_len: Optional[torch.Tensor]) -> Dict:
        """Normalize context token positions without assuming an attention backend."""
        batch_size, seq_len = input_ids.size()
        input_ids = input_ids.contiguous().reshape(-1).to(torch.int32)
        if kv_len is None:
            kv_len = get_forward_metadata().kv_len
        if kv_len is not None:
            kv_len = kv_len.to(device=input_ids.device, dtype=torch.long).reshape(batch_size, -1)
            if kv_len.shape[1] == seq_len:
                position_ids = kv_len.reshape(-1)
            else:
                start = kv_len[:, :1] - seq_len + 1
                offsets = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).view(1, seq_len)
                position_ids = (start + offsets).clamp_min(0).reshape(-1)
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).repeat(batch_size)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "forward_metadata": get_forward_metadata(),
            "kv_len": kv_len,
        }
