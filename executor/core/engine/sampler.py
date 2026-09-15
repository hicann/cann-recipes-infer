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

"""Token sampling, logprob gathering, and speculative rejection sampling."""

from dataclasses import dataclass

import torch

from executor.utils.forward_metadata import get_forward_metadata
from ..forward_data_info import Batch, LogprobsTensors, SamplingMetadata

_SAMPLING_EPS = 1e-5
_REJECTION_PROB_EPS = 1e-8


class Sampler:
    """Sampler class for token sampling operations."""

    def __init__(self, device: torch.device):
        """Initialize sampler with device.

        Args:
            device: The device to use for tensor operations.
        """
        self.device = device

    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor,
        max_num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        if token_ids.dtype != torch.int64:
            raise ValueError(
                f"Expected token_ids to be torch.int64 type, "
                f"but received type: {token_ids.dtype}"
            )
        # Find the topK values.
        # topk_logprobs: [req_num, token_num, topk]
        topk_logprobs, topk_indices = torch.topk(logprobs, max_num_logprobs, dim=-1)

        # token_ids: [req_num, token_num, 1]
        token_ids = token_ids.unsqueeze(-1)
        # logprobs of sampled token [req_num, token_num, 1]
        token_logprobs = logprobs.gather(dim=-1, index=token_ids)

        # Get the ranks of sampled tokens
        if logprobs.shape[0] < 1:
            raise ValueError("logprobs dim 0 should >= 1")
        if logprobs.shape[0] != token_logprobs.shape[0]:
            raise ValueError("logprobs.shape[0] is not equal to token_logprobs.shape[0]")
        token_ranks = (logprobs >= token_logprobs).sum(dim=-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=2)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=2)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        # indices: [req_num, token_num, max_num_logprobs + 1]
        # logprobs: [req_num, token_num, max_num_logprobs + 1]
        # token_ranks: [req_num, token_num, 1]
        return LogprobsTensors(indices, logprobs, token_ranks)

    def build_sampling_params_from_requests(
        self,
        batch: Batch,
        logits: torch.Tensor
    ) -> SamplingMetadata:
        forward_metadata = get_forward_metadata()
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if batch.is_prefill and cp_metadata is not None and cp_metadata.enabled:
            request_indices = cp_metadata.output_request_indices.detach().cpu().tolist()
        else:
            request_indices = list(range(len(batch.requests)))

        target_size = logits.shape[0]
        temperatures = []
        top_ps = []
        top_ks = []
        top_logprobs = []
        logprobs = []
        valid_top_k_ids = set()
        valid_top_p_ids = set()
        generators = {}
        for row_idx, req_idx in enumerate(request_indices):
            req = batch.requests[req_idx]
            temperatures.append(req.sampling_params.temperature)
            if req.sampling_params.top_p is not None and req.sampling_params.top_p < 1:
                valid_top_p_ids.add(req.request_id)
            top_ps.append(req.sampling_params.top_p)
            if req.sampling_params.top_k is not None and 0 < req.sampling_params.top_k < logits.size(-1):
                valid_top_k_ids.add(req.request_id)
            else:
                req.sampling_params.top_k = logits.size(-1)
            top_ks.append(req.sampling_params.top_k)
            top_logprobs.append(req.sampling_params.top_logprobs)
            logprobs.append(req.sampling_params.logprobs)
            if req.generator is not None:
                generators[row_idx] = req.generator

        pad_count = target_size - len(request_indices)
        if pad_count > 0:
            temperatures.extend([1.0] * pad_count)
            top_ps.extend([1.0] * pad_count)
            top_ks.extend([logits.size(-1)] * pad_count)
            top_logprobs.extend([0] * pad_count)
            logprobs.extend([False] * pad_count)

        temperature = torch.tensor(temperatures, dtype=torch.float32, device=self.device)
        if len(valid_top_p_ids) == 0:
            top_p = None
        else:
            top_p = torch.tensor(top_ps, dtype=torch.float32, device=self.device)
        if len(valid_top_k_ids) == 0:
            top_k = None
        else:
            top_k = torch.tensor(top_ks, dtype=torch.int64, device=self.device)
        top_logprobs_tensor = torch.tensor(top_logprobs, dtype=torch.int64, device=self.device)
        max_num_logprobs = top_logprobs_tensor.max().item()
        logprobs_tensor = torch.tensor(logprobs, dtype=torch.bool, device=self.device)
        logprobs_flag = torch.any(logprobs_tensor).item()
        all_greedy = torch.all(temperature < _SAMPLING_EPS).item()
        all_random = torch.all(temperature >= _SAMPLING_EPS).item()

        return SamplingMetadata(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            all_greedy=all_greedy,
            all_random=all_random,
            max_num_logprobs=max_num_logprobs,
            logprobs=logprobs_flag,
            generators=generators
        )

    def build_sampling_tensors(
        self,
        batch: Batch,
        batch_size: int,
        vocab_size: int,
        *,
        default_temperature: float = 1.0,
        default_top_p: float = 1.0,
        default_top_k: int = 0,
    ) -> dict[str, torch.Tensor | bool]:
        """Build dense per-request parameters for model-side sampling."""
        temperatures = []
        top_ps = []
        top_ks = []
        filter_enabled = False
        for row_idx in range(batch_size):
            if row_idx < len(batch.requests):
                params = batch.requests[row_idx].sampling_params
                temperature = float(params.temperature)
                top_p = float(params.top_p)
                top_k = int(params.top_k)
            else:
                temperature = float(default_temperature)
                top_p = float(default_top_p)
                top_k = int(default_top_k)
            normalized_top_k = top_k if 0 < top_k < vocab_size else vocab_size
            temperatures.append(temperature)
            top_ps.append(top_p)
            top_ks.append(normalized_top_k)
            filter_enabled |= top_p < 1.0 or normalized_top_k < vocab_size
        return {
            "temperature": torch.tensor(temperatures, dtype=torch.float32, device=self.device),
            "top_p": torch.tensor(top_ps, dtype=torch.float32, device=self.device),
            "top_k": torch.tensor(top_ks, dtype=torch.int64, device=self.device),
            "filter_enabled": filter_enabled,
        }

    @staticmethod
    def logits_to_probs(
        logits: torch.Tensor,
        sampling_params: dict[str, torch.Tensor | bool],
    ) -> torch.Tensor:
        """Return the distribution actually sampled after request filters."""
        temperature = sampling_params["temperature"]
        greedy_mask = temperature < _SAMPLING_EPS
        safe_temperature = torch.where(greedy_mask, torch.ones_like(temperature), temperature)
        processed_logits = logits.float() / safe_temperature.view(-1, 1, 1)
        if sampling_params["filter_enabled"]:
            processed_logits = Sampler._filter_logits_kp(
                processed_logits, sampling_params["top_k"], sampling_params["top_p"],
            )
        probs = processed_logits.softmax(dim=-1, dtype=torch.float32)
        greedy_probs = torch.zeros_like(probs)
        greedy_probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return torch.where(greedy_mask.view(-1, 1, 1), greedy_probs, probs)


    def gather_logprobs_for_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors | None:
        """Gather logprobs for caller-selected tokens using normal sampling filters."""
        if not any(request.sampling_params.logprobs for request in batch.requests):
            return None
        sampling_data = self.build_sampling_params_from_requests(batch, logits)
        if not sampling_data.logprobs or sampling_data.max_num_logprobs is None:
            return None

        processed_logits = logits.clone()
        if batch.is_prefill:
            processed_logits = processed_logits[:, -1:, :]
        if not sampling_data.all_greedy:
            greedy_mask = sampling_data.temperature < _SAMPLING_EPS
            temperatures = torch.where(
                greedy_mask,
                torch.ones_like(sampling_data.temperature),
                sampling_data.temperature,
            )
            processed_logits.div_(temperatures.unsqueeze(1).unsqueeze(1))
            processed_logits = self._filter_logits_kp(
                processed_logits,
                sampling_data.top_k,
                sampling_data.top_p,
            )

        processed_logprobs = processed_logits.log_softmax(dim=-1, dtype=torch.float32)
        processed_logprobs = processed_logprobs[:token_ids.shape[0], :token_ids.shape[1]]
        return self.gather_logprobs(
            processed_logprobs,
            max_num_logprobs=sampling_data.max_num_logprobs,
            token_ids=token_ids.long(),
        )

    def random_sample(
        self,
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        q = torch.empty_like(probs, device=self.device)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1)

    @staticmethod
    def request_generators(batch: Batch, batch_size: int) -> dict[int, torch.Generator]:
        """Return row-local RNGs shared by normal and speculative sampling."""
        return {
            row_idx: request.generator
            for row_idx, request in enumerate(batch.requests[:batch_size])
            if request.generator is not None
        }

    def random_like_by_request(
        self,
        reference: torch.Tensor,
        batch: Batch,
        distribution: str,
    ) -> torch.Tensor:
        """Draw request-local noise without coupling RNG state to graph execution."""
        if distribution not in {"exponential", "uniform"}:
            raise ValueError(f"Unsupported random distribution: {distribution}.")
        random_values = torch.empty_like(reference)
        generators = self.request_generators(batch, reference.shape[0])
        random_op = random_values.exponential_ if distribution == "exponential" else random_values.uniform_
        if len(generators) != reference.shape[0]:
            random_op()
        for row_idx, generator in generators.items():
            row_op = random_values[row_idx].exponential_ \
                if distribution == "exponential" else random_values[row_idx].uniform_
            row_op(generator=generator)
        return random_values

    @staticmethod
    def _filter_logits_kp(
        logits: torch.Tensor,
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ) -> torch.Tensor:
        if top_p is None and top_k is None:
            return logits

        sorted_logits, sort_indices = logits.sort(dim=-1, descending=False)

        if top_k is not None:
            k_cutoff = sorted_logits.size(-1) - top_k.to(torch.long)
            k_cutoff = k_cutoff.unsqueeze(-1).unsqueeze(-1).expand(
                sorted_logits.shape[:-1] + (1,)
            )
            k_cutoff = sorted_logits.gather(-1, k_cutoff)
            k_cutoff = sorted_logits < k_cutoff
            sorted_logits.masked_fill_(k_cutoff, -float("inf"))

        if top_p is not None:
            sorted_probs = sorted_logits.softmax(dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1, out=sorted_probs)
            p_cutoff = cum_probs <= 1 - top_p.unsqueeze(dim=1).unsqueeze(dim=1)
            p_cutoff[:, :, -1] = False
            sorted_logits.masked_fill_(p_cutoff, -float("inf"))

        return logits.scatter_(dim=-1, index=sort_indices, src=sorted_logits)

    def _kp_sample(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        top_k: torch.Tensor | None,
        top_p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self._filter_logits_kp(logits, top_k, top_p)
        logits_out = logits.log_softmax(dim=-1, dtype=torch.float32)
        prob_dist = logits.softmax(dim=-1, dtype=torch.float32)
        return self.random_sample(prob_dist, generators), logits_out

    def _sample_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor,
        sampling_data: SamplingMetadata
    ) -> tuple[torch.Tensor, torch.Tensor | None]: # sampled, processed_logprobs
        if batch.is_prefill:
            logits = logits[:, -1:, :]

        if sampling_data.all_greedy and sampling_data.all_random:
            raise ValueError("all_greedy and all_random cannot be True at the same time.")
        if sampling_data.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = torch.argmax(logits, dim=-1)
            if sampling_data.all_greedy:
                processed_logprobs = None
                if sampling_data.logprobs and sampling_data.max_num_logprobs is not None:
                    processed_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
                return greedy_sampled, processed_logprobs

        if sampling_data.temperature is None:
            raise ValueError("sampling_data.temperature cannot be None here")

        # Apply temperature.
        greedy_mask = sampling_data.temperature < _SAMPLING_EPS
        if not sampling_data.all_random:
            sampling_data.temperature = torch.where(
                greedy_mask, 1.0, sampling_data.temperature
            )
        logits.div_(sampling_data.temperature.unsqueeze(dim=1).unsqueeze(dim=1))

        # Apply top_k and top_p.
        random_sampled, processed_logprobs = self._kp_sample(
            logits,
            sampling_data.generators,
            sampling_data.top_k,
            sampling_data.top_p,
        )

        logprobs_out = processed_logprobs if sampling_data.logprobs else None
        if greedy_sampled is None:
            return random_sampled, logprobs_out

        greedy_mask = greedy_mask.unsqueeze(-1)
        sampled = torch.where(
            greedy_mask,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled, logprobs_out

    def sample_and_gather_logprobs(
        self,
        batch: Batch,
        logits: torch.Tensor
    ) -> tuple[torch.Tensor, LogprobsTensors | None]:
        """Sample tokens and gather logprobs for a batch.

        Args:
            batch: Batch containing requests.
            logits: Model output logits.

        Returns:
            Tuple of (next_tokens, logprobs_tensors)
        """
        if logits.shape[0] == 0:
            token_logits = logits[:, -1:, :] if batch.is_prefill else logits
            next_tokens = torch.empty(
                token_logits.shape[:-1],
                dtype=torch.long,
                device=logits.device,
            )
            return next_tokens, None

        logits = logits.clone()
        sampling_data = self.build_sampling_params_from_requests(batch, logits)
        next_tokens, processed_logprobs = self._sample_tokens(batch, logits, sampling_data)

        # Process logprobs
        next_tokens = next_tokens.long()
        raw_logprobs = processed_logprobs
        if not sampling_data.logprobs or raw_logprobs is None:
            logprobs_tensors = None
        else:
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs,
                max_num_logprobs=sampling_data.max_num_logprobs,
                token_ids=next_tokens
            )

        return next_tokens, logprobs_tensors


@dataclass(frozen=True)
class RejectionSamplingResult:
    """Result of speculative rejection sampling."""

    accepted_num: torch.Tensor
    output_tokens: torch.Tensor
    proposal_lens: torch.Tensor


class SpeculativeRejectionSampler:
    """Verify proposals from ``q`` while preserving target distribution ``p``."""

    def __init__(self, sampler: Sampler):
        self._sampler = sampler

    @staticmethod
    def _residual_probs(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the normalized positive residual of target minus draft."""
        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        residual_mass = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(residual_mass <= _REJECTION_PROB_EPS, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
        return residual / residual_mass.clamp_min(_REJECTION_PROB_EPS)

    @staticmethod
    def _truncate_accepted_at_eos(
        batch: Batch,
        draft_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        proposal_lens: torch.Tensor,
        eos_token_ids,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cap accepted and verified proposal lengths at the first accepted EOS."""
        if not eos_token_ids or draft_tokens.shape[1] == 0:
            return accepted_num, proposal_lens

        token_indices = torch.arange(
            draft_tokens.shape[1], device=draft_tokens.device,
        ).unsqueeze(0)
        accepted_mask = token_indices < accepted_num.unsqueeze(1)
        eos_mask = torch.zeros_like(accepted_mask)
        for eos_token_id in eos_token_ids:
            eos_mask |= draft_tokens == eos_token_id

        batch_size = draft_tokens.shape[0]
        if batch.requests:
            observes_eos_values = [
                not request.sampling_params.ignore_eos
                for request in batch.requests[:batch_size]
            ]
            observes_eos_values.extend(
                [False] * (batch_size - len(observes_eos_values))
            )
            observes_eos = torch.tensor(
                observes_eos_values,
                dtype=torch.bool,
                device=draft_tokens.device,
            ).unsqueeze(1)
            eos_mask &= observes_eos

        accepted_eos = eos_mask & accepted_mask
        has_accepted_eos = accepted_eos.any(dim=1)
        eos_prefix_len = accepted_eos.to(torch.int64).argmax(dim=1) + 1
        accepted_num = torch.where(has_accepted_eos, eos_prefix_len, accepted_num)
        proposal_lens = torch.where(
            has_accepted_eos,
            eos_prefix_len,
            proposal_lens,
        )
        return accepted_num, proposal_lens

    @staticmethod
    def _validate_inputs(
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        fallback_tokens: torch.Tensor,
        proposal_lens: torch.Tensor,
    ) -> None:
        batch_size, draft_len = draft_tokens.shape
        if draft_probs.shape[:2] != (batch_size, draft_len):
            raise ValueError(
                "Draft probability shape must match draft tokens, "
                f"got tokens={tuple(draft_tokens.shape)} and probs={tuple(draft_probs.shape)}."
            )
        if target_probs.shape[0] != batch_size or target_probs.shape[1] < draft_len + 1:
            raise ValueError(
                "Target probabilities must contain each draft position plus a bonus position, "
                f"got draft_len={draft_len} and target shape={tuple(target_probs.shape)}."
            )
        if draft_probs.shape[-1] != target_probs.shape[-1]:
            raise ValueError(
                "Rejection sampling requires matching draft and target vocab sizes, "
                f"got {draft_probs.shape[-1]} and {target_probs.shape[-1]}."
            )
        if fallback_tokens.shape[0] != batch_size or fallback_tokens.shape[1] < draft_len + 1:
            raise ValueError(
                "Fallback tokens must contain the full verification block, "
                f"got draft_len={draft_len} and fallback shape={tuple(fallback_tokens.shape)}."
            )
        if proposal_lens.numel() != batch_size:
            raise ValueError(
                f"Proposal lengths batch size mismatch: expected {batch_size}, "
                f"got {proposal_lens.numel()}."
            )

    def sample(
        self,
        *,
        batch: Batch,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        fallback_tokens: torch.Tensor,
        proposal_lens: torch.Tensor,
        eos_token_ids=None,
    ) -> RejectionSamplingResult:
        """Accept a proposal prefix and sample its rejection or bonus position.

        ``draft_probs`` must describe the distribution that generated each
        proposal and ``target_probs`` must describe the requested target
        distribution. The two distributions may use different temperatures;
        the acceptance ratio remains valid because it is computed from the
        actual proposal distribution ``q`` and target distribution ``p``.
        """
        self._validate_inputs(
            draft_tokens,
            draft_probs,
            target_probs,
            fallback_tokens,
            proposal_lens,
        )
        batch_size, draft_len = draft_tokens.shape
        proposal_lens = proposal_lens.to(
            device=draft_tokens.device,
            dtype=torch.int64,
        ).reshape(-1).clamp(min=0, max=draft_len)
        target_probs = target_probs[:, :draft_len + 1, :]
        selected_target_probs = target_probs[:, :draft_len, :].gather(
            dim=-1,
            index=draft_tokens.unsqueeze(-1),
        ).squeeze(-1)
        selected_draft_probs = draft_probs.gather(
            dim=-1,
            index=draft_tokens.unsqueeze(-1),
        ).squeeze(-1).clamp_min(_REJECTION_PROB_EPS)
        accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)

        token_indices = torch.arange(draft_len, device=draft_tokens.device).unsqueeze(0)
        valid_mask = token_indices < proposal_lens.unsqueeze(1)
        accept_draw = self._sampler.random_like_by_request(accept_prob, batch, "uniform")
        accept_mask = ((accept_draw < accept_prob) & valid_mask).to(torch.int64)
        accepted_num = accept_mask.cumprod(dim=1).sum(dim=1).to(torch.int64)
        accepted_num, proposal_lens = self._truncate_accepted_at_eos(
            batch,
            draft_tokens,
            accepted_num,
            proposal_lens,
            eos_token_ids,
        )

        output_tokens = fallback_tokens.clone()
        accepted_mask = token_indices < accepted_num.unsqueeze(1)
        output_tokens[:, :draft_len] = torch.where(
            accepted_mask,
            draft_tokens,
            output_tokens[:, :draft_len],
        )

        rejected_indices = accepted_num.clamp(max=draft_len - 1)
        batch_indices = torch.arange(batch_size, device=draft_tokens.device)
        rejected_target_probs = target_probs[batch_indices, rejected_indices]
        rejected_draft_probs = draft_probs[batch_indices, rejected_indices]
        bonus_target_probs = target_probs[batch_indices, proposal_lens]
        replacement_probs = torch.where(
            (accepted_num < proposal_lens).unsqueeze(1),
            self._residual_probs(rejected_target_probs, rejected_draft_probs),
            bonus_target_probs,
        )
        generators = self._sampler.request_generators(batch, batch_size)
        replacement = self._sampler.random_sample(replacement_probs, generators)
        output_tokens.scatter_(1, accepted_num.unsqueeze(1), replacement.unsqueeze(1))
        return RejectionSamplingResult(
            accepted_num=accepted_num,
            output_tokens=output_tokens,
            proposal_lens=proposal_lens,
        )
