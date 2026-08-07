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

"""Sampler for token sampling and logprobs gathering."""

import torch

from executor.utils.forward_metadata import get_forward_metadata
from ..forward_data_info import Batch, LogprobsTensors, SamplingMetadata

_SAMPLING_EPS = 1e-5


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