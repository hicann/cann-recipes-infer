# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Shared BF16 components and proposal interface for GQA/MLA DSpark models."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch_npu
from torch import nn

from .modeling_kimi_k3 import _offline_infer_config
from .modules import all_gather_first_dim, reduce_scatter_first_dim, vocab_tp_to_owner
from module.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

InferenceConfig = object
CommManager = object

_DSPARK_TP_GROUP = "dspark_tp_group"


def _max_dspark_seq_len(infer_config: InferenceConfig) -> int:
    return (
        infer_config.data_config.input_max_len
        + infer_config.data_config.max_new_tokens
        + infer_config.model_config.next_n
    )


class K3DSparkRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = float(eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual=None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = torch_npu.npu_rms_norm(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )[0]
        else:
            hidden_states, _, residual = torch_npu.npu_add_rms_norm(
                residual,
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )
        return hidden_states, residual


def _yarn_find_correction_dim(
    rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(
            beta_fast, dim, base, max_position_embeddings
        )
    )
    high = math.ceil(
        _yarn_find_correction_dim(
            beta_slow, dim, base, max_position_embeddings
        )
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_ramp(low: int, high: int, size: int) -> torch.Tensor:
    if low == high:
        high += 1
    ramp = (torch.arange(size, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0, 1)


def _yarn_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class K3DSparkRotaryEmbedding(nn.Module):
    """YaRN rotary cache sized to the configured offline inference capacity."""

    def __init__(self, config, max_seq_len: int):
        super().__init__()
        rope = config.rope_parameters or {}
        dim = config.qk_rope_head_dim
        if dim % 2:
            raise ValueError("DSpark RoPE dimension must be even")
        base = rope.get("rope_theta", config.rope_theta)
        factor = rope.get("factor", 1.0)
        original_max = rope.get(
            "original_max_position_embeddings", config.max_position_embeddings
        )
        beta_fast = rope.get("beta_fast", 32.0)
        beta_slow = rope.get("beta_slow", 1.0)

        freq_extra = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        if rope.get("rope_type", "default") == "yarn":
            freq_inter = freq_extra / factor
            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, dim, base, original_max
            )
            extrapolation = 1.0 - _yarn_ramp(low, high, dim // 2)
            inv_freq = freq_inter * (1.0 - extrapolation) + freq_extra * extrapolation
        else:
            inv_freq = freq_extra

        mscale = rope.get("mscale", 1.0)
        mscale_all_dim = rope.get("mscale_all_dim", 0.0)
        amplitude = _yarn_mscale(factor, mscale) / _yarn_mscale(
            factor, mscale_all_dim
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        # Preserve the full-dimension BF16 contract of the fused operators.
        fused_freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (fused_freqs.cos() * amplitude).to(torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (fused_freqs.sin() * amplitude).to(torch.bfloat16),
            persistent=False,
        )

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.clamp_min(0)
        flat = positions.view(-1)
        shape = (*positions.shape, self.cos_cached.shape[-1])
        cos = self.cos_cached.index_select(0, flat).view(shape)
        sin = self.sin_cached.index_select(0, flat).view(shape)
        return cos, sin


class K3DSparkMLP(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: Optional[CommManager],
        prefix: str,
    ):
        super().__init__()
        self.tp_size = infer_config.model_config.dspark_tp_size
        self.tp_rank = comm_manager.get_rank(_DSPARK_TP_GROUP)
        self.tp_group = comm_manager.get_group(_DSPARK_TP_GROUP)
        common = dict(
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            params_dtype=torch.bfloat16,
            quant_config=None,
        )
        # Pack gate first and up second: npu_swiglu activates the first half
        # and multiplies it by the second half.
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            prefix=f"{prefix}.gate_up_proj",
            **common,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Gather the DP request shards for the DSpark MLP, then scatter the
        # row-parallel partial sum back to the original request owners.
        hidden_states = all_gather_first_dim(
            hidden_states, self.tp_group, self.tp_size
        )
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(torch_npu.npu_swiglu(gate_up))
        return reduce_scatter_first_dim(
            hidden_states, self.tp_group, self.tp_size
        )


class K3DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        config,
        prefix: str,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size,
            config.markov_rank,
            config.pad_token_id,
            torch.bfloat16,
        )
        self.markov_w2 = ReplicatedLinear(
            config.markov_rank,
            self.vocab_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.markov_w2",
        )

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full-vocabulary bias and its owner-local embedding."""
        markov_embed = self.markov_w1(token_ids)
        return self.markov_w2(markov_embed), markov_embed


class K3DSparkConfidenceHead(nn.Module):
    """Predict per-token acceptance probabilities from draft/Markov features."""

    def __init__(
        self,
        config,
        prefix: str,
    ) -> None:
        super().__init__()
        self.with_markov = bool(config.confidence_head_with_markov)
        self.hidden_size = int(config.hidden_size)
        self.markov_rank = int(config.markov_rank)
        input_dim = self.hidden_size + (
            self.markov_rank if self.with_markov else 0
        )
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"DSpark confidence hidden width must be {self.hidden_size}, "
                f"got {hidden.shape[-1]}"
            )
        if self.with_markov:
            if hidden.shape[:-1] != markov_embed.shape[:-1]:
                raise ValueError(
                    "DSpark confidence hidden and Markov leading shapes must match"
                )
            if markov_embed.shape[-1] != self.markov_rank:
                raise ValueError(
                    f"DSpark confidence Markov width must be {self.markov_rank}, "
                    f"got {markov_embed.shape[-1]}"
                )

        features = (
            torch.cat([hidden, markov_embed], dim=-1)
            if self.with_markov
            else hidden
        )
        confidence_logits = self.proj(features.to(self.proj.weight.dtype))
        return torch.sigmoid(confidence_logits.float().squeeze(-1))


class K3DSparkForCausalLMBase(nn.Module):
    """Common setup and sampling; subclasses own attention, caches and loading.

    Keep modules on the existing attributes so checkpoint parameter names and
    shared target embedding/lm-head ownership remain unchanged.
    """

    _model_class: type[nn.Module]

    @staticmethod
    def update_model_cfg(config, infer_config: InferenceConfig) -> None:
        quantization = config.quantization_config or config.compression_config
        if quantization:
            raise ValueError("Kimi K3 DSpark supports BF16 weights only")

    def __init__(
        self,
        config,
        runner_settings: dict,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        infer_config = _offline_infer_config(runner_settings)
        self.update_model_cfg(config, infer_config)
        self.config = config
        self.runner_settings = runner_settings
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        world_size = int(infer_config.parallel_config.world_size)
        self.dspark_tp_size = infer_config.model_config.dspark_tp_size
        if world_size % self.dspark_tp_size:
            raise RuntimeError(
                f"DSpark world_size={world_size} must be divisible by "
                f"tp_size={self.dspark_tp_size}"
            )
        if comm_manager is None:
            raise RuntimeError("DSpark TP requires a communication manager")
        comm_manager.register_group(
            name=_DSPARK_TP_GROUP,
            group_num=world_size // self.dspark_tp_size,
            group_size=self.dspark_tp_size,
            hccl_buffer_size=comm_manager.default_hccl_buffer_size,
            group_type=3,
        )
        self.dspark_tp_rank = comm_manager.get_rank(_DSPARK_TP_GROUP)
        self.dspark_tp_group = comm_manager.get_group(_DSPARK_TP_GROUP)
        # DSpark receives target hidden states through the target model's
        # attention-TP request-DP shard and keeps that ownership internally.
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_tp_group = (
            comm_manager.get_group("attn_tp_group")
            if self.attn_tp_size > 1
            else None
        )
        self.attn_tp_rank = (
            comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        )
        self.next_n = infer_config.model_config.next_n
        self.temperature = infer_config.data_config.temperature
        self.mask_token_id = config.mask_token_id
        self.block_size = infer_config.scheduler_config.block_size
        self.execute_mode = runner_settings.get("exe_mode", "eager")
        self.model = self._model_class(
            config,
            infer_config,
            comm_manager,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.markov_head = K3DSparkMarkovHead(config, "markov_head")
        confidence_head_requested = bool(
            infer_config.model_config.custom_params.get(
                "enable_dspark_confidence_head", False
            )
        )
        if confidence_head_requested and not config.enable_confidence_head:
            raise ValueError(
                "enable_dspark_confidence_head requires checkpoint "
                "enable_confidence_head=True"
            )
        self.enable_confidence_head = confidence_head_requested
        self.confidence_head = (
            K3DSparkConfidenceHead(
                config,
                "confidence_head",
            )
            if self.enable_confidence_head
            else None
        )
        self.lm_head = None
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.lmhead_tp_group = (
            comm_manager.get_group("lmhead_tp_group")
            if self.lmhead_tp_size > 1
            else None
        )

    def set_shared_target_modules(self, main_model) -> None:
        main_model.set_draft_config(self.config)
        if main_model.config.num_hidden_layers != self.config.target_num_hidden_layers:
            raise ValueError("target_num_hidden_layers does not match the main model")
        if main_model.config.vocab_size != self.config.vocab_size:
            raise ValueError("draft and target vocab_size must match")
        self.model.embed_tokens = main_model.model.embed_tokens
        self.lm_head = main_model.lm_head

    def _full_vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("DSpark lm_head has not been shared from the target model")
        logits = self.lm_head(hidden_states)
        return vocab_tp_to_owner(
            logits, self.lmhead_tp_group, self.lmhead_tp_size
        )

    def sample(
        self,
        logits: torch.Tensor,
        sample_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(
            logits.float() / max(self.temperature, 1e-5), dim=-1
        )
        if sample_noise is None:
            sample_noise = torch.empty_like(probabilities).exponential_()
        return probabilities.div(sample_noise).argmax(dim=-1)

    def run_decode_proposal(self, decode_inputs: Dict) -> Dict:
        result = self.forward_spec_decode(**decode_inputs)
        spec_tokens, logits, confidence, kv_len, kv_len_cached = result
        return {
            "spec_tokens": spec_tokens,
            "logits": logits,
            "confidence": confidence,
            "kv_len": kv_len,
            "kv_len_cached": kv_len_cached,
        }
