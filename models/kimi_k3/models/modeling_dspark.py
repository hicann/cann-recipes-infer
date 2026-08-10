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

"""Pure-BF16 RadixArk Kimi K3 DSpark proposal model.

The target model supplies the configured intermediate hidden states. The draft
model projects their concatenation into a compact context, evaluates a fixed
proposal block with Qwen3 GQA layers, then applies a token-conditioned Markov
bias before sampling the speculative tokens.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers.utils import logging

from executor.model_loader.weight_utils import default_weight_loader
from .modeling_kimi_k3 import _offline_infer_config
from .modules import build_paged_slot_mapping
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

logger = logging.get_logger(__name__)

InferenceConfig = object
CommManager = object


def _max_dspark_seq_len(infer_config: InferenceConfig) -> int:
    return (
        infer_config.data_config.input_max_len
        + infer_config.data_config.max_new_tokens
        + infer_config.model_config.next_n
    )


def _all_gather_last_dim(tensor: torch.Tensor, group, size: int) -> torch.Tensor:
    if size <= 1:
        return tensor
    shards = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=-1)


def _all_gather_first_dim(tensor: torch.Tensor, group, size: int) -> torch.Tensor:
    if size <= 1:
        return tensor
    gathered = tensor.new_empty(tensor.shape[0] * size, *tensor.shape[1:])
    dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=group)
    return gathered


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
        self.tp_size = infer_config.parallel_config.dense_tp_size
        self.tp_rank = (
            comm_manager.get_rank("dense_tp_group") if self.tp_size > 1 else 0
        )
        self.tp_group = (
            comm_manager.get_group("dense_tp_group") if self.tp_size > 1 else None
        )
        common = dict(
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            params_dtype=torch.bfloat16,
            quant_config=None,
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            prefix=f"{prefix}.gate_proj",
            **common,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            prefix=f"{prefix}.up_proj",
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
        hidden_states = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        if self.tp_size > 1:
            dist.all_reduce(hidden_states, group=self.tp_group)
        return hidden_states


class K3DSparkAttention(nn.Module):
    """Non-causal Qwen3 GQA over committed context and one noise block."""

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: Optional[CommManager],
        prefix: str,
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.tp_size = int(infer_config.parallel_config.attn_tp_size)
        self.tp_rank = (
            comm_manager.get_rank("attn_tp_group") if self.tp_size > 1 else 0
        )
        self.tp_group = (
            comm_manager.get_group("attn_tp_group") if self.tp_size > 1 else None
        )
        self.block_size = int(infer_config.scheduler_config.block_size)
        self.softmax_scale = self.head_dim ** -0.5
        common = dict(
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
        )
        # Cache ownership follows requests, so every owner rank computes full
        # Q/K/V rather than the checkpoint's attention-TP head shards.
        self.q_proj = ReplicatedLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            prefix=f"{prefix}.q_proj",
            **common,
        )
        self.k_proj = ReplicatedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            prefix=f"{prefix}.k_proj",
            **common,
        )
        self.v_proj = ReplicatedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            prefix=f"{prefix}.v_proj",
            **common,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = K3DSparkRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = K3DSparkRMSNorm(self.head_dim, config.rms_norm_eps)
        self.attn_type = "FullAttention"

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.k_proj(context_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        key, _ = self.k_norm(key)
        value = self.v_proj(context_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        cos, sin = context_cos_sin
        dummy_query = torch.empty_like(key)
        _, key = torch_npu.npu_apply_rotary_pos_emb(
            dummy_query,
            key,
            cos.reshape(-1, 1, self.head_dim),
            sin.reshape(-1, 1, self.head_dim),
            layout="TND",
        )
        return key, value

    def _project_noise_qkv(
        self,
        hidden_states: torch.Tensor,
        draft_cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_proj(hidden_states).view(
            -1, self.total_num_heads, self.head_dim
        )
        query, _ = self.q_norm(query)
        key = self.k_proj(hidden_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        key, _ = self.k_norm(key)
        value = self.v_proj(hidden_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        cos, sin = draft_cos_sin
        query, key = torch_npu.npu_apply_rotary_pos_emb(
            query,
            key,
            cos.reshape(-1, 1, self.head_dim),
            sin.reshape(-1, 1, self.head_dim),
            layout="TND",
        )
        return query, key, value

    def _update_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        layer_cache: Dict[str, torch.Tensor],
    ) -> None:
        slots = slots.reshape(-1).to(device=key.device, dtype=torch.int64)
        torch_npu.npu_scatter_nd_update_(
            layer_cache["k_cache"].view(
                -1, self.num_kv_heads, self.head_dim
            ),
            slots.view(-1, 1),
            key.reshape(-1, self.num_kv_heads, self.head_dim),
        )
        torch_npu.npu_scatter_nd_update_(
            layer_cache["v_cache"].view(
                -1, self.num_kv_heads, self.head_dim
            ),
            slots.view(-1, 1),
            value.reshape(-1, self.num_kv_heads, self.head_dim),
        )

    def prefill_context_cache(
        self,
        context_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Dict[str, torch.Tensor],
        layer_cache: Dict[str, torch.Tensor],
    ) -> None:
        key, value = self._project_context_kv(context_states, context_cos_sin)
        self._update_cache(
            key,
            value,
            attn_metadata["context_slot_mapping"],
            layer_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        draft_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Dict[str, torch.Tensor],
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        layer_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        batch_size, draft_len = hidden_states.shape[:2]
        local_batch = context_states.shape[0]
        request_start = self.tp_rank * local_batch
        request_end = request_start + local_batch
        request_slice = slice(request_start, request_end)
        hidden_states = hidden_states[request_slice]
        query, draft_key, draft_value = self._project_noise_qkv(
            hidden_states, draft_cos_sin
        )
        context_key, context_value = self._project_context_kv(
            context_states, context_cos_sin
        )
        context_key = context_key.view(
            local_batch, -1, self.num_kv_heads, self.head_dim
        )
        context_value = context_value.view(
            local_batch, -1, self.num_kv_heads, self.head_dim
        )
        draft_key = draft_key.view(
            local_batch, draft_len, self.num_kv_heads, self.head_dim
        )
        draft_value = draft_value.view(
            local_batch, draft_len, self.num_kv_heads, self.head_dim
        )
        combined_key = torch.cat((context_key, draft_key), dim=1)
        combined_value = torch.cat((context_value, draft_value), dim=1)
        combined_slots = torch.cat(
            (
                attn_metadata["context_slot_mapping"],
                attn_metadata["draft_slot_mapping"],
            ),
            dim=1,
        )
        self._update_cache(
            combined_key,
            combined_value,
            combined_slots,
            layer_cache,
        )
        k_cache = layer_cache["k_cache"]
        v_cache = layer_cache["v_cache"]
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            k_cache.view(*k_cache.shape[:2], -1),
            v_cache.view(*v_cache.shape[:2], -1),
            atten_mask=None,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=attn_metadata["block_table"],
            num_query_heads=self.total_num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.softmax_scale,
            input_layout="TND",
            sparse_mode=0,
            block_size=self.block_size,
        )
        output = attn_output.reshape(
            local_batch, draft_len, self.total_num_heads * self.head_dim
        )
        if self.tp_size > 1:
            gathered_output = output.new_empty(
                batch_size, draft_len, output.shape[-1]
            )
            dist.all_gather_into_tensor(
                gathered_output, output.contiguous(), group=self.tp_group
            )
            local_width = gathered_output.shape[-1] // self.tp_size
            output = gathered_output.narrow(
                -1, self.tp_rank * local_width, local_width
            )
        output = self.o_proj(output)
        if self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: Optional[CommManager],
        layer_idx: int,
        prefix: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = K3DSparkAttention(
            config, infer_config, comm_manager, f"{prefix}.self_attn"
        )
        self.mlp = K3DSparkMLP(
            config, infer_config, comm_manager, f"{prefix}.mlp"
        )
        self.input_layernorm = K3DSparkRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = K3DSparkRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        context_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        draft_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Dict[str, torch.Tensor],
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        layer_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states,
            context_states,
            context_cos_sin,
            draft_cos_sin,
            attn_metadata,
            actual_seq_qlen,
            actual_seq_kvlen,
            layer_cache,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: Optional[CommManager],
        prefix: str = "model",
    ):
        super().__init__()
        self.config = config
        context_width = config.target_hidden_size * len(
            config.target_layer_ids
        )
        self.fc = ReplicatedLinear(
            context_width,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.fc",
        )
        self.hidden_norm = K3DSparkRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    config,
                    infer_config,
                    comm_manager,
                    layer_idx,
                    f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = K3DSparkRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.embed_tp_rank = (
            comm_manager.get_rank("embed_tp_group")
            if self.embed_tp_size > 1
            else 0
        )
        self.embed_tp_group = (
            comm_manager.get_group("embed_tp_group")
            if self.embed_tp_size > 1
            else None
        )

        self.block_size = infer_config.scheduler_config.block_size
        self.max_cache_len = _max_dspark_seq_len(infer_config)
        self.rotary_emb = K3DSparkRotaryEmbedding(config, self.max_cache_len)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tp_size <= 1:
            return self.embed_tokens(input_ids)
        vocab_per_rank = self.config.vocab_size // self.embed_tp_size
        local_ids = input_ids - self.embed_tp_rank * vocab_per_rank
        mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
        hidden_states = self.embed_tokens(local_ids * mask) * mask.unsqueeze(-1)
        dist.all_reduce(hidden_states, group=self.embed_tp_group)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        draft_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        context_slot_mapping: torch.Tensor,
        draft_slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
    ) -> torch.Tensor:
        attn_metadata = {
            "context_slot_mapping": context_slot_mapping,
            "draft_slot_mapping": draft_slot_mapping,
            "block_table": block_table,
        }
        context_states, _ = self.hidden_norm(
            self.fc(target_hidden_states)
        )
        hidden_states = self.embed(input_ids)
        residual = None
        for layer, layer_cache in zip(self.layers, cache_data):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                context_states,
                context_cos_sin,
                draft_cos_sin,
                attn_metadata,
                actual_seq_qlen,
                actual_seq_kvlen,
                layer_cache,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def prefill_context_cache(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        slot_block_table: torch.Tensor,
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
    ) -> None:
        batch_size, context_len = context_positions.shape
        positions = context_positions.view(batch_size, context_len)
        attn_metadata = {
            "context_slot_mapping": build_paged_slot_mapping(
                positions, slot_block_table, self.block_size
            ),
        }
        context_cos_sin = self.rotary_emb(context_positions)
        for layer, layer_cache in zip(self.layers, cache_data):
            layer.self_attn.prefill_context_cache(
                context_states,
                context_cos_sin,
                attn_metadata,
                layer_cache,
            )


class K3DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: Optional[CommManager],
        prefix: str,
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.embed_tp_rank = (
            comm_manager.get_rank("embed_tp_group")
            if self.embed_tp_size > 1
            else 0
        )
        self.lmhead_tp_rank = (
            comm_manager.get_rank("lmhead_tp_group")
            if self.lmhead_tp_size > 1
            else 0
        )
        self.embed_tp_group = (
            comm_manager.get_group("embed_tp_group")
            if self.embed_tp_size > 1
            else None
        )
        self.lmhead_tp_group = (
            comm_manager.get_group("lmhead_tp_group")
            if self.lmhead_tp_size > 1
            else None
        )
        self.markov_w1 = VocabParallelEmbedding(
            self.vocab_size,
            config.markov_rank,
            config.pad_token_id,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.embed_tp_rank,
        )
        self.markov_w2 = ColumnParallelLinear(
            config.markov_rank,
            self.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=self.lmhead_tp_rank,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.markov_w2",
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tp_size > 1:
            vocab_per_rank = self.vocab_size // self.embed_tp_size
            local_ids = token_ids - self.embed_tp_rank * vocab_per_rank
            mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
            markov_embed = self.markov_w1(local_ids * mask) * mask.unsqueeze(-1)
            dist.all_reduce(markov_embed, group=self.embed_tp_group)
        else:
            markov_embed = self.markov_w1(token_ids)
        logits = self.markov_w2(markov_embed)
        logits = _all_gather_last_dim(
            logits, self.lmhead_tp_group, self.lmhead_tp_size
        )
        return logits


class K3DSparkForCausalLM(nn.Module):
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
        # DSpark receives target hidden states through the target model's
        # attention-TP shard and restores the request axis before packing.
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
        self.model = K3DSparkModel(
            config,
            infer_config,
            comm_manager,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.markov_head = K3DSparkMarkovHead(
            config, infer_config, comm_manager, "markov_head"
        )
        self.lm_head = None
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.lmhead_tp_group = (
            comm_manager.get_group("lmhead_tp_group")
            if self.lmhead_tp_size > 1
            else None
        )
    def prepare_target_hidden_states(
        self, target_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Project the local Prefill shard before owner-directed routing."""
        context_states, _ = self.model.hidden_norm(
            self.model.fc(target_hidden_states)
        )
        return context_states

    def set_shared_target_modules(self, main_model) -> None:
        main_model.set_draft_config(self.config)
        if main_model.config.num_hidden_layers != self.config.target_num_hidden_layers:
            raise ValueError("target_num_hidden_layers does not match the main model")
        if main_model.config.vocab_size != self.config.vocab_size:
            raise ValueError("draft and target vocab_size must match")
        self.model.embed_tokens = main_model.model.embed_tokens
        self.lm_head = main_model.lm_head

    def check_model_settings(self) -> None:
        parallel = self.infer_config.parallel_config
        num_heads = int(self.config.num_attention_heads)
        num_kv_heads = int(self.config.num_key_value_heads)
        head_dim = int(self.config.head_dim)
        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise RuntimeError("DSpark GQA head counts and head_dim must be positive")
        if num_heads % num_kv_heads:
            raise RuntimeError("num_attention_heads must be divisible by num_key_value_heads")
        if (num_heads * head_dim) % int(parallel.attn_tp_size):
            raise RuntimeError("DSpark attention width must be divisible by attn_tp_size")
        if int(parallel.dense_tp_size) != int(parallel.attn_tp_size):
            raise RuntimeError("dense_tp_size must equal attn_tp_size for DSpark")
        if self.config.intermediate_size % parallel.dense_tp_size:
            raise RuntimeError("intermediate_size must be divisible by dense_tp_size")
        if self.config.vocab_size % parallel.embed_tp_size:
            raise RuntimeError("vocab_size must be divisible by embed_tp_size")
        if self.config.vocab_size % parallel.lmhead_tp_size:
            raise RuntimeError("vocab_size must be divisible by lmhead_tp_size")
        if self.config.num_hidden_layers <= 0:
            raise RuntimeError("DSpark requires a positive layer count")
        if bool(getattr(self.config, "attention_bias", False)):
            raise RuntimeError("DSpark supports bias-free attention only")
        if bool(getattr(self.config, "use_sliding_window", False)) or getattr(
            self.config, "sliding_window", None
        ) is not None:
            raise RuntimeError("RadixArk Kimi-K3 DSpark requires full attention")
        if self.block_size not in (16, 128):
            raise RuntimeError("DSpark GQA PA requires cache block_size 16 or 128")
        if self.next_n != int(self.config.block_size):
            raise RuntimeError(
                f"next_n={self.next_n} must match the checkpoint block_size="
                f"{self.config.block_size}"
            )
        if self.config.markov_head_type != "vanilla":
            raise RuntimeError("only the vanilla DSpark Markov head is supported")

    def _full_vocab_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError("DSpark lm_head has not been shared from the target model")
        logits = self.lm_head(hidden_states)
        return _all_gather_last_dim(
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

    def forward_spec_decode(
        self,
        main_hidden: torch.Tensor,
        main_next_tokens: torch.Tensor,
        cached_len: torch.Tensor,
        context_cos: torch.Tensor,
        context_sin: torch.Tensor,
        draft_cos: torch.Tensor,
        draft_sin: torch.Tensor,
        context_slot_mapping: torch.Tensor,
        draft_slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
        sample_noise: Optional[torch.Tensor] = None,
    ):
        main_next_tokens = main_next_tokens[:, 0]
        main_next_tokens = _all_gather_first_dim(
            main_next_tokens, self.attn_tp_group, self.attn_tp_size
        )
        batch_size = main_next_tokens.shape[0]
        draft_input_ids = main_next_tokens.new_full(
            (batch_size, self.next_n), self.mask_token_id
        )
        draft_input_ids[:, 0] = main_next_tokens

        lm_hidden = self.model(
            draft_input_ids,
            main_hidden,
            (context_cos, context_sin),
            (draft_cos, draft_sin),
            context_slot_mapping,
            draft_slot_mapping,
            block_table,
            actual_seq_qlen,
            actual_seq_kvlen,
            cache_data,
        )
        logits = self._full_vocab_logits(lm_hidden).float()
        output_ids = main_next_tokens.new_empty(batch_size, self.next_n + 1)
        output_ids[:, 0] = main_next_tokens
        for step in range(self.next_n):
            markov_bias = self.markov_head(output_ids[:, step])
            logits[:, step].add_(markov_bias.float())
            noise = None if sample_noise is None else sample_noise[:, step]
            output_ids[:, step + 1] = self.sample(logits[:, step], noise)

        return (
            output_ids[:, 1:],
            logits,
            cached_len + self.next_n,
            cached_len,
        )

    def propose(
        self,
        input_dict: Dict,
        main_next_tokens: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> Dict:
        context_positions = input_dict["target_hidden_positions"]
        block_table = input_dict["block_table"]
        slot_block_table = input_dict["slot_block_table"]
        cache_data = input_dict["cache_data"]
        batch_size = target_hidden_states.shape[0]
        is_prefill = bool(input_dict.get("is_prefill", False))
        if is_prefill:
            context_states = target_hidden_states
            self.model.prefill_context_cache(
                context_states,
                context_positions,
                slot_block_table,
                cache_data,
            )
            cached_len = context_positions.view(batch_size, -1).max(dim=1).values + 1
            return {
                "spec_tokens": main_next_tokens.new_empty(batch_size, 0),
                "logits": None,
                "kv_len": cached_len,
                "kv_len_cached": cached_len,
            }

        decode_batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        local_batch = decode_batch_size // self.attn_tp_size
        context_lengths = (
            context_positions.view(local_batch, -1).max(dim=1).values + 1
        )
        last_context_position = context_lengths - 1
        active_rows = last_context_position >= 0
        offsets = torch.arange(
            1,
            self.next_n + 1,
            device=context_positions.device,
            dtype=torch.long,
        )
        draft_positions = last_context_position.unsqueeze(1) + offsets.unsqueeze(0)
        draft_positions = torch.where(
            active_rows.unsqueeze(1),
            draft_positions,
            torch.full_like(draft_positions, -1),
        )
        context_cos, context_sin = self.model.rotary_emb(context_positions)
        draft_cos, draft_sin = self.model.rotary_emb(draft_positions)
        context_slot_mapping = build_paged_slot_mapping(
            context_positions,
            slot_block_table,
            self.block_size,
        )
        draft_slot_mapping = build_paged_slot_mapping(
            draft_positions,
            slot_block_table,
            self.block_size,
        )
        stable_next_tokens = main_next_tokens[:, :1]
        actual_seq_qlen = [self.next_n * (idx + 1) for idx in range(local_batch)]
        context_lengths_list = context_lengths.detach().cpu().tolist()
        actual_seq_kvlen = [
            length + self.next_n for length in context_lengths_list
        ]
        sample_noise = None
        if self.temperature > 0:
            sample_noise = torch.empty(
                decode_batch_size,
                self.next_n,
                self.config.vocab_size,
                device=target_hidden_states.device,
                dtype=torch.float32,
            ).exponential_()
        decode_inputs = {
            "main_hidden": target_hidden_states.contiguous(),
            "main_next_tokens": stable_next_tokens,
            "cached_len": context_lengths,
            "context_cos": context_cos,
            "context_sin": context_sin,
            "draft_cos": draft_cos,
            "draft_sin": draft_sin,
            "context_slot_mapping": context_slot_mapping,
            "draft_slot_mapping": draft_slot_mapping,
            "block_table": block_table,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_kvlen": actual_seq_kvlen,
            "cache_data": cache_data,
            "sample_noise": sample_noise,
        }
        if self.execute_mode != "eager":
            for value in decode_inputs.values():
                if isinstance(value, torch.Tensor):
                    torch._dynamo.mark_static(value)
                elif isinstance(value, tuple):
                    for layer_cache in value:
                        for cache_value in layer_cache.values():
                            if isinstance(cache_value, torch.Tensor):
                                torch._dynamo.mark_static(cache_value)
        result = self.forward_spec_decode(**decode_inputs)
        spec_tokens, logits, kv_len, kv_len_cached = result
        return {
            "spec_tokens": spec_tokens,
            "logits": logits,
            "kv_len": kv_len,
            "kv_len_cached": kv_len_cached,
        }

    @staticmethod
    def _weight_candidates(name: str) -> Tuple[str, ...]:
        candidates = [name]
        for source_prefix in ("draft_model.", "model.draft_model."):
            if name.startswith(source_prefix):
                candidates.append(name[len(source_prefix):])
        expanded = list(candidates)
        for candidate in candidates:
            if candidate.startswith(("fc.", "hidden_norm.", "layers.", "norm.")):
                expanded.append(f"model.{candidate}")
        return tuple(dict.fromkeys(expanded))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        ignored_fragments = (
            "embed_tokens.weight",
            "lm_head.weight",
            "confidence_head.",
        )
        for name, tensor in weights:
            if any(fragment in name for fragment in ignored_fragments):
                continue
            param_name = next(
                (candidate for candidate in self._weight_candidates(name) if candidate in params),
                None,
            )
            if param_name is None:
                logger.debug("Skip non-runtime RadixArk DSpark tensor: %s", name)
                continue
            param = params[param_name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded.add(param_name)

        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} RadixArk DSpark parameters were not loaded, "
                f"starting with {missing[:8]}"
            )
        return loaded

    def process_weights_after_loading(self) -> None:
        is_nz = self.infer_config.model_config.enable_weight_nz
        for module_name, module in self.named_modules():
            # lm_head is shared from the target model after its weights have
            # already been transposed and format-cast by the main runner.
            if module_name == "lm_head":
                continue
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(module, is_nz=is_nz)

__all__ = [
    "K3DSparkAttention",
    "K3DSparkDecoderLayer",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
]
