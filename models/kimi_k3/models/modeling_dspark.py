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

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch_npu
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
from torch import nn
from transformers.utils import logging

from executor.model_loader.weight_utils import default_weight_loader
from .modeling_dspark_common import (
    K3DSparkConfidenceHead,
    K3DSparkForCausalLMBase,
    K3DSparkMarkovHead,
    K3DSparkMLP,
    K3DSparkRMSNorm,
    K3DSparkRotaryEmbedding,
    _max_dspark_seq_len,
)
from .modules import (
    all_gather_first_dim,
    build_paged_slot_mapping,
    reduce_scatter_first_dim,
)
from module.linear import ReplicatedLinear

logger = logging.get_logger(__name__)

InferenceConfig = object
CommManager = object


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
        self.o_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
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
        attn_metadata: Dict[str, Any],
        layer_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        local_batch, draft_len = hidden_states.shape[:2]
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
        self._update_cache(
            combined_key,
            combined_value,
            attn_metadata["combined_slot_mapping"],
            layer_cache,
        )
        k_cache = layer_cache["k_cache"]
        v_cache = layer_cache["v_cache"]
        attn_output, _ = flash_attn(
            query,
            k_cache,
            v_cache,
            block_table=attn_metadata["block_table"],
            attn_mask=None,
            metadata=attn_metadata["fia_metadata"],
            cu_seqlens_q=attn_metadata["cu_seqlens_q"],
            seqused_q=attn_metadata["seqused_q"],
            seqused_kv=attn_metadata["seqused_kv"],
            softmax_scale=self.softmax_scale,
            mask_mode=0,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=draft_len,
            max_seqlen_kv=attn_metadata["max_seqlen_kv"],
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="TND",
            return_softmax_lse=False,
        )
        output = attn_output.reshape(
            local_batch, draft_len, self.total_num_heads * self.head_dim
        )
        # FIA returns all heads for this rank's owner-local requests. Project
        # locally and keep the request-DP shard between decoder submodules.
        output = output.to(device=self.o_proj.weight.device)
        output = self.o_proj(output)
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
        attn_metadata: Dict[str, Any],
        layer_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states,
            context_states,
            context_cos_sin,
            draft_cos_sin,
            attn_metadata,
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
        # The checkpoint has no draft embedding, so this boundary must use the
        # shared target TP group while returning a request-DP shard.
        input_ids = all_gather_first_dim(
            input_ids, self.embed_tp_group, self.embed_tp_size
        )
        vocab_per_rank = self.config.vocab_size // self.embed_tp_size
        local_ids = input_ids - self.embed_tp_rank * vocab_per_rank
        mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
        hidden_states = self.embed_tokens(local_ids * mask) * mask.unsqueeze(-1)
        return reduce_scatter_first_dim(
            hidden_states, self.embed_tp_group, self.embed_tp_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        draft_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        context_slot_mapping: torch.Tensor,
        draft_slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        fia_metadata: torch.Tensor,
        max_seqlen_kv: int,
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
    ) -> torch.Tensor:
        attn_metadata = {
            "combined_slot_mapping": torch.cat(
                (context_slot_mapping, draft_slot_mapping), dim=1
            ),
            "block_table": block_table,
            "fia_metadata": fia_metadata,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "max_seqlen_kv": max_seqlen_kv,
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


class K3DSparkForCausalLM(K3DSparkForCausalLMBase):
    _model_class = K3DSparkModel

    def prepare_target_hidden_states(
        self, target_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Project the local Prefill shard before owner-directed routing."""
        context_states, _ = self.model.hidden_norm(
            self.model.fc(target_hidden_states)
        )
        return context_states


    def check_model_settings(self) -> None:
        parallel = self.infer_config.parallel_config
        num_heads = int(self.config.num_attention_heads)
        num_kv_heads = int(self.config.num_key_value_heads)
        head_dim = int(self.config.head_dim)
        if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise RuntimeError("DSpark GQA head counts and head_dim must be positive")
        if num_heads % num_kv_heads:
            raise RuntimeError("num_attention_heads must be divisible by num_key_value_heads")
        if self.config.intermediate_size % self.dspark_tp_size:
            raise RuntimeError("intermediate_size must be divisible by DSpark tp_size")
        if self.config.vocab_size % parallel.embed_tp_size:
            raise RuntimeError("vocab_size must be divisible by embed_tp_size")
        if self.config.vocab_size % parallel.lmhead_tp_size:
            raise RuntimeError("vocab_size must be divisible by lmhead_tp_size")
        decode_batch_size = int(
            self.infer_config.scheduler_config.batch_size_per_dp_rank
        )
        if decode_batch_size % int(parallel.attn_tp_size):
            raise RuntimeError(
                "decode batch size must be divisible by target attn_tp_size"
            )
        if decode_batch_size % self.dspark_tp_size:
            raise RuntimeError(
                "decode batch size must be divisible by DSpark tp_size"
            )
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
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        fia_metadata: torch.Tensor,
        max_seqlen_kv: int,
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
        sample_noise: Optional[torch.Tensor] = None,
    ):
        main_next_tokens = main_next_tokens[:, 0]
        local_batch = main_next_tokens.shape[0]
        draft_input_ids = main_next_tokens.new_full(
            (local_batch, self.next_n), self.mask_token_id
        )
        draft_input_ids[:, 0] = main_next_tokens

        confidence_hidden = self.model(
            draft_input_ids,
            main_hidden,
            (context_cos, context_sin),
            (draft_cos, draft_sin),
            context_slot_mapping,
            draft_slot_mapping,
            block_table,
            cu_seqlens_q,
            seqused_q,
            seqused_kv,
            fia_metadata,
            max_seqlen_kv,
            cache_data,
        )
        lm_hidden = all_gather_first_dim(
            confidence_hidden, self.lmhead_tp_group, self.lmhead_tp_size
        )
        logits = self._full_vocab_logits(lm_hidden).float()
        output_ids = main_next_tokens.new_empty(local_batch, self.next_n + 1)
        output_ids[:, 0] = main_next_tokens
        markov_embeds = [] if self.confidence_head is not None else None
        for step in range(self.next_n):
            markov_bias, markov_embed = self.markov_head(output_ids[:, step])
            # FP32 logits keep the addition in FP32 with a BF16 Markov bias.
            logits[:, step].add_(markov_bias)
            if markov_embeds is not None:
                markov_embeds.append(markov_embed)
            noise = None if sample_noise is None else sample_noise[:, step]
            output_ids[:, step + 1] = self.sample(logits[:, step], noise)

        spec_tokens = output_ids[:, 1:]
        draft_output = spec_tokens.unsqueeze(-1) if self.temperature <= 0 else logits
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(
                confidence_hidden,
                torch.stack(markov_embeds, dim=1),
            )
        return spec_tokens, draft_output, confidence, cached_len + self.next_n, cached_len

    def prepare_decode_inputs(
        self,
        input_dict: Dict,
        main_next_tokens: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> Dict:
        context_positions = input_dict["target_hidden_positions"]
        block_table = input_dict["block_table"]
        slot_block_table = input_dict["slot_block_table"]
        cache_data = input_dict["cache_data"]
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
        # Build transformer FA metadata outside the compiled graph. Creating
        # device tensors from Python lists inside NPUGraph capture triggers a
        # synchronous host-to-device copy, which is unsupported in GLOBAL mode.
        cu_seqlens_q = torch.tensor(
            [0, *actual_seq_qlen],
            dtype=torch.int32,
            device=context_positions.device,
        )
        seqused_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqused_kv = torch.tensor(
            actual_seq_kvlen,
            dtype=torch.int32,
            device=context_positions.device,
        )
        # This bound flows through decode_inputs into a graph ATTR, so its
        # value must stay constant across steps: a per-step exact value would
        # recompile the captured graph every round. The exact per-step lengths
        # are carried by the seqused_kv tensor at runtime.
        max_seqlen_kv = self.model.max_cache_len
        fia_metadata = flash_attn_metadata(
            batch_size=local_batch,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            num_heads_q=self.model.layers[0].self_attn.total_num_heads,
            num_heads_kv=self.model.layers[0].self_attn.num_kv_heads,
            head_dim=self.model.layers[0].self_attn.head_dim,
            max_seqlen_q=self.next_n,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=0,
            win_left=-1,
            win_right=-1,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="TND",
        )
        sample_noise = None
        if self.temperature > 0:
            sample_noise = torch.empty(
                local_batch,
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
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seqused_q,
            "seqused_kv": seqused_kv,
            "fia_metadata": fia_metadata,
            "max_seqlen_kv": max_seqlen_kv,
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
        return decode_inputs


    def propose(
        self,
        input_dict: Dict,
        main_next_tokens: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> Dict:
        context_positions = input_dict["target_hidden_positions"]
        slot_block_table = input_dict["slot_block_table"]
        cache_data = input_dict["cache_data"]
        batch_size = target_hidden_states.shape[0]
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
            "confidence": None,
            "kv_len": cached_len,
            "kv_len_cached": cached_len,
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
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        def resolve(name: str) -> Optional[str]:
            return next(
                (candidate for candidate in self._weight_candidates(name) if candidate in params),
                None,
            )

        ignored_fragments = [
            "embed_tokens.weight",
            "lm_head.weight",
        ]
        if self.confidence_head is None:
            ignored_fragments.append("confidence_head.")
        for name, tensor in weights:
            if any(fragment in name for fragment in ignored_fragments):
                continue

            for fused_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, fused_name)
                param_name = resolve(mapped_name)
                if param_name is None:
                    continue
                param = params[param_name]
                param.weight_loader(param, tensor, shard_id)
                loaded.add(param_name)
                break
            else:
                param_name = resolve(name)
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
                # FRACTAL_NZ matmul does not support the confidence projection's
                # single output column. Keep its BF16 weight in ND format.
                module_is_nz = is_nz and module_name != "confidence_head.proj"
                quant_method.process_weights_after_loading(
                    module, is_nz=module_is_nz
                )

__all__ = [
    "K3DSparkAttention",
    "K3DSparkDecoderLayer",
    "K3DSparkConfidenceHead",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
]
