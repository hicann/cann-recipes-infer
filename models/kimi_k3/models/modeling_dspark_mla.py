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

"""Pure-BF16 Kimi K3 MLA DSpark proposal model.

The target model supplies the configured intermediate hidden states. The draft
model projects their concatenation into a compact context, evaluates a fixed
proposal block with owner-local MLA layers, then applies a token-conditioned
Markov bias before sampling the speculative tokens.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import cann_ops_transformer.ops
import torch
import torch_npu
from torch import nn

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
    build_dspark_mla_slot_mapping,
    reduce_scatter_first_dim,
)
from module.linear import ReplicatedLinear

InferenceConfig = object
CommManager = object


class K3DSparkAttention(nn.Module):
    """Owner-local non-causal MLA over committed context and one draft block.

    Each attention-owner rank keeps the complete 64-head query projection, but
    the paged cache stores one shared compressed KV head: a 512-wide normalized
    latent plus a 64-wide RoPE key.  ``kv_b_proj`` is absorbed into the query
    and the post-attention value projection during decode.
    """

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
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_width = self.kv_lora_rank + self.qk_rope_head_dim
        self.block_size = int(infer_config.scheduler_config.block_size)
        self.softmax_scale = self.q_head_dim ** -0.5
        rope = config.rope_parameters or {}
        if rope.get("rope_type", "default") == "yarn":
            factor = float(rope.get("factor", 1.0))
            mscale_all_dim = float(rope.get("mscale_all_dim", 0.0))
            if factor > 1.0 and mscale_all_dim:
                mscale = 0.1 * mscale_all_dim * math.log(factor) + 1.0
                self.softmax_scale *= mscale * mscale
        common = dict(
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
        )
        # Cache ownership follows requests, so every owner rank computes all
        # query heads rather than splitting them across the DSpark TP group.
        # Keep the checkpoint projections separate: MLA Prolog consumes their
        # weights directly and fuses both matmuls with norm, RoPE and cache IO.
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            prefix=f"{prefix}.q_a_proj",
            **common,
        )
        self.q_a_layernorm = K3DSparkRMSNorm(
            self.q_lora_rank, config.rms_norm_eps
        )
        self.q_b_proj = ReplicatedLinear(
            self.q_lora_rank,
            self.total_num_heads * self.q_head_dim,
            prefix=f"{prefix}.q_b_proj",
            **common,
        )
        self.kv_a_layernorm = K3DSparkRMSNorm(
            self.kv_lora_rank, config.rms_norm_eps
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_width,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
            **common,
        )
        # The module owns the checkpoint parameter.  Its raw [out, in] weight
        # is split after loading into the two absorbed batch-matmul weights.
        self.kv_b_proj = ReplicatedLinear(
            self.kv_lora_rank,
            self.total_num_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            prefix=f"{prefix}.kv_b_proj",
            **common,
        )
        self.kv_b_proj_w_k = None
        self.kv_b_proj_w_v = None
        self.weight_dq_prolog = None
        self.weight_uq_qr_prolog = None
        self.weight_dkv_kr_prolog = None
        self.o_proj = ReplicatedLinear(
            self.total_num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.attn_type = "FullAttention"

    def prepare_prolog_weights(self) -> None:
        """Match the main model's one-time MLA Prolog weight preparation."""

        def as_input_output(weight, input_size, output_size):
            weight = weight.data.contiguous()
            if tuple(weight.shape) == (output_size, input_size):
                weight = weight.transpose(0, 1).contiguous()
            if tuple(weight.shape) != (input_size, output_size):
                raise RuntimeError(
                    f"invalid DSpark MLA Prolog weight shape {tuple(weight.shape)}, "
                    f"expected {(input_size, output_size)}"
                )
            return weight

        def as_fractal_nz(weight):
            weight = weight.data
            fmt = torch_npu.get_npu_format(weight)
            fmt_id = fmt.value if hasattr(fmt, "value") else int(fmt)
            if fmt_id != 29:
                weight = torch_npu.npu_format_cast(weight.contiguous(), 29)
            return weight

        weight_dq = as_input_output(
            self.q_a_proj.weight, self.hidden_size, self.q_lora_rank
        )
        weight_dkv_kr = as_input_output(
            self.kv_a_proj_with_mqa.weight, self.hidden_size, self.kv_width
        )
        self.weight_dq_prolog = nn.Parameter(
            as_fractal_nz(weight_dq), requires_grad=False
        )
        self.weight_uq_qr_prolog = nn.Parameter(
            as_fractal_nz(self.q_b_proj.weight), requires_grad=False
        )
        self.weight_dkv_kr_prolog = nn.Parameter(
            as_fractal_nz(weight_dkv_kr), requires_grad=False
        )

    def _mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        slots: torch.Tensor,
        layer_cache: Dict[str, torch.Tensor],
    ):
        """Fuse MLA projections, norms, RoPE, absorb and PA_NZ cache update."""
        if (
            self.weight_dq_prolog is None
            or self.weight_uq_qr_prolog is None
            or self.weight_dkv_kr_prolog is None
        ):
            raise RuntimeError("DSpark MLA Prolog weights are not initialized")
        if self.kv_b_proj_w_k is None:
            raise RuntimeError("DSpark MLA absorbed KV-B weights are not initialized")

        token_x = hidden_states.reshape(-1, self.hidden_size).contiguous()
        tokens = token_x.shape[0]
        if slots.numel() != tokens:
            raise ValueError(
                f"DSpark MLA Prolog has {tokens} tokens but {slots.numel()} slots"
            )
        cos, sin = cos_sin
        if (
            cos.numel() != tokens * self.qk_rope_head_dim
            or sin.numel() != cos.numel()
        ):
            raise ValueError("DSpark MLA Prolog RoPE tensors do not match token count")

        # The first nine positional inputs and the merged PA_NZ repository
        # follow KimiMLAAttention._mla_prolog.  DSpark additionally supplies
        # RoPE tensors; Kimi K3 intentionally omits them for its NoPE path.
        return torch.ops.cann_ops_transformer.mla_prolog(
            token_x,
            self.weight_dq_prolog,
            self.weight_uq_qr_prolog,
            self.kv_b_proj_w_k,
            self.weight_dkv_kr_prolog,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            layer_cache["kv_cache"],
            layer_cache["kr_cache"],
            rope_sin=sin.reshape(tokens, self.qk_rope_head_dim).contiguous(),
            rope_cos=cos.reshape(tokens, self.qk_rope_head_dim).contiguous(),
            cache_index=slots.reshape(-1).to(torch.int64),
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            query_norm_flag=True,
            weight_quant_mode=0,
            ckvkr_repo_mode=1,
        )

    @staticmethod
    def _pa_nz_cache_view(kv_cache: torch.Tensor) -> torch.Tensor:
        """View logical PA storage with Flash MLA's physical PA_NZ layout."""
        cache_nz_dim = 16
        block_num, block_size, num_kv_heads, cache_dim = kv_cache.shape
        return kv_cache.view(
            block_num,
            num_kv_heads,
            cache_dim // cache_nz_dim,
            block_size,
            cache_nz_dim,
        )

    def _flash_mla_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_len: int,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the same cann_ops_transformer Flash MLA ABI as the main model."""
        if metadata is None:
            metadata = (
                torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
                    cache_seqlens,
                    num_heads_q=query.shape[1],
                    num_heads_kv=1,
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    max_seqlen_q=q_len,
                    max_seqlen_kv=-1,
                    head_dim_qk=self.kv_width,
                    head_dim_v=self.kv_lora_rank,
                    mask_mode=0,
                    layout_q="TND",
                )
            )
        output, _ = torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            query.contiguous(),
            self._pa_nz_cache_view(kv_cache),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            attn_mask=None,
            metadata=metadata,
            head_dim_v=self.kv_lora_rank,
            softmax_scale=self.softmax_scale,
            mask_mode=0,
            max_seqlen_q=q_len,
            max_seqlen_kv=-1,
            layout_q="TND",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=False,
        )
        return output

    def prefill_context_cache(
        self,
        context_states: torch.Tensor,
        context_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Dict[str, torch.Tensor],
        layer_cache: Dict[str, torch.Tensor],
    ) -> None:
        self._mla_prolog(
            context_states,
            context_cos_sin,
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
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        flash_mla_metadata: torch.Tensor,
        layer_cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        local_batch, draft_len = hidden_states.shape[:2]
        # Keep these calls ordered. Padding context slots intentionally target
        # scratch entries that the following draft update overwrites.
        self._mla_prolog(
            context_states,
            context_cos_sin,
            attn_metadata["context_slot_mapping"],
            layer_cache,
        )
        query, *_ = self._mla_prolog(
            hidden_states,
            draft_cos_sin,
            attn_metadata["draft_slot_mapping"],
            layer_cache,
        )
        attn_output = self._flash_mla_attention(
            query,
            layer_cache["kv_cache"],
            attn_metadata["block_table"],
            cache_seqlens,
            cu_seqlens_q,
            draft_len,
            flash_mla_metadata,
        )
        if self.kv_b_proj_w_v is None:
            raise RuntimeError("DSpark MLA absorbed KV-B weights are not initialized")
        attn_output = attn_output.to(device=self.kv_b_proj_w_v.device)
        output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).reshape(
            local_batch,
            draft_len,
            self.total_num_heads * self.v_head_dim,
        )
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
        attn_metadata: Dict[str, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        flash_mla_metadata: torch.Tensor,
        layer_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states,
            context_states,
            context_cos_sin,
            draft_cos_sin,
            attn_metadata,
            cu_seqlens_q,
            cache_seqlens,
            flash_mla_metadata,
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
        self.context_proj = ReplicatedLinear(
            context_width,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=f"{prefix}.context_proj",
        )
        self.context_norm = K3DSparkRMSNorm(
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
        self.final_norm = K3DSparkRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
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
        cache_seqlens: torch.Tensor,
        flash_mla_metadata: torch.Tensor,
        cache_data: Tuple[Dict[str, torch.Tensor], ...],
    ) -> torch.Tensor:
        attn_metadata = {
            "context_slot_mapping": context_slot_mapping,
            "draft_slot_mapping": draft_slot_mapping,
            "block_table": block_table,
        }
        context_states, _ = self.context_norm(
            self.context_proj(target_hidden_states)
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
                cu_seqlens_q,
                cache_seqlens,
                flash_mla_metadata,
                layer_cache,
            )
        hidden_states, _ = self.final_norm(hidden_states, residual)
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
            "context_slot_mapping": build_dspark_mla_slot_mapping(
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
        context_states, _ = self.model.context_norm(
            self.model.context_proj(target_hidden_states)
        )
        return context_states


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
        decode_batch_size = (
            self.infer_config.scheduler_config.batch_size_per_dp_rank
        )
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
        context_slot_mapping = build_dspark_mla_slot_mapping(
            context_positions,
            slot_block_table,
            self.block_size,
        )
        draft_slot_mapping = build_dspark_mla_slot_mapping(
            draft_positions,
            slot_block_table,
            self.block_size,
        )
        stable_next_tokens = main_next_tokens[:, :1]
        cu_seqlens_q = torch.arange(
            local_batch + 1,
            dtype=torch.int32,
            device=context_positions.device,
        ) * self.next_n
        cache_seqlens = (context_lengths + self.next_n).to(torch.int32)
        # Build the operator metadata before graph capture, matching the main
        # model's flash_mla_with_kvcache call contract. DSpark attention stays
        # non-causal, so mask_mode remains zero for the whole proposal block.
        flash_mla_metadata = (
            torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
                cache_seqlens,
                num_heads_q=self.config.num_attention_heads,
                num_heads_kv=1,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=None,
                max_seqlen_q=self.next_n,
                max_seqlen_kv=-1,
                head_dim_qk=self.config.kv_lora_rank
                + self.config.qk_rope_head_dim,
                head_dim_v=self.config.kv_lora_rank,
                mask_mode=0,
                layout_q="TND",
            )
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
            "cache_seqlens": cache_seqlens,
            "flash_mla_metadata": flash_mla_metadata,
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


    def check_model_settings(self) -> None:
        if self.config.num_hidden_layers <= 0:
            raise RuntimeError("DSpark requires a positive layer count")
        if bool(getattr(self.config, "attention_bias", False)):
            raise RuntimeError("DSpark supports bias-free attention only")
        if bool(getattr(self.config, "use_sliding_window", False)) or getattr(
            self.config, "sliding_window", None
        ) is not None:
            raise RuntimeError("Kimi K3 DSpark requires full attention")
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
        cache_seqlens: torch.Tensor,
        flash_mla_metadata: torch.Tensor,
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
            cache_seqlens,
            flash_mla_metadata,
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
            logits[:, step].add_(markov_bias.float())
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
                "confidence": None,
                "kv_len": cached_len,
                "kv_len_cached": cached_len,
            }

        decode_inputs = self.prepare_decode_inputs(
            input_dict, main_next_tokens, target_hidden_states
        )
        return self.run_decode_proposal(decode_inputs)

    @staticmethod
    def _weight_candidates(name: str) -> Tuple[str, ...]:
        candidates = [name]
        for source_prefix in ("draft_model.", "model.draft_model."):
            if name.startswith(source_prefix):
                candidates.append(name[len(source_prefix):])
        expanded = list(candidates)
        for candidate in candidates:
            if candidate.startswith(
                ("context_proj.", "context_norm.", "layers.", "final_norm.")
            ):
                expanded.append(f"model.{candidate}")
        return tuple(dict.fromkeys(expanded))

    def should_load_weight(self, name: str) -> bool:
        """Avoid materializing frozen/shared or training-only checkpoint tensors."""
        ignored_fragments = [
            "embed_tokens.weight",
            "lm_head.weight",
        ]
        if self.confidence_head is None:
            ignored_fragments.append("confidence_head.")
        return not any(fragment in name for fragment in ignored_fragments)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        loaded_shards: set[Tuple[str, int]] = set()
        unexpected: list[str] = []

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

            handled_stacked = False
            for fused_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                handled_stacked = True
                mapped_name = name.replace(weight_name, fused_name)
                param_name = resolve(mapped_name)
                if param_name is None:
                    unexpected.append(name)
                    break
                shard_key = (param_name, shard_id)
                if shard_key in loaded_shards:
                    raise RuntimeError(
                        f"duplicate DSpark checkpoint shard {shard_id} for {param_name}"
                    )
                param = params[param_name]
                param.weight_loader(param, tensor, shard_id)
                loaded.add(param_name)
                loaded_shards.add(shard_key)
                break
            if handled_stacked:
                continue

            param_name = resolve(name)
            if param_name is None:
                unexpected.append(name)
                continue
            if param_name in loaded:
                raise RuntimeError(f"duplicate DSpark checkpoint tensor for {param_name}")
            param = params[param_name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if loader is default_weight_loader and param.size() != tensor.size():
                raise RuntimeError(
                    f"DSpark checkpoint tensor {name} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(param.shape)} for {param_name}"
                )
            loader(param, tensor)
            loaded.add(param_name)

        for param_name, param in params.items():
            module_name = param_name.rsplit(".", 1)[0]
            if module_name.endswith("gate_up_proj"):
                missing_shards = {
                    shard_id
                    for shard_id in (0, 1)
                    if (param_name, shard_id) not in loaded_shards
                }
                if missing_shards:
                    raise RuntimeError(
                        f"DSpark checkpoint did not load shards "
                        f"{sorted(missing_shards)} for {param_name}"
                    )

        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} unexpected DSpark checkpoint tensors, "
                f"starting with {unexpected[:8]}"
            )

        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} K3 DSpark MLA parameters were not loaded, "
                f"starting with {missing[:8]}"
            )
        return loaded

    def process_weights_after_loading(self) -> None:
        self._split_kv_b_proj()
        is_nz = self.infer_config.model_config.enable_weight_nz
        for module_name, module in self.named_modules():
            # Shared target boundaries were processed by the main runner.
            # kv_b_proj stays in checkpoint [out, in] layout because its two
            # absorbed weights were derived above and the original linear is
            # not executed by the MLA decode path.
            if (
                module_name in ("lm_head", "model.embed_tokens")
                or "kv_b_proj" in module_name
            ):
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
        for layer in self.model.layers:
            layer.self_attn.prepare_prolog_weights()

    def _split_kv_b_proj(self) -> None:
        """Build the K-absorb and V-up weights from each raw KV-B tensor."""
        for layer in self.model.layers:
            attn = layer.self_attn
            expected_shape = (
                attn.total_num_heads
                * (attn.qk_nope_head_dim + attn.v_head_dim),
                attn.kv_lora_rank,
            )
            if tuple(attn.kv_b_proj.weight.shape) != expected_shape:
                raise RuntimeError(
                    "DSpark KV-B weight has shape "
                    f"{tuple(attn.kv_b_proj.weight.shape)}, expected "
                    f"{expected_shape}"
                )
            weight = attn.kv_b_proj.weight.T.view(
                attn.kv_lora_rank,
                attn.total_num_heads,
                attn.qk_nope_head_dim + attn.v_head_dim,
            )
            w_k, w_v = weight.split(
                [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1
            )
            attn.kv_b_proj_w_k = nn.Parameter(
                w_k.permute(1, 2, 0).contiguous(), requires_grad=False
            )
            attn.kv_b_proj_w_v = nn.Parameter(
                w_v.transpose(0, 1).contiguous(), requires_grad=False
            )

__all__ = [
    "K3DSparkAttention",
    "K3DSparkDecoderLayer",
    "K3DSparkConfidenceHead",
    "K3DSparkForCausalLM",
    "K3DSparkModel",
]
