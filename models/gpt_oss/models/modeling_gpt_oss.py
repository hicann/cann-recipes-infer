# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Callable, Iterable, Optional, Tuple, List
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch_npu
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_layers import GradientCheckpointingLayer, GenericForSequenceClassification
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    QKVParallelLinear
    )
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_gpt_oss import GptOssConfig


@use_kernel_forward_from_hub("RMSNorm")
class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GptOssRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama

    def forward(self, hidden_states, *args):
        if len(args) == 0: # only hidden_states exists
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return result
        elif len(args) == 1 and args[0] is None: # residual is None
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1: # residual is not None
            residual = args[0]
            result, _, r = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (result, r)
        else:
            raise NotImplementedError(
                f"insupportable GptOssRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GptOssExperts(nn.Module):
    def __init__(self, config, runner_settings):
        super().__init__()
        self.tp_size = runner_settings.get("parallel_config").get("tp_size", 1)
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.expert_dim_per_rank = self.expert_dim // self.tp_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim_per_rank),
            requires_grad=False
            )
        self.gate_up_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.expert_dim_per_rank, dtype=torch.float),
            requires_grad=False
            )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim_per_rank, self.hidden_size),
            requires_grad=False
            )
        self.down_proj_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.float),
            requires_grad=False
            )
        self.alpha = 1.702
        self.limit = 7.0
        if self.tp_size > 1:
            self.commute_type = "all_reduce"
        else:
            self.commute_type = "no_commute"

    def forward(self, hidden_states: torch.Tensor, expert_tokens) -> torch.Tensor:      
        mm1_mm3 = torch_npu.npu_grouped_matmul([hidden_states], [self.gate_up_proj],
            group_list=expert_tokens, split_item=3, group_type=0, bias=[self.gate_up_proj_bias])[0]
        #swiglu 
        gate, up = mm1_mm3[..., ::2], mm1_mm3[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch_npu.npu_grouped_matmul(
            [((up + 1) * glu)], [self.down_proj],
            group_list=expert_tokens, group_type=0, split_item=3, bias=[self.down_proj_bias])[0]
        if self.commute_type == "all_reduce":
            dist.all_reduce(next_states)
        return next_states


@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLP(nn.Module):
    def __init__(self, config, runner_settings, prefix):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.router = ColumnParallelLinear(
            input_size=self.hidden_dim,
            output_size=self.num_experts,
            bias=True,
            tp_size=1,
            tp_rank=0,
            prefix=f"{prefix}.router"
        )
        self.experts = GptOssExperts(config, runner_settings)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        router_scores, router_indices, row_idx = self._forward_router(hidden_states)  # (num_experts, seq_len)
        hidden_states_ordered_by_experts, routing_weights, expanded_row_idx, expert_idx \
                = self.moe_infer_fusion(hidden_states, router_indices, router_scores, row_idx)
        hidden_states = torch_npu.npu_moe_finalize_routing(
                hidden_states_ordered_by_experts, 
                skip1=None, skip2=None,
                bias=None,
                scales=routing_weights,
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=expert_idx
            )
        y = hidden_states.view(batch_size, -1, self.hidden_dim)
        return y

    def _forward_router(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)  # (seq_len, num_experts)
        topk_weight, topk_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(router_logits, None, k=self.top_k)
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
        return topk_weight, topk_idx, row_idx

    def moe_infer_fusion(self, x, topk_ids, topk_weight, row_idx):
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)

        routing_weights = topk_weight.to(x.dtype)
        expert_idx = topk_ids.int()
        if row_idx is None:
            # decode
            if sequence_length == 1: 
                row_idx = self.row_idx_decode
            else:
                row_idx_prefill_len = batch_size * sequence_length * self.top_k
                row_idx_prefill = torch.arange(
                    0, row_idx_prefill_len, dtype=torch.int32,
                    device=topk_weight.device).view(self.top_k, -1).permute(1, 0).contiguous()
                row_idx = row_idx_prefill

        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=expert_idx,
            active_num=batch_size * sequence_length
        )
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, self.num_experts)
        expert_tokens = expert_tokens.to(torch.int64)
        hidden_states_ordered_by_experts = self.experts(expanded_x, expert_tokens=expert_tokens)
        return hidden_states_ordered_by_experts, routing_weights, expanded_row_idx, expert_idx


class GptOssRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: GptOssConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(x.dtype), sin.to(x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_prefill: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if is_prefill:
            causal_mask = attention_mask[:, :, :attn_weights.shape[-2], :attn_weights.shape[-1]]
        else:
            causal_mask = attention_mask[:, :, attn_weights.shape[-1] - 1, :attn_weights.shape[-1]].unsqueeze(2)
        attn_weights = attn_weights + causal_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class GptOssAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptOssConfig, runner_settings, layer_idx: int, prefix):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = runner_settings.get("parallel_config").get("tp_size", 1)
        if self.tp_size > 1:
            self.commute_type = "all_reduce"
        else:
            self.commute_type = "no_commute"
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads_per_rank = max(config.num_key_value_heads // self.tp_size, 1)
        self.num_attention_heads_per_rank = config.num_attention_heads // self.tp_size
        self.num_key_value_groups = self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=True,
            tp_size=self.tp_size,
            tp_rank=dist.get_rank() if self.tp_size > 1 else 0,
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            tp_size=self.tp_size,
            tp_rank=dist.get_rank() if self.tp_size > 1 else 0,
            bias=True,
            input_is_parallel=True,
            prefix=f"{prefix}.o_proj"
        )
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads_per_rank))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        is_prefill: bool,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = self.merged_qkv_proj(hidden_states)
        bsz, q_len, _ = qkv.size()
        query_states, key_states, value_states = qkv.split((
            self.num_attention_heads_per_rank * self.head_dim,\
            self.num_key_value_heads_per_rank * self.head_dim,\
            self.num_key_value_heads_per_rank * self.head_dim), dim=2)
        input_shape = hidden_states.shape[:-1]
        # (bz, q_len, -1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim) 

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            past_key = past_key_values[self.layer_idx][0]
            past_value = past_key_values[self.layer_idx][1]
            torch_npu.scatter_update_(past_key, kv_len, key_states, -2)
            torch_npu.scatter_update_(past_value, kv_len, value_states, -2)
        if q_len == 1:
            key_states, value_states = past_key_values[self.layer_idx]
            key_states = key_states[:, :, :kv_len + 1, :]
            value_states = value_states[:, :, :kv_len + 1, :]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            is_prefill,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.commute_type == "all_reduce":
            attn_output = self.o_proj(attn_output)
            dist.all_reduce(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GptOssDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GptOssConfig, runner_settings, layer_idx: int, prefix):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, runner_settings, layer_idx, prefix)
        self.mlp = GptOssMLP(config, runner_settings, prefix)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        is_prefill: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        past_residual: Optional[torch.Tensor] = None, 
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.FloatTensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            kv_len=kv_len,
            is_prefill=is_prefill,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return (residual, hidden_states)


@auto_docstring
class GptOssPreTrainedModel(PreTrainedModel):
    config: GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        # "router_logits": OutputRecorder(GptOssTopKRouter, index=0),
        "hidden_states": GptOssDecoderLayer,
        "attentions": GptOssAttention,
    }
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
    _supports_flash_attention = False
    _supports_flex_attention = False

    def _init_weights(self, module):
        pass


@auto_docstring
class GptOssModel(GptOssPreTrainedModel):
    _no_split_modules = ["GptOssDecoderLayer"]

    def __init__(self, config: GptOssConfig, runner_settings, prefix: str = "",):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=self.padding_idx,
            params_dtype=torch.bfloat16,
            tp_size=1,
            tp_rank=0,
        )
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, runner_settings, layer_idx, prefix)\
                for layer_idx in range(config.num_hidden_layers)])
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        causal_mask_mapping: Optional[dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: Optional[bool] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        #（bz, seq_len, hidden_d)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        residual = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                is_prefill=is_prefill,
                past_key_values=past_key_values,
                kv_len=kv_len,
                position_embeddings=position_embeddings,
                past_residual=residual,
                **kwargs,
            )
            residual, hidden_states = layer_outputs

        hidden_states, _ = self.norm(hidden_states, residual)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states)


@auto_docstring
class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, runner_settings, prefix: str = "",):
        super().__init__(config)
        self.runner_settings = runner_settings
        self.model = GptOssModel(config, runner_settings, prefix)
        self.vocab_size = config.vocab_size
        self.lm_head_tp_size = runner_settings.get("parallel_config").get("tp_size", 1)
        self.attn_tp_size = runner_settings.get("parallel_config").get("tp_size", 1)
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.vocab_size,
            bias=False,
            tp_size=self.lm_head_tp_size,
            tp_rank=dist.get_rank() if self.lm_head_tp_size > 1 else 0
        )
        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        is_prefill: Optional[bool] = None,
        causal_mask_mapping: Optional[dict] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            causal_mask_mapping=causal_mask_mapping,
            kv_len=kv_len,
            is_prefill=is_prefill,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states)
        if self.lm_head_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lm_head_tp_size)]
            dist.all_gather(new_logits, logits)
            logits = torch.concat(new_logits, dim=-1)
        return logits

    # Adapted from vllm.model_executor.models.gpt_oss.GptOssModel._load_weights_other
    # (https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_oss.py)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
        ]

        tp_rank = dist.get_rank() if self.lm_head_tp_size > 1 else 0
        tp_size = self.lm_head_tp_size

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        intermediate_size = self.config.intermediate_size
        per_rank_intermediate_size = intermediate_size // tp_size
        # Calculate common slicing bounds for current rank
        tp_rank_start = tp_rank * per_rank_intermediate_size
        tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

        for name, weight in weights:
            if name.endswith(".gate_up_proj"):
                # Handle MLP gate and up projection weights
                # Extract gate and up projection parts               
                narrow_weight = weight[:, :, 2 * tp_rank_start:2 * tp_rank_end]
                narrow_weight = narrow_weight.contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith(".down_proj"):
                # Handle MLP down projection weights
                narrow_weight = weight[:, tp_rank_start:tp_rank_end, :]
                narrow_weight = narrow_weight.contiguous()
                param = params_dict[name]

                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith(".gate_up_proj_bias"):
                # Handle MLP gate and up projection biases
                # Extract gate and up projection bias parts
                narrow_weight = weight[:, 2 * tp_rank_start:2 * tp_rank_end]

                param = params_dict[name]
                param.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith(".down_proj_bias"):
                # Handle MLP down projection bias
                # only load on rank 0 to avoid duplication during the all-reduce phase.
                if tp_rank != 0:
                    weight.zero_()
                param = params_dict[name]
                param.copy_(weight)
                loaded_params.add(name)
                continue
            elif "sinks" in name:
                # Handle attention sinks (distributed across ranks)
                param = params_dict[name]
                narrow_weight = weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, weight)
                else:
                    weight_loader(param, weight, shard_id)
                break
            else:
                # Handle all other weights with potential renaming
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, weight)
            loaded_params.add(name)
        return loaded_params

    def init_cache(
        self,
        input_ids,
    ):
        batch_size, seq_len = input_ids.size()
        cache_seq_len = self.runner_settings.get("data_config").get("max_position_embeddings", 4096)
        past_key_value = ()
        cache_shape = (
            batch_size,
            max(self.config.num_key_value_heads // self.attn_tp_size, 1),
            cache_seq_len,
            self.config.head_dim
        )
        dtype = self.config.torch_dtype
        for _ in range(self.config.num_hidden_layers):
            key_cache = torch.zeros(cache_shape, dtype=dtype, device=input_ids.device)
            value_cache = torch.zeros(cache_shape, dtype=dtype, device=input_ids.device)
            past_key_value += ((key_cache, value_cache),)
        return past_key_value

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        kv_len=None,
        cache_position=None,
        is_prefill=None,
        causal_mask_mapping=None,
        generate_ids=None,
        curr_position=None,
        **kwargs
    ):
        model_inputs = {}
        if past_key_values is None:
            past_key_values = self.init_cache(input_ids)
        
        if is_prefill:
            if cache_position is None:
                past_seen_tokens = 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
                )
        else:
            input_ids = generate_ids[:, -1:]
            cache_position = torch.tensor([curr_position], dtype=torch.long, device=input_ids.device)
            position_ids = kv_len.unsqueeze(1)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        if is_prefill:
            position_ids = model_inputs.get("position_ids")
            kv_len = torch.zeros((position_ids.size()[0]), dtype=torch.int32, device=input_ids.device)

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "causal_mask_mapping": causal_mask_mapping,
                "kv_len": kv_len,
                "is_prefill": is_prefill
            }
        )
        return model_inputs


class GptOssForSequenceClassification(GenericForSequenceClassification, GptOssPreTrainedModel):
    pass


__all__ = ["GptOssForCausalLM", "GptOssForSequenceClassification", "GptOssModel", "GptOssPreTrainedModel"]