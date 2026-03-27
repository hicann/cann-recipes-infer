# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from executor.utils.forward_metadata import ForwardMetaData
from executor.core.config import InferenceConfig, CommManager
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    QKVParallelLinear
    )
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_gpt_oss import GptOssConfig


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
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
    def __init__(self, config, infer_config: InferenceConfig):
        super().__init__()
        self.tp_size = infer_config.parallel_config.moe_tp_size
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


class GptOssMLP(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, prefix=""):
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
        self.experts = GptOssExperts(config, infer_config)

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
        hidden_states = hidden_states.view(-1, self.hidden_dim)
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


def npu_apply_rotary_pos_emb(q, k, cos, sin, layout):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    cos = torch.concat((cos, cos), -1)
    sin = torch.concat((sin, sin), -1)

    q, k = torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout)
    return q, k


class GptOssAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptOssConfig, infer_config: InferenceConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = infer_config.parallel_config.attn_tp_size
        if self.tp_size > 1:
            self.commute_type = "all_reduce"
        else:
            self.commute_type = "no_commute"
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads_per_rank = max(config.num_key_value_heads // self.tp_size, 1)
        self.num_attention_heads_per_rank = config.num_attention_heads // self.tp_size
        self.num_key_value_groups = self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=True,
            tp_size=self.tp_size,
            tp_rank=dist.get_rank() if self.tp_size > 1 else 0,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
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
        self.k_cache = self.v_cache = torch.Tensor([])  # will be set in runner's init_kvcache
        self.cache_unit = (self.num_key_value_heads_per_rank, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(hidden_states)
        bsz, q_len, _ = qkv.size()
        query_states, key_states, value_states = qkv.split((
            self.num_attention_heads_per_rank * self.head_dim,\
            self.num_key_value_heads_per_rank * self.head_dim,\
            self.num_key_value_heads_per_rank * self.head_dim), dim=2)
        input_shape = hidden_states.shape[:-1]
        # (bz, q_len, -1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, layout="BSND")

        kv_len = forward_metadata.kv_len
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        attention_mask = forward_metadata.attention_mask

        k_cache, v_cache = self.k_cache, self.v_cache
        if not (k_cache.numel() and v_cache.numel()):
            raise RuntimeError("A BUG: k_cache and v_cache are not initialized properly.")
        torch_npu.scatter_update_(k_cache, kv_len, key_states, 1)
        torch_npu.scatter_update_(v_cache, kv_len, value_states, 1)
        if q_len == 1:
            key_states, value_states = k_cache, v_cache
            key_states = key_states[:, :kv_len + 1, :, :]
            value_states = value_states[:, :kv_len + 1, :, :]

        actual_seq_qlen = [query_states.shape[1]]

        query_states = query_states.reshape(-1, self.num_attention_heads_per_rank, self.head_dim)
        key_states = key_states.reshape(-1, self.num_key_value_heads_per_rank, self.head_dim)
        value_states = value_states.reshape(-1, self.num_key_value_heads_per_rank, self.head_dim)

        attn_output, attn_weights = torch_npu.npu_fused_infer_attention_score_v2(
            query_states,
            key_states,
            value_states,
            num_query_heads=self.num_attention_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            input_layout="TND",
            softmax_scale=self.scaling,
            sparse_mode=4 if self.sliding_window else 3,
            pre_tokens=self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max,
            next_tokens=0,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_lengths_kv,
            atten_mask=attention_mask,
            learnable_sink=self.sinks
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.commute_type == "all_reduce":
            dist.all_reduce(attn_output)
        return attn_output, attn_weights


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, infer_config: InferenceConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, infer_config, layer_idx, prefix)
        self.mlp = GptOssMLP(config, infer_config, prefix)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_residual: Optional[torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_metadata=forward_metadata,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return (residual, hidden_states)


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig, infer_config: InferenceConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
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
            [GptOssDecoderLayer(config, infer_config, layer_idx, prefix)\
                for layer_idx in range(config.num_hidden_layers)])
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # (bz, seq_len, hidden_d)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        residual = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_residual=residual,
                forward_metadata=forward_metadata,
                **kwargs,
            )
            residual, hidden_states = layer_outputs

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    def __init__(self, config: GptOssConfig, infer_config: InferenceConfig, comm_manager: CommManager = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.model = GptOssModel(config, infer_config, prefix)
        self.vocab_size = config.vocab_size
        self.lm_head_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.vocab_size,
            bias=False,
            tp_size=self.lm_head_tp_size,
            tp_rank=dist.get_rank() if self.lm_head_tp_size > 1 else 0
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            **kwargs,
        )
        bs, q_len, hidden_size = hidden_states.shape
        if forward_metadata.is_prefill:
            bs = position_ids.shape[0]
            gather_index, _ = torch.max(position_ids, dim=-1)
            seq_index = ((gather_index + 1).to(torch.int32).cumsum(-1) - 1).npu()
            hidden_states = \
                (torch.index_select(hidden_states.view(1, -1, hidden_size), 1, seq_index.view(-1))).view(bs, 1, -1)
            q_len = 1 # prefill takes th last token
        else: # combine bs and q_len axes for lm_head
            hidden_states = hidden_states.view(bs * q_len, 1, hidden_size)

        logits = self.lm_head(hidden_states)

        if self.lm_head_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lm_head_tp_size)]
            dist.all_gather(new_logits, logits)
            logits = torch.concat(new_logits, dim=-1)

        return logits.reshape(bs, q_len, -1).float()

    # Adapted from vllm.model_executor.models.gpt_oss.GptOssModel._load_weights_other
    # (https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_oss.py)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
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

    def process_weights_after_loading(self):
        for _, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module, is_nz=False)


__all__ = ["GptOssForCausalLM", "GptOssModel"]
