# coding=utf-8
# Unified Qwen dense model for Ascend NPU inference.
# Supports Qwen2/Qwen2.5/Qwen3 dense (non-MoE) variants via config-driven behavior:
#   - QK-Norm: enabled for Qwen3 (model_type=="qwen3"), disabled for Qwen2
#   - attention_bias: True for Qwen2, False for Qwen3 (read from config.attention_bias)
#
# Adapted from:
#   https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/qwen3/modeling_qwen3.py
#   https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/qwen2/modeling_qwen2.py
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

"""Unified PyTorch Qwen dense model adapted for Ascend NPU inference."""

import os
from typing import Optional, Tuple, Iterable

import torch
from torch import nn
import torch_npu
import torchair

import torch.distributed as dist

from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from executor.utils import init_comm_group, get_default_group
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_qwen import Qwen2Config, Qwen3Config

torchair.patch_for_hcom()


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def ln_npu(self, hidden_states):
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return result

    def forward(self, hidden_states, *args):
        if len(args) == 0:
            result = self.ln_npu(hidden_states)
            return result
        elif len(args) == 1 and args[0] is None:
            result = self.ln_npu(hidden_states)
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1:
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (y, x)
        else:
            raise NotImplementedError(
                f"insupportable QwenRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config, max_position_embeddings=2048, device=None):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(self, x, position_ids, max_seq_len=None):
        if position_ids.dim() != 1:
            raise RuntimeError("Qwen expects packed 1D position_ids.")
        if max_seq_len is None:
            kv_len = position_ids.max().item() + 1 if position_ids.numel() > 0 else 1
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class QwenMLP(nn.Module):
    """Dense MLP with SwiGLU activation, adapted for NPU with tensor parallelism."""

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.comm_manager = comm_manager

        from transformers.activations import ACT2FN

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size, self.intermediate_size],
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group"),
            quant_config=None,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group"),
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        intermediate_size_per_rank = gate_up.shape[-1] // 2
        gate = gate_up[..., :intermediate_size_per_rank]
        up = gate_up[..., intermediate_size_per_rank:]
        hidden_states = self.act_fn(gate) * up
        hidden_states = self.down_proj(hidden_states)
        if self.attn_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("attn_tp_group"))
        return hidden_states


class QwenAttention(nn.Module):
    """Multi-headed attention adapted for Ascend NPU, supports both Qwen2 and Qwen3."""

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_type = "FullAttention"
        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        self.block_size = infer_config.scheduler_config.block_size

        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads_per_rank = max(self.num_key_value_heads // self.attn_tp_size, 1)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attn_intermediate_size = self.head_dim * self.num_heads
        self.attn_intermediate_size_per_rank = self.attn_intermediate_size // self.attn_tp_size
        self.comm_manager = comm_manager

        # Config-driven: Qwen2 uses bias=True, Qwen3 uses bias=False
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=getattr(config, "attention_bias", False),
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group"),
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False,
        )

        # Config-driven: QK-Norm present when config.json has explicit head_dim (Qwen3)
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(
            self.attn_intermediate_size,
            config.hidden_size,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group"),
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.o_proj",
        )
        self.scale_fa = 1 / (self.head_dim ** 0.5)
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        cache_dtype = torch.bfloat16 if config.torch_dtype is None else config.torch_dtype
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
            ),
            CacheEntry(
                cache_name="v_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
            ),
        ]

    def exec_qkv(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor]] = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill if forward_metadata else False
        q_len = qkv.size(0)
        padded_q_len = q_len
        prompt_tokens = forward_metadata.prompt_tokens if forward_metadata else q_len

        if is_prefill:
            if prompt_tokens < padded_q_len:
                qkv = qkv[:prompt_tokens]
                q_len = prompt_tokens

        query_states, key_states, value_states = qkv.split(
            (
                self.num_heads_per_rank * self.head_dim,
                self.num_key_value_heads_per_rank * self.head_dim,
                self.num_key_value_heads_per_rank * self.head_dim,
            ),
            dim=1,
        )

        query_shape = (q_len, self.num_heads_per_rank, self.head_dim)
        key_value_shape = (q_len, self.num_key_value_heads_per_rank, self.head_dim)

        # Qwen3: apply QK-Norm before RoPE; Qwen2: direct reshape
        if self.use_qk_norm:
            query_states = self.q_norm(query_states.view(query_shape).contiguous())
            key_states = self.k_norm(key_states.view(key_value_shape).contiguous())
        else:
            query_states = query_states.view(query_shape)
            key_states = key_states.view(key_value_shape)
        value_states = value_states.view(key_value_shape)

        cos, sin = cos_sin
        query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(
            query_states, key_states, cos, sin, layout="TND"
        )

        attention_mask = forward_metadata.attention_mask

        k_cache, v_cache = self.k_cache, self.v_cache
        if not k_cache.numel() or not v_cache.numel():
            raise RuntimeError("A BUG: k_cache or v_cache are not initialized properly.")

        sparse_mode = 3
        fa_ops = torch.ops.npu
        if not forward_metadata.is_prefill and self.enable_gegraph:
            fa_ops = torchair.ops

        tmp_slot_mapping = slot_mapping.view(-1)
        torch_npu.npu_scatter_nd_update_(
            k_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            tmp_slot_mapping.view(-1, 1),
            key_states,
        )
        torch_npu.npu_scatter_nd_update_(
            v_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            tmp_slot_mapping.view(-1, 1),
            value_states,
        )

        key_states, value_states = k_cache, v_cache
        if not is_prefill and self.enable_npugraph_ex:
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
        else:
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q

        attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
            query_states,
            key_states.view(*key_states.shape[:2], -1),
            value_states.view(*value_states.shape[:2], -1),
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            softmax_scale=self.scale_fa,
            input_layout="TND",
            sparse_mode=sparse_mode,
            atten_mask=attention_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            block_size=self.block_size,
        )

        attn_output = attn_output.reshape(q_len, self.attn_intermediate_size_per_rank)
        attn_output = self.o_proj(attn_output)
        if is_prefill and q_len < padded_q_len:
            pad_len = padded_q_len - q_len
            attn_output = torch.cat([
                attn_output,
                torch.zeros(
                    (pad_len, attn_output.size(-1)),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                ),
            ], dim=0)
        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.merged_qkv_proj(hidden_states)
        output = self.exec_qkv(
            qkv=qkv,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            slot_mapping=slot_mapping[self.attn_type],
            block_table=block_table[self.attn_type],
        )
        return output


class QwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QwenAttention(
            config=config,
            infer_config=infer_config,
            comm_manager=comm_manager,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = QwenMLP(
            config=config,
            infer_config=infer_config,
            comm_manager=comm_manager,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            slot_mapping=slot_mapping,
            block_table=block_table,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (residual, hidden_states)
        return outputs


class QwenModel(nn.Module):
    """Transformer decoder consisting of config.num_hidden_layers layers."""

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        self.rank_id = int(os.getenv("LOCAL_RANK", "0"))
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.comm_manager = comm_manager

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=1,
            tp_rank=0,
        )
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    config, infer_config, comm_manager, layer_idx,
                    prefix=f"model.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = QwenRotaryEmbedding(config=config, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill if forward_metadata else False
        position_ids = position_ids.view(-1).long()

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        block_table = forward_metadata.block_table
        slot_mapping = forward_metadata.slot_mapping
        cos_sin = self.rotary_emb(hidden_states, position_ids, self.max_position_embeddings)
        residual = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                cos_sin=cos_sin,
                past_residual=residual,
                forward_metadata=forward_metadata,
                slot_mapping=slot_mapping,
                block_table=block_table,
                **kwargs,
            )
            residual, hidden_states = layer_outputs

        hidden_states, _ = self.norm(hidden_states, residual)
        cu_seq_lens_q = forward_metadata.actual_seq_lengths_cu_q if forward_metadata else None
        if cu_seq_lens_q is None:
            raise RuntimeError("actual_seq_lengths_cu_q is required.")

        if is_prefill:
            seq_index = cu_seq_lens_q - 1
            hidden_states = torch.index_select(hidden_states, 0, seq_index)
            hidden_states = hidden_states.view(seq_index.numel(), 1, hidden_states.size(-1))
        else:
            hidden_states = hidden_states.view(hidden_states.shape[0], 1, hidden_states.shape[-1])

        return hidden_states


class QwenForCausalLM(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.num_hidden_layers = config.num_hidden_layers
        self.config = config
        self.input_max_len = infer_config.data_config.input_truncated_len
        self.max_position_embeddings = config.max_position_embeddings
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size

        self.model = QwenModel(config, infer_config, comm_manager, prefix)

        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group")
            if self.lmhead_tp_size > 1 else 0,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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

        logits = self.lm_head(hidden_states)

        if self.lmhead_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lmhead_tp_size)]
            dist.all_gather(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.concat(new_logits, dim=-1)
        logits = logits.float()
        return logits

    def get_cache_info(
        self,
    ) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(layer.self_attn.cache_entries),
                )
            )

        return ModelCacheInfo(
            num_layers=len(layer_infos),
            block_size=self.infer_config.scheduler_config.block_size,
            layer_infos=layer_infos,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def process_weights_after_loading(self):
        is_nz = (self.infer_config.model_config.enable_weight_nz
                 and self.infer_config.model_config.exe_mode != "npugraph_ex")
        for module_name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module, is_nz=is_nz)
