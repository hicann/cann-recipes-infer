# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

""" PyTorch Qwen3_MOE model."""
import os
import math
from typing import List, Optional, Tuple, Union, Iterable

import torch
import torch.nn.functional as F
from torch import nn
import torch_npu
import torchair

import torch.distributed as dist

from module.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from executor.utils import init_comm_group, get_default_group
from executor.utils.forward_metadata import ForwardMetaData, get_forward_metadata
from executor.core.config import InferenceConfig, CommManager
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_qwen3_moe import Qwen3MoeConfig

torchair.patch_for_hcom()


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def ln(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def ln_npu(self, hidden_states):
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return result

    def forward(self, hidden_states, *args):
        if len(args) == 0: # only hidden_states exists
            result = self.ln_npu(hidden_states)
            return result
        elif len(args) == 1 and args[0] is None: # residual is None
            result = self.ln_npu(hidden_states)
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1: # residual is not None
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (y, x)
        else:
            raise NotImplementedError(
                f"insupportable Qwen3MoeRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config, max_position_embeddings=2048, device=None):
        super().__init__()
        self.config = config
        self.dim = self.config.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = self.config.rope_theta
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(self, x, seq_len, kv_len, max_seq_len=None):
        # x shape is [bs, num_attention_heads, seq_len, head_size]
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        batch_size, _, _ = x.size()
        if seq_len == 1:
            # BD -> BNSD
            cos = torch.index_select(self.cos_cached, dim=0, index=kv_len).unsqueeze(1).unsqueeze(1)
            sin = torch.index_select(self.sin_cached, dim=0, index=kv_len).unsqueeze(1).unsqueeze(1)
        else:
            # SD -> BSND
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)

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
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):

        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.hidden_dim = config.hidden_size
        self.batch_size_decode = infer_config.scheduler_config.batch_size
        self.local_rank = int(os.getenv("LOCAL_RANK", "1"))
        self.input_len = infer_config.scheduler_config.input_max_len
        self.batch_size_prefill = 1
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.comm_manager = comm_manager
        self.max_position_embeddings = config.max_position_embeddings
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.intermediate_size_per_rank = self.moe_intermediate_size // self.moe_tp_size
        self.experts_per_rank = config.num_experts // self.moe_ep_size
        self.force_eplb = infer_config.model_config.force_eplb
        self.ep_size = self.infer_config.parallel_config.moe_ep_size
        self.tp_size = self.infer_config.parallel_config.moe_tp_size
        self.experts = FusedMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            bias=False,
            quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=comm_manager.get_rank("moe_tp_group") if self.tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=comm_manager.get_rank("moe_ep_group") if self.ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )
        self.init_gate(prefix)
        self.row_idx_decode_len = self.batch_size_decode * self.top_k
        self.row_idx_decode = torch.arange(
            0, self.row_idx_decode_len,
            dtype=torch.int32).view(self.top_k, -1).permute(1, 0).contiguous().npu()

    def init_gate(self, prefix):
        self.norm_topk_prob = self.config.norm_topk_prob
        self.gating_dim = self.config.hidden_size
        self.gate = ReplicatedLinear(
            self.gating_dim,
            self.num_experts,
            bias=False,
            quant_config=None,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate"
        )

    def _forward_gate(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        logits = self.gate(hidden_states)
        topk_weight, topk_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(logits, None, k=self.top_k)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = None
        return topk_idx, topk_weight, aux_loss, row_idx

    def set_mc2_kwargs(self):
        global_rank = self.comm_manager.config.global_rank
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group")
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "moe_expert_num": self.num_experts,
                "global_bs": 0,
                "scales": None,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "quant_mode": 0
            }
        self.combine_kwargs = {
                "x_active_mask": None,
                "moe_expert_num": self.num_experts,
                "global_bs": 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "comm_quant_mode": 0
            }

    def forward(self, hidden_states, cur_topk_list=None, **kwargs):
        is_prefill = get_forward_metadata().is_prefill
        topk_idx, topk_weight, _, row_idx = self._forward_gate(hidden_states)
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)
        if self.moe_tp_size > 1:
            # MoE TP scene
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight)
        else:
            # MoE EP scene
            if is_prefill:
                return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight)
            else:
                return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight)

    def moe_infer_tp(self, hidden_states, topk_idx, topk_weight):
        batch_size, sequence_length, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        routing_args = {
            "expert_idx": topk_idx,
            "active_num": batch_size * sequence_length * self.top_k,
            "expert_num": self.num_experts,
            "expert_tokens_num_type": 1,  # 0: cumsum mode(not supported now); 1: count mode
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": -1
        }

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states, **routing_args
        )

        moe_args = {"group_list_type": 1}

        hidden_states_ordered_by_experts = self.experts(expanded_x, tokens_per_expert, **moe_args)

        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=None, skip2=None,
            bias=None,
            scales=topk_weight.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_tp_group"))

        y = hidden_states.view(batch_size, -1, self.hidden_dim)
        return y

    def dispatch_double_routing(self, tokens_per_expert, expanded_x):
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, input_splits, output_splits

    def moe_infer_double_routing(self, hidden_states, topk_ids, topk_weight):
        batch_size, sequence_length, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        bs_qlen = hidden_states.shape[0]
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=None,  # non-quant
            expert_num=self.num_experts,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=-1  # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        )
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group, gathered_tokens, input_splits, output_splits =\
            self.dispatch_double_routing(tokens_per_expert, expanded_x)

        # reroute
        hidden_states_ordered_by_experts, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1))

        # compute experts
        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
        }
        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        hidden_states = hidden_states.view(bs_qlen, self.hidden_dim)
        return hidden_states.view(batch_size, -1, h)

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight):
        """
        support ep for decode stage
        """
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)
        self.set_mc2_kwargs()

        # moe dispatch
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids, # [n*topk]
            **self.dispatch_kwargs
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        # compute experts
        gmm_args = {
            "x": expand_x,
            "expert_tokens": expert_token_num,
            "group_list_type": 1,
        }

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        # moe combine
        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "expert_ids": topk_ids,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32), # [n*topk]
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            **self.combine_kwargs
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.hidden_dim)
        return hidden_states


class Qwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: Optional[int] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size

        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"

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
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=comm_manager.get_rank("attn_tp_group"),
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.batch_size = infer_config.scheduler_config.batch_size
        self.input_len = infer_config.scheduler_config.input_max_len

        self.o_proj = RowParallelLinear(self.attn_intermediate_size,
                                        config.hidden_size,
                                        tp_size=self.attn_tp_size,
                                        tp_rank=comm_manager.get_rank("attn_tp_group"),
                                        bias=False,
                                        input_is_parallel=True,
                                        prefix=f"{prefix}.o_proj")
        self.scale_fa = 1 / (self.head_dim ** 0.5)
        self.k_cache = self.v_cache = torch.Tensor([])
        self.cache_unit = (self.head_dim * self.num_key_value_heads_per_rank,)

    def exec_qkv(
        self,
        qkv: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor]] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        bsz, q_len, _ = qkv.size()

        query_states, key_states, value_states = qkv.split((self.num_heads_per_rank * self.head_dim, \
                                                            self.num_key_value_heads_per_rank * self.head_dim, \
                                                            self.num_key_value_heads_per_rank * self.head_dim), dim=2)

        query_shape = (bsz, q_len, self.num_heads_per_rank, self.head_dim)
        key_value_shape = (bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim)

        query_states = self.q_norm(query_states.view(query_shape).contiguous())
        key_states = self.k_norm(key_states.view(key_value_shape).contiguous())

        cos, sin = cos_sin
        query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, layout='BSH')
        query_states = query_states.view(bsz, q_len, -1)
        key_states = key_states.view(bsz, q_len, -1)

        kv_len = forward_metadata.kv_len
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        attention_mask = forward_metadata.attention_mask

        k_cache, v_cache = self.k_cache, self.v_cache
        if not k_cache.numel() or not v_cache.numel():
            raise RuntimeError("A BUG: k_cache or v_cache are not initialized properly.")

        torch_npu.scatter_update_(k_cache, kv_len, key_states, -2)
        torch_npu.scatter_update_(v_cache, kv_len, value_states, -2)

        sparse_mode = 3
        fa_ops = torch.ops.npu
        if not forward_metadata.is_prefill:
            key_states, value_states = k_cache, v_cache
            attention_mask = None
            sparse_mode = 0
            if self.enable_gegraph:
                fa_ops = torchair.ops

        attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
            query_states, key_states, value_states,
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            softmax_scale=self.scale_fa,
            input_layout="BSH",
            sparse_mode=sparse_mode,
            atten_mask=attention_mask,
            actual_seq_kvlen=actual_seq_lengths_kv
        )

        attn_output = attn_output.reshape(bsz, q_len, self.attn_intermediate_size_per_rank)
        attn_output = self.o_proj(attn_output)
        bsz, q_len, h = attn_output.size()
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            # attn_TP + attn_DP
            new_output = torch.empty([bsz // self.attn_tp_size, q_len, h], dtype=attn_output.dtype, device="npu")
            dist.reduce_scatter_tensor(new_output, attn_output, group=self.comm_manager.get_group("attn_tp_group"))
            attn_output = new_output
        elif self.attn_tp_size > 1:
            # attention_TP + moe_TP
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, h = hidden_states.size()
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            # attn_TP + attn_DP
            h_dtype = hidden_states.dtype
            attn_tp_group = self.comm_manager.get_group("attn_tp_group")
            new_hidden_states = torch.empty([bsz * self.attn_tp_size, q_len, h], dtype=h_dtype, device="npu")
            dist.all_gather_into_tensor(new_hidden_states, hidden_states, group=attn_tp_group)
            hidden_states = new_hidden_states
        qkv = self.merged_qkv_proj(hidden_states)
        output = self.exec_qkv(
            qkv=qkv,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
        )
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = ""
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size

        self.self_attn = Qwen3MoeAttention(
            config=config,
            infer_config=infer_config,
            comm_manager=comm_manager,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn"
        )

        self.layer_idx = layer_idx

        self.mlp = (
            Qwen3MoeSparseMoeBlock(config, infer_config, comm_manager, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.batch_size = infer_config.scheduler_config.batch_size
        self.input_len = infer_config.scheduler_config.input_max_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            forward_metadata=forward_metadata,
            **kwargs
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)
        outputs = (residual, hidden_states)
        return outputs


class Qwen3MoeModel(nn.Module):
    """Transformer decoder consisting of config.num_hidden_layers layers."""
    def __init__(
        self,
        config: Qwen3MoeConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
        self.rank_id = int(os.getenv("LOCAL_RANK", "0"))
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.comm_manager = comm_manager
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=comm_manager.get_rank("embed_tp_group"))
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, infer_config, comm_manager, layer_idx, prefix=f"model.layers.{layer_idx}")
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config, max_position_embeddings=self.max_position_embeddings)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):

        batch_size, seq_length = input_ids.shape
        position_ids = position_ids.view(-1, seq_length).long()

        # TP+DP: Split batch within TP group, each rank processes different samples
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            samples_per_rank = batch_size // self.attn_tp_size
            rank_in_tp_group = self.comm_manager.get_rank("attn_tp_group")
            start_idx = rank_in_tp_group * samples_per_rank
            end_idx = start_idx + samples_per_rank
            input_ids = input_ids[start_idx:end_idx]
            position_ids = position_ids[start_idx:end_idx]
            batch_size = samples_per_rank

        if self.embed_tp_size > 1:
            new_input_ids = input_ids - self.rank_id * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        kv_len = forward_metadata.kv_len
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            cos_sin = self.rotary_emb(hidden_states.repeat(self.attn_tp_size, 1, 1),
                                      seq_length, kv_len, self.max_position_embeddings)
        else:
            cos_sin = self.rotary_emb(hidden_states, seq_length, kv_len, self.max_position_embeddings)
        residual = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                cos_sin=cos_sin,
                past_residual=residual,
                forward_metadata=forward_metadata,
                **kwargs
            )
            residual, hidden_states = layer_outputs

        hidden_states, _ = self.norm(hidden_states, residual)

        if hidden_states.size()[1] > 1:
            gather_index, _ = torch.max(position_ids, dim=-1)
            gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, hidden_states.shape[-1])
            hidden_states = torch.gather(hidden_states, 1, gather_index)

        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix: str = ""):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.num_hidden_layers = config.num_hidden_layers
        self.force_eplb = infer_config.model_config.force_eplb
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.input_max_len = infer_config.scheduler_config.input_max_len
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.max_position_embeddings = config.max_position_embeddings

        self.model = Qwen3MoeModel(config, infer_config, comm_manager, prefix)
        self.vocab_size_per_rank = config.vocab_size // self.lmhead_tp_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group")
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

        # TP+DP: all_gather after LM Head to reduce communication volume
        if self.attn_tp_size > 1 and self.attn_dp_size > 1:
            bsz, q_len, v = logits.size()
            new_logits = torch.empty([bsz * self.attn_tp_size, q_len, v],
                                    dtype=logits.dtype, device="npu")
            dist.all_gather_into_tensor(new_logits, logits, 
                                        group=self.comm_manager.get_group("attn_tp_group"))
            logits = new_logits

        if self.lmhead_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lmhead_tp_size)]
            dist.all_gather(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.concat(new_logits, dim=-1)
        logits = logits.float()
        return logits

    # Adapted from vllm.model_executor.models.qwen3moe.Qwen3MoeModel.load_weights
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = ()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                loaded_weight,
                                name_mapped,
                                shard_id=shard_id,
                                expert_id=expert_id,
                                )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def process_weights_after_loading(self):
        # Doing weight transpose, format cast to nz after loading weights from files
        for module_name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(
                    module, is_nz=self.infer_config.model_config.enable_weight_nz
                )
