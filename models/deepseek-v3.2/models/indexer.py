# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

""" PyTorch Index model."""
import os
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng
import custom_ops

from executor.utils import npu_stream_switch

from module.linear import ReplicatedLinear
from .modules import (_prepare_4d_causal_attention_mask, one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, apply_rotary_pos_emb, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel
                    )


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)


class Indexer(nn.Module):
    def __init__(self, config, runner_settings, layer_idx: Optional[int] = None,
                 prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache 
        self.runner_settings = runner_settings
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)

        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)

        self.mm_quant_mode = runner_settings.get("model_config").get("mm_quant_mode", "A16W16")
        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
        self.enable_gegraph = runner_settings.get("exe_mode", "ge_graph") == "ge_graph"
        self.enable_pypto = self.runner_settings.get("model_config").get("enable_pypto", False)

        self.dim: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.n_local_heads = config.index_n_heads // self.attn_tp_size
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.n_heads * self.head_dim,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(self.dim,
                                    self.head_dim,
                                    bias=False,
                                    quant_config=None,
                                    prefix=f"{prefix}.wk")
        self.weights_proj = ReplicatedLinear(self.dim,
                                    self.n_heads,
                                    bias=False,
                                    quant_config=None,
                                    prefix=f"{prefix}.weights_proj")
        self.k_norm = LayerNorm(self.head_dim)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        kv_len: torch.Tensor,
        cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        past_key_values_indexer: Optional[List[torch.FloatTensor]],
        mask: Optional[torch.Tensor],
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        cp_input_dict: Optional[Dict] = None,
        is_prefill: bool = True,
    ):
        x = x.view(kv_len.shape[0], -1, self.dim)
        bsz, seqlen, _ = x.size()

        if self.cp_size > 1 and is_prefill:
            _, _, cos, sin = cos_sin
        else:
            cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.rope_head_dim)
        sin = sin.view(-1, 1, 1, self.rope_head_dim)

        if is_prefill:
            end_pos = seqlen
        else:
            end_pos = actual_seq_lengths_kv[0]

        enable_multi_streams = self.enable_multi_streams and not is_prefill
        with npu_stream_switch(enable_multi_streams, "22"):
            # prolog for kv use multi streams
            if not is_prefill and self.enable_gegraph:
                tng.scope.npu_wait_tensor(qr, query_states[0])
                tng.scope.npu_wait_tensor(qr, key_states[0])

            # q process in new stream
            q = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # [b,s,64,128]
            q_pe, q_nope = torch.split(q, [self.rope_head_dim, \
                                        self.head_dim - self.rope_head_dim], dim=-1)  # [b,s,64,64+64]

            q_pe = q_pe.view(-1, self.n_heads, 1, self.rope_head_dim)
            # [b,s,n,d]
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(bsz, -1, self.n_heads, self.rope_head_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k = self.k_norm(k_proj)
        # [b,s,64+64]
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(bsz, -1, 1, self.rope_head_dim) # [b,s,1,d]
        k = torch.cat([k_pe, k_nope.unsqueeze(2)], dim=-1)  # [b,s,1,128]

        if self.cp_size > 1 and is_prefill:
            kv_all = k.new_empty([bsz * seqlen * self.cp_size, k.shape[-1]])
            dist.all_gather_into_tensor(kv_all, k.view(bsz * seqlen, -1), \
                                    group=self.hccl_comm_dict.get("cp_group", None))
            outputs_list = list(torch.split(kv_all, cp_input_dict["reverse_split_list"], dim=0))
            k = torch.cat([outputs_list[i] for i in cp_input_dict["cp_reverse_index"]], dim=0)

        if past_key_values_indexer is not None:
            past_key_states = past_key_values_indexer[self.layer_idx][0]
            # for long seq input, should use npu_scatter_nd_update_
            torch_npu.npu_scatter_nd_update_(past_key_states.view(-1, self.head_dim),
                                                slot_mapping.view(-1, 1),
                                                k.view(-1, k.shape[-1]))

        indexer_input = {}
        if is_prefill:
            # input format is [B, S, N, D] with seq pad to input_max_len, attention calc use the seq_len after pad. 
            # note: tnd layerout, actual_seq_lengths_q use cumsum indices
            seq_qlen_with_pad = torch.tensor([seqlen for _ in range(bsz)], dtype=kv_len.dtype, device=kv_len.device)
            actual_seq_lengths_kv = seq_qlen_with_pad
        indexer_func = self.forward_fusion
        indexer_input.update({"actual_seq_lengths_query": actual_seq_lengths_q,
                            "actual_seq_lengths_kv": actual_seq_lengths_kv,
                            "k": past_key_states,
                            "block_table": block_table,
                            })
        indexer_input.update({"k_proj": k_proj, "is_prefill": is_prefill})

        if self.cp_size > 1 and is_prefill:
            # [B, S, N, D] -> [T, N, D]
            x = x.flatten(0, 1).unsqueeze(0)
            q = q.flatten(0, 1).unsqueeze(0)
            x_prev, x_next = torch.split(x, x.size(1) // 2, dim=1)
            q_prev, q_next = torch.split(q, q.size(1) // 2, dim=1)
            indexer_input.update({
                "q": q_prev.view(bsz, -1, self.n_heads, q.shape[-1]).view(bsz, -1, q.shape[-2], q.shape[-1]),
                "x": x_prev.view(bsz, -1, x.shape[-1])})
            indexer_input.update({"actual_seq_lengths_kv": cp_input_dict["kv_len_prev"],
                                    "actual_seq_lengths_query": cp_input_dict["actual_seq_q"]})
            topk_indices_prev = indexer_func(**indexer_input)
            indexer_input.update({
                "q": q_next.view(bsz, -1, self.n_heads, q.shape[-1]),
                "x": x_next.view(bsz, -1, x.shape[-1])})
            indexer_input.update({"actual_seq_lengths_kv": cp_input_dict["kv_len_next"]})
            topk_indices_next = indexer_func(**indexer_input)
            return (topk_indices_prev, topk_indices_next)
        else:
            indexer_input.update({"q": q, "x": x})
            return indexer_func(**indexer_input)

    def forward_normal(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        k_proj: torch.Tensor,
        end_pos: int,
        mask: Optional[torch.Tensor],
        is_prefill: bool,
        **kwargs
    ):
        bsz, seqlen, _ = x.size()
        k = k.view(bsz, -1, 1, self.head_dim)

        enable_multi_streams = self.enable_multi_streams and not is_prefill
        with npu_stream_switch(enable_multi_streams, "33"):
            if not is_prefill and self.enable_gegraph:
                tng.scope.npu_wait_tensor(x, k_proj)
            weights = self.weights_proj(x) * self.n_heads ** -0.5  # [b,s,64]
            weights = weights.unsqueeze(-1) * self.softmax_scale   # [b,s,64,1]
        index_score = self.fp8_index(q, weights, k)
        if mask is not None:
            index_score += mask.squeeze(1)
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]

        return topk_indices
    
    def forward_fusion(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        k_proj: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        **kwargs
    ):
        enable_multi_streams = self.enable_multi_streams and not is_prefill
        li_input_kwargs = {
            "key": k,
            "actual_seq_lengths_query": actual_seq_lengths_query.to(torch.int32),
            "actual_seq_lengths_key": actual_seq_lengths_kv.to(torch.int32),
            "block_table": block_table,
            "layout_key": 'PA_BSND',
            "sparse_count": self.index_topk,
            "sparse_mode": 3,
        }
        if is_prefill or not self.enable_pypto:
            x = x.view(-1, self.dim)
            q = q.view(-1, self.n_heads, self.head_dim)
            with npu_stream_switch(enable_multi_streams, "33"):
                if not is_prefill and self.enable_gegraph:
                    tng.scope.npu_wait_tensor(x, k_proj)
                weights = self.weights_proj(x)
            li_input_kwargs.update({
                "query": q,
                "weights": weights,
                "layout_query": 'TND',
            })
            topk_indices = torch.ops.custom.npu_lightning_indexer(**li_input_kwargs)
            return topk_indices
        else:
            import custom_pypto
            with npu_stream_switch(enable_multi_streams, "33"):
                if not is_prefill and self.enable_gegraph:
                    tng.scope.npu_wait_tensor(x, k_proj)
                weights = self.weights_proj(x)
            li_input_kwargs.update({
                "query": q,
                "weights": weights,
                "layout_query": 'BSND',
            })
            topk_indices = torch.ops.custom_pypto.npu_lightning_indexer_pto(**li_input_kwargs)
            return topk_indices.view(-1, 1, self.index_topk)