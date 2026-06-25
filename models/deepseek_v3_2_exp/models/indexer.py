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
from typing import Optional, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import custom_ops

from executor.core.config import InferenceConfig, CommManager
from executor.utils import npu_stream_switch as npu_stream_switch_gegraph, npu_wait_tensor, get_had_pow2
from executor.utils.forward_metadata import PrefillCPMetaData
from executor.utils.stream_utils import (
    record_event,
    wait_event,
    record_stream,
    npu_stream_switch as npu_stream_switch_npugraph,
)
from module.linear import ReplicatedLinear


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
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: Optional[int] = None,
                 prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        model_config = self.infer_config.model_config
        parallel_config = self.infer_config.parallel_config
        custom_params = model_config.custom_params

        self.attn_tp_size = parallel_config.attn_tp_size
        self.cp_size = parallel_config.cp_size

        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.enable_gegraph = model_config.exe_mode == "ge_graph"
        self.enable_npugraph_ex = model_config.exe_mode == "npugraph_ex"
        self.npu_events = []
        streams = kwargs.get("npugraph_streams", {})
        self.prolog_stream = streams.get("indexer_prolog")
        self.weight_stream = streams.get("indexer_weight")
        if self.enable_multi_streams and self.enable_npugraph_ex:
            self.npu_events = [torch.npu.Event() for _ in range(4)]
        self.enable_pypto = custom_params.get("enable_pypto", False)
        self.pa_block_size = self.infer_config.scheduler_config.block_size

        self.dim: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.n_local_heads = config.index_n_heads // self.attn_tp_size
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.n_heads * self.head_dim,
                                     bias=False,
                                     quant_config=config.quant_config,
                                     prefix=f"{prefix}.wq_b")
        self.wk = ReplicatedLinear(self.dim,
                                    self.head_dim,
                                    bias=False,
                                    quant_config=config.quant_config,
                                    prefix=f"{prefix}.wk")
        self.weights_proj = ReplicatedLinear(self.dim,
                                    self.n_heads,
                                    bias=False,
                                    quant_config=config.quant_config,
                                    prefix=f"{prefix}.weights_proj")
        self.k_norm = LayerNorm(self.head_dim)
        self.softmax_scale = self.head_dim ** -0.5
        self.hadamard_matrix = nn.Parameter(get_had_pow2(128), requires_grad=False)

    def apply_hadamard(self, inp):
        matrix = self.hadamard_matrix
        init_shape = inp.shape
        inp = inp.view(-1, matrix.shape[0])
        return inp.matmul(matrix).view( \
            init_shape).to(torch.float16)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        kv_len: torch.Tensor,
        cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        c8_input_dict: Optional[Dict] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        indexer_key_scale_cache: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ):
        input_args = {
            "x": x,
            "qr": qr,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "kv_len": kv_len,
            "cos_sin": cos_sin,
            "position_ids": position_ids,
            "query_states": query_states,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
            "actual_seq_lengths_q": actual_seq_lengths_q,
            "cp_metadata": cp_metadata,
            "c8_input_dict": c8_input_dict,
            "indexer_key_cache": indexer_key_cache,
            "indexer_key_scale_cache": indexer_key_scale_cache,
            "is_prefill": is_prefill,
        }

        # indexer_prolog_pypto fusion ops only enabled under W8A8C8 scenario in decode stage
        enable_indexer_prolog_pypto = self.enable_pypto and self.kv_cache_quant_mode == "int8"
        if is_prefill or not enable_indexer_prolog_pypto:
            forward_func = self.prefill_decode_ascendc
        else:
            import custom_pypto
            forward_func = self.decode_indexer_prolog_pypto

        return forward_func(**input_args)

    def decode_indexer_prolog_pypto(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        kv_len: torch.Tensor,
        cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        c8_input_dict: Optional[Dict] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        indexer_key_scale_cache: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ):
        token_num = x.shape[0]
        cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.rope_head_dim)
        sin = sin.view(-1, 1, 1, self.rope_head_dim)

        key_states = indexer_key_cache
        key_scales = indexer_key_scale_cache

        res = torch.ops.custom_pypto.npu_lightning_indexer_prolog_pto(
            token_x=x.view(-1, self.dim),
            q_norm=qr.view(-1, self.q_lora_rank),
            q_norm_scale=c8_input_dict.get("pertoken_scale", None),
            wq_b=self.wq_b.weight,
            wq_b_scale=self.wq_b.weight_scale,
            wk=self.wk.weight,
            weights_proj=self.weights_proj.weight,
            ln_gamma_k=self.k_norm.weight,
            ln_beta_k=self.k_norm.bias,
            cos_idx_rope=cos.squeeze(1).squeeze(1),
            sin_idx_rope=sin.squeeze(1).squeeze(1),
            hadamard_q=self.hadamard_matrix,
            hadamard_k=self.hadamard_matrix,
            idx_k_cache=key_states,
            idx_k_scale_cache=key_scales,
            idx_k_cache_index=slot_mapping.view(-1),
            layernorm_epsilon_k=self.k_norm.eps,
            layout_query="TND",
            layout_key="PA_BSND"
        )
        q = res[0]
        query_dequant_scale = res[1]
        weights = res[2]

        li_ops_input = {}
        indexer_func = self.li_fusion
        key_scales_for_indexer = key_scales.squeeze(2) if key_scales.dim() == 4 else key_scales
        li_ops_input.update({"key_dequant_scale": key_scales_for_indexer,
                              "query_dequant_scale": query_dequant_scale.view(-1, self.n_heads),
                                })
        li_ops_input.update({"actual_seq_lengths_query": actual_seq_lengths_q,
                            "actual_seq_lengths_kv": actual_seq_lengths_kv,
                            "k": key_states,
                            "block_table": block_table,
                            "is_prefill": is_prefill,
                            })
        li_ops_input.update({"q": q, "weights": weights})
        return indexer_func(**li_ops_input)

    def prefill_decode_ascendc(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        kv_len: torch.Tensor,
        cos_sin: torch.Tensor,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        c8_input_dict: Optional[Dict] = None,
        indexer_key_cache: Optional[torch.Tensor] = None,
        indexer_key_scale_cache: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ):
        token_num = x.shape[0]
        enable_cp = cp_metadata is not None and getattr(cp_metadata, "enabled", False)
        cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.rope_head_dim)
        sin = sin.view(-1, 1, 1, self.rope_head_dim)
        enable_multi_streams = self.enable_multi_streams and not is_prefill
        enable_gegraph_and_multistream = enable_multi_streams and self.enable_gegraph
        enable_npugraph_and_multistream = enable_multi_streams and self.enable_npugraph_ex

        if enable_npugraph_and_multistream:
            record_stream(True, qr, self.prolog_stream)
            record_stream(True, cos, self.prolog_stream)
            record_stream(True, sin, self.prolog_stream)
            record_event(True, self.npu_events, 0)
        with (
            npu_stream_switch_npugraph(True, self.prolog_stream)
            if enable_npugraph_and_multistream
            else npu_stream_switch_gegraph(enable_gegraph_and_multistream, "22")
        ):
            # prolog for kv use multi streams
            if enable_multi_streams:
                if enable_gegraph_and_multistream:
                    qr = npu_wait_tensor(True, qr, query_states[0])
                elif enable_npugraph_and_multistream:
                    wait_event(True, self.npu_events, 0)
                else:
                    qr = npu_wait_tensor(True, qr, query_states[0])
            # q process in new stream
            q_b = self.wq_b(qr, c8_input_dict.get("pertoken_scale", None)) # [b,s,1536] @ [1536,64*128] = [b,s,64*128]

            if enable_npugraph_and_multistream:
                record_event(True, self.npu_events, 1)

            q = q_b.view(token_num, self.n_heads, self.head_dim)
            q_pe, q_nope = torch.split(q, [self.rope_head_dim, \
                                        self.head_dim - self.rope_head_dim], dim=-1)

            q_pe = q_pe.view(-1, self.n_heads, 1, self.rope_head_dim)
            q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(token_num, self.n_heads, self.rope_head_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)
            if enable_npugraph_and_multistream:
                record_event(True, self.npu_events, 2)

        if enable_npugraph_and_multistream:
            record_stream(True, x, self.weight_stream)

        with (
            npu_stream_switch_npugraph(True, self.weight_stream)
            if enable_npugraph_and_multistream
            else npu_stream_switch_gegraph(enable_gegraph_and_multistream, "33")
        ):
            if enable_multi_streams:
                if enable_gegraph_and_multistream:
                    x = npu_wait_tensor(True, x, q_b)
                elif enable_npugraph_and_multistream:
                    wait_event(True, self.npu_events, 1)
                else:
                    x = npu_wait_tensor(True, x, q_b)
            weights = self.weights_proj(x.view(-1, self.dim))
            if enable_npugraph_and_multistream:
                record_event(True, self.npu_events, 3)

        k_proj = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k = self.k_norm(k_proj)
        # [b,s,64+64]
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_rotary_mul(k_pe, cos, sin).view(token_num, 1, self.rope_head_dim)
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)
        key_dequant_scale = None
        indexer_input = {}
        key_states = indexer_key_cache

        if enable_npugraph_and_multistream:
            wait_event(True, self.npu_events, 2)
        if self.kv_cache_quant_mode == "int8":
            with (
                npu_stream_switch_npugraph(True, self.prolog_stream)
                if enable_npugraph_and_multistream
                else npu_stream_switch_gegraph(enable_gegraph_and_multistream, "22")
            ):
                # q quant
                q = self.apply_hadamard(q)
                q, query_dequant_scale = torch_npu.npu_dynamic_quant(q)
                query_dequant_scale = query_dequant_scale.type(torch.float16)
            # k quant
            k = self.apply_hadamard(k)
            k, key_dequant_scale = torch_npu.npu_dynamic_quant(k)
            key_dequant_scale = key_dequant_scale.type(torch.float16)

            if enable_cp:
                k_scale_all = key_dequant_scale.new_empty(
                    [
                        cp_metadata.local_token_num * cp_metadata.cp_size,
                        key_dequant_scale.shape[-1],
                    ]
                )
                dist.all_gather_into_tensor(
                    k_scale_all,
                    key_dequant_scale.view(token_num, -1),
                    group=self.comm_manager.get_group("cp_group"),
                )
                key_dequant_scale = torch.index_select(k_scale_all, 0, cp_metadata.restore_indices)
                query_dequant_scale_prev, query_dequant_scale_next = torch.split(
                    query_dequant_scale,
                    [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
                    dim=0,
                )
                query_dequant_scale_prev = query_dequant_scale_prev.reshape(-1, self.n_heads)
                query_dequant_scale_next = query_dequant_scale_next.reshape(-1, self.n_heads)
                c8_input_dict.update({
                                    "query_dequant_scale_prev": query_dequant_scale_prev,
                                    "query_dequant_scale_next": query_dequant_scale_next,
                                    })

            key_scales = indexer_key_scale_cache
            compute_key_scales = key_scales
            scatter_slot_mapping = (
                cp_metadata.global_slot_mapping["FullAttention"]
                if enable_cp else slot_mapping
            )
            if enable_cp:
                compute_block_num = cp_metadata.global_block_table["FullAttention"].numel()
                _, block_size, num_heads, scale_dim = key_scales.shape
                compute_key_scales = key_scales.new_zeros(
                    compute_block_num,
                    block_size,
                    num_heads,
                    scale_dim,
                )
            persistent_key_dequant_scale = (
                torch.index_select(key_dequant_scale, 0, cp_metadata.persistent_valid_indices)
                if enable_cp else key_dequant_scale
            )
            persistent_scale_slot_mapping = (
                cp_metadata.persistent_slot_mapping["FullAttention"]
                if enable_cp else scatter_slot_mapping
            )
            if persistent_scale_slot_mapping.numel() > 0:
                torch_npu.npu_scatter_nd_update_(
                    key_scales.view(-1, 1),
                    persistent_scale_slot_mapping.view(-1, 1),
                    persistent_key_dequant_scale.view(-1, key_dequant_scale.shape[-1]),
                )
            if enable_cp:
                torch_npu.npu_scatter_nd_update_(
                    compute_key_scales.view(-1, 1),
                    scatter_slot_mapping.view(-1, 1),
                    key_dequant_scale.view(-1, key_dequant_scale.shape[-1]),
                )
            key_scales_for_indexer = (
                compute_key_scales.squeeze(2) if compute_key_scales.dim() == 4 else compute_key_scales
            )
            indexer_input.update({"key_dequant_scale": key_scales_for_indexer,
                                   "query_dequant_scale": query_dequant_scale.view(-1, self.n_heads),
                                })
        if enable_cp:
            # CP prefill computes token shards on each rank. Restore the full
            # token-level KV first, then build two PA cache views below.
            kv_all = k.new_empty([cp_metadata.local_token_num * cp_metadata.cp_size,
                                  k.shape[-1]])
            dist.all_gather_into_tensor(
                kv_all,
                k.view(token_num, -1),
                group=self.comm_manager.get_group("cp_group"),
            )
            k = torch.index_select(kv_all, 0, cp_metadata.restore_indices)
        scatter_slot_mapping = (
            cp_metadata.global_slot_mapping["FullAttention"]
            if enable_cp else slot_mapping
        )
        if enable_cp:
            compute_block_num = cp_metadata.global_block_table["FullAttention"].numel()
            _, block_size, num_heads, head_dim = key_states.shape
            compute_key_states = key_states.new_zeros(
                compute_block_num,
                block_size,
                num_heads,
                head_dim,
            )
            # Persistent cache keeps only KV owned by this rank for later
            # decode; ranks without owner requests skip the empty scatter.
            persistent_k = torch.index_select(k, 0, cp_metadata.persistent_valid_indices)
            persistent_slot_mapping = cp_metadata.persistent_slot_mapping["FullAttention"]
            if persistent_slot_mapping.numel() > 0:
                torch_npu.npu_scatter_nd_update_(
                    key_states.view(-1, self.head_dim),
                    persistent_slot_mapping.view(-1, 1),
                    persistent_k.view(-1, k.shape[-1]),
                )
            # Current prefill indexer still needs full-batch KV, so scatter the
            # restored KV into a temporary compute cache.
            torch_npu.npu_scatter_nd_update_(
                compute_key_states.view(-1, self.head_dim),
                scatter_slot_mapping.view(-1, 1),
                k.view(-1, k.shape[-1]),
            )
            key_states = compute_key_states
        else:
            torch_npu.npu_scatter_nd_update_(
                key_states.view(-1, self.head_dim),
                slot_mapping.view(-1, 1),
                k.view(-1, k.shape[-1]),
            )
        indexer_func = self.li_fusion
        indexer_input.update({"actual_seq_lengths_query": actual_seq_lengths_q,
                            "actual_seq_lengths_kv": actual_seq_lengths_kv,
                            "k": key_states,
                            "block_table": block_table,
                            "k_proj": k_proj,
                            "is_prefill": is_prefill,
                            })

        if enable_npugraph_and_multistream:
            wait_event(True, self.npu_events, 3)
        if enable_cp:
            weights_prev, weights_next = torch.split(
                weights,
                [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
                dim=0,
            )
            weights_prev = weights_prev.reshape(-1, weights_prev.shape[-1])
            weights_next = weights_next.reshape(-1, weights_next.shape[-1])
            q_prev, q_next = torch.split(
                q,
                [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
                dim=0,
            )
            indexer_input.update({
                "q": q_prev.view(-1, self.n_heads, q.shape[-1]),
                "weights": weights_prev,
                })
            indexer_input.update({"actual_seq_lengths_kv": cp_metadata.kv_len_prev,
                                    "actual_seq_lengths_query": cp_metadata.actual_seq_q_prev})
            if self.kv_cache_quant_mode == "int8":
                indexer_input.update({"query_dequant_scale": c8_input_dict["query_dequant_scale_prev"]})
            topk_indices_prev = indexer_func(**indexer_input)
            indexer_input.update({
                "q": q_next.view(-1, self.n_heads, q.shape[-1]),
                "weights": weights_next,
                })
            indexer_input.update({"actual_seq_lengths_kv": cp_metadata.kv_len_next,
                                  "actual_seq_lengths_query": cp_metadata.actual_seq_q_next})
            if self.kv_cache_quant_mode == "int8":
                indexer_input.update({"query_dequant_scale": c8_input_dict["query_dequant_scale_next"]})
            topk_indices_next = indexer_func(**indexer_input)
            return (topk_indices_prev, topk_indices_next)
        else:
            indexer_input.update({"q": q, "weights": weights})
            return indexer_func(**indexer_input)

    def li_fusion(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        **kwargs
    ):
        use_pto = self.enable_pypto and not is_prefill and not self.kv_cache_quant_mode == "int8"
        if not use_pto:
            q = q.view(-1, self.n_heads, self.head_dim)
            layout_query = 'TND'
        else:
            import custom_pypto
            layout_query = 'BSND'

        li_input_kwargs = {
            "key": k,
            "actual_seq_lengths_query": actual_seq_lengths_query.to(torch.int32),
            "actual_seq_lengths_key": actual_seq_lengths_kv.to(torch.int32),
            "block_table": block_table,
            "layout_key": 'PA_BSND',
            "sparse_count": self.index_topk,
            "sparse_mode": 3,
            "query": q,
            "weights": weights,
            "layout_query": layout_query,
        }
        if self.kv_cache_quant_mode == "int8":
            key_dequant_scale = kwargs.get("key_dequant_scale", None)
            query_dequant_scale = kwargs.get("query_dequant_scale", None)
            li_input_kwargs.update({
                "key_dequant_scale": key_dequant_scale,
                "key_quant_mode": 0,
                "query_dequant_scale": query_dequant_scale,
                "query_quant_mode": 0,
                "weights": weights.type(torch.float16),
            })
            return torch_npu.npu_quant_lightning_indexer(**li_input_kwargs)
        elif not use_pto:
            return torch_npu.npu_lightning_indexer(**li_input_kwargs)[0]
        else:
            topk_indices = torch.ops.custom_pypto.npu_lightning_indexer_pto(**li_input_kwargs)
            return topk_indices.view(-1, 1, self.index_topk)
