# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
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
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn

import torch_npu

from executor.core.config import InferenceConfig, CommManager
from executor.utils import get_had_pow2
from executor.utils.forward_metadata import PrefillCPMetaData
from executor.utils.stream_utils import (
    create_event, wait_tensor, npu_stream_switch,
    record_event, record_stream, wait_event)
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


class GlmMoeDsaIndexer(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache
        self.comm_manager = comm_manager
        model_config = infer_config.model_config
        custom_params = model_config.custom_params

        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.exe_mode = model_config.exe_mode
        self.enable_gegraph = self.exe_mode == "ge_graph"
        self.enable_npugraph_ex = self.exe_mode == "npugraph_ex"
        self.indexer_stream = kwargs.get("indexer_stream", None)
        self.weights_stream = kwargs.get("weights_stream", None)
        self.indexer_events = tuple(create_event(self.exe_mode, self.enable_multi_streams) for _ in range(4))

        self.dim: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
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
                                    quant_config=None,
                                    prefix=f"{prefix}.weights_proj")
        self.k_norm = LayerNorm(self.head_dim)
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
        return self.prefill_decode_ascendc(**input_args)

    def apply_hadamard_quant(self, x, use_float16_scale=False):
        init_shape = x.size()
        x = self.apply_hadamard(x)
        x = x.view(-1, self.head_dim)
        if self.li_cache_quant_mode == "int8":
            x, dequant_scale = torch_npu.npu_dynamic_quant(x)
        elif "float8" in self.li_cache_quant_mode:
            x, dequant_scale = torch_npu.npu_dynamic_block_quant(x, dst_type=torch.float8_e4m3fn)
        dequant_scale = dequant_scale.type(torch.float16) if use_float16_scale else dequant_scale
        return x.view(*init_shape), dequant_scale.view(*init_shape[:-1], -1)

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
        c8_input_dict = c8_input_dict or {}
        cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.rope_head_dim)
        sin = sin.view(-1, 1, 1, self.rope_head_dim)
        enable_multi_streams = self.enable_multi_streams and not is_prefill

        record_stream(enable_multi_streams, qr, self.indexer_stream, exe_mode=self.exe_mode)
        record_stream(enable_multi_streams, cos, self.indexer_stream, exe_mode=self.exe_mode)
        record_stream(enable_multi_streams, sin, self.indexer_stream, exe_mode=self.exe_mode)
        record_event(enable_multi_streams, self.indexer_events, 0, exe_mode=self.exe_mode)
        with npu_stream_switch(enable_multi_streams, self.indexer_stream, exe_mode=self.exe_mode):
            # prolog for kv use multi streams
            wait_event(enable_multi_streams, self.indexer_events, 0, exe_mode=self.exe_mode)
            wait_tensor(enable_multi_streams, qr, query_states[0], exe_mode=self.exe_mode)
            q_b = self.wq_b(qr, c8_input_dict.get("pertoken_scale", None)) # [b,s,1536] @ [1536,64*128] = [b,s,64*128]

            record_stream(enable_multi_streams, q_b, self.weights_stream, exe_mode=self.exe_mode)
            record_stream(enable_multi_streams, x, self.weights_stream, exe_mode=self.exe_mode)
            record_event(enable_multi_streams, self.indexer_events, 1, exe_mode=self.exe_mode)

            q = q_b.view(token_num, self.n_heads, self.head_dim)  # [T,64,128]
            q_pe, q_nope = torch.split(q, [self.rope_head_dim, \
                                        self.head_dim - self.rope_head_dim], dim=-1)  # [T,64,64+64]

            q_pe = q_pe.view(-1, self.n_heads, 1, self.rope_head_dim)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(token_num, self.n_heads, self.rope_head_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)
            record_event(enable_multi_streams, self.indexer_events, 2, exe_mode=self.exe_mode)
        with npu_stream_switch(enable_multi_streams, self.weights_stream, exe_mode=self.exe_mode):
            wait_event(enable_multi_streams, self.indexer_events, 1, exe_mode=self.exe_mode)
            wait_tensor(enable_multi_streams, x, q_b, exe_mode=self.exe_mode)
            weights = self.weights_proj(x.view(-1, self.dim))
            record_event(enable_multi_streams, self.indexer_events, 3, exe_mode=self.exe_mode)

        k_proj = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k = self.k_norm(k_proj)
        # [b,s,64+64]
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(token_num, 1, self.rope_head_dim)
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [T,1,128]
        indexer_input = {}
        key_states = indexer_key_cache

        wait_event(enable_multi_streams, self.indexer_events, 2, exe_mode=self.exe_mode)
        if self.li_cache_quant_mode != "unquant":
            use_float16_scale = True if self.li_cache_quant_mode == "int8" else False
            with npu_stream_switch(enable_multi_streams, self.indexer_stream, exe_mode=self.exe_mode):
                # q quant
                q, query_dequant_scale = self.apply_hadamard_quant(q, use_float16_scale=use_float16_scale)
            # k quant
            k, key_dequant_scale = self.apply_hadamard_quant(k, use_float16_scale=use_float16_scale)

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
                c8_input_dict.update({
                    "query_dequant_scale_prev": query_dequant_scale_prev.reshape(-1, self.n_heads),
                    "query_dequant_scale_next": query_dequant_scale_next.reshape(-1, self.n_heads),
                })

            key_scales = indexer_key_scale_cache
            compute_key_scales = key_scales
            scatter_slot_mapping = (
                cp_metadata.global_slot_mapping["FullAttention"] if enable_cp else slot_mapping
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
                cp_metadata.persistent_slot_mapping["FullAttention"] if enable_cp else scatter_slot_mapping
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
            kv_all = k.new_empty([cp_metadata.local_token_num * cp_metadata.cp_size, k.shape[-1]])
            dist.all_gather_into_tensor(
                kv_all,
                k.view(token_num, -1),
                group=self.comm_manager.get_group("cp_group"),
            )
            k = torch.index_select(kv_all, 0, cp_metadata.restore_indices)

            compute_block_num = cp_metadata.global_block_table["FullAttention"].numel()
            _, block_size, num_heads, head_dim = key_states.shape
            compute_key_states = key_states.new_zeros(
                compute_block_num,
                block_size,
                num_heads,
                head_dim,
            )
            persistent_k = torch.index_select(k, 0, cp_metadata.persistent_valid_indices)
            persistent_slot_mapping = cp_metadata.persistent_slot_mapping["FullAttention"]
            if persistent_slot_mapping.numel() > 0:
                torch_npu.npu_scatter_nd_update_(
                    key_states.view(-1, self.head_dim),
                    persistent_slot_mapping.view(-1, 1),
                    persistent_k.view(-1, k.shape[-1]),
                )
            torch_npu.npu_scatter_nd_update_(
                compute_key_states.view(-1, self.head_dim),
                cp_metadata.global_slot_mapping["FullAttention"].view(-1, 1),
                k.view(-1, k.shape[-1]),
            )
            key_states = compute_key_states
        else:
            torch_npu.npu_scatter_nd_update_(key_states.view(-1, self.head_dim),
                                            slot_mapping.view(-1, 1),
                                            k.view(-1, k.shape[-1]))

        indexer_func = self.li_fusion
        indexer_input.update({"actual_seq_lengths_query": actual_seq_lengths_q,
                            "actual_seq_lengths_kv": actual_seq_lengths_kv,
                            "k": key_states,
                            "block_table": block_table,
                            "k_proj": k_proj,
                            "is_prefill": is_prefill,
                            })
        wait_event(enable_multi_streams, self.indexer_events, 3, exe_mode=self.exe_mode)
        if enable_cp:
            weights_prev, weights_next = torch.split(
                weights,
                [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
                dim=0,
            )
            q_prev, q_next = torch.split(
                q,
                [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
                dim=0,
            )
            indexer_input.update({
                "q": q_prev.view(-1, self.n_heads, q.shape[-1]),
                "weights": weights_prev.reshape(-1, weights_prev.shape[-1]),
                "actual_seq_lengths_kv": cp_metadata.kv_len_prev,
                "actual_seq_lengths_query": cp_metadata.actual_seq_q_prev,
            })
            if self.li_cache_quant_mode != "unquant":
                indexer_input.update({"query_dequant_scale": c8_input_dict["query_dequant_scale_prev"]})
            topk_indices_prev = indexer_func(**indexer_input)

            indexer_input.update({
                "q": q_next.view(-1, self.n_heads, q.shape[-1]),
                "weights": weights_next.reshape(-1, weights_next.shape[-1]),
                "actual_seq_lengths_kv": cp_metadata.kv_len_next,
                "actual_seq_lengths_query": cp_metadata.actual_seq_q_next,
            })
            if self.li_cache_quant_mode != "unquant":
                indexer_input.update({"query_dequant_scale": c8_input_dict["query_dequant_scale_next"]})
            topk_indices_next = indexer_func(**indexer_input)
            return (topk_indices_prev, topk_indices_next)

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
        q = q.view(-1, self.n_heads, self.head_dim)
        layout_query = 'TND'

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
        if self.li_cache_quant_mode != "unquant":
            key_dequant_scale = kwargs.get("key_dequant_scale", None)
            query_dequant_scale = kwargs.get("query_dequant_scale", None)
            li_input_kwargs.update({
                "key_dequant_scale": key_dequant_scale,
                "key_quant_mode": 0,
                "query_dequant_scale": query_dequant_scale,
                "query_quant_mode": 0,
                "weights": weights.type(torch.float16) if self.li_cache_quant_mode == "int8" else weights,
            })
            return torch_npu.npu_quant_lightning_indexer(**li_input_kwargs)
        else:
            topk_indices, _ = torch_npu.npu_lightning_indexer(**li_input_kwargs)
            return topk_indices
