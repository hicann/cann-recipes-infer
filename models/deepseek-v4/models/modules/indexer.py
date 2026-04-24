# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng

from transformers.cache_utils import Cache
from executor.utils import get_had_pow2, limit_core_num
from executor.utils.stream_utils import npu_stream_switch, record_event, wait_event, record_stream
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm, apply_rotary_emb, rotate_activation
from .compressor import Compressor


class Indexer(nn.Module):
    def __init__(self, config, runner_settings, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"
        self.runner_settings = runner_settings
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.partial_slice = [self.head_dim - self.rope_head_dim, self.head_dim]
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.wq_b = ReplicatedLinear(self.q_lora_rank,
                                     self.n_heads * self.head_dim,
                                     params_dtype=torch.bfloat16,
                                     quant_config=config.quant_config,
                                     prefix=f"{prefix}.wq_b")
        self.weights_proj = ReplicatedLinear(self.dim,
                                             self.n_heads,
                                             params_dtype=torch.bfloat16,
                                             quant_config=None,
                                             prefix=f"{prefix}.weights_proj")
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.max_seq_len = runner_settings.get("data_config").get("max_position_embeddings", 2048)
        self.hadamard_matrix = get_had_pow2(self.head_dim)
        self.compressor = Compressor(config, runner_settings, layer_idx, compress_ratio, rotate=True,
                                     head_dim = self.head_dim, prefix=f"{prefix}.compressor", **kwargs)
        self.enable_pypto = self.runner_settings.get("model_config").get("enable_pypto", False)
        self.enable_multi_streams = self.runner_settings.get("model_config").get("enable_multi_streams", False) and \
                                not self.enable_pypto
        self.platform_version = self.runner_settings.get("model_config").get("enable_multi_streams", False)
        self.enable_limit_core = self.runner_settings.get("model_config").get("enable_limit_core", False)
        aic_total = 24 # enable_limit_core only suppots A3
        aiv_to_aic_ratio = 2 # aiv_num is 2 * aic_num
        self.cmpr_aic_num = 16
        self.cmpr_aiv_num = self.cmpr_aic_num * aiv_to_aic_ratio
        self.rope_aic_num = aic_total - self.cmpr_aic_num
        self.rope_aiv_num = self.rope_aic_num * aiv_to_aic_ratio
        self.indexer_events = []
        if self.enable_multi_streams:
            # 2 is number of events used for event synchronization
            self.indexer_events = [torch.npu.Event(), torch.npu.Event()]

        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.global_rank = kwargs.get("global_rank")

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor,
        cache_data: Tuple[Dict],
        attn_metadata: Dict,
        cmpr_switch_flag: bool,
        cmpr_event: tuple[torch.npu.Event],
        cmpr_event_idx: int,
        is_prefill: bool = True,
    ):
        enable_multi_streams = self.enable_multi_streams and not is_prefill
        enable_limit_core = self.enable_limit_core and not is_prefill

        if is_prefill and self.cp_size > 1:
            cos_prev, sin_prev = attn_metadata["prev"]["cos_sin"]["comp"]
            cos_next, sin_next = attn_metadata["next"]["cos_sin"]["comp"]
            cos = torch.cat([cos_prev, cos_next], dim=0)
            sin = torch.cat([sin_prev, sin_next], dim=0)
        else:
            cos, sin = attn_metadata["cos_sin"]["comp"]

        with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aiv_num):
            # weight project
            if is_prefill and self.cp_size > 1:
                q_len = qr.shape[1] // 2
                if attn_metadata["prev"]["is_start"]:
                    x_for_weights_proj = torch.cat([x[0][:, :q_len], x[1][:, -q_len:]], dim=1)
                else:
                    x_for_weights_proj = torch.cat([x[0][:, -q_len:], x[1][:, -q_len:]], dim=1)
            else:
                x_for_weights_proj = x
            weights = self.weights_proj(x_for_weights_proj) * (self.softmax_scale * self.n_heads ** -0.5)
            if self.li_cache_quant_mode == "int8":
                weights = weights.flatten(0, 1).to(torch.float16)
            else:
                weights = weights.flatten(0, 1).to(torch.float32)
            # compressor
            self.compressor(x, cache_data, attn_metadata, is_prefill)

        # li event 0 and input tensors are recorded in Attention.mal_prolog function, after calling mla qb
        cur_stream = torch.npu.current_stream()
        with npu_stream_switch(enable_multi_streams, attn_metadata.get('indexer_stream', None)):
            wait_event(enable_multi_streams, self.indexer_events, 0)
            with limit_core_num(enable_limit_core, self.rope_aic_num, self.rope_aiv_num):
                q = self.wq_b(qr, dynamic_scale=qr_scale)
                q = q.view(qr.shape[0], -1, self.n_heads, self.head_dim)

                # rope
                torch.ops.custom.inplace_partial_rotary_mul(   # x: (T, 1, N, D); cos(T, 1, 1, D)
                    q.flatten(0, 1).unsqueeze(2), cos, sin,
                    rotary_mode="interleave",
                    partial_slice=self.partial_slice,
                )
                q = rotate_activation(q, self.hadamard_matrix)
                wait_event(cmpr_switch_flag, cmpr_event, cmpr_event_idx) # separate sfa compressor and li qb dynamic quant
                if self.li_cache_quant_mode == "int8":
                    q, q_scale = torch_npu.npu_dynamic_quant(q.flatten(0, 1))  # B,S,N,D -> T,N,D
                    q_scale = q_scale.to(torch.float16)
                else: # fp8
                    q, q_scale = torch_npu.npu_dynamic_block_quant(q.view(-1, q.shape[-1]), dst_type=cache_data["li_cmp_kv"].dtype)
                    q = q.view(-1, self.n_heads, self.head_dim)
                    q_scale = q_scale.view(-1, self.n_heads)
            record_stream(enable_multi_streams, q, cur_stream)
            record_stream(enable_multi_streams, q_scale, cur_stream)
            record_event(enable_multi_streams, self.indexer_events, 1)

        # LI fusion kernel
        wait_event(enable_multi_streams, self.indexer_events, 1)
        if is_prefill and self.cp_size > 1:
            q_prev, q_next = q.split(q.shape[0] // 2, dim=0)
            q_dict = {"prev": q_prev, "next": q_next}
            q_scale_prev, q_scale_next = q_scale.split(q_scale.shape[0] // 2, dim=0)
            q_scale_dict = {"prev": q_scale_prev, "next": q_scale_next}
            weights_prev, weights_next = weights.split(weights.shape[0] // 2, dim=0)
            weights_dict = {"prev": weights_prev, "next": weights_next}
            topk_list = []
            for zz_flag in ["prev", "next"]:
                topk_idxs = self.forward_li_quant(q_dict[zz_flag], q_scale_dict[zz_flag], cache_data["li_cmp_kv"],
                                                  cache_data["li_key_dequant_scale"], weights_dict[zz_flag], attn_metadata[zz_flag])
                topk_list.append(topk_idxs.view(-1, 1, topk_idxs.shape[-1]))
            return topk_list
        else:
            topk_idxs = self.forward_li_quant(
                q, q_scale, cache_data["li_cmp_kv"], cache_data["li_key_dequant_scale"], weights, attn_metadata)
            return topk_idxs.view(-1, 1, topk_idxs.shape[-1])

    def forward_li_quant(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: Dict
    ):
        actual_seq_q = attn_metadata["actual_seq_q"]
        actual_seq_k = attn_metadata["actual_seq_k"]
        li_input_kwargs = {
            "query": q,
            "key": k,
            "weights": weights,
            "query_dequant_scale": q_scale,
            "key_dequant_scale": k_scale.squeeze(-2),
            "actual_seq_lengths_query": actual_seq_q,
            "actual_seq_lengths_key": actual_seq_k,
            "block_table": attn_metadata["block_table"]["c4a_cmp_kv"],
            "layout_key": 'PA_BSND',
            "sparse_count": self.index_topk,
            "sparse_mode": 3,
            "layout_query": "TND",
            "cmp_ratio": 4, # only c4a have li module
            "key_quant_mode": 0,
            "query_quant_mode": 0,
            "metadata": attn_metadata["kernel_metadata"]["lightning_indexer_quant"]
        }
        topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(**li_input_kwargs)
        return topk_idxs.view(q.shape[0], -1, self.index_topk)