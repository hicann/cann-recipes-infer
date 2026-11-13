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
from typing import Callable, List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng
import cann_ops_transformer

from transformers.cache_utils import Cache
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
from executor.utils import get_had_pow2
from executor.utils.stream_utils import (
    limit_core_num, npu_stream_switch, record_event, wait_event, record_stream)
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm, apply_rotary_emb, rotate_activation, \
    partial_rotary_mul_quant, get_total_aic_num
from .compressor import Compressor

# MXFP4 uses one E8M0 scale for every 32 logical elements.  The operator
# exposes the scales as two lanes of d/64 rather than a flat d/32 dimension.
MXFP4_SCALE_GROUP_SIZE = 32
MXFP4_SCALE_LANES = 2


class Indexer(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                prefix: Optional[str] = "", comm_manager: CommManager = None,
                cache_getter: Callable[[str], torch.Tensor] = None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.cache_getter = cache_getter
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
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
                                             quant_config=config.quant_config \
                                                if self.mm_quant_mode == "w8a8hifloat8" else None,
                                             prefix=f"{prefix}.weights_proj")
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.max_seq_len = self.infer_config.model_config.custom_params.get("max_position_embeddings", 2048)
        self.hadamard_matrix = get_had_pow2(self.head_dim)
        self.compressor = Compressor(config, self.infer_config, layer_idx, compress_ratio, rotate=True,
                                     head_dim=self.head_dim, prefix=f"{prefix}.compressor",
                                     comm_manager=self.comm_manager, cache_getter=self.cache_getter, **kwargs)
        self.enable_pypto = self.infer_config.model_config.custom_params.get("enable_pypto", False)
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False) \
            and not self.enable_pypto
        self.platform_version = self.infer_config.model_config.platform_version
        self.enable_limit_core = self.infer_config.model_config.custom_params.get("enable_limit_core", False)

        aic_total = get_total_aic_num()
        aiv_to_aic_ratio = 2 # aiv_num is 2 * aic_num
        self.cmpr_aic_num = 16
        self.cmpr_aiv_num = self.cmpr_aic_num * aiv_to_aic_ratio
        self.rope_aic_num = aic_total - self.cmpr_aic_num
        self.rope_aiv_num = self.rope_aic_num * aiv_to_aic_ratio
        self.indexer_events = []
        if self.enable_multi_streams:
            # 2 is number of events used for event synchronization
            self.indexer_events = [torch.npu.Event(), torch.npu.Event()]

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.global_rank = kwargs.get("global_rank")

        if self.mm_quant_mode == "w8a8hifloat8":
            self.register_buffer(
                "q_scale_ones",
                torch.ones([1], dtype=torch.float32),
            )
        if self.li_cache_quant_mode == "hifloat8":
            self.register_buffer(
                "hif8_quant_scale", torch.ones(1, dtype=torch.bfloat16), persistent=False
            )

    def get_cache(self, cache_name: str) -> torch.Tensor:
        return self.cache_getter(cache_name)

    def get_runtime_cache(self, cache_name: str, attn_metadata: Dict, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.cp_size > 1 and not self.is_online:
            cp_metadata = attn_metadata.get("cp_metadata")
            tmp_cache = (
                cp_metadata.get("cp_tmp_cache")
                if isinstance(cp_metadata, dict) else None
            )
            if (
                not isinstance(tmp_cache, dict)
                or not tmp_cache
                or not isinstance(tmp_cache.get("4"), dict)
            ):
                raise ValueError(
                    f"Missing CP temporary cache for indexer layer {self.layer_idx}, "
                    "ratio 4 during offline prefill."
                )
            return tmp_cache["4"][cache_name]
        return self.get_cache(cache_name)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor,
        attn_metadata: Dict,
        cmpr_switch_flag: bool,
        cmpr_event: tuple[torch.npu.Event],
        cmpr_event_idx: int,
        is_prefill: bool = True,
        *,
        indexer_current_x: Optional[torch.Tensor] = None,
    ):
        enable_multi_streams = self.enable_multi_streams and not is_prefill
        enable_limit_core = self.enable_limit_core and not is_prefill

        if is_prefill and self.cp_size > 1:
            cos, sin = attn_metadata["cp_cos_sin"]["comp"]
        else:
            cos, sin = attn_metadata["cos_sin"]["comp"]

        with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aiv_num,
                            exe_mode=self.infer_config.model_config.exe_mode):
            # weight project
            if is_prefill and self.cp_size > 1:
                # Reuse the packed [prev, next] current-segment input.
                if indexer_current_x is not None:
                    x_for_weights_proj = indexer_current_x
                else:
                    # Fallback for callers that provide only augmented streams.
                    q_len = qr.shape[0] // 2
                    if attn_metadata["prev"]["is_start"]:
                        x_for_weights_proj = torch.cat(
                            [x[0][:q_len], x[1][-q_len:]], dim=0
                        )
                    else:
                        x_for_weights_proj = torch.cat(
                            [x[0][-q_len:], x[1][-q_len:]], dim=0
                        )
            else:
                x_for_weights_proj = x
            # compressor
            self.compressor(x, attn_metadata, is_prefill)

        # li event 0 and input tensors are recorded in Attention.mal_prolog function, after calling mla qb
        record_stream(enable_multi_streams, x_for_weights_proj, attn_metadata.get('indexer_stream', None))
        cur_stream = torch.npu.current_stream()
        with npu_stream_switch(enable_multi_streams, attn_metadata.get('indexer_stream', None)):
            wait_event(enable_multi_streams, self.indexer_events, 0)
            with limit_core_num(enable_limit_core, self.rope_aic_num, self.rope_aiv_num,
                                exe_mode=self.infer_config.model_config.exe_mode):
                q = self.wq_b(qr, dynamic_scale=qr_scale)
                q = q.view(-1, self.n_heads, self.head_dim)

                # rope
                if self.li_cache_quant_mode == "hifloat8":
                    q = partial_rotary_mul_quant(   # x: (T, 1, N, D); cos(T, 1, 1, D)
                        q.unsqueeze(2), cos, sin,
                        partial_slice=self.partial_slice,
                        platform_version=self.platform_version,
                        origin_shape=q.shape,
                        scale=self.hif8_quant_scale,
                    )
                else:
                    torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
                        q.unsqueeze(2), cos, sin,
                        rotary_mode="interleave",
                        partial_slice=self.partial_slice,
                    )
                # hif8 covers outliers natively, no need for hadamard
                if self.mm_quant_mode != "w8a8hifloat8":
                    q = rotate_activation(q, self.hadamard_matrix)
                weights = self.weights_proj(x_for_weights_proj) * (self.softmax_scale * self.n_heads ** -0.5)
                if self.li_cache_quant_mode == "int8":
                    weights = weights.to(torch.float16)
                else:
                    weights = weights.to(torch.float32)
                # separate sfa compressor and li qb dynamic quant
                wait_event(
                    cmpr_switch_flag and self.platform_version != PlatformVersion.ASCEND_950,
                    cmpr_event,
                    cmpr_event_idx)
                if self.li_cache_quant_mode == "int8":
                    q, q_scale = torch_npu.npu_dynamic_quant(q)  # T,N,D
                    q_scale = q_scale.to(torch.float16)
                elif self.li_cache_quant_mode == "hifloat8":
                    q = q.view(-1, self.n_heads, self.head_dim)
                    q_scale = self.q_scale_ones
                elif self.li_cache_quant_mode == "mxfloat4":
                    q, q_scale = torch_npu.npu_dynamic_mx_quant(q, dst_type=torch_npu.float4_e2m1fn_x2)
                    q = q.view(-1, self.n_heads, q.shape[-1])
                    q_scale = q_scale.view(torch.float8_e8m0fnu).view(
                        -1,
                        self.n_heads,
                        self.head_dim // (MXFP4_SCALE_GROUP_SIZE * MXFP4_SCALE_LANES),
                        MXFP4_SCALE_LANES,
                    )
                else: # fp8
                    q, q_scale = torch_npu.npu_dynamic_block_quant(
                        q.view(-1, q.shape[-1]),
                        dst_type=self.get_runtime_cache("li_cmp_kv", attn_metadata, is_prefill).dtype,
                    )
                    q = q.view(-1, self.n_heads, self.head_dim)
                    q_scale = q_scale.view(-1, self.n_heads)
            record_stream(enable_multi_streams, q, cur_stream)
            record_stream(enable_multi_streams, q_scale, cur_stream)
            record_stream(enable_multi_streams, weights, cur_stream)
            record_event(enable_multi_streams, self.indexer_events, 1)

        # LI fusion kernel
        wait_event(enable_multi_streams, self.indexer_events, 1)
        if is_prefill and self.cp_size > 1:
            cp_metadata = attn_metadata.get("cp_metadata")
            if not isinstance(cp_metadata, dict):
                cp_metadata = {}
            indexer_plan = cp_metadata.get("indexer_plan", {}) \
                if isinstance(cp_metadata, dict) else None
            local_half_tokens = cp_metadata.get("local_prev_token_num") \
                if isinstance(cp_metadata, dict) else None
            if local_half_tokens is None and isinstance(indexer_plan, dict):
                for flag in ("prev", "next"):
                    flag_plan = indexer_plan.get(flag, {})
                    if isinstance(flag_plan, dict) and flag_plan.get("local_half_tokens") is not None:
                        local_half_tokens = flag_plan["local_half_tokens"]
                        break
            if isinstance(local_half_tokens, torch.Tensor):
                # Prefer the host shape value to avoid per-layer device ``item()``.
                if local_half_tokens.numel() == 1 and local_half_tokens.device.type == "cpu":
                    local_half_tokens = int(local_half_tokens)
                else:
                    local_half_tokens = None
            if local_half_tokens is None:
                segment_lens = cp_metadata.get("segment_lens")
                if segment_lens is not None:
                    local_half_tokens = sum(segment_lens)
            # Fall back to the tensor shape when metadata is stale or incomplete.
            if not isinstance(local_half_tokens, int) \
                    or local_half_tokens <= 0 \
                    or local_half_tokens >= q.shape[0] \
                    or local_half_tokens * 2 != q.shape[0]:
                local_half_tokens = q.shape[0] // 2

            def split_rows(tensor, preferred_half):
                # Derive a complete split for legacy or alternate layouts.
                half = (
                    preferred_half
                    if tensor.shape[0] == 2 * preferred_half
                    else tensor.shape[0] // 2
                )
                return torch.split(
                    tensor, (half, tensor.shape[0] - half), dim=0
                )

            q_prev, q_next = split_rows(q, local_half_tokens)
            q_dict = {"prev": q_prev, "next": q_next}
            if self.li_cache_quant_mode == "hifloat8":
                q_scale_prev, q_scale_next = q_scale, q_scale
            else:
                q_scale_prev, q_scale_next = split_rows(q_scale, local_half_tokens)
            q_scale_dict = {"prev": q_scale_prev, "next": q_scale_next}
            weights_prev, weights_next = split_rows(weights, local_half_tokens)
            weights_dict = {"prev": weights_prev, "next": weights_next}
            topk_list = []
            for zz_flag in ["prev", "next"]:
                topk_idxs = self.forward_li_quant(
                    q_dict[zz_flag],
                    q_scale_dict[zz_flag],
                    self.get_runtime_cache("li_cmp_kv", attn_metadata, is_prefill),
                    self.get_runtime_cache("li_key_dequant_scale", attn_metadata, is_prefill),
                    weights_dict[zz_flag],
                    attn_metadata[zz_flag],
                )
                topk_list.append(topk_idxs.view(-1, 1, topk_idxs.shape[-1]))
            return topk_list
        else:
            topk_idxs = self.forward_li_quant(
                q,
                q_scale,
                self.get_runtime_cache("li_cmp_kv", attn_metadata, is_prefill),
                self.get_runtime_cache("li_key_dequant_scale", attn_metadata, is_prefill),
                weights,
                attn_metadata,
            )
            return topk_idxs.view(-1, 1, topk_idxs.shape[-1])

    def forward_li_quant(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: Dict,
    ):
        ratio_key = str(self.compress_ratio)
        # seqused_k is the valid length in the compressed cache; cmp_residual_k
        # carries the remainder needed to reconstruct the original sequence length.
        seq_used_k = attn_metadata["compressed_seq_lens"][ratio_key]
        cu_seqlens_q = attn_metadata["cu_seq_lens_q"]
        seq_used_q = attn_metadata["seq_used_q"]
        block_table_key = f"c{self.compress_ratio}a_cmp_kv"
        if attn_metadata.get("tmp_block_table", None) and not self.is_online:
            # enable prefill cp on offline mode
            block_table = attn_metadata["tmp_block_table"][block_table_key]
        else:
            block_table = attn_metadata["block_table"][block_table_key]
        # Lightning Indexer quant modes: 1: FP8, 2: INT8, 4: HiFloat8,
        # 5: MXFP4.
        quant_mode = {
            "int8": 2,
            "hifloat8": 4,
            "mxfloat4": 5,
        }.get(self.li_cache_quant_mode, 1)
        if self.li_cache_quant_mode == "hifloat8":
            k_scale = self.q_scale_ones.to(
                device=k_scale.device, dtype=k_scale.dtype
            ).contiguous()
        elif self.li_cache_quant_mode == "mxfloat4":
            k_scale = k_scale.view(
                k_scale.shape[0],
                k_scale.shape[1],
                k_scale.shape[2],
                self.head_dim // (MXFP4_SCALE_GROUP_SIZE * MXFP4_SCALE_LANES),
                MXFP4_SCALE_LANES,
            ).view(torch.float8_e8m0fnu)
        else:
            k_scale = k_scale.squeeze(-2)
        li_input_kwargs = {
            "query": q,
            "key": k,
            "weights": weights,
            "query_dequant_scale": q_scale,
            "key_dequant_scale": k_scale,
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_q": seq_used_q,
            "seqused_k": seq_used_k,
            "cmp_residual_k": attn_metadata["compressed_seq_remainders"][ratio_key],
            "block_table": block_table,
            "topk": self.index_topk,
            "quant_mode": quant_mode,
            "mask_mode": 3,  # 3: causal mask
            "layout_q": "TND",
            "layout_k": "PA_BBND",
            "cmp_ratio": self.compress_ratio, # only c4a have li module
            "metadata": attn_metadata["kernel_metadata"]["lightning_indexer_quant"]
        }
        topk_idxs, _ = torch.ops.cann_ops_transformer.quant_lightning_indexer(**li_input_kwargs)
        return topk_idxs.view(q.shape[0], -1, self.index_topk)
