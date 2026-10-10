# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Flash-Chat
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/longcat_flash/modeling_longcat_flash.py
# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
#
# The multimodal generation section below (search for "# Multimodal generation") is copied from
# the official LongCat-Next release (https://huggingface.co/meituan-longcat/LongCat-Next,
# modeling_longcat_next.py) and keeps its notice: Copyright (c) 2026 Meituan, released under the
# MIT License (see that repository's LICENSE file).
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import math
import os
import logging
import warnings
from typing import Iterable, Optional, Tuple

from dataclasses import dataclass
from typing import Union
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch_npu
import torchair as tng
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from tqdm import tqdm
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from executor.utils.forward_metadata import ForwardMetaData
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
    VocabParallelEmbedding,
)
from module.fuse_moe_gmm import FusedMoEGMM
from module.utils import get_moe_num_chunks, split_moe_tensors
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import (
    calc_moe_hccl_buffer_size, get_init_attn_mask,
    limit_core_num, npu_stream_switch, npu_wait_tensor,
)
from .configuration_longcat_next import LongcatNextConfig


ENABLE_NGRAM_EMBEDDING = True


def _mark_static(tensor):
    """Wrap torch._dynamo.mark_static so torch.compile sees a static tensor."""
    torch._dynamo.mark_static(tensor)  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class LongcatFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, *args) -> torch.Tensor:
        if len(args) == 0:
            # Pure RMSNorm (no residual)
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return result
        elif len(args) == 1 and args[0] is None:
            # First layer: residual is None, init residual from hidden_states
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1:
            # Fused residual add + RMSNorm
            residual = args[0]
            result, _, r = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (result, r)
        else:
            raise NotImplementedError(
                f"Unsupported LongcatFlashRMSNorm call with {len(args) + 1} arguments"
            )


# ---------------------------------------------------------------------------
# RoPE  (yarn + interleaved)
# ---------------------------------------------------------------------------

class LongcatFlashRotaryEmbedding(nn.Module):
    """Yarn RoPE: pre-compute cos/sin cache at init, forward is index_select."""
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config, max_seq_len, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        # transformers 5.0 yarn rope_init returns attention_scaling=1.0; force the
        # yarn mscale so cos/sin match the softmax_scale (base * m^2) convention.
        scaling_factor = self.config.rope_parameters.get("factor", 1)
        mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
        if self.rope_type != "default" and mscale_all_dim:
            self.attention_scaling = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_seq_len, device, dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ---------------------------------------------------------------------------
# MLA (Multi-head Latent Attention) with TP support
# ---------------------------------------------------------------------------

class LongcatFlashMLA(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str = ""):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = infer_config.parallel_config.attn_tp_size
        self.tp_rank = (dist.get_rank() % self.tp_size) if (self.tp_size > 1 and dist.is_initialized()) else 0
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        # Prefill replaces AllReduce(O_output) with
        # ReduceScatter(O_output) + AllGather(next_layer_input).
        # Decode keeps AllReduce because batch_per_dp_rank may be < attn_tp.
        self.is_sp = self.tp_size > 1 and self.attn_dp_size > 1
        self.comm_manager = comm_manager

        # Use tng.ops for FA calls in ge_graph mode (accepts Tensor for actual_seq_lengths_kv)
        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        self.fa_ops = tng.ops if self.enable_gegraph else torch.ops.npu

        # MLA prolog is an opt-in fused decode path.  The eager correctness
        # baseline must use the same split absorb sequence as the infer
        # LongCat-Flash reference; forcing prolog in eager makes it impossible
        # to distinguish a cache/parallelism issue from a fused-kernel issue.
        custom_params = getattr(infer_config.model_config, "custom_params", {}) or {}
        self.enable_mla_prolog = bool(custom_params.get("enable_mla_prolog", False))

        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.tp_size
        self.num_key_value_heads = 1  # MLA shared KV latent

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        # LongCat-Flash-Lite always uses Q LoRA; the no-q-lora variant is not
        # supported by this implementation.
        if self.q_lora_rank is None:
            raise ValueError("LongcatFlashMLA requires config.q_lora_rank to be set")

        # MLA parallel projections live under prefixes listed in the
        # compressed-tensors ``ignore`` set of the W8A8 deploy, so passing
        # ``quant_config`` here is safe — the framework resolves them to
        # ``UnquantizedLinearMethod`` and weights stay BF16. This keeps
        # npu_mla_prolog_v3 (which has no W8A8 args) working.
        quant_config = getattr(config, "quant_config", None)

        # Q path: compression is replicated, expansion is TP-split by heads.
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = LongcatFlashRMSNorm(config.q_lora_rank)
        self.q_b_proj = ColumnParallelLinear(
            config.q_lora_rank, self.num_heads * self.qk_head_dim,
            bias=False, tp_size=self.tp_size, tp_rank=self.tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        # KV path: compression is replicated (MQA shared), expansion is TP-split by heads
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = LongcatFlashRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, tp_size=self.tp_size, tp_rank=self.tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # Output: RowParallel (input is already head-parallel) + all_reduce
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            tp_size=self.tp_size, tp_rank=self.tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Scaling (softmax_scale for FA)
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        rope_type = self.config.rope_parameters.get("rope_type", "default")
        mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
        if rope_type != "default":
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        elif mscale_all_dim:
            # rope_type=default but mscale_all_dim is set → likely a stale YaRN
            # field that the inference path silently ignores. Warn so a misset
            # rope_parameters does not produce a wrong softmax_scale (skill F.5).
            warnings.warn(
                f"rope_parameters.mscale_all_dim={mscale_all_dim} is set but "
                f"rope_type=default — mscale is NOT applied to softmax_scale; "
                "confirm this matches the training-time setup.",
                RuntimeWarning,
            )

        self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5

        # PA configuration — sourced from scheduler_config so the framework's
        # allocator and MLA cache views see one consistent value.
        self.block_size = infer_config.scheduler_config.block_size

        # NZ-format weights consumed by torch_npu.npu_mla_prolog_v3.
        # Initialised in process_weights_after_loading (see CausalLM wrapper).
        self.weight_dq_prolog = None        # (He, Hcq) NZ — from q_a_proj.weight^T
        self.weight_dkv_kr_prolog = None    # (He, Hckv+Dr) NZ — from kv_a_proj_with_mqa.weight^T

        # Absorb weight placeholders (set in init_absorb_weights)
        self.kv_b_proj_w_k = None  # (num_heads_per_rank, qk_nope_head_dim, kv_lora_rank)
        self.kv_b_proj_w_v = None  # (num_heads_per_rank, kv_lora_rank, v_head_dim)

        # PA cache buffers populated by the unified cache framework via tensor_setter.
        # Each MLA shares one nope/rope cache, both keyed under attn_type "FullAttention".
        self.attn_type = "FullAttention"
        self.cache_nope = torch.Tensor()
        self.cache_rope = torch.Tensor()
        cache_dtype = self.config.torch_dtype
        self.cache_entries = [
            CacheEntry(
                cache_name="cache_nope",
                attn_type=self.attn_type,
                dim=self.kv_lora_rank,
                num_head=1,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=lambda t, layer=self: setattr(layer, "cache_nope", t),
            ),
            CacheEntry(
                cache_name="cache_rope",
                attn_type=self.attn_type,
                dim=self.qk_rope_head_dim,
                num_head=1,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=lambda t, layer=self: setattr(layer, "cache_rope", t),
            ),
        ]

    def init_absorb_weights(self):
        """Split kv_b_proj weight into K and V parts for absorb optimization.

        Called after weight loading in process_weights_after_loading.
        """
        # Called BEFORE process_weights_after_loading, so weight is still in original
        # (out_features, in_features) = (num_heads_per_rank * (nope+v), kv_lora_rank) layout.
        # .T gives (kv_lora_rank, num_heads_per_rank * (nope+v)).
        kv_b_proj_weight = self.kv_b_proj.weight.T.contiguous()  # (kv_lora_rank, num_heads_per_rank * (nope+v))
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_k, w_v = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # kv_b_proj_w_k: (num_heads_per_rank, qk_nope_head_dim, kv_lora_rank)
        # Used in absorb: q_nope @ kv_b_proj_w_k => (B, S, N, kv_lora_rank)
        self.kv_b_proj_w_k = w_k.permute(1, 2, 0).contiguous()
        # kv_b_proj_w_v: (num_heads_per_rank, kv_lora_rank, v_head_dim)
        # Used in output: attn_output @ kv_b_proj_w_v => (N, B*S, v_head_dim)
        self.kv_b_proj_w_v = w_v.transpose(0, 1).contiguous()

    def prepare_prolog_weights(self, enable_nz: bool = True):
        """Prepare weight handles required by torch_npu.npu_mla_prolog_v3.

        Must be called AFTER ``process_weights_after_loading`` so that
        ``q_b_proj.weight`` is already laid out as ``(in_features, out_features)``
        in NZ format. The plain ``nn.Linear`` weights (q_a_proj, kv_a_proj_with_mqa)
        keep their original ``(out_features, in_features)`` shape, so we transpose
        them here and (optionally) cast to NZ to match the kernel's expectation
        of ``(He, Hcq)`` and ``(He, Hckv + Dr)``.
        """
        # weight_dq: (He, Hcq) — derived from q_a_proj.weight (Hcq, He)
        wdq = self.q_a_proj.weight.data.transpose(0, 1).contiguous()
        # weight_dkv_kr: (He, Hckv + Dr) — derived from kv_a_proj_with_mqa.weight
        wdkv = self.kv_a_proj_with_mqa.weight.data.transpose(0, 1).contiguous()

        if enable_nz:
            wdq = torch_npu.npu_format_cast(wdq, 29)  # 29 == FRACTAL_NZ
            wdkv = torch_npu.npu_format_cast(wdkv, 29)

        self.weight_dq_prolog = wdq
        self.weight_dkv_kr_prolog = wdkv


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
        **kwargs,
    ) -> torch.Tensor:
        # Dispatch by phase flag (not by shape): decode is also packed (T>1) for
        # multi-batch, so shape alone cannot distinguish prefill from decode.
        if forward_metadata.is_prefill:
            return self._forward_prefill(hidden_states, position_embeddings, forward_metadata)
        if self.enable_mla_prolog and self.weight_dq_prolog is not None:
            return self._forward_decode_prolog(hidden_states, position_embeddings, forward_metadata)
        return self._forward_decode_absorb(hidden_states, position_embeddings, forward_metadata)

    def _forward_decode_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
    ) -> torch.Tensor:
        """Decode using the split MLA absorb path used by infer LongCat-Flash.

        This is the correctness baseline for eager mode.  The latent KV cache
        is written with ``npu_kv_rmsnorm_rope_cache_v2`` and read by FA v2 in
        ``TND_NTD`` form; only the K projection is absorbed into Q and the V
        projection is applied after attention.
        """
        num_tokens = hidden_states.shape[0]
        cos, sin = position_embeddings

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        q = q.view(num_tokens, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        q_nope = q_nope * self.mla_scale_q_lora
        q_pe = q_pe * self.mla_scale_q_lora

        # Q absorb: q_nope @ W_uk.  Keep the small-shape fused matmul used by
        # the official reference and fall back to torch.matmul for larger
        # products that exceed its index limit.
        if self.kv_b_proj_w_k.shape[0] * self.kv_b_proj_w_k.shape[1] <= 65535:
            q_nope = torch_npu.npu_transpose_batchmatmul(
                q_nope,
                self.kv_b_proj_w_k,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            ).view(num_tokens, self.num_heads_per_rank, self.kv_lora_rank)
        else:
            q_nope = (
                torch.matmul(q_nope.transpose(0, 1), self.kv_b_proj_w_k)
                .transpose(0, 1)
                .view(num_tokens, self.num_heads_per_rank, self.kv_lora_rank)
            )

        q_pe = q_pe.unsqueeze(0).transpose(1, 2)
        cos = cos.view(1, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(1, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.view(
            1, self.num_heads_per_rank, -1, self.qk_rope_head_dim
        ).transpose(1, 2).view(num_tokens, self.num_heads_per_rank, self.qk_rope_head_dim)

        latent_cache = latent_cache.view(
            num_tokens, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )
        cos_kv = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin_kv = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        slot_mapping = forward_metadata.slot_mapping[self.attn_type].view(-1)

        k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache_v2(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos_kv,
            sin_kv,
            slot_mapping.to(torch.int64),
            self.cache_rope,
            self.cache_nope,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            is_output_kv=True,
        )

        if self.mla_scale_kv_lora is not None:
            k_nope = k_nope * self.mla_scale_kv_lora


        # Decode FA reads the complete latent cache through the paged block
        # table.  The cache tensor itself remains unscaled; scaling the view
        # preserves one identical physical-cache convention for prefill and
        # decode.
        block_num, block_size, _, _ = self.cache_nope.size()
        kv_cache_nz_dim = 16
        cache_nope_nz = self.cache_nope.view(
            block_num, 1, self.kv_lora_rank // kv_cache_nz_dim,
            block_size, kv_cache_nz_dim
        )
        cache_rope_nz = self.cache_rope.view(
            block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim,
            block_size, kv_cache_nz_dim
        )
        if self.mla_scale_kv_lora is not None:
            cache_nope_nz = cache_nope_nz * self.mla_scale_kv_lora

        if self.enable_npugraph_ex:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_list_kv
            actual_seq_lengths_q_cu = forward_metadata.actual_seq_lengths_cu_list_q
        else:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
            actual_seq_lengths_q_cu = forward_metadata.actual_seq_lengths_cu_q

        attn_output, _ = self.fa_ops.npu_fused_infer_attention_score_v2(
            q_nope.contiguous(),
            cache_nope_nz,
            cache_nope_nz,
            query_rope=q_pe.contiguous(),
            key_rope=cache_rope_nz,
            atten_mask=None,
            actual_seq_kvlen=actual_seq_lengths_kv,
            actual_seq_qlen=actual_seq_lengths_q_cu,
            block_table=forward_metadata.block_table[self.attn_type],
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads,
            softmax_scale=self.softmax_scale,
            input_layout="TND_NTD",
            sparse_mode=0,
            block_size=self.block_size,
            query_quant_mode=0,
            key_quant_mode=0,
            value_quant_mode=0,
        )

        attn_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        attn_output = self.o_proj(attn_output.view(num_tokens, -1))
        if self.tp_size > 1:
            dist.all_reduce(
                attn_output,
                group=self.comm_manager.get_group("attn_tp_group"),
            )
        return attn_output

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
    ) -> torch.Tensor:
        """Prefill: non-absorb path. K/V are expanded via ``kv_b_proj`` and FA
        runs in NTD_TND layout over full per-head K/V. Inputs are packed 2D
        ``(T, H)``; a B=1 wrap is applied locally where a kernel requires 4D.
        """
        if self.is_sp:
            local_t, h_dim = hidden_states.shape
            full_t = local_t * self.tp_size
            gathered = torch.empty(
                full_t, h_dim, dtype=hidden_states.dtype, device=hidden_states.device,
            )
            dist.all_gather_into_tensor(
                gathered, hidden_states,
                group=self.comm_manager.get_group("attn_tp_group"),
            )
            hidden_states = gathered

        padded_q_len = hidden_states.shape[0]
        cos, sin = position_embeddings  # (B, S, D_rope) from rotary_emb; flatten below

        # SP path inflates T_local -> T_padded via all_gather, but framework
        # metadata (slot_mapping / cu_kv) is sized for T_real (sum of seq_lens).
        # Slice hidden_states / cos / sin back to T_real before FA + cache write;
        # re-pad just before reduce_scatter so the inverse comm shape matches.
        prompt_tokens = getattr(forward_metadata, "prompt_tokens", padded_q_len)
        if self.is_sp and 0 < prompt_tokens < padded_q_len:
            hidden_states = hidden_states[:prompt_tokens]
            cos = cos[:prompt_tokens]
            sin = sin[:prompt_tokens]

        num_tokens = hidden_states.shape[0]

        # --- Q LoRA path (init raises if q_lora_rank unset) ---
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        # (T, N, qk_head_dim) — TND form
        q_states = q_states.view(num_tokens, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # MLA LoRA scaling on Q
        q_nope = q_nope * self.mla_scale_q_lora
        q_pe = q_pe * self.mla_scale_q_lora

        # Q RoPE: npu_interleave_rope needs 4D (B, N, T, D). Wrap with B=1.
        q_pe = q_pe.unsqueeze(0).transpose(1, 2)  # (1, T, N, D_rope) → (1, N, T, D_rope)
        cos_q = cos.view(1, 1, num_tokens, self.qk_rope_head_dim)
        sin_q = sin.view(1, 1, num_tokens, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos_q, sin_q)
        q_pe = q_pe.transpose(1, 2).squeeze(0)  # back to (T, N, D_rope)

        # --- KV ---
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        # (T, 1, 1, kv_lora_rank + qk_rope_head_dim) for npu_kv_rmsnorm_rope_cache
        latent_cache = latent_cache.view(num_tokens, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)

        slot_mapping = forward_metadata.slot_mapping[self.attn_type]
        cos_kv = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin_kv = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        # Write to cache and get current KV outputs
        _, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos_kv,
            sin_kv,
            slot_mapping.view(-1).to(torch.int64),
            self.cache_rope,
            self.cache_nope,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            is_output_kv=True,
        )

        # Apply KV LoRA scaling
        k_nope = k_nope * self.mla_scale_kv_lora

        # Expand compressed latent to full K and V via kv_b_proj weights
        # k_nope: (B*S, 1, kv_lora_rank) -> expand to (1, B*S, kv_lora_rank) for matmul
        k_nope_2d = k_nope.view(1, -1, self.kv_lora_rank)
        # kv_b_proj_w_k: (N, qk_nope_head_dim, kv_lora_rank)
        # k_nope_out: (N, B*S, qk_nope_head_dim)
        k_nope_out = torch.matmul(k_nope_2d, self.kv_b_proj_w_k.permute(0, 2, 1))
        # v_out: (N, B*S, v_head_dim)
        v_out = torch.matmul(k_nope_2d, self.kv_b_proj_w_v)

        # k_rope: (B*S, 1, qk_rope_head_dim) -> (N, B*S, qk_rope_head_dim)
        k_rope = k_rope.view(1, -1, self.qk_rope_head_dim).repeat(self.num_heads_per_rank, 1, 1)

        # FA Prefill: NTD_TND layout (non-absorb, expanded K/V).
        # q_nope/q_pe are already (T, N, D); transpose to (N, T, D) for the
        # NTD-side of the layout.
        q_nope_ntd = q_nope.permute(1, 0, 2).contiguous()
        q_pe_ntd = q_pe.permute(1, 0, 2).contiguous()

        # NTD_TND attention: actual_seq_lengths{,_kv} must be cumulative
        # offsets ([T1, T1+T2, ...]) — they tell FA where each sequence ends
        # in the packed T dimension. The framework also exposes the per-request
        # form in actual_seq_lengths_kv; for B=1 the two are numerically equal,
        # but for B>1 we need the *_cu_kv field or attention crosses request
        # boundaries and produces garbage tokens.
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_cu_kv
        attention_mask = forward_metadata.attention_mask
        # For Prefill with NTD layout, use 2D causal mask [2048, 2048]
        if attention_mask is not None and attention_mask.dim() > 2:
            # Get the 2D causal mask portion
            attn_mask_2d = get_init_attn_mask(2048, hidden_states.device)
        else:
            attn_mask_2d = attention_mask

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_nope_ntd, k_nope_out.contiguous(), v_out.contiguous(),
            query_rope=q_pe_ntd, key_rope=k_rope.contiguous(),
            num_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_heads_per_rank,
            input_layout="NTD_TND",
            atten_mask=attn_mask_2d, sparse_mode=3,
            actual_seq_lengths=actual_seq_lengths_kv,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            next_tokens=0,
        )

        # FA with NTD_TND returns output in TND layout: (T, N, v_head_dim).
        # Flatten N*v_head_dim → packed 2D (T, N*v_head_dim) for o_proj; the
        # caller (DecoderLayer) threads packed 2D throughout.
        attn_output = attn_output.reshape(num_tokens, -1).contiguous()

        # --- Output ---
        attn_output = self.o_proj(attn_output)
        if self.is_sp:
            # Re-pad T_real -> T_padded so reduce_scatter sees the same shape
            # all_gather produced at entry.
            if attn_output.shape[0] < padded_q_len:
                pad = torch.zeros(
                    padded_q_len - attn_output.shape[0], attn_output.shape[-1],
                    dtype=attn_output.dtype, device=attn_output.device,
                )
                attn_output = torch.cat([attn_output, pad], dim=0)
            full_t, h_dim = attn_output.shape
            local_t = full_t // self.tp_size
            scattered = torch.empty(
                local_t, h_dim, dtype=attn_output.dtype, device=attn_output.device,
            )
            dist.reduce_scatter_tensor(
                scattered, attn_output,
                group=self.comm_manager.get_group("attn_tp_group"),
            )
            attn_output = scattered
        elif self.tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))
        return attn_output

    def _forward_decode_prolog(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
    ) -> torch.Tensor:
        """Decode path using ``torch_npu.npu_mla_prolog_v3``.

        The fused kernel performs (q_a -> rmsnorm -> q_b -> split + interleaved
        rope -> absorb-via-weight_uk) and (kv_a -> rmsnorm -> rope -> scatter
        into PA cache) in a single call.

        ``cache_nope`` is stored **unscaled** to match the prefill write path
        (``npu_kv_rmsnorm_rope_cache``). The kv_lora scale is therefore applied
        on the ``q_nope`` output below rather than via ``kc_scale``, so cache
        slots written by prefill and decode share a single scale convention.
        """
        # Inputs are packed 2D (T, H); prolog returns TND outputs directly.
        num_tokens = hidden_states.shape[0]
        hidden_states_2d = hidden_states
        cos, sin = position_embeddings

        # TND-friendly rope tables: (num_tokens, qk_rope_head_dim).
        rope_cos = cos.view(num_tokens, self.qk_rope_head_dim)
        rope_sin = sin.view(num_tokens, self.qk_rope_head_dim)

        slot_mapping = forward_metadata.slot_mapping[self.attn_type]
        cache_index = slot_mapping.view(-1).to(torch.int64)

        # qc_qr_scale folds the Q-LoRA scale into q_nope/q_pe. kc_scale=1.0
        # leaves the cache write unscaled; the kv_lora scale is applied to
        # q_nope below instead, so prefill and decode cache slots share scale.
        q_nope, q_pe, _, _, _ = torch_npu.npu_mla_prolog_v3(
            hidden_states_2d,
            self.weight_dq_prolog,
            self.q_b_proj.weight,
            self.kv_b_proj_w_k,
            self.weight_dkv_kr_prolog,
            self.q_a_layernorm.weight,
            self.kv_a_layernorm.weight,
            rope_sin,
            rope_cos,
            self.cache_nope,
            self.cache_rope,
            cache_index=cache_index,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            qc_qr_scale=float(self.mla_scale_q_lora),
            kc_scale=1.0,
        )
        # Apply the kv_lora scale on q_nope; cache stays unscaled.
        q_nope = q_nope * self.mla_scale_kv_lora
        # q_nope: (num_tokens, N, kv_lora_rank), q_pe: (num_tokens, N, qk_rope_head_dim).

        # Cache view as PA_NZ 5D for FA v2 TND_NTD.
        kv_cache_nz_dim = 16  # bf16/fp16 NZ inner dim
        cache_nope_nz = self.cache_nope.view(
            self.cache_nope.shape[0], 1,
            self.kv_lora_rank // kv_cache_nz_dim,
            self.cache_nope.shape[1], kv_cache_nz_dim,
        )
        cache_rope_nz = self.cache_rope.view(
            self.cache_rope.shape[0], 1,
            self.qk_rope_head_dim // kv_cache_nz_dim,
            self.cache_rope.shape[1], kv_cache_nz_dim,
        )

        # npugraph_ex requires actual_seq_lengths to be a Python list (Dynamo
        # SymInt[] schema). ge_graph + eager accept Tensor. Framework populates
        # the _list_ variants only when exe_mode == "npugraph_ex".
        if self.enable_npugraph_ex:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_list_kv
            actual_seq_lengths_q_cu = forward_metadata.actual_seq_lengths_cu_list_q
        else:
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
            actual_seq_lengths_q_cu = forward_metadata.actual_seq_lengths_cu_q

        # FA v2 with input_layout="TND_NTD":
        # query is TND (T, N, kv_lora) but the OUTPUT layout is NTD
        # (N, T, kv_lora) — the trailing "_NTD" suffix denotes the output
        # layout, not the cache layout. V absorb below handles the swap
        # via npu_transpose_batchmatmul.
        attn_output, _ = self.fa_ops.npu_fused_infer_attention_score_v2(
            q_nope.contiguous(), cache_nope_nz, cache_nope_nz,
            query_rope=q_pe.contiguous(), key_rope=cache_rope_nz,
            atten_mask=None,
            actual_seq_kvlen=actual_seq_lengths_kv,
            actual_seq_qlen=actual_seq_lengths_q_cu,
            block_table=forward_metadata.block_table[self.attn_type],
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads,
            softmax_scale=self.softmax_scale,
            input_layout="TND_NTD",
            sparse_mode=0,
            block_size=self.block_size,
            query_quant_mode=0, key_quant_mode=0, value_quant_mode=0,
        )
        # attn_output: (N, num_tokens, kv_lora_rank) — NTD output layout.

        # Cache is unscaled (kc_scale=1.0); q_nope*kv_lora covers the QK side, but
        # V (softmax @ cache_nope) still needs the kv_lora fold to match prefill.
        attn_output = attn_output * self.mla_scale_kv_lora

        # V absorb: attn_output (N, T, kv_lora) bmm kv_b_proj_w_v
        # (N, kv_lora, v_head_dim) → (T, N*v_head_dim) packed 2D for o_proj
        # (perm_y permutes (N, T) → (T, N) and flattens the last two dims).
        attn_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

        attn_output = self.o_proj(attn_output.view(num_tokens, -1))
        if self.tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))
        # Return packed 2D (T, H); DecoderLayer threads packed 2D throughout.
        return attn_output


# ---------------------------------------------------------------------------
# Dense MLP with TP support
# ---------------------------------------------------------------------------

class LongcatFlashMLP(nn.Module):
    """Dense MLP using its own TP knob (dense_tp_size).

    Decoupled from attn_tp_size so deployments can run e.g.
    ``attn_tp=4 + attn_dp=2`` (DP attention) alongside ``dense_tp=1``
    (replicated dense MLP, no AllReduce). When ``dense_tp_size == attn_tp_size``
    the comm_manager reuses ``attn_tp_group`` for ``dense_tp_group``, so this
    is a no-op for the existing 8tp deployment.
    """


    def __init__(  # pylint: disable=huawei-too-many-arguments
            self, config, infer_config: InferenceConfig, comm_manager: CommManager,
            hidden_size=None, intermediate_size=None, prefix=""):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.ffn_hidden_size if intermediate_size is None else intermediate_size
        self.tp_size = infer_config.parallel_config.dense_tp_size
        self.tp_rank = (dist.get_rank() % self.tp_size) if (self.tp_size > 1 and dist.is_initialized()) else 0
        self.comm_manager = comm_manager

        quant_config = getattr(config, "quant_config", None)
        self.mm_quant_mode = (
            quant_config.mm_quant_mode if quant_config is not None else "w16a16"
        )

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size, bias=False,
            tp_size=self.tp_size, tp_rank=self.tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x):
        if self.mm_quant_mode == "w8a8int8":
            # W8A8 fast path: gate_up returns int32 + per-token scale, fused
            # dequant+swiglu+quant produces int8 input for down_proj.
            merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
            intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                merged_x,
                weight_scale=self.gate_up_proj.weight_scale,
                quant_scale=self.down_proj.smooth_scales,
                quant_mode=1,
                activate_left=True,
                activation_scale=pertoken_scale,
            )
            result = self.down_proj(intermediate_hidden_states, pertoken_scale)
        else:
            merged_x = self.gate_up_proj(x)
            intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
            result = self.down_proj(intermediate_hidden_states)
        if self.tp_size > 1:
            dist.all_reduce(result, group=self.comm_manager.get_group("dense_tp_group"))
        return result


# ---------------------------------------------------------------------------
# MoE Router (replicated — all ranks must agree on routing)
# ---------------------------------------------------------------------------

class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.n_routed_experts + (config.zero_expert_num or 0)
        self.top_k = config.moe_topk
        self.routed_scaling_factor = config.routed_scaling_factor
        self.router_bias = getattr(config, "router_bias", False)
        self.classifier = nn.Linear(
            config.hidden_size,
            self.num_experts,
            bias=self.router_bias,
            dtype=torch.float32,
        )
        # register_buffer not in named_parameters()
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((self.num_experts,), dtype=torch.float32)
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(
            hidden_states.float(),
            self.classifier.weight.float(),
            self.classifier.bias.float() if self.classifier.bias is not None else None,
        )
        topk_weights, topk_indices, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            k=self.top_k,
            bias=self.e_score_correction_bias.float(),
            renorm=0,  # 0: softmax->topk
            norm_type=0,  # 0: softmax
            routed_scaling_factor=self.routed_scaling_factor,
            eps=float(1e-20),
        )
        return topk_weights, topk_indices.to(torch.int32)


# ---------------------------------------------------------------------------
# MoE with full fusion (gating_top_k + init_routing_v2 + grouped_matmul + finalize_routing)
# ---------------------------------------------------------------------------

class LongcatFlashMoE(nn.Module):
    """MoE with TP and EP support, switched by yaml parallel_config.

    TP path (moe_tp_size > 1): fused gating_top_k + init_routing_v2 +
        grouped_matmul + finalize_routing + all_reduce.
    EP path (moe_tp_size == 1, moe_ep_size > 1):
        - prefill: init_routing + AllToAll dispatch + re_routing + GMM +
            AllToAll combine + finalize_routing.
        - decode: MC2 npu_moe_distribute_dispatch_v2 / combine_v2 with
            ``copy_expert_num`` for identity (zero) experts.
    """


    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num or 0
        self.total_experts = self.n_routed_experts + self.zero_expert_num
        self.comm_manager = comm_manager

        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.use_ep = (self.moe_tp_size == 1 and self.moe_ep_size > 1)

        self.tp_rank = (dist.get_rank() % self.moe_tp_size) if (self.moe_tp_size > 1 and dist.is_initialized()) else 0
        self.ep_rank = (dist.get_rank() % self.moe_ep_size) if (self.moe_ep_size > 1 and dist.is_initialized()) else 0
        # Backward-compat alias: pre-merge code referenced self.tp_size for moe TP size.
        self.tp_size = self.moe_tp_size

        self.router = LongcatFlashTopkRouter(config)

        if self.use_ep:
            self.experts_per_rank = self.n_routed_experts // self.moe_ep_size
            # MC2 op kwargs are filled lazily (need group_name from comm_manager).
            self.dispatch_kwargs = None
            self.combine_kwargs = None
            # MC2 npu_moe_distribute_dispatch_v2 / combine_v2 caps
            # experts_per_rank <= 24 on A2 unlayered tiling; A3 has no such
            # cap. Default is ON unless the runtime platform is A2 and the
            # model exceeds the A2 cap. Override via
            # ``custom_params.enable_moe_mc2_dispatch``.
            platform = infer_config.model_config.platform_version
            platform = getattr(platform, "value", platform)
            cp = infer_config.model_config.custom_params or {}
            default_mc2 = not (platform == "A2" and self.experts_per_rank > 24)
            self.enable_moe_mc2_dispatch = bool(cp.get("enable_moe_mc2_dispatch", default_mc2))

        # W8A8 routing flag — when the framework attaches a CompressedTensors
        # quant_config (gmm_quant_mode == "w8a8int8"), the dispatch/prefill EP
        # paths run npu_dynamic_quant before the GMM call and forward the
        # per-token scale into FusedMoEGMM. ``w16a16`` keeps the BF16 path.
        quant_config = getattr(config, "quant_config", None)
        self.gmm_quant_mode = (
            quant_config.gmm_quant_mode if quant_config is not None else "w16a16"
        )

        # Single FusedMoEGMM constructor that takes both tp and ep knobs;
        # FusedMoEGMM internally routes shard layout based on tp/ep sizes.
        # Matches qwen3_moe / longcat-flash convention.
        self.experts = FusedMoEGMM(
            num_experts=self.n_routed_experts,
            hidden_size=self.hidden_size,
            intermediate_size=config.expert_ffn_hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.moe_ep_size,
            ep_rank=self.ep_rank,
            prefix="experts",
        )

    def forward(
        self,
        hidden_states,
        is_prefill: bool = False,
        prefill_moe_global_chunks=None,
    ):
        """MoE forward — accepts and returns packed 2D ``(T, H)``.

        The TP / EP-prefill / EP-decode paths all work on flat tokens
        internally; callers (DecoderLayer) thread packed 2D throughout
        so the entry / exit shape stays consistent.
        """
        if self.use_ep:
            # Prefill: always double-routing AllToAll (unconstrained, every chip).
            # Decode: MC2 dispatch_v2/combine_v2 when enabled (A3 graph mode);
            # otherwise reuse the prefill double-routing path so EP runs on A2.
            if is_prefill or not self.enable_moe_mc2_dispatch:
                return self._forward_ep_prefill(
                    hidden_states,
                    prefill_moe_global_chunks=prefill_moe_global_chunks,
                )
            return self._forward_ep_decode(hidden_states)
        return self._forward_tp(hidden_states)

    # ----- TP path (existing fast path: init_routing_v2 + GMM + finalize_routing) -----

    def _forward_tp(self, hidden_states):
        # ``hidden_states`` is packed 2D ``(T, H)``; the router does its own
        # final view, all downstream ops are flat-token native.
        hidden_states_2d = hidden_states

        # Step 1: Gating (fused softmax + bias + topk + scaling)
        topk_weight, topk_idx = self.router(hidden_states)
        topk_idx = topk_idx.to(torch.int32)

        # Step 2: Identity expert mask preparation (graph-safe, no boolean index assignment)
        # Identity experts have idx >= n_routed_experts
        # zero_expert_weight: keep only weights for identity experts (idx >= n_routed_experts), zero out routed ones
        identity_mask = (topk_idx >= self.n_routed_experts)  # True for identity experts
        zero_expert_weight = topk_weight * identity_mask.to(topk_weight.dtype)

        # Mask out identity experts from routed weights (keep only routed expert weights)
        routed_mask = (topk_idx < self.n_routed_experts)  # True for routed experts
        topk_weight = topk_weight * routed_mask.to(topk_weight.dtype)

        # Match the EP-prefill routing contract: keep all T*K rows assigned
        # to real experts. Excluding identity IDs via active_expert_range
        # produces -1 gather indices and an unused expanded_x tail, whereas
        # finalize_routing(drop_pad_mode=2) requires nonnegative row indices.
        # Identity entries take expert 0 with zero routed weight; their real
        # contribution is still added exactly once after the TP reduction.
        routed_topk_idx = topk_idx * routed_mask.to(topk_idx.dtype)

        # Step 3: Init routing (token-major expansion, as in EP prefill)
        expanded_x, expanded_row_idx, expert_tokens_num, _ = \
            torch_npu.npu_moe_init_routing_v2(
                hidden_states_2d,
                expert_idx=routed_topk_idx,
                active_num=topk_idx.shape[0] * topk_idx.shape[1],
                expert_num=self.n_routed_experts,
                expert_tokens_num_type=1,  # 1: count mode
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.n_routed_experts],
                row_idx_type=0,  # token-major gather map for finalize mode 2
                quant_mode=-1,  # BF16, no quantization
            )

        # Step 4: Expert computation (grouped_matmul + swiglu)
        expert_output = self.experts(
            x=expanded_x,
            expert_tokens=expert_tokens_num,
            group_list_type=1,
        )

        # Step 5: Finalize routing (weighted aggregation back to original order)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            expert_output, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(expert_output.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2,
        )

        # Step 6: all_reduce for TP on routed expert output
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_tp_group"))

        # Step 7: Identity expert contribution (input * sum of identity weights)
        identity_weight = zero_expert_weight.sum(dim=1, keepdim=True).to(hidden_states.dtype)
        hidden_states = hidden_states + hidden_states_2d * identity_weight

        return hidden_states

    # ----- EP path: prefill (double-routing AllToAll) and decode (MC2 dispatch_v2) -----

    def _compute_identity_output(self, hidden_states_2d, topk_idx, topk_weight):
        """Identity (zero) expert contribution, computed locally before EP dispatch.

        Identity experts have idx >= n_routed_experts; they output ``input * weight``
        with no FFN. Done locally so dispatch only handles routed experts.
        """
        identity_mask = (topk_idx >= self.n_routed_experts)
        # weight per token across top_k positions (zero on routed entries)
        identity_weights = topk_weight * identity_mask.to(topk_weight.dtype)
        identity_scale = identity_weights.sum(dim=-1, keepdim=True).to(hidden_states_2d.dtype)
        return hidden_states_2d * identity_scale

    def set_mc2_kwargs(self):
        """Build kwargs for npu_moe_distribute_dispatch_v2 / combine_v2 (decode path).

        Uses the independent ``moe_ep_group_mc2`` comm domain (framework allocates
        a dedicated HCCL buffer for it) so MC2 dispatch / combine run on a
        non-shared group, which is required by ``comm_alg=fullmesh_v2``.

        ``copy_expert_num`` exposes LongCat's identity (zero) experts to the op,
        which yields ``input * weight`` for IDs in
        [n_routed_experts, n_routed_experts + zero_expert_num) without an
        AllToAll hop.
        """
        global_rank = dist.get_rank()
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        copy_expert_num = self.zero_expert_num
        common = {
            "x_active_mask": None,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.n_routed_experts,
            "global_bs": 0,
            "group_ep": mc2_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": global_rank // self.moe_tp_size,
            "group_tp": mc2_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": global_rank % self.moe_tp_size,
            "zero_expert_num": 0,
            "copy_expert_num": copy_expert_num,
            "const_expert_num": 0,
        }
        self.dispatch_kwargs = {**common, "scales": None, "quant_mode": 0, "comm_alg": "fullmesh_v2"}
        self.combine_kwargs = dict(common)

    def _dispatch_double_routing(self, tokens_per_expert, expanded_x):
        """Prefill EP: AllToAll dispatch of routed-expert tokens."""
        ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=ep_group)

        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]

        gathered_tokens = expanded_x.new_empty(int(all_tokens.item()), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=ep_group)
        return tokens_per_expert_group, gathered_tokens, input_splits, output_splits

    def _combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        """Prefill EP: AllToAll combine after expert computation."""
        ep_group = self.comm_manager.get_group("moe_ep_group")
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=ep_group)
        return gathered_tokens

    def _forward_ep_prefill(self, hidden_states, prefill_moe_global_chunks=None):
        """Run EP prefill with identical routing rounds on every EP rank.

        The routing operator may produce different local token counts when the
        framework packs requests unevenly.  Flash keeps the collective sequence
        aligned by synchronizing a global chunk count and splitting the three
        token-aligned routing inputs together.  Keep identity-expert output on
        the original full sequence and concatenate routed results in token order.
        """
        hidden_states_2d = hidden_states

        # Gating is done once on the full local sequence so every chunk retains
        # the original token order and router decisions.
        topk_weight, topk_idx = self.router(hidden_states_2d)
        topk_idx = topk_idx.to(torch.int32)
        identity_output = self._compute_identity_output(
            hidden_states_2d, topk_idx, topk_weight
        )

        num_chunks = 1 if prefill_moe_global_chunks is None else prefill_moe_global_chunks
        routed_outputs = []
        for chunk_hidden, chunk_topk_idx, chunk_topk_weight in zip(
            *split_moe_tensors(
                hidden_states_2d,
                topk_idx,
                topk_weight,
                num_chunks=num_chunks,
            )
        ):
            # For EP routing path, only consider routed experts. Clamp identity
            # IDs to 0 and zero their weights so they do not perturb dispatch.
            routed_mask = chunk_topk_idx < self.n_routed_experts
            routed_topk_idx = chunk_topk_idx * routed_mask.to(chunk_topk_idx.dtype)
            routed_topk_weight = chunk_topk_weight * routed_mask.to(chunk_topk_weight.dtype)

            expanded_x, expanded_row_idx, tokens_per_expert, _ = \
                torch_npu.npu_moe_init_routing_v2(
                    chunk_hidden,
                    expert_idx=routed_topk_idx,
                    active_num=routed_topk_idx.shape[0] * routed_topk_idx.shape[1],
                    scale=None,
                    expert_num=self.n_routed_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, self.n_routed_experts],
                    quant_mode=-1,
                )

            # AllToAll dispatch
            tokens_per_expert_group, gathered_tokens, input_splits, output_splits = \
                self._dispatch_double_routing(tokens_per_expert, expanded_x)

            # Re-routing: align tokens to local expert order
            hidden_ordered, _, ids_unsort, tokens_per_local_expert = \
                torch_npu.npu_moe_re_routing(
                    gathered_tokens,
                    tokens_per_expert_group.view(self.moe_ep_size, -1),
                    per_token_scales=None,
                )

            # GMM on local experts. W8A8: per-rank dynamic quant after
            # re_routing; the dispatch remains BF16.
            if "a8" in self.gmm_quant_mode:
                hidden_ordered_q, hidden_ordered_scale = torch_npu.npu_dynamic_quant(
                    hidden_ordered
                )
                expert_output = self.experts(
                    x=hidden_ordered_q,
                    expert_tokens=tokens_per_local_expert,
                    group_list_type=1,
                    pertoken_scale=hidden_ordered_scale,
                )
            else:
                expert_output = self.experts(
                    x=hidden_ordered,
                    expert_tokens=tokens_per_local_expert,
                    group_list_type=1,
                )

            # Unsort back to dispatch order, then AllToAll combine.
            new_x = torch.index_select(
                expert_output, 0, ids_unsort.float().argsort().int()
            )
            gathered_tokens = self._combine_double_routing(
                new_x, expanded_x, input_splits, output_splits
            )

            # Finalize routing back to this chunk's original token positions.
            routed_outputs.append(
                torch_npu.npu_moe_finalize_routing(
                    gathered_tokens,
                    skip1=None,
                    skip2=None,
                    bias=None,
                    scales=routed_topk_weight.to(gathered_tokens.dtype),
                    expanded_src_to_dst_row=expanded_row_idx,
                    export_for_source_row=None,
                    drop_pad_mode=2,
                )
            )

        routed_output = torch.cat(routed_outputs, dim=0)
        return routed_output + identity_output

    def _forward_ep_decode(self, hidden_states):
        # ``hidden_states`` is packed 2D ``(T, H)``.
        hidden_states_2d = hidden_states

        # Gating
        topk_weight, topk_idx = self.router(hidden_states)
        topk_idx = topk_idx.to(torch.int32)

        if self.dispatch_kwargs is None:
            self.set_mc2_kwargs()

        dispatch_args = {
            "x": hidden_states_2d,
            "expert_ids": topk_idx,
            **self.dispatch_kwargs,
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, _, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        # GMM expert computation. We dispatched in BF16 (quant_mode=0), so do
        # the W8A8 activation quant per-rank here before the grouped matmul.
        # dispatch_v2's built-in W8A8 path expects a global per-expert
        # smooth_scale across all routed experts — we only own a shard, so
        # we keep alltoall in BF16 and pay one npu_dynamic_quant.
        if "a8" in self.gmm_quant_mode:
            expand_x_q, expand_x_scale = torch_npu.npu_dynamic_quant(expand_x)
            expert_output = self.experts(
                x=expand_x_q,
                expert_tokens=expert_token_num,
                group_list_type=1,
                pertoken_scale=expand_x_scale,
            )
        else:
            expert_output = self.experts(
                x=expand_x,
                expert_tokens=expert_token_num,
                group_list_type=1,
            )

        # combine_v2 needs ori_x to fold in copy_expert (identity) contributions.
        combine_args = {
            "expand_x": expert_output,
            "expert_ids": topk_idx,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32),
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            "ori_x": hidden_states_2d,
            **self.combine_kwargs,
        }
        hidden_states_2d = torch_npu.npu_moe_distribute_combine_v2(**combine_args)
        return hidden_states_2d


# ---------------------------------------------------------------------------
# Decoder Layer  (dual-sublayer + shortcut MoE)
# ---------------------------------------------------------------------------

class LongcatFlashDecoderLayer(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str = ""):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # MoE multi-stream overlap (Plan-A): hide MoE (dispatch+GMM+combine)
        # behind sub-layer 1 (dense_a + mla[1] + dense_b) on a side stream,
        # synced at the layer-end three-way add. Decode-only; prefill keeps the
        # serial path. Off by default — opt-in via custom_params.
        custom = getattr(infer_config.model_config, "custom_params", {}) or {}
        self.enable_moe_stream_overlap = bool(custom.get("enable_moe_stream_overlap", False))
        self.decompose_post_attn0_rmsnorm = bool(
            custom.get("decompose_post_attn0_rmsnorm", True)
        )
        # opt-out switch for limit_core_num (core split between the MoE side
        # stream and the main stream's sub-layer 1; required by multi-stream
        # overlap, default on).
        self.enable_limit_core_num = bool(custom.get("enable_limit_core_num", True))
        # Core split between MoE side stream and main stream's sub-layer 1 (non-AFD path).
        self._moe_aic_num = str(custom.get("moe_aic_num", 8))
        self._moe_aiv_num = str(custom.get("moe_aiv_num", 16))
        self._main_aic_num = str(custom.get("main_aic_num", 16))
        self._main_aiv_num = str(custom.get("main_aiv_num", 32))

        self.mlp = LongcatFlashMoE(config, infer_config, comm_manager)

        self.self_attn = nn.ModuleList([
            LongcatFlashMLA(config, infer_config, comm_manager, layer_idx * 2 + i,
                            prefix=f"{prefix}.self_attn.{i}")
            for i in [0, 1]
        ])
        self.mlps = nn.ModuleList([
            LongcatFlashMLP(config, infer_config, comm_manager,
                            prefix=f"{prefix}.mlps.{i}")
            for i in [0, 1]
        ])
        self.input_layernorm = nn.ModuleList([
            LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in [0, 1]
        ])
        self.post_attention_layernorm = nn.ModuleList([
            LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in [0, 1]
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData,
        prefill_moe_global_chunks=None,
        **kwargs,
    ) -> torch.Tensor:
        # Residual passing pattern using npu_add_rms_norm:
        # input_layernorm(hidden_states, residual) returns (normed, new_residual)
        # where new_residual = residual + hidden_states (fused add+norm)
        is_prefill = forward_metadata.is_prefill
        # Decode-only MoE/sub-layer-1 overlap; prefill keeps serial path.
        moe_overlap = self.enable_moe_stream_overlap and not is_prefill

        residual = None  # Will be initialized on first call

        # --- sub-layer 0 ---
        hidden_states, residual = self.input_layernorm[0](hidden_states, residual)
        hidden_states = self.self_attn[0](
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            forward_metadata=forward_metadata,
        )

        if self.decompose_post_attn0_rmsnorm:
            # The decomposed path is the validated default for the official
            # LongCat-Next entrypoint.  It preserves the same math while
            # avoiding the fused add+rms-norm graph conversion divergence.
            residual = residual + hidden_states
            hidden_states = torch_npu.npu_rms_norm(
                residual,
                self.post_attention_layernorm[0].weight,
                self.post_attention_layernorm[0].variance_epsilon,
            )[0]
        else:
            hidden_states, residual = self.post_attention_layernorm[0](hidden_states, residual)

        _limit_core_active = moe_overlap and self.enable_limit_core_num

        # -- MoE on side stream, sub-layer 1 on main stream (merge at three-way add) --
        with npu_stream_switch(moe_overlap, "moe"):
            with limit_core_num(_limit_core_active, self._moe_aic_num, self._moe_aiv_num):
                hidden_states = npu_wait_tensor(moe_overlap, hidden_states, residual)
                shortcut_mlp_output = self.mlp(
                    hidden_states,
                    is_prefill=is_prefill,
                    prefill_moe_global_chunks=prefill_moe_global_chunks,
                )

        # Main stream: sub-layer 1, limit-core split so it overlaps the MoE side stream.
        with limit_core_num(_limit_core_active, self._main_aic_num, self._main_aiv_num):
            hidden_states = self.mlps[0](hidden_states)
            hidden_states, residual = self.input_layernorm[1](hidden_states, residual)
            hidden_states = self.self_attn[1](
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                forward_metadata=forward_metadata,
            )
            hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)
            hidden_states = self.mlps[1](hidden_states)

        # Three-way add: residual + hidden_states + shortcut_mlp_output
        hidden_states = residual + hidden_states + shortcut_mlp_output

        return hidden_states


# ---------------------------------------------------------------------------
# N-gram Embedding (replicated, base embedding handled by caller with TP)
# ---------------------------------------------------------------------------

class NgramEmbedding(nn.Module):
    """N-gram sub-table embedding.

    Sub-tables can shard along an independent TP axis (ngram_embed_tp_size)
    distinct from base embedding's embed_tp_size — letting deployments e.g.
    DP-replicate the base table (embed_tp=1) while still sharding ngram
    sub-tables (ngram_embed_tp=8). When ngram_tp_size/group are not provided,
    falls back to base_embeddings' tp_size/tp_rank for backward compat.
    """

    def __init__(  # pylint: disable=huawei-too-many-arguments
        self,
        config,
        base_embeddings,
        comm_manager: Optional[CommManager] = None,
        ngram_tp_size: Optional[int] = None,
        ngram_tp_rank: Optional[int] = None,
        ngram_tp_group=None,
    ):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings
        self.comm_manager = comm_manager
        # ngram TP knobs default to base embedding's TP for backward compat
        # (legacy behavior: ngram sub-tables follow the base embed sharding).
        if ngram_tp_size is None:
            ngram_tp_size = getattr(base_embeddings, "tp_size", 1)
        if ngram_tp_rank is None:
            ngram_tp_rank = getattr(base_embeddings, "tp_rank", 0)
        self.embed_tp_size = ngram_tp_size
        self.embed_tp_rank = ngram_tp_rank
        # Cache the comm group: None => fall back to embed_tp_group via
        # comm_manager (legacy path); explicit group for independent ngram TP.
        self._ngram_tp_group = ngram_tp_group

        # LongCat-Next hashes text token IDs, not the full multimodal
        # embedding vocabulary.  The official HF implementation uses
        # ``text_vocab_size`` here; using ``vocab_size`` would create tables
        # with the wrong global row count and load the N-gram weights into
        # incorrect TP shards.
        self.m = config.ngram_vocab_size_ratio * config.text_vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num

        # Official LongCat-Next excludes multimodal control tokens from the
        # rolling N-gram state. Keep the shipped contiguous range as static
        # bounds so decode uses comparisons instead of torch.isin/nonzero.
        ignored_ids = getattr(config, "oe_ignored_token_ids", None)
        if ignored_ids is None:
            ignored_ids = range(
                int(config.text_vocab_size),
                int(getattr(config, "text_vocab_plus_multimodal_special_token_size", config.text_vocab_size)),
            )
        self._oe_ignored_ids = tuple(int(token_id) for token_id in ignored_ids)
        self._oe_ignored_lo = int(config.text_vocab_size)
        self._oe_ignored_hi = int(
            getattr(config, "text_vocab_plus_multimodal_special_token_size", config.text_vocab_size)
        )
        self._oe_ignored_is_contiguous = self._oe_ignored_ids == tuple(
            range(self._oe_ignored_lo, self._oe_ignored_hi)
        )

        num_embedders = self.k * (self.n - 1)
        emb_dim = config.hidden_size // num_embedders
        self.emb_dim = emb_dim

        embedders = []
        post_projs = []
        self.embedder_vocab_sizes = []
        self.embedder_padded_vocab_sizes = []
        for i in range(num_embedders):
            vocab_size = int(self.m + i * 2 + 1)
            padded_vocab_size = (
                (vocab_size + self.embed_tp_size - 1) // self.embed_tp_size
            ) * self.embed_tp_size
            self.embedder_vocab_sizes.append(vocab_size)
            self.embedder_padded_vocab_sizes.append(padded_vocab_size)
            if self.embed_tp_size > 1:
                embedders.append(
                    VocabParallelEmbedding(
                        padded_vocab_size,
                        emb_dim,
                        config.pad_token_id,
                        torch.bfloat16,
                        tp_size=self.embed_tp_size,
                        tp_rank=self.embed_tp_rank,
                    )
                )
            else:
                embedders.append(nn.Embedding(vocab_size, emb_dim, padding_idx=config.pad_token_id))
            post_projs.append(nn.Linear(emb_dim, config.hidden_size, bias=False))

        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)
        self._vocab_mods_cache = None

        # Ngram rolling window of last (n-1) tokens per sequence.
        # Registered as a buffer so torch.compile / ge_graph can trace the
        # in-place `copy_` updates below. The batch dimension is finalized
        # in init_ngram_cache() once the runtime batch size is known.
        self.register_buffer(
            "ngram_context",
            torch.zeros(1, max(self.n - 1, 1), dtype=torch.long),
            persistent=False,
        )

    def init_ngram_cache(self, batch_size: int, device=None):
        """Finalize ngram_context buffer once batch_size is known.

        Called from ModelWorker during kv-cache setup, before graph tracing,
        so the buffer shape matches what the compiled decode graph will see.
        """
        if device is None:
            device = self.ngram_context.device
        self.ngram_context = torch.zeros(
            batch_size, max(self.n - 1, 1),
            dtype=torch.long, device=device,
        )

    def is_oe_ignored(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return the ignored-token mask without invoking ``torch.isin``."""
        if self._oe_ignored_is_contiguous:
            return (token_ids >= self._oe_ignored_lo) & (token_ids < self._oe_ignored_hi)
        if not self._oe_ignored_ids:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        mask = token_ids == self._oe_ignored_ids[0]
        for token_id in self._oe_ignored_ids[1:]:
            mask = mask | (token_ids == token_id)
        return mask

    def precompute_vocab_mods(self):
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache
        vocab_mods = {}
        # Keep the rolling hash domain identical to the official Next model.
        vocab_size = self.config.text_vocab_size
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def shift_right_ignore_eos_unrolled(self, tensor: torch.Tensor, n: int, eos_token_id: int = 2) -> torch.Tensor:
        """Graph-capturable shift-right with EOS-aware segment boundaries.

        Semantics:
          - Split ``tensor`` along the seq dimension into segments delimited by EOS
            (each EOS is *included* in its preceding segment).
          - Within each segment of length L:
              * if L > n: positions [seg_start + n, seg_start + L) are filled with
                ``tensor[seg_start : seg_start + L - n]``.
              * else: all positions stay zero.
        """
        batch_size, seq_len = tensor.shape
        device = tensor.device

        special_mask = (tensor == 0)
        is_eos = (tensor == eos_token_id)  # (B, S)
        boundary = is_eos | special_mask
        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, S)

        # segment_start[i, j] = start index of the segment containing position j.
        # A new segment begins at position j iff the *previous* position was EOS.
        # cummax over (eos_at_prev_position ? j : -1) gives us the most recent such index.
        boundary_at_prev = torch.cat(
            [torch.zeros_like(boundary[:, :1]), boundary[:, :-1]],
            dim=-1,
        )  # (B, S): shift-right of is_eos, padding False at the start
        candidates = torch.where(boundary_at_prev, pos, torch.full_like(pos, -1))  # (B, S)
        # Replacement for torch.cummax: build the prefix max as a chain of
        # element-wise maxes. seq_len is fixed at trace time (decode-only graph,
        # n=4 for the longcat config), so the loop unrolls into a few ops.
        prefix_parts = [candidates[:, :1]]
        for i in range(1, seq_len):
            prefix_parts.append(torch.maximum(prefix_parts[-1], candidates[:, i:i + 1]))
        segment_start = torch.cat(prefix_parts, dim=-1).clamp_min(0)

        # shift_mask[i, j] = True iff offset within segment >= n (so we copy from j-n)
        offset = pos - segment_start
        shift_mask = offset >= n  # (B, S)

        # Source index for gather (always valid when mask is True; clamp prevents OOB when False)
        src_idx = (pos - n).clamp_min(0)  # (B, S)
        gathered = tensor.gather(dim=-1, index=src_idx)

        result = torch.where(shift_mask & ~special_mask, gathered, torch.zeros_like(tensor))
        return result

    def get_ngram_ids(self, input_ids, shifted_ids, vocab_mods, ngram):
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def _lookup_embedding(self, embedding: nn.Module, input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Lookup and all-reduce one sub-table (legacy helper)."""
        embeds = self.lookup_embedding_local(embedding, input_ids)
        if self.embed_tp_size <= 1 or not isinstance(embedding, VocabParallelEmbedding):
            return embeds
        if self._ngram_tp_group is not None:
            dist.all_reduce(embeds, group=self._ngram_tp_group)
        else:
            dist.all_reduce(embeds, group=self.comm_manager.get_group("embed_tp_group"))
        return embeds

    def lookup_embedding_local(self, embedding: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Lookup a local N-gram shard without launching a collective.

        Hash ID 0 is the sentinel used for missing history/control tokens; the
        official embedding-with-mask returns zero for it instead of exposing
        embedding row 0, and this local shard lookup does the same.
        """
        valid_ids = input_ids > 0
        if self.embed_tp_size <= 1 or not isinstance(embedding, VocabParallelEmbedding):
            return embedding(input_ids) * valid_ids.unsqueeze(-1)

        vocab_size_per_rank = embedding.input_size_per_partition
        local_input_ids = input_ids - self.embed_tp_rank * vocab_size_per_rank
        mask = (
            valid_ids
            & (local_input_ids >= 0)
            & (local_input_ids < vocab_size_per_rank)
        )
        local_input_ids = local_input_ids * mask
        return embedding(local_input_ids) * mask.unsqueeze(-1)

    def reduce_ngram_embeddings(self, embeddings):
        """Reduce all N-gram sub-table shards in one equivalent collective.

        Each sub-table projection is linear, so reducing the concatenated
        local embeddings before the individual projections is exactly equal
        to reducing each embedding separately. The payload is unchanged while
        twelve HCCL launches become one.
        """
        if self.embed_tp_size <= 1:
            return embeddings
        merged = torch.cat(embeddings, dim=-1)
        if self._ngram_tp_group is not None:
            dist.all_reduce(merged, group=self._ngram_tp_group)
        else:
            dist.all_reduce(merged, group=self.comm_manager.get_group("embed_tp_group"))
        return list(merged.split(self.emb_dim, dim=-1))

    def forward(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
        base_embeds: Optional[torch.Tensor] = None,
        actual_seq_lengths_cu_q: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Packed-1D entry: ``input_ids`` is ``(T,)`` and ``base_embeds`` is
        ``(T, H)``. Returns packed 2D ``(T, H)``.

        Dispatches by ``is_prefill``:
        - decode: each request is one token (T == B). Reshape to ``(B, 1)``
          and run BSND under graph mode (shift-right needs a fixed-shape
          unrolled max-chain because GE backend has no ``cummax``).
        - prefill: cu_q-driven packed shift-right with combined EOS +
          segment-start reset; ngram_context update via per-segment gather.
          Stays in packed 1D end-to-end since prefill is eager and request
          lengths can differ.
        """
        if not is_prefill:
            # Decode: B is static (from ngram_context), reshape to (B, 1).
            batch_size = self.ngram_context.shape[0]
            input_ids_bsnd = input_ids.view(batch_size, 1)
            base_embeds_bsnd = base_embeds.view(batch_size, 1, -1) if base_embeds is not None else None
            out_bsnd = self._forward_bsnd_decode(input_ids_bsnd, base_embeds=base_embeds_bsnd)
            return out_bsnd.view(-1, out_bsnd.shape[-1])

        # Prefill: packed 1D end-to-end (variable-length-safe).
        # Framework's ``actual_seq_lengths_cu_q`` is ``cumsum(seq_lens)``
        # *without* a leading zero (so single-segment T=256 arrives as
        # ``[256]``, not ``[0, 256]``). Normalise by prepending zero so the
        # ``cu_q[1:] - cu_q[:-1]`` segment-length math is well-defined.
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        if actual_seq_lengths_cu_q is not None and actual_seq_lengths_cu_q.numel() > 0:
            cu_q = actual_seq_lengths_cu_q.to(dtype=torch.long, device=device)
            if int(cu_q[0].item()) != 0:
                cu_q = torch.cat(
                    [torch.zeros(1, dtype=cu_q.dtype, device=device), cu_q]
                )
        else:
            cu_q = torch.tensor([0, num_tokens], dtype=torch.long, device=device)
        return self._forward_prefill_packed(input_ids, base_embeds, cu_q)

    def _forward_bsnd_decode(
        self,
        input_ids: torch.Tensor,
        base_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """BSND decode forward — accepts ``(B, 1)`` input_ids. Used by the
        packed 1D entry; decode-only because shift-right needs a fixed-shape
        unrolled max-chain (GE graph backend has no ``cummax``).
        Returns ``(B, 1, H)``.
        """
        seq_len = input_ids.size(-1)

        # Prepend the cached n-1 prior tokens to form the n-token context
        # window. Multimodal control tokens are represented as zero in this
        # context, matching the official LongCat-Next implementation.
        context = torch.cat([self.ngram_context[..., -(self.n - 1):], input_ids], dim=-1)
        context = torch.where(
            self.is_oe_ignored(context), torch.zeros_like(context), context
        )
        oe_ignored_mask = self.is_oe_ignored(input_ids)

        # Base embedding (from embed TP or direct lookup)
        if base_embeds is not None:
            x = base_embeds.clone()
        else:
            x = self.word_embeddings(input_ids).clone()

        vocab_mods = self.precompute_vocab_mods()

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self.shift_right_ignore_eos_unrolled(
                context, i - 1, eos_token_id=self.config.eos_token_id,
            )

        local_ngram_embeddings = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self.get_ngram_ids(context, shifted_ids, vocab_mods[(i, j)], ngram=i)
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.lookup_embedding_local(self.embedders[index], new_ids)
                local_ngram_embeddings.append(x_ngram)

        for index, x_ngram in enumerate(
            self.reduce_ngram_embeddings(local_ngram_embeddings)
        ):
            x_proj = self.post_projs[index](x_ngram)
            x = x + x_proj

        denominator = 1 + self.k * (self.n - 1)
        x = torch.where(
            oe_ignored_mask.unsqueeze(-1),
            x,
            x / denominator,
        )

        # Update ngram context (in-place so torch.compile can capture the mutation)
        new_tokens = torch.where(
            oe_ignored_mask, torch.zeros_like(input_ids), input_ids
        )
        self.ngram_context.copy_(
            torch.cat([self.ngram_context[:, 1:], new_tokens], dim=-1)
        )

        return x

    def _forward_prefill_packed(
        self,
        input_ids: torch.Tensor,
        base_embeds: torch.Tensor,
        cu_q: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length prefill — ``input_ids`` is ``(T,)`` 1D and
        ``base_embeds`` is ``(T, H)`` 2D. ``cu_q`` is ``(B+1,)`` cumulative
        offsets (with leading 0). Returns ``(T, H)``.
        """
        device = input_ids.device

        # Context uses zero for multimodal control tokens, matching the HF
        # implementation. Keep input_ids unchanged for the base embedding.
        oe_ignored_mask = self.is_oe_ignored(input_ids)
        context = torch.where(oe_ignored_mask, torch.zeros_like(input_ids), input_ids)

        x = base_embeds.clone()
        vocab_mods = self.precompute_vocab_mods()

        # cu_q-aware shift-right: reset at EOS-at-prev AND segment-start.
        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self.shift_right_ignore_eos_packed(
                context,
                shift_n=i - 1,
                eos_token_id=self.config.eos_token_id,
                cu_q=cu_q,
            )

        local_ngram_embeddings = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                ngram_ids = self.get_ngram_ids(context, shifted_ids, vocab_mods[(i, j)], ngram=i)
                new_ids = ngram_ids % emb_vocab_dim  # (T,) — no -seq_len trim needed (context == input)
                x_ngram = self.lookup_embedding_local(self.embedders[index], new_ids)
                local_ngram_embeddings.append(x_ngram)

        for index, x_ngram in enumerate(
            self.reduce_ngram_embeddings(local_ngram_embeddings)
        ):
            x_proj = self.post_projs[index](x_ngram)
            x = x + x_proj

        denominator = 1 + self.k * (self.n - 1)
        x = torch.where(
            oe_ignored_mask.unsqueeze(-1),
            x,
            x / denominator,
        )

        # ngram_context update: per-segment last (n-1) tokens via cu_q.
        n_minus_1 = max(self.n - 1, 1)
        batch_size = cu_q.shape[0] - 1
        end_idx = cu_q[1:] - 1                                              # (batch_size,) last packed pos per segment
        offsets = torch.arange(-n_minus_1 + 1, 1, device=device)            # (n-1,) [-(n-2), ..., 0]
        gather_idx = end_idx.unsqueeze(1) + offsets.unsqueeze(0)            # (batch_size, n-1)
        # Clamp underflow for the gather, but keep missing history as zero;
        # duplicating the segment's first token would create false N-grams on
        # the first decode step of a short segment.
        segment_starts = cu_q[:-1].unsqueeze(1)
        valid_context = gather_idx >= segment_starts
        safe_gather_idx = torch.maximum(gather_idx, segment_starts)
        gathered_context = input_ids[safe_gather_idx.view(-1)].view(batch_size, n_minus_1)
        gathered_context = torch.where(
            valid_context, gathered_context, torch.zeros_like(gathered_context)
        )
        new_context = torch.where(
            self.is_oe_ignored(gathered_context),
            torch.zeros_like(gathered_context),
            gathered_context,
        )
        self.ngram_context.copy_(new_context)

        return x

    def shift_right_ignore_eos_packed(
        self,
        input_ids: torch.Tensor,
        shift_n: int,
        eos_token_id: int,
        cu_q: torch.Tensor,
    ) -> torch.Tensor:
        """Shift-right by ``shift_n`` over packed 1D with combined reset.

        Reset happens at (a) each segment start derived from ``cu_q`` and
        (b) right after every EOS within a segment. Runs in prefill (eager) only.
        """
        num_tokens = input_ids.shape[0]
        device = input_ids.device

        special_mask = (input_ids == 0)                                     # (num_tokens,)
        is_eos = (input_ids == eos_token_id)                                # (num_tokens,)
        is_eos_at_prev = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=device), (is_eos | special_mask)[:-1]]
        )                                                                   # (num_tokens,)

        is_seg_start = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        is_seg_start[cu_q[:-1]] = True

        is_reset = is_seg_start | is_eos_at_prev                            # (num_tokens,)

        pos = torch.arange(num_tokens, device=device)
        candidates = torch.where(is_reset, pos, torch.full_like(pos, -1))
        segment_start, _ = candidates.cummax(dim=0)
        segment_start = segment_start.clamp_min(0)

        offset = pos - segment_start
        shift_mask = offset >= shift_n
        src_idx = (pos - shift_n).clamp_min(0)
        gathered = input_ids.gather(0, src_idx)
        return torch.where(
            shift_mask & ~special_mask,
            gathered,
            torch.zeros_like(input_ids),
        )


# ---------------------------------------------------------------------------
# Text backbone (28 decoder layers, dual MLA + MoE) with TP/EP support
# ---------------------------------------------------------------------------

class LongcatNextModel(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.embed_tp_rank = (
            (dist.get_rank() % self.embed_tp_size)
            if (self.embed_tp_size > 1 and dist.is_initialized())
            else 0
        )
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.attn_tp_rank = (
            (dist.get_rank() % self.attn_tp_size)
            if (self.attn_tp_size > 1 and dist.is_initialized())
            else 0
        )
        self.is_sp = self.attn_tp_size > 1 and self.attn_dp_size > 1

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.embed_tp_rank,
        )

        # N-gram TP knob: read from model_config.custom_params (kept out of
        # the framework's ParallelConfig per "no public framework changes"
        # rule). Default 0 means "follow embed_tp_size" (legacy behavior).
        custom = getattr(infer_config.model_config, "custom_params", {}) or {}
        ngram_tp = int(custom.get("ngram_embed_tp_size", 0))
        if ngram_tp == 0 or ngram_tp == self.embed_tp_size:
            # Reuse embed_tp_group (legacy path; NgramEmbedding falls back to
            # base_embeddings tp_size/rank when args are None).
            ngram_tp_size = None
            ngram_tp_rank = None
            ngram_tp_group = None
        else:
            ngram_tp_size = ngram_tp
            ngram_tp_rank = (dist.get_rank() % ngram_tp) if dist.is_initialized() else 0
            world = infer_config.parallel_config.world_size
            if not dist.is_initialized() or ngram_tp == world:
                # All ranks form a single group → use default world group
                # (None → dist.all_reduce uses the world group).
                ngram_tp_group = None
            else:
                ngram_tp_group = comm_manager.get_group("ngram_embed_tp_group")
        self.ngram_embed_tp_size = ngram_tp_size if ngram_tp_size is not None else self.embed_tp_size
        self.ngram_embed_tp_rank = ngram_tp_rank if ngram_tp_rank is not None else self.embed_tp_rank
        self.ngram_embed_tp_group = ngram_tp_group  # None == use embed_tp_group fallback

        self.ngram_embeddings = NgramEmbedding(
            config, self.embed_tokens, comm_manager,
            ngram_tp_size=ngram_tp_size,
            ngram_tp_rank=ngram_tp_rank,
            ngram_tp_group=ngram_tp_group,
        ) if ENABLE_NGRAM_EMBEDDING else None

        self.layers = nn.ModuleList([
            LongcatFlashDecoderLayer(config, infer_config, comm_manager, layer_idx,
                                     prefix=f"model.layers.{layer_idx}")
            for layer_idx in range(config.num_layers)
        ])
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Bound cos/sin cache by active reach (~KB), not max_position_embeddings (~82 MB BF16).
        rope_max_seq_len = (
            infer_config.data_config.input_truncated_len
            + infer_config.scheduler_config.max_new_tokens
        )
        self.rotary_emb = LongcatFlashRotaryEmbedding(
            config=config, max_seq_len=rope_max_seq_len,
        )

    def get_prefill_moe_global_chunks(self, hidden_states, is_prefill):
        """Return an EP-synchronized prefill chunk count.

        All EP ranks must execute the same number of collective rounds.  The
        helper uses the configured maximum chunk length and an EP MAX reduction
        to account for uneven packed-token counts across ranks.
        """
        if not is_prefill or self.moe_ep_size <= 1:
            return None
        custom_params = self.infer_config.model_config.custom_params or {}
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        return get_moe_num_chunks(
            hidden_states,
            moe_chunk_max_len,
            moe_ep_group,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_metadata: ForwardMetaData,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill

        # ``input_ids`` and ``position_ids`` are framework-packed 1D ``(T,)``;
        # the model now consumes them as-is throughout. NgramEmbedding picks
        # the equal-length fast path or the cu_q-driven variable-length slow
        # path internally.
        if input_ids.ndim != 1:
            input_ids = input_ids.view(-1)
        if position_ids.ndim != 1:
            position_ids = position_ids.view(-1)

        # SP-pad is applied AFTER embed + ngram_embeddings (both expect T_real
        # input_ids). qwen3_moe convention: only pad input embed_outputs for the
        # per-rank scatter; position_ids stays T_real so rotary_emb returns
        # T_real cos/sin (MLA prefill consumes them after gather + slice-back).
        sp_prefill = self.is_sp and is_prefill
        t_real = input_ids.shape[0]
        if sp_prefill:
            t_padded = (
                (t_real + self.attn_tp_size - 1) // self.attn_tp_size
            ) * self.attn_tp_size
            pad_len = t_padded - t_real
        else:
            pad_len = 0

        # Embedding with TP support — operates element-wise on 1D ids, returns
        # packed 2D ``(T, H)``.
        if inputs_embeds is not None:
            # The multimodal adapter already applied the official N-gram and
            # placeholder rules. Do not embed/update the context a second time.
            if inputs_embeds.ndim != 2 or inputs_embeds.shape != (t_real, self.config.hidden_size):
                raise ValueError("inputs_embeds must be packed [T_real, hidden_size]")
            base_embeds = inputs_embeds
        elif self.embed_tp_size > 1:
            new_input_ids = input_ids - self.embed_tp_rank * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            base_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(base_embeds, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            base_embeds = self.embed_tokens(input_ids)

        if inputs_embeds is not None:
            pass
        elif self.ngram_embeddings is not None:
            inputs_embeds = self.ngram_embeddings(
                input_ids,
                is_prefill=is_prefill,
                base_embeds=base_embeds,
                actual_seq_lengths_cu_q=forward_metadata.actual_seq_lengths_cu_q,
            )
        else:
            inputs_embeds = base_embeds

        # hidden_states is packed 2D (T, H) end-to-end from here.
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if sp_prefill and pad_len > 0:
            # Pad T_real -> T_padded along T dim so scatter math works.
            h_pad = torch.zeros(
                pad_len, hidden_states.shape[-1],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            hidden_states = torch.cat([hidden_states, h_pad], dim=0)

        if sp_prefill:
            tokens_per_rank = hidden_states.shape[0] // self.attn_tp_size
            start = self.attn_tp_rank * tokens_per_rank
            hidden_states = hidden_states[start:start + tokens_per_rank].contiguous()

        prefill_moe_global_chunks = self.get_prefill_moe_global_chunks(
            hidden_states, is_prefill
        )
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                forward_metadata=forward_metadata,
                prefill_moe_global_chunks=prefill_moe_global_chunks,
            )

        # norm + last-token gather run on packed 2D (T, H).
        hidden_states = self.norm(hidden_states)
        if forward_metadata.is_prefill:
            if sp_prefill:
                t_local, h_dim = hidden_states.shape
                t_full = t_local * self.attn_tp_size
                full = torch.empty(
                    t_full, h_dim,
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
                dist.all_gather_into_tensor(
                    full, hidden_states.contiguous(),
                    group=self.comm_manager.get_group("attn_tp_group"),
                )
                hidden_states = full

            # Prefill: select the last hidden state of each segment via
            # cumulative q-lengths (cu_q[i] is the end offset of segment i,
            # so cu_q[i]-1 is the last token's index in packed [T] order).
            cu_q = forward_metadata.actual_seq_lengths_cu_q.to(
                dtype=torch.long, device=hidden_states.device
            )
            hidden_states = torch.index_select(hidden_states, 0, cu_q - 1)
        # Decode: every token is already its segment's last; no gather needed.

        # Reshape to (B, 1, H) so the downstream lm_head + framework
        # `logits[:, -1:, :]` slicing keep working without changes.
        # (B equals num requests for prefill; equals batch_size for decode.)
        return hidden_states.view(hidden_states.shape[0], 1, hidden_states.shape[-1])


# ---------------------------------------------------------------------------
# LongcatFlashNgramForCausalLM  (top-level) with TP support
# ---------------------------------------------------------------------------

class LongcatNextForCausalLM(nn.Module):
    _can_compile_fullgraph = True

    def __init__(
        self,
        config: LongcatNextConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        # Framework EPLB hook reads these directly off the ForCausalLM instance.
        self.num_experts = config.n_routed_experts + (config.zero_expert_num or 0)
        self.num_experts_per_tok = config.moe_topk

        # PA block size: single source of truth for get_cache_info and the MLA
        # sublayers (both must agree with the framework's block allocator).
        self.block_size = infer_config.scheduler_config.block_size

        # Parallel sizes consumed by init_parallel_comm_group() below.
        self.world_size = infer_config.parallel_config.world_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.lmhead_tp_rank = (
            (dist.get_rank() % self.lmhead_tp_size)
            if (self.lmhead_tp_size > 1 and dist.is_initialized())
            else 0
        )
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.moe_tp_rank = (
            (dist.get_rank() % self.moe_tp_size)
            if (self.moe_tp_size > 1 and dist.is_initialized())
            else 0
        )

        self.init_parallel_comm_group()
        self.model = LongcatNextModel(config, infer_config, comm_manager)
        self.vocab_size = config.vocab_size
        # The multimodal embedding table contains visual/audio code tokens,
        # but the causal-LM head only predicts text plus multimodal control
        # tokens (the official HF implementation uses this smaller size).
        # TP requires the output dimension to be divisible by lmhead_tp_size;
        # pad the parameter rows and trim the gathered logits back to the
        # checkpoint's real vocabulary before sampling.
        self.lm_head_raw_vocab_size = int(
            getattr(config, "text_vocab_plus_multimodal_special_token_size", config.vocab_size)
        )
        self.lm_head_vocab_size = (
            (self.lm_head_raw_vocab_size + self.lmhead_tp_size - 1)
            // self.lmhead_tp_size
        ) * self.lmhead_tp_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.lm_head_vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=self.lmhead_tp_rank,
        )

    def check_model_settings(self):
        """Validate LongCat-Next constraints before allocating NPU runtime state."""
        parallel = self.infer_config.parallel_config
        model_config = self.infer_config.model_config
        custom = model_config.custom_params or {}

        if model_config.exe_mode not in {"eager", "ge_graph", "npugraph_ex"}:
            raise RuntimeError(
                f"Unsupported exe_mode={model_config.exe_mode!r}; expected eager, ge_graph or npugraph_ex."
            )
        if parallel.cp_size != 1:
            raise RuntimeError(f"LongCat-Next currently requires cp_size=1, got {parallel.cp_size}.")
        if model_config.next_n != 0:
            raise RuntimeError(
                f"LongCat-Next does not provide an MTP draft model; got next_n={model_config.next_n}."
            )

        parallel_sizes = {
            "attn_tp_size": parallel.attn_tp_size,
            "dense_tp_size": parallel.dense_tp_size,
            "moe_tp_size": parallel.moe_tp_size,
            "embed_tp_size": parallel.embed_tp_size,
            "lmhead_tp_size": parallel.lmhead_tp_size,
            "o_proj_tp_size": parallel.o_proj_tp_size,
        }
        for name, size in parallel_sizes.items():
            if size <= 0 or parallel.world_size % size != 0:
                raise RuntimeError(
                    f"{name}={size} must be positive and divide world_size={parallel.world_size}."
                )
        if self.config.num_attention_heads % parallel.attn_tp_size != 0:
            raise RuntimeError(
                f"num_attention_heads={self.config.num_attention_heads} must be divisible by "
                f"attn_tp_size={parallel.attn_tp_size}."
            )
        if self.config.vocab_size % parallel.embed_tp_size != 0:
            raise RuntimeError(
                f"vocab_size={self.config.vocab_size} must be divisible by "
                f"embed_tp_size={parallel.embed_tp_size}."
            )
        if parallel.attn_tp_size > 1 and parallel.attn_tp_size != parallel.o_proj_tp_size:
            raise RuntimeError(
                "attn_tp_size and o_proj_tp_size must match when attention TP is enabled."
            )

        moe_chunk_max_len = custom.get("moe_chunk_max_len", 65536)
        if not isinstance(moe_chunk_max_len, int) or moe_chunk_max_len <= 0:
            raise RuntimeError(
                f"moe_chunk_max_len must be a positive integer, got {moe_chunk_max_len!r}."
            )
        if model_config.exe_mode == "eager" and model_config.enable_cache_compile:
            raise RuntimeError("enable_cache_compile is not supported with exe_mode='eager'.")

    def init_parallel_comm_group(self):
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=self.world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
        )
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=self.world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
        )
        if self.dense_tp_size > 1:
            self.comm_manager.register_group(
                name="dense_tp_group",
                group_num=self.world_size // self.dense_tp_size,
                group_size=self.dense_tp_size,
            )
        if self.moe_tp_size > 1:
            self.comm_manager.register_group(
                name="moe_tp_group",
                group_num=self.world_size // self.moe_tp_size,
                group_size=self.moe_tp_size,
            )
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
            )
            # Keep the default path aligned with the infer repository LongCat-Flash, which
            # materializes the dedicated MC2 domain for EP. An explicit
            # ``enable_moe_mc2_dispatch: false`` is a diagnostic/compatibility
            # override and must skip both group creation and MC2 kwargs setup.
            custom_params = self.infer_config.model_config.custom_params or {}
            platform = self.infer_config.model_config.platform_version
            platform = getattr(platform, "value", platform)
            experts_per_rank = self.config.n_routed_experts // self.moe_ep_size
            default_mc2 = not (platform == "A2" and experts_per_rank > 24)
            enable_moe_mc2_dispatch = bool(
                custom_params.get("enable_moe_mc2_dispatch", default_mc2)
            )
            if self.moe_tp_size == 1 and enable_moe_mc2_dispatch:
                moe_ep_mc2_buffer_size = calc_moe_hccl_buffer_size(
                    self.infer_config, self.config, is_full_mesh_v2=False
                )
                self.comm_manager.register_group(
                    name="moe_ep_group_mc2",
                    group_num=moe_ep_group_num,
                    group_size=self.moe_ep_size,
                    group_stride=self.moe_tp_size,
                    return_name=True,
                    allow_physical_reuse=False,
                    hccl_buffer_size=moe_ep_mc2_buffer_size,
                    group_type=None,
                )

        custom = getattr(self.infer_config.model_config, "custom_params", {}) or {}
        ngram_tp = int(custom.get("ngram_embed_tp_size", 0))
        if (
            ngram_tp > 0
            and ngram_tp != self.embed_tp_size
            and ngram_tp != self.world_size
        ):
            self.comm_manager.register_group(
                name="ngram_embed_tp_group",
                group_num=self.world_size // ngram_tp,
                group_size=ngram_tp,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_metadata: ForwardMetaData,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            **kwargs,
        )
        logits = self.forward_lm_head(hidden_states)
        return logits

    def forward_features(self, input_ids, position_ids, forward_metadata, inputs_embeds):
        """Tensor-only multimodal boundary; keeps the text forward contract intact."""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            inputs_embeds=inputs_embeds,
        )


    def forward_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """LM head: (B, 1, H) → (B, 1, V) via all_gather over column-parallel
        vocab shards. NgramModel.forward emits the BSND wrap (S=1) right
        after the last-token gather so this path keeps the framework's
        ``logits[:, -1:, :]`` sampler slicing unchanged.
        """
        logits = self.lm_head(hidden_states)
        if self.lmhead_tp_size > 1:
            tp = self.lmhead_tp_size
            # each rank: (B, 1, V/tp); gather along dim 0 → (tp*B, 1, V/tp)
            gathered = torch.empty(
                [logits.shape[0] * tp, *logits.shape[1:]],
                dtype=logits.dtype, device=logits.device,
            )
            dist.all_gather_into_tensor(
                gathered, logits,
                group=self.comm_manager.get_group("lmhead_tp_group"),
            )
            bsz = logits.shape[0]
            logits = gathered.view(tp, bsz, *logits.shape[1:]).permute(1, 2, 0, 3).reshape(
                bsz, logits.shape[1], -1
            )
        # The padded rows are implementation-only rows and must never be
        # visible to the sampler.
        logits = logits[..., :self.lm_head_raw_vocab_size]
        return logits

    def process_weights_after_loading(self):
        # Initialize absorb weights BEFORE transpose/NZ conversion.
        # init_absorb_weights expects kv_b_proj.weight in its original
        # (out_features, in_features) layout. After quant_method processing,
        # the weight gets transposed and cast to NZ format, making it
        # difficult to recover the correct data layout.
        for module in self.modules():
            if isinstance(module, LongcatFlashMLA):
                module.init_absorb_weights()

        # enable_weight_nz is a top-level ModelConfig field
        # (yaml: model_config.enable_weight_nz), not a custom_params entry.
        enable_weight_nz = self.infer_config.model_config.enable_weight_nz
        # W8A8 dense MLP needs fp32 scales for npu_dequant_swiglu_quant: cast
        # weight_scale on gate_up_proj and smooth_scale on down_proj.
        # FusedMoEGMM is special-cased — its smooth_scales feed
        # npu_dynamic_quant / dispatch_v2 which expect bf16/fp16, so we must
        # NOT force them to fp32. Mirrors deepseek-r1 / qwen3-moe convention.
        float_scale_modules = ("gate_up_proj",)
        float_smooth_scale_modules = ("down_proj",)
        for name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            if isinstance(module, FusedMoEGMM):
                quant_method.process_weights_after_loading(
                    module, is_transpose=True, is_nz=enable_weight_nz,
                )
                continue
            scales_dtype = {}
            if any(s in name for s in float_scale_modules):
                scales_dtype["scale_dtype"] = torch.float
            if any(s in name for s in float_smooth_scale_modules):
                scales_dtype["smooth_scale_dtype"] = torch.float
            try:
                quant_method.process_weights_after_loading(
                    module, is_nz=enable_weight_nz, scales_dtype=scales_dtype,
                )
            except TypeError:
                # UnquantizedLinearMethod (and other quant methods that don't
                # take scales_dtype) — fall back to the no-cast path.
                quant_method.process_weights_after_loading(module, is_nz=enable_weight_nz)

        # Build prolog NZ-format weights and mark absorb weights as static
        # for graph mode.
        for module in self.modules():
            if isinstance(module, LongcatFlashMLA):
                module.prepare_prolog_weights(enable_nz=enable_weight_nz)
                # frozen tensors used by graph compile
                if module.kv_b_proj_w_k is not None:
                    _mark_static(module.kv_b_proj_w_k)
                if module.kv_b_proj_w_v is not None:
                    _mark_static(module.kv_b_proj_w_v)
                if module.weight_dq_prolog is not None:
                    _mark_static(module.weight_dq_prolog)
                if module.weight_dkv_kr_prolog is not None:
                    _mark_static(module.weight_dkv_kr_prolog)

        # MLA must stay BF16 (npu_mla_prolog_v3 has no W8A8 inputs); fail at
        # load time if the W8A8 ignore set misses any MLA projection.
        for name, module in self.named_modules():
            if not isinstance(module, LongcatFlashMLA):
                continue
            for proj in ("q_b_proj", "kv_b_proj", "o_proj"):
                qm = getattr(getattr(module, proj), "quant_method", None)
                if not isinstance(qm, UnquantizedLinearMethod):
                    raise RuntimeError(
                        f"{name}.{proj} resolved to {type(qm).__name__}, expected "
                        f"UnquantizedLinearMethod — check the W8A8 checkpoint's "
                        f"quantization_config.ignore covers all MLA projections"
                    )

        # Eagerly prepare MC2 dispatch_v2/combine_v2 kwargs for every MoE module
        # that will use them. Doing this here — after HCCL groups are up but
        # before the first forward — avoids paying the kwargs-build cost inside
        # the traced decode region (where it would otherwise fire a graph guard
        # miss on the first decode step).
        for module in self.modules():
            if not isinstance(module, LongcatFlashMoE):
                continue
            if not getattr(module, "use_ep", False):
                continue
            if not getattr(module, "enable_moe_mc2_dispatch", False):
                continue
            if module.dispatch_kwargs is not None:
                continue
            module.set_mc2_kwargs()

        # Resize ngram rolling-window buffer to actual batch_size now that the model
        # is on its target device. Cache tensors themselves are allocated later by
        # the framework via cache_entries / tensor_setter.
        try:
            batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        except AttributeError:
            batch_size = 1
        device = next(self.parameters()).device
        ngram_emb = getattr(self.model, "ngram_embeddings", None)
        if ngram_emb is not None and hasattr(ngram_emb, "init_ngram_cache"):
            ngram_emb.init_ngram_cache(batch_size, device=device)

    def get_cache_info(self) -> ModelCacheInfo:
        """Expose KV cache layout so the framework can allocate paged storage.

        Each MLA sub-layer registers two CacheEntry objects (cache_nope and
        cache_rope) sharing `attn_type='FullAttention'`. The framework iterates
        ModelCacheInfo, sizes the block pool, and binds the allocated tensors
        back via each entry's `tensor_setter`.
        """
        # LongCat-Lite has 2 MLA sublayers per LongcatFlashDecoderLayer (dual
        # MLA, ModuleList of 2). We flatten to 28 cache entries spanning
        # 14 logical layers × 2 sublayers — the flat layer_idx becomes the
        # canonical key the framework hashes for PD prefix-cache lookup.
        layer_infos = []
        layer_idx = 0
        for layer in self.model.layers:
            for mla in layer.self_attn:
                layer_infos.append(
                    LayerCacheInfo(layer_idx=layer_idx, caches=list(mla.cache_entries))
                )
                layer_idx += 1
        return ModelCacheInfo(
            num_layers=layer_idx,
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        # Dense MLP: gate_proj/up_proj -> gate_up_proj (MergedColumnParallelLinear)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # MoE experts: per-expert gate_proj/up_proj/down_proj -> FusedMoEGMM w13/w2
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set = set()

        for name, loaded_weight in weights:
            if not ENABLE_NGRAM_EMBEDDING and ".ngram_embeddings." in name:
                continue

            # Official LongCat-Next checkpoints store lm_head with the text
            # vocabulary (131125 rows), while the multimodal embedding table
            # is larger (config.vocab_size).  Pad only the transient loaded
            # tensor so ColumnParallelLinear can shard it evenly.
            if name.endswith("lm_head.weight"):
                if loaded_weight.shape[0] > self.lm_head_vocab_size:
                    raise RuntimeError(
                        f"lm_head checkpoint rows {loaded_weight.shape[0]} exceed "
                        f"configured rows {self.lm_head_vocab_size}"
                    )
                if loaded_weight.shape[0] < self.lm_head_vocab_size:
                    padded_weight = torch.zeros(
                        (self.lm_head_vocab_size, loaded_weight.shape[1]),
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    )
                    padded_weight[:loaded_weight.shape[0]].copy_(loaded_weight)
                    loaded_weight = padded_weight

            if ".ngram_embeddings.embedders." in name and name.endswith(".weight"):
                if name not in params_dict:
                    continue
                parts = name.split(".")
                embedder_idx = int(parts[parts.index("embedders") + 1])
                padded_vocab_size = self.model.ngram_embeddings.embedder_padded_vocab_sizes[embedder_idx]
                if loaded_weight.shape[0] < padded_vocab_size:
                    padded_weight = torch.zeros(
                        (padded_vocab_size, loaded_weight.shape[1]),
                        dtype=loaded_weight.dtype,
                        device=loaded_weight.device,
                    )
                    padded_weight[:loaded_weight.shape[0]].copy_(loaded_weight)
                    loaded_weight = padded_weight

            # Dense MLP stacked params (gate_proj/up_proj -> gate_up_proj)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip MoE expert weights (handled below)
                if "mlp.experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # MoE expert params (per-expert -> FusedMoEGMM w13/w2)
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, name,
                                  shard_id=shard_id, expert_id=expert_id)
                    break
                else:
                    # Standard params (parallel layers, buffers, etc.)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


__all__ = ["LongcatNextForCausalLM"]


# ===========================================================================
# Multimodal generation (copied from the official LongCat-Next release, which
# ships next to the weights; Copyright (c) 2026 Meituan, MIT License).
# Local changes: the backbone base classes of the
# official inheritance chain are gone, so the generation model derives from
# transformers directly; the three dataclasses, NgramCache and the N-gram
# prefill shift are carried here so this file is the only model definition.
# ===========================================================================


logger = logging.getLogger(__name__)


@dataclass
class LongcatNextForCausalLMOutputWithPast(CausalLMOutputWithPast):
    visual_loss: Optional[torch.FloatTensor] = None
    visual_logits: Optional[torch.FloatTensor] = None
    visual_ids: Optional[torch.LongTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    audio_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    visual_ids: Optional[torch.LongTensor] = None
    audio_ids: Optional[torch.LongTensor] = None
    audio_text_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerateEncoderDecoderOutput(GenerateEncoderDecoderOutput):
    visual_ids: Optional[torch.LongTensor] = None
    audio_ids: Optional[torch.LongTensor] = None
    audio_text_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerationStatus:
    mode: str = "text"
    current_image_token_num: int = -1
    audio_parallel_decoding: bool = False
    is_audio_text_end: bool = False
    is_audio_start: bool = False
    last_step_mode: str = None

    def __init__(self, visual_generation_config, audio_generation_config):
        self.visual_generation_config = visual_generation_config
        self.h = self.visual_generation_config.custom_params["token_h"]
        self.w = self.visual_generation_config.custom_params["token_w"]
        self.anyres_prefix = self.visual_generation_config.custom_params["anyres_prefix"].format(h=self.h, w=self.w)
        self.audio_generation_config = audio_generation_config
        self.audio_parallel_decoding = audio_generation_config.audio_parallel_decoding
        # Keep the multimodal route after media placeholders enter the sequence.
        self.has_placeholder = False

    def switch_to(self, modal):
        if not (modal in ["text", "visual", "audio"]):
            raise AssertionError
        self.mode = modal
        if modal in ("visual", "audio"):
            self.has_placeholder = True
        self.current_image_token_num = 0 if modal == "visual" else -1
        self.is_audio_text_end = False
        self.is_audio_start = False

    @property
    def is_img_newline(self):
        return ((self.current_image_token_num + 1) % (self.w + 1)) == 0 and not self.is_img_end

    @property
    def is_img_end(self):
        return (self.current_image_token_num + 1) / (self.w + 1) == self.h


class NgramCache(DynamicCache):
    """
    Extended DynamicCache for storing N-gram context alongside KV cache.
    """
    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        # Keep only n-1 tokens (minimum needed for N-gram computation)
        self.max_context_len = config.emb_neighbor_num - 1
        self.oe_ignored_token_ids = torch.tensor(config.oe_ignored_token_ids, dtype=torch.long)


    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        """
        Update N-gram context with window management.

        Args:
            new_tokens: New tokens to append, shape (batch_size, seq_len)
        """
        new_tokens = new_tokens.clone()
        new_tokens[torch.isin(new_tokens, self.oe_ignored_token_ids.to(new_tokens.device))] = 0

        if self.ngram_context is None:
            self.ngram_context = new_tokens
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)

        # Truncate to maintain constant memory footprint
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len:]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        """Reorder cache for beam search."""
        # Reorder parent's KV cache
        super().reorder_cache(beam_idx)

        # Reorder N-gram context
        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(0, beam_idx.to(self.ngram_context.device))

        return self


def ngram_prefill_shift(tensor: torch.Tensor, n: int, eos_token_id: int = 2) -> torch.Tensor:
    p, q = tensor.shape
    # special_token / modal set 0
    special_tokens = 0

    if n == 0:
        return tensor.clone()

    if n >= q:
        return torch.zeros_like(tensor)

    # Find all special_token/modal/EOS locations
    special_mask = (tensor == special_tokens)
    total_mask = (tensor == eos_token_id) | special_mask

    pos = torch.arange(q, device=tensor.device).unsqueeze(0).expand(p, q)

    # NPU adaptation: segment start per position via cummax;
    # scatter_reduce(amin) falls back to CPU on NPU. Segment boundary = total_mask,
    # segment start = the position right after the most recent boundary.
    is_reset_at_prev = torch.cat([
        torch.zeros(p, 1, dtype=torch.bool, device=tensor.device),
        total_mask[:, :-1]
    ], dim=1)
    candidates = torch.where(is_reset_at_prev, pos, torch.full_like(pos, -1))
    segment_start_per_pos = candidates.cummax(dim=1).values.clamp_min(0)

    # Calculate the offset of each position within the segment
    offset_in_segment = pos - segment_start_per_pos

    # Data for each position should be taken from the position offset -n within the segment
    source_offset = offset_in_segment - n
    valid_mask = source_offset >= 0

    # Calculate the actual source index
    source_indices = segment_start_per_pos + torch.clamp(source_offset, min=0)

    # Data is collected by source_indices
    result = torch.gather(tensor, 1, source_indices)

    # Set invalid position to zero
    result = result * valid_mask * (~special_mask)

    return result


class LongcatNextMultimodalModel(nn.Module):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatNextConfig

    def _init_multimodal_constants(self, config):
        name2id_dict = {
            "image_newline_token_id": self.config.visual_config.image_newline_token_id,
            "image_end_token_id": self.config.visual_config.image_end_token_id,
            "image_pad_token_id": self.config.visual_config.image_pad_token_id,
            "audiotext_start_token_id": config.audio_config.audiotext_start_token_id,
            "audiotext_pad_token_id": self.config.audio_config.audiotext_pad_token_id,
            "audiogen_end_token_id": config.audio_config.audiogen_end_token_id,
            "audio_pad_token_id": self.config.audio_config.audio_pad_token_id,
        }
        for k, v in name2id_dict.items():
            self.register_buffer(k, torch.tensor([v], dtype=torch.long), persistent=False)
        visual_offset_list = [config.visual_offset] + config.visual_config.vq_config.codebook_sizes[:-1]
        visual_offset_vals = torch.cumsum(torch.tensor(visual_offset_list, dtype=torch.long), dim=0)
        self.register_buffer("visual_offset_vals", visual_offset_vals, persistent=False)
        audio_offset_list = [config.audio_offset] + config.audio_config.vq_config.codebook_sizes[:-1]
        audio_offset_vals = torch.cumsum(torch.tensor(audio_offset_list, dtype=torch.long), dim=0)
        self.register_buffer("audio_offset_vals", audio_offset_vals, persistent=False)
        logger.debug("visual_offset_vals=%s", self.visual_offset_vals)
        logger.debug("audio_offset_vals=%s", self.audio_offset_vals)

    def forward(  # pylint: disable=huawei-too-many-arguments
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        visual_inputs=None,
        visual_ids=None,
        audio_inputs=None,
        audio_ids=None,
        audio_text_ids=None,
        multimodal_generation_status=None,
        **kwargs
    ) -> BaseModelOutputWithPast:

        if input_ids is None:
            raise ValueError("You must specify input_ids")

        # Extract N-gram context if available
        ngram_context = None
        if isinstance(past_key_values, NgramCache) and past_key_values.ngram_context is not None:
            ngram_context = past_key_values.ngram_context

        # assert input_ids.size(0) == 1, "only support bs=1 for now" # but when
        # bs=2, idx=1 is for uncond_image_generation
        special_visual_mask, special_audio_mask, special_audio_text_start_mask, \
            special_audio_text_pad_mask = self.get_placeholder_mask(input_ids[:1])  # seq-dim

        if inputs_embeds is None:
            input_ids[:, special_visual_mask | special_audio_mask |
                      special_audio_text_pad_mask | special_audio_text_start_mask] = 0
            filled_text_pad_mask = torch.ones_like(special_audio_mask)
            audio_text_position_mask = (special_audio_text_pad_mask |
                                        special_audio_text_start_mask | special_audio_mask)

            if audio_text_ids is not None and audio_text_ids.size(1) > 0 and audio_text_position_mask.sum() > 0:
                filled_text = audio_text_ids[:, -audio_text_position_mask.sum():]
                filled_text_pad_mask = (filled_text == self.config.audio_config.audiotext_pad_token_id)[0]
                input_ids[:, audio_text_position_mask] = filled_text
                input_ids[input_ids == self.config.audio_config.audiotext_pad_token_id] = 0

            inputs_embeds = self.ngram_embeddings(input_ids, ngram_context=ngram_context)
            inputs_embeds[:, (special_visual_mask | (special_audio_mask & filled_text_pad_mask))] = 0

        if special_audio_text_start_mask.sum() > 0:
            audio_text_start_embedding = self.embed_tokens(self.audiotext_start_token_id)
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_audio_text_start_mask] += audio_text_start_embedding
            else:
                inputs_embeds[:, special_audio_text_start_mask] += audio_text_start_embedding

        if visual_inputs is not None:
            visual_ids = self.get_visual_ids(**visual_inputs) # [<bs=1>*seq, lev]

        if visual_ids is not None and special_visual_mask.sum() > 0:
            visual_embeddings = self.get_visual_embeddings(visual_ids[-special_visual_mask.sum():]) # -> [seq, dim]
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_visual_mask] = visual_embeddings.to(inputs_embeds.device)
            else:
                inputs_embeds[:, special_visual_mask] = visual_embeddings.to(inputs_embeds.device)

        if audio_inputs is not None:
            audio_ids = self.get_audio_ids(**audio_inputs) # -> [<bs=1>*seq, lev]

        if audio_ids is not None and special_audio_mask.sum() > 0:
            audio_embeddings = self.get_audio_embeddings(audio_ids[-special_audio_mask.sum():]) # -> [seq, dim]
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_audio_mask] += audio_embeddings.to(inputs_embeds.device)
            else:
                inputs_embeds[:, special_audio_mask] += audio_embeddings.to(inputs_embeds.device)

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(config=self.config)

        # Update N-gram context
        if use_cache and isinstance(past_key_values, NgramCache):
            past_key_values.update_ngram_context(input_ids)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs
        )

    def get_visual_ids(self, pixel_values, visual_grid_thw, offset=True):
        visual_ids = self.visual_tokenizer.encode(pixel_values, visual_grid_thw)
        if offset:
            visual_ids += self.visual_offset_vals.to(visual_ids.device)
        return visual_ids

    def get_audio_ids(self, audio, encoder_length, bridge_length, offset=True):
        audio_ids = self.audio_tokenizer.encode(audio, encoder_length, bridge_length)
        if offset:
            audio_ids += self.audio_offset_vals.to(audio_ids.device)
        return audio_ids

    @torch.no_grad()
    def decode_visual_ids_and_save(
        self,
        visual_ids,
        save_prefix,
        token_h,
        token_w,
        **kwargs,
    ):
        visual_ids -= self.visual_offset_vals.to(visual_ids.device)

        if not (save_prefix.startswith("./") or save_prefix.startswith("/")):
            save_prefix = f"./{save_prefix}"
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        return self.visual_tokenizer.lazy_decode_and_save(visual_ids, token_h, token_w, f"{save_prefix}_{0}.png")

    @torch.no_grad()
    def decode_audio_ids_and_save(
        self,
        audio_ids,
        save_prefix,
        sampling_rate,
        wave_concat_overlap,
        **kwargs,
    ):
        audio_ids -= self.audio_offset_vals.to(audio_ids.device)

        if not (save_prefix.startswith("./") or save_prefix.startswith("/")):
            save_prefix = f"./{save_prefix}"
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        save_path = f"{save_prefix}_{0}.wav"
        self.audio_tokenizer.lazy_decode_and_save(audio_ids, sampling_rate, wave_concat_overlap, save_path)
        return [save_path]

    def get_visual_embeddings(self, visual_ids):
        visual_embeddings = self.embed_tokens(visual_ids).sum(dim=1) # [seq, lev] -> [seq, lev, dim] -> [seq, dim]
        visual_embeddings = self.visual_tokenizer.visual_embedding_layer(visual_embeddings)
        return visual_embeddings

    def get_audio_embeddings(self, audio_ids):
        audio_embeddings = self.embed_tokens(audio_ids).sum(dim=1)
        return audio_embeddings

    def get_placeholder_mask(self, input_ids: torch.LongTensor):
        special_image_mask = (input_ids == self.config.visual_config.image_pad_token_id).squeeze(0)
        special_audio_mask = (input_ids == self.config.audio_config.audio_pad_token_id).squeeze(0)
        special_audio_text_start_mask = (input_ids == self.config.audio_config.audiotext_start_token_id).squeeze(0)
        special_audio_text_pad_mask = (input_ids == self.config.audio_config.audiotext_pad_token_id).squeeze(0)
        return special_image_mask, special_audio_mask, special_audio_text_start_mask, special_audio_text_pad_mask


class LongcatNextForGeneration(PreTrainedModel, GenerationMixin):
    # Capabilities declared by the official base class this generation model used to
    # inherit (the NPU LongCat-Flash backend). PreTrainedModel validates the requested
    # attention implementation against them while constructing the model, so they must
    # stay declared now that the class derives from transformers directly.
    base_model_prefix = "model"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False

    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatNextConfig


    @can_return_tuple
    @auto_docstring
    def forward(  # pylint: disable=huawei-too-many-arguments
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        visual_inputs=None,
        visual_ids=None,
        audio_inputs=None,
        audio_ids=None,
        audio_text_ids=None,
        multimodal_generation_status: LongcatNextForCausalLMGenerationStatus = None,
        visual_generation_config: GenerationConfig = None,
        audio_generation_config: GenerationConfig = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        visual_inputs (`BatchFeature`, *optional*):
            Visual inputs returned by the processor, containing pixel values and grid metadata for image encoding.
        visual_ids (`torch.LongTensor` of shape `(num_visual_tokens, num_codebooks)`, *optional*):
            Quantized visual token ids from the visual tokenizer, used to build visual embeddings during generation.
        audio_inputs (`BatchFeature`, *optional*):
            Audio inputs returned by the processor, containing mel-spectrogram features and length metadata.
        audio_ids (`torch.LongTensor` of shape `(num_audio_tokens, num_codebooks)`, *optional*):
            Quantized audio token ids from the audio tokenizer, used to build audio embeddings during generation.
        audio_text_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token ids for the audio text transcript generated alongside audio tokens.
        multimodal_generation_status (`LongcatNextForCausalLMGenerationStatus`, *optional*):
            Stateful object tracking the current multimodal generation mode (text / visual / audio) and
            associated counters used to route logits to the correct head during auto-regressive decoding.
        visual_generation_config (`GenerationConfig`, *optional*):
            Generation configuration for the visual head, controlling sampling parameters such as
            `temperature`, `top_k`, `top_p`, and custom parameters like `cfg_scale` and `anyres_config`.
        audio_generation_config (`GenerationConfig`, *optional*):
            Generation configuration for the audio head, controlling sampling parameters such as
            `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `audio_parallel_decoding`.
        """

        if (multimodal_generation_status.mode == "visual" and visual_generation_config.custom_params["cfg_scale"] != 1.0
                and input_ids.size(0) == 1):
            input_ids = input_ids.repeat((2, 1))

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_inputs=visual_inputs,
            visual_ids=visual_ids,
            audio_inputs=audio_inputs,
            audio_ids=audio_ids,
            audio_text_ids=audio_text_ids,
            multimodal_generation_status=multimodal_generation_status,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        slice_hidden_states = hidden_states[:, slice_indices, :]

        loss, logits = None, None
        if multimodal_generation_status.mode == "visual" and \
            (not multimodal_generation_status.is_img_newline) and (not multimodal_generation_status.is_img_end):
            visual_ids = self.get_multimodal_logits_and_ids(
                self.visual_head, visual_ids, slice_hidden_states, visual_generation_config, "visual")
        else:
            logits = self.lm_head(slice_hidden_states)

        if multimodal_generation_status.mode == "audio" and multimodal_generation_status.is_audio_start:
            audio_ids = self.get_multimodal_logits_and_ids(
                self.audio_head, audio_ids, slice_hidden_states, audio_generation_config, "audio")

        return LongcatNextForCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            visual_ids=visual_ids,
            audio_ids=audio_ids,
        )

    def get_multimodal_logits_and_ids(
        self,
        head_model,
        multimodal_ids,
        hidden_states,
        generation_config,
        modality,
    ):
        """Sample one code group with ``head_model`` for ``modality``.

        ``modality`` selects the codebook sizes, the codebook offsets and the
        embedding table; all three come from this model, so callers pass only
        what changes from step to step.
        """
        modality_config = (self.config.visual_config if modality == "visual"
                           else self.config.audio_config)
        codebook_sizes = modality_config.vq_config.codebook_sizes
        offset_vals = (self.model.visual_offset_vals if modality == "visual"
                       else self.model.audio_offset_vals)
        multimodal_embedding_layer = self.model.embed_tokens
        next_token_ids = torch.zeros(
            hidden_states.size(0),
            len(codebook_sizes),
            dtype=torch.long,
            device=hidden_states.device)
        multimodal_embedding_layer = multimodal_embedding_layer.to(hidden_states.device)

        for level, _ in enumerate(codebook_sizes):
            logits = head_model(hidden_states, next_token_ids, multimodal_embedding_layer, level) # -> (bs, 1, dim)
            next_token_id = self.inner_sample(
                logits, multimodal_ids[None, :, level] - offset_vals[level], generation_config) # (bs, 1)
            next_token_id += offset_vals[level]
            next_token_ids[:, level] = next_token_id

        return next_token_ids[:1]

    def inner_sample(
        self,
        next_token_logits: torch.Tensor,
        multimodal_ids: torch.LongTensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        logits_processor = self._get_logits_processor(generation_config)

        if "cfg_scale" in generation_config.custom_params and generation_config.custom_params["cfg_scale"] != 1.0:
            cond_logits, uncond_logits = next_token_logits.chunk(2, dim=0)
            next_token_logits = generation_config.custom_params["cfg_scale"] * \
                (cond_logits - uncond_logits) + uncond_logits

        next_token_scores = logits_processor(multimodal_ids, next_token_logits.to(multimodal_ids.device))
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        return next_tokens

    @torch.no_grad()
    def generate(self, inputs=None, **kwargs):
        """Override to ensure NgramCache is used."""

        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = NgramCache(config=self.config)

        return super().generate(
            inputs=inputs,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        generation_config,
        attention_mask,
        cache_position,
        **kwargs,
    ):
        """Advance the multimodal state and build this step's model inputs.

        The multimodal containers travel through ``**kwargs``, the same channel
        HF uses for extra model kwargs, so the declared signature stays small.
        """
        visual_ids = kwargs.pop("visual_ids", None)
        audio_ids = kwargs.pop("audio_ids", None)
        audio_text_ids = kwargs.pop("audio_text_ids", None)
        multimodal_generation_status = kwargs.pop("multimodal_generation_status", None)
        if multimodal_generation_status is None:
            raise ValueError("prepare_inputs_for_generation requires multimodal_generation_status")
        extra_new_tokens = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)
        if visual_ids is None:
            visual_ids = torch.empty(0, len(self.config.visual_config.vq_config.codebook_sizes),
                                     dtype=torch.long, device=input_ids.device)
        if audio_ids is None:
            audio_ids = torch.empty(0, len(self.config.audio_config.vq_config.codebook_sizes),
                                    dtype=torch.long, device=input_ids.device)
        if audio_text_ids is None:
            audio_text_ids = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)

        def insert_ids(new_ids, _input_ids, _attention_mask, _cache_position, position=0):
            if position < 0:
                parts = [_input_ids[:, :position], new_ids, _input_ids[:, position:]]
            else:
                parts = [_input_ids, new_ids]
            _input_ids = torch.cat(parts, dim=1)
            insert_len = new_ids.size(1)
            _attention_mask = F.pad(_attention_mask, (0, insert_len), value=1)
            insert_position = _cache_position[-1] + 1 + torch.arange(insert_len, device=_cache_position.device)
            _cache_position = torch.cat([_cache_position, insert_position])
            return _input_ids, _attention_mask, _cache_position

        # multimodal generation status change
        if cache_position[0] != 0:
            multimodal_generation_status.last_step_mode = multimodal_generation_status.mode

        if multimodal_generation_status.mode == "visual":
            multimodal_generation_status.current_image_token_num += 1

        if (input_ids[:, -1] == self.config.visual_config.image_start_token_id).all():
            multimodal_generation_status.switch_to("visual")
            anyres_prefix_ids = self.text_tokenizer.encode(
                multimodal_generation_status.anyres_prefix, return_tensors="pt")
            anyres_prefix_ids = anyres_prefix_ids.to(input_ids.device)
            extra_new_tokens = torch.cat([extra_new_tokens, anyres_prefix_ids], dim=1)
            input_ids, attention_mask, cache_position = insert_ids(
                anyres_prefix_ids, input_ids, attention_mask, cache_position, position=-1)
            if input_ids.size(0) == 1: # cfg, change bs=1 -> 2
                input_ids = input_ids.repeat((2, 1))
                input_ids[1, :-(anyres_prefix_ids.size(-1) + 1)] = 0
                logger.debug("Enabled visual CFG with input_ids shape=%s", tuple(input_ids.shape))
                attention_mask = attention_mask.repeat((2, 1))

        elif (input_ids[:, -1] == self.config.audio_config.audiogen_start_token_id).all():
            multimodal_generation_status.switch_to("audio")

        elif (input_ids[:, -1] == self.config.audio_config.audiotext_start_token_id).all():
            multimodal_generation_status.is_audio_start = True

        elif ((input_ids[:, -1] == self.config.visual_config.image_end_token_id)
              | (input_ids[:, -1] == self.config.audio_config.audiogen_end_token_id)).all():
            multimodal_generation_status.switch_to("text")

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            visual_ids=visual_ids,
            audio_ids=audio_ids,
            audio_text_ids=audio_text_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        if model_inputs["cache_position"][0] != 0:
            model_inputs["visual_inputs"] = None
            model_inputs["audio_inputs"] = None

        return model_inputs, multimodal_generation_status, extra_new_tokens

    def _sample(  # pylint: disable=huawei-too-many-arguments
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        visual_ids=None,
        audio_ids=None,
        audio_text_ids=None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor  # pylint: disable=protected-access
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":  # pylint: disable=protected-access
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning(
                        "When using Flash Attention 2 and a static cache, "
                        "you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        visual_generation_config = GenerationConfig(**generation_config.visual_generation_config)
        audio_generation_config = GenerationConfig(**generation_config.audio_generation_config)
        multimodal_generation_status = LongcatNextForCausalLMGenerationStatus(
            visual_generation_config, audio_generation_config)

        # Validate request-wide invariants once; decode only reads this Python
        # flag instead of synchronizing NPU reductions at every token.
        placeholder_mask = self.model.get_placeholder_mask(input_ids[:1])
        multimodal_generation_status.has_placeholder = bool(
            torch.stack(placeholder_mask).any().item())
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask.bool().all().item()):
            raise ValueError("Multimodal text requires an unpadded request")

        pbar = tqdm(iter(int, 1), desc="Generating", unit="tok")
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs, multimodal_generation_status, extra_new_tokens = self.prepare_inputs_for_generation(
                input_ids,
                generation_config,
                visual_ids=visual_ids,
                audio_ids=audio_ids,
                audio_text_ids=audio_text_ids,
                multimodal_generation_status=multimodal_generation_status,
                **model_kwargs,
            )
            if extra_new_tokens.size(1) > 0:
                input_ids = torch.cat([input_ids[:, :-1], extra_new_tokens, input_ids[:, -1:]], dim=1)
                model_kwargs["attention_mask"] = model_inputs["attention_mask"]
                model_kwargs["cache_position"] = model_inputs["cache_position"]

            if multimodal_generation_status.mode == "text" and multimodal_generation_status.last_step_mode == "visual":
                next_tokens = generation_config._eos_token_tensor  # pylint: disable=protected-access
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                break

            visual_ids = model_inputs["visual_ids"]
            audio_ids = model_inputs["audio_ids"]
            audio_text_ids = model_inputs["audio_text_ids"]

            if is_prefill:
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    multimodal_generation_status=multimodal_generation_status,
                    visual_generation_config=visual_generation_config,
                    audio_generation_config=audio_generation_config)
                is_prefill = False
            else:
                outputs = model_forward(
                    **model_inputs,
                    return_dict=True,
                    multimodal_generation_status=multimodal_generation_status,
                    visual_generation_config=visual_generation_config,
                    audio_generation_config=audio_generation_config)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=1,
            )
            if synced_gpus and this_peer_finished:
                continue


            # multimodal generation
            if multimodal_generation_status.mode == "text" or \
                    (multimodal_generation_status.mode == "audio"
                     and not multimodal_generation_status.is_audio_text_end):
                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large
                # for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # Known issue (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']".
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # audio_text_ids done
                if multimodal_generation_status.mode == "audio" and (
                        next_tokens == self.config.audio_config.audiotext_pad_token_id).all():
                    multimodal_generation_status.is_audio_text_end = True

            elif multimodal_generation_status.mode == "visual":
                if multimodal_generation_status.is_img_end:
                    next_tokens = self.model.image_end_token_id.to(input_ids.device)

                elif multimodal_generation_status.is_img_newline:
                    next_tokens = self.model.image_newline_token_id.to(input_ids.device)

                else:
                    visual_ids = torch.cat([visual_ids, outputs.visual_ids], dim=0) # [seq, lev]
                    next_tokens = self.model.image_pad_token_id.to(input_ids.device)

            else: # mode == "audio" and multimodal_generation_status.is_audio_text_end
                next_tokens = self.model.audio_pad_token_id.to(input_ids.device)


            if multimodal_generation_status.mode == "audio":
                # audio_text_ids update
                audio_text_next_tokens = self.model.audiotext_pad_token_id.to(input_ids.device)
                if not multimodal_generation_status.is_audio_text_end:
                    audio_text_next_tokens, next_tokens = next_tokens, audio_text_next_tokens
                audio_text_ids = torch.cat((audio_text_ids, audio_text_next_tokens[:, None]), dim=1)

                # audio_ids update
                if multimodal_generation_status.is_audio_start:
                    if outputs.audio_ids[-1, 0] == (self.model.audio_offset_vals[1]): # offset + (level_1_len)
                        next_tokens = self.model.audiogen_end_token_id.to(input_ids.device)
                    else:
                        next_tokens = self.model.audio_pad_token_id.to(input_ids.device)
                    audio_ids = torch.cat([audio_ids, outputs.audio_ids], dim=0)

                elif (multimodal_generation_status.audio_parallel_decoding) or \
                        (not multimodal_generation_status.audio_parallel_decoding
                         and multimodal_generation_status.is_audio_text_end):
                    next_tokens = self.model.audiotext_start_token_id.to(input_ids.device)


            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # streaming mm ids are not supported yet
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            pbar.update(1)
            pbar.set_postfix({
                "recent_5toks": f"{input_ids[:, -5:].tolist()}",
            })

        pbar.close()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return LongcatNextForCausalLMGenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    visual_ids=visual_ids,
                    audio_ids=audio_ids,
                    audio_text_ids=audio_text_ids,
                )
            else:
                return LongcatNextForCausalLMGenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    visual_ids=visual_ids,
                    audio_ids=audio_ids,
                    audio_text_ids=audio_text_ids,
                )
        else:
            return input_ids, visual_ids, audio_ids, audio_text_ids

__all__ = ["LongcatNextForCausalLM", "LongcatNextMultimodalModel", "LongcatNextForGeneration",
           "LongcatNextForCausalLMOutputWithPast", "NgramCache", "ngram_prefill_shift"]
