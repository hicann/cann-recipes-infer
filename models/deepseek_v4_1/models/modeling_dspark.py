# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-DSpark/tree/main
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2023 DeepSeek. All rights reserved.
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

"""DeepSeek-V4.1 DSpark proposal model.

The target model supplies selected hidden states, DSpark proposes a token block,
and the target model verifies that block before the next proposal step.
"""

import json
import os
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, Iterable, NamedTuple, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import cann_ops_transformer
import cann_ops_nn
from torch import nn
from transformers.utils import logging

from executor.core.config import CommManager, InferenceConfig, PlatformVersion
from executor.core.engine.sampler import Sampler
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils.forward_metadata import get_forward_metadata
from executor.utils.stream_utils import npu_stream_switch, record_event, wait_event, record_stream
from executor.model_loader.weight_utils import default_weight_loader
from module.fuse_moe_gmm import FusedMoEGMM
from module.linear import ColumnParallelLinear, ReplicatedLinear, VocabParallelEmbedding

from .configuration_deepseek import DeepseekV41Config
from .modeling_deepseek import (
    Attention,
    DeepseekV41DecoderLayer,
    DeepseekV41Model,
    DeepseekV41ForCausalLM,
    KVCache,
    get_max_position_embeddings
)
from .modules import DeepseekV41RMSNorm, _init_rope
from .modules import PACKED_KV_COMPUTE_DTYPE, get_kv_cache_dim
from .modules.registry import OpKernel
from .modules.op_impls.mhc import make_identity_pre_mix, hc_pre_mix
from executor.utils import weight_dequant, calc_moe_hccl_buffer_size


logger = logging.get_logger(__name__)


def sample(
    logits: torch.Tensor,
    sampling_params: Dict[str, torch.Tensor | bool],
    sample_noise: Optional[torch.Tensor] = None,
):
    """Sample with the same distribution transform used for verification q."""
    probs = Sampler.logits_to_probs(logits.unsqueeze(1), sampling_params).squeeze(1)
    if sample_noise is None:
        return probs.argmax(dim=-1)
    return probs.div_(sample_noise).argmax(dim=-1)

def resolve_dspark_proposal_num_layers(config: DeepseekV41Config, infer_config: InferenceConfig) -> int:
    """Resolve proposal depth from the draft checkpoint configuration.

    A standalone draft checkpoint may expose ``n_mtp_layers`` in its own model
    config. The official integrated DeepSeek-V4.1 DSpark checkpoint instead keeps
    target and proposal weights in one directory and stores this metadata in
    ``inference/config.json``.
    """
    proposal_num_layers = getattr(config, "num_nextn_predict_layers", None)

    try:
        proposal_num_layers = int(proposal_num_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DeepSeek-V4.1 DSpark proposal config must define an integer n_mtp_layers."
        ) from exc
    if proposal_num_layers <= 0:
        raise ValueError(
            "DeepSeek-V4.1 DSpark proposal config requires n_mtp_layers greater than 0."
        )
    return proposal_num_layers


class DSparkSamplingContext(NamedTuple):
    """Sampling metadata and request-local noise for one draft block."""

    params: Optional[Dict[str, torch.Tensor | bool]]
    noise: Optional[torch.Tensor]


def _slice_rope_tensor(tensor: torch.Tensor, bsz: int, total_len: int, start: int, end: int):
    return tensor.view(bsz, total_len, *tensor.shape[1:])[:, start:end].flatten(0, 1).contiguous()


class DSparkAttention(Attention):
    """
    DSpark attention follows the official proposal cache contract:
    projected main-model hidden states write the verified-token/window KV,
    while draft tokens attend over that window plus the block-local KV.
    """
    def __init__(self, config: DeepseekV41Config, infer_config: InferenceConfig, layer_idx: Optional[int] = None,
                 prefix: Optional[str] = "", comm_manager: CommManager = None,
                 dspark_stage_idx: int = 0, **kwargs):
        # Each stage uses the native MTP attention role; prefix and stage index
        # keep its weights and framework-managed KV cache independent.
        proposal_layer_idx = config.num_hidden_layers + dspark_stage_idx
        # Proposal stages use win cache layout. The target config only lists
        # native decoder layers, so extend its per-layer attention metadata for
        # the DSpark indices before delegating construction to Attention.
        required_layers = proposal_layer_idx + 1
        if len(config.compress_ratios) < required_layers:
            config.compress_ratios.extend([0] * (required_layers - len(config.compress_ratios)))
            config.attention_types.extend(["win"] * (required_layers - len(config.attention_types)))
        super().__init__(
            config,
            infer_config,
            comm_manager,
            proposal_layer_idx,
            prefix,
            **kwargs,
        )
        self.dspark_stage_idx = dspark_stage_idx

    def _apply_win_rope(self, tensor: torch.Tensor, cos_sin: Tuple):
        rope_input = tensor.unsqueeze(1)
        cos, sin = cos_sin
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            rope_input, cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        return tensor

    def _project_dspark_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        kv = self.wkv(x)
        kv = self.kv_norm(kv).view(-1, 1, self.head_dim)
        return self._apply_win_rope(kv, (cos, sin))

    def _project_dspark_q(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # The shared quantization path consumes packed [tokens, hidden] inputs.
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.q_num_heads, self.head_dim))
        return self._apply_win_rope(q, (cos, sin))

    def prepare_fa_kwargs(
        self,
        q: torch.Tensor,
        win_kv: torch.Tensor,
        cmp_kv: Optional[torch.Tensor],
        cmp_sparse_indices: Optional[torch.Tensor],
        attn_metadata: Dict,
        is_prefill: bool,
    ):
        win_key = "win_kv"
        attn_kwargs = {
            "q": q,
            "ori_kv": win_kv.view(torch.uint8),
            "ori_sparse_indices": attn_metadata["dspark_fused_fa_inputs"]["win_sparse_indices"],
            "ori_block_table": attn_metadata["block_table"][win_key],
            "cu_seqlens_q": attn_metadata["dspark_fused_fa_inputs"]["cu_seqlens_q"],
            "seqused_q": None,
            "seqused_ori_kv": None,
            "ori_topk_length": attn_metadata["dspark_fused_fa_inputs"]["win_topk_length"],
            "sinks": self.attn_sink.detach(),  # The DSL ABI requires a plain Tensor.
            "quant_mode": 1,
            "softmax_scale": self.softmax_scale,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
            "return_softmax_lse": False,
            "metadata": attn_metadata["dspark_fused_fa_inputs"]["metadata"],
        }
        return attn_kwargs

    def sparse_attn(self, q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill):
        if self.dspark_stage_idx == 0:
            wait_event(self.enable_multi_streams, attn_metadata["metadata_event"], 1)
        return super().sparse_attn(
            q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill
        )

    def forward(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ):
        is_prefill = kwargs.get("is_prefill", False)
        cos_sin = attn_metadata["cos_sin"]
        if is_prefill:
            # Prefill only seeds the proposal cache; token proposals start in decode.
            cos, sin = cos_sin["win"]
            main_kv = self._project_dspark_kv(main_x, cos, sin)
            cache_slots = attn_metadata["dspark_prefill_slot_mapping"]
            self.update_win_kv(main_kv, cache_slots, kv_cache.win_kv)
            return x

        rope_slices = attn_metadata["dspark_rope_slices"]
        main_cos, main_sin = rope_slices["main"]
        draft_cos, draft_sin = rope_slices["draft"]

        main_kv = self._project_dspark_kv(main_x, main_cos, main_sin)
        q = self._project_dspark_q(x, draft_cos, draft_sin)
        draft_kv = self._project_dspark_kv(x, draft_cos, draft_sin)

        pa_inputs = attn_metadata["dspark_pa_inputs"]
        self.update_win_kv(main_kv, pa_inputs["main_slot_mapping"], kv_cache.win_kv)
        self.update_win_kv(draft_kv, pa_inputs["draft_slot_mapping"], kv_cache.win_kv)

        attn_output = self.sparse_attn(q, kv_cache.win_kv, None, None, attn_metadata, False)

        output = self.attn_post(attn_output, attn_metadata, True)
        return output


class DSparkMarkovHead(nn.Module):
    """Add a token-conditioned low-rank vocabulary bias to each proposal step.

    The embedding follows the model TP group.  The vocabulary projection has
    an independent TP setting, controlled by
    ``speculative_config.markov_lmhead_tp_size``.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        infer_config: InferenceConfig,
        prefix: str,
        comm_manager: CommManager = None,
    ):
        super().__init__()
        self.comm_manager = comm_manager
        parallel_config = infer_config.parallel_config
        self.embed_tp_size = parallel_config.embed_tp_size
        self.attn_dp_size = parallel_config.attn_dp_size
        self.markov_lmhead_tp_size = int(
            infer_config.speculative_config.markov_lmhead_tp_size
        )
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.embed_tp_rank = (
            comm_manager.get_rank("embed_tp_group")
            if self.embed_tp_size > 1 else 0
        )
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            config.pad_token_id,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.embed_tp_rank,
        )
        self.markov_w2 = ColumnParallelLinear(
            config.dspark_markov_rank,
            config.vocab_size,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            tp_size=self.markov_lmhead_tp_size,
            tp_rank=(comm_manager.get_rank("markov_lmhead_tp_group")
                     if self.markov_lmhead_tp_size > 1 else 0),
            prefix=f"{prefix}.markov_w2",
        )

    def forward(self, token_ids: torch.Tensor):
        if self.embed_tp_size > 1:
            new_token_ids = token_ids - self.embed_tp_rank * self.vocab_size_per_rank
            mask = (new_token_ids >= 0) & (new_token_ids < self.vocab_size_per_rank)
            markov_embed = self.markov_w1(new_token_ids * mask) * mask.unsqueeze(-1)
            dist.all_reduce(markov_embed, group=self.comm_manager.get_group("embed_tp_group"))
        else:
            markov_embed = self.markov_w1(token_ids)

        if self.attn_dp_size == 1 or self.markov_lmhead_tp_size == 1:
            markov_hidden = markov_embed
        else:
            markov_hidden = torch.empty_like(markov_embed).repeat(self.markov_lmhead_tp_size, 1)
            dist.all_gather_into_tensor(
                markov_hidden,
                markov_embed,
                group=self.comm_manager.get_group("markov_lmhead_tp_group"),
            )

        logits = self.markov_w2(markov_hidden.float())
        if self.markov_lmhead_tp_size > 1:
            if self.attn_dp_size == 1:
                gathered_logits = torch.empty_like(logits).repeat(self.markov_lmhead_tp_size, 1)
                dist.all_gather_into_tensor(
                    gathered_logits,
                    logits,
                    group=self.comm_manager.get_group("markov_lmhead_tp_group"),
                )
            else:
                gathered_logits = torch.empty_like(logits).view(-1)
                dist.all_to_all_single(
                    gathered_logits,
                    logits.view(-1),
                    group=self.comm_manager.get_group("markov_lmhead_tp_group"),
                )
            logits = gathered_logits.reshape(
                self.markov_lmhead_tp_size, markov_embed.shape[0], -1
            ).permute(1, 0, 2).reshape(markov_embed.shape[0], self.vocab_size)
        return logits, markov_embed


class DSparkConfidenceHead(nn.Module):
    """Score the contiguous proposal prefix used by confidence-based truncation."""

    def __init__(self, input_dim: int, prefix: str):
        super().__init__()
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.proj",
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DeepseekV41DSparkProposalLayer(DeepseekV41DecoderLayer):
    """One DSpark proposal block: HC attention, HC MoE, and optional output heads."""

    def __init__(self, config: DeepseekV41Config, infer_config: InferenceConfig, stage_idx: int, **kwargs):
        proposal_layer_idx = config.num_hidden_layers + stage_idx
        required_layers = proposal_layer_idx + 1
        if len(config.compress_ratios) < required_layers:
            missing = required_layers - len(config.compress_ratios)
            config.compress_ratios.extend([0] * missing)
            config.attention_types.extend(["win"] * missing)
        layer_idx = config.num_hidden_layers + stage_idx
        child_kwargs = {k: v for k, v in kwargs.items() if k != "comm_manager"}
        super().__init__(
            config=config,
            infer_config=infer_config,
            comm_manager=kwargs.get("comm_manager"),
            layer_idx=layer_idx,
            prefix=f"mtp.{stage_idx}",
            **child_kwargs,
        )
        self.config = config
        self.infer_config = infer_config
        self.stage_idx = stage_idx
        self.prefix = f"mtp.{stage_idx}"
        self.dspark_num_layers = config.n_mtp_layers
        self.dspark_block_size = config.dspark_block_size
        self.dspark_markov_rank = config.dspark_markov_rank
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.attn = DSparkAttention(
            config=config,
            infer_config=self.infer_config,
            comm_manager=kwargs.get("comm_manager"),
            layer_idx=layer_idx,
            prefix=f"{self.prefix}.attn",
            dspark_stage_idx=stage_idx,
            **child_kwargs,
        )

        # Only the first block projects the target hidden states shared by all
        # proposal blocks. Output heads are owned by the final block.
        if stage_idx == 0:
            target_layer_num = len(config.dspark_target_layer_ids)
            self.main_proj = ReplicatedLinear(
                config.hidden_size * target_layer_num,
                config.hidden_size,
                bias=False,
                quant_config=config.quant_config,
                prefix=f"{self.prefix}.main_proj",
            )
            self.main_norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if stage_idx == self.dspark_num_layers - 1:
            self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                config, self.infer_config, prefix=f"{self.prefix}.markov_head",
                comm_manager=kwargs.get("comm_manager"))
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + self.dspark_markov_rank,
                prefix=f"{self.prefix}.confidence_head",
            )

    def project_main_hidden(self, main_hidden: torch.Tensor):
        return self.main_norm(self.main_proj(main_hidden))

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """Project target hidden states and build the fixed draft block."""
        main_x = self.project_main_hidden(main_hidden)
        draft_ids = input_ids.new_full(
            (input_ids.size(0), self.dspark_block_size),
            self.config.dspark_noise_token_id,
        )
        draft_ids[:, 0] = input_ids
        x = self.embed(draft_ids)
        return x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1), main_x

    def prefill_main_cache(self, main_x: torch.Tensor,
                           attn_metadata: Optional[Dict] = None,
                           kv_cache: Optional[KVCache] = None):
        self.attn(
            x=main_x,
            main_x=main_x,
            attn_metadata=attn_metadata,
            is_prefill=True,
            kv_cache=kv_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        main_x: torch.Tensor,
        pre_mix: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        kv_cache: Optional[KVCache] = None,
        image_mask: Optional[torch.Tensor] = None,
        prefill_moe_global_chunks: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        cur_topk_list = kwargs.get("cur_topk_list")
        residual = hidden_states

        hidden_states, post, comb, attn_pre = OpKernel.hc_pre(hidden_states, pre_mix,
                                                                self.hc_attn_fn, self.hc_attn_scale,
                                                                self.hc_attn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                                self.norm_eps, self.hc_eps)

        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            x=hidden_states,
            main_x=main_x,
            attn_metadata=attn_metadata,
            is_prefill=False,
            kv_cache=kv_cache,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb, ffn_pre = OpKernel.hc_pre(hidden_states, attn_pre, self.hc_ffn_fn, self.hc_ffn_scale,
                                                    self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                    self.norm_eps, self.hc_eps)

        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states,
            is_prefill=False,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
            image_mask=image_mask,
            shared_expert_stream=attn_metadata.get('shared_expert_stream', None),
            prefill_moe_global_chunks=prefill_moe_global_chunks,
            moe_events=attn_metadata.get("moe_events"),
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        return hidden_states, ffn_pre

    def forward_proposal_head(
        self,
        logits: torch.Tensor,
        confidence_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        sampling_context: DSparkSamplingContext,
    ):
        """Generate the block autoregressively with Markov bias and confidence."""
        output_ids = input_ids.new_empty(input_ids.size(0), self.dspark_block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for idx in range(self.dspark_block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, idx])
            logits[:, idx].add_(logits_bias.float())
            markov_embeds.append(markov_embed)
            step_noise = None if sampling_context.noise is None else sampling_context.noise[:, idx]
            output_ids[:, idx + 1] = sample(
                logits[:, idx], sampling_params=sampling_context.params, sample_noise=step_noise)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(confidence_hidden, markov_embed)
        return output_ids, confidence


class DeepseekV41DSparkModel(DeepseekV41Model):
    """DSpark decoder that owns proposal stages and their paged KV caches."""

    def __init__(self, config: DeepseekV41Config, infer_config: InferenceConfig, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = kwargs.get("comm_manager")
        self.global_rank = kwargs.get("global_rank", 0)
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.embed_dp_size = infer_config.parallel_config.embed_dp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.world_size = infer_config.parallel_config.world_size
        self.cp_size = 1
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = config.vocab_size // self.embed_tp_size
        self.max_position_embeddings = get_max_position_embeddings(infer_config)
        self.block_size = infer_config.scheduler_config.block_size
        self.window_size = config.sliding_window
        self.rotary_emb = None
        self.compress_rotary_emb = None
        _init_rope(self)
        self.hc_mult = config.hc_mult
        self.embed_tokens = None
        self.layers = nn.ModuleList([
            DeepseekV41DSparkProposalLayer(config, infer_config, stage_idx, **kwargs)
            for stage_idx in range(config.n_mtp_layers)
        ])
        self.kv_cache = {
            stage_idx: KVCache(stage_idx)
            for stage_idx in range(config.n_mtp_layers)
        }
        self._init_model_cache_layout()

    def _init_model_cache_layout(self):
        """Register one framework-managed sliding-window cache per MTP stage.

        DSpark proposal attention only uses ``win_kv``.  Keep the cache entry
        next to its stage-local ``KVCache`` object so framework allocation is
        wired directly to the cache consumed by ``DSparkAttention``.
        """
        win_cache_dim = get_kv_cache_dim(self.config.head_dim)
        self.cache_entries_by_layer = {}
        for stage_idx, kv_cache in self.kv_cache.items():
            cache_entries = [
                CacheEntry(
                    cache_name=f"dspark_win_kv_layer{stage_idx}",
                    attn_type="SlidingWindow",
                    dim=win_cache_dim,
                    num_head=1,
                    dtype=PACKED_KV_COMPUTE_DTYPE,
                    needs_block=True,
                    block_size=self.block_size,
                    manager_key="win_kv",
                    tensor_setter=lambda tensor, cache=kv_cache: setattr(
                        cache, "win_kv", tensor),
                    sliding_window=self.window_size,
                )
            ]
            # Keep ownership and registration together on each stage cache.
            kv_cache.cache_entries = cache_entries
            self.cache_entries_by_layer[stage_idx] = cache_entries

    @property
    def mtp(self):
        return self.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        attn_metadata: dict,
        visual_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        main_next_tokens = kwargs.get("main_next_tokens")
        cur_topk_list = kwargs.get("cur_topk_list")

        if main_next_tokens is None:
            main_next_tokens = input_ids[:, :1] if input_ids.dim() > 1 else input_ids.view(-1, 1)

        draft_input_ids = input_ids
        if draft_input_ids.dim() == 1:
            draft_input_ids = draft_input_ids.view(main_next_tokens.shape[0], -1)
        batch_size, draft_seq_len = draft_input_ids.shape
        input_ids = main_next_tokens[:, :1].reshape(-1)

        main_x = self.layers[0].project_main_hidden(main_hidden)
        if main_x.dim() == 2:
            main_x = main_x.view(batch_size, -1, main_x.shape[-1])

        # Embedding kernels consume packed token rows; restore [B, S, H] only
        # for the shared RoPE construction below.
        hidden_states = self.calc_input_embeddings(draft_input_ids.reshape(-1), False)
        hidden_states = hidden_states.view(batch_size, draft_seq_len, -1)
        image_mask = None
        if visual_embeddings is not None:
                    hidden_states, image_mask = self.merge_visual_embeddings(
                        hidden_states, input_ids, visual_embeddings,
                    )
        cos_sin = self.generate_cos_sin(
            attn_metadata,
            torch.cat([main_x, hidden_states], dim=1),
            True
        )
        attn_metadata["cos_sin"] = cos_sin
        main_len = main_x.shape[1]
        draft_len = hidden_states.shape[1]
        total_len = main_len + draft_len
        main_cos = _slice_rope_tensor(cos_sin["win"][0], batch_size, total_len, 0, main_len)
        main_sin = _slice_rope_tensor(cos_sin["win"][1], batch_size, total_len, 0, main_len)
        draft_cos = _slice_rope_tensor(cos_sin["win"][0], batch_size, total_len, main_len, total_len)
        draft_sin = _slice_rope_tensor(cos_sin["win"][1], batch_size, total_len, main_len, total_len)
        draft_neg_sin = _slice_rope_tensor(
            cos_sin["win_neg_sin"], batch_size, total_len, main_len, total_len)
        attn_metadata["dspark_rope_slices"] = {
            "main": (main_cos, main_sin),
            "draft": (draft_cos, draft_sin),
            "draft_neg_sin": draft_neg_sin,
        }
        # mhc init one-hot pre_mix
        # MHC, attention, and MoE consume packed [T, H] rows.
        main_x = main_x.reshape(-1, main_x.shape[-1])
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        draft_input_ids = draft_input_ids.reshape(-1)
        if image_mask is not None:
            image_mask = image_mask.reshape(-1)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states, self.hc_mult)

        for stage_idx, layer in enumerate(self.layers):
            hidden_states, pre_mix = layer(
                hidden_states,
                main_x=main_x,
                pre_mix=pre_mix,
                attn_metadata=attn_metadata,
                kv_cache=self.kv_cache[stage_idx],
                cur_topk_list=cur_topk_list,
                input_ids=draft_input_ids,
                image_mask=image_mask,
                visual_embeddings=visual_embeddings
            )
        confidence_states = hc_pre_mix(hidden_states, pre_mix)
        last_mtp = self.mtp[-1]
        hidden_states = last_mtp.norm(confidence_states)
        return hidden_states, confidence_states

    def prefill_update_cache(self, main_hidden: torch.Tensor, attn_metadata: dict, **kwargs):
        main_x = self.layers[0].project_main_hidden(main_hidden)
        cos_sin = self.generate_cos_sin(attn_metadata, main_x, is_mtp=True)
        attn_metadata.update({'cos_sin': cos_sin})
        for stage_idx, layer in enumerate(self.layers):
            _ = layer.prefill_main_cache(
                main_x,
                attn_metadata=attn_metadata,
                kv_cache=self.kv_cache[stage_idx],
            )


class DeepseekV41DSparkProposalModel(DeepseekV41ForCausalLM):
    """
    DSpark proposal model entry.

    The infer pipeline calls ``propose`` after the main model produces verified
    hidden states. DSpark uses a block-level speculative head, which requires
    dedicated attention/cache kernels and DSpark weight mapping. Those kernels
    are intentionally not emulated with the existing token-by-token MTP path.
    """

    @staticmethod
    def update_model_cfg(config, infer_config: InferenceConfig):
        """Validate DSpark-only model metadata before draft construction."""
        config.n_mtp_layers = resolve_dspark_proposal_num_layers(config, infer_config)
        target_layer_ids = config.dspark_target_layer_ids
        next_n = infer_config.speculative_config.num_speculative_tokens
        if next_n != config.dspark_block_size:
            raise ValueError(
                f"DSpark requires num_speculative_tokens={next_n} equal to "
                f"dspark_block_size={config.dspark_block_size}."
            )
        if (isinstance(config.dspark_markov_rank, bool)
                or not isinstance(config.dspark_markov_rank, int)
                or config.dspark_markov_rank <= 0):
            raise ValueError("DSpark markov rank must be greater than 0.")
        if (isinstance(config.dspark_noise_token_id, bool)
                or not isinstance(config.dspark_noise_token_id, int)
                or not 0 <= config.dspark_noise_token_id < config.vocab_size):
            raise ValueError(
                "DSpark noise token id must be within the model vocabulary range."
            )
        markov_lmhead_tp_size = int(
            infer_config.speculative_config.markov_lmhead_tp_size
        )
        world_size = infer_config.parallel_config.world_size
        if markov_lmhead_tp_size <= 0 or world_size % markov_lmhead_tp_size != 0:
            raise ValueError(
                f"markov_lmhead_tp_size={markov_lmhead_tp_size} must be positive "
                f"and divide world_size={world_size}."
            )
        if config.vocab_size % markov_lmhead_tp_size != 0:
            raise ValueError(
                f"vocab_size={config.vocab_size} must be divisible by "
                f"markov_lmhead_tp_size={markov_lmhead_tp_size}."
            )

    def init_parallel_comm_group(self):
        super().init_parallel_comm_group()
        markov_lmhead_tp_size = int(
            self.infer_config.speculative_config.markov_lmhead_tp_size
        )
        if markov_lmhead_tp_size <= 0 or self.world_size % markov_lmhead_tp_size != 0:
            raise ValueError(
                f"markov_lmhead_tp_size={markov_lmhead_tp_size} must be positive "
                f"and divide world_size={self.world_size}."
            )
        if markov_lmhead_tp_size > 1:
            self.comm_manager.register_group(
                name="markov_lmhead_tp_group",
                group_num=self.world_size // markov_lmhead_tp_size,
                group_size=markov_lmhead_tp_size,
                platform_version=self.platform_version,
            )

        # create independent ep group for dspark megamoe and mc2
        platform_version = self.platform_version
        if not self.low_latency_tp:
            moe_ep_mc2_group_type = None if self.platform_version != PlatformVersion.ASCEND_950 else 3
            is_full_mesh_v2 = self.platform_version != PlatformVersion.ASCEND_950
            hccl_buffer_size = calc_moe_hccl_buffer_size(
                self.infer_config, self.config, is_full_mesh_v2=is_full_mesh_v2
            )
            self.comm_manager.register_group(
                name="dspark_moe_ep_group_mc2",
                group_num=self.moe_tp_size,
                group_size=self.moe_ep_size,
                group_stride=self.moe_tp_size,
                return_name=True,
                allow_physical_reuse=False,
                hccl_buffer_size=hccl_buffer_size,
                group_type=moe_ep_mc2_group_type,
                platform_version=platform_version,
            )
        if self.enable_mega_moe and self.moe_ep_size > 1:
            self.comm_manager.register_group(
                name="dspark_megamoe_ep_group",
                group_num=self.moe_tp_size,
                group_size=self.moe_ep_size,
                group_stride=self.moe_tp_size,
                return_name=True,
                allow_physical_reuse=False,
                platform_version=platform_version,
            )

    def __init__(self, config: DeepseekV41Config, infer_config: InferenceConfig, **kwargs):
        super().__init__(config, infer_config, comm_manager=kwargs.get("comm_manager"), is_mtp=True)
        self.dspark_block_size = config.dspark_block_size
        self.dspark_target_layer_ids = config.dspark_target_layer_ids
        self.dspark_noise_token_id = config.dspark_noise_token_id
        self.dspark_markov_rank = config.dspark_markov_rank
        self.dspark_num_layers = config.n_mtp_layers
        self.ignore_share_weight = True

        self.compiled_forward_spec_decode = None

        dspark_kwargs = {
            **kwargs,
            "global_rank": self.global_rank,
            "is_mtp": True,
            "comm_manager": self.comm_manager,
        }
        self.model = DeepseekV41DSparkModel(
            config, infer_config, **dspark_kwargs,
        )
        self.mtp = self.model.layers
        self.kv_cache = self.model.kv_cache
        self.lm_head = None
        self.top_k = config.dspark_n_activated_experts
        self.num_experts_per_tok = config.dspark_n_activated_experts
        self.num_experts = config.dspark_n_routed_experts

    def get_cache_info(
            self,
        ) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx in range(len(self.model.layers)):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(self.model.cache_entries_by_layer[layer_idx]),
                )
            )

        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    def forward_spec_decode_graph(self, input_ids: torch.LongTensor, main_hidden: torch.Tensor, **kwargs):
        return self._forward_spec_decode_impl(input_ids, main_hidden, kwargs)

    def prepare_proposal_inputs(self, proposal_inputs: Dict, context_inputs: Dict) -> Dict:
        """Convert worker tensors to V4 execution inputs outside timing and graphs.

        proposal_inputs contains input_ids [B, N], main_next_tokens [B, 1],
        main_hidden [B, S, H], target_hidden_positions [B, S] (-1 for padding),
        draft_positions [B, N], is_prefill and sampling/execution parameters.
        context_inputs contains flattened input_ids/position_ids, forward_metadata
        and the normalized proposal kv_len used by inherited preprocessing.

        Returns a new argument dictionary for propose; request state, sampling
        RNG and KV tensors are unchanged. Only model-specific metadata is built.
        """
        prepared = dict(proposal_inputs)
        target_hidden_positions = prepared.pop("target_hidden_positions")
        draft_positions = prepared.pop("draft_positions")
        model_inputs = dict(context_inputs)
        kv_len = model_inputs.pop("kv_len")
        is_prefill = prepared.get("is_prefill", False)
        forward_metadata = model_inputs["forward_metadata"]
        cp_metadata = forward_metadata.cp_metadata
        if is_prefill and cp_metadata is not None and cp_metadata.enabled:
            model_inputs["forward_metadata"] = self.tail_forward_metadata(
                forward_metadata, target_hidden_positions)
        model_inputs = self.preprocess_model_inputs(
            model_inputs, is_prefill=bool(is_prefill),
        )
        try:
            attn_metadata = model_inputs["attn_metadata"]
        except KeyError as exc:
            raise KeyError("DSpark requires attn_metadata from model preprocessing.") from exc
        block_tables = attn_metadata.get("block_table")
        if not isinstance(block_tables, dict) or "win_kv" not in block_tables:
            raise KeyError("DSpark requires framework block_table['win_kv'].")
        main_hidden = prepared.get("main_hidden")
        if not is_prefill and kv_len is not None:
            self._override_dspark_decode_attn_metadata(
                attn_metadata, kv_len, main_hidden.shape[0], draft_positions.device,
            )
        if is_prefill:
            prefill_context, main_hidden = self._prepare_prefill_cache_inputs(
                main_hidden, target_hidden_positions, attn_metadata, draft_positions,
            )
            prepared["prefill_context"] = prefill_context
        else:
            self._prepare_dspark_decode_shared_inputs(
                attn_metadata, target_hidden_positions, draft_positions,
            )
        prepared["main_hidden"] = main_hidden
        prepared["attn_metadata"] = attn_metadata
        return prepared

    @staticmethod
    def tail_forward_metadata(forward_metadata, target_hidden_positions):
        """Describe the tail window prefill CP hands over as the draft's own prefill batch."""
        lengths = forward_metadata.actual_seq_lengths_q
        tail_lens = (target_hidden_positions >= 0).sum(dim=1).to(
            device=lengths.device, dtype=lengths.dtype)
        return replace(
            forward_metadata,
            actual_seq_lengths_q=tail_lens,
            actual_seq_lengths_kv=tail_lens,
            actual_seq_lengths_cu_q=tail_lens.cumsum(0),
        )

    def _prepare_prefill_cache_inputs(self, main_hidden, target_hidden_positions, attn_metadata, draft_positions):
        """Prepare prompt-tail writes and the first proposal without touching KV tensors."""
        prefill_positions = target_hidden_positions.to(
            device=main_hidden.device, dtype=torch.int32).reshape(main_hidden.shape[0], -1)
        main_hidden, prefill_positions = self._select_prefill_context_tail(main_hidden, prefill_positions)
        block_tables = attn_metadata["block_table"]
        attn_metadata["dspark_prefill_slot_mapping"] = self._slot_mapping_from_positions(
            block_tables["win_kv"], prefill_positions, self.block_size,
        )
        prefill_attn_metadata = dict(attn_metadata)
        safe_prefill_positions = torch.where(
            prefill_positions >= 0, prefill_positions, torch.zeros_like(prefill_positions),
        )
        prefill_attn_metadata["position_ids"] = safe_prefill_positions
        prefill_attn_metadata["kv_len"] = safe_prefill_positions + 1
        prefill_attn_metadata["start_pos"] = safe_prefill_positions[:, 0].to(torch.int32)
        prefill_context = {
            "main_hidden": main_hidden,
            "target_hidden_positions": prefill_positions,
            "attn_metadata": prefill_attn_metadata,
        }
        # Prompt KV is written by the model; the first proposal must not write it again.
        proposal_hidden = main_hidden[:, :1].new_zeros(main_hidden.shape[0], 1, main_hidden.shape[-1])
        proposal_positions = prefill_positions[:, :1].new_full((main_hidden.shape[0], 1), -1)
        self._prepare_dspark_decode_shared_inputs(
            attn_metadata, proposal_positions, draft_positions,
        )
        return prefill_context, proposal_hidden

    def _select_prefill_context_tail(
        self,
        main_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep the prompt tail needed by SlidingWindow KV.

        The cache has a maximum sliding-window capacity, but prefill does not
        need to materialize that capacity when the prompt is shorter.  Use the
        actual maximum valid length in the batch, capped by ``window_size``;
        only shorter rows in a mixed-length batch retain row-local padding.
        """
        batch_size, seq_len = positions.shape
        window_size = self.window_size
        valid_len = (positions >= 0).sum(dim=1)
        # A dense batch needs one rectangular sequence length.  This is the
        # longest real prompt row, rather than the fixed cache capacity.
        target_len = min(window_size, int(valid_len.max().item()))
        if target_len == 0:
            raise ValueError("DSpark prefill requires at least one valid prompt position")
        offsets = torch.arange(target_len, device=positions.device).view(1, target_len)
        source_idx = (valid_len.view(batch_size, 1) - target_len + offsets).clamp(
            min=0, max=seq_len - 1)
        valid_tail = offsets >= (target_len - valid_len.clamp(max=target_len)).view(batch_size, 1)
        tail_positions = torch.gather(positions, 1, source_idx)
        tail_hidden = torch.gather(
            main_hidden,
            1,
            source_idx.unsqueeze(-1).expand(-1, -1, main_hidden.shape[-1]),
        )
        tail_positions = torch.where(
            valid_tail, tail_positions, tail_positions.new_full(tail_positions.shape, -1))
        tail_hidden = torch.where(
            valid_tail.unsqueeze(-1), tail_hidden, torch.zeros_like(tail_hidden))
        return tail_hidden, tail_positions

    def _override_dspark_decode_attn_metadata(
        self,
        attn_metadata: Dict,
        kv_len: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Build DSpark proposal metadata independently from main decode metadata.

        The framework decode metadata describes the current target-model query
        token. DSpark proposes a full block, so its attention metadata must keep
        the block-level q/k lengths used by the draft attention and lm-head
        postprocess path.
        """
        proposal_len = int(self.dspark_block_size) + 1
        kv_len = kv_len.to(device=device, dtype=torch.long).reshape(batch_size, -1)
        if kv_len.shape[1] >= proposal_len:
            position_ids = kv_len[:, :proposal_len]
        else:
            start = (kv_len[:, -1:] - 1).clamp_min(0)
            offsets = torch.arange(proposal_len, device=device, dtype=torch.long).view(1, proposal_len)
            position_ids = start + offsets

        seq_used_q = torch.full((batch_size,), proposal_len, device=device, dtype=torch.int32)
        actual_seq_q = torch.cumsum(seq_used_q, dim=0)
        cu_seq_lens_q = torch.cat([torch.zeros_like(actual_seq_q[:1]), actual_seq_q], dim=0)
        attn_metadata.update({
            "position_ids": position_ids.to(torch.int32),
            "kv_len": (position_ids + 1).to(torch.int32),
            "start_pos": position_ids[:, 0].to(torch.int32),
            "actual_seq_q": actual_seq_q.to(torch.int32),
            "actual_seq_k": (position_ids[:, -1] + 1).to(torch.int32),
            "cu_seq_lens_q": cu_seq_lens_q.to(torch.int32),
            "seq_used_q": seq_used_q,
        })

    def build_sliding_topk_ids(
        self,
        position_ids: torch.Tensor,
        prefix_len: int = 128,
    ) -> torch.Tensor:
        B, S = position_ids.shape
        device = position_ids.device
        dtype = torch.int32
        start_pos = position_ids[:, 0]
        offsets = torch.arange(-prefix_len, 0, device=device, dtype=dtype)
        prefix_pos = start_pos.unsqueeze(1) + offsets.unsqueeze(0) # [B, prefix_len]

        current_pos = position_ids # [B, S]
        row = torch.cat([prefix_pos, current_pos], dim=1) # [B, prefix_len + S]

        invalid = row < 0 # [B, L] bool
        order = invalid.to(torch.int8).argsort(dim=1, stable=True)
        row = row.gather(1, order)

        row = torch.where(row < 0, torch.full_like(row, -1), row).to(dtype)

        topk_ids = row.unsqueeze(1).expand(B, S, -1).contiguous()
        return topk_ids.view(B * S, -1)

    def _prepare_dspark_decode_shared_inputs(
        self,
        attn_metadata: Dict,
        target_hidden_positions: torch.Tensor,
        draft_positions: torch.Tensor,
    ) -> None:
        """Prepare dynamic inputs once for all proposal layers.

        This converts worker positions into PA slot mappings and fused-FA layout
        metadata before entering the compiled decode forward.
        """
        batch_size, draft_len = draft_positions.shape
        device = draft_positions.device
        main_positions = target_hidden_positions.to(device=device, dtype=torch.long)
        valid = main_positions >= 0
        safe_positions = torch.where(valid, main_positions, torch.zeros_like(main_positions))
        position_ids = torch.cat([safe_positions, draft_positions], dim=1).to(torch.int32)
        attn_metadata["position_ids"] = position_ids
        attn_metadata["kv_len"] = position_ids + 1
        start_pos = draft_positions[:, 0].to(torch.int32)
        attn_metadata["start_pos"] = start_pos

        block_table = attn_metadata["block_table"]["win_kv"].to(device=device, dtype=torch.int32)
        main_slot_mapping = self._slot_mapping_from_positions(
            block_table, main_positions, self.block_size)
        draft_slot_mapping = self._slot_mapping_from_positions(
            block_table, draft_positions, self.block_size)
        attn_metadata["dspark_pa_inputs"] = {
            "main_slot_mapping": main_slot_mapping,
            "draft_slot_mapping": draft_slot_mapping,
        }

        # Persistent cache ownership stays with the framework. Attention reads
        # a compact, fixed-shape PA view: valid context first, then ALL draft KV,
        win_topk_ids = self.build_sliding_topk_ids(draft_positions, self.window_size)

        win_sparse_indices = win_topk_ids.unsqueeze(1)
        seqused_q = torch.full(
            (batch_size,),
            draft_len,
            device=device,
            dtype=torch.int32,
        )
        cu_seqused_q = torch.cumsum(seqused_q, dim=0)
        cu_seq_lens_q = torch.cat(
            [torch.zeros_like(cu_seqused_q[:1]), cu_seqused_q],
            dim=0,
        ).to(torch.int32)
        win_topk_length = (win_topk_ids >= 0).sum(dim=-1).unsqueeze(1).to(torch.int32)

        attn_metadata["dspark_fused_fa_inputs"] = {
            "win_sparse_indices": win_sparse_indices,
            "cu_seqlens_q": cu_seq_lens_q,
            "win_topk_length": win_topk_length
        }

    @staticmethod
    def _slot_mapping_from_positions(
        block_table: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Map absolute token positions to framework-owned PA cache slots."""
        positions = positions.to(device=block_table.device, dtype=torch.long)
        valid = positions >= 0
        safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
        block_indices = safe_positions // block_size
        rows = torch.arange(
            positions.shape[0], device=positions.device, dtype=torch.long,
        ).unsqueeze(1).expand_as(positions)
        block_ids = block_table.to(torch.long)[rows, block_indices]
        slots = block_ids * block_size + safe_positions.remainder(block_size)
        return torch.where(valid, slots, slots.new_full(slots.shape, -1)).to(torch.int32)

    def _prefill_proposal_cache(self, main_hidden, target_hidden_positions, attn_metadata):
        """Project and write the prepared prompt context to each stage's PA cache."""
        self.model.prefill_update_cache(main_hidden, attn_metadata)

    def forward_spec(self, input_ids: torch.LongTensor, main_hidden: torch.Tensor, **kwargs):
        """Execute proposal computation using worker-prepared inputs."""
        if kwargs.get("is_prefill", False):
            self._prefill_proposal_cache(**kwargs["prefill_context"])

        spec_tokens, logits, confidence = self._forward_spec_decode_impl(input_ids, main_hidden, kwargs)
        return {
            "spec_tokens": spec_tokens,
            "logits": logits,
            "confidence": confidence,
        }

    def _forward_spec_decode_impl(
        self,
        input_ids: torch.LongTensor,
        main_hidden: torch.Tensor,
        kwargs: Dict,
    ):
        attn_metadata = kwargs.get("attn_metadata")
        sample_noise = kwargs.get("sample_noise")
        sampling_params = kwargs.get("sampling_params")

        self.generate_dspark_kernel_metadata(attn_metadata)
        hidden_states, confidence_states = self.model(
            input_ids,
            main_hidden,
            attn_metadata,
            main_next_tokens=kwargs.get("main_next_tokens"),
            cur_topk_list=kwargs.get("cur_topk_list"),
        )
        logits = self.forward_lm_head(
            outputs=hidden_states,
            kv_len=attn_metadata["kv_len"],
            is_prefill=False,
            attn_metadata=attn_metadata,
        ).float()
        last_block = self.model.layers[-1]
        batch_size = input_ids.shape[0]
        draft_len = logits.shape[1]
        confidence_states = confidence_states.view(batch_size, draft_len, -1)
        main_next_tokens = kwargs.get("main_next_tokens")
        seed_input_ids = (
            main_next_tokens[:, 0]
            if main_next_tokens is not None
            else (input_ids[:, 0] if input_ids.dim() > 1 else input_ids)
        )
        output_ids, confidence = last_block.forward_proposal_head(
            logits,
            confidence_states,
            seed_input_ids,
            DSparkSamplingContext(params=sampling_params, noise=sample_noise),
        )
        return output_ids[:, 1:], logits, confidence

    def generate_dspark_kernel_metadata(self, attn_metadata: Dict) -> None:
        """Generate the window-only fused-attention metadata once per proposal."""
        fused_fa_inputs = attn_metadata["dspark_fused_fa_inputs"]
        metadata_stream = attn_metadata.get("metadata_stream")
        main_stream = torch.npu.current_stream() if self.enable_multi_streams else None

        record_event(self.enable_multi_streams, self.metadata_event, 0)
        with npu_stream_switch(self.enable_multi_streams, metadata_stream):
            wait_event(self.enable_multi_streams, self.metadata_event, 0)
            win_topk_length = fused_fa_inputs["win_topk_length"]
            metadata = self.sparse_attn_metadata_ops(
                win_topk_length,
                torch.zeros_like(win_topk_length),
                cu_seqlens_q=fused_fa_inputs["cu_seqlens_q"],
                num_heads_q=(
                    self.config.num_attention_heads
                    if self.low_latency_tp
                    else self.config.num_attention_heads // self.infer_config.parallel_config.attn_tp_size
                ),
                num_heads_kv=1,
                head_dim=self.config.head_dim,
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=False,
            )
            metadata = metadata.to(attn_metadata["position_ids"].device, non_blocking=True)
            fused_fa_inputs["metadata"] = metadata
            record_stream(self.enable_multi_streams, metadata, main_stream)
            record_event(self.enable_multi_streams, self.metadata_event, 1)

    def _load_weight_map(self):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]

        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.num_experts)

        return stacked_params_mapping, expert_params_mapping

    def _validate_dspark_loaded_weights(self, loaded_params: Set[str]):
        last_stage_idx = self.dspark_num_layers - 1
        required_params = {
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            f"mtp.{last_stage_idx}.norm.weight",
            f"mtp.{last_stage_idx}.hc_attn_fn",
            f"mtp.{last_stage_idx}.hc_attn_base",
            f"mtp.{last_stage_idx}.hc_attn_scale",
            f"mtp.{last_stage_idx}.hc_ffn_fn",
            f"mtp.{last_stage_idx}.hc_ffn_base",
            f"mtp.{last_stage_idx}.hc_ffn_scale",
            f"mtp.{last_stage_idx}.markov_head.embed.weight",
            f"mtp.{last_stage_idx}.markov_head.head.weight",
            f"mtp.{last_stage_idx}.confidence_head.proj.weight",
        }
        required_params.update(
            f"mtp.{stage_idx}.attn.attn_sink" for stage_idx in range(self.dspark_num_layers)
        )
        missing_required = sorted(required_params - loaded_params)
        if missing_required:
            raise ValueError(
                f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}"
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load DSpark stage weights and adapt checkpoint tensor layouts."""
        stacked_params_mapping, expert_params_mapping = self._load_weight_map()
        # The model owns the layers below ``dspark_model.layers`` while the
        # checkpoint intentionally uses the public ``mtp.<stage>`` namespace.
        # adapt_safetensors_field_dspark returns that public namespace as keys.
        params_dict = adapt_safetensors_field_dspark(dict(self.named_parameters()))
        loaded_params: Set[str] = set()
        dequant_cache = {}
        is_replace_expert_scale_name = any("w13_weight_scale" in key for key in params_dict)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".ffn.shared_experts." in name or ".attn." in name or "main_proj." in name:
                name = name.replace(".scale", ".weight_scale")
                if name.endswith(".weight_scale"):
                    # Scales are repeated offline; preserve their E8M0 bits when loading.
                    loaded_weight = loaded_weight.view(torch.uint8)
            elif name.endswith(".engram.wkv.scale"):
                loaded_weight = loaded_weight.view(torch.uint8)

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
                if (("ffn.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if self.config.quant_config.mm_quant_mode != "w8a8float8":
                    name = name.replace(".scale", ".weight_scale")

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_replace_expert_scale_name:
                        name = name.replace("w13_scale", "w13_weight_scale").replace("w2_scale", "w2_weight_scale")

                    if name not in params_dict:
                        continue
                    is_gmm_w4mxfloat = ("w4" in self.config.quant_config.gmm_quant_mode and
                                        "mxfloat" in self.config.quant_config.gmm_quant_mode)
                    if is_gmm_w4mxfloat:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                    loaded_weight,
                                    name,
                                    shard_id=shard_id,
                                    expert_id=expert_id)
                    break
                else:
                    # The npu_transpose_batchmatmul op doesn't support the fp8 data type. The weight of wo_a needs
                    # to be converted to bf16.
                    if "wo_a" in name and self.config.quant_config.mm_quant_mode == "w8a8float8":
                        base_name, attr = name.rsplit(".", 1)
                        if base_name not in dequant_cache:
                            dequant_cache[base_name] = {}
                        dequant_cache[base_name][attr] = loaded_weight
                        if "weight" in dequant_cache[base_name] and "scale" in dequant_cache[base_name]:
                            data = dequant_cache.pop(base_name)
                            q_weight = data["weight"]
                            scale = data["scale"]
                            loaded_weight = weight_dequant(q_weight, scale)
                            name = f"{base_name}.weight"

                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if self.config.quant_config.mm_quant_mode != "w8a8float8":
                        name = name.replace(".scale", ".weight_scale")

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # add checkpoint load check
        weights_not_loaded = set(params_dict.keys()) - loaded_params
        key_weights = {"smooth_scale", "w2_alpha"}
        if weights_not_loaded:
            if all(any(key in name for key in key_weights) for name in weights_not_loaded):
                logger.warning(
                    "Smooth scales were not initialized from checkpoint.")
            else:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")
        self._validate_dspark_loaded_weights(loaded_params)
        return loaded_params

    def propose(self, model_inputs):
        """Run worker-prepared proposal inputs through eager or compiled execution."""
        is_prefill = model_inputs.get("is_prefill", False)
        if not is_prefill and self.compiled_forward_spec_decode is not None:
            is_warm_up = getattr(get_forward_metadata(), "is_warm_up", False)
            compile_context = (
                torch.compiler.set_stance(skip_guard_eval_unsafe=True)
                if self.enable_npugraph_ex and self.enable_cache_compile and not is_warm_up
                else nullcontext()
            )
            with compile_context:
                spec_tokens, logits, confidence = self.compiled_forward_spec_decode(**model_inputs)
            return {
                "spec_tokens": spec_tokens,
                "logits": logits,
                "confidence": confidence,
            }
        return self.forward_spec(**model_inputs)


def adapt_safetensors_field_dspark(params_dict: Dict):
    """Build a checkpoint-name view of DSpark's registered parameters.

    DSpark checkpoints expose ``mtp.<stage>``.  Depending on registration order
    PyTorch reports the same parameters as ``dspark_model.layers.<stage>`` (or
    through the inherited ``model.`` prefix).  Keep the actual Parameter values
    while returning aliases in the checkpoint namespace.
    """
    fix_dict = {}
    for k, v in params_dict.items():
        if k.startswith("layers."):
            k = "mtp." + k[len("layers."):]
        elif k.startswith("model.layers."):
            k = "mtp." + k[len("model.layers."):]
        elif "model." in k and not k.startswith("model."):
            k = k.replace("model.", "", 1)

        replacements = (
            ("tid2eid", "gate.tid2eid"),
            ("e_score_correction_bias", "bias"),
            ("shared_experts.down_proj", "shared_experts.w2"),
            ("markov_head.markov_w1", "markov_head.embed"),
            ("markov_head.markov_w2", "markov_head.head"),
        )
        for old, new in replacements:
            k = k.replace(old, new)
        # Avoid overwriting a canonical alias when the module is reachable via
        # both ``mtp`` and ``dspark_model`` registration paths.
        fix_dict.setdefault(k, v)
    return fix_dict
