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

from executor.core.config import InferenceConfig, CommManager
from executor.utils import get_had_pow2, limit_core_num
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm, apply_rotary_emb, rotate_activation


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
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.owns_k = layer_idx in config.kv_source_layers
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_candidate_source = layer_idx == config.candidate_source_layer
        self.uses_candidates = 0 <= config.candidate_source_layer < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.freqs_cis: torch.Tensor | None = None
        self.eps = config.rms_norm_eps
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
        if self.owns_k:
            self.wk = ReplicatedLinear(config.head_dim,
                                       self.head_dim,
                                       params_dtype=torch.bfloat16,
                                       quant_config=None,
                                       prefix=f"{prefix}.wk")
            self.k_norm = DeepseekV3RMSNorm(self.head_dim, self.eps)

        self.softmax_scale = self.head_dim ** -0.5

        self.max_seq_len = self.infer_config.model_config.custom_params.get("max_position_embeddings", 2048)
        self.hadamard_matrix = get_had_pow2(self.head_dim)

    def update_indexer_cache(
        self,
        kv: torch.Tensor,
        indexer_slot_mapping: torch.Tensor,
        indexer_cache,
    ) -> Tuple[torch.Tensor, None]:
        slot_mapping = indexer_slot_mapping.view(-1)
        valid_mask = slot_mapping != -1
        valid_slots = slot_mapping[valid_mask].long()
        valid_kv = kv.view(-1, self.head_dim)[valid_mask]
        indexer_cache.view(-1, self.head_dim).index_copy_(
            0, valid_slots, valid_kv)
        return kv, None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_metadata: Dict,
        latent: torch.Tensor,
        kv_cache,
        offset: int = 0
    ):
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        x = x.unsqueeze(0)
        indexer_cache = kv_cache.indexer_cache

        bsz, seq_len, _ = x.size()
        cos, sin = attn_metadata["cos_sin"]["comp"]
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)

        # rope
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            q.unsqueeze(2), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        q = q.view(bsz, -1, self.n_heads, self.head_dim)

        start_pos = attn_metadata["start_pos"] // self.compress_ratio
        end_pos = (attn_metadata["start_pos"] + attn_metadata["seq_used_q"]) // self.compress_ratio
        cos_k, sin_k = attn_metadata["cos_sin"][f"c{self.compress_ratio}a"]
        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            k = k.view(-1, 1, 1, self.head_dim)

            torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
                k, cos_k, sin_k,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            k = k.view(bsz, -1, self.head_dim)
            slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]
            self.update_indexer_cache(k, slot_mapping, indexer_cache)

        block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_kv"]
        index_k = indexer_cache[block_table].view(-1, *indexer_cache.shape[2:]) # T, H, D
        
        index_k = index_k[: end_pos]
        T, H, D = index_k.shape
        index_k = index_k.view(bsz, T, D)

        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        if start_pos == 0:
            compress_lens = (torch.arange(1, seq_len + 1, device=x.device) // self.compress_ratio).unsqueeze(-1)
            mask = torch.arange(index_score.size(-1), device=x.device) >= compress_lens
            index_score.masked_fill_(mask, -torch.inf)
        else:
            compress_lens = end_pos

        if self.is_candidate_source:
            candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
            kv_cache.candidates = candidates
        elif self.uses_candidates:
            candidates = kv_cache.candidates
            index_score = index_score.masked_fill(~candidates, -torch.inf)

        topk = min(self.index_topk, end_pos)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


def select_candidate_blocks(
        logits: torch.Tensor,
        compress_lens: torch.Tensor | int,
        topk_blocks: int,
        block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
