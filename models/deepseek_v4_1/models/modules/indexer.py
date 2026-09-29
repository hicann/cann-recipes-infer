# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
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
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng
import cann_ops_transformer
import cann_ops_transformer_experimental
from cann_ops_transformer.ops.ds41 import (
    indexer_prologue_k,
    quant_lightning_indexer,
    quant_sparse_lightning_indexer,
)

from executor.core.config import InferenceConfig, CommManager
from executor.utils.stream_utils import wait_event
from module.linear import ReplicatedLinear
from .common_modules import (DeepseekV41RMSNorm, apply_rotary_emb, gather_cp_segments, get_cmp_block_table,
                             get_cp_tmp_cache, has_cp_decode_requests, rotate_activation, scatter_cache_rows)
from .registry import OpKernel

# MXFP4 layout used by the Indexer cache and QLI/QSLI operators.
MXFP4_VALUES_PER_BYTE = 2
MXFP4_SCALE_GROUP_SIZE = 32
MXFP4_SCALE_PAIR_GROUPS = 2
MXFP4_SCALE_PAIR_SIZE = MXFP4_SCALE_GROUP_SIZE * MXFP4_SCALE_PAIR_GROUPS


class Indexer(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                prefix: Optional[str] = "", comm_manager: CommManager = None,
                cache_getter: Callable[[str], torch.Tensor] = None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.cache_getter = cache_getter
        self.enable_metadata_multi_streams = infer_config.model_config.custom_params.get(
            "enable_multi_streams", False
        )
        self.is_first_indexer = layer_idx == min(config.index_source_layers)
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        self.layer_idx = layer_idx
        self.use_fused_prolog_qw = False
        self.indexer_prologue_k_op = indexer_prologue_k
        self.quant_lightning_indexer_ops = quant_lightning_indexer
        self.quant_sparse_lightning_indexer_ops = quant_sparse_lightning_indexer
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.cp_size = self.infer_config.parallel_config.cp_size
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
        self.mxfp4_packed_dim = self.head_dim // MXFP4_VALUES_PER_BYTE
        self.mxfp4_scale_dim = self.head_dim // MXFP4_SCALE_GROUP_SIZE
        self.mxfp4_scale_pair_dim = self.head_dim // MXFP4_SCALE_PAIR_SIZE
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
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, self.eps)

        self.softmax_scale = self.head_dim ** -0.5

        self.max_seq_len = self.infer_config.model_config.custom_params.get("max_position_embeddings", 2048)

    def write_prologue_k(self, latent, cos_k, sin_k, k_cache, k_scale_cache, cache_index,
                         storage_mode=0, combined_block_size=-1):
        """Project, norm, rope and quantize index K of one segment straight into a cache."""
        self.indexer_prologue_k_op(
            latent=latent.view(-1, self.wk.input_size),
            wk=self.wk.weight.detach(),
            norm_weight=self.k_norm.weight.detach(),
            # RotaryEmbedding returns [T, 1, 1, Dr], while the fused K
            # prologue ABI requires flat [T, Dr] RoPE tensors.
            rope_sin=sin_k.reshape(-1, self.rope_head_dim),
            rope_cos=cos_k.reshape(-1, self.rope_head_dim),
            k_cache=k_cache,
            k_scale_cache=k_scale_cache,
            cache_index=cache_index,
            norm_eps=self.eps,
            storage_mode=storage_mode,
            combined_block_size=combined_block_size,
        )

    def update_cp_indexer_cache(self, latent_list, attn_metadata: Dict, kv_cache, tmp_cache):
        """Quantize index K of both zigzag segments into scratch, gather, then write the cache.

        The fused K prologue writes into a cache and returns nothing, so the rows are quantized
        before they travel. Identity slots keep this rank's scratch rows contiguous for the gather.
        """
        ratio_key = f"{self.compress_ratio}"
        cp_metadata = attn_metadata["cp_metadata"]
        scratch, scratch_scale = tmp_cache["indexer_scratch"], tmp_cache["indexer_scratch_scale"]
        rows = 0
        for zigzag_flag, latent in zip(("prev", "next"), latent_list):
            seg_metadata = attn_metadata[zigzag_flag]
            cos_k, sin_k = seg_metadata["cos_sin"][f"c{ratio_key}a"]
            # The row count follows latent itself, so the operator never reads past it.
            seg_rows = latent.shape[0]
            self.write_prologue_k(
                latent, cos_k, sin_k, scratch, scratch_scale,
                torch.arange(rows, rows + seg_rows, dtype=torch.int64, device=latent.device))
            rows += seg_rows

        # K and its scale live in two buffers; they travel as one row and split on arrival.
        local_rows = torch.cat([
            scratch.view(-1, self.mxfp4_packed_dim)[:rows],
            scratch_scale.view(-1, self.mxfp4_scale_dim)[:rows],
        ], dim=-1)
        all_rows = gather_cp_segments(
            local_rows, cp_metadata, self.comm_manager.get_group("cp_group"))
        all_k = all_rows[:, :self.mxfp4_packed_dim].contiguous()
        all_scale = all_rows[:, self.mxfp4_packed_dim:].contiguous()

        slot_mapping = cp_metadata["slot_mapping_cmp"][ratio_key]
        scatter_cache_rows(tmp_cache["indexer_cache"], slot_mapping, all_k)
        scatter_cache_rows(tmp_cache["indexer_cache_scale"], slot_mapping, all_scale)
        self.repack_qsli_cache(tmp_cache, cp_metadata["batch_size"])
        if has_cp_decode_requests(cp_metadata):
            decode_slots = cp_metadata["slot_mapping_cmp_for_decode"][ratio_key]
            scatter_cache_rows(kv_cache.indexer_cache, decode_slots, all_k)
            scatter_cache_rows(kv_cache.indexer_cache_scale, decode_slots, all_scale)
            self.update_decode_qsli_cache(tmp_cache, attn_metadata, kv_cache, cp_metadata)

    def update_decode_qsli_cache(self, tmp_cache, attn_metadata: Dict, kv_cache, cp_metadata):
        """Copy the combined rows of this rank's own requests into the persistent cache."""
        qsli = tmp_cache.get("qsli_cache")
        persistent = getattr(kv_cache, "indexer_qsli_cache", None)
        if qsli is None or persistent is None:
            return
        _, qsli_blocks, rows_per_block = tmp_cache["qsli_layout"]
        block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_qsli_combined_kv"]
        combined = qsli.view(-1, qsli.shape[-1])
        rows = min(qsli_blocks, block_table.shape[-1]) * rows_per_block
        row_idx = torch.arange(rows, device=block_table.device)
        for request in cp_metadata["owner_request_indices"].tolist():
            src = (request * qsli_blocks + 1) * rows_per_block
            slots = (block_table[request, row_idx // rows_per_block].long() * rows_per_block
                     + row_idx % rows_per_block)
            scatter_cache_rows(persistent, slots, combined[src:src + rows])

    def repack_qsli_cache(self, tmp_cache, batch_size):
        """Rewrite the rows QSLI reads: one entry holds a group of keys followed by their scales."""
        qsli = tmp_cache.get("qsli_cache")
        if qsli is None:
            return
        cmp_blocks, qsli_blocks, rows_per_block = tmp_cache["qsli_layout"]
        group = self.candidate_block_size
        cache = tmp_cache["indexer_cache"]
        block_size = cache.shape[1]
        keys = cache.view(-1, self.mxfp4_packed_dim)
        scales = tmp_cache["indexer_cache_scale"].view(-1, self.mxfp4_scale_dim)
        combined = qsli.view(-1, qsli.shape[-1])
        rows = cmp_blocks * block_size // group
        for request in range(batch_size):
            src = (request * cmp_blocks + 1) * block_size
            dst = (request * qsli_blocks + 1) * rows_per_block
            combined[dst:dst + rows] = torch.cat([
                keys[src:src + rows * group].view(rows, -1),
                scales[src:src + rows * group].view(rows, -1),
            ], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_metadata: Dict,
        latent,
        kv_cache,
        offset: int = 0
    ):
        if self.is_first_indexer:
            wait_event(
                self.enable_metadata_multi_streams,
                attn_metadata.get("metadata_event"),
                2,
            )
        cos, sin = attn_metadata["cos_sin"]["comp"]
        # Fused scope: MXFP8 quantization of qr, MXFP8 Q projection with wq_b,
        # partial RoPE on Q, MXFP4 Q quantization, and weights_proj(x) with
        # softmax scaling. The native implementation preserves this same scope.
        q, q_scale, weights = OpKernel.indexer_prolog_qw(self, x, qr, cos, sin)

        if attn_metadata["is_prefill"] and self.cp_size > 1:
            return self.forward_cp(q, q_scale, weights, latent, attn_metadata, kv_cache, offset)

        if self.owns_k and latent is not None:
            self.update_indexer_cache(latent, attn_metadata, kv_cache)
        return self.compute_sparse_indices(
            q, q_scale, weights, kv_cache.indexer_cache, kv_cache.indexer_cache_scale,
            attn_metadata, offset, getattr(kv_cache, "indexer_qsli_cache", None))

    def forward_cp(self, q, q_scale, weights, latent_list, attn_metadata: Dict, kv_cache, offset):
        """Score both zigzag segments of this rank against the full length temporary cache."""
        tmp_cache = get_cp_tmp_cache(attn_metadata, self.compress_ratio)
        if self.owns_k and latent_list is not None:
            self.update_cp_indexer_cache(latent_list, attn_metadata, kv_cache, tmp_cache)
        # Local tokens are [prev | next]; each zigzag segment scores the cache with its own metadata.
        indices = [
            self.compute_sparse_indices(
                q_seg, q_scale_seg, weights_seg, tmp_cache["indexer_cache"],
                tmp_cache["indexer_cache_scale"], attn_metadata[zigzag_flag], offset,
                tmp_cache.get("qsli_cache"))
            for zigzag_flag, q_seg, q_scale_seg, weights_seg in zip(
                ("prev", "next"), q.chunk(2, dim=0), q_scale.chunk(2, dim=0), weights.chunk(2, dim=0))
        ]
        return torch.cat(indices, dim=0)

    def update_indexer_cache(self, latent, attn_metadata: Dict, kv_cache):
        """Write index K of every local token into the persistent caches."""
        cos_k, sin_k = attn_metadata["cos_sin"][f"c{self.compress_ratio}a"]
        # QLI keeps K and K-scale in separate caches.  A source layer can
        # also serve QSLI consumers, in which case the same K must be
        # written to the combined cache as well.
        slot_mapping_key = f"c{self.compress_ratio}a_cmp_kv"
        self.write_prologue_k(
            latent, cos_k, sin_k, kv_cache.indexer_cache, kv_cache.indexer_cache_scale,
            attn_metadata["indexer_slot_mapping"][slot_mapping_key])

        qsli_cache = getattr(kv_cache, "indexer_qsli_cache", None)
        if qsli_cache is None:
            return
        # The source layer currently updates the split QLI cache and the packed QSLI cache
        # separately. This compatibility path will be consolidated at layer 20 so one K update
        # serves both cache layouts.
        qsli_slot_mapping_key = f"c{self.compress_ratio}a_qsli_combined_kv"
        qsli_slot_mapping = attn_metadata["indexer_slot_mapping"].get(qsli_slot_mapping_key)
        if qsli_slot_mapping is None:
            raise KeyError(f"Missing Indexer QSLI slot mapping: {qsli_slot_mapping_key}")
        self.write_prologue_k(
            latent, cos_k, sin_k, qsli_cache, None, qsli_slot_mapping,
            storage_mode=1, combined_block_size=self.candidate_block_size)

    def compute_sparse_indices(self, q, q_scale, weights, indexer_cache, indexer_cache_scale,
                               attn_metadata, offset, qsli_cache):
        if self.uses_candidates:
            if qsli_cache is None:
                raise RuntimeError(
                    "QSLI requires the source layer's combined indexer cache"
                )
            return self.quant_sparse_lightning_indexer(
                q, q_scale, weights, qsli_cache, attn_metadata, offset
            )
        return self.quant_lightning_indexer(
            q, q_scale, weights, indexer_cache, indexer_cache_scale, attn_metadata, offset
        )

    def prepare_weights(self, x):
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        return weights.float()

    def prepare_query(self, qr, cos, sin):
        q = self.wq_b(qr)
        q = q.view(-1, self.n_heads, self.head_dim)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            q.unsqueeze(2), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        q, q_scale = torch_npu.npu_dynamic_mx_quant(q, dst_type=torch_npu.float4_e2m1fn_x2)
        return q.view(torch.uint8), q_scale

    def select_indices(self, q, q_scale, weights, attn_metadata, kv_cache, offset=0):
        # Separate MLA calls this after preparing Q/W and K on independent streams.
        if self.is_first_indexer:
            wait_event(
                self.enable_metadata_multi_streams,
                attn_metadata.get("metadata_event"),
                2,
            )
        return self.compute_sparse_indices(
            q, q_scale, weights, kv_cache.indexer_cache, kv_cache.indexer_cache_scale,
            attn_metadata, offset, getattr(kv_cache, "indexer_qsli_cache", None))

    def quant_lightning_indexer(
        self, q, q_scale, weights, indexer_cache, indexer_cache_scale, attn_metadata, offset=0
    ):
        if self.head_dim % MXFP4_SCALE_PAIR_SIZE:
            raise ValueError(
                "QLI E8M0 scale pairs require index_head_dim divisible by "
                f"{MXFP4_SCALE_PAIR_SIZE}"
            )
        packed_dim = self.mxfp4_packed_dim

        q = q.view(-1, self.n_heads, packed_dim)
        q_descale = q_scale.view(torch.uint8).view(
            q.shape[0], self.n_heads, self.mxfp4_scale_pair_dim, MXFP4_SCALE_PAIR_GROUPS,
        )

        k = indexer_cache
        k_descale = indexer_cache_scale.view(torch.uint8).view(
            *indexer_cache_scale.shape[:3], self.mxfp4_scale_pair_dim,
            MXFP4_SCALE_PAIR_GROUPS,
        )

        ratio_key = str(self.compress_ratio)
        seqused_k = (attn_metadata["actual_seq_k"] if self.compress_ratio == 1
                     else attn_metadata["compressed_seq_lens"][ratio_key])
        block_table = get_cmp_block_table(attn_metadata, self.compress_ratio)
        cmp_residual_k = (
            None if self.compress_ratio == 1
            else attn_metadata["compressed_seq_remainders"][ratio_key]
        )

        result = self.quant_lightning_indexer_ops(
            q=q,
            k=k,
            w=weights.view(-1, self.n_heads).float(),
            descale_q=q_descale,
            descale_k=k_descale,
            cu_seqlens_q=attn_metadata["cu_seq_lens_q"].to(torch.int32),
            seqused_q=None,
            seqused_k=seqused_k.to(torch.int32),
            cmp_residual_k=(None if cmp_residual_k is None
                            else cmp_residual_k.to(torch.int32)),
            block_table=block_table.to(torch.int32),
            topk=self.index_topk,
            quant_mode=1,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
            metadata=attn_metadata["kernel_metadata"][
                f"c{self.compress_ratio}a_qli_candidate_metadata"
                if self.is_candidate_source else f"c{self.compress_ratio}a_qli_metadata"
            ],
            candidate_topk_blocks=self.candidate_topk_blocks if self.is_candidate_source else -1,
            candidate_block_size=self.candidate_block_size if self.is_candidate_source else -1,
        )
        sparse_indices, _, candidate_indices, candidate_lengths = result
        if self.is_candidate_source:
            attn_metadata["li_candidates"] = candidate_indices
            attn_metadata["li_candidate_lengths"] = candidate_lengths

        return sparse_indices.view(q.shape[0], self.index_topk)

    def quant_sparse_lightning_indexer(
        self, q, q_scale, weights, indexer_qsli_cache, attn_metadata, offset
    ):
        if self.head_dim % MXFP4_SCALE_PAIR_SIZE:
            raise ValueError(
                "QSLI E8M0 scale pairs require index_head_dim divisible by "
                f"{MXFP4_SCALE_PAIR_SIZE}"
            )
        packed_dim = self.mxfp4_packed_dim

        q = q.view(-1, self.n_heads, packed_dim)
        descale_q = q_scale.view(torch.uint8).view(
            q.shape[0], self.n_heads, self.mxfp4_scale_pair_dim, MXFP4_SCALE_PAIR_GROUPS,
        )

        k = indexer_qsli_cache
        if k.dim() == 4 and k.shape[-2] == 1:
            k = k.squeeze(-2)

        block_table = get_cmp_block_table(
            attn_metadata, self.compress_ratio, suffix="qsli_combined_kv")
        seqused_k = attn_metadata["compressed_seq_lens"][str(self.compress_ratio)]

        result = self.quant_sparse_lightning_indexer_ops(
            q=q,
            k=k,
            w=weights.view(-1, self.n_heads).float(),
            descale_q=descale_q,
            candidate_block_indices=attn_metadata["li_candidates"],
            candidate_block_length=attn_metadata["li_candidate_lengths"],
            cu_seqlens_q=attn_metadata["cu_seq_lens_q"].to(torch.int32),
            seqused_q=None,
            seqused_k=seqused_k.to(torch.int32),
            cmp_residual_k=None,
            block_table=block_table.to(torch.int32),
            output_idx_offset=None,
            metadata=attn_metadata["kernel_metadata"][
                f"c{self.compress_ratio}a_qsli_metadata"
            ],
            topk=self.index_topk,
            candidate_block_size=self.candidate_block_size,
            quant_mode=1,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=self.compress_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
        )

        sparse_indices, _ = result
        return sparse_indices.view(q.shape[0], self.index_topk)
