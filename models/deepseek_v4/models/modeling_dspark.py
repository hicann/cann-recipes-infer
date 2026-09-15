# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/tree/main
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

"""DeepSeek-V4 DSpark proposal model.

The target model supplies selected hidden states, DSpark proposes a token block,
and the target model verifies that block before the next proposal step.
"""

import json
import os
from contextlib import nullcontext
from typing import Dict, Iterable, NamedTuple, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import cann_ops_transformer  # noqa: F401  # Registers torch.ops.cann_ops_transformer.
from torch import nn
from transformers.utils import logging

from executor.core.config import CommManager, InferenceConfig
from executor.core.engine.sampler import Sampler
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils.forward_metadata import get_forward_metadata
from executor.utils.stream_utils import npu_stream_switch, record_event, record_stream, wait_event
from executor.model_loader.weight_utils import default_weight_loader
from module.fuse_moe_gmm import FusedMoEGMM
from module.linear import ColumnParallelLinear, ReplicatedLinear, VocabParallelEmbedding

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import Attention, DeepseekV3ForCausalLM, DeepseekV3Model, DeepseekV3MoE
from .modules import DeepseekV3RMSNorm
from .modules.registry import OpKernel

logger = logging.get_logger(__name__)

HADAMARD_SIZE = 128
MXFP8_GROUP_SIZE = 32


def resolve_dspark_proposal_num_layers(config: DeepseekV3Config, infer_config: InferenceConfig) -> int:
    """Resolve proposal depth from the draft checkpoint configuration.

    A standalone draft checkpoint may expose ``n_mtp_layers`` in its own model
    config. The official integrated DeepSeek-V4 DSpark checkpoint instead keeps
    target and proposal weights in one directory and stores this metadata in
    ``inference/config.json``.
    """
    proposal_num_layers = getattr(config, "n_mtp_layers", None)
    if proposal_num_layers is None:
        model_path = (
            infer_config.speculative_config.draft_model_path
            or infer_config.model_config.model_path
        )
        config_path = os.path.join(model_path, "inference", "config.json")
        try:
            with open(config_path, encoding="utf-8") as config_file:
                proposal_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "DeepSeek-V4 DSpark could not load proposal metadata from "
                f"{config_path!r}. The config must define a positive n_mtp_layers."
            ) from exc
        proposal_num_layers = proposal_config.get("n_mtp_layers")

    try:
        proposal_num_layers = int(proposal_num_layers)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DeepSeek-V4 DSpark proposal config must define an integer n_mtp_layers."
        ) from exc
    if proposal_num_layers <= 0:
        raise ValueError(
            "DeepSeek-V4 DSpark proposal config requires n_mtp_layers greater than 0."
        )
    return proposal_num_layers


class DSparkSamplingContext(NamedTuple):
    """Sampling metadata and request-local noise for one draft block."""

    params: Optional[Dict[str, torch.Tensor | bool]]
    noise: Optional[torch.Tensor]


def _dequant_dspark_wo_a_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        weight
        .unflatten(0, (-1, HADAMARD_SIZE))
        .unflatten(-1, (-1, HADAMARD_SIZE))
        .float()
        .mul(scale[:, None, :, None].float())
        .flatten(2, 3)
        .flatten(0, 1)
        .bfloat16()
    )


def _slice_rope_tensor(tensor: torch.Tensor, bsz: int, total_len: int, start: int, end: int):
    return tensor.view(bsz, total_len, *tensor.shape[1:])[:, start:end].flatten(0, 1).contiguous()


class DSparkAttention(Attention):
    """
    DSpark attention follows the official proposal cache contract:
    projected main-model hidden states write the verified-token/window KV,
    while draft tokens attend over that window plus the block-local KV.
    """
    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, layer_idx: Optional[int] = None,
                 prefix: Optional[str] = "", comm_manager: CommManager = None,
                 dspark_stage_idx: int = 0, **kwargs):
        # Each stage uses the native MTP attention role; prefix and stage index
        # keep its weights and framework-managed KV cache independent.
        super().__init__(config, infer_config, comm_manager, config.num_hidden_layers, prefix, **kwargs)
        # DSpark SparseFlashMLA consumes BF16 PA_BBND cache. Do not inherit a
        # target-side KV quantization mode for this separate cache tensor.
        self.kv_cache_quant_mode = "unquant"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        self.wait_for_dspark_metadata = dspark_stage_idx == 0
        self.dspark_sparse_attn_ops = torch.ops.cann_ops_transformer.sparse_flash_mla
        self.dspark_sparse_attn_metadata_ops = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata

    def _init_cache_entries(self):
        """Register DSpark KV in the framework SlidingWindow cache group.

        Target and proposal layers use different cache tensors, but share the
        request-owned ``win_kv`` block table and lifecycle. This is the same
        separation used by paged speculative proposers: allocation is common,
        while KV contents remain model-specific.
        """
        self.dspark_kv_cache = torch.Tensor([])
        self.cache_entries = [
            CacheEntry(
                cache_name="dspark_win_kv",
                attn_type="SlidingWindow",
                dim=self.head_dim,
                num_head=1,
                dtype=torch.bfloat16,
                needs_block=True,
                block_size=self.block_size,
                manager_key="win_kv",
                tensor_setter=lambda tensor, layer=self: setattr(
                    layer, "dspark_kv_cache", tensor),
                sliding_window=self.window_size,
                cache_layout="BnBsND",
            )
        ]

    def _get_dspark_cache(self) -> torch.Tensor:
        cache = self.dspark_kv_cache
        if cache is None or cache.numel() == 0:
            raise RuntimeError(
                "DSpark SlidingWindow cache was not allocated by KVCacheManager."
            )
        return cache

    def _write_dspark_cache(self, kv: torch.Tensor, slot_mapping: torch.Tensor):
        """Write projected KV through the framework-provided PA slot mapping."""
        cache = self._get_dspark_cache()
        # PA block 0 is reserved by BlockPool as the null block. Route padded
        # positions there because aclnnScatterNdUpdate rejects negative indices.
        safe_slots = slot_mapping.to(
            device=kv.device, dtype=torch.int32).reshape(-1).clamp_min(0)
        torch_npu.npu_scatter_nd_update_(
            cache.view(-1, cache.shape[-1]),
            safe_slots.view(-1, 1),
            kv.view(-1, self.head_dim),
        )

    @staticmethod
    def _merge_rope_tensors(main_tensor: torch.Tensor, draft_tensor: torch.Tensor, bsz: int):
        main_tensor = main_tensor.view(bsz, -1, *main_tensor.shape[1:])
        draft_tensor = draft_tensor.view(bsz, -1, *draft_tensor.shape[1:])
        return torch.cat([main_tensor, draft_tensor], dim=1).flatten(0, 1).contiguous()

    def _apply_c1a_rope(self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rope_input = tensor.flatten(0, 1).unsqueeze(2) if tensor.dim() == 4 \
            else tensor.view(-1, 1, 1, self.head_dim)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            rope_input, cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        return tensor

    def _project_dspark_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        return self._apply_c1a_rope(kv, cos, sin)

    def _project_dspark_q(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # The shared quantization path consumes packed [tokens, hidden] inputs.
        qa = self.wq_a(x).flatten(0, 1)
        qr, qr_scale = self.apply_norm_dynamic_quant(qa)
        q = self.wq_b(qr, dynamic_scale=qr_scale).view(
            *x.shape[:2], self.num_heads_per_rank, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        return self._apply_c1a_rope(q, cos, sin)

    def _project_dspark_q_kv_multistream(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        attn_metadata: Dict,
    ):
        enable_multistream = self.enable_multi_streams and self.enable_npugraph_ex
        mla_stream = attn_metadata.get("mla_stream")
        rope_slices = attn_metadata["dspark_rope_slices"]
        main_cos, main_sin = rope_slices["main"]
        draft_cos, draft_sin = rope_slices["draft"]
        kv_input = torch.cat([main_x, x], dim=1)
        main_len = main_x.shape[1]

        # Keep Q projection on the current stream while main/draft KV projection
        # runs on the MLA stream. Both paths join before fused attention.
        # The shared quantization path consumes packed [tokens, hidden] inputs.
        qa = self.wq_a(x).flatten(0, 1)
        record_stream(enable_multistream, kv_input, mla_stream)
        record_event(enable_multistream, self.mla_events, 0)
        cur_stream = torch.npu.current_stream()
        with npu_stream_switch(enable_multistream, mla_stream):
            wait_event(enable_multistream, self.mla_events, 0)
            kv = self.wkv(kv_input)
            kv = self.kv_norm(kv)
            kv_cos = self._merge_rope_tensors(main_cos, draft_cos, x.shape[0])
            kv_sin = self._merge_rope_tensors(main_sin, draft_sin, x.shape[0])
            kv = self._apply_c1a_rope(kv, kv_cos, kv_sin)
            main_kv, draft_kv = kv.split([main_len, x.shape[1]], dim=1)
            record_stream(enable_multistream, main_kv, cur_stream)
            record_stream(enable_multistream, draft_kv, cur_stream)
            record_event(enable_multistream, self.mla_events, 1)

        qr, qr_scale = self.apply_norm_dynamic_quant(qa)
        q = self.wq_b(qr, dynamic_scale=qr_scale).view(
            *x.shape[:2], self.num_heads_per_rank, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = self._apply_c1a_rope(q, draft_cos, draft_sin)
        wait_event(enable_multistream, self.mla_events, 1)
        return main_kv, q, draft_kv

    def generate_dspark_fused_fa_metadata(self, attn_metadata: Dict):
        """Generate one metadata object shared by all DSpark proposal layers."""
        metadata_inputs = attn_metadata["dspark_fused_fa_inputs"]
        seqused_q = metadata_inputs["seqused_q"]
        seqused_ori_kv = metadata_inputs["seqused_ori_kv"]
        metadata = self.dspark_sparse_attn_metadata_ops(
            num_heads_q=self.num_heads_per_rank,
            num_heads_kv=1,
            head_dim=self.head_dim,
            cu_seqlens_q=None,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            batch_size=metadata_inputs["batch_size"],
            max_seqlen_q=metadata_inputs["max_seqlen_q"],
            max_seqlen_ori_kv=metadata_inputs["max_seqlen_ori_kv"],
            cmp_ratio=1,
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=-1,
            ori_win_right=-1,
            layout_q="BSND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        attn_metadata["dspark_fused_fa_metadata"] = (
            metadata_inputs["block_table"],
            seqused_q,
            seqused_ori_kv,
            metadata,
        )

    def _gather_dspark_kv(self, inputs: Dict) -> torch.Tensor:
        """Collect this stage's context window and full draft block for all queries."""
        cache = self._get_dspark_cache()
        valid_mask = inputs.get("valid_kv_mask")
        compact_kv = cache.reshape(-1, self.head_dim).index_select(
            0, inputs.get("gather_indices"),
        ).view(*valid_mask.shape[:2], self.head_dim)
        # Invalid slots can contain NaN; select zero rather than multiplying a mask.
        return torch.where(valid_mask, compact_kv, 0).reshape(
            -1, self.block_size, 1, self.head_dim,
        )

    def _dspark_fused_attention(self, q: torch.Tensor, attn_metadata: Dict):
        wait_event(
            self.enable_multi_streams and self.wait_for_dspark_metadata,
            attn_metadata.get("metadata_event"),
            1,
        )
        block_table, seqused_q, seqused_ori_kv, metadata = \
            attn_metadata["dspark_fused_fa_metadata"]

        compact_kv = self._gather_dspark_kv(attn_metadata["dspark_fused_fa_inputs"])

        return self.dspark_sparse_attn_ops(
            q=q,
            ori_kv=compact_kv,
            cmp_kv=None,
            ori_sparse_indices=None,
            cmp_sparse_indices=None,
            ori_block_table=block_table,
            cmp_block_table=None,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            sinks=self.attn_sink,
            metadata=metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=1,
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=-1,
            ori_win_right=-1,
            layout_q="BSND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        )[0].to(q.dtype)

    def _dspark_attn_post_eager(self, attn_output: torch.Tensor, attn_metadata: Dict):
        if self.oproj_tp_size > 1:
            output = self.attn_post(
                attn_output.flatten(0, 1), attn_metadata, is_prefill=False)
            return output.unflatten(0, attn_output.shape[:2])

        bsz, seq_len = attn_output.shape[:2]
        cos_sin = attn_metadata["cos_sin"]
        cos = cos_sin["c1a"][0] if self.compress_ratio == 1 else cos_sin["comp"][0]
        sin = cos_sin["c1a_neg_sin"] if self.compress_ratio == 1 else cos_sin["comp_neg_sin"]
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            attn_output.flatten(0, 1).unsqueeze(2),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        group_dim = self.num_heads_per_rank * self.head_dim // self.num_groups_per_rank
        attn_output = attn_output.view(bsz * seq_len, self.num_groups_per_rank, group_dim).to(torch.bfloat16)

        if self.mm_quant_mode == "w8a8mxfloat8" and hasattr(self.wo_a, "weight_scale"):
            attn_output, output_scale = torch_npu.npu_dynamic_mx_quant(
                attn_output, dst_type=torch.float8_e4m3fn)
            attn_output = torch_npu.npu_transpose_quant_batchmatmul(
                attn_output,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                x1_scale=output_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                group_sizes=(0, 0, MXFP8_GROUP_SIZE),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            ).view(bsz, seq_len, -1)
        else:
            attn_output = torch_npu.npu_transpose_batchmatmul(
                attn_output,
                self.wo_a.weight,
                perm_x1=(1, 0, 2),
                perm_y=(1, 0, 2),
            ).view(bsz, seq_len, -1)
        return self.wo_b(attn_output)

    def _draft_attn_metadata(self, attn_metadata: Dict):
        rope_slices = attn_metadata["dspark_rope_slices"]
        draft_metadata = dict(attn_metadata)
        draft_metadata["cos_sin"] = {
            "c1a": rope_slices["draft"],
            "c1a_neg_sin": rope_slices["draft_neg_sin"],
        }
        return draft_metadata

    def forward(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        **kwargs,
    ):
        is_prefill = kwargs.get("is_prefill", True)
        cos_sin = attn_metadata["cos_sin"]
        if is_prefill:
            # Prefill only seeds the proposal cache; token proposals start in decode.
            cos, sin = cos_sin["c1a"]
            main_kv = self._project_dspark_kv(main_x, cos, sin)
            cache_slots = attn_metadata["dspark_prefill_slot_mapping"]
            self._write_dspark_cache(main_kv, cache_slots)
            return x

        rope_slices = attn_metadata["dspark_rope_slices"]
        main_cos, main_sin = rope_slices["main"]
        draft_cos, draft_sin = rope_slices["draft"]

        enable_projection_multistream = self.enable_multi_streams and self.enable_npugraph_ex
        if enable_projection_multistream:
            main_kv, q, draft_kv = self._project_dspark_q_kv_multistream(
                x,
                main_x,
                attn_metadata,
            )
        else:
            main_kv = self._project_dspark_kv(main_x, main_cos, main_sin)
            q = self._project_dspark_q(x, draft_cos, draft_sin)
            draft_kv = self._project_dspark_kv(x, draft_cos, draft_sin)

        pa_inputs = attn_metadata["dspark_pa_inputs"]
        self._write_dspark_cache(main_kv, pa_inputs["main_slot_mapping"])
        self._write_dspark_cache(draft_kv, pa_inputs["draft_slot_mapping"])

        if q.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"DSpark SparseFlashMLA does not support query dtype {q.dtype}.")
        attn_output = self._dspark_fused_attention(q, attn_metadata)

        output = self._dspark_attn_post_eager(
            attn_output,
            self._draft_attn_metadata(attn_metadata),
        )
        return output


class DSparkMarkovHead(nn.Module):
    """Add a token-conditioned low-rank vocabulary bias to each proposal step.

    Embedding and vocabulary projections follow the model TP groups so the
    returned logits retain the full-vocabulary layout expected by sampling.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        prefix: str,
        comm_manager: CommManager = None,
    ):
        super().__init__()
        self.comm_manager = comm_manager
        parallel_config = infer_config.parallel_config
        self.embed_tp_size = parallel_config.embed_tp_size
        self.lmhead_tp_size = parallel_config.lmhead_tp_size
        self.attn_dp_size = parallel_config.attn_dp_size
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
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
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

        if self.attn_dp_size == 1 or self.lmhead_tp_size == 1:
            markov_hidden = markov_embed
        else:
            markov_hidden = torch.empty_like(markov_embed).repeat(self.lmhead_tp_size, 1)
            dist.all_gather_into_tensor(
                markov_hidden,
                markov_embed,
                group=self.comm_manager.get_group("lmhead_tp_group"),
            )

        logits = self.markov_w2(markov_hidden.float())
        if self.lmhead_tp_size > 1:
            if self.attn_dp_size == 1:
                gathered_logits = torch.empty_like(logits).repeat(self.lmhead_tp_size, 1)
                dist.all_gather_into_tensor(
                    gathered_logits,
                    logits,
                    group=self.comm_manager.get_group("lmhead_tp_group"),
                )
            else:
                gathered_logits = torch.empty_like(logits).view(-1)
                dist.all_to_all_single(
                    gathered_logits,
                    logits.view(-1),
                    group=self.comm_manager.get_group("lmhead_tp_group"),
                )
            logits = gathered_logits.reshape(
                self.lmhead_tp_size, markov_embed.shape[0], -1
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


class DeepseekV4DSparkProposalLayer(nn.Module):
    """One DSpark proposal block: HC attention, HC MoE, and optional output heads."""

    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, stage_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.stage_idx = stage_idx
        self.prefix = f"mtp.{stage_idx}"
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.dspark_num_layers = config.n_mtp_layers
        self.dspark_block_size = config.dspark_block_size
        self.dspark_markov_rank = config.dspark_markov_rank

        layer_idx = config.num_hidden_layers + stage_idx
        child_kwargs = {k: v for k, v in kwargs.items() if k != "comm_manager"}
        self.attn = DSparkAttention(
            config=config,
            infer_config=self.infer_config,
            comm_manager=kwargs.get("comm_manager"),
            layer_idx=layer_idx,
            prefix=f"{self.prefix}.attn",
            dspark_stage_idx=stage_idx,
            **child_kwargs,
        )
        self.ffn = DeepseekV3MoE(
            config,
            self.infer_config,
            comm_manager=kwargs.get("comm_manager"),
            prefix=f"{self.prefix}.ffn",
            layer_idx=layer_idx,
            **child_kwargs,
        )
        self.attn_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

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
            self.main_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if stage_idx == self.dspark_num_layers - 1:
            self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(
                config, self.infer_config, prefix=f"{self.prefix}.markov_head",
                comm_manager=kwargs.get("comm_manager"))
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + self.dspark_markov_rank,
                prefix=f"{self.prefix}.confidence_head",
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def project_main_hidden(self, main_hidden: torch.Tensor):
        return self.main_norm(self.main_proj(main_hidden))

    def prefill_main_cache(self, main_x: torch.Tensor,
                           attn_metadata: Optional[Dict] = None,
                           target_hidden_positions: Optional[torch.Tensor] = None):
        self.attn(
            x=main_x,
            main_x=main_x,
            attn_metadata=attn_metadata,
            is_prefill=True,
            target_hidden_positions=target_hidden_positions,
        )

    def _hc_pre(self, hidden_states, hc_fn, hc_scale, hc_base):
        """Run shared mHC preprocessing while preserving the proposal axis."""
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_states, post, comb = OpKernel.hc_pre(
            hidden_states.flatten(0, 1),
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        hidden_states = hidden_states.unflatten(0, (batch_size, seq_len))
        post = post.unflatten(0, (batch_size, seq_len))
        comb = comb.unflatten(0, (batch_size, seq_len))
        return hidden_states, post, comb

    def forward(
        self,
        hidden_states: torch.Tensor,
        main_x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        **kwargs,
    ):
        cur_topk_list = kwargs.get("cur_topk_list")
        input_ids = kwargs.get("input_ids")

        # Apply the HC mixture around proposal attention.
        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            x=hidden_states,
            main_x=main_x,
            attn_metadata=attn_metadata,
            is_prefill=False,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        # Apply the same HC residual pattern around the proposal MoE.
        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output_shape = hidden_states.shape
        hidden_states = self.ffn(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            is_prefill=False,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
            shared_expert_stream=attn_metadata.get('shared_expert_stream', None),
        )
        hidden_states = hidden_states.reshape(*ffn_output_shape[:-1], hidden_states.shape[-1])
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        return hidden_states

    def forward_head_hidden(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pre-norm confidence features and normalized LM features."""
        hidden_states = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(hidden_states, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        hidden_states = torch.sum(pre.unsqueeze(-1) * hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], self.hc_mult, self.config.hidden_size), dim=2)
        confidence_hidden = hidden_states.to(torch.bfloat16)
        # Official DSpark feeds pre-norm HC output to the confidence head, while
        # the shared LM head consumes the normalized representation.
        lm_hidden = self.norm(confidence_hidden)
        return confidence_hidden, lm_hidden

    def forward_proposal_head(
        self,
        logits: torch.Tensor,
        confidence_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        proposal_model: "DeepseekV4DSparkProposalModel",
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
            output_ids[:, idx + 1] = proposal_model.sample(
                logits[:, idx], sampling_params=sampling_context.params, sample_noise=step_noise)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(confidence_hidden, markov_embed)
        return output_ids, confidence


class DSparkInputModel(DeepseekV3Model):
    """Reuse packed embedding and RoPE helpers without a native decoder network."""

    def _init_decoder(self, config: DeepseekV3Config, **kwargs):
        # Bound to the main model embedding by the common worker after loading.
        self.embed_tokens = None


class DeepseekV4DSparkProposalModel(DeepseekV3ForCausalLM):
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
        if not infer_config.model_config.platform_version.is_ascend_950():
            raise ValueError(
                "DSpark currently supports Ascend 950 series only; "
                "disable speculative decoding or use a supported MTP configuration."
            )
        DeepseekV3ForCausalLM.update_model_cfg(config, infer_config)
        if infer_config.parallel_config.cp_size != 1:
            raise ValueError("DSpark requires cp_size=1; proposal prefill does not restore CP partitions.")
        config.n_mtp_layers = resolve_dspark_proposal_num_layers(config, infer_config)
        target_layer_ids = config.dspark_target_layer_ids
        if not target_layer_ids or list(target_layer_ids) != sorted(set(target_layer_ids)):
            raise ValueError("DSpark target layer ids must be unique and strictly increasing.")
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

    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, **kwargs):
        super().__init__(config, infer_config, is_mtp=True, comm_manager=kwargs.get("comm_manager"))
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
        self.mtp = nn.ModuleList([
            DeepseekV4DSparkProposalLayer(config, infer_config, stage_idx, **dspark_kwargs)
            for stage_idx in range(self.dspark_num_layers)
        ])

    def _build_decoder(self, config: DeepseekV3Config, prefix: str, **kwargs):
        return DSparkInputModel(
            config, self.infer_config, self.comm_manager, prefix, **kwargs,
        )

    def get_cache_info(self) -> ModelCacheInfo:
        """Expose proposal attention caches to the framework KVCacheManager."""
        layer_infos = [
            LayerCacheInfo(
                layer_idx=stage_idx,
                caches=list(block.attn.cache_entries),
            )
            for stage_idx, block in enumerate(self.mtp)
        ]
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    @staticmethod
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
        model_inputs = self.preprocess_model_inputs(
            model_inputs, is_prefill=bool(is_prefill), is_mtp=True,
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
        """Keep only the valid prompt tail needed by SlidingWindow KV."""
        batch_size, seq_len = positions.shape
        window_size = self.window_size
        valid_len = (positions >= 0).sum(dim=1)
        offsets = torch.arange(window_size, device=positions.device).view(1, window_size)
        source_idx = (valid_len.view(batch_size, 1) - window_size + offsets).clamp(
            min=0, max=seq_len - 1)
        valid_tail = offsets >= (window_size - valid_len.clamp(max=window_size)).view(batch_size, 1)
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
        attn_metadata["start_pos"] = draft_positions[:, 0].to(torch.int32)

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
        # with padding only at the end (excluded by seqused_ori_kv).
        context_end = draft_positions[:, :1]
        context_len = context_end.clamp(min=0, max=self.window_size)
        max_total_tokens = self.window_size + draft_len
        blocks_per_request = (max_total_tokens + self.block_size - 1) // self.block_size
        offsets = torch.arange(
            blocks_per_request * self.block_size, device=device, dtype=torch.long,
        ).unsqueeze(0)
        gather_positions = context_end - context_len + offsets
        valid_kv = offsets < context_len + draft_len
        gather_positions = torch.where(
            valid_kv, gather_positions, torch.full_like(gather_positions, -1),
        )
        gather_slots = self._slot_mapping_from_positions(
            block_table, gather_positions, self.block_size,
        )
        compact_block_table = torch.arange(
            batch_size * blocks_per_request, device=device, dtype=torch.int32,
        ).view(batch_size, blocks_per_request)
        seqused_q = torch.full(
            (batch_size,),
            draft_len,
            device=device,
            dtype=torch.int32,
        )
        seqused_ori_kv = (context_len.squeeze(1) + draft_len).to(torch.int32)
        attn_metadata["dspark_fused_fa_inputs"] = {
            "batch_size": batch_size,
            "max_seqlen_q": draft_len,
            "max_seqlen_ori_kv": max_total_tokens,
            "block_table": compact_block_table,
            "gather_indices": gather_slots.clamp_min(0).reshape(-1).long(),
            "valid_kv_mask": (gather_slots >= 0).unsqueeze(-1),
            "seqused_q": seqused_q,
            "seqused_ori_kv": seqused_ori_kv,
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
        main_x = self.mtp[0].project_main_hidden(main_hidden)
        cos_sin = self.model.generate_cos_sin(attn_metadata, main_x, is_mtp=True)
        attn_metadata.update({'cos_sin': cos_sin})
        for block in self.mtp:
            block.prefill_main_cache(
                main_x,
                attn_metadata=attn_metadata,
                target_hidden_positions=target_hidden_positions,
            )

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
        decode_inputs: Dict,
    ):
        attn_metadata = decode_inputs.get("attn_metadata")
        main_next_tokens = decode_inputs.get("main_next_tokens")
        cur_topk_list = decode_inputs.get("cur_topk_list")
        sample_noise = decode_inputs.get("sample_noise")
        sampling_params = decode_inputs.get("sampling_params")
        is_prefill = False
        draft_input_ids = input_ids
        input_ids = main_next_tokens[:, :1].reshape(-1)
        batch_size = input_ids.shape[0]

        draft_attn_metadata = dict(attn_metadata)
        # Metadata runs independently and is consumed only when the first
        # proposal attention reaches its fused FA call.
        enable_metadata_stream = self.enable_multi_streams
        record_event(enable_metadata_stream, draft_attn_metadata.get("metadata_event"), 0)
        with npu_stream_switch(
            enable_metadata_stream,
            draft_attn_metadata.get("metadata_stream"),
        ):
            wait_event(enable_metadata_stream, draft_attn_metadata.get("metadata_event"), 0)
            self.mtp[0].attn.generate_dspark_fused_fa_metadata(draft_attn_metadata)
            record_event(enable_metadata_stream, draft_attn_metadata.get("metadata_event"), 1)

        main_x = self.mtp[0].project_main_hidden(main_hidden)
        hidden_states = self.model.calc_input_embeddings(
            draft_input_ids.reshape(-1), is_prefill,
        ).unflatten(0, (batch_size, self.dspark_block_size))

        # RoPE is shared by all DSpark layers and is generated once before the layer loop.
        cos_sin = self.model.generate_cos_sin(
            draft_attn_metadata,
            torch.cat([main_x, hidden_states], dim=1),
            is_mtp=True,
        )
        draft_attn_metadata["cos_sin"] = cos_sin
        main_len = main_x.shape[1]
        draft_len = hidden_states.shape[1]
        total_len = main_len + draft_len
        main_cos = _slice_rope_tensor(cos_sin["c1a"][0], batch_size, total_len, 0, main_len)
        main_sin = _slice_rope_tensor(cos_sin["c1a"][1], batch_size, total_len, 0, main_len)
        draft_cos = _slice_rope_tensor(cos_sin["c1a"][0], batch_size, total_len, main_len, total_len)
        draft_sin = _slice_rope_tensor(cos_sin["c1a"][1], batch_size, total_len, main_len, total_len)
        draft_neg_sin = _slice_rope_tensor(
            cos_sin["c1a_neg_sin"], batch_size, total_len, main_len, total_len)
        draft_attn_metadata["dspark_rope_slices"] = {
            "main": (main_cos, main_sin),
            "draft": (draft_cos, draft_sin),
            "draft_neg_sin": draft_neg_sin,
        }

        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        # All proposal blocks reuse projected target context, RoPE, and FA metadata.
        for block in self.mtp:
            hidden_states = block(
                hidden_states,
                main_x=main_x,
                attn_metadata=draft_attn_metadata,
                cur_topk_list=cur_topk_list,
                input_ids=draft_input_ids,
            )
        last_block = self.mtp[-1]
        # The final block produces separate pre-norm confidence and normalized
        # LM inputs, and owns the token-conditioned Markov/confidence heads.
        confidence_hidden, lm_hidden = last_block.forward_head_hidden(hidden_states)
        logits = self.forward_lm_head(
            outputs=lm_hidden.flatten(0, 1),
            kv_len=draft_attn_metadata["kv_len"],
            is_prefill=False,
            attn_metadata=draft_attn_metadata,
        ).float()
        output_ids, confidence = last_block.forward_proposal_head(
            logits,
            confidence_hidden,
            input_ids,
            self,
            DSparkSamplingContext(params=sampling_params, noise=sample_noise),
        )
        return output_ids[:, 1:], logits, confidence

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
            num_experts=self.config.n_routed_experts)

        return stacked_params_mapping, expert_params_mapping

    def _validate_dspark_loaded_weights(self, loaded_params: Set[str]):
        last_stage_idx = self.dspark_num_layers - 1
        required_params = {
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            f"mtp.{last_stage_idx}.norm.weight",
            f"mtp.{last_stage_idx}.hc_head_fn",
            f"mtp.{last_stage_idx}.hc_head_base",
            f"mtp.{last_stage_idx}.hc_head_scale",
            f"mtp.{last_stage_idx}.markov_head.markov_w1.weight",
            f"mtp.{last_stage_idx}.markov_head.markov_w2.weight",
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
        params_dict = adapt_safetensors_field_dspark(dict(self.named_parameters()))
        loaded_params: Set[str] = set()
        dequant_cache = {}
        is_replace_expert_scale_name = any("w13_weight_scale" in key for key in params_dict)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.ignore_share_weight and any(substring in name for substring in ["lm_head.weight", "emb.tok_emb"]):
                continue
            if not name.startswith("mtp."):
                continue

            # Map stacked dense/MoE shards first, then handle DSpark-specific
            # projection layouts and direct one-to-one parameters.
            loaded_name = name
            stacked_loaded = False
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in loaded_name:
                    continue
                if ("ffn.experts." in loaded_name) and loaded_name not in params_dict:
                    continue
                mapped_name = loaded_name.replace(weight_name, param_name)
                if self.mm_quant_mode != "w8a8float8":
                    mapped_name = mapped_name.replace(".scale", ".weight_scale")
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                stacked_loaded = True
            if stacked_loaded:
                continue

            for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
                if weight_name not in loaded_name:
                    continue
                if expert_id != -1 and f"experts.{expert_id}." not in loaded_name:
                    continue
                mapped_name = loaded_name.replace(weight_name, param_name)
                if is_replace_expert_scale_name:
                    mapped_name = mapped_name.replace("scale", "weight_scale")
                elif self.mm_quant_mode != "w8a8float8":
                    mapped_name = mapped_name.replace(".scale", ".weight_scale")
                if mapped_name not in params_dict:
                    continue
                is_gmm_w4mxfloat = ("w4" in self.config.quant_config.gmm_quant_mode and
                                    "mxfloat" in self.config.quant_config.gmm_quant_mode)
                if is_gmm_w4mxfloat:
                    loaded_weight = loaded_weight.view(torch.uint8)
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, mapped_name, shard_id=shard_id, expert_id=expert_id)
                loaded_params.add(mapped_name)

            if ".attn.wo_a." in loaded_name and self.mm_quant_mode == "w8a8float8":
                base_name, attr = loaded_name.rsplit(".", 1)
                if attr in ("weight", "scale"):
                    if base_name not in dequant_cache:
                        dequant_cache[base_name] = {}
                    dequant_cache[base_name][attr] = loaded_weight
                    if "weight" in dequant_cache[base_name] and "scale" in dequant_cache[base_name]:
                        data = dequant_cache.pop(base_name)
                        mapped_name = f"{base_name}.weight"
                        if mapped_name in params_dict:
                            param = params_dict[mapped_name]
                            dequant_weight = _dequant_dspark_wo_a_weight(data["weight"], data["scale"])
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, dequant_weight)
                            loaded_params.add(mapped_name)
                    continue

            mapped_name = loaded_name
            if self.mm_quant_mode != "w8a8float8":
                mapped_name = mapped_name.replace(".scale", ".weight_scale")
            if mapped_name in params_dict:
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped_name)

        missing = set(params_dict.keys()) - loaded_params
        optional_keys = ("model.", "head.", "embed_tokens", "lm_head")
        missing = {name for name in missing if not any(key in name for key in optional_keys)}
        # Official DSpark checkpoints use q_norm followed by an explicit RMS on q,
        # so the inherited target-attention q_b_norm parameter is intentionally unused.
        missing = {name for name in missing if "attn.q_b_norm." not in name}
        if missing:
            logger.warning(f"DSpark weights not initialized from checkpoint: {missing}")
        self._validate_dspark_loaded_weights(loaded_params)
        return loaded_params

    def propose(self, model_inputs):
        """Run worker-prepared proposal inputs through eager or compiled execution."""
        is_prefill = model_inputs.get("is_prefill", False)
        with torch.no_grad():
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
    """Normalize checkpoint field names without changing parameter values."""
    fix_dict = {}
    for k, v in params_dict.items():
        if "model." in k and not k.startswith("model."):
            k = k.replace("model.", "")
        if "tid2eid" in k:
            k = k.replace("tid2eid", "gate.tid2eid")
        if "e_score_correction_bias" in k:
            k = k.replace("e_score_correction_bias", "bias")
        if "shared_experts.down_proj" in k:
            k = k.replace("shared_experts.down_proj", "shared_experts.w2")
        if "input_layernorm" in k:
            k = k.replace("input_layernorm", "attn_norm")
        if "post_attention_layernorm" in k:
            k = k.replace("post_attention_layernorm", "ffn_norm")
        fix_dict[k] = v
    return fix_dict
