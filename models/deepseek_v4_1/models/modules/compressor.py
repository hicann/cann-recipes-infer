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
from typing import Optional, Dict

import torch
import torch.distributed as dist

from torch import nn

import torch_npu
from executor.core.config import InferenceConfig, CommManager
from module.linear import ReplicatedLinear
from .common_modules import (DeepseekV41RMSNorm, gather_cp_segments, get_cp_tmp_cache,
                             has_cp_decode_requests, scatter_cache_rows)


def write_compressed_kv(kv, slot_mapping, cache):
    """Quantize post-RoPE KV into its shared PA cache."""
    torch.ops.custom.kv_compress_epilog_v2(
        cache.view(-1, cache.shape[-1]),
        kv.view(-1, kv.shape[-1]),
        slot_mapping.view(-1),
        quant_group_size=16,
        quant_mode="mxfp4_bf16",
        round_scale=True,
        x_scale=1.0,
    )


class Compressor(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                rotate: bool = False, head_dim: int = 512, prefix: Optional[str] = "",
                comm_manager: CommManager = None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - config.qk_rope_head_dim
        self.partial_slice = [self.nope_head_dim, self.head_dim]
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.comm_manager = comm_manager

        from cann_ops_transformer.ops.ds41 import compressor
        import custom_ops  # Registers the repository's KV quantization writer.
        self.compressor_ops = compressor
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")

        # wkv and wgate in checkpoint is stored in bf16, stored in fp32 for convenient
        self.wkv = ReplicatedLinear(self.dim,
                                    self.head_dim,
                                    params_dtype=torch.bfloat16,
                                    quant_config=None,
                                    prefix=f"{prefix}.wkv")
        if self.compress_ratio > 1:
            self.wgate = ReplicatedLinear(self.dim,
                                          self.head_dim,
                                          params_dtype=torch.bfloat16,
                                          quant_config=None,
                                          prefix=f"{prefix}.wgate")
        else:
            self.wgate = None

        self.norm = DeepseekV41RMSNorm(self.head_dim, config.rms_norm_eps)

    def update_compress_cache(
        self,
        kv: torch.Tensor,
        cmp_slot_mapping: torch.Tensor,
        cmp_cache: torch.Tensor,
    ):
        torch.ops.custom.kv_compress_epilog_v2(
            cmp_cache.view(-1, cmp_cache.shape[-1]),
            kv.view(-1, self.head_dim),
            cmp_slot_mapping.view(-1),
            quant_group_size=16,
            quant_mode="mxfp4_bf16",
            round_scale=True,
            x_scale=1.0,
        )

    def compressor_epilog(self, kv: torch.Tensor, attn_metadata: Dict, kv_cache):
        cmp_slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]
        self.update_compress_cache(kv, cmp_slot_mapping, kv_cache.cmp_cache)

    def compressor_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        kv_cache,
        tmp_cache=None,
    ):
        # tmp_cache is set only by the CP path, where attn_metadata is one zigzag segment's metadata.
        ratio_key = f"{self.compress_ratio}"
        cos_sin = attn_metadata["cos_sin"]
        rope_cos, rope_sin = cos_sin[f"c{ratio_key}a"]

        if self.compress_ratio == 1:
            kv = torch.nn.functional.linear(x.view(-1, self.dim), self.wkv.weight)
            if tmp_cache is not None:
                # Slots and rope follow the packed rows the compressor operator emits, while this
                # projection keeps the CP padding of every request, so drop it the same way.
                valid_rows = attn_metadata["valid_rows"]
                kv = torch.nn.functional.pad(
                    kv.index_select(0, valid_rows), (0, 0, 0, kv.shape[0] - valid_rows.numel()))
        else:
            if tmp_cache is not None:
                cu_seqlens = attn_metadata["cu_seq_lens"][ratio_key]
                seq_used_q = attn_metadata["cmp_seq_used_q"][ratio_key]
                start_pos = attn_metadata["start_pos"][ratio_key]
                state_cache = tmp_cache["state_cache_by_layer"][self.layer_idx]["state_cache"]
                state_block_table = attn_metadata["tmp_block_table"][f"c{ratio_key}a_cmp_state"]
            else:
                cu_seqlens = attn_metadata["cu_seq_lens_q"]
                seq_used_q = attn_metadata["seq_used_q"]
                start_pos = attn_metadata["start_pos"]
                state_cache = kv_cache.state_cache
                state_block_table = attn_metadata["block_table"][f"c{ratio_key}a_cmp_state"]
            cmpr_input_kwargs = {
                "x": x.view(-1, self.dim),
                "wkv": self.wkv.weight,
                "wgate": self.wgate.weight,
                "seqused": seq_used_q,
                "start_pos": start_pos,
                "cmp_ratio": self.compress_ratio,
                "cu_seqlens": cu_seqlens,
                "state_cache": state_cache.flatten(-2),
                "state_block_table": state_block_table[:, 0],
            }
            kv = self.compressor_ops(**cmpr_input_kwargs)

        kv = torch_npu.npu_rms_norm(
            kv,
            self.norm.weight,
            self.norm.variance_epsilon,
        )[0]
        latent = kv.clone()
        kv_rope_view = kv.view(-1, 1, 1, self.head_dim)
        rope_cos = rope_cos.view(-1, 1, 1, self.rope_head_dim)
        rope_sin = rope_sin.view(-1, 1, 1, self.rope_head_dim)
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            kv_rope_view,
            rope_cos,
            rope_sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        return kv, latent

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool,
        kv_cache,
    ):
        if is_prefill and self.cp_size > 1:
            return self.forward_cp(x, attn_metadata, kv_cache)
        kv, latent = self.compressor_prolog(x, attn_metadata, kv_cache)
        self.compressor_epilog(kv, attn_metadata, kv_cache)
        return latent

    def forward_cp(self, x: torch.Tensor, attn_metadata: Dict, kv_cache):
        """Compress the two zigzag segments in turn, then gather the rows of the whole sequence.

        Returns the per-segment latent, which the Indexer turns into index K.
        """
        ratio_key = f"{self.compress_ratio}"
        cp_metadata = attn_metadata["cp_metadata"]
        tmp_cache = get_cp_tmp_cache(attn_metadata, self.compress_ratio)
        segments = x.chunk(2, dim=0)
        remainders = self.gather_remainders(segments, cp_metadata)
        latent_list, kv_list = [], []
        for zigzag_flag, x_seg in zip(("prev", "next"), segments):
            seg_metadata = attn_metadata[zigzag_flag]
            # Written right before the segment that reads it, so no tail state is overwritten.
            self.write_remainder_state(seg_metadata, remainders, tmp_cache)
            kv, latent = self.compressor_prolog(x_seg, seg_metadata, kv_cache, tmp_cache)
            latent_list.append(latent)
            kv_list.append(kv)

        # Quantize this rank's rows before they travel: identity slots keep them contiguous,
        # so the gather moves 320 byte rows instead of 1024 byte bf16 ones and each rank
        # quantizes only its own share.
        scratch = tmp_cache["cmp_scratch"]
        rows = 0
        for kv in kv_list:
            self.update_compress_cache(
                kv,
                torch.arange(rows, rows + kv.shape[0], dtype=torch.int32, device=kv.device),
                scratch,
            )
            rows += kv.shape[0]

        all_kv = gather_cp_segments(
            scratch.view(-1, scratch.shape[-1])[:rows], cp_metadata,
            self.comm_manager.get_group("cp_group"))
        scatter_cache_rows(tmp_cache["cmp_cache"], cp_metadata["slot_mapping_cmp"][ratio_key], all_kv)
        if has_cp_decode_requests(cp_metadata):
            scatter_cache_rows(
                kv_cache.cmp_cache, cp_metadata["slot_mapping_cmp_for_decode"][ratio_key], all_kv)
        if self.compress_ratio > 1:
            self.sync_cp_state(attn_metadata, kv_cache, tmp_cache)
        return latent_list

    def gather_remainders(self, segments, cp_metadata):
        """Collect the last hidden of every segment, which completes the next segment's group.

        One carry token per segment completes a group of at most two.
        """
        if self.compress_ratio <= 1 or not cp_metadata["needs_remainder"]:
            return None
        segment_lens = cp_metadata["segment_lens"]
        tail_index, offset = [], 0
        for segment_len in segment_lens:
            offset += segment_len
            tail_index.append(offset - 1)
        tail_index = torch.tensor(tail_index, dtype=torch.long, device=segments[0].device)
        tails = torch.cat([seg.view(-1, self.dim).index_select(0, tail_index) for seg in segments], dim=0)
        all_tails = gather_cp_segments(tails, cp_metadata, self.comm_manager.get_group("cp_group"))
        return all_tails.view(-1, len(segment_lens), self.dim)

    def write_remainder_state(self, seg_metadata, remainders, tmp_cache):
        """Project the preceding segment's last hidden and write it as this segment's carry."""
        if remainders is None:
            return
        remainder = seg_metadata["remainder"][f"{self.compress_ratio}"]
        if remainder is None:
            return
        source = remainders[seg_metadata["segment_idx"] - 1].index_select(0, remainder["requests"]).float()
        state = torch.cat([
            torch.nn.functional.linear(source, self.wkv.weight.float()),
            torch.nn.functional.linear(source, self.wgate.weight.float()),
        ], dim=-1)
        state_cache = tmp_cache["state_cache_by_layer"][self.layer_idx]["state_cache"]
        scatter_cache_rows(state_cache.flatten(-2), remainder["slots"], state)

    def sync_cp_state(self, attn_metadata: Dict, kv_cache, tmp_cache):
        """Keep the state of the segment holding the last valid token and hand it to decode."""
        cp_group = self.comm_manager.get_group("cp_group")
        cp_size = dist.get_world_size(cp_group)
        cp_metadata = attn_metadata["cp_metadata"]
        state_key = f"c{self.compress_ratio}a_cmp_state"
        state_cache = tmp_cache["state_cache_by_layer"][self.layer_idx]["state_cache"]
        gathered = state_cache.new_empty([state_cache.shape[0] * cp_size, *state_cache.shape[1:]])
        dist.all_gather_into_tensor(gathered, state_cache, group=cp_group)
        # Requests end on different ranks, so every state block takes its own source rank.
        state_blocks = cp_metadata["tmp_block_table"][state_key][:, 0].long()
        source_ranks = torch.tensor(
            cp_metadata["last_segment_rank"], dtype=torch.long, device=state_cache.device)
        state_cache[state_blocks] = gathered.view(cp_size, *state_cache.shape)[source_ranks, state_blocks]

        if attn_metadata["is_warm_up"] or not has_cp_decode_requests(cp_metadata):
            return
        # One persistent state block per owned request, so both sides index by the same requests.
        owned_requests = cp_metadata["owner_request_indices"].to(state_blocks.device)
        dst_block_ids = attn_metadata["block_table"][state_key][owned_requests, 0].long()
        kv_cache.state_cache[dst_block_ids] = state_cache[state_blocks[owned_requests]]
