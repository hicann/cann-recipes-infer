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
from typing import Optional, Tuple, Dict

import torch

from torch import nn

import torch_npu
from cann_ops_transformer import compressor
from executor.core.config import InferenceConfig, CommManager
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm

FALL_BACK_EPILOG = 1
FALL_BACK_COMPRESSOR = 1


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

        self.use_fused_kernel_compressor = (
            self.infer_config.model_config.custom_params.get("kernel_config", {}).get("compressor", "native")
            == "ascendc"
        )
        self.block_size = self.infer_config.scheduler_config.block_size
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

        self.norm = DeepseekV3RMSNorm(self.head_dim, config.rms_norm_eps)

        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

    def update_compress_cache(
        self,
        kv: torch.Tensor,
        cmp_slot_mapping: torch.Tensor,
        kv_cache,
    ) -> Tuple[torch.Tensor, None]:
        slot_mapping = cmp_slot_mapping.view(-1)
        valid_mask = slot_mapping != -1
        valid_slots = slot_mapping[valid_mask].long()
        valid_kv = kv.view(-1, self.head_dim)[valid_mask]
        kv_cache.cmp_cache.view(-1, self.head_dim).index_copy_(
            0, valid_slots, valid_kv)

        return kv, None

    def compressor_epilog(self, kv: torch.Tensor, attn_metadata: Dict, kv_cache):
        cmp_slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]
        if FALL_BACK_EPILOG:
            kv, k_scale = self.update_compress_cache(kv, cmp_slot_mapping, kv_cache)
        return None

    def _native_compressor_fallback(
        self,
        x_flat: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_used_q: torch.Tensor,
        start_pos: torch.Tensor,
        state_block_table: torch.Tensor,
        state_cache: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        ratio = self.compress_ratio
        head_dim = self.head_dim
        dtype = x_flat.dtype
        device = x_flat.device

        bsz = seq_used_q.shape[0]
        t = x_flat.shape[0]
        cmp_size = min(t, t // ratio + bsz)

        out = torch.zeros(cmp_size, head_dim, dtype=dtype, device=device)

        x_fp32 = x_flat.float()
        kv_proj = torch.nn.functional.linear(x_fp32, self.wkv.weight)
        score_proj = torch.nn.functional.linear(x_fp32, self.wgate.weight)

        out_idx = 0

        if is_prefill:
            for b in range(bsz):
                seq_len = seq_used_q[b].item()
                if seq_len == 0:
                    continue
                seq_start = cu_seqlens[b].item()
                kv_b = kv_proj[seq_start:seq_start + seq_len]
                score_b = score_proj[seq_start:seq_start + seq_len]

                remainder = seq_len % ratio
                cutoff = seq_len - remainder

                if remainder > 0:
                    block_id = state_block_table[b, (seq_len - 1) // self.block_size].item()
                    for j in range(remainder):
                        state_cache[block_id, j, 0] = kv_b[cutoff + j]
                        state_cache[block_id, j, 1] = score_b[cutoff + j]

                if cutoff > 0:
                    kv_groups = kv_b[:cutoff].view(-1, ratio, head_dim)
                    score_groups = score_b[:cutoff].view(-1, ratio, head_dim)
                    pooled = (kv_groups * score_groups.softmax(dim=1)).sum(dim=1)
                    num_groups = pooled.shape[0]
                    out[out_idx:out_idx + num_groups] = pooled.to(dtype)
                    out_idx += num_groups
        else:
            seqlen = x_flat.shape[0] // bsz
            for b in range(bsz):
                s_len = seq_used_q[b].item()
                if s_len == 0:
                    continue
                sp = start_pos[b].item()
                offset = b * seqlen

                for t_idx in range(s_len):
                    pos = sp + t_idx
                    slot = pos % ratio
                    block_id = state_block_table[b, pos // self.block_size].item()
                    state_cache[block_id, slot, 0] = kv_proj[offset + t_idx]
                    state_cache[block_id, slot, 1] = score_proj[offset + t_idx]
                    if (pos + 1) % ratio == 0:
                        kv_state = state_cache[block_id, :ratio, 0]
                        score_state = state_cache[block_id, :ratio, 1]
                        pooled = (kv_state * score_state.softmax(dim=0)).sum(dim=0)
                        out[out_idx] = pooled.to(dtype)
                        out_idx += 1

        return out

    def compressor_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool,
        kv_cache,
    ):
        cos_sin = attn_metadata["cos_sin"]
        rope_cos, rope_sin = cos_sin[f"c{self.compress_ratio}a"]

        if self.compress_ratio == 1:
            kv = torch.nn.functional.linear(x.view(-1, self.dim), self.wkv.weight)
        else:
            cu_seqlens = attn_metadata["cu_seq_lens_q"]
            seq_used_q = attn_metadata["seq_used_q"]
            start_pos = attn_metadata["start_pos"]
            state_block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"]
            cmpr_input_kwargs = {
                "x": x.view(-1, self.dim),
                "wkv": self.wkv.weight,
                "wgate": self.wgate.weight,
                "ape": None,
                "state_block_table": state_block_table,
                "seqused": seq_used_q,
                "start_pos": start_pos,
                "cmp_ratio": self.compress_ratio,
                "coff": 1,
                "cache_mode": 1,  # 1: contiguous buffer; 2: ring buffer
                "cu_seqlens": cu_seqlens,
            }
            cmpr_input_kwargs.update({
                "state_cache": kv_cache.state_cache.flatten(-2),
            })
            if FALL_BACK_COMPRESSOR:
                kv = self._native_compressor_fallback(
                    x.view(-1, self.dim),
                    cu_seqlens,
                    seq_used_q,
                    start_pos,
                    state_block_table,
                    kv_cache.state_cache,
                    is_prefill,
                )
            else:
                kv = compressor(**cmpr_input_kwargs)

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
        kv, latent = self.compressor_prolog(x, attn_metadata, is_prefill, kv_cache)
        self.compressor_epilog(kv, attn_metadata, kv_cache)
        return latent
