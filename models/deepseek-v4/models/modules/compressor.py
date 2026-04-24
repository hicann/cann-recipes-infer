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
from executor.utils import npu_stream_switch, get_had_pow2
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm, apply_rotary_emb, rotate_activation


class Compressor(nn.Module):
    def __init__(self, config, runner_settings, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                rotate: bool = False, head_dim: int = 512, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.dim = config.hidden_size
        self.head_dim = head_dim # 128 if indexer else 512
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.prefix = prefix
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )

        self.use_fused_kernel_compressor = runner_settings.get("kernel_config", {}).get("compressor", "native") == "ascendc"
        self.platform_version = runner_settings.get("model_config").get("platform_version", "A3")
        self.block_size = runner_settings.get("model_config").get("pa_block_size", 128)
        # wkv and wgate in checkpoint is stored in bf16, stored in fp32 for convenient
        self.wkv = ReplicatedLinear(self.dim,
                                    coff * self.head_dim,
                                    params_dtype=torch.bfloat16,
                                    quant_config=None,
                                    prefix=f"{prefix}.wkv")
        self.wgate = ReplicatedLinear(self.dim,
                                      coff * self.head_dim,
                                      params_dtype=torch.bfloat16,
                                      quant_config=None,
                                      prefix=f"{prefix}.wgate")

        self.norm = DeepseekV3RMSNorm(self.head_dim, config.rms_norm_eps)

        self.hadamard_matrix = get_had_pow2(self.head_dim)
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
        self.global_rank = kwargs.get("global_rank")
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window

    def compressor_epilog(self, kv: torch.Tensor, cache_data: Tuple[Dict], attn_metadata: Dict, is_prefill: bool):
        if is_prefill and self.cp_size > 1:
            cmp_slot_mapping = attn_metadata["cp_metadata"]["slot_mapping_cmp"][f"{self.compress_ratio}"]
        else:
            cmp_slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]
        if "index" in self.prefix:
            li_cmp_cache = cache_data["li_cmp_kv"]
            scale_cache = cache_data["li_key_dequant_scale"]
            if self.li_cache_quant_mode == "float8":
                torch.ops.custom.indexer_compress_epilog(
                    x=kv,
                    slot_mapping=cmp_slot_mapping,
                    indexer_compress_cache=li_cmp_cache,
                    indexer_compress_scale=scale_cache
                )
            else:
                kv, k_scale = torch_npu.npu_dynamic_quant(kv)
                k_scale = k_scale.to(torch.float16)
                torch.ops.custom.scatter_nd_update_asc(scale_cache.view(-1, scale_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                k_scale.view(-1, scale_cache.shape[-1]))
                torch.ops.custom.scatter_nd_update_asc(li_cmp_cache.view(-1, li_cmp_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                kv.view(-1, li_cmp_cache.shape[-1]))
        else: # SFA compress
            sfa_cmp_cache = cache_data["sfa_cmp_kv"]
            if self.kv_cache_quant_mode == "float8":
                epilog_out = torch.ops.custom.kv_compress_epilog(
                    x=kv,
                    slot_mapping=cmp_slot_mapping,
                    kv_compress_cache=sfa_cmp_cache
                )
            else:
                torch.ops.custom.scatter_nd_update_asc(sfa_cmp_cache.view(-1, sfa_cmp_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                kv.view(-1, sfa_cmp_cache.shape[-1]))
        return None

    def compressor_prolog(
        self,
        x: torch.Tensor,
        cache_data: Tuple[Dict],
        attn_metadata: Dict,
        is_prefill: bool
    ):
        if is_prefill and self.cp_size > 1:
            cos_sin = attn_metadata["cos_sin"]
            offset = attn_metadata["cmp_in_offset"][f"{self.compress_ratio}"]
            x = x[:, offset:]
            cu_seqlens = attn_metadata["cu_seq_lens"][f"{self.compress_ratio}"]
            seq_used_q = attn_metadata["seq_used_q"][f"{self.compress_ratio}"]
            start_pos = attn_metadata["start_pos"][f"{self.compress_ratio}"]
        else:
            cos_sin = attn_metadata["cos_sin"]
            cu_seqlens = attn_metadata["cu_seq_lens_q"]
            seq_used_q = attn_metadata["seq_used_q"] if is_prefill else None
            start_pos = attn_metadata["start_pos"]
        cos, sin = cos_sin[f"c{self.compress_ratio}a"]

        cmpr_input_kwargs = {
            "x": x.view(-1, self.dim),
            "wkv": self.wkv.weight,
            "wgate": self.wgate.weight,
            "ape": self.ape,
            "norm_weight": self.norm.weight,
            "rope_cos": cos.view(-1, self.rope_head_dim),
            "rope_sin": sin.view(-1, self.rope_head_dim),
            "state_block_table": attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"],
            "cu_seqlens": cu_seqlens,
            "seqused": seq_used_q,
            "start_pos": start_pos,
            "rope_head_dim": self.rope_head_dim,
            "cmp_ratio": self.compress_ratio,
            "coff": 1 + self.overlap,
            "norm_eps": self.norm.variance_epsilon,
            "rotary_mode": 2, # 1: half; 2: interleave
            "cache_mode": 1,  # 1: contiguous buffer; 2: ring buffer
        }
        if "index" in self.prefix: # LI
            cmpr_input_kwargs.update({
                "state_cache": cache_data["li_kv_state"].flatten(-3),
            })
        else: # SFA
            cmpr_input_kwargs.update({
                "state_cache": cache_data["sfa_kv_state"].flatten(-3),
            })

        kv = torch.ops.custom.compressor(**cmpr_input_kwargs) # (T, self.head_dim)
        return kv

    def forward(
        self,
        x: torch.Tensor,
        cache_data: Tuple[Dict],
        attn_metadata: Dict,
        is_prefill: bool
    ):
        # compressor
        if is_prefill and self.cp_size > 1:
            pad_kv_list = []
            pstr = "li" if "index" in self.prefix else "sfa"
            cur_kv_state = cache_data[pstr + "_kv_state"].clone().flatten(0, -3).flatten(-2)
            for zz_flag in ["prev", "next"]:
                x_seg = x[0] if zz_flag == "prev" else x[1]
                kv = self.compressor_prolog(x_seg, cache_data, attn_metadata[zz_flag], is_prefill)
                cp_metadata = attn_metadata[zz_flag]
                if cp_metadata["is_end"]:
                    cur_kv_state = cache_data[pstr + "_kv_state"].flatten(0, -3).flatten(-2)
                cmp_out_pad = cp_metadata["cmp_out_pad"][f"{self.compress_ratio}"]
                pad_tensor = cmp_out_pad[0] if "index" in self.prefix else cmp_out_pad[1]
                pad_kv = torch.cat([pad_tensor, kv], dim=0)
                pad_kv_list.append(pad_kv)
            pad_kv = torch.cat(pad_kv_list, dim=0)
            all_cmp_kv = kv.new_empty([pad_kv.shape[0] * self.cp_size, kv.shape[-1]])
            dist.all_gather_into_tensor(all_cmp_kv, pad_kv, group=self.hccl_comm_dict["cp_group"])

            # reverse
            all_cmp_kv = all_cmp_kv.view(-1, pad_kv.shape[0] // 2,
                                         kv.shape[-1])[attn_metadata["cp_metadata"]["reverse_index"]]
            kv = all_cmp_kv.flatten(0, 1)

            all_ks = cur_kv_state.new_empty([cur_kv_state.shape[0] * self.cp_size, cur_kv_state.shape[-1]])
            dist.all_gather_into_tensor(all_ks, cur_kv_state, group=self.hccl_comm_dict["cp_group"])
            last_ks = all_ks.view(self.cp_size, -1, cur_kv_state.shape[-1])[attn_metadata["cp_metadata"]["last_rank_zz"]]
            kv_state = last_ks

            cache_data[pstr + "_kv_state"][:] = kv_state.view(cache_data[pstr + "_kv_state"].shape)
        else:
            kv = self.compressor_prolog(x, cache_data, attn_metadata, is_prefill)

        # hardmard
        if self.rotate:
            kv = rotate_activation(kv, self.hadamard_matrix)

        # epilog
        return self.compressor_epilog(kv, cache_data, attn_metadata, is_prefill)