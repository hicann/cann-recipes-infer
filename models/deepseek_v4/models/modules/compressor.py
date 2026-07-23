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
from transformers.cache_utils import Cache
from executor.core.config import InferenceConfig, CommManager
from executor.utils import npu_stream_switch, get_had_pow2
from module.linear import ReplicatedLinear
from .common_modules import DeepseekV3RMSNorm, apply_rotary_emb, rotate_activation


class Compressor(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, layer_idx: Optional[int] = None, compress_ratio: int = 4,
                rotate: bool = False, head_dim: int = 512, prefix: Optional[str] = "",
                comm_manager: CommManager = None, cache_getter: Callable[[str], torch.Tensor] = None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        self.cache_getter = cache_getter
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

        self.use_fused_kernel_compressor = (
            self.infer_config.model_config.custom_params.get("kernel_config", {}).get("compressor", "native")
            == "ascendc"
        )
        self.platform_version = self.infer_config.model_config.platform_version
        self.block_size = self.infer_config.scheduler_config.block_size
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")

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

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.global_rank = kwargs.get("global_rank")
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window

    def get_cache(self, cache_name: str) -> torch.Tensor:
        if self.cache_getter is None:
            raise RuntimeError(f"cache_getter is required when accessing {cache_name}.")
        return self.cache_getter(cache_name)

    def _get_cp_state_block_ids(self, attn_metadata: Dict) -> torch.Tensor:
        state_key = f"c{self.compress_ratio}a_cmp_state"
        block_table = attn_metadata["block_table"][state_key]
        block_ids = torch.unique(block_table[block_table > 0], sorted=True)
        return block_ids.to(dtype=torch.long)

    def _gather_cp_state_blocks(self, state_cache: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        return torch.index_select(state_cache, 0, block_ids).contiguous()

    def _scatter_cp_state_blocks(
        self,
        state_cache: torch.Tensor,
        block_ids: torch.Tensor,
        state_blocks: torch.Tensor,
    ) -> None:
        state_cache.index_copy_(0, block_ids, state_blocks)

    def get_tmp_state_cache(self, tmp_cache: Dict, cache_name: str) -> torch.Tensor:
        ratio_cache = tmp_cache[f"{self.compress_ratio}"]
        state_cache_by_layer = ratio_cache.get("state_cache_by_layer", None)
        if state_cache_by_layer is None:
            return ratio_cache[cache_name]
        if self.layer_idx not in state_cache_by_layer:
            raise KeyError(
                f"Missing CP temporary state cache for layer {self.layer_idx}, "
                f"ratio {self.compress_ratio}."
            )
        layer_state_cache = state_cache_by_layer[self.layer_idx]
        if cache_name not in layer_state_cache:
            raise KeyError(
                f"Missing CP temporary state cache {cache_name} for layer "
                f"{self.layer_idx}, ratio {self.compress_ratio}."
            )
        return layer_state_cache[cache_name]

    def get_runtime_cache(self, cache_name: str, attn_metadata: Dict, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.cp_size > 1 and not self.is_online: # only for offline
            tmp_cache = attn_metadata["cp_metadata"].get("cp_tmp_cache", None)
            if tmp_cache is None:
                raise ValueError("When cp is enabled, a temporary cache is required, but no temporary cache is found.")
            if "state" in cache_name:
                return self.get_tmp_state_cache(tmp_cache, cache_name)
            return tmp_cache[f"{self.compress_ratio}"][cache_name]
        else:
            return self.get_cache(cache_name)

    def get_state_cache_online(self, cache_name: str, attn_metadata: Dict):
        state_block_ids = self._get_cp_state_block_ids(attn_metadata)
        return self._gather_cp_state_blocks(self.get_cache(cache_name), state_block_ids)

    def get_decode_token_indices(self, attn_metadata: Dict) -> torch.Tensor:
        cp_metadata = attn_metadata["cp_metadata"]
        if isinstance(cp_metadata, dict):
            return cp_metadata.get("decode_token_indices", None)
        return getattr(cp_metadata, "decode_token_indices", None)

    def get_ordered_block_ids(self, block_table: torch.Tensor) -> torch.Tensor:
        ordered_ids = []
        seen_ids = set()
        for block_id in block_table.reshape(-1).detach().cpu().tolist():
            block_id = int(block_id)
            if block_id != 0 and block_id not in seen_ids:
                ordered_ids.append(block_id)
                seen_ids.add(block_id)
        return torch.tensor(ordered_ids, dtype=torch.long, device=block_table.device)

    def update_decode_state_cache(
        self,
        tmp_state_cache: torch.Tensor,
        decode_state_cache: torch.Tensor,
        attn_metadata: Dict,
    ):
        # In offline scenarios, when CP is enabled for prefill, the temporary state cache values 
        # must be synchronized and flushed to the real cache used for decode inference.
        state_key = f"c{self.compress_ratio}a_cmp_state"
        tmp_block_table = attn_metadata["cp_metadata"]["tmp_block_table"][state_key]
        schedule_block_table = attn_metadata["block_table"][state_key]

        # If the system is currently in the warmup phase, skip the subsequent update.
        # During warmup, the block_table contains all zeros, which does not satisfy
        # the validation criteria and would cause incorrect updates if processed.
        if attn_metadata["is_warm_up"]:
            return

        tmp_block_ids = self.get_ordered_block_ids(tmp_block_table)
        schedule_block_ids = self.get_ordered_block_ids(schedule_block_table)
        src_block_ids = tmp_block_ids
        if self.compress_ratio == 4:
            dst_block_ids = schedule_block_ids
        elif self.compress_ratio == 128:
            dst_block_ids = schedule_block_ids[-1:]
        else:
            raise ValueError(f"Unsupported state cache ratio: {self.compress_ratio}.")

        decode_state_cache[dst_block_ids] = tmp_state_cache[src_block_ids]

    def update_compress_cache(
        self,
        kv: torch.Tensor,
        cmp_slot_mapping: torch.Tensor,
        is_prefill: bool,
        cache_getter: Callable[[str], torch.Tensor],
        quantized_kv: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if "index" in self.prefix:
            li_cmp_cache = cache_getter("li_cmp_kv")
            scale_cache = cache_getter("li_key_dequant_scale")
            if self.li_cache_quant_mode == "float8":
                torch.ops.custom.indexer_compress_epilog(
                    x=kv,
                    slot_mapping=cmp_slot_mapping,
                    indexer_compress_cache=li_cmp_cache,
                    indexer_compress_scale=scale_cache
                )
                return kv, None
            elif self.li_cache_quant_mode == "hifloat8":
                torch.ops.custom.indexer_compress_epilog(
                    x=kv,
                    slot_mapping=cmp_slot_mapping,
                    indexer_compress_cache=li_cmp_cache,
                    indexer_compress_scale=scale_cache,
                    quant_mode=1,
                    round_scale=True
                )
                return kv, None
            else:
                if quantized_kv is None and k_scale is None:
                    quantized_kv, k_scale = torch_npu.npu_dynamic_quant(kv)
                    k_scale = k_scale.to(torch.float16)
                torch.ops.custom.scatter_nd_update_asc(scale_cache.view(-1, scale_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                k_scale.view(-1, scale_cache.shape[-1]))
                torch.ops.custom.scatter_nd_update_asc(li_cmp_cache.view(-1, li_cmp_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                quantized_kv.view(-1, li_cmp_cache.shape[-1]))
                return quantized_kv, k_scale
        else:
            sfa_cmp_cache = cache_getter("sfa_cmp_kv")
            if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
                import cann_ops_transformer
                torch.ops.cann_ops_transformer.kv_compress_epilog(
                    x=kv if is_prefill else kv.view(-1, self.head_dim),
                    slot_mapping=cmp_slot_mapping if is_prefill else cmp_slot_mapping.view(-1),
                    cache=sfa_cmp_cache.view(torch.uint8),
                    quant_mode="fp8_bf16"
                )
            else:
                torch.ops.custom.scatter_nd_update_asc(sfa_cmp_cache.view(-1, sfa_cmp_cache.shape[-1]),
                                                cmp_slot_mapping.view(-1, 1),
                                                kv.view(-1, sfa_cmp_cache.shape[-1]))
            return kv, None

    def compressor_epilog(self, kv: torch.Tensor, attn_metadata: Dict, is_prefill: bool):
        if is_prefill and self.cp_size > 1:
            # temp cache
            cmp_slot_mapping = attn_metadata["cp_metadata"]["slot_mapping_cmp"][f"{self.compress_ratio}"]
            # decode_token_indices marks the cache indices updated by the current rank in CP.
            # Under the cp_mini_batch=1 constraint, a non-empty value means this rank must
            # update the decode cache for the current request.
            decode_token_indices = self.get_decode_token_indices(attn_metadata)
            has_decode_requests = decode_token_indices is not None and decode_token_indices.numel() > 0
        else:
            cmp_slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]

        kv, k_scale = self.update_compress_cache(
            kv,
            cmp_slot_mapping,
            is_prefill,
            lambda cache_name: self.get_runtime_cache(cache_name, attn_metadata, is_prefill),
        )

        if is_prefill and self.cp_size > 1 and has_decode_requests:
            decode_slot_mapping = attn_metadata["cp_metadata"].get("slot_mapping_cmp_for_decode", {}).get(
                f"{self.compress_ratio}", None
            )
            if decode_slot_mapping is not None:
                # In offline scenarios, when CP is enabled for prefill, the temporary compressed cache values 
                # must be synchronized and flushed to the real cache used for decode inference.
                self.update_compress_cache(
                    kv,
                    decode_slot_mapping,
                    is_prefill,
                    self.get_cache,
                    quantized_kv=kv if "index" in self.prefix else None,
                    k_scale=k_scale,
                )

        return None

    def compressor_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool,
        attn_metadata_ori: Dict = None
    ):
        if is_prefill and self.cp_size > 1:
            cos_sin = attn_metadata["cos_sin"]
            offset = attn_metadata["cmp_in_offset"][f"{self.compress_ratio}"]
            x = x[offset:]
            cu_seqlens = attn_metadata["cu_seq_lens"][f"{self.compress_ratio}"]
            seq_used_q = attn_metadata["seq_used_q"][f"{self.compress_ratio}"]
            start_pos = attn_metadata["start_pos"][f"{self.compress_ratio}"]
            if not self.is_online:
                state_block_table = attn_metadata["tmp_block_table"][f"c{self.compress_ratio}a_cmp_state"]
            else:
                state_block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"]
        else:
            attn_metadata_ori = attn_metadata
            cos_sin = attn_metadata["cos_sin"]
            cu_seqlens = attn_metadata["cu_seq_lens_q"]
            seq_used_q = attn_metadata["seq_used_q"]
            start_pos = attn_metadata["start_pos"]
            state_block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"]
        cos, sin = cos_sin[f"c{self.compress_ratio}a"]
        bsz = seq_used_q.shape[0]
        rope_cos = cos.view(-1, self.rope_head_dim) if is_prefill else cos.view(bsz, -1, self.rope_head_dim)
        rope_sin = sin.view(-1, self.rope_head_dim) if is_prefill else sin.view(bsz, -1, self.rope_head_dim)
        cmpr_input_kwargs = {
            "x": x.view(-1, self.dim) if is_prefill else x.view(bsz, -1, self.dim),
            "wkv": self.wkv.weight,
            "wgate": self.wgate.weight,
            "ape": self.ape,
            "norm_weight": self.norm.weight.to(torch.float32),
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "state_block_table": state_block_table,
            "seqused": seq_used_q,
            "start_pos": start_pos,
            "rope_head_dim": self.rope_head_dim,
            "cmp_ratio": self.compress_ratio,
            "coff": 1 + self.overlap,
            "norm_eps": self.norm.variance_epsilon,
            "rotary_mode": 2, # 1: half; 2: interleave
            "cache_mode": 1,  # 1: contiguous buffer; 2: ring buffer
        }
        if is_prefill:   # tnd format in prefill stage while bsnd format in decode stage
            cmpr_input_kwargs['cu_seqlens'] = cu_seqlens
        if "index" in self.prefix: # LI
            cmpr_input_kwargs.update({
                "state_cache": self.get_runtime_cache("li_kv_state", attn_metadata_ori, is_prefill).flatten(-3),
            })
        else: # SFA
            cmpr_input_kwargs.update({
                "state_cache": self.get_runtime_cache("sfa_kv_state", attn_metadata_ori, is_prefill).flatten(-3),
            })
        # (T, self.head_dim) in prefill stage, while (B, S, self.head_dim) in decode stage
        kv = torch.ops.custom.compressor(**cmpr_input_kwargs)
        return kv

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool
    ):
        # compressor
        if is_prefill and self.cp_size > 1:
            pad_kv_list = []
            pstr = "li" if "index" in self.prefix else "sfa"
            if self.is_online:
                cur_kv_state = self.get_state_cache_online(pstr + "_kv_state", attn_metadata)
            else:
                kv_state_cache = self.get_runtime_cache(pstr + "_kv_state", attn_metadata, is_prefill)
                cur_kv_state = kv_state_cache.clone()

            for zz_flag in ["prev", "next"]:
                x_seg = x[0] if zz_flag == "prev" else x[1]
                kv = self.compressor_prolog(x_seg, attn_metadata[zz_flag], is_prefill, attn_metadata)
                cp_metadata = attn_metadata[zz_flag]
                if cp_metadata["is_end"]:
                    if self.is_online:
                        cur_kv_state = self.get_state_cache_online(pstr + "_kv_state", attn_metadata)
                    else:
                        cur_kv_state = kv_state_cache
                cmp_out_pad = cp_metadata["cmp_out_pad"][f"{self.compress_ratio}"]
                pad_tensor = cmp_out_pad[0] if "index" in self.prefix else cmp_out_pad[1]
                pad_kv = torch.cat([pad_tensor, kv], dim=0)
                pad_kv_list.append(pad_kv)
            pad_kv = torch.cat(pad_kv_list, dim=0)
            all_cmp_kv = kv.new_empty([pad_kv.shape[0] * self.cp_size, kv.shape[-1]])
            dist.all_gather_into_tensor(all_cmp_kv, pad_kv, group=self.comm_manager.get_group("cp_group"))

            # reverse
            all_cmp_kv = all_cmp_kv.view(-1, pad_kv.shape[0] // 2,
                                         kv.shape[-1])[attn_metadata["cp_metadata"]["reverse_index"]]
            kv = all_cmp_kv.flatten(0, 1)

            all_ks = cur_kv_state.new_empty([cur_kv_state.shape[0] * self.cp_size, *cur_kv_state.shape[1:]])
            dist.all_gather_into_tensor(all_ks, cur_kv_state, group=self.comm_manager.get_group("cp_group"))
            last_ks = all_ks.view(self.cp_size, *cur_kv_state.shape)[attn_metadata["cp_metadata"]["last_rank_zz"]]
            kv_state = last_ks

            if self.is_online:
                state_block_ids = self._get_cp_state_block_ids(attn_metadata)
                self._scatter_cp_state_blocks(self.get_cache(pstr + "_kv_state"), state_block_ids, kv_state)
            else:
                tmp_kv_state_cache = kv_state_cache
                tmp_kv_state_cache[:] = kv_state.view(tmp_kv_state_cache.shape)
                decode_token_indices = self.get_decode_token_indices(attn_metadata)
                if decode_token_indices is not None and decode_token_indices.numel() > 0:
                    decode_kv_state_cache = self.get_cache(pstr + "_kv_state")
                    self.update_decode_state_cache(
                        tmp_kv_state_cache,
                        decode_kv_state_cache,
                        attn_metadata,
                    )
        else:
            kv = self.compressor_prolog(x, attn_metadata, is_prefill)

        # hardmard (hif8 covers outliers natively, no need for hadamard)
        if self.rotate and self.mm_quant_mode != "w8a8hifloat8":
            kv = rotate_activation(kv, self.hadamard_matrix)

        # epilog
        return self.compressor_epilog(kv, attn_metadata, is_prefill)
