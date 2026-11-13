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
from cann_ops_transformer import compressor
from transformers.cache_utils import Cache
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
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
        self.partial_slice = [self.nope_head_dim, self.head_dim]
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
                                    quant_config=config.quant_config \
                                        if self.mm_quant_mode == "w8a8hifloat8" else None,
                                    prefix=f"{prefix}.wkv")
        self.wgate = ReplicatedLinear(self.dim,
                                      coff * self.head_dim,
                                      params_dtype=torch.bfloat16,
                                      quant_config=config.quant_config \
                                        if self.mm_quant_mode == "w8a8hifloat8" else None,
                                      prefix=f"{prefix}.wgate")

        self.norm = DeepseekV3RMSNorm(self.head_dim, config.rms_norm_eps)

        if self.mm_quant_mode == "w8a8hifloat8":
            self.register_buffer("x_descale", torch.ones(1, dtype=torch.float32), persistent=False)

        self.hadamard_matrix = get_had_pow2(self.head_dim)
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.global_rank = kwargs.get("global_rank")
        self.layer_idx = layer_idx
        self.window_size = config.sliding_window

    def get_cache(self, cache_name: str) -> torch.Tensor:
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
        # ``state_cache_by_layer`` is materialized per (layer_idx, ratio) by
        # the CP planner, so the entries below always exist by construction.
        return state_cache_by_layer[self.layer_idx][cache_name]

    def get_runtime_cache(self, cache_name: str, attn_metadata: Dict, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.cp_size > 1 and not self.is_online: # only for offline
            cp_metadata = attn_metadata.get("cp_metadata")
            tmp_cache = (
                cp_metadata.get("cp_tmp_cache")
                if isinstance(cp_metadata, dict) else None
            )
            # Offline CP prefill must use the request-major temporary cache.
            # Falling back to the persistent cache for an ordinary compressed
            # layer would silently mix two different slot layouts.  Ratio-1
            # MTP layers do not allocate a CP compressor cache and are the only
            # explicit compatibility case where the persistent cache is valid.
            if not isinstance(tmp_cache, dict) or not tmp_cache:
                if self.compress_ratio == 1:
                    return self.get_cache(cache_name)
                raise ValueError(
                    f"Missing CP temporary cache for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio} during offline prefill."
                )
            ratio_cache = tmp_cache.get(str(self.compress_ratio))
            if not isinstance(ratio_cache, dict):
                raise ValueError(
                    f"Missing CP temporary cache ratio {self.compress_ratio} "
                    f"for layer {self.layer_idx} during offline prefill."
                )
            if "state" in cache_name:
                return self.get_tmp_state_cache(tmp_cache, cache_name)
            return ratio_cache[cache_name]
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

    @staticmethod
    def _get_cp_ratio_plan(attn_metadata: Dict, ratio: int) -> Dict:
        """Return the immutable per-ratio CP plan, if one was materialized.

        The planner is intentionally allowed to evolve its nesting while the
        model is migrated.  Consumers therefore accept both integer and string
        ratio keys, but never derive row coordinates from tensor shapes when a
        plan is available.
        """
        cp_metadata = attn_metadata.get("cp_metadata", {})
        if not isinstance(cp_metadata, dict):
            return {}
        plans = cp_metadata.get("compressor_plans") or {}
        plan = plans.get(str(ratio), plans.get(ratio, {}))
        return plan if isinstance(plan, dict) else {}

    def _get_cp_state_write_plan(self, attn_metadata: Dict) -> Dict:
        """Find the owner-local state write plan for this compressor ratio.

        ``state_write_plans`` is produced by ``AttentionData`` after temporary
        and scheduled state tables are available.  A ratio-local plan may also
        be embedded in ``compressor_plans``.  Return only dictionaries that
        contain the fixed source/destination pairing; CP prefill callers must
        provide this plan before entering the layer.
        """
        cp_metadata = attn_metadata.get("cp_metadata", {})
        if not isinstance(cp_metadata, dict):
            return {}
        ratio_key = str(self.compress_ratio)
        candidates = []
        state_plans = cp_metadata.get("state_write_plans") or {}
        if isinstance(state_plans, dict):
            candidates.extend((
                state_plans.get(ratio_key),
                state_plans.get(self.compress_ratio),
            ))
        ratio_plan = self._get_cp_ratio_plan(attn_metadata, self.compress_ratio)
        if ratio_plan:
            candidates.extend((
                ratio_plan.get("state_write_plan"),
                ratio_plan.get("state_write_plans", {}).get(ratio_key)
                if isinstance(ratio_plan.get("state_write_plans"), dict) else None,
                ratio_plan,
            ))
        for candidate in candidates:
            if isinstance(candidate, dict) and all(
                isinstance(candidate.get(key), torch.Tensor)
                for key in ("src_block_ids", "dst_block_ids", "valid_mask")
            ):
                return candidate
        return {}

    def _get_cp_state_gather_plan(self, attn_metadata: Dict) -> Dict:
        """Return the fixed device-side rank/index map for state gathering."""
        ratio_plan = self._get_cp_ratio_plan(attn_metadata, self.compress_ratio)
        if ratio_plan and (
            ("state_rank_for_block" in ratio_plan or "rank_for_block" in ratio_plan)
            and (
                "safe_state_block_indices" in ratio_plan
                or "state_block_indices" in ratio_plan
                or "block_indices" in ratio_plan
            )
        ):
            return ratio_plan
        # Accept the complete nested state-write plan, but reject partial plans.
        plan = self._get_cp_state_write_plan(attn_metadata)
        if plan and (
            ("state_rank_for_block" in plan or "rank_for_block" in plan)
            and (
                "safe_state_block_indices" in plan
                or "state_block_indices" in plan
                or "block_indices" in plan
            )
        ):
            return plan
        return {}

    def _get_persistent_decode_slot_mapping(self, attn_metadata: Dict) -> Optional[torch.Tensor]:
        """Get the planner-masked natural compressed mapping, when present."""
        plan = self._get_cp_ratio_plan(attn_metadata, self.compress_ratio)
        mapping = plan.get("persistent_decode_slot_mapping")
        if mapping is None:
            mapping = plan.get("persistent_slot_mapping")
        return mapping if isinstance(mapping, torch.Tensor) else None

    def _get_persistent_decode_valid_mask(self, attn_metadata: Dict) -> Optional[torch.Tensor]:
        """Return the natural-row mask paired with the persistent mapping."""
        plan = self._get_cp_ratio_plan(attn_metadata, self.compress_ratio)
        mask = plan.get("persistent_decode_valid_mask")
        if mask is None:
            mask = plan.get("persistent_slot_valid_mask")
        return mask if isinstance(mask, torch.Tensor) else None

    def _select_cp_state_from_plan(
        self,
        state_by_rank: torch.Tensor,
        attn_metadata: Dict,
    ) -> torch.Tensor:
        """Gather final state rows using the planner's reusable rank map.

        ``state_by_rank`` is ``[C, state_block_count, ...]``.  The plan stores
        one rank and block index per state row; invalid rows use rank zero and
        are masked by the plan, preserving the legacy zero-state behavior while
        keeping the gather shape static.
        """
        plan = self._get_cp_state_gather_plan(attn_metadata)
        rank_values = plan.get("state_rank_for_block", plan.get("rank_for_block"))
        block_values = plan.get(
            "safe_state_block_indices",
            plan.get("state_block_indices", plan.get("block_indices")),
        )
        if not isinstance(rank_values, torch.Tensor) or not isinstance(block_values, torch.Tensor):
            # Legacy metadata keeps the original per-block rank selection.
            cp_metadata = attn_metadata.get("cp_metadata", {})
            if not isinstance(cp_metadata, dict):
                cp_metadata = {}
            last_ranks = cp_metadata.get("last_rank_zz")
            if last_ranks is None:
                last_ranks = cp_metadata.get("last_cp_ranks")
            tmp_tables = cp_metadata.get("tmp_block_table", {})
            state_key = f"c{self.compress_ratio}a_cmp_state"
            tmp_table = tmp_tables.get(state_key) if isinstance(tmp_tables, dict) else None
            if last_ranks is None:
                raise ValueError(
                    f"Missing CP state gather plan and legacy state metadata for "
                    f"layer {self.layer_idx}, ratio {self.compress_ratio}."
                )
            last_ranks = torch.as_tensor(
                last_ranks, device=state_by_rank.device, dtype=torch.long
            ).view(-1)
            state_block_count = state_by_rank.shape[1]
            if last_ranks.numel() == state_block_count and tmp_table is None:
                if torch.any((last_ranks < 0) | (last_ranks >= state_by_rank.shape[0])):
                    raise ValueError(
                        f"Invalid CP last-rank metadata for layer {self.layer_idx}, "
                        f"ratio {self.compress_ratio}: cp_size={state_by_rank.shape[0]}."
                    )
                block_indices = torch.arange(
                    state_block_count, device=state_by_rank.device, dtype=torch.long
                )
                return state_by_rank[last_ranks, block_indices]
            if last_ranks.numel() == 1:
                rank_value = last_ranks[0]
                if rank_value < 0 or rank_value >= state_by_rank.shape[0]:
                    raise ValueError(
                        f"Invalid CP last-rank metadata for layer {self.layer_idx}, "
                        f"ratio {self.compress_ratio}: cp_size={state_by_rank.shape[0]}."
                    )
                # Original single-request path.
                return state_by_rank[rank_value]
            if tmp_table is None:
                raise ValueError(
                    f"Missing legacy CP state table for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio}."
                )
            tmp_table = tmp_table.to(device=state_by_rank.device, dtype=torch.long)
            if tmp_table.dim() == 1:
                tmp_table = tmp_table.view(1, -1)
            if tmp_table.shape[0] != last_ranks.numel():
                # Legacy callers may provide one rank per state block.
                if last_ranks.numel() == state_block_count:
                    if torch.any((last_ranks < 0) | (last_ranks >= state_by_rank.shape[0])):
                        raise ValueError(
                            f"Invalid CP last-rank metadata for layer {self.layer_idx}, "
                            f"ratio {self.compress_ratio}: cp_size={state_by_rank.shape[0]}."
                        )
                    block_indices = torch.arange(
                        state_block_count, device=state_by_rank.device, dtype=torch.long
                    )
                    return state_by_rank[last_ranks, block_indices]
                raise ValueError(
                    f"Mismatched legacy CP state rows for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio}: table_rows={tmp_table.shape[0]}, "
                    f"rank_rows={last_ranks.numel()}."
                )
            if torch.any((last_ranks < 0) | (last_ranks >= state_by_rank.shape[0])):
                raise ValueError(
                    f"Invalid CP last-rank metadata for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio}: cp_size={state_by_rank.shape[0]}."
                )
            state_block_count = state_by_rank.shape[1]
            block_ids = tmp_table.reshape(-1)
            request_ranks = last_ranks.view(-1, 1).expand_as(tmp_table).reshape(-1)
            valid = (block_ids > 0) & (block_ids < state_block_count)
            rank_for_block = torch.zeros(
                state_block_count, device=state_by_rank.device, dtype=torch.long
            )
            if torch.any(valid):
                rank_for_block = rank_for_block.scatter(
                    0, block_ids[valid], request_ranks[valid]
                )
            block_indices = torch.arange(
                state_block_count, device=state_by_rank.device, dtype=torch.long
            )
            return state_by_rank[rank_for_block, block_indices]

        rank_values = rank_values.view(-1)
        block_values = block_values.view(-1)
        if rank_values.numel() != block_values.numel():
            raise ValueError(
                f"Mismatched CP state gather plan lengths for layer {self.layer_idx}, "
                f"ratio {self.compress_ratio}: ranks={rank_values.numel()}, "
                f"blocks={block_values.numel()}."
            )
        if rank_values.numel() == 0:
            return state_by_rank.new_empty((0, *state_by_rank.shape[2:]))
        # Invalid rows map to reserved block 0, keeping a fixed gather shape.
        # Older metadata retains the compatibility mask path.
        if "safe_state_block_indices" not in plan:
            valid_mask = plan.get("state_valid_mask")
            if not isinstance(valid_mask, torch.Tensor):
                # Older complete plans have valid rows by construction.
                valid_mask = torch.ones(
                    rank_values.numel(), dtype=torch.bool, device=rank_values.device
                )
            else:
                valid_mask = valid_mask.view(-1)
                if valid_mask.numel() != rank_values.numel():
                    raise ValueError(
                        f"Mismatched CP state valid mask for layer {self.layer_idx}, "
                        f"ratio {self.compress_ratio}: mask={valid_mask.numel()}, "
                        f"rows={rank_values.numel()}."
                    )
            rank_values = torch.where(
                valid_mask, rank_values, torch.zeros_like(rank_values)
            )
        return state_by_rank[rank_values, block_values]

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
        # Flush CP prefill's temporary state cache to the decode cache offline.
        state_key = f"c{self.compress_ratio}a_cmp_state"

        # Warmup block tables are zero-filled; skip the persistent update.
        if attn_metadata["is_warm_up"]:
            return

        # Reuse the owner-local plan built once per prefill; block 0 is the
        # safe sentinel for invalid rows.
        state_plan = self._get_cp_state_write_plan(attn_metadata)
        required_keys = ("src_block_ids", "dst_block_ids", "valid_mask")
        if not all(
            isinstance(state_plan.get(key), torch.Tensor)
            for key in required_keys
        ):
            # Compatibility path for metadata without a complete write plan.
            cp_metadata = attn_metadata.get("cp_metadata", {})
            if not isinstance(cp_metadata, dict):
                cp_metadata = {}
            tmp_tables = cp_metadata.get("tmp_block_table")
            schedule_tables = attn_metadata.get("block_table")
            if not isinstance(tmp_tables, dict):
                tmp_tables = {}
            if not isinstance(schedule_tables, dict):
                schedule_tables = {}
            tmp_block_table = tmp_tables.get(state_key)
            schedule_block_table = schedule_tables.get(state_key)
            if tmp_block_table is None or schedule_block_table is None:
                raise ValueError(
                    f"Missing CP state write plan/tables for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio}."
                )
            tmp_block_ids = self.get_ordered_block_ids(tmp_block_table)
            schedule_block_ids = self.get_ordered_block_ids(schedule_block_table)
            if self.compress_ratio == 4:
                dst_block_ids = schedule_block_ids
            elif self.compress_ratio == 128:
                dst_block_ids = schedule_block_ids[-1:]
            else:
                raise ValueError(
                    f"Unsupported state cache ratio {self.compress_ratio} for "
                    f"layer {self.layer_idx}."
                )
            if tmp_block_ids.numel() != dst_block_ids.numel():
                raise ValueError(
                    f"Mismatched CP state block tables for layer {self.layer_idx}, "
                    f"ratio {self.compress_ratio}: src={tmp_block_ids.numel()}, "
                    f"dst={dst_block_ids.numel()}."
                )
            decode_state_cache[dst_block_ids] = tmp_state_cache[tmp_block_ids]
            return

        src_block_ids = state_plan["src_block_ids"].view(-1)
        dst_block_ids = state_plan["dst_block_ids"].view(-1)
        valid_mask = state_plan["valid_mask"].view(-1)
        if not (
            src_block_ids.numel() == dst_block_ids.numel() == valid_mask.numel()
        ):
            raise ValueError(
                f"Mismatched CP state write plan lengths for layer {self.layer_idx}, "
                f"ratio {self.compress_ratio}."
            )

        # Keep invalid rows at their old values while preserving graph shape.
        new_blocks = tmp_state_cache.index_select(0, src_block_ids)
        old_blocks = decode_state_cache.index_select(0, dst_block_ids)
        broadcast_mask = valid_mask
        while broadcast_mask.dim() < new_blocks.dim():
            broadcast_mask = broadcast_mask.unsqueeze(-1)
        merged_blocks = torch.where(broadcast_mask, new_blocks, old_blocks)
        decode_state_cache.index_copy_(0, dst_block_ids, merged_blocks)

    def update_compress_cache(
        self,
        kv: torch.Tensor,
        cmp_slot_mapping: torch.Tensor,
        is_prefill: bool,
        cache_getter: Callable[[str], torch.Tensor],
        quantized_kv: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        valid_row_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        flat_slots = cmp_slot_mapping.reshape(-1)
        flat_kv = kv.reshape(-1, kv.shape[-1])
        if flat_slots.numel() == 0:
            # Short prefills may produce no compressed row; skip empty writes.
            return kv, None
        if valid_row_indices is not None:
            # Reuse CP's precomputed valid rows; avoid per-layer mask reduction.
            if (valid_row_indices.device != flat_slots.device
                    or valid_row_indices.dtype != torch.long):
                valid_row_indices = valid_row_indices.to(
                    device=flat_slots.device, dtype=torch.long
                )
            valid_row_indices = valid_row_indices.view(-1)
            if valid_row_indices.numel() == 0:
                return kv, None
            if valid_row_indices.numel() != flat_slots.numel():
                flat_slots = flat_slots.index_select(0, valid_row_indices)
                flat_kv = flat_kv.index_select(0, valid_row_indices)
                if quantized_kv is not None:
                    quantized_kv = quantized_kv.view(-1, quantized_kv.shape[-1]) \
                        .index_select(0, valid_row_indices)
                if k_scale is not None:
                    k_scale = k_scale.view(-1, k_scale.shape[-1]) \
                        .index_select(0, valid_row_indices)
        cmp_slot_mapping = flat_slots
        kv = flat_kv
        scatter = torch_npu.npu_scatter_nd_update_
        if self.platform_version == PlatformVersion.A3:
            import custom_ops  # Registers the A3 custom operator binding.
            scatter = torch.ops.custom.scatter_nd_update_asc

        if "index" in self.prefix:
            li_cmp_cache = cache_getter("li_cmp_kv")
            scale_cache = cache_getter("li_key_dequant_scale")
            if self.li_cache_quant_mode in ("float8", "mxfloat4", "hifloat8"):
                quant_cache_kwargs = {
                    "x": kv,
                    "slot_mapping": cmp_slot_mapping,
                    "cache": li_cmp_cache,
                    "cache_scale": scale_cache,
                    "quant_mode": {
                        "float8": "fp8",
                        "mxfloat4": "mxfp4",
                        "hifloat8": "hifloat8",
                    }[self.li_cache_quant_mode],
                }
                if self.li_cache_quant_mode == "hifloat8":
                    quant_cache_kwargs["slot_mapping"] = cmp_slot_mapping.to(
                        device=li_cmp_cache.device, dtype=torch.int32
                    ).contiguous()
                    quant_cache_kwargs["round_scale"] = True
                torch.ops.cann_ops_transformer.indexer_quant_cache(
                    **quant_cache_kwargs,
                )
                return kv, None
            else:
                if quantized_kv is None and k_scale is None:
                    quantized_kv, k_scale = torch_npu.npu_dynamic_quant(kv)
                    k_scale = k_scale.to(torch.float16)
                scatter(scale_cache.view(-1, scale_cache.shape[-1]),
                        cmp_slot_mapping.view(-1, 1),
                        k_scale.view(-1, scale_cache.shape[-1]))
                scatter(li_cmp_cache.view(-1, li_cmp_cache.shape[-1]),
                        cmp_slot_mapping.view(-1, 1),
                        quantized_kv.view(-1, li_cmp_cache.shape[-1]))
                return quantized_kv, k_scale
        else:
            sfa_cmp_cache = cache_getter("sfa_cmp_kv")
            if self.kv_cache_quant_mode == "hifloat8":
                cmp_slot_mapping = cmp_slot_mapping.to(
                    device=sfa_cmp_cache.device, dtype=torch.int32
                ).contiguous()
                sfa_cache_scale = torch.empty(
                    (*sfa_cmp_cache.shape[:-1], 1),
                    dtype=torch.float32,
                    device=sfa_cmp_cache.device,
                )
                sfa_x = kv.view(-1, self.head_dim)
                torch.ops.cann_ops_transformer.indexer_quant_cache(
                    cache=sfa_cmp_cache,
                    cache_scale=sfa_cache_scale,
                    x=sfa_x,
                    slot_mapping=cmp_slot_mapping.view(-1),
                    quant_mode="hifloat8",
                    x_scale=1.0,
                )
            elif self.kv_cache_quant_mode == "float8":
                torch.ops.cann_ops_transformer.kv_compress_epilog(
                    x=kv.view(-1, self.head_dim),
                    slot_mapping=cmp_slot_mapping.view(-1),
                    cache=sfa_cmp_cache,
                    quant_mode="fp8_bf16"
                )
            else:
                scatter(
                    sfa_cmp_cache.view(-1, sfa_cmp_cache.shape[-1]),
                    cmp_slot_mapping.view(-1, 1),
                    kv.view(-1, sfa_cmp_cache.shape[-1]),
                )
            return kv, None

    def compressor_epilog(self, kv: torch.Tensor, attn_metadata: Dict, is_prefill: bool):
        cp_plan = self._get_cp_ratio_plan(attn_metadata, self.compress_ratio)
        temp_valid_row_indices = None
        if is_prefill and self.cp_size > 1:
            # temp cache
            cmp_slot_mapping = attn_metadata["cp_metadata"]["slot_mapping_cmp"][f"{self.compress_ratio}"]
            temp_valid_row_indices = cp_plan.get("temp_valid_row_indices")
            # decode_token_indices marks the cache indices updated by the current rank in CP.
            # A non-empty value means this rank owns at least one request and
            # must update the corresponding decode cache rows.
            decode_token_indices = self.get_decode_token_indices(attn_metadata)
            has_decode_requests = decode_token_indices is not None and decode_token_indices.numel() > 0
        else:
            cmp_slot_mapping = attn_metadata['slot_mapping'][f"c{self.compress_ratio}a_cmp_kv"]
            if is_prefill:
                temp_valid_row_indices = attn_metadata[
                    "cmp_valid_row_indices"
                ][str(self.compress_ratio)]

        # Keep the aggregate payload intact; the planner supplies owner-row
        # coordinates for the optional persistent write.
        source_kv = kv
        self.update_compress_cache(
            kv,
            cmp_slot_mapping,
            is_prefill,
            lambda cache_name: self.get_runtime_cache(cache_name, attn_metadata, is_prefill),
            valid_row_indices=temp_valid_row_indices,
        )

        if is_prefill and self.cp_size > 1 and not self.is_online and has_decode_requests:
            # Consume the planner-filtered natural mapping to keep rows aligned
            # with the all-gather restore plan.
            planned_decode_mapping = self._get_persistent_decode_slot_mapping(attn_metadata)
            persistent_valid_row_indices = cp_plan.get("persistent_valid_row_indices")
            # Persist planner-masked rows in natural source_kv order.
            if planned_decode_mapping is not None:
                self.update_compress_cache(
                    source_kv,
                    planned_decode_mapping,
                    is_prefill,
                    self.get_cache,
                    quantized_kv=None,
                    k_scale=None,
                    valid_row_indices=persistent_valid_row_indices,
                )
        return None

    def compressor_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool,
        attn_metadata_ori: Dict = None
    ):
        cos_sin = attn_metadata["cos_sin"]
        if is_prefill and self.cp_size > 1:
            ratio_key = f"{self.compress_ratio}"
            input_indices = attn_metadata.get("cmp_input_indices", {}).get(ratio_key)
            input_is_identity = attn_metadata.get(
                "cmp_input_indices_is_identity", {}
            ).get(ratio_key, False)
            input_contiguous = attn_metadata.get(
                "cmp_input_contiguous", {}
            ).get(ratio_key)
            if input_is_identity:
                # Planner-proven physical order: no gather needed.
                pass
            elif input_contiguous is not None:
                # Ratio-128/no-borrow fast path: remove the prefix with a view.
                prefix_len = int(input_contiguous["prefix_len"])
                segment_len = int(input_contiguous["segment_len"])
                batch_size = int(input_contiguous["batch_size"])
                x = x.view(
                    batch_size, prefix_len + segment_len, x.shape[-1]
                ).narrow(1, prefix_len, segment_len).reshape(-1, x.shape[-1])
            elif input_indices is not None:
                x = torch.index_select(x, 0, input_indices.to(device=x.device))
            else:
                # Compatibility with older metadata.
                offset = attn_metadata["cmp_in_offset"][ratio_key]
                x = x[offset:]
            cu_seqlens = attn_metadata["cu_seq_lens"][f"{self.compress_ratio}"]
            seq_used_q = attn_metadata["cmp_seq_used_q"][f"{self.compress_ratio}"]
            start_pos = attn_metadata["start_pos"][f"{self.compress_ratio}"]
            if not self.is_online:
                state_block_table = attn_metadata["tmp_block_table"][f"c{self.compress_ratio}a_cmp_state"]
            else:
                state_block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"]
        else:
            attn_metadata_ori = attn_metadata
            cu_seqlens = attn_metadata["cu_seq_lens_q"]
            seq_used_q = attn_metadata["seq_used_q"]
            start_pos = attn_metadata["start_pos"]
            state_block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_state"]
        bsz = seq_used_q.shape[0]

        # Keep the compressor's fixed physical rows through RoPE, then trim
        # to CP's valid-row count before aggregation.
        cp_valid_len = None
        if is_prefill and self.cp_size > 1:
            cp_valid_len = attn_metadata["cmp_valid_lens"][f"{self.compress_ratio}"]
        cmpr_input_kwargs = {
            "x": x.view(-1, self.dim) if is_prefill else x.view(bsz, -1, self.dim),
            "wkv": self.wkv.weight,
            "wgate": self.wgate.weight,
            "ape": self.ape,
            "state_block_table": state_block_table,
            "seqused": seq_used_q,
            "start_pos": start_pos,
            "cmp_ratio": self.compress_ratio,
            "coff": 1 + self.overlap,
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
        rope_cos, rope_sin = cos_sin[f"c{self.compress_ratio}a"]
        if self.mm_quant_mode == "w8a8hifloat8":
            quant_keys = (
                "x", "wkv", "wgate", "ape", "state_block_table", "seqused",
                "start_pos", "cmp_ratio", "coff", "cache_mode", "state_cache",
            )
            quant_kwargs = {key: cmpr_input_kwargs.get(key) for key in quant_keys}
            if is_prefill:
                quant_kwargs["cu_seqlens"] = cmpr_input_kwargs.get("cu_seqlens")
            quant_kwargs.update({
                "x_descale": self.x_descale,
                "wkv_descale": self.wkv.weight_scale.view(-1),
                "wgate_descale": self.wgate.weight_scale.view(-1),
                "quant_mode": 1,
            })
            kv = torch.ops.cann_ops_transformer.quant_compressor(**quant_kwargs)
            kv = torch_npu.npu_rms_norm(
                kv, self.norm.weight, self.norm.variance_epsilon)[0]
            torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
                kv.view(-1, 1, 1, self.head_dim),
                rope_cos,
                rope_sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            if cp_valid_len is not None:
                kv = kv[:cp_valid_len]
        else:
            kv = compressor(**cmpr_input_kwargs)
            kv = torch_npu.npu_rms_norm(
                kv,
                self.norm.weight,
                self.norm.variance_epsilon,
            )[0]
            kv_rope_view = kv.view(*((-1, 1) if is_prefill else (bsz, -1)), 1, self.head_dim)
            rope_cos = rope_cos.view(*((-1, 1) if is_prefill else (bsz, -1)), 1, self.rope_head_dim)
            rope_sin = rope_sin.view(*((-1, 1) if is_prefill else (bsz, -1)), 1, self.rope_head_dim)
            torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
                kv_rope_view,
                rope_cos,
                rope_sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            if cp_valid_len is not None:
                kv = kv[:cp_valid_len]
        return kv

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Dict,
        is_prefill: bool
    ):
        # compressor
        if is_prefill and self.cp_size > 1:
            # Fill fixed zigzag blocks in place; callers without block_len use
            # the compatibility concatenation path.
            compressor_plan = attn_metadata["cp_metadata"].get(
                "compressor_plans", {}
            ).get(str(self.compress_ratio), {})
            planned_block_len = compressor_plan.get("block_len")
            pad_kv = None
            pad_kv_parts = [] if planned_block_len is None else None
            pstr = "li" if "index" in self.prefix else "sfa"
            if self.is_online:
                cur_kv_state = self.get_state_cache_online(pstr + "_kv_state", attn_metadata)
            else:
                kv_state_cache = self.get_runtime_cache(pstr + "_kv_state", attn_metadata, is_prefill)
                cur_kv_state = kv_state_cache.clone()

            for zz_flag in ["prev", "next"]:
                x_seg = x[0] if zz_flag == "prev" else x[1]
                cp_metadata = attn_metadata[zz_flag]
                has_compressor_input = cp_metadata["has_compressor_input"][
                    f"{self.compress_ratio}"
                ]
                if has_compressor_input:
                    kv = self.compressor_prolog(x_seg, cp_metadata, is_prefill, attn_metadata)
                else:
                    # Keep fully padded tails empty; some compressors reject
                    # empty TND input, while cmp_out_pad supplies collective rows.
                    kv = x_seg.new_empty((0, self.head_dim))
                has_end = cp_metadata.get("has_end")
                if has_end is None:
                    # Compatibility fallback.
                    has_end = bool(torch.as_tensor(cp_metadata["is_end"]).any())
                if has_end:
                    if self.is_online:
                        cur_kv_state = self.get_state_cache_online(pstr + "_kv_state", attn_metadata)
                    else:
                        cur_kv_state = kv_state_cache
                cmp_out_pad = cp_metadata["cmp_out_pad"][f"{self.compress_ratio}"]
                pad_tensor = cmp_out_pad[0] if "index" in self.prefix else cmp_out_pad[1]
                if planned_block_len is None:
                    pad_kv_parts.extend((pad_tensor, kv))
                else:
                    if pad_kv is None:
                        pad_kv = pad_tensor.new_empty(
                            (2 * int(planned_block_len), pad_tensor.shape[-1])
                        )
                    half_start = (
                        0 if zz_flag == "prev" else int(planned_block_len)
                    )
                    half = pad_kv.narrow(0, half_start, int(planned_block_len))
                    pad_len = int(pad_tensor.shape[0])
                    if pad_len:
                        half.narrow(0, 0, pad_len).copy_(pad_tensor)
                    kv_len = int(kv.shape[0])
                    if kv_len:
                        half.narrow(0, pad_len, kv_len).copy_(kv)
            if pad_kv is None:
                pad_kv = torch.cat(pad_kv_parts, dim=0)
            all_cmp_kv = kv.new_empty([pad_kv.shape[0] * self.cp_size, kv.shape[-1]])
            dist.all_gather_into_tensor(all_cmp_kv, pad_kv, group=self.comm_manager.get_group("cp_group"))

            # Gather rank-major [prev_B, next_B], then restore natural order.
            # Use the explicit block length; zero is valid for short prompts.
            block_len = attn_metadata["cp_metadata"].get("compressor_plans", {}) \
                .get(str(self.compress_ratio), {}).get("block_len")
            if block_len is None:
                block_len = pad_kv.shape[0] // 2
            aggregate_kv = all_cmp_kv.view(
                self.cp_size, 2, int(block_len), kv.shape[-1]
            ).flatten(0, 2)
            natural_indices = attn_metadata["cp_metadata"].get(
                "compressor_plans", {}
            ).get(str(self.compress_ratio), {}).get("aggregate_to_natural")
            if natural_indices is None:
                kv = aggregate_kv
            else:
                natural_indices = natural_indices.to(device=aggregate_kv.device, dtype=torch.long)
                kv = aggregate_kv.index_select(0, natural_indices)

            all_ks = cur_kv_state.new_empty([cur_kv_state.shape[0] * self.cp_size, *cur_kv_state.shape[1:]])
            dist.all_gather_into_tensor(all_ks, cur_kv_state, group=self.comm_manager.get_group("cp_group"))
            state_by_rank = all_ks.view(self.cp_size, *cur_kv_state.shape)
            if self.is_online or attn_metadata.get("is_warm_up", False):
                # Warmup/online CP has no offline persistent state map.
                last_ranks = attn_metadata["cp_metadata"].get("last_cp_ranks")
                if last_ranks is None:
                    last_ranks = attn_metadata["cp_metadata"].get("last_rank_zz", 0)
                last_ranks = torch.as_tensor(
                    last_ranks, device=state_by_rank.device, dtype=torch.long
                ).view(-1)
                state_key = f"c{self.compress_ratio}a_cmp_state"
                state_table = attn_metadata["cp_metadata"].get(
                    "tmp_block_table", {}
                ).get(state_key)
                if state_table is None:
                    kv_state = state_by_rank.index_select(0, last_ranks[:1]).squeeze(0)
                else:
                    state_table = state_table.to(
                        device=state_by_rank.device, dtype=torch.long
                    )
                    state_block_count = state_by_rank.shape[1]
                    flat_table = state_table.view(-1)
                    valid_blocks = flat_table > 0
                    block_ids = flat_table[valid_blocks]
                    request_ranks = last_ranks.view(-1, 1).expand_as(
                        state_table
                    ).reshape(-1)[valid_blocks]
                    rank_for_block = torch.zeros(
                        state_block_count,
                        dtype=torch.long,
                        device=state_by_rank.device,
                    )
                    if block_ids.numel() > 0:
                        rank_for_block = rank_for_block.scatter(
                            0, block_ids, request_ranks
                        )
                    block_indices = torch.arange(
                        state_block_count,
                        dtype=torch.long,
                        device=state_by_rank.device,
                    )
                    kv_state = state_by_rank[rank_for_block, block_indices]
            else:
                # Offline metadata carries the fixed plan built for this batch.
                kv_state = self._select_cp_state_from_plan(state_by_rank, attn_metadata)

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
