# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

""" PyTorch Attention input model."""
import os
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
from executor.utils import align_up
from .cp_metadata import (
    CP_COMPRESS_RATIOS,
    build_prefill_cp_query_metadata,
    build_prefill_cp_window_plan,
)
from .common_modules import PACKED_KV_STORAGE_DTYPE, get_kv_cache_dim, is_packed_kv_layout


def pad_tensor_tnd(ori_tensor, total_len, pad_value):
    new_tensor = torch.full((total_len,), pad_value, dtype=ori_tensor.dtype, device=ori_tensor.device)
    new_tensor[:len(ori_tensor)] = ori_tensor
    return new_tensor


def calc_num_ratio_group(next_n, overlap):
    return (1 if next_n == 0 else 2) + overlap # additional +1 for overlap


class AttnMetaData(nn.Module):
    def __init__(self, config, comm_manager, infer_config: InferenceConfig, is_mtp=False):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        self.block_size = self.infer_config.scheduler_config.block_size
        self.next_n = self.infer_config.model_config.next_n
        self.is_mtp = is_mtp
        self.slot_mapping_pad_value = -1
        self.position_ids_pad_value = 1
        self.platform_version = self.infer_config.model_config.platform_version

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.window_size = config.sliding_window

        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.metadata_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.shared_expert_stream = torch.npu.Stream() if self.enable_multi_streams else None
        # mla_stream, compressor_stream and indexer_stream are exclusively used in the decoding phase.
        self.mla_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.compressor_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.indexer_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16"
        )
        self.update_kv_quant_settings()
        self.update_gmm_quant_mode()
        self.pre_pad_num_block = 1
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode
        self.init_cache_dim()
        self.cache_dtype_map = {
            "int8": torch.int8,
            "float8": torch.float8_e4m3fn,
            "unquant": torch.bfloat16,
            "hifloat8": torch.uint8,
            "mxfloat4": torch.float4_e2m1fn_x2
        }

    def update_kv_quant_settings(self):
        # if quant to hif8/fp8/mxfp8, set kv cache and li cache accordingly; if quant to int8, set li cache to int8
        if self.platform_version == PlatformVersion.ASCEND_950 and "hifloat" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "hifloat8"
            self.config.quant_config.li_cache_quant_mode = "hifloat8"
        elif self.platform_version == PlatformVersion.ASCEND_950 and "float" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "float8"
            if (self.config.quant_config.li_cache_quant_mode is None or
                    self.config.quant_config.li_cache_quant_mode == "unquant"):
                self.config.quant_config.li_cache_quant_mode = "float8"
            elif "float4" in self.config.quant_config.li_cache_quant_mode:
                # MXFP4 LI cache is independent of the Linear weight quant mode.
                self.config.quant_config.li_cache_quant_mode = "mxfloat4"
        else:
            self.config.quant_config.li_cache_quant_mode = "int8"

    def update_gmm_quant_mode(self):
        if self.platform_version == PlatformVersion.ASCEND_950 and \
            "w4" in self.config.quant_config.gmm_quant_mode and "mx" not in self.config.quant_config.gmm_quant_mode:
            self.config.quant_config.gmm_quant_mode = \
                self.config.quant_config.gmm_quant_mode.replace("float", "mxfloat")

    def init_cache_dim(self):
        self.cache_dim = get_kv_cache_dim(
            self.config.head_dim, self.config.qk_rope_head_dim, self.kv_cache_quant_mode)

    def get_cmp_kv_dtype(self):
        use_fused_kernel_compressor = (
            self.infer_config.model_config.custom_params.get("kernel_config", {}).get("compressor", "native")
            == "ascendc"
        )
        cmp_kv_dtype = torch.float8_e4m3fn if (
            self.platform_version == PlatformVersion.ASCEND_950 and use_fused_kernel_compressor
        ) else torch.bfloat16
        if self.kv_cache_quant_mode in ("float8", "hifloat8"):
            cmp_kv_dtype = PACKED_KV_STORAGE_DTYPE
        return cmp_kv_dtype

    def get_kv_cache_dtype(self):
        if is_packed_kv_layout(self.kv_cache_quant_mode):
            return PACKED_KV_STORAGE_DTYPE
        return self.cache_dtype_map[self.kv_cache_quant_mode]

    def create_cache(self, block_num, dim, dtype, device):
        shape = (block_num, self.block_size, 1, dim)
        # Ascend does not implement zeros for E8M0. Allocate its one-byte
        # physical storage as uint8, then restore the logical dtype for kernels.
        if dtype == torch.float8_e8m0fnu:
            return torch.zeros(shape, dtype=torch.uint8, device=device).view(dtype)
        return torch.zeros(shape, dtype=dtype, device=device)

    def create_state_cache(self, state_block_num, compress_ratio, cache_dim, device):
        assert compress_ratio in CP_COMPRESS_RATIOS
        overlap_num = 2 if compress_ratio == 4 else 1
        return torch.zeros(
            (state_block_num, self.block_size, 2, overlap_num, cache_dim),
            dtype=torch.float32,
            device=device,
        )

    def build_cp_tmp_block_table(self, batch_size, block_num_per_batch, device):
        block_ids = torch.arange(
            batch_size * block_num_per_batch,
            dtype=torch.int32,
            device=device,
        ).view(batch_size, block_num_per_batch)
        return block_ids + 1

    def build_cp_tmp_state_block_table(
        self,
        padded_token_num,
        ratio,
        block_num_per_batch,
        batch_size_per_rank,
        attn_metadata,
    ):
        actual_seq_len = attn_metadata['start_pos'] + attn_metadata['seq_used_q']
        block_table_len = math.ceil(padded_token_num / self.block_size)
        compressed_len = ratio * 2 if ratio == 4 else ratio  # 2: when ratio is 4 require overlap
        actual_block_end_pos = \
            ((actual_seq_len - 1) // self.block_size).view(batch_size_per_rank, 1).repeat(1, block_table_len)
        block_pos_ids = torch.arange(
            block_table_len, dtype=torch.int32, device=actual_seq_len.device
        ).repeat(batch_size_per_rank, 1)
        block_table_offset = torch.arange(
            batch_size_per_rank * block_num_per_batch,
            dtype=torch.int32,
            device=actual_seq_len.device,
        ).view(batch_size_per_rank, -1) + self.pre_pad_num_block
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table_offset = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]

        block_table = torch.zeros(
            (batch_size_per_rank, block_table_len), dtype=torch.int32, device=actual_seq_len.device
        )
        if ratio == 128:
            # When compression ratio is 128, the c128a scheme uses a single block
            # to store the compressed remainder data. Therefore, only the last
            # valid block of each sequence needs to be marked in block_table.
            actual_block_end_pos = ((actual_seq_len.clamp_min(1) - 1) // self.block_size).view(-1)
            row_ids = torch.arange(batch_size_per_rank, dtype=torch.long, device=actual_seq_len.device)
            block_ids = row_ids * block_num_per_batch + self.pre_pad_num_block
            block_table[row_ids, actual_block_end_pos.to(torch.long).clamp_max(block_table_len - 1)] = block_ids.to(torch.int32)
            return block_table
        actual_block_start_pos = ((actual_seq_len - compressed_len - 1) // self.block_size).\
            view(batch_size_per_rank, 1).repeat(1, block_table_len)
        block_table = torch.where(
            block_pos_ids >= actual_block_start_pos,
            block_table_offset,
            block_table
        )
        block_table = torch.where(
            block_pos_ids <= actual_block_end_pos,
            block_table,
            0
        )
        return block_table

    @staticmethod
    def _cp_table_rows_for_owners(
        table: torch.Tensor,
        owner_indices: torch.Tensor,
        batch_size: int,
        table_name: str,
    ) -> torch.Tensor:
        """Return owner-local state rows while preserving block IDs and order."""
        if table is None:
            return None
        if table.dim() == 1:
            table = table.view(1, -1)
        owner_indices = owner_indices.to(device=table.device, dtype=torch.long).view(-1)
        owner_count = owner_indices.numel()
        if owner_count == 0:
            return table[:0]
        if table.shape[0] == batch_size:
            return table.index_select(0, owner_indices)
        if table.shape[0] == owner_count:
            return table
        raise ValueError(
            f"{table_name} must have either all-request rows ({batch_size}) "
            f"or owner-local rows ({owner_count}), got {table.shape[0]}."
        )

    def _build_cp_state_write_plan(
        self,
        ratio: int,
        tmp_table: torch.Tensor,
        schedule_table: torch.Tensor,
        owner_indices: torch.Tensor,
        last_cp_ranks: torch.Tensor,
        batch_size: int,
        cp_tmp_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build fixed owner writes and reusable state-gather maps.

        Block IDs are 1-based; block 0 is the null sentinel. Ratio 4 pairs
        ordered non-null blocks, while ratio 128 keeps each owner's final one.
        Empty rows use block 0 with ``valid_mask=False`` to preserve shape.
        """
        if tmp_table is None or schedule_table is None:
            return {}
        owner_indices = owner_indices.to(device=tmp_table.device, dtype=torch.long).view(-1)
        owner_count = owner_indices.numel()
        tmp_owner = self._cp_table_rows_for_owners(
            tmp_table, owner_indices, batch_size, f"temporary c{ratio}a state table"
        )
        schedule_owner = self._cp_table_rows_for_owners(
            schedule_table, owner_indices, batch_size, f"scheduled c{ratio}a state table"
        )
        tmp_rows = tmp_owner.detach().cpu().tolist()
        schedule_rows = schedule_owner.detach().cpu().tolist()

        def ordered_nonzero(row):
            """Deduplicate cyclic block-table entries without changing order."""
            ordered = []
            seen = set()
            for block_id in row:
                block_id = int(block_id)
                if block_id > 0 and block_id not in seen:
                    ordered.append(block_id)
                    seen.add(block_id)
            return ordered

        # Tables may be cyclically extended; deduplicate IDs before pairing so
        # ratio-4 writes do not depend on physical table width.
        tmp_owner_rows = [ordered_nonzero(row) for row in tmp_rows]
        schedule_owner_rows = [ordered_nonzero(row) for row in schedule_rows]

        src_ids = []
        dst_ids = []
        valid_pairs = []
        owner_row_ids = []
        if last_cp_ranks is None:
            raise ValueError(
                f"last_cp_ranks is required to build c{ratio}a state plan."
            )
        all_ranks = last_cp_ranks.to(device=tmp_table.device, dtype=torch.long).view(-1)
        if all_ranks.numel() == batch_size:
            owner_rank_values = all_ranks.index_select(0, owner_indices)
        elif all_ranks.numel() == owner_count:
            # Preserve row order when tables and ranks are owner-local.
            owner_rank_values = all_ranks
        for owner_row, (tmp_valid, schedule_valid) in enumerate(
            zip(tmp_owner_rows, schedule_owner_rows)
        ):
            if ratio == 4:
                for src_block, dst_block in zip(tmp_valid, schedule_valid):
                    src_ids.append(src_block)
                    dst_ids.append(dst_block)
                    valid_pairs.append(True)
                    owner_row_ids.append(owner_row)
            elif ratio == 128:
                # C128A has one state result per request: use its final block.
                src_block = tmp_valid[-1] if tmp_valid else 0
                dst_block = schedule_valid[-1] if schedule_valid else 0
                src_ids.append(src_block)
                dst_ids.append(dst_block)
                valid_pairs.append(bool(src_block and dst_block))
                owner_row_ids.append(owner_row)

        device = tmp_table.device
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(dst_ids, dtype=torch.long, device=device)
        pair_mask = torch.tensor(valid_pairs, dtype=torch.bool, device=device)
        owner_row_tensor = torch.tensor(owner_row_ids, dtype=torch.long, device=device)

        # Build a capacity-sized rank map indexed by temporary block ID; block
        # 0 remains invalid/safe. Prefer the allocated cache capacity.
        state_capacity = 0
        if cp_tmp_cache:
            state_by_layer = cp_tmp_cache.get("state_cache_by_layer", {})
            for layer_cache in state_by_layer.values():
                for state_cache in layer_cache.values():
                    if isinstance(state_cache, torch.Tensor):
                        state_capacity = int(state_cache.shape[0])
                        break
                if state_capacity:
                    break
        # Include all temporary rows for all-gather; owner filtering applies
        # only to persistent writes.
        if tmp_table.dim() == 1:
            full_tmp_table = tmp_table.view(1, -1)
        else:
            full_tmp_table = tmp_table
        full_batch_size = int(full_tmp_table.shape[0])
        if full_batch_size == batch_size:
            full_tmp_rows = full_tmp_table.detach().cpu().tolist()
            full_request_indices = list(range(batch_size))
        elif full_batch_size == owner_count:
            # Owner-local tables cannot describe non-owner IDs; leave them
            # invalid rather than assigning unknown blocks to rank 0.
            full_tmp_rows = full_tmp_table.detach().cpu().tolist()
            full_request_indices = [int(value) for value in owner_indices.detach().cpu().tolist()]
        else:
            raise ValueError(
                f"temporary c{ratio}a state table must have either all-request rows "
                f"({batch_size}) or owner-local rows ({owner_count}), "
                f"got {full_batch_size}."
            )

        full_tmp_rows = [ordered_nonzero(row) for row in full_tmp_rows]
        max_tmp_id = max((block_id for row in full_tmp_rows for block_id in row), default=0)
        if state_capacity <= 0:
            state_capacity = max_tmp_id + 1
        if all_ranks.numel() == batch_size:
            rank_by_request = {
                request_idx: int(all_ranks[request_idx].detach().cpu().item())
                for request_idx in range(batch_size)
            }
        elif all_ranks.numel() == owner_count and len(full_request_indices) == owner_count:
            rank_by_request = {
                request_idx: int(all_ranks[row_idx].detach().cpu().item())
                for row_idx, request_idx in enumerate(full_request_indices)
            }
        else:
            raise ValueError(
                f"last_cp_ranks cannot map the temporary state table: "
                f"entries={all_ranks.numel()}, table_rows={full_batch_size}, "
                f"batch_size={batch_size}, owner_count={owner_count}."
            )
        state_block_indices = torch.arange(state_capacity, dtype=torch.long, device=device)
        state_rank_for_block = torch.zeros(state_capacity, dtype=torch.long, device=device)
        state_valid_mask = torch.zeros(state_capacity, dtype=torch.bool, device=device)
        block_owner = {}
        for request_idx, tmp_row in zip(full_request_indices, full_tmp_rows):
            rank_value = rank_by_request.get(request_idx, 0)
            for block_id in tmp_row:
                block_id = int(block_id)
                if 0 < block_id < state_capacity:
                    previous_rank = block_owner.get(block_id)
                    block_owner[block_id] = rank_value
                    state_rank_for_block[block_id] = rank_value
                    state_valid_mask[block_id] = True

        # Fold invalid rows into reserved block 0 once so layers keep a fixed
        # gather shape without per-layer masking.
        safe_state_block_indices = state_block_indices.masked_fill(
            ~state_valid_mask, 0
        )

        return {
            "ratio": int(ratio),
            "owner_request_indices": owner_indices,
            "owner_tmp_state_block_table": tmp_owner,
            "owner_schedule_state_block_table": schedule_owner,
            "src_block_ids": src_tensor,
            "dst_block_ids": dst_tensor,
            "valid_mask": pair_mask,
            "owner_row_indices": owner_row_tensor,
            "state_block_indices": state_block_indices,
            "block_indices": state_block_indices,
            "safe_state_block_indices": safe_state_block_indices,
            "state_rank_for_block": state_rank_for_block,
            "rank_for_block": state_rank_for_block,
            "state_valid_mask": state_valid_mask,
        }

    def build_cp_tmp_state_cache_by_layer(
        self,
        ratio,
        state_block_num,
        device,
    ):
        tmp_state_cache_by_layer = {}
        compress_ratios = self.config.compress_ratios
        for layer_idx, layer_ratio in enumerate(compress_ratios):
            if layer_ratio != ratio:
                continue
            state_cache = {
                "sfa_kv_state": self.create_state_cache(
                    state_block_num, ratio, self.config.head_dim, device),
            }
            if ratio == 4:
                state_cache["li_kv_state"] = self.create_state_cache(
                    state_block_num, ratio, self.config.index_head_dim, device)
            tmp_state_cache_by_layer[layer_idx] = state_cache
        return tmp_state_cache_by_layer

    def build_cp_prefill_tmp_cache(self, input_ids, attn_metadata, cp_metadata):
        """Allocate full-length temporary compressed KV/state cache for CP prefill."""
        if cp_metadata is None or not getattr(cp_metadata, "enabled", False):
            return None

        batch_size = attn_metadata["start_pos"].numel()
        padded_seq_lens = list(cp_metadata.padded_seq_lens)
        padded_token_num = sum(padded_seq_lens)
        device = input_ids.device
        cmp_kv_dtype = self.get_cmp_kv_dtype()
        li_cache_dtype = self.cache_dtype_map[self.li_cache_quant_mode]

        tmp_cache = {}
        tmp_block_table = {}
        for ratio in CP_COMPRESS_RATIOS:
            cmp_cache_lens = [math.ceil(length / ratio) for length in padded_seq_lens]
            cmp_block_num_per_batch = max(
                (math.ceil(length / self.block_size) for length in cmp_cache_lens), default=0
            )
            cmp_block_num = cmp_block_num_per_batch * batch_size + 1
            # state_cache_size = calc_num_ratio_group(self.next_n, overlap=1 if ratio == 4 else 0) * ratio
            state_block_num_per_batch = 2 if ratio == 4 else 1#math.ceil(state_cache_size / self.block_size)
            state_block_num = state_block_num_per_batch * batch_size + 1

            ratio_cache = {
                "sfa_cmp_kv": self.create_cache(cmp_block_num, self.cache_dim, cmp_kv_dtype, device),
                "state_cache_by_layer": self.build_cp_tmp_state_cache_by_layer(
                    ratio, state_block_num, device),
            }
            if ratio == 4:
                li_cache_dim = self.config.index_head_dim
                if self.li_cache_quant_mode == "mxfloat4":
                    li_cache_dim = self.config.index_head_dim // 2
                scale_dtype = (
                    torch.float8_e8m0fnu
                    if self.li_cache_quant_mode == "mxfloat4"
                    else torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
                )
                ratio_cache.update({
                    "li_cmp_kv": self.create_cache(
                        cmp_block_num, li_cache_dim, li_cache_dtype, device)
                })
                if self.li_cache_quant_mode in ["int8", "float8", "hifloat8", "mxfloat4"]:
                    scale_dim = self.config.index_head_dim // 32 if self.li_cache_quant_mode == "mxfloat4" else 1
                    ratio_cache["li_key_dequant_scale"] = self.create_cache(
                        cmp_block_num, scale_dim, scale_dtype, device)

            tmp_cache[str(ratio)] = ratio_cache

            tmp_block_table[f"c{ratio}a_cmp_kv"] = self.build_cp_tmp_block_table(
                batch_size, cmp_block_num_per_batch, device)
            tmp_block_table[f"c{ratio}a_cmp_state"] = self.build_cp_tmp_state_block_table(
                max(padded_seq_lens, default=0),
                ratio,
                state_block_num_per_batch,
                batch_size,
                attn_metadata
            )
            attn_metadata["cp_metadata"].update({
                "tmp_block_table": tmp_block_table
            })

        return tmp_cache

    def pad_tensor_tnd(self, ori_tensor, total_len, pad_value):
        new_tensor = torch.full((total_len,), pad_value, dtype=ori_tensor.dtype, device=ori_tensor.device)
        new_tensor[:len(ori_tensor)] = ori_tensor
        return new_tensor

    def generate_compressed_position_ids(
        self,
        attn_metadata,
        ratio
    ):
        """
        TND padding format, put all pads values at the back.

        Padding example with batch_size 3, max_seq_len 3, kv_len [2, 3, 1], pad value 1
        TND padding format: [0, 1, 0, 1, 2, 0, 1, 1, 1]
        BNSD padding format: [0, 1, 1, 0, 1, 2, 0, 1, 1]
        """
        start_pos = attn_metadata["start_pos"] // ratio
        bsz = start_pos.shape[0]
        end_pos = (attn_metadata["start_pos"] + attn_metadata["seq_used_q"]) // ratio
        compressed_len = end_pos - start_pos
        offsets = torch.nn.functional.pad(torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0))[:-1]
        expanded_starts = torch.repeat_interleave(start_pos, compressed_len)
        expanded_offsets = torch.repeat_interleave(offsets, compressed_len)
        flat_range = torch.arange(compressed_len.sum(), dtype=torch.int32, device=compressed_len.device)
        compressed_ids = flat_range - expanded_offsets + expanded_starts
        # The TND compressor allocates its physical output as
        # min(T, T // ratio + B), including tail padding.  Its post-op RoPE
        # input must have the same row count; slot mapping marks the extra
        # rows as invalid.  CP keeps the valid-row count separately for its
        # aggregate plan and trims the physical tail after RoPE.
        total_len = int(attn_metadata["cu_seq_lens_q"][-1].item())
        max_len = min(total_len, total_len // ratio + bsz)
        position_ids_cmp = self.pad_tensor_tnd(compressed_ids, max_len, self.position_ids_pad_value)
        return compressed_len, position_ids_cmp

    def generate_compressed_position_ids_bsnd(
        self,
        attn_metadata,
        ratio
    ):
        """
        Padding example with batch_size 3, max_seq_len 3, kv_len [2, 3, 1], pad value 1
        BNSD padding format: [0, 1, 1, 0, 1, 2, 0, 1, 1]
        """
        start_pos = attn_metadata["start_pos"] // ratio
        bsz = start_pos.shape[0]
        end_pos = (attn_metadata["start_pos"] + attn_metadata["seq_used_q"]) // ratio
        compressed_len = end_pos - start_pos
        seq_len = 1 + self.infer_config.model_config.next_n
        max_len = (seq_len + ratio - 1) // ratio
        idx = torch.arange(max_len, dtype=torch.int32, device=start_pos.device).expand(bsz, max_len)
        mask = idx < compressed_len.unsqueeze(1)
        idx = idx + start_pos.unsqueeze(1)
        position_ids_cmp = torch.where(mask, idx, self.position_ids_pad_value)
        return mask, position_ids_cmp

    def get_slot_mapping_from_block_table_bsnd(
        self,
        mask,
        position_ids_cmp,
        block_table
    ):
        bsz, seq_len = position_ids_cmp.shape
        row_indices = torch.arange(bsz, dtype=torch.int32, device="npu").view(-1, 1)
        # slot mapping without padding
        block_idx = position_ids_cmp // self.block_size
        block_offset = position_ids_cmp % self.block_size

        slot_mapping = block_table[row_indices, block_idx] * self.block_size + block_offset
        slot_mapping = slot_mapping.view(bsz, seq_len)
        return torch.where(mask, slot_mapping, self.slot_mapping_pad_value)

    def get_slot_mapping_from_block_table(
        self,
        q_len,
        position_ids,
        block_table
    ):
        bsz = q_len.shape[0]
        position_ids = position_ids.view(-1)
        row_indices = torch.repeat_interleave(torch.arange(bsz, dtype=q_len.dtype, device=q_len.device), q_len)
        # slot mapping without padding
        pad_length = position_ids.shape[0] - row_indices.shape[0]
        if pad_length > 0:
            indices = position_ids[:(-pad_length)]
        else:
            indices = position_ids
        slot_mapping = (
            block_table[row_indices, indices // self.block_size] * self.block_size
            + indices % self.block_size
        )
        return self.pad_tensor_tnd(
            slot_mapping.view(-1).to(torch.int32), position_ids.shape[0], self.slot_mapping_pad_value
            )

    def get_cmp_metadata(self, attn_metadata, is_prefill):
        attn_metadata['position_ids_c'] = {}
        if is_prefill:
            # Prefill compressor output is packed at the front of the fixed
            # physical buffer; materialize its valid row coordinates once so
            # every compressor layer can skip padding reductions.
            attn_metadata['cmp_valid_row_indices'] = {}
        for ratio in CP_COMPRESS_RATIOS:
            # 1. position_ids
            position_ids_func = self.generate_compressed_position_ids if is_prefill \
                else self.generate_compressed_position_ids_bsnd
            mask, position_ids_cmp = position_ids_func(attn_metadata, ratio)
            attn_metadata['position_ids_c'].update(
                {str(ratio): position_ids_cmp * ratio}
            )
            # 2. block_table
            kv_block_table = attn_metadata["block_table"][f"c{ratio}a_cmp_kv"]

            slot_mapping_func = self.get_slot_mapping_from_block_table if is_prefill \
                else self.get_slot_mapping_from_block_table_bsnd
            cmp_slot_mapping = slot_mapping_func(
                mask,
                position_ids_cmp,
                kv_block_table,
            )
            attn_metadata["slot_mapping"][f"c{ratio}a_cmp_kv"] = cmp_slot_mapping
            if is_prefill:
                valid_len = int(mask.sum().item())
                attn_metadata['cmp_valid_row_indices'][str(ratio)] = torch.arange(
                    valid_len, dtype=torch.long, device=cmp_slot_mapping.device
                )

    def build_prefill_full_kv_metadata(self, input_ids, actual_seq_lengths_q):
        actual_seq_lengths_q = actual_seq_lengths_q.view(-1)
        block_nums = (actual_seq_lengths_q + self.block_size - 1) // self.block_size
        full_block_num = int(block_nums.sum().item())
        max_block_num_per_batch = int(block_nums.max().item()) if block_nums.numel() > 0 else 0
        bsz = actual_seq_lengths_q.shape[0]

        full_block_table = torch.zeros(
            (bsz, max_block_num_per_batch),
            dtype=torch.int32,
            device=input_ids.device,
        )
        slot_mappings = []
        next_block_id = 1
        for batch_idx, (seq_len, block_num) in enumerate(zip(actual_seq_lengths_q.tolist(), block_nums.tolist())):
            if block_num == 0:
                continue

            block_ids = torch.arange(
                next_block_id,
                next_block_id + block_num,
                dtype=torch.int32,
                device=input_ids.device,
            )
            full_block_table[batch_idx, :block_num] = block_ids

            local_position_ids = torch.arange(
                seq_len,
                dtype=torch.int64,
                device=input_ids.device,
            )
            block_indices = local_position_ids // self.block_size
            position_offsets = local_position_ids % self.block_size
            cur_slot_mapping = block_ids[block_indices].to(position_offsets.dtype) * self.block_size + position_offsets
            slot_mappings.append(cur_slot_mapping.to(torch.int64))
            next_block_id += block_num

        full_slot_mapping = torch.cat(slot_mappings, dim=0) if slot_mappings else torch.empty(
            (0,),
            dtype=torch.int64,
            device=input_ids.device,
        )

        full_kv_cache = torch.zeros(
            (
                full_block_num + 1,
                self.block_size,
                1,
                self.cache_dim,
            ),
            dtype=self.get_kv_cache_dtype(),
            device=input_ids.device,
        )
        return full_kv_cache, full_block_table, full_slot_mapping

    def build_attn_metadata(self, input_ids, position_ids, forward_metadata):
        is_prefill = forward_metadata.is_prefill
        metadata_get = forward_metadata.get if hasattr(forward_metadata, "get") else \
            lambda key: getattr(forward_metadata, key)

        actual_seq_lengths_q = metadata_get("actual_seq_lengths_q").to(torch.int32)
        actual_seq_lengths_cu_q = metadata_get("actual_seq_lengths_cu_q").to(torch.int32)
        actual_seq_lengths_kv = metadata_get("actual_seq_lengths_kv").to(torch.int32)
        bsz = actual_seq_lengths_q.shape[0]
        cp_forward_meta = metadata_get("cp_metadata") if is_prefill and self.cp_size > 1 else None

        cu_seq_lens_q = torch.cat(
            [torch.zeros_like(actual_seq_lengths_cu_q[:1]), actual_seq_lengths_cu_q],
            dim=0,
        )

        if is_prefill:
            start_pos = torch.zeros([bsz], device=input_ids.device, dtype=torch.int32)
            full_lengths = actual_seq_lengths_q
            if cp_forward_meta is not None and getattr(cp_forward_meta, "padded_seq_lens", None):
                full_lengths = torch.tensor(
                    cp_forward_meta.padded_seq_lens,
                    dtype=actual_seq_lengths_q.dtype,
                    device=input_ids.device,
                )
            full_kv_cache, full_block_table, full_slot_mapping = self.build_prefill_full_kv_metadata(
                input_ids, full_lengths
            )
        else:
            start_pos = (
                metadata_get("kv_len")
                - self.infer_config.model_config.next_n
            ).to(dtype=torch.int32)
        if is_prefill and self.cp_size > 1:
            attn_metadata = {
                "is_prefill": is_prefill,
                "is_warm_up": forward_metadata.is_warm_up,
                "batch_size_per_rank": self.infer_config.scheduler_config.batch_size_per_dp_rank,
                "slot_mapping": metadata_get("slot_mapping"),
                "block_table": metadata_get("block_table"),
                "position_ids": position_ids, # padded
                "kv_len": actual_seq_lengths_q,
                "start_pos": start_pos,
                "actual_seq_q": actual_seq_lengths_q,
                "actual_seq_k": actual_seq_lengths_kv,
                "cu_seq_lens_q": cu_seq_lens_q,
                "seq_used_q": actual_seq_lengths_q,
                "kernel_metadata": {}
            }
        else:
            attn_metadata = {
                "is_prefill": is_prefill,
                "batch_size_per_rank": self.infer_config.scheduler_config.batch_size_per_dp_rank,
                "slot_mapping": metadata_get("slot_mapping"),
                "block_table": metadata_get("block_table"),
                "position_ids": position_ids,
                "kv_len": actual_seq_lengths_cu_q if is_prefill else position_ids + 1,
                "start_pos": start_pos,
                "actual_seq_q": actual_seq_lengths_cu_q,
                "actual_seq_k": actual_seq_lengths_kv,
                "cu_seq_lens_q": cu_seq_lens_q,
                "seq_used_q": actual_seq_lengths_q,
                "kernel_metadata": {}
            }
        if not self.is_mtp and not (is_prefill and self.cp_size > 1):
            self.get_cmp_metadata(attn_metadata, is_prefill)
        attn_metadata["shared_expert_stream"] = self.shared_expert_stream
        attn_metadata["mla_stream"] = self.mla_stream

        if is_prefill:
            if attn_metadata["block_table"] is None:
                attn_metadata["block_table"] = {}
            if attn_metadata["slot_mapping"] is None:
                attn_metadata["slot_mapping"] = {}
            attn_metadata["block_table"].update({"full_kv": full_block_table})
            attn_metadata["slot_mapping"].update({"full_kv": full_slot_mapping})
            attn_metadata["full_kv_cache"] = full_kv_cache
        else:
            attn_metadata["metadata_stream"] = self.metadata_stream
            attn_metadata["compressor_stream"] = self.compressor_stream
            attn_metadata["indexer_stream"] = self.indexer_stream

        if is_prefill and self.cp_size > 1:
            attn_metadata = self.get_cp_metadata(
                input_ids,
                attn_metadata,
                metadata_get("cp_metadata"),
                self.is_mtp,
            )
            for zigzag_flag in ["prev", "next"]:
                zigzag_metadata = attn_metadata.get(zigzag_flag)
                if zigzag_metadata is None:
                    continue
                self._add_compressed_seq_metadata(zigzag_metadata)
        else:
            self._add_compressed_seq_metadata(attn_metadata)

        return attn_metadata

    def _add_compressed_seq_metadata(self, attn_metadata):
        # Normalize the shared packed-sequence length fields once.
        for key in ("cu_seq_lens_q", "seq_used_q", "actual_seq_k"):
            value = attn_metadata[key]
            if value.dtype != torch.int32:
                value = value.to(dtype=torch.int32)
                attn_metadata[key] = value
        actual_seq_k = attn_metadata["actual_seq_k"]
        compressed_seq_lens = {}
        compressed_seq_remainders = {}
        for ratio in dict.fromkeys(self.config.compress_ratios):
            if ratio <= 1:
                continue
            ratio_key = f"{ratio}"
            compressed_seq_lens[ratio_key] = actual_seq_k // ratio
            compressed_seq_remainders[ratio_key] = actual_seq_k % ratio
        attn_metadata["compressed_seq_lens"] = compressed_seq_lens
        attn_metadata["compressed_seq_remainders"] = compressed_seq_remainders

    def _set_cp_persistent_window_slots(self, attn_metadata, cp_metadata, seq_lens, device):
        """Project compact scheduler slots to one fixed W-row block/request."""
        compact_win_slots = attn_metadata.get("slot_mapping", {}).get("win_kv")
        if compact_win_slots is None:
            return

        owner_requests = set(
            cp_metadata.persistent_request_indices.detach().cpu().tolist()
            if cp_metadata.persistent_request_indices is not None
            else range(len(seq_lens))
        )
        compact_starts = list(cp_metadata.compact_starts)
        last_win_slots = []
        for request_idx, seq_len in enumerate(seq_lens):
            take = min(self.window_size, seq_len)
            if request_idx not in owner_requests:
                last_win_slots.append(torch.full(
                    (self.window_size,), -1,
                    dtype=compact_win_slots.dtype,
                    device=device,
                ))
                continue
            pad = torch.full(
                (self.window_size - take,), -1,
                dtype=compact_win_slots.dtype,
                device=device,
            )
            start = compact_starts[request_idx]
            last_win_slots.append(torch.cat((
                pad,
                compact_win_slots[start + seq_len - take:start + seq_len],
            )))
        # Keep the compact scheduler mapping immutable so Main and MTP can
        # rebuild metadata from the same ForwardMetaData.
        attn_metadata.setdefault("cp_metadata", {})[
            "persistent_win_slot_mapping"
        ] = torch.cat(last_win_slots)

    @staticmethod
    def _cp_comp_len(starts, segment_idx, current_valid_len, valid_lens, ratio):
        """Return valid same-request overlap rows for one CP segment."""
        if segment_idx == 0:
            return torch.zeros_like(current_valid_len)
        overlap_len = ratio if ratio == 4 else 0
        raw = (starts % ratio) + overlap_len
        prev_valid = valid_lens[:, segment_idx - 1].to(
            device=starts.device, dtype=starts.dtype
        )
        comp_len = torch.minimum(torch.minimum(raw, starts), prev_valid)
        return torch.where(current_valid_len > 0, comp_len, torch.zeros_like(comp_len))

    def get_cp_metadata(self, input_ids, attn_metadata, cp_metadata, is_mtp):
        attn_metadata_ori = attn_metadata
        device = input_ids.device
        segment_num = 2 * self.cp_size
        batch_size = int(attn_metadata_ori["start_pos"].numel())
        if cp_metadata is None:
            # Only bypass callers reach this branch; Engine attaches metadata.
            return attn_metadata_ori

        split_list_hidden = list(cp_metadata.split_list)
        split_position_ids = list(attn_metadata_ori["position_ids"].split(split_list_hidden, dim=-1))
        segment_lens = list(cp_metadata.segment_lens)
        valid_lens = cp_metadata.segment_valid_lens.to(device=device, dtype=torch.int32)
        seq_lens = [int(v) for v in attn_metadata_ori["actual_seq_k"].detach().cpu().tolist()]

        # Materialize immutable planner fields on the input device once.
        def _materialize_plan_value(value, dtype=None):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                if value.device == device and (dtype is None or value.dtype == dtype):
                    return value
                return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
            return torch.tensor(value, dtype=dtype, device=device) if dtype is not None else torch.tensor(value, device=device)

        planner_window_plan = getattr(cp_metadata, "window_plan", None)
        if not planner_window_plan:
            planner_window_plan = build_prefill_cp_window_plan(
                segment_lens,
                valid_lens.detach().cpu().tolist(),
                self.cp_size,
                int(cp_metadata.cp_rank),
                self.window_size,
                last_segment_indices=cp_metadata.last_segment_indices.detach().cpu().tolist(),
                device=device,
            )

        def _normalize_window_plan(raw_plan):
            normalized = {}
            if not isinstance(raw_plan, dict):
                return normalized
            tensor_dtypes = {
                "segment_indices": torch.long,
                "segment_offsets": torch.long,
                "current_valid_lens": torch.int32,
                "previous_valid_lens": torch.int32,
                "pre_window_lens": torch.int32,
                "augmented_split_sizes": torch.int32,
                "augmented_offsets": torch.long,
                "current_source_indices": torch.long,
                "current_row_indices": torch.long,
                "augmented_current_source_indices": torch.long,
                "augmented_current_dest_indices": torch.long,
                "indexer_source_indices": torch.long,
                "current_dest_indices": torch.long,
                "current_row_offsets": torch.long,
                "pre_window_source_indices": torch.long,
                "pre_window_dest_indices": torch.long,
                "tail_source_indices": torch.long,
                "tail_valid_mask": torch.bool,
                "tail_padding_mask": torch.bool,
                "window_value_source_indices": torch.long,
            }
            for key, value in raw_plan.items():
                if key == "last":
                    continue
                copied = dict(value) if isinstance(value, dict) else {}
                for field, dtype in tensor_dtypes.items():
                    if field in copied:
                        copied[field] = _materialize_plan_value(copied[field], dtype)
                normalized[key] = copied
            last_raw = raw_plan.get("last")
            if isinstance(last_raw, dict):
                last = dict(last_raw)
                for field, dtype in {
                    "source_indices": torch.long,
                    "valid_mask": torch.bool,
                    "last_window_source_indices": torch.long,
                    "last_window_valid_mask": torch.bool,
                    "last_window_padding_mask": torch.bool,
                    "take_prev": torch.int32,
                    "take_cur": torch.int32,
                    "segment_indices": torch.long,
                    "current_valid_lens": torch.int32,
                    "previous_valid_lens": torch.int32,
                }.items():
                    if field in last:
                        last[field] = _materialize_plan_value(last[field], dtype)
                # Keep both names during migration; consumers can select the
                # explicit aliases without guessing which coordinate is used.
                if "last_window_source_indices" not in last and "source_indices" in last:
                    last["last_window_source_indices"] = last["source_indices"]
                if "last_window_valid_mask" not in last and "valid_mask" in last:
                    last["last_window_valid_mask"] = last["valid_mask"]
                if "source_indices" not in last and "last_window_source_indices" in last:
                    last["source_indices"] = last["last_window_source_indices"]
                if "valid_mask" not in last and "last_window_valid_mask" in last:
                    last["valid_mask"] = last["last_window_valid_mask"]
                normalized["last"] = last
            return normalized

        canonical_window_plan = _normalize_window_plan(planner_window_plan)
        segment_offsets_plan = getattr(cp_metadata, "segment_offsets", None)
        if segment_offsets_plan is None:
            segment_offsets_plan = torch.tensor(
                [0] + list(torch.cumsum(
                    torch.tensor(segment_lens, dtype=torch.int32, device=device), 0
                ).detach().cpu().tolist()),
                dtype=torch.long,
                device=device,
            )
        else:
            segment_offsets_plan = _materialize_plan_value(segment_offsets_plan, torch.long)
        last_local_indices = _materialize_plan_value(
            getattr(cp_metadata, "last_token_local_indices", None), torch.long
        )
        if last_local_indices is None:
            last_local_indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        last_valid_rows = _materialize_plan_value(
            getattr(cp_metadata, "last_token_valid_row_indices", None), torch.long
        )
        last_gather_rows = _materialize_plan_value(
            getattr(cp_metadata, "last_token_gather_rows", None), torch.long
        )

        # Preserve the canonical local map when converting metadata to the
        # dictionary consumed by model code.
        local_indices = _materialize_plan_value(
            getattr(cp_metadata, "local_indices", None), torch.long
        )

        cp_input_dict = {
            "cp_rank": cp_metadata.cp_rank,
            "split_list": split_list_hidden,
            "segment_lens": segment_lens,
            "zigzag_idx": cp_metadata.zigzag_idx,
            "zigzag_indices": cp_metadata.zigzag_indices,
            "gather_to_natural": cp_metadata.gather_to_natural,
            "segment_valid_lens": valid_lens,
            "last_segment_indices": cp_metadata.last_segment_indices,
            "last_cp_ranks": cp_metadata.last_cp_ranks,
            "last_flags": cp_metadata.last_flags,
            "last_token_local_indices": last_local_indices,
            "last_token_valid_row_indices": last_valid_rows,
            "last_token_gather_rows": last_gather_rows,
            "last_token_slot_capacity": int(
                getattr(cp_metadata, "last_token_slot_capacity", 0) or 0
            ),
            "local_indices": local_indices,
            "decode_token_indices": cp_metadata.persistent_valid_indices,
            "persistent_request_indices": cp_metadata.persistent_request_indices,
            "state_write_plans": cp_metadata.state_write_plans or {},
            "segment_offsets": segment_offsets_plan,
            "window_plan": canonical_window_plan,
            "indexer_plan": cp_metadata.indexer_plan or {
                flag: {
                    "current_row_indices": canonical_window_plan.get(flag, {}).get(
                        "current_row_indices"
                    ),
                    "indexer_source_indices": canonical_window_plan.get(flag, {}).get(
                        "indexer_source_indices"
                    ),
                    "current_row_offsets": canonical_window_plan.get(flag, {}).get(
                        "current_row_offsets"
                    ),
                    "local_half_tokens": canonical_window_plan.get(flag, {}).get(
                        "local_half_tokens"
                    ),
                }
                for flag in ("prev", "next")
                if canonical_window_plan.get(flag)
            },
        }

        # Keep legacy keys for existing consumers; retain multi-request shape.
        cp_input_dict.update({
            "split_kv_len": valid_lens,
            "last_rank": cp_metadata.last_segment_indices,
            "last_rank_flag": cp_metadata.last_flags,
            "last_rank_zz": cp_metadata.last_cp_ranks,
        })
        attn_metadata["cp_metadata"] = cp_input_dict
        # These host shape contracts are shared by flag-local and top-level
        # consumers of the augmented stream.
        cp_input_dict["local_prev_token_num"] = int(
            canonical_window_plan.get("prev", {}).get(
                "local_half_tokens", sum(segment_lens)
            )
        )
        cp_input_dict["local_next_token_num"] = int(
            canonical_window_plan.get("next", {}).get(
                "local_half_tokens", sum(segment_lens)
            )
        )
        cp_input_dict["prev_augmented_total_rows"] = int(
            canonical_window_plan.get("prev", {}).get(
                "augmented_total_rows", sum(segment_lens)
            )
        )
        cp_input_dict["next_augmented_total_rows"] = int(
            canonical_window_plan.get("next", {}).get(
                "augmented_total_rows", sum(segment_lens)
            )
        )
        attn_metadata["prev"], attn_metadata["next"] = {}, {}
        cp_input_dict["cp_tmp_cache"] = {}
        cp_input_dict["tmp_block_table"] = {}
        if not self.is_mtp and not self.is_online:
            cp_input_dict["cp_tmp_cache"] = self.build_cp_prefill_tmp_cache(
                input_ids, attn_metadata, cp_metadata
            )

        # Complete the owner-local state plan once per prefill; layers reuse
        # it. Warmup has no persistent rows and keeps the legacy no-write path.
        if not attn_metadata.get("is_warm_up", False):
            owner_indices = cp_metadata.persistent_request_indices
            if owner_indices is None:
                owner_indices = torch.empty(0, dtype=torch.long, device=device)
            else:
                owner_indices = owner_indices.to(device=device, dtype=torch.long).view(-1)
            state_plans = cp_input_dict.setdefault("state_write_plans", {})
            tmp_tables = cp_input_dict.get("tmp_block_table", {})
            schedule_tables = attn_metadata_ori.get("block_table", {}) or {}
            tmp_cache_by_ratio = cp_input_dict.get("cp_tmp_cache", {}) or {}
            for ratio in CP_COMPRESS_RATIOS:
                state_key = f"c{ratio}a_cmp_state"
                tmp_state_table = tmp_tables.get(state_key)
                schedule_state_table = schedule_tables.get(state_key)
                if tmp_state_table is None or schedule_state_table is None:
                    continue
                ratio_cache = tmp_cache_by_ratio.get(str(ratio), {})
                state_plan = self._build_cp_state_write_plan(
                    ratio,
                    tmp_state_table,
                    schedule_state_table,
                    owner_indices,
                    cp_metadata.last_cp_ranks,
                    batch_size,
                    cp_tmp_cache=ratio_cache,
                )
                if state_plan:
                    # Store fixed-shape fields once; executor remains
                    # independent of model-specific state-table layouts.
                    state_plans.setdefault(str(ratio), {}).update(state_plan)

        rank = int(cp_metadata.cp_rank)
        for flag in ("prev", "next"):
            segment_indices = [
                rank if flag == "prev" else segment_num - rank - 1
            ] * batch_size
            current_positions = [split_position_ids[b * segment_num + segment_indices[b]]
                                 for b in range(batch_size)]
            flag_plan = canonical_window_plan.get(flag, {})
            planned_segment_indices = flag_plan.get("segment_indices")
            if isinstance(planned_segment_indices, torch.Tensor):
                segment_indices = planned_segment_indices.detach().cpu().tolist()
            current_valid = flag_plan.get("current_valid_lens")
            if current_valid is None:
                current_valid = torch.tensor(
                    [int(valid_lens[b, segment_indices[b]].item()) for b in range(batch_size)],
                    dtype=torch.int32,
                    device=device,
                )
            else:
                current_valid = _materialize_plan_value(current_valid, torch.int32)
            logical_kv = torch.tensor(
                [min(seq_lens[b], (segment_indices[b] + 1) * segment_lens[b])
                 for b in range(batch_size)], dtype=torch.int32, device=device
            )
            position_with_pre = []
            pre_lengths = []
            for b, segment_idx in enumerate(segment_indices):
                if segment_idx == 0:
                    pre_lengths.append(0)
                    position_with_pre.append(current_positions[b])
                    continue
                prev_valid = int(valid_lens[b, segment_idx - 1].item())
                take = min(self.window_size, prev_valid)
                prev = split_position_ids[b * segment_num + segment_idx - 1]
                # Padding follows the valid prefix; take tails from that
                # prefix so padding cannot become window context.
                tail = prev[max(prev_valid - take, 0):prev_valid] if take else prev[:0]
                pad = torch.full(
                    (self.window_size - take,), self.position_ids_pad_value,
                    dtype=prev.dtype, device=device
                )
                pre_lengths.append(self.window_size)
                position_with_pre.append(torch.cat((pad, tail, current_positions[b])))

            last_positions = []
            for b in range(batch_size):
                last_segment = int(cp_metadata.last_segment_indices[b].item() % segment_num)
                last_valid = int(valid_lens[b, last_segment].item())
                current = split_position_ids[b * segment_num + last_segment]
                take_cur = min(self.window_size, last_valid)
                valid_current = current[:last_valid]
                cur_tail = valid_current[max(valid_current.numel() - take_cur, 0):] if take_cur else current[:0]
                need_prev = self.window_size - take_cur
                if need_prev and last_segment > 0:
                    prev = split_position_ids[b * segment_num + last_segment - 1]
                    prev_valid = int(valid_lens[b, last_segment - 1].item())
                    take_prev = min(need_prev, prev_valid)
                    prev_tail = prev[max(prev_valid - take_prev, 0):] if take_prev else prev[:0]
                else:
                    prev_tail = current[:0]
                    take_prev = 0
                pad = torch.full(
                    (self.window_size - take_prev - take_cur,), self.position_ids_pad_value,
                    dtype=current.dtype, device=device
                )
                last_positions.append(torch.cat((pad, prev_tail, cur_tail)))

            cur_flat = torch.cat(current_positions)
            pre_flat = torch.cat(position_with_pre)
            last_flat = torch.cat(last_positions)
            cu_q, seq_used_q = build_prefill_cp_query_metadata(
                segment_lens, current_valid, device=device
            )
            physical_kv = torch.tensor(
                [pre_lengths[b] + segment_lens[b] for b in range(batch_size)],
                dtype=torch.int32, device=device
            )
            block_table = attn_metadata_ori["block_table"]
            slot_mapping_ori_kv = self.get_slot_mapping_from_block_table(
                physical_kv,
                pre_flat,
                block_table["full_kv"],
            )
            valid_rows = []
            for b in range(batch_size):
                pre_len = pre_lengths[b]
                prev_valid = pre_len and min(self.window_size, int(valid_lens[b, segment_indices[b] - 1].item()))
                valid_rows.extend([False] * (pre_len - prev_valid))
                valid_rows.extend([True] * prev_valid)
                valid_rows.extend([True] * int(current_valid[b].item()))
                valid_rows.extend([False] * (segment_lens[b] - int(current_valid[b].item())))
            slot_mapping_ori_kv = torch.where(
                torch.tensor(valid_rows, dtype=torch.bool, device=device),
                slot_mapping_ori_kv,
                torch.full_like(slot_mapping_ori_kv, -1),
            )
            is_end_flags = [
                int(cp_metadata.last_segment_indices[b].item())
                == b * segment_num + segment_indices[b]
                for b in range(batch_size)
            ]
            attn_metadata[flag].update({
                "is_start": flag == "prev" and rank == 0,
                "is_start_per_request": torch.tensor(
                    [segment_idx == 0 for segment_idx in segment_indices],
                    dtype=torch.bool, device=device,
                ),
                "is_end": torch.tensor(
                    is_end_flags, dtype=torch.bool, device=device,
                ),
                # Host-side plan consumed by Compressor.forward without
                # per-layer reductions or device-to-host conversion.
                "has_end": any(is_end_flags),
                "cur_kv_len": current_valid,
                "last_kv_len": torch.tensor(
                    [int(valid_lens[b, int(cp_metadata.last_segment_indices[b].item() % segment_num)].item())
                     for b in range(batch_size)], dtype=torch.int32, device=device,
                ),
                "block_table": block_table,
                "full_kv_cache": attn_metadata_ori["full_kv_cache"],
                "cp_tmp_cache": cp_input_dict["cp_tmp_cache"],
                "tmp_block_table": cp_input_dict["tmp_block_table"],
                "kernel_metadata": {},
                "position_ids_with_pre_win": pre_flat,
                "position_ids_last_win": last_flat,
                "position_ids_cur": cur_flat,
                "slot_mapping_ori_kv": slot_mapping_ori_kv,
                # Cache writers consume this fixed row map directly.
                "slot_mapping_ori_kv_valid_row_indices": torch.arange(
                    slot_mapping_ori_kv.numel(), dtype=torch.long, device=device
                )[slot_mapping_ori_kv.view(-1) >= 0],
                "actual_seq_k": logical_kv,
                "actual_seq_q": cu_q[1:],
                "cu_seq_lens_q": cu_q,
                "seq_used_q": seq_used_q,
                "window_pre_lengths": torch.tensor(pre_lengths, dtype=torch.int32, device=device),
                "augmented_offsets": torch.cat((
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.cumsum(
                        torch.tensor(physical_kv, dtype=torch.int32, device=device),
                        0,
                        dtype=torch.int32,
                    ),
                ))[:-1],
            })
            # Extend the canonical plan; retain aliases for existing callers.
            flag_plan.update({
                "segment_indices_host": segment_indices,
                "current_valid_lens": current_valid,
                "pre_window_lens": _materialize_plan_value(
                    flag_plan.get("pre_window_lens", pre_lengths), torch.int32
                ),
                "last_window_positions": last_flat,
                "augmented_total_rows": int(
                    flag_plan.get("augmented_total_rows", sum(pre_lengths) + sum(segment_lens))
                ),
            })
            last_plan = canonical_window_plan.get("last", {})
            if isinstance(last_plan, dict):
                for alias, source_key in (
                    ("last_window_source_indices", "last_window_source_indices"),
                    ("last_window_valid_mask", "last_window_valid_mask"),
                ):
                    if source_key in last_plan:
                        flag_plan[alias] = last_plan[source_key]

        # Materialize request-layout mappings in packed [prev, next] order once.
        prev_slot_mapping = attn_metadata["prev"]["slot_mapping_ori_kv"]
        next_slot_mapping = attn_metadata["next"]["slot_mapping_ori_kv"]
        cp_input_dict["full_kv_slot_mapping"] = torch.cat(
            (prev_slot_mapping, next_slot_mapping), dim=-1
        )
        cp_input_dict["full_kv_valid_row_indices"] = torch.cat((
            attn_metadata["prev"]["slot_mapping_ori_kv_valid_row_indices"],
            attn_metadata["next"]["slot_mapping_ori_kv_valid_row_indices"]
            + prev_slot_mapping.numel(),
        ), dim=0)

        # MTP reuses the CP window layout; prepare its owner-local B*W mapping.
        self._set_cp_persistent_window_slots(attn_metadata_ori, cp_metadata, seq_lens, device)
        persistent_win_slots = cp_input_dict.get("persistent_win_slot_mapping")
        if persistent_win_slots is not None:
            value_indices = torch.arange(
                batch_size * self.window_size, dtype=torch.long, device=device
            )
            cp_input_dict["window_plan"]["last_window_update_plan"] = {
                "slot_indices": persistent_win_slots,
                "value_indices": value_indices,
                "request_index": torch.arange(
                    batch_size, dtype=torch.long, device=device
                ).repeat_interleave(self.window_size),
            }
            cp_input_dict["persistent_win_valid_row_indices"] = torch.arange(
                persistent_win_slots.numel(), dtype=torch.long, device=device
            )[persistent_win_slots.view(-1) >= 0]
        if is_mtp:
            return attn_metadata

        # Build per-request compressor metadata, then pad once to fixed
        # rank/flag blocks for equal-shape collectives.
        slot_mapping_cmp_dict, slot_mapping_cmp_for_decode_dict = {}, {}
        persistent_decode_mapping_dict, persistent_decode_mask_dict = {}, {}
        cp_input_dict["compressor_plans"] = {}
        for flag in ("prev", "next"):
            attn_metadata[flag]["cmp_out_pad"] = {}
            attn_metadata[flag]["cmp_in_offset"] = {}
        for ratio in CP_COMPRESS_RATIOS:
            expected_by_rank_flag = {}
            request_plans = {}
            overlap_len = ratio if ratio == 4 else 0
            for candidate_rank in range(self.cp_size):
                for flag in ("prev", "next"):
                    natural_segments = [
                        candidate_rank if flag == "prev" else segment_num - candidate_rank - 1
                    ] * batch_size
                    expected = 0
                    row_starts = []
                    borrowed_out_lens = []
                    valid_out_lens = []
                    compressed_lens = []
                    input_lens = []
                    start_positions = []
                    for b, segment_idx in enumerate(natural_segments):
                        current_valid_len = int(valid_lens[b, segment_idx].item())
                        start = segment_idx * segment_lens[b]
                        # Empty tails contribute no borrowed rows; keep this
                        # plan aligned with get_cmp_param() and collective rows.
                        comp_len = 0
                        if current_valid_len > 0 and segment_idx > 0:
                            comp_len = min(
                                start % ratio + overlap_len,
                                start,
                                int(valid_lens[b, segment_idx - 1].item()),
                            )
                        input_len = current_valid_len + comp_len
                        start_pos = start - comp_len
                        compressed_len = max(
                            0,
                            (start_pos + input_len) // ratio - start_pos // ratio,
                        )
                        borrowed_out_len = max(
                            0,
                            (start_pos + comp_len) // ratio - start_pos // ratio,
                        )
                        valid_out_len = max(0, compressed_len - borrowed_out_len)
                        row_starts.append(expected)
                        borrowed_out_lens.append(borrowed_out_len)
                        valid_out_lens.append(valid_out_len)
                        compressed_lens.append(compressed_len)
                        input_lens.append(input_len)
                        start_positions.append(start_pos)
                        expected += compressed_len
                    expected_by_rank_flag[(candidate_rank, flag)] = expected
                    request_plans[(candidate_rank, flag)] = {
                        "segment_indices": natural_segments,
                        "row_start": row_starts,
                        "borrowed_out_len": borrowed_out_lens,
                        "valid_out_len": valid_out_lens,
                        "compressed_len": compressed_lens,
                        "input_len": input_lens,
                        "start_pos": start_positions,
                        "expected_len": expected,
                    }
            block_len = max(expected_by_rank_flag.values(), default=0)
            cp_input_dict["compressor_plans"][str(ratio)] = {
                "block_len": block_len,
                "expected_len": {
                    f"{candidate_rank}:{flag}": expected
                    for (candidate_rank, flag), expected in expected_by_rank_flag.items()
                },
                "flags": {
                    f"{candidate_rank}:{flag}": plan
                    for (candidate_rank, flag), plan in request_plans.items()
                },
            }

            local_slot_parts, local_decode_parts = [], []
            for flag in ("prev", "next"):
                segment_idx = rank if flag == "prev" else segment_num - rank - 1
                res_dict = self.get_cmp_param(
                    input_ids, segment_idx, attn_metadata_ori, split_list_hidden, valid_lens,
                )
                attn_metadata[flag].update(res_dict)
                expected = expected_by_rank_flag[(rank, flag)]
                pad_len = block_len - expected
                request_plan = request_plans[(rank, flag)]
                attn_metadata[flag].setdefault(
                    "has_compressor_input", {}
                )[str(ratio)] = any(
                    input_len > 0 for input_len in request_plan["input_len"]
                )
                attn_metadata[flag]["cmp_out_pad"][str(ratio)] = (
                    torch.zeros((pad_len, self.config.index_head_dim), dtype=torch.bfloat16, device=device),
                    torch.zeros((pad_len, self.config.head_dim), dtype=torch.bfloat16, device=device),
                )
                attn_metadata[flag]["cmp_in_offset"][str(ratio)] = res_dict["comp_lens"][str(ratio)]
                raw_slots = res_dict["slot_mapping_cmp"][str(ratio)]
                raw_decode_slots = res_dict["slot_mapping_cmp_for_decode"][str(ratio)]
                aggregate_slots = torch.full(
                    (block_len,), -1, dtype=torch.int32, device=device
                )
                aggregate_decode_slots = torch.full_like(aggregate_slots, -1)
                for request_idx, (row_start, compressed_len, borrowed_len) in enumerate(zip(
                    request_plan["row_start"],
                    request_plan["compressed_len"],
                    request_plan["borrowed_out_len"],
                )):
                    if compressed_len <= 0:
                        continue
                    src_start = row_start
                    dst_start = pad_len + row_start
                    aggregate_slots[dst_start:dst_start + compressed_len] = raw_slots[
                        src_start:src_start + compressed_len
                    ]
                    aggregate_decode_slots[dst_start:dst_start + compressed_len] = raw_decode_slots[
                        src_start:src_start + compressed_len
                    ]
                    if borrowed_len > 0:
                        aggregate_slots[dst_start:dst_start + borrowed_len] = -1
                        aggregate_decode_slots[dst_start:dst_start + borrowed_len] = -1
                local_slot_parts.append(aggregate_slots)
                if not self.is_online:
                    local_decode_parts.append(aggregate_decode_slots)

            local_slots = torch.cat(local_slot_parts)
            gathered_slots = local_slots.new_empty(local_slots.numel() * self.cp_size)
            dist.all_gather_into_tensor(gathered_slots, local_slots, group=self.comm_manager.get_group("cp_group"))
            gathered_slots = gathered_slots.view(self.cp_size, 2, block_len)
            # Restore valid collective rows from rank-major [prev_B, next_B]
            # to natural request/segment order.
            gathered_slots_flat = gathered_slots.view(-1)
            if not self.is_online:
                local_decode = torch.cat(local_decode_parts)
                gathered_decode = local_decode.new_empty(local_decode.numel() * self.cp_size)
                dist.all_gather_into_tensor(
                    gathered_decode, local_decode, group=self.comm_manager.get_group("cp_group")
                )
                gathered_decode = gathered_decode.view(self.cp_size, 2, block_len)
                gathered_decode_flat = gathered_decode.view(-1)

            # Owners write all CP segments for their requests; keep complete
            # gathered mappings and apply the writer mask in compressor_epilog.
            owner_requests = set(
                cp_metadata.persistent_request_indices.detach().cpu().tolist()
                if cp_metadata.persistent_request_indices is not None
                else range(batch_size)
            )
            persistent_write_mask = torch.zeros(
                self.cp_size * 2 * block_len, dtype=torch.bool, device=device
            )
            for candidate_rank in range(self.cp_size):
                for flag_idx, flag in enumerate(("prev", "next")):
                    flag_plan = request_plans[(candidate_rank, flag)]
                    expected = expected_by_rank_flag[(candidate_rank, flag)]
                    base = (candidate_rank * 2 + flag_idx) * block_len
                    pad_len = block_len - expected
                    for request_idx, compressed_len in enumerate(flag_plan["compressed_len"]):
                        if request_idx not in owner_requests or compressed_len <= 0:
                            continue
                        start = base + pad_len + flag_plan["row_start"][request_idx]
                        persistent_write_mask[start:start + compressed_len] = True
            cp_input_dict.setdefault("persistent_cmp_write_mask", {})[str(ratio)] = persistent_write_mask

            # Keep aggregate blocks rank-major only for the collective; expose
            # explicit natural row indices to consumers.
            natural_rows = []
            natural_row_starts = [0]
            for request_idx in range(batch_size):
                for segment_idx in range(segment_num):
                    candidate_rank = segment_idx if segment_idx < self.cp_size else segment_num - segment_idx - 1
                    flag_idx = 0 if segment_idx < self.cp_size else 1
                    flag_plan = request_plans[(candidate_rank, "prev" if flag_idx == 0 else "next")]
                    expected = expected_by_rank_flag[(candidate_rank, "prev" if flag_idx == 0 else "next")]
                    source_start = (
                        (candidate_rank * 2 + flag_idx) * block_len
                        + (block_len - expected)
                        + flag_plan["row_start"][request_idx]
                    )
                    row_count = flag_plan["compressed_len"][request_idx]
                    natural_rows.extend(range(source_start, source_start + row_count))
                    natural_row_starts.append(natural_row_starts[-1] + row_count)
            natural_rows_tensor = torch.tensor(natural_rows, dtype=torch.long, device=device)
            compressor_plan = cp_input_dict["compressor_plans"][str(ratio)]
            natural_slot_mapping = (
                gathered_slots_flat.index_select(0, natural_rows_tensor)
                if natural_rows_tensor.numel() else gathered_slots_flat[:0]
            )
            slot_mapping_cmp_dict[str(ratio)] = natural_slot_mapping
            # Materialize valid row coordinates once so layers can index rows
            # without reducing an NPU mask to Python.
            natural_row_indices = torch.arange(
                natural_slot_mapping.numel(), dtype=torch.long, device=device
            )
            temp_valid_row_indices = natural_row_indices[
                natural_slot_mapping >= 0
            ]
            if not self.is_online:
                natural_decode_mapping = (
                    gathered_decode_flat.index_select(0, natural_rows_tensor)
                    if natural_rows_tensor.numel() else gathered_decode_flat[:0]
                )
                slot_mapping_cmp_for_decode_dict[str(ratio)] = natural_decode_mapping
                write_mask = cp_input_dict["persistent_cmp_write_mask"][str(ratio)]
                cp_input_dict["persistent_cmp_write_mask"][str(ratio)] = (
                    write_mask.index_select(0, natural_rows_tensor)
                    if natural_rows_tensor.numel() else write_mask[:0]
                )
                natural_write_mask = cp_input_dict["persistent_cmp_write_mask"][str(ratio)]
                # Apply owner, borrowed-row, and padding masks after restore.
                # Mapping matches source_kv; -1 excludes writes, while slot 0 is valid.
                persistent_decode_mask = natural_write_mask & (natural_decode_mapping >= 0)
                persistent_decode_mapping = torch.where(
                    persistent_decode_mask,
                    natural_decode_mapping,
                    torch.full_like(natural_decode_mapping, -1),
                )
                persistent_valid_row_indices = natural_row_indices[
                    persistent_decode_mask
                ]
                persistent_decode_mapping_dict[str(ratio)] = persistent_decode_mapping
                persistent_decode_mask_dict[str(ratio)] = persistent_decode_mask
            compressor_plan.update({
                "natural_row_starts": natural_row_starts,
                "aggregate_to_natural": natural_rows_tensor,
                "natural_slot_mapping": natural_slot_mapping,
                "temp_valid_row_indices": temp_valid_row_indices,
            })
            if not self.is_online:
                compressor_plan.update({
                    "persistent_decode_slot_mapping": persistent_decode_mapping_dict[str(ratio)],
                    "persistent_decode_valid_mask": persistent_decode_mask_dict[str(ratio)],
                    "persistent_slot_mapping": persistent_decode_mapping_dict[str(ratio)],
                    "persistent_slot_valid_mask": persistent_decode_mask_dict[str(ratio)],
                    "persistent_valid_row_indices": persistent_valid_row_indices,
                })
            state_plan = cp_input_dict.get("state_write_plans", {}).get(str(ratio))
            if isinstance(state_plan, dict) and state_plan:
                compressor_plan["state_write_plan"] = state_plan
        attn_metadata["cp_metadata"].update({"slot_mapping_cmp": slot_mapping_cmp_dict})
        if not self.is_online:
            attn_metadata["cp_metadata"].update({
                "slot_mapping_cmp_for_decode": slot_mapping_cmp_for_decode_dict,
                "persistent_decode_slot_mapping": persistent_decode_mapping_dict,
                "persistent_decode_valid_mask": persistent_decode_mask_dict,
            })

        return attn_metadata

    def get_cmp_param(
        self,
        input_ids,
        rank,
        attn_metadata,
        split_list_hidden,
        split_kv_len,
    ):
        device = input_ids.device
        batch_size = len(split_list_hidden) // (2 * self.cp_size)
        segment_num = 2 * self.cp_size
        segment_lens = [int(split_list_hidden[b * segment_num + rank]) for b in range(batch_size)]
        cur_kv_len = split_kv_len[:, rank].to(device=device, dtype=torch.int32)
        cu_seq_lens_dict, seq_used_q_dict, start_pos_dict = {}, {}, {}
        position_ids_cmp_for_rope_dict = {}
        slot_mapping_cmp_dict, slot_mapping_used_cmp_dict = {}, {}
        comp_len_dict, cmp_input_indices_dict = {}, {}
        cmp_input_identity_dict, cmp_input_contiguous_dict = {}, {}
        cmp_valid_len_dict = {}

        # Use request-major augmented offsets to skip segment and window padding.
        flag = "prev" if rank < self.cp_size else "next"
        for ratio in CP_COMPRESS_RATIOS:
            overlap_len = ratio if ratio == 4 else 0
            starts = torch.tensor(
                [rank * segment_lens[b] for b in range(batch_size)],
                dtype=torch.int32, device=device,
            )
            comp_len = self._cp_comp_len(
                starts,
                rank,
                cur_kv_len,
                split_kv_len,
                ratio,
            )
            seq_used_q = torch.where(cur_kv_len > 0, cur_kv_len + comp_len, torch.zeros_like(cur_kv_len))
            input_lens = seq_used_q
            cu_seq_lens = torch.cat((
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(input_lens, dim=0, dtype=torch.int32),
            ))
            start_pos = torch.where(
                cur_kv_len > 0,
                starts - comp_len,
                torch.zeros_like(starts),
            )
            compressed_len, position_ids_cmp = self.generate_compressed_position_ids(
                {"start_pos": start_pos, "seq_used_q": seq_used_q, "cu_seq_lens_q": cu_seq_lens}, ratio
            )
            block_table_tmp = attn_metadata["cp_metadata"]["tmp_block_table"].get(
                f"c{ratio}a_cmp_kv"
            )
            block_table_used = attn_metadata.get("block_table", {}).get(f"c{ratio}a_cmp_kv")
            if block_table_tmp is None:
                # MTP/online callers may only provide the scheduled table.
                block_table_tmp = block_table_used

            # Owners allocate persistent blocks, but every rank needs the
            # complete table to map its gathered segment rows.
            if block_table_used is not None and not self.is_online:
                block_table_used = block_table_used.clone()
                dist.all_reduce(
                    block_table_used,
                    group=self.comm_manager.get_group("cp_group"),
                )
            slot_mapping_cmp = self.get_slot_mapping_from_block_table(
                compressed_len, position_ids_cmp, block_table_tmp
            )
            if block_table_used is None:
                slot_mapping_used = slot_mapping_cmp.clone()
            else:
                slot_mapping_used = self.get_slot_mapping_from_block_table(
                    compressed_len, position_ids_cmp, block_table_used
                )
                request_has_blocks = block_table_used.to(torch.int32).sum(dim=-1) > 0
                request_valid_rows = torch.repeat_interleave(request_has_blocks, compressed_len)
                physical_pad_len = position_ids_cmp.numel() - request_valid_rows.numel()
                if physical_pad_len > 0:
                    request_valid_rows = torch.cat((
                        request_valid_rows,
                        torch.zeros(physical_pad_len, dtype=torch.bool, device=device),
                    ))
                slot_mapping_used = torch.where(
                    request_valid_rows,
                    slot_mapping_used,
                    torch.full_like(slot_mapping_used, -1),
                )
            cu_seq_lens_dict[str(ratio)] = cu_seq_lens
            seq_used_q_dict[str(ratio)] = seq_used_q
            start_pos_dict[str(ratio)] = start_pos
            comp_len_dict[str(ratio)] = comp_len
            cmp_valid_len_dict[str(ratio)] = int(compressed_len.sum().item())
            position_ids_cmp_for_rope_dict[str(ratio)] = position_ids_cmp * ratio
            slot_mapping_cmp_dict[str(ratio)] = slot_mapping_cmp
            slot_mapping_used_cmp_dict[str(ratio)] = slot_mapping_used

            # Select borrowed rows from the preceding tail and current rows
            # from each segment's valid prefix.
            input_indices = []
            all_current_valid = True
            all_no_overlap = True
            for b in range(batch_size):
                aug_offset = sum((self.window_size if rank > 0 else 0) + segment_lens[j]
                                 for j in range(b))
                cur_valid = int(cur_kv_len[b].item())
                borrowed = int(comp_len[b].item())
                all_current_valid = (
                    all_current_valid and cur_valid == segment_lens[b]
                )
                all_no_overlap = all_no_overlap and borrowed == 0
                if borrowed:
                    input_indices.extend(range(aug_offset + self.window_size - borrowed, aug_offset + self.window_size))
                input_indices.extend(range(aug_offset + (self.window_size if rank > 0 else 0),
                                           aug_offset + (self.window_size if rank > 0 else 0) + cur_valid))
            cmp_input_indices_dict[str(ratio)] = torch.tensor(input_indices, dtype=torch.long, device=device)

            # Keep gather bypass decisions in metadata. The contiguous fast
            # path currently applies to ratio-128 segments without overlap.
            augmented_lens = [
                (self.window_size if rank > 0 else 0) + segment_lens[b]
                for b in range(batch_size)
            ]
            total_augmented_len = sum(augmented_lens)
            cmp_input_identity_dict[str(ratio)] = (
                input_indices == list(range(total_augmented_len))
            )
            cmp_input_contiguous_dict[str(ratio)] = (
                {
                    "prefix_len": self.window_size,
                    "segment_len": segment_lens[0],
                    "batch_size": batch_size,
                }
                if (
                    ratio == 128
                    and rank > 0
                    and batch_size > 0
                    and len(set(segment_lens)) == 1
                    and all_current_valid
                    and all_no_overlap
                ) else None
            )

        return {
            "cu_seq_lens": cu_seq_lens_dict,
            "cmp_seq_used_q": seq_used_q_dict,
            "start_pos": start_pos_dict,
            "position_ids_cmp_for_rope": position_ids_cmp_for_rope_dict,
            "slot_mapping_cmp": slot_mapping_cmp_dict,
            "slot_mapping_cmp_for_decode": slot_mapping_used_cmp_dict,
            "comp_lens": comp_len_dict,
            "cmp_valid_lens": cmp_valid_len_dict,
            "cmp_input_indices": cmp_input_indices_dict,
            "cmp_input_indices_is_identity": cmp_input_identity_dict,
            "cmp_input_contiguous": cmp_input_contiguous_dict,
        }

    def get_zigzag_idx(self, origin_idx, cp_segment_num):
        midpoint = cp_segment_num // 2 - 1
        if origin_idx <= midpoint:
            return origin_idx, "prev"
        else:
            return midpoint + 1 - (origin_idx - midpoint), "next"
