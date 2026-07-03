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
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
from executor.utils import align_up


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
            "int8": torch.int8, "float8": torch.float8_e4m3fn, "unquant": torch.bfloat16, "hifloat8": torch.uint8
        }

    def update_kv_quant_settings(self):
        # set li_cache_quant_mode to quant_config
        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")

        # if quant to hif8/fp8/mxfp8, set kv cache and li cache accordingly; if quant to int8, set li cache to int8
        if self.platform_version == PlatformVersion.ASCEND_950 and "hifloat" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "hifloat8"
            self.config.quant_config.li_cache_quant_mode = "hifloat8"
        elif self.platform_version == PlatformVersion.ASCEND_950 and "float" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "float8"
            self.config.quant_config.li_cache_quant_mode = "float8"
        else:
            self.config.quant_config.li_cache_quant_mode = "int8"

    def update_gmm_quant_mode(self):
        if self.platform_version == PlatformVersion.ASCEND_950 and \
            "w4" in self.config.quant_config.gmm_quant_mode and "mx" not in self.config.quant_config.gmm_quant_mode:
            self.config.quant_config.gmm_quant_mode = \
                self.config.quant_config.gmm_quant_mode.replace("float", "mxfloat")

    def init_cache_dim(self):
        cache_dim = self.config.head_dim
        if "float" in self.kv_cache_quant_mode:
            rope_dim = self.config.qk_rope_head_dim
            nope_dim = self.config.head_dim - rope_dim
            cache_dim = align_up(nope_dim + 2 * rope_dim + nope_dim // 64, 128)
        self.cache_dim = cache_dim

    def get_cmp_kv_dtype(self):
        use_fused_kernel_compressor = (
            self.infer_config.model_config.custom_params.get("kernel_config", {}).get("compressor", "native")
            == "ascendc"
        )
        cmp_kv_dtype = torch.float8_e4m3fn if (
            self.platform_version == PlatformVersion.ASCEND_950 and use_fused_kernel_compressor
        ) else torch.bfloat16
        if "float" in self.kv_cache_quant_mode:
            cmp_kv_dtype = torch.float8_e4m3fn
        return cmp_kv_dtype

    def get_kv_cache_dtype(self):
        if self.kv_cache_quant_mode == "hifloat8":
            return torch.float8_e4m3fn
        return self.cache_dtype_map[self.kv_cache_quant_mode]

    def create_cache(self, block_num, dim, dtype, device):
        return torch.zeros((block_num, self.block_size, 1, dim), dtype=dtype, device=device)

    def create_state_cache(self, state_block_num, compress_ratio, cache_dim, device):
        assert compress_ratio in (4, 128)
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
        block_pos_ids = torch.arange(block_table_len, dtype=torch.int32, device="npu").repeat(batch_size_per_rank, 1)
        block_table_offset = torch.arange(
            batch_size_per_rank * block_num_per_batch,
            dtype=torch.int32,
            device="npu",
        ).view(batch_size_per_rank, -1) + self.pre_pad_num_block
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table_offset = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]

        block_table = torch.zeros((batch_size_per_rank, block_table_len), dtype=torch.int32, device="npu")
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

    def verify_schedule_statecache(self, tmp_state_block_table, schedule_block_table, ratio):
        state_key = f"c{ratio}a_cmp_state"
        if state_key not in schedule_block_table:
            raise KeyError(f"Missing scheduled state block table: {state_key}.")

        def count_allocated_blocks(block_table):
            block_ids = block_table[block_table != 0]
            return torch.unique(block_ids).numel()

        tmp_block_num = count_allocated_blocks(tmp_state_block_table)
        schedule_block_num = count_allocated_blocks(schedule_block_table[state_key])
        if ratio == 4:
            is_valid = schedule_block_num == tmp_block_num
            expect = "equal to"
        elif ratio == 128:
            is_valid = schedule_block_num >= tmp_block_num
            expect = "greater than"
        else:
            raise ValueError(f"Unsupported state cache ratio: {ratio}.")

        if not bool(is_valid):
            raise ValueError(
                f"Invalid {state_key} block allocation for CP prefill decode cache update: "
                f"scheduled block num {int(schedule_block_num)} must be {expect} "
                f"tmp block num {int(tmp_block_num)}."
            )

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
        if batch_size != 1:
            raise ValueError(
                "DeepSeek-V4 CP temporary cache currently follows the CP metadata "
                f"minibatch=1 constraint, got batch_size={batch_size}."
            )
        padded_token_num = input_ids.shape[-1]
        device = input_ids.device
        cmp_kv_dtype = self.get_cmp_kv_dtype()
        li_cache_dtype = self.cache_dtype_map[self.li_cache_quant_mode]

        tmp_cache = {}
        tmp_block_table = {}
        for ratio in (4, 128):
            cmp_cache_len = math.ceil(padded_token_num / ratio)
            cmp_block_num_per_batch = math.ceil(cmp_cache_len / self.block_size)
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
                ratio_cache.update({
                    "li_cmp_kv": self.create_cache(
                        cmp_block_num, self.config.index_head_dim, li_cache_dtype, device),
                })
                if self.li_cache_quant_mode in ["int8", "float8", "hifloat8"]:
                    scale_dtype = torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
                    ratio_cache["li_key_dequant_scale"] = self.create_cache(
                        cmp_block_num, 1, scale_dtype, device)

            tmp_cache[str(ratio)] = ratio_cache

            tmp_block_table[f"c{ratio}a_cmp_kv"] = self.build_cp_tmp_block_table(
                batch_size, cmp_block_num_per_batch, device)
            tmp_block_table[f"c{ratio}a_cmp_state"] = self.build_cp_tmp_state_block_table(
                padded_token_num,
                ratio,
                state_block_num_per_batch,
                batch_size,
                attn_metadata
            )
            attn_metadata["cp_metadata"].update({
                "tmp_block_table": tmp_block_table
            })
            decode_token_indices = attn_metadata["cp_metadata"].get(
                "decode_token_indices",
                getattr(cp_metadata, "persistent_valid_indices", None),
            )
            has_decode_requests = decode_token_indices is not None and decode_token_indices.numel() > 0
            if has_decode_requests and not attn_metadata["is_warm_up"]:
                self.verify_schedule_statecache(
                    tmp_block_table[f"c{ratio}a_cmp_state"],
                    attn_metadata["block_table"],
                    ratio,
                )

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
        flat_range = torch.arange(compressed_len.sum(), dtype=torch.int32, device="npu")
        compressed_ids = flat_range - expanded_offsets + expanded_starts
        max_len = min(attn_metadata["cu_seq_lens_q"][-1], attn_metadata["cu_seq_lens_q"][-1] // ratio + bsz)
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
        idx = torch.arange(max_len, dtype=torch.int32, device="npu").expand(bsz, max_len)
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
        return self.pad_tensor_tnd(slot_mapping.view(-1), position_ids.shape[0], self.slot_mapping_pad_value)

    def get_cmp_metadata(self, attn_metadata, is_prefill):
        attn_metadata['position_ids_c'] = {}
        for ratio in (4, 128):
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
            attn_metadata["slot_mapping"][f"c{ratio}a_cmp_kv"] = slot_mapping_func(
                mask,
                position_ids_cmp,
                kv_block_table,
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

        actual_seq_lengths_q = metadata_get("actual_seq_lengths_q")
        actual_seq_lengths_cu_q = metadata_get("actual_seq_lengths_cu_q")
        bsz = actual_seq_lengths_q.shape[0]
        if is_prefill and self.cp_size > 1:
            # generate padded params
            seq_len = int(actual_seq_lengths_q[0])
            total_len = position_ids.shape[-1]
            valid_ids = torch.arange(seq_len, dtype=position_ids.dtype, device=position_ids.device)
            pad_len = total_len - seq_len
            pad_ids = torch.full((pad_len,), 1, dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([valid_ids, pad_ids])
            actual_seq_lengths_q = torch.tensor([position_ids.shape[-1]], device=actual_seq_lengths_q.device)
            actual_seq_lengths_cu_q = actual_seq_lengths_q

        cu_seq_lens_q = torch.cat(
            [torch.zeros_like(actual_seq_lengths_cu_q[:1]), actual_seq_lengths_cu_q],
            dim=0,
        )

        if is_prefill:
            start_pos = torch.zeros([bsz], device=input_ids.device, dtype=torch.int32)
            full_kv_cache, full_block_table, full_slot_mapping = self.build_prefill_full_kv_metadata(
                input_ids,
                actual_seq_lengths_q,
            )
        else:
            start_pos = metadata_get("kv_len") - self.infer_config.model_config.next_n
        if is_prefill and self.cp_size > 1:
            attn_metadata = {
                "is_prefill": is_prefill,
                "is_warm_up": forward_metadata.is_warm_up,
                "batch_size_per_rank": self.infer_config.scheduler_config.batch_size_per_dp_rank,
                "slot_mapping": metadata_get("slot_mapping"),
                "block_table": metadata_get("block_table"),
                "position_ids": position_ids, # padded
                "kv_len": metadata_get("actual_seq_lengths_q").to(torch.int32),
                "start_pos": start_pos.to(torch.int32),
                "actual_seq_q": actual_seq_lengths_q.to(torch.int32),
                "actual_seq_k": actual_seq_lengths_q.to(torch.int32),
                "cu_seq_lens_q": cu_seq_lens_q.to(torch.int32),
                "seq_used_q": metadata_get("actual_seq_lengths_q").to(torch.int32),
                "kernel_metadata": {}
            }
        else:
            attn_metadata = {
                "is_prefill": is_prefill,
                "batch_size_per_rank": self.infer_config.scheduler_config.batch_size_per_dp_rank,
                "slot_mapping": metadata_get("slot_mapping"),
                "block_table": metadata_get("block_table"),
                "position_ids": position_ids,
                "kv_len": actual_seq_lengths_cu_q.to(torch.int32) if is_prefill else position_ids + 1,
                "start_pos": start_pos.to(torch.int32),
                "actual_seq_q": actual_seq_lengths_cu_q.to(torch.int32),
                "actual_seq_k": metadata_get("actual_seq_lengths_kv").to(torch.int32),
                "cu_seq_lens_q": cu_seq_lens_q.to(torch.int32),
                "seq_used_q": actual_seq_lengths_q.to(torch.int32),
                "kernel_metadata": {}
            }
        if not self.is_mtp:
            self.get_cmp_metadata(attn_metadata, is_prefill)
        attn_metadata["shared_expert_stream"] = self.shared_expert_stream
        attn_metadata["mla_stream"] = self.mla_stream

        if is_prefill:
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
                position_ids,
                attn_metadata,
                metadata_get("cp_metadata"),
                self.is_mtp,
            )

        return attn_metadata

    def get_cp_metadata(self, input_ids, is_prefill, attn_metadata, cp_metadata, is_mtp):
        attn_metadata_ori = attn_metadata
        # Process sequence with long padding

        seq_len = input_ids.shape[-1]
        batch_size = attn_metadata_ori["start_pos"].numel()
        assert batch_size == 1, (
            "DeepSeek-V4 CP splitting is only supported during prefill with minibatch size 1; "
            f"got batch_size={batch_size}. Please set prefill minibatch size to 1 or disable CP."
        )

        kv_len = attn_metadata_ori["kv_len"]
        cp_input_dict = {}
        position_ids = attn_metadata_ori["position_ids"]
        cp_segment_num = self.cp_size * 2 # zigzag
        segment_len = seq_len // cp_segment_num

        # (even) split list for hidden_states, position_ids and input_ids
        split_list_hidden = [segment_len] * cp_segment_num * batch_size
        cp_input_dict.update({"split_list": split_list_hidden})
        split_position_ids = list(position_ids.split(split_list_hidden, dim=-1))

        # generate zigzag gather index
        zigzag_idx = list(range(self.global_rank, self.global_rank + batch_size * cp_segment_num, cp_segment_num)) + \
                    list(range(cp_segment_num - self.global_rank - 1, batch_size * cp_segment_num, cp_segment_num))
        cp_input_dict.update({"zigzag_idx": zigzag_idx})

        reverse_index = torch.tensor(
            list(range(0, cp_segment_num, 2)) + list(range(cp_segment_num - 1, 0, -2)),
            device="npu",
        )
        cp_input_dict.update({"reverse_index": reverse_index})

        # Split kv_len for each rank
        split_kv_len = torch.min((torch.arange(cp_segment_num, device="npu") + 1) * segment_len, kv_len)\
                                    - torch.min(torch.arange(cp_segment_num, device="npu") * segment_len, kv_len)
        split_kv_len = split_kv_len.to(torch.int32)
        last_rank_before_zz = ((split_kv_len > 0).sum(dim=0) - 1).item()
        last_kv_len = split_kv_len[last_rank_before_zz].item()
        last_rank_zz, last_rank_flag = self.get_zigzag_idx(last_rank_before_zz, cp_segment_num)
        cp_input_dict.update({
            "split_kv_len": split_kv_len,
            "last_rank": last_rank_before_zz, # the last segment index with kv_len > 0
            "last_rank_flag": last_rank_flag,
            "last_rank_zz": last_rank_zz,
            "decode_token_indices": cp_metadata.persistent_valid_indices
        })

        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]
        if last_rank_before_zz == 0:
            pad_len = align_up(win_kv_slot_mapping.numel(), self.config.sliding_window) - win_kv_slot_mapping.numel()
            if pad_len > 0:
                win_kv_slot_mapping = torch.cat([
                    win_kv_slot_mapping,
                    torch.zeros(
                        pad_len,
                        dtype=win_kv_slot_mapping.dtype,
                        device=win_kv_slot_mapping.device,
                    ),
                ], dim=0)
        attn_metadata["slot_mapping"]["win_kv"] = win_kv_slot_mapping[-self.config.sliding_window:]

        for ratio in self.config.compress_ratios:
            if ratio <= 1:
                continue
            block_table_key = f"c{ratio}a_cmp_kv"
            block_table = attn_metadata_ori["block_table"][block_table_key]
            compressed_len = math.ceil(seq_len / ratio)
            need_blocks = math.ceil(compressed_len / self.block_size)
            pad_blocks = need_blocks - block_table.shape[-1]
            if pad_blocks > 0:
                attn_metadata_ori["block_table"][block_table_key] = F.pad(
                    block_table, (0, pad_blocks), value=0)

        attn_metadata["cp_metadata"] = cp_input_dict
        attn_metadata["prev"] = {}
        attn_metadata["next"] = {}
        attn_metadata["cp_metadata"]["cp_tmp_cache"] = {}
        attn_metadata["cp_metadata"]["tmp_block_table"] = {}
        if not self.is_mtp and not self.is_online:
            # offline and main model
            attn_metadata["cp_metadata"]["cp_tmp_cache"] = self.build_cp_prefill_tmp_cache(
                input_ids,
                attn_metadata,
                cp_metadata,
            )

        for zigzag_flag in ["prev", "next"]:
            segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

            attn_metadata[zigzag_flag].update({
                "is_start": segment_idx == 0, # if current segment is the start
                "is_end": segment_idx == last_rank_before_zz, # if current segment is the last rank
                "cur_kv_len": split_kv_len[segment_idx].item(),
                "block_table": attn_metadata_ori["block_table"],
                "full_kv_cache": attn_metadata_ori["full_kv_cache"],
                "cp_tmp_cache": attn_metadata["cp_metadata"]["cp_tmp_cache"], # only for offline
                "tmp_block_table": attn_metadata["cp_metadata"]["tmp_block_table"], # only for offline
                "kernel_metadata": {}
            })

            if segment_idx > 0:
                position_ids_with_pre_win = torch.cat([split_position_ids[segment_idx - 1][-self.window_size:],
                                                        split_position_ids[segment_idx]], dim=-1)
            else:
                position_ids_with_pre_win = split_position_ids[segment_idx]
            if last_kv_len >= self.window_size:
                # TND
                position_ids_last_win = split_position_ids[last_rank_before_zz][
                    last_kv_len - self.window_size:last_kv_len
                ]
            elif last_rank_before_zz == 0:
                # Keep valid tokens at the start of win_kv when the whole valid sequence is in segment 0.
                position_ids_last_win = split_position_ids[last_rank_before_zz][:self.window_size]
            else:
                position_ids_last_win = split_position_ids[last_rank_before_zz][:last_kv_len]
                position_ids_last_win = torch.cat([
                    split_position_ids[last_rank_before_zz - 1][-(self.window_size - last_kv_len):],
                    position_ids_last_win,
                ], dim=-1)
            attn_metadata[zigzag_flag].update({
                "position_ids_with_pre_win": position_ids_with_pre_win,
                "position_ids_last_win": position_ids_last_win,
                "position_ids_cur": split_position_ids[segment_idx],
                "last_kv_len": last_kv_len,
            })

            # Calculate actual_seq_k, actual_seq_q and cu_seq_lens_q for fa and li
            cur_segment_len = split_list_hidden[segment_idx]
            if segment_idx == 0:
                ori_kv_len = split_kv_len[segment_idx]
            else:
                valid_win_len = self.window_size + split_kv_len[segment_idx - 1] - cur_segment_len
                ori_kv_len = valid_win_len + split_kv_len[segment_idx]
            ori_kv_len = torch.tensor([ori_kv_len] * batch_size, dtype=torch.int32, device="npu")

            slot_mapping_ori_kv = self.get_slot_mapping_from_block_table(
                ori_kv_len, position_ids_with_pre_win, attn_metadata_ori["block_table"]["full_kv"])
            actual_seq_k_val = sum(split_list_hidden[:segment_idx + 1])
            actual_seq_k = torch.full([batch_size], actual_seq_k_val, dtype=torch.int32, device="npu")
            actual_seq_q = torch.arange(
                cur_segment_len,
                cur_segment_len * (batch_size + 1),
                cur_segment_len,
                dtype=torch.int32,
                device="npu",
            )
            cu_seq_lens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device="npu"), actual_seq_q])
            attn_metadata[zigzag_flag].update({
                "slot_mapping_ori_kv": slot_mapping_ori_kv,
                "actual_seq_k": actual_seq_k.to(torch.int32),
                "actual_seq_q": actual_seq_q.to(torch.int32),
                "cu_seq_lens_q": cu_seq_lens_q.to(torch.int32),
            })

        if is_mtp:
            return attn_metadata

        # metadata for compressor
        slot_mapping_cmp_dict = {}
        slot_mapping_cmp_for_decode_dict = {}
        need_decode_slot_mapping = not self.is_online

        for zigzag_flag in ["prev", "next"]:
            attn_metadata[zigzag_flag]["cmp_out_pad"] = {}
            attn_metadata[zigzag_flag]["cmp_in_offset"] = {}

        for ratio in [4, 128]:
            # calculate max output len of compressor and pad len
            overlap_len = ratio if ratio == 4 else 0 # overlap for c4a
            in_lens = (
                torch.tensor([p[0] % ratio + overlap_len for p in split_position_ids])
                + torch.tensor(split_list_hidden)
            )
            in_lens[0] = split_list_hidden[0]
            out_lens = torch.min(torch.stack([in_lens, in_lens // ratio + batch_size], dim=1), dim=1)[0]
            max_out_len = out_lens.max()
            pad_len = max_out_len - out_lens

            slot_mapping_cmp_list = []
            slot_mapping_cmp_for_decode_list = []

            for zigzag_flag in ["prev", "next"]:
                segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

                # Calculate cu_seq_lens, seq_used_q, start_pos, cmp_position_ids, slot_mapping for compressor
                # The slot_mapping of all segments should be concatenated for epilog of the full compressed sequence
                # Since the length of compressor out can be different, we should pad them
                # into the same length before all-gather.
                # Also, the input of compressor should be sliced properly
                res_dict = self.get_cmp_param(
                    input_ids, segment_idx, attn_metadata_ori, split_list_hidden, split_position_ids, split_kv_len)
                attn_metadata[zigzag_flag].update(res_dict)

                # initialize pad tensor for all-gather
                pad_tensor_li = torch.zeros(
                    (pad_len[segment_idx], self.config.index_head_dim),
                    dtype=torch.bfloat16,
                    device="npu",
                )
                pad_tensor_sfa = torch.zeros(
                    (pad_len[segment_idx], self.config.head_dim),
                    dtype=torch.bfloat16,
                    device="npu",
                )
                attn_metadata[zigzag_flag]["cmp_out_pad"][f"{ratio}"] = (pad_tensor_li, pad_tensor_sfa)

                # slot_mapping of each segment should be padded to max_out_len
                cur_slot_mapping_cmp = attn_metadata[zigzag_flag]["slot_mapping_cmp"][f"{ratio}"]

                pad_tensor = torch.full([pad_len[segment_idx]], -1, dtype=torch.int32, device="npu")
                cur_slot_mapping_cmp_pad = torch.cat([pad_tensor, cur_slot_mapping_cmp], dim=0)

                slot_mapping_cmp_list.append(cur_slot_mapping_cmp_pad)
                if need_decode_slot_mapping:
                    cur_slot_mapping_cmp_for_decode = attn_metadata[zigzag_flag][
                        "slot_mapping_cmp_for_decode"
                    ][f"{ratio}"]
                    cur_slot_mapping_cmp_for_decode_pad = torch.cat([
                        pad_tensor,
                        cur_slot_mapping_cmp_for_decode,
                    ], dim=0)
                    slot_mapping_cmp_for_decode_list.append(cur_slot_mapping_cmp_for_decode_pad)

                comp_len = attn_metadata[zigzag_flag]["comp_lens"][f"{ratio}"]
                if segment_idx == 0:
                    cmp_in_offset = torch.zeros([batch_size], dtype=torch.int32, device="npu")
                else:
                    cmp_in_offset = self.window_size - comp_len
                attn_metadata[zigzag_flag]["cmp_in_offset"][f"{ratio}"] = cmp_in_offset[0].item()
            # gather slot_mapping_cmp of all segments, and each of them has been padded to the same length
            cur_slot_mapping_cmp = torch.cat(slot_mapping_cmp_list, dim=0)
            if need_decode_slot_mapping:
                cur_slot_mapping_cmp_for_decode = torch.cat(slot_mapping_cmp_for_decode_list, dim=0)
                all_slot_mapping_cmp_for_decode = cur_slot_mapping_cmp_for_decode.new_empty([
                    cur_slot_mapping_cmp_for_decode.shape[0] * self.cp_size
                ])
                dist.all_gather_into_tensor(
                    all_slot_mapping_cmp_for_decode,
                    cur_slot_mapping_cmp_for_decode,
                    group=self.comm_manager.get_group("cp_group"),
                )
                all_slot_mapping_cmp_for_decode = \
                    all_slot_mapping_cmp_for_decode.view(
                        -1,
                        cur_slot_mapping_cmp_for_decode.shape[0] // 2,
                    )[reverse_index]
                slot_mapping_cmp_for_decode_dict[f"{ratio}"] = all_slot_mapping_cmp_for_decode.flatten(0, 1)

            all_slot_mapping_cmp = cur_slot_mapping_cmp.new_empty([cur_slot_mapping_cmp.shape[0] * self.cp_size])
            dist.all_gather_into_tensor(
                all_slot_mapping_cmp,
                cur_slot_mapping_cmp,
                group=self.comm_manager.get_group("cp_group"),
            )
            all_slot_mapping_cmp = all_slot_mapping_cmp.view(-1, cur_slot_mapping_cmp.shape[0] // 2)[reverse_index]
            slot_mapping_cmp_dict[f"{ratio}"] = all_slot_mapping_cmp.flatten(0, 1)
        cp_metadata_update = {"slot_mapping_cmp": slot_mapping_cmp_dict}
        if need_decode_slot_mapping:
            cp_metadata_update["slot_mapping_cmp_for_decode"] = slot_mapping_cmp_for_decode_dict
        attn_metadata["cp_metadata"].update(cp_metadata_update)
        return attn_metadata

    def get_cmp_param(
        self,
        input_ids,
        rank,
        attn_metadata,
        split_list_hidden,
        split_position_ids,
        split_kv_len,
    ):
        seq_len = input_ids.shape
        batch_size = 1

        # cu_seq_lens, seq_used_q, start_pos and cmp_position_ids for compressor
        cu_seq_lens_dict = {}
        seq_used_q_dict = {}
        start_pos_dict = {}
        position_ids_cmp_for_rope_dict = {}
        slot_mapping_cmp_dict = {}
        slot_mapping_used_cmp_dict = {}
        comp_len_dict = {}

        cur_kv_len = split_kv_len[rank].unsqueeze(0)
        cur_position_ids = split_position_ids[rank].unsqueeze(0)
        cur_segment_len = split_list_hidden[rank]
        for ratio in [4, 128]:
            # length to be complemented to compress the pre remainder of current rank
            # rank 0 should not be padded by left
            if rank == 0:
                comp_len = torch.zeros([batch_size], dtype=torch.int32, device="npu")
            else:
                overlap_len = ratio if ratio == 4 else 0 # overlap for c4a
                comp_len = cur_position_ids[:, 0] % ratio + overlap_len
            comp_len_dict[f"{ratio}"] = comp_len.to(torch.int32)

            # actual seq len to be compressed
            seq_used_q = cur_kv_len + comp_len
            seq_used_q[cur_kv_len == 0] = 0
            seq_used_q_dict[f"{ratio}"] = seq_used_q.to(torch.int32)

            cu_seq_lens = cur_segment_len + comp_len
            cu_seq_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device="npu"), cu_seq_lens])
            cu_seq_lens_dict[f"{ratio}"] = cu_seq_lens.to(torch.int32)

            if rank == 0:
                start_pos = torch.zeros([batch_size], dtype=torch.int32, device="npu")
            else:
                start_pos = sum(split_list_hidden[:rank]) - comp_len
            start_pos_dict[f"{ratio}"] = start_pos.to(torch.int32)

            # position_ids_cmp and slot_mapping for compressor rope and epilog, both in tnd padding format
            compressed_len, position_ids_cmp = self.generate_compressed_position_ids(
                {"start_pos": start_pos, "seq_used_q": seq_used_q, "cu_seq_lens_q": cu_seq_lens}, ratio
            )
            decode_token_indices = attn_metadata["cp_metadata"].get("decode_token_indices", None)
            has_decode_requests = decode_token_indices is not None and decode_token_indices.numel() > 0

            if not self.is_online:
                if has_decode_requests:

                    block_table_used = attn_metadata['block_table'][f'c{ratio}a_cmp_kv']
                    block_table_tmp = attn_metadata["cp_metadata"]["tmp_block_table"][f"c{ratio}a_cmp_kv"]
                    pad_len = block_table_tmp.shape[-1] - block_table_used.shape[-1]

                    if pad_len > 0:
                        pad_block = torch.zeros(
                            1,
                            pad_len,
                            dtype=block_table_tmp.dtype,
                            device=block_table_tmp.device,
                        )
                        block_table_used = torch.concat([block_table_used, pad_block], dim=-1)
                    else:
                        block_table_used = block_table_used[:, :block_table_tmp.shape[-1]]
                else:
                    block_table_tmp = attn_metadata["cp_metadata"]["tmp_block_table"][f"c{ratio}a_cmp_kv"]
                    pad_len = block_table_tmp.shape[-1]
                    block_table_used = torch.zeros(
                        1,
                        pad_len,
                        dtype=block_table_tmp.dtype,
                        device=block_table_tmp.device,
                    )

                dist.all_reduce(
                    block_table_used,
                    # op=dist.ReduceOp.SUM,
                    group=self.comm_manager.get_group("cp_group"),
                )
                slot_mapping_used_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                    compressed_len, position_ids_cmp, block_table_used)
                slot_mapping_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                    compressed_len,
                    position_ids_cmp,
                    attn_metadata["cp_metadata"]["tmp_block_table"][f"c{ratio}a_cmp_kv"],
                )
                if ratio == 4 and rank > 0:
                    offsets = torch.nn.functional.pad(
                        torch.cumsum(compressed_len, dim=0, dtype=torch.int32),
                        (1, 0),
                    )[:-1]
                    slot_mapping_cmp_dict[f"{ratio}"][offsets] = -1
                    slot_mapping_used_cmp_dict[f"{ratio}"][offsets] = -1
                position_ids_cmp_for_rope_dict[f"{ratio}"] = position_ids_cmp * ratio
            else:
                block_table_used = attn_metadata['block_table'][f'c{ratio}a_cmp_kv']
                slot_mapping_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                    compressed_len, position_ids_cmp, block_table_used)
                if ratio == 4 and rank > 0:
                    offsets = torch.nn.functional.pad(
                        torch.cumsum(compressed_len, dim=0, dtype=torch.int32),
                        (1, 0),
                    )[:-1]
                    slot_mapping_cmp_dict[f"{ratio}"][offsets] = -1
                position_ids_cmp_for_rope_dict[f"{ratio}"] = position_ids_cmp * ratio

        res_dict = {
            "cu_seq_lens": cu_seq_lens_dict,
            "seq_used_q": seq_used_q_dict,
            "start_pos": start_pos_dict,
            "position_ids_cmp_for_rope": position_ids_cmp_for_rope_dict,
            "slot_mapping_cmp": slot_mapping_cmp_dict, # tnd padding
            "slot_mapping_cmp_for_decode": slot_mapping_used_cmp_dict, # only for offline
            "comp_lens": comp_len_dict,
        }
        return res_dict

    def get_zigzag_idx(self, origin_idx, cp_segment_num):
        midpoint = cp_segment_num // 2 - 1
        if origin_idx <= midpoint:
            return origin_idx, "prev"
        else:
            return midpoint + 1 - (origin_idx - midpoint), "next"
