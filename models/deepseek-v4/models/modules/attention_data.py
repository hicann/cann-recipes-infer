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
from executor.utils import align_up

def pad_tensor_tnd(ori_tensor, total_len, pad_value):
    new_tensor = torch.full((total_len,), pad_value, dtype=ori_tensor.dtype, device=ori_tensor.device)
    new_tensor[:len(ori_tensor)] = ori_tensor
    return new_tensor

def calc_num_ratio_group(next_n, overlap):
    return (1 if next_n == 0 else 2) + overlap # additional +1 for overlap

class CacheData(nn.Module):
    def __init__(self, config, runner_settings, is_mtp=False, kv_cache_quant_mode=None, li_cache_quant_mode=None):
        super().__init__()
        self.config = config
        self.runner_settings = runner_settings
        self.next_n = self.runner_settings.get("model_config").get("next_n", 0)
        self.is_mtp = is_mtp
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
        self.pre_pad_num_block = 1 # block_id 0 is pad
        self.kv_cache_quant_mode = kv_cache_quant_mode
        self.li_cache_quant_mode = li_cache_quant_mode
        use_fused_kernel_compressor = self.runner_settings.get("kernel_config", {}).get("compressor", "native") == "ascendc"
        platform_version = self.runner_settings.get("model_config").get("platform_version", "A3")
        self.cmp_kv_dtype = torch.float8_e4m3fn if (platform_version == "950" and use_fused_kernel_compressor == True) else torch.bfloat16
        self.update_cache_param()
        self.window_size = config.sliding_window

    def update_cache_param(self):
        cache_dtype_map = {
            "int8": torch.int8, "float8": torch.float8_e4m3fn, "unquant": torch.bfloat16
        }
        self.cache_dtype = cache_dtype_map[self.kv_cache_quant_mode]
        self.li_cache_dtype = cache_dtype_map[self.li_cache_quant_mode]
        self.cache_dim = self.config.head_dim

        if self.kv_cache_quant_mode == "float8":
            rope_dim = self.config.qk_rope_head_dim
            nope_dim = self.config.head_dim - rope_dim
            self.cmp_kv_dtype = self.cache_dtype
            # when FA FP8 quant is enabled, nope_cache, rope_cache, and scales
            # are concatenated and passed via the kv input (FP8).
            # nope(FP8-8bit) + rope(BF16-2*8bit) + scales(e8m0-8bit) // 64(tile_size) + pad(align_up 128)
            self.cache_dim = align_up(nope_dim + 2 * rope_dim + nope_dim // 64, 128)

    def create_cache(self, block_num, dim, dtype):
        cache_shape = (
            block_num,
            self.block_size,
            1,                  # MQA: num_kv_head = 1
            dim
        )
        return torch.zeros(cache_shape, dtype=dtype, device="npu")

    def create_state_cache(self, state_block_num, compress_ratio, cache_dim):
        assert compress_ratio == 4 or compress_ratio == 128
        overlap_num = 2 if compress_ratio == 4 else 1
        cache_shape = (
            state_block_num,
            self.block_size,
            2,              # state + score
            overlap_num,
            cache_dim
        )
        return torch.zeros(cache_shape, dtype=torch.float32, device="npu")

    def get_block_num(self, cache_size):
        return math.ceil(cache_size / self.block_size) * self.batch_size_per_rank + self.pre_pad_num_block # pad block 0

    def update_win_cache(self, cache_dict):
        win_block_num = self.get_block_num(self.window_size + self.next_n)
        cache_dict.update({
            "win_kv": self.create_cache(win_block_num, self.cache_dim, self.cache_dtype)
        })
        return cache_dict

    def init_cache_c1a(self, cache_dict):
        return self.update_win_cache(cache_dict)

    def init_full_buffer_c1a(self):
        full_block_num = self.get_block_num(self.pa_max_length)
        return self.create_cache(full_block_num, self.cache_dim, self.cache_dtype)

    def init_cache_c4a(self, cache_dict):
        compress_ratio = 4
        cmp_block_num = self.get_block_num(self.pa_max_length // compress_ratio)
        num_ratio_group = calc_num_ratio_group(next_n=self.next_n, overlap=1)
        state_block_num = self.get_block_num(num_ratio_group * compress_ratio)
        cache_dict = self.update_win_cache(cache_dict)
        cache_dict.update({
            "sfa_cmp_kv": self.create_cache(cmp_block_num, self.cache_dim, self.cmp_kv_dtype),
            "sfa_kv_state": self.create_state_cache(state_block_num, compress_ratio, self.config.head_dim),
            "li_cmp_kv": self.create_cache(cmp_block_num, self.config.index_head_dim, self.li_cache_dtype),
            "li_kv_state": self.create_state_cache(state_block_num, compress_ratio, self.config.index_head_dim),
        })
        if self.li_cache_quant_mode in ["int8", "float8"]:
            dtype = torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
            cache_dict.update({
                "li_key_dequant_scale": self.create_cache(cmp_block_num, 1, dtype)
            })
        return cache_dict

    def init_cache_c128a(self, cache_dict):
        compress_ratio = 128
        cmp_block_num = self.get_block_num(self.pa_max_length // compress_ratio)
        num_ratio_group = calc_num_ratio_group(next_n=self.next_n, overlap=0)
        state_block_num = self.get_block_num(num_ratio_group * compress_ratio)
        cache_dict = self.update_win_cache(cache_dict)
        cache_dict.update({
            "sfa_cmp_kv": self.create_cache(cmp_block_num, self.cache_dim, self.cmp_kv_dtype),
            "sfa_kv_state": self.create_state_cache(state_block_num, compress_ratio, self.config.head_dim),
        })
        return cache_dict

    def init_cache_single_layer(self, ratio, cache_dict):
        method_name = f"init_cache_c{ratio}a"
        init_func = getattr(self, method_name, None)

        if init_func and callable(init_func):
            return init_func(cache_dict)
        else:
            raise ValueError(f"{ratio=} must in [1, 4, 128]")

    def init_cache_data(self, num_hidden_layers=43):
        cache_data = ()
        compress_ratios = self.config.compress_ratios
        for layer_id in range(num_hidden_layers):
            ratio = compress_ratios[layer_id] if not self.is_mtp else 1  # MTP layer use window attention
            cache_dict = {
                "win_kv": None, "sfa_cmp_kv": None, "li_cmp_kv": None, "sfa_kv_state": None,
                "sfa_score_state": None, "li_kv_state": None, "li_score_state": None,
                "li_key_dequant_scale": None
            }
            cache_dict = self.init_cache_single_layer(ratio, cache_dict)
            cache_data += (cache_dict,)
        return cache_data

    def create_tmp_cache(self, minibatch=1, num_hidden_layers=43):
        # tmp cache for prefill minibatch, containing all kinds of cache
        self.batch_size_per_rank = minibatch

        cache_dict_c1a = self.init_cache_c1a({
                "win_kv": None, "sfa_cmp_kv": None, "li_cmp_kv": None, "sfa_kv_state": None,
                "sfa_score_state": None, "li_kv_state": None, "li_score_state": None,
                "li_key_dequant_scale": None
            })
        cache_dict_c4a = self.init_cache_c4a({
                "win_kv": None, "sfa_cmp_kv": None, "li_cmp_kv": None, "sfa_kv_state": None,
                "sfa_score_state": None, "li_kv_state": None, "li_score_state": None,
                "li_key_dequant_scale": None
            })
        cache_dict_c128a = self.init_cache_c128a({
                "win_kv": None, "sfa_cmp_kv": None, "li_cmp_kv": None, "sfa_kv_state": None,
                "sfa_score_state": None, "li_kv_state": None, "li_score_state": None,
                "li_key_dequant_scale": None
            })

        cache_data = ()
        compress_ratios = self.config.compress_ratios
        for layer_id in range(num_hidden_layers):
            ratio = compress_ratios[layer_id] if not self.is_mtp else 1  # MTP layer use window attention
            if ratio == 1:
                cache_data += (cache_dict_c1a,)
            elif ratio == 4:
                cache_data += (cache_dict_c4a,)
            else:
                cache_data += (cache_dict_c128a,)

        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)

        return cache_data


class AttnMetaData(nn.Module):
    def __init__(self, config, runner_settings, is_mtp=False):
        super().__init__()
        self.config = config
        self.block_size = runner_settings.get("model_config").get("pa_block_size", 128)
        self.pre_pad_num_block = 1 # block_id 0 is pad
        self.pa_max_length = runner_settings.get("model_config").get("pa_max_length", 2048)
        self.next_n = runner_settings.get("model_config").get("next_n", 0)
        self.is_mtp = is_mtp
        self.window_size = config.sliding_window
        self.slot_mapping_pad_value = -1
        self.position_ids_pad_value = 1

        self.cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset

        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)

        self.metadata_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.shared_expert_stream = torch.npu.Stream() if self.enable_multi_streams else None
        # mla_stream, compressor_stream and indexer_stream are exclusively used in the decoding phase.
        self.mla_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.compressor_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.indexer_stream = torch.npu.Stream() if self.enable_multi_streams else None

    def get_block_num_per_batch(self, cache_size):
        return math.ceil(cache_size / self.block_size)

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
        slot_mapping = block_table[row_indices, indices // self.block_size] * self.block_size + indices % self.block_size
        return pad_tensor_tnd(slot_mapping.view(-1), position_ids.shape[0], self.slot_mapping_pad_value)

    def calc_full_buffer_block_table(
        self,
        block_num_per_batch,
        batch_size_per_rank
    ):
        """
        Used in kv cache for prefill win attn and compressed kv cache for c4a/c128a

       Suppose block_num_per_batch = 3, batch_size = 2. Block table will look like:
            1 2 3
            4 5 6
        """
        # compressed kv cache for c4a/c128a
        block_table = torch.arange(0, batch_size_per_rank * block_num_per_batch, dtype=torch.int32, device="npu").\
            view(batch_size_per_rank, -1) + self.pre_pad_num_block

        return block_table

    def calc_ring_buffer_block_table(
        self,
        block_num_per_batch,
        batch_size_per_rank
    ):
        """
        Used in kv cache for decode win attn and state cache for c4a/c128a

        Suppose block_num_per_batch = 2, block_table_len = 6, batch_size = 3. Block table will look like:
            1 2 1 2 1 2
            3 4 3 4 3 4
            5 6 5 6 5 6
        """
        block_table_len = math.ceil(self.pa_max_length / self.block_size)
        block_table_offset = torch.arange(batch_size_per_rank * block_num_per_batch, dtype=torch.int32, device="npu").\
            view(batch_size_per_rank, -1) + self.pre_pad_num_block
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]

        return block_table

    def calc_state_block_table(
            self,
            block_num_per_batch,
            batch_size_per_rank,
            attn_metadata,
            is_prefill,
    ):
        actual_seq_len = attn_metadata['start_pos'] + attn_metadata['seq_used_q']
        block_table_len = math.ceil(self.pa_max_length / self.block_size)

        actual_block_start_pos = (attn_metadata['start_pos'] // self.block_size).view(batch_size_per_rank, 1).repeat(1, block_table_len)
        actual_block_end_pos = ((actual_seq_len - 1) // self.block_size).view(batch_size_per_rank, 1).repeat(1, block_table_len)
        block_pos_ids = torch.arange(block_table_len, dtype=torch.int32, device="npu").repeat(batch_size_per_rank, 1)
        block_table_offset = torch.arange(batch_size_per_rank * block_num_per_batch, dtype=torch.int32, device="npu").\
            view(batch_size_per_rank, -1) + self.pre_pad_num_block
        repeat_num = math.ceil(block_table_len / block_num_per_batch)
        block_table_offset = block_table_offset.repeat(1, repeat_num)[:, :block_table_len]

        block_table = torch.zeros((batch_size_per_rank, block_table_len), dtype=torch.int32, device="npu")
        if is_prefill:
            block_table = torch.where(
                block_pos_ids == actual_block_end_pos,
                block_table_offset,
                block_table
            )
        else:
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

    def init_win_kv_block_table_and_slot_mapping(
        self,
        attn_metadata,
        is_prefill
    ):
        batch_size_per_rank = attn_metadata["batch_size_per_rank"]
        position_ids = attn_metadata["position_ids"]
        q_len = attn_metadata["cu_seq_lens_q"].diff() # note: cu_seq_lens_q uses bsnd-format padding
        block_num_per_batch_win = self.get_block_num_per_batch(self.window_size + self.next_n)
        block_table_win = self.calc_ring_buffer_block_table(block_num_per_batch_win, batch_size_per_rank)
        attn_metadata['block_table'].update({'win_kv': block_table_win})
        if is_prefill:
            block_num_per_batch_full = self.get_block_num_per_batch(self.pa_max_length)
            bsz, seq_len = position_ids.shape[0], position_ids.shape[1]
            position_ids = torch.arange(seq_len, device=position_ids.device).repeat(bsz, 1)
            block_table_full = self.calc_full_buffer_block_table(block_num_per_batch_full, batch_size_per_rank)
            slot_mapping_full = self.get_slot_mapping_from_block_table(q_len, position_ids, block_table_full)
            attn_metadata['block_table'].update({'full_kv': block_table_full})
            attn_metadata['slot_mapping'].update({'full_kv': slot_mapping_full})

            # slot_mapping to extact tail cache from full cache
            kv_len = attn_metadata['kv_len']
            bsz = kv_len.shape[0]
            start_pos = torch.clamp(kv_len - self.window_size, min=0)
            token_indices = torch.arange(self.window_size, dtype=torch.int32, device="npu")
            gather_indices = start_pos.unsqueeze(1) + token_indices.unsqueeze(0)
            attn_metadata['slot_mapping'].update({'full_kv_gather_indices': gather_indices})
            q_len = torch.full((kv_len.shape[0],), self.window_size, dtype=torch.int32, device="npu")
        else:
            gather_indices = position_ids

        slot_mapping = self.get_slot_mapping_from_block_table(q_len, gather_indices, block_table_win)
        attn_metadata['slot_mapping'].update({'win_kv': slot_mapping})

    def calc_qkv_lengths(
        self,
        input_ids,
        attention_mask,
        kv_len,
        is_prefill
    ):
        batch_size, seq_len = input_ids.size()
        if is_prefill:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, self.position_ids_pad_value)
            # Obtain the actual length of the request
            kv_len = torch.max(position_ids, axis=1)[0] + 1
            seq_used_q = kv_len.to(torch.int32)
            actual_seq_lengths_kv = torch.full((batch_size,), seq_len, dtype=torch.int32, device="npu")
        else:
            actual_seq_lengths_kv = torch.max(kv_len, axis=1)[0] if seq_len > 1 else kv_len
            position_ids = kv_len.view(-1, seq_len) - 1
            seq_used_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device="npu")

        actual_seq_lengths_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device="npu") * seq_len
        cu_seq_lens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device="npu"), actual_seq_lengths_q])
        qkv_lengths = {
            'kv_len': kv_len.to(torch.int32),
            "actual_seq_q": actual_seq_lengths_q, # padded q
            "actual_seq_k": actual_seq_lengths_kv.to(torch.int32), # padded kv
            "position_ids": position_ids.to(torch.int32),
            "cu_seq_lens_q": cu_seq_lens_q, # li/sfa/cmpr; shape is batchsize + 1
            "seq_used_q": seq_used_q, # sfa/cmpr, not padded
            "start_pos": position_ids[:, 0].to(torch.int32), # cmpr
        }
        return qkv_lengths

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
        end_pos = (attn_metadata["start_pos"] +  attn_metadata["seq_used_q"]) // ratio
        compressed_len = end_pos - start_pos
        offsets = torch.nn.functional.pad(torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0))[:-1]
        expanded_starts = torch.repeat_interleave(start_pos, compressed_len)
        expanded_offsets = torch.repeat_interleave(offsets, compressed_len)
        flat_range = torch.arange(compressed_len.sum(), dtype=torch.int32, device="npu")
        compressed_ids = flat_range - expanded_offsets + expanded_starts
        max_len = min(attn_metadata["cu_seq_lens_q"][-1], attn_metadata["cu_seq_lens_q"][-1] // ratio + bsz)
        position_ids_cmp = pad_tensor_tnd(compressed_ids, max_len, self.position_ids_pad_value)
        return compressed_len, position_ids_cmp

    def get_cmp_attn_metadata(
        self,
        ratio,
        overlap,
        attn_metadata,
        is_prefill
    ):
        # 1. position_ids
        q_len, position_ids_cmp = self.generate_compressed_position_ids(attn_metadata, ratio)
        attn_metadata['position_ids_c'].update(
            {str(ratio): position_ids_cmp * ratio}
        )

        # 2. block_table and slot_mapping
        batch_size_per_rank = attn_metadata["batch_size_per_rank"]
        # cmp_kv
        kv_block_num_per_batch = self.get_block_num_per_batch(self.pa_max_length // ratio)
        kv_block_table = self.calc_full_buffer_block_table(kv_block_num_per_batch, batch_size_per_rank)
        attn_metadata["block_table"][f"c{ratio}a_cmp_kv"] = kv_block_table
        attn_metadata["slot_mapping"][f"c{ratio}a_cmp_kv"] = \
            self.get_slot_mapping_from_block_table(q_len, position_ids_cmp, kv_block_table)

        # kv_state and score state
        num_ratio_group = calc_num_ratio_group(self.next_n, overlap)
        state_block_num_per_batch = self.get_block_num_per_batch(num_ratio_group * ratio)
        attn_metadata["block_table"][f"c{ratio}a_cmp_state"] = \
            self.calc_state_block_table(state_block_num_per_batch, batch_size_per_rank, attn_metadata, is_prefill)

    def get_attn_metadata(
        self,
        input_ids,
        attention_mask,
        kv_len,
        is_prefill=True
    ):
        """
        Generate attn_metadata, a dict with the following structure.

        attn_metadata = {
            "block_table": {
                "win_kv": win_kv_cache,
                "c4a_kv_cache": c4a_cv_cache,
                ...
            }
            "slot_mapping": {
                "win_kv": win_kv_cache,
                "c4a_kv_cache": c4a_cv_cache,
                ...
            }
            "position_ids": position_ids,
            "position_ids_c": position_ids_c,
            "cu_seq_lens_q": cu_seq_lens_q,
            ...
        }
        """
        batch_size, _ = input_ids.size()
        attn_metadata = {"batch_size_per_rank": batch_size}

        attn_metadata['shared_expert_stream'] = self.shared_expert_stream
        if not is_prefill:
            attn_metadata['metadata_stream'] = self.metadata_stream
            attn_metadata['mla_stream'] = self.mla_stream
            attn_metadata['compressor_stream'] = self.compressor_stream
            attn_metadata['indexer_stream'] = self.indexer_stream
        attn_metadata['kernel_metadata'] = {}

        # initialize seq_len and position_ids etc.
        misc_len = self.calc_qkv_lengths(input_ids, attention_mask, kv_len, is_prefill)
        attn_metadata.update(misc_len)

        attn_metadata['block_table'] = {}
        attn_metadata['slot_mapping'] = {}

        # c1a block_table/slot_mapping
        self.init_win_kv_block_table_and_slot_mapping(attn_metadata, is_prefill)

        if self.is_mtp:
            return attn_metadata

        attn_metadata['position_ids_c'] = {}
        # c4a position_ids and block_table/slot_mapping
        self.get_cmp_attn_metadata(
            ratio=4,
            overlap=1,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill
        )
        # c128a position_ids and block_table/slot_mapping
        self.get_cmp_attn_metadata(
            ratio=128,
            overlap=0,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill
        )

        return attn_metadata

    def get_cp_metadata(self, input_ids, is_prefill, attn_metadata, is_mtp, hccl_comm_dict):
        attn_metadata_ori = attn_metadata
        # Process sequence with long padding
        # only support minibatch = 1
        if not is_prefill or self.cp_size == 1:
            attn_metadata.update({"cp_metadata": None})
            return

        batch_size, seq_len = input_ids.shape
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

        reverse_index = torch.tensor(list(range(0, cp_segment_num, 2)) + list(range(cp_segment_num - 1, 0, -2)), device="npu")
        cp_input_dict.update({"reverse_index": reverse_index})

        # Split kv_len for each rank
        split_kv_len = torch.min((torch.arange(cp_segment_num, device="npu").unsqueeze(0) + 1) * segment_len, kv_len)\
                                    - torch.min(torch.arange(cp_segment_num, device="npu").unsqueeze(0) * segment_len, kv_len)
        split_kv_len = split_kv_len.to(torch.int32)
        last_rank_before_zz = ((split_kv_len > 0).sum(dim=1) - 1).item()
        last_rank_zz, last_rank_flag = self.get_zigzag_idx(last_rank_before_zz, cp_segment_num)
        cp_input_dict.update({
            "split_kv_len": split_kv_len,
            "last_rank": last_rank_before_zz, # the last segment index with kv_len > 0, should be >= 1,
            "last_rank_flag": last_rank_flag,
            "last_rank_zz": last_rank_zz,
        })
        attn_metadata["cp_metadata"] = cp_input_dict
        attn_metadata["prev"] = {}
        attn_metadata["next"] = {}

        for zigzag_flag in ["prev", "next"]:
            segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

            attn_metadata[zigzag_flag].update({
                "is_start": segment_idx == 0, # if current segment is the start
                "is_end": segment_idx == last_rank_before_zz, # if current segment is the last rank
                "cur_kv_len": split_kv_len[0, segment_idx].item(),
                "block_table": attn_metadata_ori["block_table"],
                "full_kv_cache": attn_metadata_ori["full_kv_cache"],
                "kernel_metadata": {}
            })

            if segment_idx > 0:
                position_ids_with_pre_win = torch.cat([split_position_ids[segment_idx - 1][:, -self.window_size:],
                                                        split_position_ids[segment_idx]], dim=-1)
            else:
                position_ids_with_pre_win = split_position_ids[segment_idx]
            last_kv_len = split_kv_len[:, last_rank_before_zz]
            if last_kv_len >= self.window_size:
                position_ids_last_win = split_position_ids[last_rank_before_zz][:, last_kv_len - self.window_size:last_kv_len]
            else:
                position_ids_last_win = split_position_ids[last_rank_before_zz][:, :last_kv_len]
                position_ids_last_win = torch.cat([split_position_ids[last_rank_before_zz - 1][:, -(self.window_size - last_kv_len):], position_ids_last_win], dim=-1)
            attn_metadata[zigzag_flag].update({
                "position_ids_with_pre_win": position_ids_with_pre_win,
                "position_ids_last_win": position_ids_last_win,
                "position_ids_cur": split_position_ids[segment_idx],
                "last_kv_len": split_kv_len[0, last_rank_before_zz].item(),
            })

            # Calculate actual_seq_k, actual_seq_q and cu_seq_lens_q for fa and li
            cur_segment_len = split_list_hidden[segment_idx]
            ori_kv_len = cur_segment_len + self.window_size if segment_idx > 0 else cur_segment_len
            ori_kv_len = torch.tensor([ori_kv_len] * batch_size, dtype=torch.int32, device="npu") # only for minibatch = 1
            slot_mapping_ori_kv = self.get_slot_mapping_from_block_table(
                ori_kv_len, position_ids_with_pre_win, attn_metadata_ori["block_table"]["full_kv"])
            actual_seq_k_val = sum(split_list_hidden[:segment_idx + 1])
            actual_seq_k = torch.full([batch_size], actual_seq_k_val, dtype=torch.int32, device="npu")
            actual_seq_q = torch.arange(cur_segment_len, cur_segment_len * (batch_size + 1), cur_segment_len, dtype=torch.int32, device="npu")
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
        for zigzag_flag in ["prev", "next"]:
            attn_metadata[zigzag_flag]["cmp_out_pad"] = {}
            attn_metadata[zigzag_flag]["cmp_in_offset"] = {}

        for ratio in [4, 128]:
            # calculate max output len of compressor and pad len
            overlap_len = ratio if ratio == 4 else 0 # overlap for c4a
            in_lens = torch.tensor([p[0, 0] % ratio + overlap_len for p in split_position_ids]) + torch.tensor(split_list_hidden)
            in_lens[0] = split_list_hidden[0]
            out_lens = torch.min(torch.stack([in_lens, in_lens // ratio + batch_size], dim=1), dim=1)[0]
            max_out_len = out_lens.max()
            pad_len = max_out_len - out_lens

            slot_mapping_cmp_list = []
            for zigzag_flag in ["prev", "next"]:
                segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

                # Calculate cu_seq_lens, seq_used_q, start_pos, cmp_position_ids, slot_mapping for compressor
                # The slot_mapping of all segments should be concatenated for epilog of the full compressed sequence
                # Since the length of compressor out can be different, we should pad them into the same length before all-gather
                # Also, the input of compressor should be sliced properly
                res_dict = self.get_cmp_param(
                    input_ids, segment_idx, attn_metadata_ori, split_list_hidden, split_position_ids, split_kv_len)
                attn_metadata[zigzag_flag].update(res_dict)

                # initialize pad tensor for all-gather
                pad_tensor_li = torch.zeros((pad_len[segment_idx], self.config.index_head_dim), dtype=torch.bfloat16, device="npu")
                pad_tensor_sfa = torch.zeros((pad_len[segment_idx], self.config.head_dim), dtype=torch.bfloat16, device="npu")
                attn_metadata[zigzag_flag]["cmp_out_pad"][f"{ratio}"] = (pad_tensor_li, pad_tensor_sfa)

                # slot_mapping of each segment should be padded to max_out_len
                cur_slot_mapping_cmp = attn_metadata[zigzag_flag]["slot_mapping_cmp"][f"{ratio}"]
                pad_tensor = torch.full([pad_len[segment_idx]], -1, dtype=torch.int32, device="npu")
                cur_slot_mapping_cmp_pad = torch.cat([pad_tensor, cur_slot_mapping_cmp], dim=0)
                slot_mapping_cmp_list.append(cur_slot_mapping_cmp_pad)

                comp_len = attn_metadata[zigzag_flag]["comp_lens"][f"{ratio}"]
                if segment_idx == 0:
                    cmp_in_offset = torch.zeros([batch_size], dtype=torch.int32, device="npu")
                else:
                    cmp_in_offset = self.window_size - comp_len
                attn_metadata[zigzag_flag]["cmp_in_offset"][f"{ratio}"] = cmp_in_offset[0].item()
            # gather slot_mapping_cmp of all segments, and each of them has been padded to the same length
            cur_slot_mapping_cmp = torch.cat(slot_mapping_cmp_list, dim=0)
            all_slot_mapping_cmp = cur_slot_mapping_cmp.new_empty([cur_slot_mapping_cmp.shape[0] * self.cp_size])
            dist.all_gather_into_tensor(all_slot_mapping_cmp, cur_slot_mapping_cmp, group=hccl_comm_dict["cp_group"])
            all_slot_mapping_cmp = all_slot_mapping_cmp.view(-1, cur_slot_mapping_cmp.shape[0] // 2)[reverse_index]
            slot_mapping_cmp_dict[f"{ratio}"] = all_slot_mapping_cmp.flatten(0, 1)
        attn_metadata["cp_metadata"].update({
            "slot_mapping_cmp": slot_mapping_cmp_dict,
        })
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
        batch_size, seq_len = input_ids.shape

        # cu_seq_lens, seq_used_q, start_pos and cmp_position_ids for compressor
        cu_seq_lens_dict = {}
        seq_used_q_dict = {}
        start_pos_dict = {}
        position_ids_cmp_for_rope_dict = {}
        slot_mapping_cmp_dict = {}
        comp_len_dict = {}

        cur_kv_len = split_kv_len[:, rank]
        cur_position_ids = split_position_ids[rank]
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
            slot_mapping_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                compressed_len, position_ids_cmp, attn_metadata["block_table"][f"c{ratio}a_cmp_kv"])
            if ratio == 4 and rank > 0:
                offsets = torch.nn.functional.pad(torch.cumsum(compressed_len, dim=0, dtype=torch.int32), (1, 0))[:-1]
                slot_mapping_cmp_dict[f"{ratio}"][offsets] = -1
            position_ids_cmp_for_rope_dict[f"{ratio}"] = position_ids_cmp * ratio

        res_dict = {
            "cu_seq_lens": cu_seq_lens_dict,
            "seq_used_q": seq_used_q_dict,
            "start_pos": start_pos_dict,
            "position_ids_cmp_for_rope": position_ids_cmp_for_rope_dict,
            "slot_mapping_cmp": slot_mapping_cmp_dict, # tnd padding
            "comp_lens": comp_len_dict,
        }
        return res_dict

    def get_zigzag_idx(self, origin_idx, cp_segment_num):
        midpoint = cp_segment_num // 2 - 1
        if origin_idx <= midpoint:
            return origin_idx, "prev"
        else:
            return midpoint + 1 - (origin_idx - midpoint), "next"