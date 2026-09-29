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
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from executor.core.config import InferenceConfig, PlatformVersion
from .common_modules import (PACKED_KV_STORAGE_DTYPE, PACKED_KV_COMPUTE_DTYPE, get_kv_cache_dim,
                             gather_cp_segments, has_cp_decode_requests)
from .indexer import MXFP4_VALUES_PER_BYTE, MXFP4_SCALE_GROUP_SIZE


def _engram_decode_prefix_row(ids, prefix_len, image_token_id, engram_pad_id):
    # all_token_ids includes the current query token t; Engram needs the
    # tokens immediately before it. n-grams must not span image tokens: an
    # image token and everything farther away is padded out, and a short
    # history is left-padded with engram_pad_id.
    tail = ids[:-1][-prefix_len:]
    row, blocked = [], False
    for tid in reversed(tail):
        blocked = blocked or tid == image_token_id
        row.append(engram_pad_id if blocked else tid)
    row.reverse()
    return [engram_pad_id] * (prefix_len - len(tail)) + row


def build_engram_hash_inputs(config, input_ids, is_prefill,
                             actual_seq_lengths_q, batch, padded_seq_lengths=None):
    """Build the hash inputs consumed by models.modules.engram.

    Returns (prefix_input_ids, ngram_shift_mask); decode builds
    prefix_input_ids [bsz, ngram_size-1] from the cached request
    histories, prefill builds ngram_shift_mask [total_tokens,
    ngram_size] (True = the token at that shift is usable) over the
    full sequence, before CP splits it. Either output is None when
    ngram_size <= 1 disables engram hashing entirely.

    CP pads every request at its own tail, so padded_seq_lengths gives the row
    stride of each request while actual_seq_lengths_q gives its valid span.
    """
    prefix_input_ids = None
    ngram_shift_mask = None
    prefix_len = max(0, int(config.engram_max_ngram_size) - 1)
    if prefix_len:
        if is_prefill:
            # Packed prefill tokens must not borrow history across a request
            # boundary or an image token.
            total_tokens = input_ids.reshape(-1).numel()
            ngram_shift_mask = torch.zeros(
                total_tokens,
                prefix_len + 1,
                dtype=torch.bool,
                device=input_ids.device,
            )
            flat_ids = input_ids.reshape(-1)
            image_token_id = config.image_token_id
            seq_lens = [int(seq_len) for seq_len in actual_seq_lengths_q.tolist()]
            strides = padded_seq_lengths if padded_seq_lengths is not None else seq_lens
            offset = 0
            for seq_len, stride in zip(seq_lens, strides):
                local_pos = torch.arange(seq_len, device=input_ids.device)
                order = torch.arange(prefix_len + 1, device=input_ids.device)
                image_mask = flat_ids[offset:offset + seq_len] == image_token_id
                distance_from_blocker = local_pos - torch.cummax(
                    local_pos.masked_fill(~image_mask, -1), 0
                ).values
                # A shift is valid only if it stays after the nearest blocker.
                ngram_shift_mask[offset:offset + seq_len] = (
                    distance_from_blocker[:, None] > order[None, :]
                )
                offset += stride
        else:
            bsz = actual_seq_lengths_q.shape[0]
            requests = getattr(batch, "requests", None) if batch is not None else None
            if requests is not None:
                histories = [request.get_all_token_ids() for request in requests]
            else:
                histories = []
            histories = histories[:bsz] + [[]] * max(0, bsz - len(histories))
            image_token_id = config.image_token_id
            engram_pad_id = config.engram_pad_id
            prefix_input_ids = torch.tensor(
                [
                    _engram_decode_prefix_row(ids, prefix_len, image_token_id, engram_pad_id)
                    for ids in histories
                ],
                dtype=torch.long, device=input_ids.device)
    return prefix_input_ids, ngram_shift_mask


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
        self.slot_mapping_pad_value = -1
        self.position_ids_pad_value = 1
        self.platform_version = self.infer_config.model_config.platform_version

        # The draft model consumes what prefill CP hands over, so it never splits segments itself.
        self.cp_size = 1 if is_mtp else self.infer_config.parallel_config.cp_size
        self.is_mtp = is_mtp
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.window_size = config.sliding_window
        # Compression ratios with a compressed cache, e.g. [2, 1] for C2A and C1A.
        self.cmp_ratios = [ratio for ratio in dict.fromkeys(config.compress_ratios) if ratio > 0]
        self.candidate_block_size = config.candidate_block_size
        # Only the source layers a QSLI consumer reads need the combined layout.
        self.qsli_ratios = {
            config.compress_ratios[layer] for layer in config.kv_source_layers
            if layer >= config.candidate_source_layer
        }
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.low_latency_tp = bool(
            self.infer_config.model_config.custom_params.get("low_latency_tp", False)
        )
        self.metadata_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.shared_expert_stream = torch.npu.Stream() if self.enable_multi_streams else None
        self.moe_events = {}
        if self.enable_multi_streams:
            self.moe_events = {
                "shared_expert": [torch.npu.Event() for _ in range(2)],
                "shared_expert_internal": [torch.npu.Event()],
                "shared_pipeline": [torch.npu.Event()],
                "moe_gmm1": [torch.npu.Event()],
            }
        self.attention_events = {}
        if self.low_latency_tp:
            # Decoder layers execute serially, so every Attention instance can
            # reuse these event pools after its record/wait pairs complete.
            self.attention_events = {
                "separate_mla": [torch.npu.Event() for _ in range(3)],
                "compressor": [torch.npu.Event() for _ in range(2)],
                "indexer_weights": [torch.npu.Event() for _ in range(2)],
            }
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16"
        )
        self.update_kv_quant_settings()
        self.update_gmm_quant_mode()
        self.init_cache_dim()

    def update_kv_quant_settings(self):
        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")

    def update_gmm_quant_mode(self):
        if self.platform_version == PlatformVersion.ASCEND_950 and \
            "w4" in self.config.quant_config.gmm_quant_mode and "mx" not in self.config.quant_config.gmm_quant_mode:
            self.config.quant_config.gmm_quant_mode = \
                self.config.quant_config.gmm_quant_mode.replace("float", "mxfloat")

    def init_cache_dim(self):
        self.cache_dim = get_kv_cache_dim(self.config.head_dim)
        self.cmp_cache_dim = get_kv_cache_dim(self.config.head_dim, is_compressed=True)

    def get_cmp_kv_dtype(self):
        return PACKED_KV_STORAGE_DTYPE

    def get_win_kv_cache_dtype(self):
        return PACKED_KV_COMPUTE_DTYPE

    def create_cache(self, block_num, dim, dtype, device):
        return torch.zeros((block_num, self.block_size, 1, dim), dtype=dtype, device=device)

    def create_state_cache(self, state_block_num, state_block_size, cache_dim, device):
        # RingCache layout: one block per request, kv and score state per row.
        return torch.zeros(
            (state_block_num, state_block_size, 2, cache_dim),
            dtype=torch.float32,
            device=device,
        )

    def get_state_block_size(self, ratio):
        """Mirror the block size of the RingCache state entry the model registers for this ratio."""
        return ratio if self.next_n == 0 else self.next_n + 1

    def build_cp_tmp_block_table(self, batch_size, block_num_per_batch, device):
        block_ids = torch.arange(
            batch_size * block_num_per_batch,
            dtype=torch.int32,
            device=device,
        ).view(batch_size, block_num_per_batch)
        return block_ids + 1

    def verify_schedule_statecache(self, owned_request_num, schedule_block_table, ratio):
        state_key = f"c{ratio}a_cmp_state"
        if state_key not in schedule_block_table:
            raise KeyError(f"Missing scheduled state block table: {state_key}.")

        # RingCache holds one state block per request, and only the owner rank allocates it.
        block_ids = schedule_block_table[state_key]
        schedule_block_num = torch.unique(block_ids[block_ids != 0]).numel()
        if schedule_block_num != owned_request_num:
            raise ValueError(
                f"Invalid {state_key} block allocation for CP prefill decode cache update: "
                f"scheduled block num {int(schedule_block_num)} must be equal to "
                f"owned request num {owned_request_num}."
            )

    def build_cp_tmp_state_cache_by_layer(
        self,
        ratio,
        state_block_num,
        device,
    ):
        tmp_state_cache_by_layer = {}
        compress_ratios = self.config.compress_ratios
        for layer_idx in self.config.kv_source_layers:
            if compress_ratios[layer_idx] != ratio:
                continue
            tmp_state_cache_by_layer[layer_idx] = {
                "state_cache": self.create_state_cache(
                    state_block_num, self.get_state_block_size(ratio), self.config.head_dim, device),
            }
        return tmp_state_cache_by_layer

    def build_cp_prefill_tmp_cache(self, input_ids, attn_metadata):
        """Allocate full-length temporary compressed KV/state cache for CP prefill."""
        batch_size = attn_metadata["start_pos"].numel()
        # Requests are split independently, so each one is padded to its own segment length.
        padded_token_nums = [
            seg_len * self.cp_size * 2 for seg_len in attn_metadata["cp_metadata"]["segment_lens"]
        ]
        device = input_ids.device
        cmp_kv_dtype = self.get_cmp_kv_dtype()
        indexer_cache_dim = self.config.index_head_dim // MXFP4_VALUES_PER_BYTE
        indexer_cache_scale_dim = self.config.index_head_dim // MXFP4_SCALE_GROUP_SIZE

        # One cache per ratio: source layer groups run one after another on the main stream,
        # so a later group of the same ratio only overwrites rows its predecessors already consumed.
        tmp_cache = {}
        tmp_block_table = {}
        for ratio in self.cmp_ratios:
            # Every request gets the same number of blocks, sized for the longest one.
            cmp_block_num_per_batch = max(
                math.ceil(math.ceil(padded_token_num / ratio) / self.block_size)
                for padded_token_num in padded_token_nums
            )
            cmp_block_num = cmp_block_num_per_batch * batch_size + 1

            # Rows are quantized into a scratch buffer before they travel, so the gather moves
            # quantized bytes. Identity slots keep this rank's rows contiguous in the scratch.
            # Both segments emit the padded row count the compressor metadata uses, so size the
            # scratch by that same expression.
            segment_tokens = sum(attn_metadata["cp_metadata"]["segment_lens"])
            scratch_rows = 2 * min(segment_tokens, segment_tokens // ratio + batch_size)
            scratch_block_num = math.ceil(scratch_rows / self.block_size)

            ratio_cache = {
                "cmp_cache": self.create_cache(cmp_block_num, self.cmp_cache_dim, cmp_kv_dtype, device),
                "cmp_scratch": self.create_cache(
                    scratch_block_num, self.cmp_cache_dim, cmp_kv_dtype, device),
                "indexer_cache": self.create_cache(cmp_block_num, indexer_cache_dim, torch.uint8, device),
                "indexer_cache_scale": self.create_cache(
                    cmp_block_num, indexer_cache_scale_dim, torch.uint8, device),
                "indexer_scratch": self.create_cache(
                    scratch_block_num, indexer_cache_dim, torch.uint8, device),
                "indexer_scratch_scale": self.create_cache(
                    scratch_block_num, indexer_cache_scale_dim, torch.uint8, device),
            }
            tmp_block_table[f"c{ratio}a_cmp_kv"] = self.build_cp_tmp_block_table(
                batch_size, cmp_block_num_per_batch, device)

            if ratio in self.qsli_ratios:
                # QSLI reads the same rows in candidate-sized groups: one entry holds the K of
                # candidate_block_size positions followed by their scales.
                rows_per_block = (self.block_size * ratio) // self.candidate_block_size
                qsli_block_num_per_batch = max(
                    math.ceil(math.ceil(math.ceil(padded_token_num / ratio)
                                        / self.candidate_block_size) / rows_per_block)
                    for padded_token_num in padded_token_nums
                )
                ratio_cache["qsli_cache"] = torch.zeros(
                    (qsli_block_num_per_batch * batch_size + 1, rows_per_block, 1,
                     self.candidate_block_size * (indexer_cache_dim + indexer_cache_scale_dim)),
                    dtype=torch.uint8, device=device)
                # Where each request starts in either cache, so the repack can walk both.
                ratio_cache["qsli_layout"] = (
                    cmp_block_num_per_batch, qsli_block_num_per_batch, rows_per_block)
                tmp_block_table[f"c{ratio}a_qsli_combined_kv"] = self.build_cp_tmp_block_table(
                    batch_size, qsli_block_num_per_batch, device)

            if ratio > 1:
                # One RingCache state block per request, plus the null block.
                state_block_num = batch_size + 1
                ratio_cache["state_cache_by_layer"] = self.build_cp_tmp_state_cache_by_layer(
                    ratio, state_block_num, device)
                tmp_block_table[f"c{ratio}a_cmp_state"] = self.build_cp_tmp_block_table(batch_size, 1, device)

            tmp_cache[str(ratio)] = ratio_cache
            if (ratio > 1 and has_cp_decode_requests(attn_metadata["cp_metadata"])
                    and not attn_metadata["is_warm_up"]):
                self.verify_schedule_statecache(
                    attn_metadata["cp_metadata"]["owner_request_indices"].numel(),
                    attn_metadata["block_table"],
                    ratio,
                )

        attn_metadata["cp_metadata"]["tmp_block_table"] = tmp_block_table
        return tmp_cache

    def pad_tensor_tnd(self, ori_tensor, total_len, pad_value):
        new_tensor = torch.full((total_len,), pad_value, dtype=ori_tensor.dtype, device=ori_tensor.device)
        new_tensor[:len(ori_tensor)] = ori_tensor
        return new_tensor

    def generate_win_topk_ids(self, position_ids, topk=128):
        """Generate causal sliding-window token indices for each query token."""
        position_ids = position_ids.reshape(-1)
        offsets = torch.arange(
            topk,
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        valid_len = torch.clamp(position_ids + 1, max=topk)
        window_start = position_ids - valid_len + 1
        topk_ids = window_start.unsqueeze(1) + offsets.unsqueeze(0)
        valid_mask = offsets.unsqueeze(0) < valid_len.unsqueeze(1)
        return torch.where(valid_mask, topk_ids, torch.full_like(topk_ids, -1))

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
        return self.pad_tensor_tnd(
            slot_mapping.view(-1).to(torch.int32), position_ids.shape[0], self.slot_mapping_pad_value
            )

    def get_cmp_metadata(self, attn_metadata, is_prefill):
        attn_metadata['position_ids_c'] = {}
        # 1. position_ids
        position_ids_func = self.generate_compressed_position_ids
        mask, position_ids_cmp = position_ids_func(attn_metadata, 2)
        attn_metadata['position_ids_c'].update(
            {str(2): position_ids_cmp * 2}
        )
        # 2. block_table
        kv_block_table = attn_metadata["block_table"][f"c2a_cmp_kv"]

        slot_mapping_func = self.get_slot_mapping_from_block_table
        attn_metadata["slot_mapping"][f"c2a_cmp_kv"] = slot_mapping_func(
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

        # Temporary full-length Win KV cache used during prefill.
        full_kv_cache = torch.zeros(
            (
                full_block_num + 1,
                self.block_size,
                1,
                self.cache_dim,
            ),
            dtype=self.get_win_kv_cache_dtype(),
            device=input_ids.device,
        )
        return full_kv_cache, full_block_table, full_slot_mapping

    def get_window_metadata(self, attn_metadata, is_prefill):
        """Map each packed query's causal window to [T, 1, W] physical slots."""
        positions = attn_metadata["position_ids"].reshape(-1).long()
        cu_lengths = attn_metadata["cu_seq_lens_q"]
        request_ids = torch.repeat_interleave(
            torch.arange(cu_lengths.numel() - 1, device=positions.device),
            (cu_lengths[1:] - cu_lengths[:-1]).long(), output_size=positions.numel())
        window = positions[:, None] + torch.arange(1 - self.window_size, 1, device=positions.device)
        safe_window = window.clamp_min(0)
        table = attn_metadata["block_table"]["full_kv" if is_prefill else "win_kv"]
        blocks = table[request_ids[:, None], safe_window // self.block_size].long()
        slots = blocks * self.block_size + safe_window % self.block_size
        attn_metadata["query_request_ids"] = request_ids
        attn_metadata["win_idx"] = slots.masked_fill((window < 0) | (blocks == 0), -1).unsqueeze(1).int()

    def build_attn_metadata(self, input_ids, position_ids, forward_metadata, batch):
        is_prefill = forward_metadata.is_prefill

        metadata_get = forward_metadata.get if hasattr(forward_metadata, "get") else \
            lambda key: getattr(forward_metadata, key)

        actual_seq_lengths_q = metadata_get("actual_seq_lengths_q")
        actual_seq_lengths_cu_q = metadata_get("actual_seq_lengths_cu_q")
        bsz = actual_seq_lengths_q.shape[0]
        is_cp_prefill = is_prefill and self.cp_size > 1

        padded_lens = None
        if is_cp_prefill:
            padded_lens = [
                seg_len * self.cp_size * 2 for seg_len in self.get_cp_segment_lens(metadata_get("cp_metadata"))
            ]

        engram_prefix_ids, engram_shift_mask = None, None
        if not self.is_mtp:
            engram_prefix_ids, engram_shift_mask = build_engram_hash_inputs(
                self.config, input_ids, is_prefill,
                actual_seq_lengths_q, batch, padded_lens,
            )

        if is_cp_prefill:
            # Rebuild the position ids of every request: real positions for its valid tokens,
            # then a constant for the CP padding that follows them.
            padded_position_ids = []
            for seq_len, padded_len in zip(actual_seq_lengths_q.tolist(), padded_lens):
                padded_position_ids.append(
                    torch.arange(seq_len, dtype=position_ids.dtype, device=position_ids.device))
                padded_position_ids.append(
                    torch.full((padded_len - seq_len,), self.position_ids_pad_value,
                               dtype=position_ids.dtype, device=position_ids.device))
            position_ids = torch.cat(padded_position_ids)
            actual_seq_lengths_q = torch.tensor(
                padded_lens, dtype=actual_seq_lengths_q.dtype, device=actual_seq_lengths_q.device)
            actual_seq_lengths_cu_q = actual_seq_lengths_q.cumsum(dim=0)

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

        if not self.is_mtp:
            engram_metadata = {
                "prefix_input_ids": engram_prefix_ids,
                "shift_mask": engram_shift_mask,
                "actual_seq_lengths": actual_seq_lengths_q.to(
                    device=input_ids.device,
                    dtype=torch.int32,
                ),
            }
        else:
            engram_metadata = {}

        if is_cp_prefill:
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
                "engram": engram_metadata,
                "kernel_metadata": {},
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
                "engram":engram_metadata,
                "kernel_metadata": {},
            }
        # Prefill CP builds these per segment instead.
        if not is_cp_prefill:
            win_topk_ids = self.generate_win_topk_ids(
                position_ids, topk=self.window_size)
            attn_metadata["win_topk_ids"] = win_topk_ids
            attn_metadata["win_sparse_indices"] = win_topk_ids.unsqueeze(1).int()
            attn_metadata["win_topk_length"] = (win_topk_ids >= 0).sum(dim=-1, dtype=torch.int32).unsqueeze(1)
        attn_metadata["max_seqlen_q"] = int(actual_seq_lengths_q.max().item())
        if not self.is_mtp:
            self.get_cmp_metadata(attn_metadata, is_prefill)
        attn_metadata["shared_expert_stream"] = self.shared_expert_stream
        attn_metadata["metadata_stream"] = self.metadata_stream
        attn_metadata["moe_events"] = self.moe_events
        attn_metadata["attention_events"] = self.attention_events

        if is_prefill:
            attn_metadata["block_table"].update({"full_kv": full_block_table})
            attn_metadata["slot_mapping"].update({"full_kv": full_slot_mapping})
            attn_metadata["full_kv_cache"] = full_kv_cache

        if not is_cp_prefill:
            self.get_window_metadata(attn_metadata, is_prefill)

        if is_cp_prefill:
            attn_metadata = self.get_cp_metadata(
                input_ids,
                attn_metadata,
                metadata_get("cp_metadata"),
            )
            for zigzag_flag in ["prev", "next"]:
                seg_metadata = attn_metadata.get(zigzag_flag)
                if seg_metadata is None:
                    raise KeyError(f"Missing CP attention metadata for {zigzag_flag}.")
                self._add_compressed_seq_metadata(seg_metadata)
                self._add_cp_segment_topk_metadata(seg_metadata, input_ids.device)
        else:
            self._add_compressed_seq_metadata(attn_metadata)

        attn_metadata["indexer_slot_mapping"] = self._prepare_indexer_slot_mapping(
            attn_metadata, input_ids.device
        )
        attn_metadata["cache_slot_mapping"] = self._prepare_cache_slot_mapping(attn_metadata)
        if not is_cp_prefill:
            attn_metadata["cmp_topk_length_by_ratio"] = self._prepare_cmp_topk_lengths(
                attn_metadata, input_ids.device
            )
        return attn_metadata

    def _prepare_indexer_slot_mapping(self, attn_metadata, device):
        """Build flattened int64 cache-write slots for Indexer cache updates."""
        indexer_slots = {
            key: slots.view(-1).to(device=device, dtype=torch.int64)
            for key, slots in attn_metadata["slot_mapping"].items()
            if key.startswith("c") and key.endswith("a_cmp_kv")
        }

        # QSLI stores one candidate-sized group of K values and scales in each
        # cache element.  Its cache manager therefore has a different block
        # size from the split QLI cache and needs slots based on compressed
        # positions rather than the ordinary token positions.
        for key in attn_metadata["slot_mapping"]:
            if not (key.startswith("c") and key.endswith("a_qsli_combined_kv")):
                continue
            try:
                ratio = int(key[1:key.index("a_")])
            except (ValueError, IndexError):
                continue
            position_ids_cmp = attn_metadata.get("position_ids_c", {}).get(str(ratio))
            block_table = attn_metadata["block_table"].get(key)
            if position_ids_cmp is None or block_table is None:
                indexer_slots[key] = attn_metadata["slot_mapping"][key].view(-1).to(
                    device=device, dtype=torch.int64
                )
                continue

            start_pos = attn_metadata["start_pos"] // ratio
            end_pos = (attn_metadata["start_pos"] + attn_metadata["seq_used_q"]) // ratio
            compressed_len = (end_pos - start_pos).to(torch.int32)
            valid_token_count = int(compressed_len.sum().item())
            valid_positions = position_ids_cmp[:valid_token_count]
            row_indices = torch.repeat_interleave(
                torch.arange(
                    compressed_len.numel(), dtype=torch.int32, device=position_ids_cmp.device
                ),
                compressed_len,
            )
            combined_block_size = (
                self.block_size * ratio
            ) // int(getattr(self.config, "candidate_block_size", 8))
            block_indices = valid_positions // combined_block_size
            block_offsets = valid_positions % combined_block_size
            block_ids = block_table[row_indices, block_indices]
            slots = block_ids * combined_block_size + block_offsets
            indexer_slots[key] = self.pad_tensor_tnd(
                slots.to(torch.int64), position_ids_cmp.numel(), self.slot_mapping_pad_value
            ).to(device=device, dtype=torch.int64)

        return indexer_slots

    def _prepare_cache_slot_mapping(self, attn_metadata):
        """Normalize cache-write slots after window and CP metadata updates."""
        cache_slot_mapping = {}
        for key in ("win_kv", "full_kv"):
            slot_mapping = attn_metadata["slot_mapping"].get(key)
            if slot_mapping is None:
                continue
            slots = slot_mapping.reshape(-1).long()
            valid = slots >= self.block_size
            cache_slot_mapping[key] = slots.masked_fill(~valid, -1)
        return cache_slot_mapping

    def _add_cp_segment_topk_metadata(self, seg_metadata, device):
        """Build per-segment window indices and top-k lengths for the two CP attention calls."""
        # Kernel metadata is built per segment and reads the positions under this key.
        seg_metadata["position_ids"] = seg_metadata["position_ids_cur"]
        win_topk_ids = self.generate_win_topk_ids(seg_metadata["position_ids_cur"], topk=self.window_size)
        seg_metadata["win_sparse_indices"] = win_topk_ids.unsqueeze(1).int()
        seg_metadata["win_topk_length"] = (win_topk_ids >= 0).sum(dim=-1, dtype=torch.int32).unsqueeze(1)
        seg_metadata["cmp_topk_length_by_ratio"] = self._prepare_cmp_topk_lengths(seg_metadata, device)

    def _prepare_cmp_topk_lengths(self, attn_metadata, device):
        """Compute valid compressed top-k lengths for every query once per step."""
        positions = attn_metadata["position_ids"].reshape(-1).to(device=device)
        lengths = {"0": torch.zeros_like(attn_metadata["win_topk_length"])}
        for ratio in dict.fromkeys(self.config.compress_ratios):
            if ratio <= 0:
                continue
            valid_length = (positions + 1) // ratio
            lengths[str(ratio)] = (
                valid_length.clamp_max(self.config.index_topk)
                .to(torch.int32)
                .view(-1, 1)
            )
        return lengths

    def _add_compressed_seq_metadata(self, attn_metadata):
        actual_seq_k = attn_metadata["actual_seq_k"]
        compressed_seq_lens = {}
        compressed_seq_remainders = {}
        for ratio in dict.fromkeys(self.config.compress_ratios):
            if ratio < 1:
                continue
            ratio_key = f"{ratio}"
            compressed_seq_lens[ratio_key] = actual_seq_k // ratio
            compressed_seq_remainders[ratio_key] = actual_seq_k % ratio
        attn_metadata["compressed_seq_lens"] = compressed_seq_lens
        attn_metadata["compressed_seq_remainders"] = compressed_seq_remainders

    def gather_cp_rows(self, local_rows, cp_metadata):
        """All-gather one value per compressed row and restore the segment order."""
        return gather_cp_segments(
            local_rows.unsqueeze(-1), cp_metadata, self.comm_manager.get_group("cp_group")).squeeze(-1)

    def get_cp_segment_lens(self, cp_metadata):
        """Zigzag segment length of every request, as the framework computed it."""
        prev_cu = cp_metadata.actual_seq_q_prev
        return torch.diff(prev_cu, prepend=prev_cu.new_zeros(1)).tolist()

    def build_cp_slot_mapping(self, block_table, block_positions, valid_lens):
        """Slots for per-request blocks that keep their padding at the end of each block.

        CP pads inside every request, so the packed helper, which expects all padding at the end
        of the batch, does not apply here.
        """
        mappings = []
        for row, (positions, valid_len) in enumerate(zip(block_positions, valid_lens)):
            valid_positions = positions[:valid_len]
            slots = (
                block_table[row, valid_positions // self.block_size] * self.block_size
                + valid_positions % self.block_size
            ).to(torch.int32)
            mappings.append(self.pad_tensor_tnd(slots, positions.shape[0], self.slot_mapping_pad_value))
        return torch.cat(mappings, dim=0)

    def get_cp_metadata(self, input_ids, attn_metadata, cp_metadata):
        if self.is_online:
            raise NotImplementedError("Prefill CP is only implemented for offline inference.")
        attn_metadata_ori = attn_metadata
        # Process sequence with long padding

        batch_size = attn_metadata_ori["start_pos"].numel()
        kv_len = attn_metadata_ori["kv_len"]
        cp_input_dict = {}
        position_ids = attn_metadata_ori["position_ids"]
        cp_segment_num = self.cp_size * 2 # zigzag
        # Each request is split on its own, so segment lengths differ between requests.
        segment_lens = self.get_cp_segment_lens(cp_metadata)

        # Split list for hidden_states, position_ids and input_ids, request major:
        # all segments of request 0, then all segments of request 1, and so on.
        split_list_hidden = [seg_len for seg_len in segment_lens for _ in range(cp_segment_num)]
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

        # Valid token count of every segment of every request, shaped [batch, cp_segment_num].
        seg_len_col = torch.tensor(segment_lens, dtype=torch.int32, device="npu").view(-1, 1)
        bounds = torch.arange(cp_segment_num + 1, dtype=torch.int32, device="npu").view(1, -1) * seg_len_col
        kv_len_col = kv_len.to(torch.int32).view(-1, 1)
        split_kv_len = torch.minimum(bounds[:, 1:], kv_len_col) - torch.minimum(bounds[:, :-1], kv_len_col)
        split_kv_len_list = split_kv_len.tolist()
        last_segment_idx = [sum(1 for n in row if n > 0) - 1 for row in split_kv_len_list]
        last_kv_len = [split_kv_len_list[b][last_segment_idx[b]] for b in range(batch_size)]
        last_segment_rank, last_segment_half = [], []
        for segment_idx in last_segment_idx:
            rank, half = self.get_zigzag_idx(segment_idx, cp_segment_num)
            last_segment_rank.append(rank)
            last_segment_half.append(half)
        cp_input_dict.update({
            "segment_lens": segment_lens,
            "batch_size": batch_size,
            # When a segment length is not a multiple of the ratio, later segments start mid group.
            "needs_remainder": any(
                seg_len % ratio for seg_len in segment_lens for ratio in self.cmp_ratios if ratio > 1),
            # Per request: the last segment with kv_len > 0, and the rank and local half it sits in.
            "last_segment_idx": last_segment_idx,
            "last_segment_half": last_segment_half,
            "last_segment_rank": last_segment_rank,
            "last_kv_len": last_kv_len,
            "decode_token_indices": cp_metadata.persistent_valid_indices,
            "owner_request_indices": cp_metadata.output_request_indices,
        })

        # Window handed to decode: the last window_size valid slots of every request, padded at the
        # tail with the null slot when a request holds fewer tokens than one window.
        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]
        last_win_slots, offset = [], 0
        for seq_len in kv_len.tolist():
            req_slots = win_kv_slot_mapping[offset:offset + seq_len]
            offset += seq_len
            if seq_len < self.window_size:
                req_slots = F.pad(
                    req_slots, (0, self.window_size - seq_len), value=self.slot_mapping_pad_value)
            last_win_slots.append(req_slots[-self.window_size:])
        cp_input_dict["slot_mapping_last_win"] = torch.cat(last_win_slots, dim=0)

        for ratio in self.cmp_ratios:
            block_table_key = f"c{ratio}a_cmp_kv"
            block_table = attn_metadata_ori["block_table"][block_table_key]
            need_blocks = max(
                math.ceil(math.ceil(seg_len * cp_segment_num / ratio) / self.block_size)
                for seg_len in segment_lens
            )
            pad_blocks = need_blocks - block_table.shape[-1]
            if pad_blocks > 0:
                attn_metadata_ori["block_table"][block_table_key] = F.pad(
                    block_table, (0, pad_blocks), value=0)

        attn_metadata["cp_metadata"] = cp_input_dict
        attn_metadata["prev"] = {}
        attn_metadata["next"] = {}
        attn_metadata["cp_metadata"]["cp_tmp_cache"] = self.build_cp_prefill_tmp_cache(
            input_ids,
            attn_metadata,
        )

        for zigzag_flag in ["prev", "next"]:
            segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

            # This segment of every request, in the request major order the local tokens follow.
            flat_idx = [b * cp_segment_num + segment_idx for b in range(batch_size)]
            cur_segment_lens = [split_list_hidden[i] for i in flat_idx]
            cur_kv_len = [split_kv_len_list[b][segment_idx] for b in range(batch_size)]

            # Rows of this segment that hold a valid token, in the packed order the
            # compressor emits. Requests sit one segment length apart in the input.
            valid_rows, row_offset = [], 0
            for segment_len, kv_len_b in zip(cur_segment_lens, cur_kv_len):
                valid_rows.append(torch.arange(
                    row_offset, row_offset + kv_len_b, dtype=torch.long, device="npu"))
                row_offset += segment_len

            attn_metadata[zigzag_flag].update({
                "is_start": segment_idx == 0, # if current segment is the start
                "segment_idx": segment_idx,
                "cur_kv_len": cur_kv_len,
                "segment_lens": cur_segment_lens,
                "valid_rows": torch.cat(valid_rows, dim=0),
                "block_table": attn_metadata_ori["block_table"],
                "full_kv_cache": attn_metadata_ori["full_kv_cache"],
                "tmp_block_table": attn_metadata["cp_metadata"]["tmp_block_table"],
                "kernel_metadata": {}
            })
            attn_metadata[zigzag_flag].update({
                "position_ids_cur": torch.cat([split_position_ids[i] for i in flat_idx], dim=-1),
            })

            # Slots of the window that ends in the previous segment of the same request, followed
            # by the slots of the segment itself. Valid tokens sit at the front of each request
            # block, so the two are built together and split afterwards.
            if segment_idx == 0:
                kv_blocks = [split_position_ids[i] for i in flat_idx]
                block_valid_lens = cur_kv_len
            else:
                kv_blocks = [
                    torch.cat([split_position_ids[i - 1][-self.window_size:], split_position_ids[i]], dim=-1)
                    for i in flat_idx
                ]
                # A segment with valid tokens always follows a fully valid segment, so its whole
                # leading window is valid. A segment without them reads nothing and writes nothing.
                block_valid_lens = [
                    self.window_size + cur_kv_len[b] if cur_kv_len[b] > 0 else 0
                    for b in range(batch_size)
                ]
            block_slots = self.build_cp_slot_mapping(
                attn_metadata_ori["block_table"]["full_kv"], kv_blocks, block_valid_lens)
            pre_win_slots, seg_slots, offset = [], [], 0
            for cur_len in cur_segment_lens:
                if segment_idx > 0:
                    pre_win_slots.append(block_slots[offset:offset + self.window_size])
                    offset += self.window_size
                seg_slots.append(block_slots[offset:offset + cur_len])
                offset += cur_len

            # Both halves are sized per request by the framework.
            actual_seq_k = cp_metadata.kv_len_prev if zigzag_flag == "prev" else cp_metadata.kv_len_next
            actual_seq_q = (
                cp_metadata.actual_seq_q_prev if zigzag_flag == "prev" else cp_metadata.actual_seq_q_next)
            attn_metadata[zigzag_flag].update({
                "slot_mapping_seg": torch.cat(seg_slots, dim=0),
                "slot_mapping_pre_win": torch.cat(pre_win_slots, dim=0) if pre_win_slots else None,
                "actual_seq_k": actual_seq_k.to(torch.int32),
                "cu_seq_lens_q": F.pad(actual_seq_q.to(torch.int32), (1, 0)),
            })

        # metadata for compressor
        slot_mapping_cmp_dict = {}
        slot_mapping_cmp_for_decode_dict = {}

        for ratio in self.cmp_ratios:
            slot_mapping_cmp_list = []
            slot_mapping_cmp_for_decode_list = []
            cmp_request_ids_list = []

            for zigzag_flag in ["prev", "next"]:
                segment_idx = self.global_rank if zigzag_flag == "prev" else 2 * self.cp_size - 1 - self.global_rank

                # Calculate cu_seq_lens, seq_used_q, start_pos, cmp_position_ids, slot_mapping for compressor
                # The slot_mapping of all segments should be concatenated for epilog of the full compressed sequence
                res_dict = self.get_cmp_param(
                    segment_idx, attn_metadata_ori, segment_lens, split_kv_len_list)
                attn_metadata[zigzag_flag].update(res_dict)

                # One compressor call covers the whole batch, so all segments emit the same row
                # count; the rows a segment does not fill carry the -1 slot and are dropped.
                slot_mapping_cmp_list.append(attn_metadata[zigzag_flag]["slot_mapping_cmp"][f"{ratio}"])
                slot_mapping_cmp_for_decode_list.append(
                    attn_metadata[zigzag_flag]["slot_mapping_cmp_for_decode"][f"{ratio}"])
                cmp_request_ids_list.append(attn_metadata[zigzag_flag]["cmp_request_ids"][f"{ratio}"])

            # gather slot_mapping_cmp of all segments, and each of them has the same length
            cur_slot_mapping_cmp = torch.cat(slot_mapping_cmp_list, dim=0)
            cur_slot_mapping_cmp_for_decode = torch.cat(slot_mapping_cmp_for_decode_list, dim=0)
            all_slot_mapping_cmp_for_decode = self.gather_cp_rows(
                cur_slot_mapping_cmp_for_decode, cp_input_dict)
            # Every rank holds the mapping of all requests, but may only write its own.
            all_request_ids = self.gather_cp_rows(torch.cat(cmp_request_ids_list, dim=0), cp_input_dict)
            owned = torch.zeros([batch_size], dtype=torch.bool, device=all_request_ids.device)
            owner_request_indices = cp_metadata.output_request_indices
            if owner_request_indices is not None and owner_request_indices.numel() > 0:
                owned[owner_request_indices] = True
            keep = (all_request_ids >= 0) & owned[all_request_ids.clamp_min(0).long()]
            slot_mapping_cmp_for_decode_dict[f"{ratio}"] = torch.where(
                keep, all_slot_mapping_cmp_for_decode,
                torch.full_like(all_slot_mapping_cmp_for_decode, self.slot_mapping_pad_value))

            slot_mapping_cmp_dict[f"{ratio}"] = self.gather_cp_rows(cur_slot_mapping_cmp, cp_input_dict)
        attn_metadata["cp_metadata"].update({
            "slot_mapping_cmp": slot_mapping_cmp_dict,
            "slot_mapping_cmp_for_decode": slot_mapping_cmp_for_decode_dict,
        })
        return attn_metadata

    def get_cmp_param(
        self,
        segment_idx,
        attn_metadata,
        segment_lens,
        split_kv_len_list,
    ):
        batch_size = len(segment_lens)

        # cu_seq_lens, seq_used_q, start_pos and cmp_position_ids for compressor
        cu_seq_lens_dict = {}
        seq_used_q_dict = {}
        start_pos_dict = {}
        position_ids_cmp_for_rope_dict = {}
        slot_mapping_cmp_dict = {}
        slot_mapping_used_cmp_dict = {}
        request_ids_dict = {}
        remainder_dict = {}

        # Segment start in the coordinates of its own request.
        segment_starts = [segment_len * segment_idx for segment_len in segment_lens]
        cur_kv_len = torch.tensor(
            [split_kv_len_list[b][segment_idx] for b in range(batch_size)], dtype=torch.int32, device="npu")
        cur_segment_lens = torch.tensor(segment_lens, dtype=torch.int32, device="npu")
        for ratio in self.cmp_ratios:
            # The segment is fed as is; a group it starts inside is completed from the state cache.
            seq_used_q_dict[f"{ratio}"] = cur_kv_len
            cu_seq_lens = F.pad(cur_segment_lens.cumsum(dim=0).to(torch.int32), (1, 0))
            cu_seq_lens_dict[f"{ratio}"] = cu_seq_lens
            if ratio > 1:
                remainder_dict[f"{ratio}"] = self.get_remainder_param(
                    attn_metadata, ratio, segment_idx, segment_starts, split_kv_len_list)

            start_pos = torch.tensor(segment_starts, dtype=torch.int32, device="npu")
            start_pos_dict[f"{ratio}"] = start_pos

            # position_ids_cmp and slot_mapping for compressor rope and epilog, both in tnd padding format
            compressed_len, position_ids_cmp = self.generate_compressed_position_ids(
                {"start_pos": start_pos, "seq_used_q": cur_kv_len, "cu_seq_lens_q": cu_seq_lens}, ratio
            )
            block_table_tmp = attn_metadata["cp_metadata"]["tmp_block_table"][f"c{ratio}a_cmp_kv"]
            if has_cp_decode_requests(attn_metadata["cp_metadata"]):
                block_table_used = attn_metadata['block_table'][f'c{ratio}a_cmp_kv']
                # get_cp_metadata already widened the scheduled table to the temporary one.
                # Copy before the in-place all-reduce, otherwise the reduction would write
                # other owners' block ids into the scheduled block table.
                block_table_used = block_table_used[:, :block_table_tmp.shape[-1]].clone()
            else:
                block_table_used = torch.zeros_like(block_table_tmp)

            # Persistent blocks live on the owner rank only, so the owner's block table is
            # shared with every rank and each rank builds the mapping of its own segments.
            # The sum relies on the scheduler leaving the rows of requests a rank does not
            # own at zero, which is how it allocates the persistent caches.
            dist.all_reduce(
                block_table_used,
                group=self.comm_manager.get_group("cp_group"),
            )
            slot_mapping_used_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                compressed_len, position_ids_cmp, block_table_used)
            # Block ids of different owners are not comparable, so the request each row belongs
            # to travels with the mapping and selects the rows a rank may write.
            request_ids_dict[f"{ratio}"] = self.pad_tensor_tnd(
                torch.repeat_interleave(
                    torch.arange(batch_size, dtype=torch.int32, device="npu"),
                    compressed_len.to(torch.int64),
                ),
                position_ids_cmp.shape[0],
                self.slot_mapping_pad_value,
            )
            slot_mapping_cmp_dict[f"{ratio}"] = self.get_slot_mapping_from_block_table(
                compressed_len, position_ids_cmp, block_table_tmp)
            position_ids_cmp_for_rope_dict[f"{ratio}"] = position_ids_cmp * ratio

        res_dict = {
            "cu_seq_lens": cu_seq_lens_dict,
            "cmp_seq_used_q": seq_used_q_dict,
            "start_pos": start_pos_dict,
            "position_ids_cmp_for_rope": position_ids_cmp_for_rope_dict,
            "slot_mapping_cmp": slot_mapping_cmp_dict, # tnd padding
            "slot_mapping_cmp_for_decode": slot_mapping_used_cmp_dict,
            "cmp_request_ids": request_ids_dict, # the request every compressed row belongs to
            "remainder": remainder_dict, # None per ratio unless the segment starts inside a group
        }
        return res_dict

    def get_remainder_param(self, attn_metadata, ratio, segment_idx, segment_starts, split_kv_len_list):
        """Requests whose segment starts inside a compression group, and their state cache slots."""
        # A segment without valid tokens is skipped: writing it would overwrite the tail state.
        requests = [
            b for b, start in enumerate(segment_starts)
            if start % ratio and split_kv_len_list[b][segment_idx] > 0
        ]
        if not requests:
            return None
        state_block_table = attn_metadata["cp_metadata"]["tmp_block_table"][f"c{ratio}a_cmp_state"]
        device = state_block_table.device
        request_index = torch.tensor(requests, dtype=torch.long, device=device)
        # The RingCache row of a token is its own position taken modulo the block size.
        block_size = self.get_state_block_size(ratio)
        rows = torch.tensor(
            [(segment_starts[b] - 1) % block_size for b in requests], dtype=torch.long, device=device)
        return {
            "requests": request_index,
            "slots": state_block_table[:, 0].index_select(0, request_index).long() * block_size + rows,
        }

    def get_zigzag_idx(self, origin_idx, cp_segment_num):
        midpoint = cp_segment_num // 2 - 1
        if origin_idx <= midpoint:
            return origin_idx, "prev"
        else:
            return midpoint + 1 - (origin_idx - midpoint), "next"
