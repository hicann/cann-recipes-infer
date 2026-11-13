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

"""DeepSeek-V4 private CP prefill metadata and planner helpers.

V4 segment/window/indexer plans extend the executor's public CP metadata
without adding model-specific fields to other models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from executor.utils.forward_metadata import PrefillCPMetaData


# Model-specific compression ratios shared by V4 planners and AttentionData.
CP_COMPRESS_RATIOS = (4, 128)


def build_prefill_cp_query_metadata(
    segment_lens: List[int],
    current_valid_lens: Any,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build physical TND boundaries and per-request valid query lengths.

    ``cu_seq_lens_q`` indexes physical rows; ``seq_used_q`` excludes tail
    padding, including partially valid CP segments.
    """
    if device is None and isinstance(current_valid_lens, torch.Tensor):
        device = current_valid_lens.device

    physical_q_lens = torch.tensor(segment_lens, dtype=torch.int32, device=device)
    cu_seq_lens_q = torch.cat((
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(physical_q_lens, dim=0, dtype=torch.int32),
    ))
    if isinstance(current_valid_lens, torch.Tensor):
        seq_used_q = current_valid_lens.view(-1)
    else:
        seq_used_q = torch.tensor(current_valid_lens, dtype=torch.int32, device=device).view(-1)
    return cu_seq_lens_q, seq_used_q


def build_prefill_cp_window_plan(
    segment_lens: List[int],
    segment_valid_lens: Any,
    cp_size: int,
    cp_rank: int,
    window_size: int,
    last_segment_indices: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Build fixed-shape window/indexer coordinates for one CP prefill.

    Tail sources index unaugmented local rows; current/indexer rows index the
    augmented stream, including physical padding. Final-window sources use
    natural window coordinates; combined maps preserve ``[prev, next]`` order.
    Plans are reused across layers. Invalid rows use safe index ``0`` plus a
    boolean mask; ``-1`` is reserved for cache slot mappings.
    """
    window_size = int(window_size)
    segment_lens = [int(length) for length in segment_lens]
    batch_size = len(segment_lens)
    segment_num = 2 * int(cp_size)

    if isinstance(segment_valid_lens, torch.Tensor):
        valid_rows = segment_valid_lens.detach().cpu().tolist()
    else:
        valid_rows = segment_valid_lens
    valid_rows = [[int(value) for value in row] for row in valid_rows]
    if last_segment_indices is None:
        last_segments = []
        for row in valid_rows:
            seen = [idx for idx, value in enumerate(row) if value > 0]
            last_segments.append(seen[-1] if seen else 0)
    elif isinstance(last_segment_indices, torch.Tensor):
        raw_last = last_segment_indices.detach().cpu().tolist()
        last_segments = [int(value) % segment_num for value in raw_last]
    else:
        last_segments = [int(value) % segment_num for value in last_segment_indices]

    segment_offsets = [0]
    for segment_len in segment_lens:
        segment_offsets.append(segment_offsets[-1] + segment_len)

    def plan_tensor(values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=device)

    local_layouts = {}

    def build_flag_plan(flag: str) -> Dict[str, Any]:
        segment_indices = [
            int(cp_rank) if flag == "prev" else segment_num - int(cp_rank) - 1
        ] * batch_size
        current_valid = [valid_rows[b][segment_indices[b]] for b in range(batch_size)]
        previous_valid = [
            valid_rows[b][segment_indices[b] - 1] if segment_indices[b] > 0 else 0
            for b in range(batch_size)
        ]
        pre_window_lens = [0 if segment_idx == 0 else window_size for segment_idx in segment_indices]
        augmented_split_sizes = [
            pre_window_lens[b] + segment_lens[b] for b in range(batch_size)
        ]
        augmented_offsets = [0]
        current_row_offsets = [0]
        for b in range(batch_size):
            augmented_offsets.append(augmented_offsets[-1] + augmented_split_sizes[b])
            current_row_offsets.append(current_row_offsets[-1] + segment_lens[b])

        tail_source_indices = []
        tail_valid_mask = []
        for b in range(batch_size):
            take = min(window_size, current_valid[b])
            row_indices = [0] * window_size
            row_mask = [False] * window_size
            first_output = window_size - take
            for col in range(take):
                row_indices[first_output + col] = (
                    segment_offsets[b] + current_valid[b] - take + col
                )
                row_mask[first_output + col] = True
            tail_source_indices.append(row_indices)
            tail_valid_mask.append(row_mask)

        # Source rows index the local half; indexer/model rows index its
        # augmented form. Keep these coordinates separate.
        current_source_indices = []
        current_row_indices = []
        pre_window_source_indices = []
        pre_window_dest_indices = []
        for b in range(batch_size):
            source_start = segment_offsets[b]
            dest_start = augmented_offsets[b] + pre_window_lens[b]
            current_source_indices.extend(
                range(source_start, source_start + segment_lens[b])
            )
            current_row_indices.extend(
                range(dest_start, dest_start + segment_lens[b])
            )
            if pre_window_lens[b]:
                segment_idx = segment_indices[b]
                natural_window_start = (b * segment_num + segment_idx - 1) * window_size
                pre_window_source_indices.extend(
                    range(natural_window_start, natural_window_start + window_size)
                )
                pre_window_dest_indices.extend(range(augmented_offsets[b], dest_start))

        local_layouts[flag] = {
            "augmented_total_rows": augmented_offsets[-1],
            "current_row_indices": list(current_row_indices),
            "pre_window_source_indices": list(pre_window_source_indices),
            "pre_window_dest_indices": list(pre_window_dest_indices),
            "tail_source_indices": [list(row) for row in tail_source_indices],
            "tail_valid_mask": [list(row) for row in tail_valid_mask],
        }

        current_row_indices = plan_tensor(current_row_indices, torch.long)
        current_source_indices = plan_tensor(current_source_indices, torch.long)
        current_row_offsets = plan_tensor(current_row_offsets, torch.long)
        segment_offsets_tensor = plan_tensor(segment_offsets, torch.long)
        augmented_offsets_tensor = plan_tensor(augmented_offsets, torch.long)
        current_valid_tensor = plan_tensor(current_valid, torch.int32)
        previous_valid_tensor = plan_tensor(previous_valid, torch.int32)
        pre_window_tensor = plan_tensor(pre_window_lens, torch.int32)
        augmented_sizes_tensor = plan_tensor(augmented_split_sizes, torch.int32)
        pre_window_source_indices = plan_tensor(pre_window_source_indices, torch.long)
        pre_window_dest_indices = plan_tensor(pre_window_dest_indices, torch.long)
        tail_source_indices = plan_tensor(tail_source_indices, torch.long)
        tail_valid_mask = plan_tensor(tail_valid_mask, torch.bool)
        return {
            "source_coordinate": "local_segment_rows",
            "destination_coordinate": "augmented_local_rows",
            "segment_indices": plan_tensor(segment_indices, torch.long),
            "segment_offsets": segment_offsets_tensor,
            "current_valid_lens": current_valid_tensor,
            "previous_valid_lens": previous_valid_tensor,
            "pre_window_lens": pre_window_tensor,
            "augmented_split_sizes": augmented_sizes_tensor,
            "augmented_offsets": augmented_offsets_tensor,
            "augmented_total_rows": augmented_offsets[-1],
            "current_source_indices": current_source_indices,
            "current_row_indices": current_row_indices,
            "augmented_current_source_indices": current_source_indices,
            "augmented_current_dest_indices": current_row_indices,
            "indexer_source_indices": current_row_indices,
            "current_dest_indices": current_row_indices,
            "current_row_offsets": current_row_offsets,
            "local_half_tokens": sum(segment_lens),
            "pre_window_source_indices": pre_window_source_indices,
            "pre_window_dest_indices": pre_window_dest_indices,
            "tail_source_indices": tail_source_indices,
            "tail_valid_mask": tail_valid_mask,
        }

    plan = {"prev": build_flag_plan("prev"), "next": build_flag_plan("next")}
    local_half_tokens = sum(segment_lens)
    prev_layout = local_layouts["prev"]
    next_layout = local_layouts["next"]
    prev_augmented_rows = prev_layout["augmented_total_rows"]

    combined_tail_source = [source for row in prev_layout["tail_source_indices"] for source in row] + \
        [local_half_tokens + source for row in next_layout["tail_source_indices"] for source in row]
    combined_tail_mask = [valid for row in prev_layout["tail_valid_mask"] for valid in row] + \
        [valid for row in next_layout["tail_valid_mask"] for valid in row]
    combined_current_dest = prev_layout["current_row_indices"] + \
        [prev_augmented_rows + index for index in next_layout["current_row_indices"]]
    combined_pre_source = prev_layout["pre_window_source_indices"] + next_layout["pre_window_source_indices"]
    combined_pre_dest = prev_layout["pre_window_dest_indices"] + \
        [prev_augmented_rows + index for index in next_layout["pre_window_dest_indices"]]
    plan["combined"] = {
        "source_coordinate": "packed_local_rows",
        "destination_coordinate": "combined_augmented_local_rows",
        "tail_source_indices": plan_tensor(combined_tail_source, torch.long),
        "tail_padding_mask": plan_tensor(
            [not valid for valid in combined_tail_mask], torch.bool
        ),
        "tail_all_valid": all(combined_tail_mask),
        "augmented_current_dest_indices": plan_tensor(combined_current_dest, torch.long),
        "pre_window_source_indices": plan_tensor(combined_pre_source, torch.long),
        "pre_window_dest_indices": plan_tensor(combined_pre_dest, torch.long),
        "augmented_total_rows": (
            prev_augmented_rows + next_layout["augmented_total_rows"]
        ),
    }
    # Equal, fully valid segments use a tail view; variable/padded segments
    # use the explicit gather plan.
    uniform_segment_len = bool(segment_lens) and len(set(segment_lens)) == 1
    local_current_segments = {
        int(cp_rank),
        segment_num - int(cp_rank) - 1,
    }
    combined_current_all_valid = all(
        valid_rows[b][segment_idx] == segment_lens[b]
        for b in range(batch_size)
        for segment_idx in local_current_segments
    )
    combined_tail_contiguous = (
        uniform_segment_len
        and all(combined_tail_mask)
        and combined_current_all_valid
        and segment_lens[0] >= window_size
    )
    plan["combined"].update({
        "tail_contiguous": combined_tail_contiguous,
        "tail_segment_len": segment_lens[0] if uniform_segment_len else 0,
        "tail_current_all_valid": combined_current_all_valid,
    })
    last_source_indices = []
    last_valid_mask = []
    take_prev_list = []
    take_cur_list = []
    current_valid_list = []
    previous_valid_list = []
    for b, last_segment in enumerate(last_segments):
        current_valid = valid_rows[b][last_segment]
        take_cur = min(window_size, current_valid)
        previous_valid = valid_rows[b][last_segment - 1] if last_segment > 0 else 0
        take_prev = min(window_size - take_cur, previous_valid) if last_segment > 0 else 0
        row_indices = [0] * window_size
        row_mask = [False] * window_size
        output_col = window_size - take_prev - take_cur
        if take_prev:
            previous_row = (b * segment_num + last_segment - 1) * window_size
            for col in range(take_prev):
                row_indices[output_col + col] = previous_row + window_size - take_prev + col
                row_mask[output_col + col] = True
            output_col += take_prev
        if take_cur:
            current_row = (b * segment_num + last_segment) * window_size
            for col in range(take_cur):
                row_indices[output_col + col] = current_row + window_size - take_cur + col
                row_mask[output_col + col] = True
        last_source_indices.append(row_indices)
        last_valid_mask.append(row_mask)
        take_prev_list.append(take_prev)
        take_cur_list.append(take_cur)
        current_valid_list.append(current_valid)
        previous_valid_list.append(previous_valid)

    # Map natural windows to all_gather's rank-major
    # [rank0-prev-B, rank0-next-B, ...] layout.
    rank_major_segments = []
    for rank in range(cp_size):
        rank_major_segments.extend(
            b * segment_num + rank for b in range(batch_size)
        )
        rank_major_segments.extend(
            b * segment_num + segment_num - rank - 1
            for b in range(batch_size)
        )
    natural_to_rank_major = {
        natural_segment: rank_major_index
        for rank_major_index, natural_segment in enumerate(rank_major_segments)
    }

    def to_rank_major_rows(natural_rows):
        return [
            natural_to_rank_major[row // window_size] * window_size
            + row % window_size
            for row in natural_rows
        ]

    pre_window_value_sources = to_rank_major_rows(combined_pre_source)
    last_window_value_sources = to_rank_major_rows([
        source for row in last_source_indices for source in row
    ])
    plan["combined"].update({
        "window_value_source_indices": plan_tensor(
            pre_window_value_sources + last_window_value_sources, torch.long
        ),
        "pre_window_value_count": len(pre_window_value_sources),
    })
    last_window_all_valid = all(
        valid for row in last_valid_mask for valid in row
    )
    last_window_padding_mask = [
        [not valid for valid in row] for row in last_valid_mask
    ]
    last_source_indices = plan_tensor(last_source_indices, torch.long)
    last_valid_mask = plan_tensor(last_valid_mask, torch.bool)
    take_prev_tensor = plan_tensor(take_prev_list, torch.int32)
    take_cur_tensor = plan_tensor(take_cur_list, torch.int32)
    current_valid_tensor = plan_tensor(current_valid_list, torch.int32)
    previous_valid_tensor = plan_tensor(previous_valid_list, torch.int32)
    last_segment_tensor = plan_tensor(last_segments, torch.long)
    plan["last"] = {
        "source_coordinate": "natural_windows_flat",
        "source_shape": [batch_size * segment_num * window_size],
        "source_indices": last_source_indices,
        "valid_mask": last_valid_mask,
        "last_window_source_indices": last_source_indices,
        "last_window_valid_mask": last_valid_mask,
        "last_window_padding_mask": plan_tensor(
            last_window_padding_mask, torch.bool
        ),
        "last_window_all_valid": last_window_all_valid,
        "take_prev": take_prev_tensor,
        "take_cur": take_cur_tensor,
        "segment_indices": last_segment_tensor,
        "current_valid_lens": current_valid_tensor,
        "previous_valid_lens": previous_valid_tensor,
    }
    return plan


@dataclass
class DeepseekV4PrefillCPMetaData(PrefillCPMetaData):
    """DeepSeek-V4 extension of the executor's small public CP contract."""

    cp_rank: int = 0
    padded_seq_lens: Optional[List[int]] = None
    segment_lens: Optional[List[int]] = None
    split_list: Optional[List[int]] = None
    padded_starts: Optional[List[int]] = None
    compact_starts: Optional[List[int]] = None
    segment_offsets: Optional[torch.Tensor] = None
    segment_offsets_host: Optional[List[int]] = None
    segment_valid_lens: Optional[torch.Tensor] = None
    zigzag_indices: Optional[torch.Tensor] = None
    zigzag_idx: Optional[List[int]] = None
    last_segment_indices: Optional[torch.Tensor] = None
    last_cp_ranks: Optional[torch.Tensor] = None
    last_flags: Optional[List[str]] = None
    last_token_local_indices: Optional[torch.Tensor] = None
    last_token_valid_row_indices: Optional[torch.Tensor] = None
    last_token_slot_capacity: int = 0
    last_token_gather_rows: Optional[torch.Tensor] = None
    persistent_request_indices: Optional[torch.Tensor] = None
    window_plan: Optional[Dict[str, Any]] = None
    compressor_plans: Optional[Dict[str, Any]] = None
    state_write_plans: Optional[Dict[str, Any]] = None
    indexer_plan: Optional[Dict[str, Any]] = None

    @property
    def gather_to_natural(self) -> Optional[torch.Tensor]:
        """V4 name for the executor's public ``restore_indices`` contract."""
        return self.restore_indices

    @classmethod
    def from_forward_metadata(
        cls,
        forward_metadata,
        cp_rank: int,
        window_size: int,
    ) -> "DeepseekV4PrefillCPMetaData":
        """Derive V4 segment plans from public query lengths, reusing the
        executor's local/restore/valid maps.
        """
        public = getattr(forward_metadata, "cp_metadata", None)
        if public is None or not getattr(public, "enabled", False):
            raise ValueError("DeepSeek-V4 CP metadata requires enabled public CP metadata.")

        seq_lengths = getattr(forward_metadata, "actual_seq_lengths_kv", None)
        if seq_lengths is None:
            raise ValueError("actual_seq_lengths_kv is required for V4 CP planning.")
        seq_lengths = seq_lengths.view(-1)
        batch_size = int(seq_lengths.numel())
        cp_size = int(public.cp_size)
        segment_num = 2 * cp_size
        if cp_size <= 1:
            raise ValueError("DeepSeek-V4 private CP metadata requires cp_size > 1.")

        prev_cumulative = getattr(public, "actual_seq_q_prev", None)
        if prev_cumulative is None or prev_cumulative.numel() != batch_size:
            raise ValueError(
                "Public CP metadata must provide cumulative actual_seq_q_prev "
                "for V4 segment planning."
            )
        cumulative_lens = [
            int(value) for value in prev_cumulative.detach().cpu().tolist()
        ]
        segment_lens = [
            end - start
            for start, end in zip([0] + cumulative_lens[:-1], cumulative_lens)
        ]
        if any(length <= 0 for length in segment_lens):
            raise ValueError(f"Invalid CP segment lengths: {segment_lens}")

        seq_lens = seq_lengths.to(dtype=torch.int32)
        seq_lens_host = [int(value) for value in seq_lens.detach().cpu().tolist()]
        padded_seq_lens = [length * segment_num for length in segment_lens]
        padded_starts = [0]
        compact_starts = [0]
        for padded_len, seq_len in zip(padded_seq_lens, seq_lens_host):
            padded_starts.append(padded_starts[-1] + int(padded_len))
            compact_starts.append(compact_starts[-1] + int(seq_len))

        valid_lens = []
        last_segments = []
        for seq_len, segment_len in zip(seq_lens_host, segment_lens):
            valid_lens.append([
                min(max(int(seq_len) - segment * segment_len, 0), segment_len)
                for segment in range(segment_num)
            ])
            last_segments.append(
                min((int(seq_len) - 1) // segment_len, segment_num - 1)
                if int(seq_len) > 0 else 0
            )

        zigzag_idx = [
            request_idx * segment_num + int(cp_rank)
            for request_idx in range(batch_size)
        ] + [
            request_idx * segment_num + segment_num - int(cp_rank) - 1
            for request_idx in range(batch_size)
        ]
        zigzag_indices = torch.tensor(
            zigzag_idx,
            dtype=torch.long,
            device=seq_lengths.device,
        )

        last_cp_ranks = []
        last_flags = []
        last_token_local_indices = []
        local_half_tokens = sum(segment_lens)
        for request_idx, segment_idx in enumerate(last_segments):
            if segment_idx < cp_size:
                last_rank = segment_idx
                last_flag = "prev"
                half_offset = 0
            else:
                last_rank = segment_num - segment_idx - 1
                last_flag = "next"
                half_offset = local_half_tokens
            last_cp_ranks.append(last_rank)
            last_flags.append(last_flag)
            if last_rank == int(cp_rank) and valid_lens[request_idx][segment_idx] > 0:
                last_token_local_indices.append(
                    half_offset
                    + sum(segment_lens[:request_idx])
                    + valid_lens[request_idx][segment_idx]
                    - 1
                )
            else:
                last_token_local_indices.append(-1)

        last_token_counts = [0] * cp_size
        for last_rank in last_cp_ranks:
            last_token_counts[last_rank] += 1
        last_token_slot_capacity = max(last_token_counts, default=0)
        last_token_gather_rows = []
        last_token_slots = [0] * cp_size
        for last_rank in last_cp_ranks:
            last_token_gather_rows.append(
                last_rank * last_token_slot_capacity + last_token_slots[last_rank]
            )
            last_token_slots[last_rank] += 1

        device = seq_lengths.device
        valid_lens_tensor = torch.tensor(valid_lens, dtype=torch.int32, device=device)
        last_segment_indices = torch.tensor(
            [b * segment_num + segment for b, segment in enumerate(last_segments)],
            dtype=torch.long,
            device=device,
        )
        window_plan = build_prefill_cp_window_plan(
            segment_lens,
            valid_lens_tensor,
            cp_size,
            int(cp_rank),
            int(window_size),
            last_segment_indices=last_segment_indices,
            device=device,
        )
        indexer_plan = {
            flag: {
                "source_coordinate": "augmented_local_rows",
                "current_row_indices": flag_plan["current_row_indices"],
                "indexer_source_indices": flag_plan["indexer_source_indices"],
                "current_row_offsets": flag_plan["current_row_offsets"],
                "local_half_tokens": flag_plan["local_half_tokens"],
            }
            for flag, flag_plan in window_plan.items()
            if flag in ("prev", "next")
        }

        # Public Q/KV coordinates retain physical padding for other models;
        # clamp only V4's private valid-length view to real prompt lengths.
        public_kv_prev = public.kv_len_prev.to(device=device, dtype=seq_lens.dtype)
        public_kv_next = public.kv_len_next.to(device=device, dtype=seq_lens.dtype)
        kv_len_prev = torch.minimum(public_kv_prev, seq_lens)
        kv_len_next = torch.minimum(public_kv_next, seq_lens)

        owner_indices = getattr(public, "persistent_request_indices", None)
        if owner_indices is None:
            owner_indices = public.output_request_indices
        if owner_indices is not None:
            owner_indices = owner_indices.to(device=device, dtype=torch.long)

        return cls(
            enabled=public.enabled,
            cp_size=cp_size,
            global_padded_token_num=public.global_padded_token_num,
            local_token_num=public.local_token_num,
            local_indices=public.local_indices,
            restore_indices=public.restore_indices,
            global_valid_indices=public.global_valid_indices,
            global_slot_mapping=public.global_slot_mapping,
            global_block_table=public.global_block_table,
            persistent_valid_indices=public.persistent_valid_indices,
            persistent_slot_mapping=public.persistent_slot_mapping,
            output_request_indices=public.output_request_indices,
            actual_seq_q_prev=public.actual_seq_q_prev.to(dtype=torch.int32),
            actual_seq_q_next=public.actual_seq_q_next.to(dtype=torch.int32),
            kv_len_prev=kv_len_prev,
            kv_len_next=kv_len_next,
            local_prev_token_num=public.local_prev_token_num,
            local_next_token_num=public.local_next_token_num,
            cp_rank=int(cp_rank),
            padded_seq_lens=padded_seq_lens,
            segment_lens=segment_lens,
            split_list=[length for length in segment_lens for _ in range(segment_num)],
            padded_starts=padded_starts,
            compact_starts=compact_starts,
            segment_offsets=torch.tensor(
                [0] + cumulative_lens,
                dtype=torch.long,
                device=device,
            ),
            segment_offsets_host=[0] + cumulative_lens,
            segment_valid_lens=valid_lens_tensor,
            zigzag_indices=zigzag_indices,
            zigzag_idx=zigzag_idx,
            last_segment_indices=last_segment_indices,
            last_cp_ranks=torch.tensor(last_cp_ranks, dtype=torch.long, device=device),
            last_flags=last_flags,
            last_token_local_indices=torch.tensor(
                last_token_local_indices, dtype=torch.long, device=device
            ),
            last_token_valid_row_indices=torch.tensor(
                [idx for idx in last_token_local_indices if idx >= 0],
                dtype=torch.long,
                device=device,
            ),
            last_token_slot_capacity=last_token_slot_capacity,
            last_token_gather_rows=torch.tensor(
                last_token_gather_rows, dtype=torch.long, device=device
            ),
            persistent_request_indices=owner_indices,
            window_plan=window_plan,
            state_write_plans={},
            indexer_plan=indexer_plan,
        )


def prepare_prefill_cp_inputs(
    model_inputs: Dict[str, Any],
    infer_config,
    window_size: int,
) -> Dict[str, Any]:
    """Attach V4 private CP plans to public model inputs in place.

    Add private plans and normalize padding positions after executor padding.
    Mutate the original inputs because Engine/MTP reuse ``forward_metadata``.
    """
    forward_metadata = model_inputs.get("forward_metadata")
    if forward_metadata is None or not getattr(forward_metadata, "is_prefill", False):
        return model_inputs
    public = getattr(forward_metadata, "cp_metadata", None)
    if public is None or not getattr(public, "enabled", False):
        return model_inputs
    if isinstance(public, DeepseekV4PrefillCPMetaData):
        return model_inputs

    cp_size = int(public.cp_size)
    if cp_size <= 1:
        return model_inputs
    disagg_config = getattr(infer_config, "disagg_config", None)
    disagg_mode = getattr(disagg_config, "disaggregation_mode", "NONE")
    if disagg_mode != "NONE":
        raise ValueError(
            "DeepSeek-V4 private CP prefill planner only supports offline mode; "
            f"got disaggregation_mode={disagg_mode}."
        )

    parallel_config = getattr(infer_config, "parallel_config", None)
    cp_rank = int(getattr(parallel_config, "global_rank", 0)) % cp_size
    private = DeepseekV4PrefillCPMetaData.from_forward_metadata(
        forward_metadata,
        cp_rank=cp_rank,
        window_size=window_size,
    )
    forward_metadata.cp_metadata = private

    position_ids = model_inputs.get("position_ids")
    valid_indices = private.global_valid_indices
    if position_ids is not None and valid_indices is not None:
        original_shape = position_ids.shape
        flat_positions = position_ids.reshape(-1)
        valid_indices = valid_indices.to(device=flat_positions.device, dtype=torch.long)
        padded_positions = torch.ones_like(flat_positions)
        padded_positions.index_copy_(
            0,
            valid_indices,
            flat_positions.index_select(0, valid_indices),
        )
        model_inputs["position_ids"] = padded_positions.view(original_shape)
    model_inputs["forward_metadata"] = forward_metadata
    return model_inputs
