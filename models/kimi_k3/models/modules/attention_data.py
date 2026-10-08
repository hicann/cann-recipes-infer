# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Model-local cache allocation and attention metadata for offline Kimi K3."""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist

from executor.utils.stream_utils import (
    create_event,
    create_stream,
    npu_stream_switch,
    record_event,
    record_stream,
    wait_event,
)

from ..configuration_kimi_k3 import mla_uses_mxfp8
from ..dspark_registry import DSPARK_DRAFT_MODEL_TYPES


_NULL_BLOCKS = 1
_BF16_CACHE_NZ_DIM = 16
# Kept for callers that imported the historical private constant.
_KV_CACHE_NZ_DIM = _BF16_CACHE_NZ_DIM


def build_paged_slot_mapping(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map int64 request positions to physical DSpark cache slots."""
    if positions.dtype != torch.int64 or block_table.dtype != torch.int64:
        raise TypeError("DSpark slot positions and block table must be int64")
    block_indices = positions.clamp_min(0) // block_size
    valid = (positions >= 0) & (block_indices < block_table.shape[1])
    safe_block_indices = block_indices.clamp_max(block_table.shape[1] - 1)
    block_ids = torch.gather(block_table, 1, safe_block_indices)
    offsets = positions.clamp_min(0) % block_size
    null_block_ids = torch.zeros_like(block_ids)
    return torch.where(valid, block_ids, null_block_ids) * block_size + torch.where(
        valid, offsets, torch.zeros_like(offsets)
    )


def build_dspark_mla_slot_mapping(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map DSpark MLA positions to unique slots for one fused cache write.

    Padding positions cannot all target the null slot because
    ``npu_kv_rmsnorm_rope_cache`` requires indices within one invocation to be
    unique. Each padding element therefore uses the next position in its own
    request row. The following draft write overwrites those scratch positions
    for active requests; GQA keeps its historical null-slot mapping above.
    """
    if positions.dtype != torch.int64 or block_table.dtype != torch.int64:
        raise TypeError("DSpark MLA positions and block table must be int64")
    in_range = (positions >= 0) & (
        positions // block_size < block_table.shape[1]
    )
    valid_positions = torch.where(
        in_range, positions, torch.full_like(positions, -1)
    )
    last_valid = valid_positions.max(dim=1, keepdim=True).values
    invalid_rank = (~in_range).to(torch.int64).cumsum(dim=1) - 1
    scratch_positions = last_valid + 1 + invalid_rank
    effective_positions = torch.where(in_range, positions, scratch_positions)
    block_indices = effective_positions // block_size
    safe_block_indices = block_indices.clamp_max(block_table.shape[1] - 1)
    block_ids = torch.gather(block_table, 1, safe_block_indices)
    offsets = effective_positions % block_size
    return block_ids * block_size + offsets


def gather_sp_shards_to_owner(
    local_hidden: torch.Tensor,
    input_len: int,
    owner_rank: int,
    tp_rank: int,
    tp_size: int,
    group=None,
):
    """Route one packed Prefill cycle's SP shards only to the selected DP owner."""
    if tp_size == 1:
        return local_hidden[:input_len].contiguous()
    if not dist.is_initialized():
        raise RuntimeError("DSpark Prefill hidden routing requires initialized distributed")
    flat = local_hidden.contiguous().view(-1)
    shard_elements = flat.numel()
    input_splits = [0] * tp_size
    input_splits[owner_rank] = shard_elements
    if tp_rank == owner_rank:
        output = flat.new_empty(shard_elements * tp_size)
        output_splits = [shard_elements] * tp_size
    else:
        output = flat.new_empty(0)
        output_splits = [0] * tp_size
    dist.all_to_all_single(
        output,
        flat,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    if tp_rank != owner_rank:
        return None
    return output.view(-1, local_hidden.shape[-1])[:input_len].contiguous()


def _settings_get(settings: dict, section: str, key: str, default=None):
    return settings.get(section, {}).get(key, default)


def validate_mega_kda_replayssm_switch(
    custom_params: dict,
    draft_model_type: str,
    next_n: int,
    error_type=ValueError,
) -> bool:
    """Validate the small switch contract shared by model and cache setup.

    Full platform, shape, and scheduler validation belongs to ``infer.py``.
    This helper only protects callers that construct the model/cache directly.
    """
    custom_params = custom_params or {}
    enabled = custom_params.get("enable_mega_kda_replayssm", False)
    if not isinstance(enabled, bool):
        raise error_type("enable_mega_kda_replayssm must be a boolean")
    if not enabled:
        return False
    enable_mega_kda = custom_params.get("enable_mega_kda", False)
    if not isinstance(enable_mega_kda, bool):
        raise error_type("enable_mega_kda must be a boolean")
    if not enable_mega_kda:
        raise error_type("enable_mega_kda_replayssm requires enable_mega_kda=True")
    if draft_model_type not in ("dspark", "dspark_gqa"):
        raise error_type("enable_mega_kda_replayssm is only supported with DSpark")
    if int(next_n) != 7:
        raise error_type("enable_mega_kda_replayssm requires next_n=7 (Verify width 8)")
    return True


def _mega_kda_replayssm_enabled(settings: dict) -> bool:
    """Return the validated mode used to select the replay cache layout."""
    model = settings.get("model_config", {})
    custom_params = model.get("custom_params") or {}
    return validate_mega_kda_replayssm_switch(
        custom_params,
        model.get("draft_model_type", "none"),
        model.get("next_n", 0),
    )


class CacheData:
    """Allocate the fixed offline cache owned by the model process.

    KDA keeps one state row per request on every attention-TP rank because each
    rank owns a head shard. MLA keeps only the requests owned by this rank; its
    latent cache is paged so decode can continue using the absorbed FA path.
    """

    def __init__(self, config, runner_settings: dict, device="npu"):
        self.config = config
        self.runner_settings = runner_settings
        self.device = device
        self.batch_size = _settings_get(
            runner_settings,
            "data_config",
            "batch_size_per_rank",
            _settings_get(runner_settings, "data_config", "batch_size", 1),
        )
        self.input_max_len = _settings_get(runner_settings, "data_config", "input_max_len", 128)
        self.max_new_tokens = _settings_get(runner_settings, "data_config", "max_new_tokens", 128)
        self.next_n = _settings_get(runner_settings, "model_config", "next_n", 0)
        self.draft_model_type = _settings_get(
            runner_settings, "model_config", "draft_model_type", "none"
        )
        self.verify_size = (
            self.next_n + 1
            if self.draft_model_type in DSPARK_DRAFT_MODEL_TYPES
            else 1
        )
        self.max_total_len = self.input_max_len + self.max_new_tokens + self.next_n
        self.block_size = _settings_get(runner_settings, "model_config", "pa_block_size", 128)
        self.prefill_chunk_size = _settings_get(
            runner_settings, "model_config", "prefill_chunk_size", 0
        )
        self.prefill_batch_size = _settings_get(
            runner_settings, "model_config", "prefill_mini_batch_size", 0
        ) or self.batch_size
        custom_params = _settings_get(
            runner_settings, "model_config", "custom_params", {}
        ) or {}
        self.use_mla_w8a8c8 = mla_uses_mxfp8(config)
        self.enable_mega_kda_replayssm = _mega_kda_replayssm_enabled(
            runner_settings
        )
        self.attn_tp_size = _settings_get(runner_settings, "parallel_config", "attn_tp_size", 1)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.attn_tp_rank = self.global_rank % self.attn_tp_size

        if self.batch_size % self.attn_tp_size:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"attn_tp_size={self.attn_tp_size}"
            )
        self.mla_batch_per_rank = self.batch_size // self.attn_tp_size
        self.blocks_per_request = math.ceil(self.max_total_len / self.block_size)

    def _new_zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def _init_kda_cache(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        linear = self.config.linear_attn_config
        local_heads = linear["num_heads"] // self.attn_tp_size
        head_dim = linear["head_dim"]
        projection_size = local_heads * head_dim
        kernel_size = linear["short_conv_kernel_size"]
        cache = {
            "attn_type": "KDA",
            "layer_idx": layer_idx,
            "conv_state": self._new_zeros(
                (
                    self.batch_size + _NULL_BLOCKS,
                    kernel_size - 1 + self.next_n,
                    3 * projection_size,
                ),
                torch.bfloat16,
            ),
            "recurrent_state": self._new_zeros(
                (
                    self.batch_size
                    if self.enable_mega_kda_replayssm
                    else _NULL_BLOCKS + self.batch_size * self.verify_size,
                    local_heads,
                    head_dim,
                    head_dim,
                ),
                torch.float32,
            ),
        }
        if self.enable_mega_kda_replayssm:
            replay_shape = (
                self.batch_size,
                self.verify_size,
                local_heads,
                head_dim,
            )
            cache.update(
                replay_u=self._new_zeros(replay_shape, torch.float16),
                replay_k=self._new_zeros(replay_shape, torch.float16),
                replay_decay=self._new_zeros(replay_shape, torch.float32),
            )
        return cache

    def _init_mla_cache(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        block_num = _NULL_BLOCKS + self.mla_batch_per_rank * self.blocks_per_request
        nope_dtype = torch.bfloat16
        nope_nz_dim = _BF16_CACHE_NZ_DIM
        if self.config.kv_lora_rank % nope_nz_dim:
            raise ValueError(
                "MLA latent cache width must be divisible by its PA_NZ inner "
                f"dimension: kv_lora_rank={self.config.kv_lora_rank}, "
                f"nz_dim={nope_nz_dim}"
            )
        if self.config.qk_rope_head_dim % _BF16_CACHE_NZ_DIM:
            raise ValueError(
                "MLA auxiliary-key cache width must be divisible by "
                f"{_BF16_CACHE_NZ_DIM}: "
                f"qk_rope_head_dim={self.config.qk_rope_head_dim}"
            )
        combined_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim
        cache = {
            "attn_type": "MLA",
            "layer_idx": layer_idx,
            "kv_cache": self._new_zeros(
                (block_num, self.block_size, 1, combined_dim),
                torch.float8_e4m3fn if self.use_mla_w8a8c8 else torch.bfloat16
            ),
            "kr_cache": torch.empty(
                (0, self.block_size, 1, self.config.qk_rope_head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            ),
        }
        if self.prefill_chunk_size > 0:
            prefill_blocks = _NULL_BLOCKS + self.prefill_batch_size * self.blocks_per_request
            prefill_nope_shape = (
                prefill_blocks, self.block_size, 1, self.config.kv_lora_rank
            )
            prefill_rope_shape = (
                prefill_blocks, self.block_size, 1, self.config.qk_rope_head_dim
            )
            cache["prefill_nope_cache"] = self._new_zeros(
                prefill_nope_shape,
                torch.bfloat16,
            )
            cache["prefill_rope_cache"] = self._new_zeros(
                prefill_rope_shape,
                torch.bfloat16,
            )
        return cache

    def init_cache_data(self) -> Tuple[Dict[str, torch.Tensor], ...]:
        cache_data = []
        for layer_idx in range(self.config.num_hidden_layers):
            if self.config.is_kda_layer(layer_idx):
                cache_data.append(self._init_kda_cache(layer_idx))
            else:
                cache_data.append(self._init_mla_cache(layer_idx))
        return tuple(cache_data)

    def init_dspark_cache_data(self, draft_config) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Allocate owner-local cache tensors for the selected DSpark model."""
        if self.draft_model_type == "dspark_mla":
            return self._init_dspark_mla_cache_data(draft_config)
        return self._init_dspark_gqa_cache_data(draft_config)

    def _init_dspark_gqa_cache_data(
        self, draft_config
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Allocate the historical SGLang/Qwen GQA cache layout."""
        block_num = _NULL_BLOCKS + self.mla_batch_per_rank * self.blocks_per_request
        caches = []
        for layer_idx in range(draft_config.num_hidden_layers):
            caches.append({
                "attn_type": "DSparkGQA",
                "layer_idx": layer_idx,
                "k_cache": self._new_zeros(
                    (
                        block_num,
                        self.block_size,
                        draft_config.num_key_value_heads,
                        draft_config.head_dim,
                    ),
                    torch.bfloat16,
                ),
                "v_cache": self._new_zeros(
                    (
                        block_num,
                        self.block_size,
                        draft_config.num_key_value_heads,
                        draft_config.head_dim,
                    ),
                    torch.bfloat16,
                ),
            })
        return tuple(caches)

    def _init_dspark_mla_cache_data(
        self, draft_config
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Allocate the merged logical PA tensor used by Flash MLA."""
        if draft_config.kv_lora_rank % _BF16_CACHE_NZ_DIM:
            raise ValueError(
                "DSpark MLA kv_lora_rank must be divisible by the BF16 NZ width"
            )
        if draft_config.qk_rope_head_dim % _BF16_CACHE_NZ_DIM:
            raise ValueError(
                "DSpark MLA qk_rope_head_dim must be divisible by the BF16 NZ width"
            )
        block_num = _NULL_BLOCKS + self.mla_batch_per_rank * self.blocks_per_request
        combined_dim = draft_config.kv_lora_rank + draft_config.qk_rope_head_dim
        caches = []
        for layer_idx in range(draft_config.num_hidden_layers):
            caches.append({
                "attn_type": "DSparkMLA",
                "layer_idx": layer_idx,
                "kv_cache": self._new_zeros(
                    (
                        block_num,
                        self.block_size,
                        1,
                        combined_dim,
                    ),
                    torch.bfloat16,
                ),
                # MLA Prolog uses the merged ckv+kr repository. Keep the
                # separate RoPE cache as an empty, shape-stable placeholder,
                # matching the main model's decode contract.
                "kr_cache": torch.empty(
                    (
                        0,
                        self.block_size,
                        1,
                        draft_config.qk_rope_head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            })
        return tuple(caches)

    @staticmethod
    def reset_cache(cache_data: Iterable[Dict[str, torch.Tensor]]) -> None:
        """Clear cache contents in-place while preserving graph-visible addresses."""
        for layer_cache in cache_data:
            for value in layer_cache.values():
                if isinstance(value, torch.Tensor):
                    value.zero_()


class AttnMetaData:
    """Build one forward step's model-local metadata dictionaries."""

    def __init__(self, config, runner_settings: dict, device="npu"):
        self.config = config
        self.runner_settings = runner_settings
        self.device = device
        self.batch_size = _settings_get(
            runner_settings,
            "data_config",
            "batch_size_per_rank",
            _settings_get(runner_settings, "data_config", "batch_size", 1),
        )
        self.input_max_len = _settings_get(runner_settings, "data_config", "input_max_len", 128)
        self.max_new_tokens = _settings_get(runner_settings, "data_config", "max_new_tokens", 128)
        self.next_n = _settings_get(runner_settings, "model_config", "next_n", 0)
        self.draft_model_type = _settings_get(
            runner_settings, "model_config", "draft_model_type", "none"
        )
        self.verify_size = (
            self.next_n + 1
            if self.draft_model_type in DSPARK_DRAFT_MODEL_TYPES
            else 1
        )
        self.max_total_len = self.input_max_len + self.max_new_tokens + self.next_n
        self.block_size = _settings_get(runner_settings, "model_config", "pa_block_size", 128)
        self.prefill_chunk_size = _settings_get(
            runner_settings, "model_config", "prefill_chunk_size", 0
        )
        self.prefill_batch_size = _settings_get(
            runner_settings, "model_config", "prefill_mini_batch_size", 0
        ) or self.batch_size
        self.attn_tp_size = _settings_get(runner_settings, "parallel_config", "attn_tp_size", 1)
        self.exe_mode = runner_settings.get("exe_mode", "eager")
        custom_params = _settings_get(
            runner_settings, "model_config", "custom_params", {}
        ) or {}
        self.use_mla_w8a8c8 = mla_uses_mxfp8(config)
        self.enable_mega_kda_replayssm = _mega_kda_replayssm_enabled(
            runner_settings
        )
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.enable_mega_kda = custom_params.get("enable_mega_kda", False)
        self.metadata_stream = (
            create_stream("mla_metadata", self.exe_mode)
            if self.enable_multi_streams and self.exe_mode != "ge_graph"
            else None
        )
        self.metadata_events = (
            tuple(create_event(self.exe_mode, True) for _ in range(2))
            if self.metadata_stream is not None
            else None
        )
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.attn_tp_rank = self.global_rank % self.attn_tp_size
        if self.batch_size % self.attn_tp_size:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"attn_tp_size={self.attn_tp_size}"
            )
        self.mla_batch_per_rank = self.batch_size // self.attn_tp_size
        self.blocks_per_request = math.ceil(self.max_total_len / self.block_size)
        self.attention_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=self.device)
        )
        self.flash_attention_mask = None
        # Flash MLA uses nonzero to mean masked.
        self.flash_attention_mask = self.attention_mask.to(torch.int8)
        self.kda_conv_block_table = torch.arange(
            1, self.batch_size + 1, dtype=torch.int32, device=self.device
        ).view(self.batch_size, 1)
        if self.enable_mega_kda_replayssm:
            # ReplaySSM keeps one committed checkpoint per request. Repeat the
            # request row across Verify positions so the common metadata shape
            # remains stable; the ReplaySSM operators do not consume this table.
            self.kda_recurrent_block_table = torch.arange(
                self.batch_size, dtype=torch.int32, device=self.device
            ).view(self.batch_size, 1).expand(-1, self.verify_size).clone()
        else:
            self.kda_recurrent_block_table = torch.arange(
                1,
                1 + self.batch_size * self.verify_size,
                dtype=torch.int32,
                device=self.device,
            ).view(self.batch_size, self.verify_size)
        self.kda_recurrent_state_indices = self.kda_recurrent_block_table.to(
            torch.int64
        )
        self.mla_block_table = self._build_mla_block_table(torch.int32)
        self.mla_slot_block_table = self._build_mla_block_table(torch.int64)
        self.prefill_mla_block_table = self._build_prefill_mla_block_table()
        self.decode_kda_conv_block_table = self.kda_conv_block_table.clone()
        self.decode_kda_recurrent_block_table = self.kda_recurrent_block_table.clone()
        self.decode_kda_recurrent_state_indices = (
            self.kda_recurrent_state_indices.clone()
        )
        self.decode_mla_block_table = self.mla_block_table.clone()
        self.decode_num_accepted_tokens = torch.ones(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.first_verify = torch.zeros((), dtype=torch.bool, device=self.device)
        self.replay_record_valid_mask = (
            torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            if self.enable_mega_kda_replayssm
            else None
        )

    def prepare_flash_mla_metadata(self, metadata: dict) -> None:
        """Build merged Flash MLA metadata inside the Decode execution path.

        In npugraph_ex this method is called from the compiled forward so the
        metadata stream and its record/wait events share the graph's
        capture/replay lifecycle.
        """
        if metadata.get("is_prefill", False):
            return
        mla_metadata = metadata.get("mla_decode_metadata")
        if mla_metadata is None or "flash_mla_metadata" in mla_metadata:
            return

        cache_seqlens = mla_metadata["actual_seq_lengths_kv"]
        cu_seqlens_q = mla_metadata["query_start_loc"]
        q_len = self.verify_size
        mask_mode = 0 if q_len == 1 else 3

        def build_metadata():
            quant_args = {}
            if self.use_mla_w8a8c8:
                op = torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache_metadata
                quant_args = {"quant_mode": 1}
            else:
                op = torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata
            return op(
                cache_seqlens,
                num_heads_q=self.config.num_attention_heads,
                num_heads_kv=1,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=mla_metadata["actual_seq_lengths_q"] if self.use_mla_w8a8c8 else None,
                max_seqlen_q=q_len,
                max_seqlen_kv=-1,
                head_dim_qk=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
                head_dim_v=self.config.kv_lora_rank,
                mask_mode=mask_mode,
                layout_q="TND",
                **quant_args,
            )

        if self.metadata_stream is None:
            flash_metadata = build_metadata()
        else:
            record_event(True, self.metadata_events, 0, self.exe_mode)
            with npu_stream_switch(
                True, self.metadata_stream, exe_mode=self.exe_mode
            ):
                wait_event(True, self.metadata_events, 0, self.exe_mode)
                flash_metadata = build_metadata()
                record_event(True, self.metadata_events, 1, self.exe_mode)
            record_stream(
                True,
                flash_metadata,
                torch.npu.current_stream(),
                self.exe_mode,
            )

        mla_metadata["flash_mla_metadata"] = flash_metadata
        mla_metadata["flash_mla_metadata_async"] = self.metadata_stream is not None

    def _build_mla_block_table(self, dtype: torch.dtype) -> torch.Tensor:
        block_ids = torch.arange(
            1,
            1 + self.mla_batch_per_rank * self.blocks_per_request,
            dtype=dtype,
            device=self.device,
        )
        return block_ids.view(self.mla_batch_per_rank, self.blocks_per_request)

    def _build_prefill_mla_block_table(self) -> torch.Tensor | None:
        if self.prefill_chunk_size <= 0:
            return None
        block_ids = torch.arange(
            1,
            1 + self.prefill_batch_size * self.blocks_per_request,
            dtype=torch.int32,
            device=self.device,
        )
        return block_ids.view(self.prefill_batch_size, self.blocks_per_request)

    def _slot_mapping(
        self,
        position_ids: torch.Tensor,
        cu_q: torch.Tensor,
        block_rows: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map packed positions to int64 indices in this rank's MLA cache."""
        slots = []
        start = 0
        if block_rows is None:
            block_rows = torch.arange(
                cu_q.shape[0], dtype=torch.long, device=self.device
            )
        if block_table is None:
            block_table = self.mla_block_table
        if block_rows.numel() != cu_q.numel():
            raise ValueError("block_rows must contain one row for every request")
        for packed_idx, end_tensor in enumerate(cu_q):
            end = int(end_tensor.item())
            request_positions = position_ids[start:end].to(torch.int64)
            block_indices = request_positions // self.block_size
            offsets = request_positions % self.block_size
            cache_row = int(block_rows[packed_idx].item())
            block_ids = block_table[cache_row].gather(0, block_indices)
            slots.append(block_ids * self.block_size + offsets)
            start = end
        if not slots:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.cat(slots)

    def _owner_slice(self):
        start = self.attn_tp_rank * self.mla_batch_per_rank
        return start, start + self.mla_batch_per_rank

    def _prefill_owner_metadata(
        self,
        input_lens: torch.Tensor,
        kv_len: torch.Tensor,
        request_indices: torch.Tensor,
    ):
        """Select this rank's MLA writes from one global-request mini batch."""
        owner_start, owner_end = self._owner_slice()
        owner_token_indices = []
        owner_lens = []
        owner_cache_rows = []
        packed_start = 0
        owner_starts = []
        for request_id_tensor, length_tensor, start_tensor in zip(
            request_indices, input_lens, kv_len
        ):
            request_id = int(request_id_tensor.item())
            length = int(length_tensor.item())
            if owner_start <= request_id < owner_end:
                owner_token_indices.append(
                    torch.arange(
                        packed_start,
                        packed_start + length,
                        dtype=torch.long,
                        device=self.device,
                    )
                )
                owner_lens.append(length)
                owner_starts.append(int(start_tensor.item()))
                owner_cache_rows.append(request_id - owner_start)
            packed_start += length

        if not owner_lens:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.int32, device=self.device),
            )

        owner_indices = torch.cat(owner_token_indices)
        owner_lens = torch.tensor(owner_lens, dtype=torch.int32, device=self.device)
        owner_cu = owner_lens.cumsum(0)
        owner_positions = torch.cat(
            [
                torch.arange(start, start + int(length.item()), dtype=torch.int32, device=self.device)
                for start, length in zip(owner_starts, owner_lens)
            ]
        )
        owner_cache_rows = torch.tensor(
            owner_cache_rows, dtype=torch.long, device=self.device
        )
        owner_slots = self._slot_mapping(
            owner_positions, owner_cu, block_rows=owner_cache_rows
        )
        return owner_indices, owner_slots

    def get_attn_metadata(
        self,
        input_ids: torch.Tensor,
        input_lens: torch.Tensor,
        kv_len: torch.Tensor | None,
        is_prefill: bool,
        request_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        first_verify: bool = False,
        active_mask: torch.Tensor | None = None,
    ) -> dict:
        if is_prefill:
            actual_q = input_lens
            chunked_prefill = kv_len is not None
            past_kv = torch.zeros_like(actual_q) if kv_len is None else kv_len
            if request_indices is None:
                request_indices = torch.arange(
                    actual_q.shape[0], dtype=torch.long, device=self.device
                )
            if request_indices.numel() != actual_q.numel():
                raise ValueError(
                    "request_indices must contain one entry for every Prefill request"
                )
            if request_indices.numel() and (
                int(request_indices.min().item()) < 0
                or int(request_indices.max().item()) >= self.batch_size
            ):
                raise ValueError(
                    f"Prefill request indices must be in [0, {self.batch_size})"
                )
            actual_kv = past_kv + actual_q
            cu_q = actual_q.cumsum(0)
            cu_kv = actual_kv.cumsum(0)
            position_ids = torch.cat(
                [
                    torch.arange(
                        int(start.item()),
                        int(start.item()) + int(length.item()),
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for start, length in zip(past_kv, actual_q)
                ]
            )
            segment_end_indices = torch.tensor(
                [end - 1 for end in cu_q.tolist()],
                dtype=torch.int64,
                device=self.device,
            )
            owner_indices, owner_slots = self._prefill_owner_metadata(
                actual_q, past_kv, request_indices
            )
            prefill_slots = None
            if chunked_prefill:
                prefill_slots = self._slot_mapping(
                    position_ids,
                    cu_q,
                    block_table=self.prefill_mla_block_table,
                )
            kda_conv_block_table = self.kda_conv_block_table.index_select(
                0, request_indices
            )
            kda_recurrent_block_table = self.kda_recurrent_block_table.index_select(
                0, request_indices
            )[:, :1]
            metadata = {
                "is_prefill": True,
                "is_chunked_prefill": chunked_prefill,
                "prefill_has_initial_state": (past_kv > 0).to(torch.int32),
                "attention_mask": self.attention_mask,
                "flash_attention_mask": self.flash_attention_mask,
                "position_ids": position_ids,
                "kv_len": actual_kv,
                "actual_seq_lengths_q": actual_q,
                "actual_seq_lengths_kv": actual_kv,
                "actual_seq_lengths_cu_q": cu_q,
                "actual_seq_lengths_cu_kv": cu_kv,
                "actual_seq_lengths_list_q": None,
                "actual_seq_lengths_list_kv": actual_kv.tolist(),
                "actual_seq_lengths_cu_list_q": cu_q.tolist(),
                "actual_seq_lengths_cu_list_kv": cu_kv.tolist(),
                "prompt_tokens": int(actual_q.sum().item()),
                "query_start_loc": torch.cat(
                    (cu_q.new_zeros(1), cu_q)
                ).to(torch.int32),
                "query_boundaries": [0, *cu_q.tolist()],
                "segment_end_indices": segment_end_indices,
                "block_table": {
                    "Mamba": kda_conv_block_table,
                    "KDAConv": kda_conv_block_table,
                    "KDARecurrent": kda_recurrent_block_table,
                    "FullAttention": self.mla_block_table,
                    "PrefillFullAttention": self.prefill_mla_block_table,
                },
                "slot_mapping": {
                    "FullAttention": owner_slots,
                    "PrefillFullAttention": prefill_slots,
                },
                "mla_owner_token_indices": owner_indices,
                "mla_decode_metadata": None,
            }
            return metadata

        if kv_len is None:
            raise ValueError("decode requires kv_len")
        q_len = self.verify_size
        actual_q = torch.full(
            (self.batch_size,), q_len, dtype=torch.int32, device=self.device
        )
        actual_kv = kv_len + q_len
        cu_q = actual_q.cumsum(0)
        cu_kv = actual_kv.cumsum(0)
        position_ids = (
            kv_len.unsqueeze(1)
            + torch.arange(q_len, dtype=torch.int32, device=self.device).unsqueeze(0)
        ).view(-1)

        owner_start, owner_end = self._owner_slice()
        local_q = actual_q[owner_start:owner_end]
        local_kv = actual_kv[owner_start:owner_end]
        local_cu_q = local_q.cumsum(0)
        local_cu_kv = local_kv.cumsum(0)
        local_positions = position_ids.view(self.batch_size, q_len)[
            owner_start:owner_end
        ].view(-1)
        local_slots = self._slot_mapping(local_positions, local_cu_q)
        if active_mask is None:
            active_mask = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )
        else:
            active_mask = active_mask
        local_active = active_mask[owner_start:owner_end]
        local_slots = torch.where(
            local_active.repeat_interleave(q_len), local_slots, torch.zeros_like(local_slots)
        )
        kda_conv_table = self.decode_kda_conv_block_table
        kda_recurrent_table = self.decode_kda_recurrent_block_table
        fused_kda_state_indices = self.decode_kda_recurrent_state_indices
        kda_conv_table.copy_(self.kda_conv_block_table)
        kda_recurrent_table.copy_(self.kda_recurrent_block_table)
        fused_kda_state_indices.copy_(self.kda_recurrent_state_indices)
        kda_conv_table[~active_mask] = 0
        kda_recurrent_table[~active_mask] = 0
        fused_kda_state_indices[~active_mask] = 0
        local_mla_block_table = self.decode_mla_block_table
        local_mla_block_table.copy_(self.mla_block_table)
        local_mla_block_table[~local_active] = 0
        recurrent_count = (
            self.decode_num_accepted_tokens
            if self.draft_model_type not in DSPARK_DRAFT_MODEL_TYPES
            else None
        )
        conv_count = None
        if self.draft_model_type in DSPARK_DRAFT_MODEL_TYPES:
            recurrent_count = (
                torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
                if num_accepted_tokens is None
                else num_accepted_tokens
            )
            # Keep the metadata alias pattern stable so the first and later
            # Verify rounds reuse the same compiled main-model graph.
            conv_count = recurrent_count.clone()
            if first_verify:
                conv_count = torch.full_like(recurrent_count, q_len)
        if self.enable_mega_kda:
            # The demo ABI takes int32 counts for both state caches, including
            # S=1 decode. First Verify retains conv=S and recurrent=1 so that
            # it reads the Prefill convolution tail and recurrent checkpoint.
            recurrent_count = recurrent_count.to(torch.int32)
            conv_count = (
                recurrent_count.clone() if conv_count is None else conv_count.to(torch.int32)
            )
        replay_count = recurrent_count
        if self.enable_mega_kda_replayssm:
            # accepted counts describe the preceding Verify record. A request
            # can become inactive after producing that record, so filtering by
            # the current active_mask would drop its final valid state update.
            replay_count = (
                torch.zeros_like(recurrent_count)
                if first_verify
                else torch.where(
                    self.replay_record_valid_mask,
                    recurrent_count,
                    torch.zeros_like(recurrent_count),
                )
            )
            self.replay_record_valid_mask.copy_(active_mask)
        ssm_state_indices = kda_recurrent_table[:, :q_len].reshape(-1).clone()
        fused_ssm_state_indices = (
            fused_kda_state_indices[:, :q_len].reshape(-1).clone()
        )
        self.first_verify.fill_(first_verify)
        mla_decode = {
            "is_prefill": False,
            "attention_mask": None if q_len == 1 else self.attention_mask,
            "flash_attention_mask": None
            if q_len == 1
            else self.flash_attention_mask,
            "position_ids": local_positions,
            "kv_len": kv_len[owner_start:owner_end],
            "actual_seq_lengths_q": local_q,
            "actual_seq_lengths_kv": local_kv,
            "actual_seq_lengths_cu_q": local_cu_q,
            "actual_seq_lengths_cu_kv": local_cu_kv,
            "actual_seq_lengths_list_q": local_q.tolist(),
            "actual_seq_lengths_list_kv": local_kv.tolist(),
            "actual_seq_lengths_cu_list_q": local_cu_q.tolist(),
            "actual_seq_lengths_cu_list_kv": local_cu_kv.tolist(),
            "prompt_tokens": 0,
            "query_start_loc": torch.cat(
                (local_cu_q.new_zeros(1), local_cu_q)
            ).to(torch.int32),
            "query_boundaries": [0, *local_cu_q.tolist()],
            "block_table": {
                "Mamba": kda_conv_table,
                "KDAConv": kda_conv_table,
                "KDARecurrent": kda_recurrent_table,
                "FullAttention": local_mla_block_table,
            },
            "slot_mapping": {"FullAttention": local_slots},
            "mla_owner_token_indices": None,
            "mla_decode_metadata": None,
            "oproj_output_rows": (
                self.mla_batch_per_rank * q_len * self.attn_tp_size
            ),
            "num_accepted_tokens": recurrent_count,
            "conv_num_accepted_tokens": conv_count,
            "ssm_state_indices": ssm_state_indices,
            "fused_ssm_state_indices": fused_ssm_state_indices,
            "first_verify": self.first_verify,
        }
        if self.enable_mega_kda_replayssm:
            mla_decode["replay_num_accepted_tokens"] = replay_count
        metadata = {
            "is_prefill": False,
            "attention_mask": None if q_len == 1 else self.attention_mask,
            "flash_attention_mask": None
            if q_len == 1
            else self.flash_attention_mask,
            "position_ids": position_ids,
            "kv_len": kv_len,
            "actual_seq_lengths_q": actual_q,
            "actual_seq_lengths_kv": actual_kv,
            "actual_seq_lengths_cu_q": cu_q,
            "actual_seq_lengths_cu_kv": cu_kv,
            "actual_seq_lengths_list_q": actual_q.tolist(),
            "actual_seq_lengths_list_kv": actual_kv.tolist(),
            "actual_seq_lengths_cu_list_q": cu_q.tolist(),
            "actual_seq_lengths_cu_list_kv": cu_kv.tolist(),
        "prompt_tokens": 0,
        "query_start_loc": torch.cat(
            (cu_q.new_zeros(1), cu_q)
        ).to(torch.int32),
            "query_boundaries": [0, *cu_q.tolist()],
            "block_table": {
                "Mamba": kda_conv_table,
                "KDAConv": kda_conv_table,
                "KDARecurrent": kda_recurrent_table,
                "FullAttention": local_mla_block_table,
            },
            "slot_mapping": {"FullAttention": local_slots},
            "mla_owner_token_indices": None,
            "mla_decode_metadata": mla_decode,
            "num_accepted_tokens": recurrent_count,
            "conv_num_accepted_tokens": conv_count,
            "ssm_state_indices": ssm_state_indices,
            "fused_ssm_state_indices": fused_ssm_state_indices,
            "first_verify": self.first_verify,
        }
        if self.enable_mega_kda_replayssm:
            metadata["replay_num_accepted_tokens"] = replay_count
        return metadata
