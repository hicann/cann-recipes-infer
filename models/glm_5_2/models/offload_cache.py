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

import torch
import torch.nn as nn
import torch_npu

from .shared_indexer_offload import shared_indexer_offload_enabled


_DSA_DEFAULT_POOL_SIZE = 8192
_DSA_LARGE_POOL_SIZE = 16384
_DSA_SET_ASSOCIATIVITY = 16
_DSA_MIN_ID_RANGE = 131072
_DEFAULT_PA_MAX_LENGTH = 2048


def resolve_dsa_pool_size(model_config=None):
    model_config = model_config or {}
    return _DSA_LARGE_POOL_SIZE if bool(model_config.get("enlarge_pool_size", False)) else _DSA_DEFAULT_POOL_SIZE


def resolve_dsa_id_range(model_config):
    pa_max_length = int(model_config.get("pa_max_length", _DEFAULT_PA_MAX_LENGTH))
    if pa_max_length <= 0:
        raise ValueError(f"pa_max_length must be positive for DSA shared offload, got {pa_max_length}")
    return max(_DSA_MIN_ID_RANGE, pa_max_length)


def build_dsa_shared_group_specs(indexer_types, num_hidden_layers, is_mtp=False):
    if is_mtp:
        return (), tuple(-1 for _ in range(num_hidden_layers))

    if indexer_types is None:
        specs = tuple({
            "group_id": layer_idx,
            "owner_layer": layer_idx,
            "layers": (layer_idx,),
            "group_kind": 0,
        } for layer_idx in range(num_hidden_layers))
        return specs, tuple(range(num_hidden_layers))

    if len(indexer_types) < num_hidden_layers:
        raise ValueError(
            f"indexer_types length {len(indexer_types)} is shorter than num_hidden_layers {num_hidden_layers}"
        )

    specs = []
    layer_to_group = [-1 for _ in range(num_hidden_layers)]
    current_layers = []
    current_owner = None

    def close_group():
        if current_owner is None:
            return
        group_id = len(specs)
        specs.append({
            "group_id": group_id,
            "owner_layer": current_owner,
            "layers": tuple(current_layers),
            "group_kind": 0,
        })
        for layer in current_layers:
            layer_to_group[layer] = group_id

    for layer_idx, indexer_type in enumerate(indexer_types[:num_hidden_layers]):
        if indexer_type == "full":
            close_group()
            current_owner = layer_idx
            current_layers = [layer_idx]
        elif indexer_type == "shared":
            if current_owner is None:
                raise ValueError("DSA shared layer cannot appear before a full indexer layer")
            current_layers.append(layer_idx)
        else:
            raise ValueError(f"unsupported indexer type for DSA group metadata: {indexer_type}")
    close_group()
    return tuple(specs), tuple(layer_to_group)


class OffloadCache(nn.Module):
    def __init__(self, runner_settings, model):
        super().__init__()
        self.runner_settings = runner_settings
        self.config = model.config
        self.is_mtp = model.is_mtp

        self.num_hidden_layers = self.config.num_nextn_predict_layers if self.is_mtp else self.config.num_hidden_layers
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.index_topk = self.config.index_topk
        self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)

        # num of selected blocks per query token
        self.s_maxblocknum = (self.index_topk + self.block_size - 1) // self.block_size
        self.next_n = self.runner_settings.get("model_config").get("next_n", 0)
        # bsz*seq
        batchseq = self.batch_size_per_rank * (1 + self.next_n)
        # total num of selected blocks
        self.selection_num_blocks = self.s_maxblocknum * batchseq

        self.selection_kv_block_table = ()
        for _ in range(self.num_hidden_layers):
            self.selection_kv_block_table += (torch.arange(0, self.selection_num_blocks
                                                     ).reshape(batchseq, -1).to(device="npu", dtype=torch.int32),)
        self.selection_kv_block_status = ()
        for _ in range(self.num_hidden_layers):
            size = (self.batch_size_per_rank, 1 + self.next_n, 1, self.index_topk + 1) # bsnd
            self.selection_kv_block_status += (torch.full(size, -1).to(device="npu", dtype=torch.int32),)

        self.d2h_stream = torch.npu.Stream(device="npu")
        self.d2h_event = torch.npu.Event(blocking=True, enable_timing=False)

        model_config = self.runner_settings.get("model_config", {})
        self.pa_max_length = int(model_config.get("pa_max_length", _DEFAULT_PA_MAX_LENGTH))
        # num of blocks of full kv in each batch
        self.cache_len = self.pa_max_length // self.block_size
        self.kv_cache_num_block = self.cache_len * self.batch_size_per_rank

        self.prefill_mini_batch_size = runner_settings.get("model_config").get("prefill_mini_batch_size", 0)
        self.mini_batch = self.prefill_mini_batch_size \
            if self.prefill_mini_batch_size > 0 else self.batch_size_per_rank
        self.batch_len = self.cache_len * self.mini_batch

        self.default_topk_indices = torch.arange(self.index_topk, dtype=torch.int32, device="npu")\
                                    .view(1, -1).repeat(batchseq, 1)

        self.kv_cache_quant_mode = self.config.quant_config.kv_cache_quant_mode \
            if self.config.quant_config is not None else "unquant"
        self.empty_rope = torch.tensor([], dtype=torch.int8, device="npu")
        self.dsa_pool_size = resolve_dsa_pool_size(model_config)
        self.dsa_id_range = resolve_dsa_id_range(model_config)
        self.use_dsa_shared_pool = (
            shared_indexer_offload_enabled(model_config)
            and not self.is_mtp
            and self.kv_cache_quant_mode == "unquant"
        )
        if self.use_dsa_shared_pool and self.dsa_pool_size % self.block_size != 0:
            raise ValueError(
                f"DSA pool size {self.dsa_pool_size} must be divisible by PA block size {self.block_size}"
            )
        if self.use_dsa_shared_pool:
            self.dsa_shared_group_specs, self.dsa_layer_to_group = build_dsa_shared_group_specs(
                getattr(self.config, "indexer_types", None),
                self.num_hidden_layers,
                self.is_mtp,
            )
        else:
            self.dsa_shared_group_specs = ()
            self.dsa_layer_to_group = tuple(-1 for _ in range(self.num_hidden_layers))
        self.dsa_shared_pool_ids = ()
        self.dsa_shared_id_to_slot = ()
        self.dsa_shared_lru_counter = ()
        self.dsa_shared_pending_plans = ()
        self.dsa_pool_key_values = ()

        # MTP gather reuse: on MTP draft steps the IndexShare loop freezes the first decode step's
        # top-k (reuse_shared_topk). Because the frozen top-k selects a fixed set of historical
        # positions whose KV is immutable, the selection buffer gathered on that first step stays
        # exactly valid for the later draft steps, so the per-step re-gather is redundant.
        self.enable_mtp_gather_reuse = self.runner_settings.get("model_config")\
            .get("enable_mtp_gather_reuse", True)

    def init_cache(
        self,
        cache_device,
    ):
        dtype = torch.int8 if self.kv_cache_quant_mode == "int8" else self.config.torch_dtype

        past_key_values = ()
        self.temp_kv_cache = None
        self.selected_key_values = ()
        self.dsa_pool_key_values = ()
        self.past_key_values_unmapped = ()

        # When the kvcache INT8 quantization is enabled
        # nope_cache, rope_cache, and nope_scale need to be concatenated for SFA/MLAprolog kernel in INT8 dtype.
        # kv_lora_rank(INT8) + qk_rope_head_dim(BF16) * 2(->INT8) + kv_scale(FP32) * 4(->INT8)
        cache_last_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim * 2 + 4 * 4 \
            if self.kv_cache_quant_mode == "int8" else self.config.kv_lora_rank

        cache_nope_shape = (
                        self.kv_cache_num_block,
                        self.block_size,
                        1,
                        cache_last_dim
                    )

        cache_rope_shape = (
                        self.kv_cache_num_block,
                        self.block_size,
                        1,
                        self.config.qk_rope_head_dim
                    )

        # temp cache for prefill
        temp_nope = torch.zeros((
                    self.batch_len,
                    self.block_size,
                    1,
                    cache_last_dim
                ), dtype=dtype, device=cache_device)
        if self.kv_cache_quant_mode == "int8":
            temp_rope = torch.tensor([], dtype=torch.int8, device=cache_device)
        else:
            temp_rope = torch.zeros((
                        self.batch_len,
                        self.block_size,
                        1,
                        self.config.qk_rope_head_dim
                    ), dtype=dtype, device=cache_device)
        self.temp_kv_cache = (temp_nope, temp_rope,)

        for _ in range(self.num_hidden_layers):
            cache_nope = torch_npu.empty_with_swapped_memory(cache_nope_shape, dtype=dtype, device=cache_device)
            if self.kv_cache_quant_mode == "int8":
                cache_rope = None
            else:
                cache_rope = torch_npu.empty_with_swapped_memory(cache_rope_shape, dtype=dtype, device=cache_device)
            past_key_values += ((cache_nope, cache_rope),)

            selected_nope = torch.zeros((self.selection_num_blocks, self.block_size, cache_last_dim),
                                        dtype=dtype, device=cache_device)
            if self.kv_cache_quant_mode == "int8":
                selected_rope = torch.tensor([], dtype=torch.int8, device=cache_device)
            else:
                selected_rope = torch.zeros((self.selection_num_blocks, self.block_size, self.config.qk_rope_head_dim),
                                            dtype=dtype, device=cache_device)
            self.selected_key_values += ((selected_nope, selected_rope),)
            if self.use_dsa_shared_pool:
                dsa_pool_blocks = self.batch_size_per_rank * self.dsa_pool_size // self.block_size
                dsa_pool_nope = torch.zeros(
                    (dsa_pool_blocks, self.block_size, cache_last_dim),
                    dtype=dtype,
                    device=cache_device,
                )
                dsa_pool_rope = torch.zeros(
                    (dsa_pool_blocks, self.block_size, self.config.qk_rope_head_dim),
                    dtype=dtype,
                    device=cache_device,
                )
                self.dsa_pool_key_values += ((dsa_pool_nope, dsa_pool_rope),)
            else:
                self.dsa_pool_key_values += ((None, None),)

        self._init_dsa_shared_metadata(cache_device)
        return past_key_values

    def _init_dsa_shared_metadata(self, cache_device):
        self.dsa_shared_pool_ids = ()
        self.dsa_shared_id_to_slot = ()
        self.dsa_shared_lru_counter = ()
        self.dsa_shared_pending_plans = ()
        for _ in self.dsa_shared_group_specs:
            self.dsa_shared_pool_ids += (torch.full(
                (self.batch_size_per_rank, self.dsa_pool_size),
                -1,
                dtype=torch.int32,
                device=cache_device,
            ),)
            self.dsa_shared_id_to_slot += (torch.full(
                (self.batch_size_per_rank, self.dsa_id_range),
                -1,
                dtype=torch.int32,
                device=cache_device,
            ),)
            self.dsa_shared_pending_plans += (None,)
            self.dsa_shared_lru_counter += (torch.zeros(
                (self.batch_size_per_rank, self.dsa_pool_size // _DSA_SET_ASSOCIATIVITY),
                dtype=torch.int32,
                device=cache_device,
            ),)

    def get_dsa_shared_group_id(self, layer_idx):
        if layer_idx < 0 or layer_idx >= len(self.dsa_layer_to_group):
            raise IndexError(f"layer_idx {layer_idx} is outside DSA group metadata range")
        group_id = self.dsa_layer_to_group[layer_idx]
        if group_id < 0:
            raise ValueError(f"layer_idx {layer_idx} has no DSA shared group")
        return group_id

    def get_dsa_shared_group_metadata(self, layer_idx):
        group_id = self.get_dsa_shared_group_id(layer_idx)
        return (
            group_id,
            self.dsa_shared_pool_ids[group_id],
            self.dsa_shared_id_to_slot[group_id],
            self.dsa_shared_lru_counter[group_id],
        )

    def get_dsa_layer_pool_values(self, layer_idx):
        if not self.use_dsa_shared_pool:
            raise ValueError("DSA shared resident pool is not initialized for this offload cache")
        if layer_idx < 0 or layer_idx >= len(self.dsa_pool_key_values):
            raise IndexError(f"layer_idx {layer_idx} is outside DSA pool range")
        pool = self.dsa_pool_key_values[layer_idx]
        if pool[0] is None or pool[1] is None:
            raise ValueError(f"layer_idx {layer_idx} has no DSA resident pool")
        return pool

    def update_dsa_layer_pool_values(self, layer_idx, pool_nope, pool_rope):
        if layer_idx < 0 or layer_idx >= len(self.dsa_pool_key_values):
            raise IndexError(f"layer_idx {layer_idx} is outside DSA pool range")
        pools = list(self.dsa_pool_key_values)
        pools[layer_idx] = (pool_nope, pool_rope)
        self.dsa_pool_key_values = tuple(pools)

    def get_dsa_shared_group_spec(self, group_id):
        if group_id < 0 or group_id >= len(self.dsa_shared_group_specs):
            raise IndexError(f"group_id {group_id} is outside DSA group metadata range")
        return self.dsa_shared_group_specs[group_id]

    def is_dsa_shared_group_owner(self, layer_idx):
        group_id = self.get_dsa_shared_group_id(layer_idx)
        return int(self.dsa_shared_group_specs[group_id]["owner_layer"]) == int(layer_idx)

    def is_dsa_shared_group_last_layer(self, layer_idx):
        group_id = self.get_dsa_shared_group_id(layer_idx)
        layers = self.dsa_shared_group_specs[group_id]["layers"]
        return int(layers[-1]) == int(layer_idx)

    def get_dsa_shared_group_plan(self, group_id):
        if group_id < 0 or group_id >= len(self.dsa_shared_pending_plans):
            raise IndexError(f"group_id {group_id} is outside DSA pending-plan range")
        plan_state = self.dsa_shared_pending_plans[group_id]
        if plan_state is None:
            raise ValueError(f"DSA shared group {group_id} has no pending plan")
        return plan_state

    def set_dsa_shared_group_plan(
        self,
        group_id,
        plan,
        install_records,
        selection_kv_actual_seq,
    ):
        if group_id < 0 or group_id >= len(self.dsa_shared_pending_plans):
            raise IndexError(f"group_id {group_id} is outside DSA pending-plan range")
        pending = list(self.dsa_shared_pending_plans)
        pending[group_id] = (
            plan,
            install_records,
            selection_kv_actual_seq,
        )
        self.dsa_shared_pending_plans = tuple(pending)

    def clear_dsa_shared_group_plan(self, group_id):
        if group_id < 0 or group_id >= len(self.dsa_shared_pending_plans):
            raise IndexError(f"group_id {group_id} is outside DSA pending-plan range")
        pending = list(self.dsa_shared_pending_plans)
        pending[group_id] = None
        self.dsa_shared_pending_plans = tuple(pending)

    def update_dsa_shared_group_metadata(self, group_id, pool_ids_next, id_to_slot_next, lru_counter_next):
        if group_id < 0 or group_id >= len(self.dsa_shared_group_specs):
            raise IndexError(f"group_id {group_id} is outside DSA group metadata range")
        pool_ids = list(self.dsa_shared_pool_ids)
        id_to_slot = list(self.dsa_shared_id_to_slot)
        lru_counter = list(self.dsa_shared_lru_counter)
        pool_ids[group_id] = pool_ids_next
        id_to_slot[group_id] = id_to_slot_next
        lru_counter[group_id] = lru_counter_next
        self.dsa_shared_pool_ids = tuple(pool_ids)
        self.dsa_shared_id_to_slot = tuple(id_to_slot)
        self.dsa_shared_lru_counter = tuple(lru_counter)

    def update_selected_key_values(self, layer_idx, selected_nope, selected_rope):
        if layer_idx < 0 or layer_idx >= len(self.selected_key_values):
            raise IndexError(f"layer_idx {layer_idx} is outside selected KV range")
        selected = list(self.selected_key_values)
        selected[layer_idx] = (selected_nope, selected_rope)
        self.selected_key_values = tuple(selected)

    def reinit_status(self):
        for i in range(self.num_hidden_layers):
            status = self.selection_kv_block_status[i]
            status.fill_(-1)
        for i in range(len(self.dsa_shared_group_specs)):
            self.dsa_shared_pool_ids[i].fill_(-1)
            self.dsa_shared_id_to_slot[i].fill_(-1)
            self.dsa_shared_lru_counter[i].zero_()
        if self.use_dsa_shared_pool:
            for dsa_pool_nope, dsa_pool_rope in self.dsa_pool_key_values:
                dsa_pool_nope.zero_()
                dsa_pool_rope.zero_()
        self.dsa_shared_pending_plans = tuple(None for _ in self.dsa_shared_group_specs)
