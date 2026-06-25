# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass

import torch
import torch.nn as nn

from executor.core.config import InferenceConfig
from executor.core.kv_cache.cache_info import MemoryBudgetItem, OffloadWorkspaceMemoryInfo


@dataclass(frozen=True)
class OffloadWorkspaceSpec:
    dtype: torch.dtype
    cache_last_dim: int
    num_hidden_layers: int
    batchseq: int
    selection_num_blocks: int
    block_size: int
    index_topk: int
    batch_len: int
    qk_rope_head_dim: int
    kv_cache_quant_mode: str

    @staticmethod
    def _dtype_itemsize(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def memory_info(self) -> OffloadWorkspaceMemoryInfo:
        dtype_size = self._dtype_itemsize(self.dtype)
        int32_size = self._dtype_itemsize(torch.int32)
        selected_blocks_per_token = (self.index_topk + self.block_size - 1) // self.block_size
        items = [
            MemoryBudgetItem(
                name="deepseek_v3_2_exp.offload.temp_nope",
                bytes=self.batch_len * self.block_size * self.cache_last_dim * dtype_size,
            ),
            MemoryBudgetItem(
                name="deepseek_v3_2_exp.offload.selected_nope",
                bytes=self.num_hidden_layers * self.selection_num_blocks
                * self.block_size * self.cache_last_dim * dtype_size,
            ),
            MemoryBudgetItem(
                name="deepseek_v3_2_exp.offload.selection_kv_block_table",
                bytes=self.num_hidden_layers * self.batchseq * selected_blocks_per_token * int32_size,
            ),
            MemoryBudgetItem(
                name="deepseek_v3_2_exp.offload.selection_kv_block_status",
                bytes=self.num_hidden_layers * self.batchseq * (self.index_topk + 1) * int32_size,
            ),
            MemoryBudgetItem(
                name="deepseek_v3_2_exp.offload.default_topk_indices",
                bytes=self.batchseq * self.index_topk * int32_size,
            ),
        ]
        if self.kv_cache_quant_mode != "int8":
            items.extend([
                MemoryBudgetItem(
                    name="deepseek_v3_2_exp.offload.temp_rope",
                    bytes=self.batch_len * self.block_size * self.qk_rope_head_dim * dtype_size,
                ),
                MemoryBudgetItem(
                    name="deepseek_v3_2_exp.offload.selected_rope",
                    bytes=self.num_hidden_layers * self.selection_num_blocks
                    * self.block_size * self.qk_rope_head_dim * dtype_size,
                ),
            ])
        return OffloadWorkspaceMemoryInfo(items=items)


class OffloadCache(nn.Module):
    def __init__(self, infer_config: InferenceConfig, model):
        super().__init__()
        self.infer_config = infer_config
        self.config = model.config
        self.spec = self.build_workspace_spec(infer_config, model)

        self.num_hidden_layers = self.spec.num_hidden_layers
        self.index_topk = self.spec.index_topk
        self.block_size = self.spec.block_size
        self.selection_num_blocks = self.spec.selection_num_blocks
        self.batch_len = self.spec.batch_len
        self.kv_cache_quant_mode = self.spec.kv_cache_quant_mode

        batchseq = self.spec.batchseq

        self.selection_kv_block_table = ()
        for _ in range(self.num_hidden_layers):
            self.selection_kv_block_table += (torch.arange(0, self.selection_num_blocks
                                                     ).reshape(batchseq, -1).to(device="npu", dtype=torch.int32),)
        self.selection_kv_block_status = ()
        for _ in range(self.num_hidden_layers):
            size = (batchseq, 1, self.index_topk + 1)
            self.selection_kv_block_status += (torch.full(size, -1).to(device="npu", dtype=torch.int32),)

        self.d2h_stream = torch.npu.Stream(device="npu")
        self.d2h_event = torch.npu.Event(blocking=True, enable_timing=False)

        self.default_topk_indices = torch.arange(self.index_topk, dtype=torch.int32, device="npu")\
                                    .view(1, -1).repeat(batchseq, 1)

        self.empty_rope = torch.tensor([], dtype=torch.int8, device="npu")

    @staticmethod
    def build_workspace_spec(infer_config: InferenceConfig, model) -> OffloadWorkspaceSpec:
        config = model.config
        kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode if config.quant_config is not None else "unquant"
        dtype = torch.int8 if kv_cache_quant_mode == "int8" else config.torch_dtype
        cache_last_dim = config.kv_lora_rank + config.qk_rope_head_dim * 2 + 4 * 4 \
            if kv_cache_quant_mode == "int8" else config.kv_lora_rank
        num_hidden_layers = config.num_nextn_predict_layers if model.is_mtp else config.num_hidden_layers
        batch_size_per_rank = infer_config.scheduler_config.batch_size_per_dp_rank
        index_topk = config.index_topk
        block_size = infer_config.scheduler_config.block_size
        s_maxblocknum = (index_topk + block_size - 1) // block_size
        batchseq = batch_size_per_rank * (1 + infer_config.model_config.next_n)
        selection_num_blocks = s_maxblocknum * batchseq
        pa_max_length = infer_config.model_config.custom_params.get(
            "pa_max_length",
            infer_config.data_config.input_truncated_len
            + infer_config.scheduler_config.max_new_tokens * (infer_config.model_config.next_n + 1)
            + infer_config.model_config.next_n,
        )
        cache_len = (pa_max_length + block_size - 1) // block_size
        # Keep the temp PA cache layout aligned with the framework KVCacheManager:
        # block 0 is reserved as the null block, and real request blocks start from 1.
        batch_len = cache_len * batch_size_per_rank + 1
        return OffloadWorkspaceSpec(
            dtype=dtype,
            cache_last_dim=cache_last_dim,
            num_hidden_layers=num_hidden_layers,
            batchseq=batchseq,
            selection_num_blocks=selection_num_blocks,
            block_size=block_size,
            index_topk=index_topk,
            batch_len=batch_len,
            qk_rope_head_dim=config.qk_rope_head_dim,
            kv_cache_quant_mode=kv_cache_quant_mode,
        )

    def init_workspace(
        self,
        cache_device,
    ):
        self.temp_kv_cache = None
        self.selected_key_values = ()

        # temp cache for prefill
        temp_nope = torch.zeros((
                    self.batch_len,
                    self.block_size,
                    1,
                    self.spec.cache_last_dim
                ), dtype=self.spec.dtype, device=cache_device)
        if self.kv_cache_quant_mode == "int8":
            temp_rope = torch.tensor([], dtype=torch.int8, device=cache_device)
        else:
            temp_rope = torch.zeros((
                        self.batch_len,
                        self.block_size,
                        1,
                        self.spec.qk_rope_head_dim
                    ), dtype=self.spec.dtype, device=cache_device)
        self.temp_kv_cache = (temp_nope, temp_rope,)

        for _ in range(self.num_hidden_layers):
            selected_nope = torch.zeros((self.selection_num_blocks, self.block_size, self.spec.cache_last_dim),
                                        dtype=self.spec.dtype, device=cache_device)
            if self.kv_cache_quant_mode == "int8":
                selected_rope = torch.tensor([], dtype=torch.int8, device=cache_device)
            else:
                selected_rope = torch.zeros((self.selection_num_blocks, self.block_size, self.spec.qk_rope_head_dim),
                                            dtype=self.spec.dtype, device=cache_device)
            self.selected_key_values += ((selected_nope, selected_rope),)

    def reinit_status(self):
        for i in range(self.num_hidden_layers):
            status = self.selection_kv_block_status[i]
            status.fill_(-1)
