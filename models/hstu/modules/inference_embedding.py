# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/modules/inference_embedding.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import math
from typing import Dict, List
import torch
import torch_npu
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import JaggedTensor


class VocabParallelEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        padding_idx=None,
        params_dtype=torch.float32,
        tp_size=1,
        tp_rank=0,
    ):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.original_vocab_size = vocab_size
        self.input_size_per_partition = math.ceil(vocab_size / tp_size)
        super().__init__(
            num_embeddings=self.input_size_per_partition,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            dtype=params_dtype
        )
        self.vocab_start_index = self.tp_rank * self.input_size_per_partition
        self.vocab_end_index = self.vocab_start_index + self.input_size_per_partition

    def forward(self, input_id):
        return super().forward(input_id)
    

class InferenceEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding_configs: List,
        hccl_comm_dict: Dict = None, 
        embed_tp_size: int = 1,
        small_table_threshold: int = 1000
    ):
        super(InferenceEmbedding, self).__init__()
        self._embedding_layers = torch.nn.ModuleDict()
        self._feature_to_table_map: Dict[str, str] = {}
        self.embed_tp_size = embed_tp_size
        self.hccl_comm_dict = hccl_comm_dict
        self.small_table_threshold = small_table_threshold
        self._is_table_sharded: Dict[str, bool] = {}

        if self.embed_tp_size > 1 and self.hccl_comm_dict is not None:
            self.embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            self.tp_rank = dist.get_rank(self.embed_tp_group)
        else:
            self.embed_tp_group = None
            self.tp_rank = 0

        for config in embedding_configs:
            if config.table_name not in self._embedding_layers:
                should_shard = (self.embed_tp_size > 1) and (config.vocab_size >= self.small_table_threshold)
                self._is_table_sharded[config.table_name] = should_shard
                if should_shard:
                    self._embedding_layers[config.table_name] = VocabParallelEmbedding(
                        vocab_size=config.vocab_size,
                        hidden_size=config.dim,
                        tp_size=self.embed_tp_size,
                        tp_rank=self.tp_rank,
                        params_dtype=torch.float32,
                        # device=torch.device(f"npu:{torch.npu.current_device()}"),
                    )
                else:
                    self._embedding_layers[config.table_name] = torch.nn.Embedding(
                        num_embeddings=config.vocab_size,
                        embedding_dim=config.dim,
                        dtype=torch.float32,
                        device=torch.device(f"npu:{torch.npu.current_device()}"),
                    )

            for feature_name in config.feature_names:
                self._feature_to_table_map[feature_name] = config.table_name
    
    def to_empty(self, device: torch.device, recurse: bool = True):
        super().to_empty(device=device, recurse=recurse)

        @torch.no_grad()
        def truncated_normal_(tensor: torch.Tensor,
                             mean: float = 0.0,
                             std: float = 0.02,
                             lower: float = -2.0,
                             upper: float = 2.0) -> None:
            """
            Fill `tensor` with samples from N(mean, std^2) truncated to [lower, upper]
            where bounds are in standard-deviation units.
            """
            if tensor is None or tensor.numel() == 0:
                return

            # Standard normal CDF for the truncation bounds.
            def _phi(x: float) -> float:
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

            lower_cdf = _phi(lower)
            upper_cdf = _phi(upper)

            # Sample in FP32 for numerical stability, then cast back.
            tmp = torch.empty_like(tensor, dtype=torch.float32, device=tensor.device)
            tmp.uniform_(lower_cdf, upper_cdf)

            # Inverse CDF via erfinv: Φ^{-1}(u) = sqrt(2) * erfinv(2u - 1)
            tmp.mul_(2.0).sub_(1.0)
            tmp.erfinv_()
            tmp.mul_(math.sqrt(2.0))

            tmp.mul_(std).add_(mean)
            tensor.copy_(tmp.to(dtype=tensor.dtype))

        @torch.no_grad()
        def init_embedding_weights(m: torch.nn.Module) -> None:
            """
            Initialize embedding weights and handle padding row if applicable.
            """
            if isinstance(m, torch.nn.Embedding):
                truncated_normal_(m.weight, mean=0.0, std=0.02, lower=-2.0, upper=2.0)
                if m.padding_idx is not None and m.padding_idx >= 0:
                    m.weight[m.padding_idx].zero_()

        self.apply(init_embedding_weights)

    def _get_global_max_length(self, local_length, device):
        len_tensor = torch.tensor([local_length], dtype=torch.long, device=device)
        dist.all_reduce(len_tensor, op=dist.ReduceOp.MAX, group=self.embed_tp_group)
        return len_tensor.item()
    
    def forward(self, kjt):
        if kjt.device().type == 'cpu':
            current_divice = torch_npu.npu.current_device()
            kjt = kjt.to(device=current_divice)
        output_embeddings = {}
        kjt_dict = kjt.to_dict()

        for feature_name, jagged_tensor in kjt_dict.items():
            if feature_name not in self._feature_to_table_map:
                continue
        
            table_name = self._feature_to_table_map[feature_name]
            embedding_layer = self._embedding_layers[table_name]
            is_sharded = self._is_table_sharded.get(table_name, False)
            input_ids = jagged_tensor.values()

            if is_sharded:
                local_len = input_ids.shape[0]
                max_len = self._get_global_max_length(local_len, input_ids.device)
                if local_len < max_len:
                    padding_size = max_len - local_len
                    padded_input_ids = torch.nn.functional.pad(input_ids, (0, padding_size), value=0)
                else:
                    padded_input_ids = input_ids
                
                all_input_ids = torch.empty(
                    max_len * self.embed_tp_size,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                dist.all_gather_into_tensor(all_input_ids, padded_input_ids, group=self.embed_tp_group)
                vocab_size_per_rank = embedding_layer.input_size_per_partition
                
                new_input_ids = all_input_ids - embedding_layer.vocab_start_index
                mask = (new_input_ids >= 0) & (new_input_ids < vocab_size_per_rank)
                safe_input_ids = torch.where(mask, new_input_ids, torch.tensor(0, device=input_ids.device))
                
                tp_embeddings = embedding_layer(safe_input_ids)
                tp_embeddings = tp_embeddings * mask.unsqueeze(-1).to(tp_embeddings.dtype)
                embeddings_padded = torch.empty(
                    max_len,
                    tp_embeddings.shape[-1],
                    dtype=tp_embeddings.dtype,
                    device=tp_embeddings.device
                )
                
                dist.reduce_scatter_tensor(embeddings_padded, tp_embeddings, group=self.embed_tp_group)
                embeddings = embeddings_padded[:local_len]
            
            else:
                embeddings = embedding_layer(input_ids)
            
            output_embeddings[feature_name] = JaggedTensor(
                values=embeddings,
                lengths=jagged_tensor.lengths(),
                offsets=jagged_tensor.offsets(),
            )
        return output_embeddings
