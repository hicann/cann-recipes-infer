# coding=utf-8
# Adapted from
# https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/modules/paged_hstu_infer_layer.py
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
import os
import builtins
import torch
import torch_npu
import torch.nn.functional as F
from configs import InferenceHSTUConfig, KVCacheConfig
from modules.jagged_data import JaggedData

lib_fbgemm_npu_api_so_path = os.getenv('LIB_FBGEMM_NPU_API_SO_PATH')
torch.ops.load_library(lib_fbgemm_npu_api_so_path)


def fused_enabled(name: str) -> bool:
    return name in getattr(builtins, "ENABLED_FUSED_OPS", set())


def build_pos_tables(max_pos: int, page_size: int, device, dtype=torch.long):
    pos = torch.arange(max_pos, dtype=torch.long, device="cpu")
    pos2page = (pos // page_size).to(dtype=dtype).to(device, non_blocking=False)
    pos2entry = (pos % page_size).to(dtype=dtype).to(device, non_blocking=False)
    return pos2page, pos2entry
    

@torch.no_grad()
def append_kvcache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    seqlen_offsets: torch.Tensor,
    nnz: int,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_layout: int,
    pos2page: torch.Tensor,
    pos2entry: torch.Tensor,
) -> None:
    if kv_layout != 0:
        raise NotImplementedError("Only support kv_layout == 0 now")
    device = paged_k_cache.device
    if paged_v_cache.device != device:
        raise AssertionError("paged_k_cache and paged_v_cache must be on the same device")
    
    _, page_size, _, _ = paged_k_cache.shape
    nnz_true = min(int(nnz), batch_indices.numel(), positions.numel())
    if nnz_true <= 0:
        return
    
    def _as_device_long(x: torch.Tensor) -> torch.Tensor:
        # Long + same device is required for indexing ops below.
        if x.device != device:
            x = x.to(device=device, non_blocking=True)
        if x.dtype != torch.long:
            x = x.to(dtype=torch.long)
        return x

    def _as_device(x: torch.Tensor) -> torch.Tensor:
        if x.device != device:
            x = x.to(device=device, non_blocking=True)
        return x

    batch = _as_device_long(batch_indices[:nnz_true])
    pos = _as_device_long(positions[:nnz_true])
    seqlen_offsets = _as_device_long(seqlen_offsets)
    kv_indices = _as_device_long(kv_indices)
    kv_indptr = _as_device_long(kv_indptr)
    
    # Compute token indices in (append_key/append_value) to read from
    row_idx = torch.arange(nnz_true, device=device, dtype=torch.long)
    token_idx = row_idx + seqlen_offsets.index_select(0, batch)
    
    # get page + entry index inside page for each token position
    page_iter = pos2page.index_select(0, pos)
    entry_idx = pos2entry.index_select(0, pos)
    
    # Map (batch, page_iter) -> page_id via CSR-like indptr/indices
    base_ptr = kv_indptr.index_select(0, batch)
    page_ptr = base_ptr + page_iter
    page_id = kv_indices.index_select(0, page_ptr)

    append_key = _as_device(append_key)
    append_value = _as_device(append_value)

    src_k = append_key.index_select(0, token_idx)
    src_v = append_value.index_select(0, token_idx)

    # Kernel expects int32 slot ids
    slot_i32 = (page_id * page_size + entry_idx).to(torch.int32)

    torch_npu._npu_reshape_and_cache(src_k, src_v, paged_k_cache, paged_v_cache, slot_i32)


class PagedHSTUInferLayer(torch.nn.Module):
    """
    x = ln(x)
    u,v,q,k = silu(linear_bias(x))
    attn_output = hstu_attn.hstu_attn_varlen_func(q,k,v,offsets,max_seqlen)
    normed_out = ln_mul_dropout(attn_output)
    out = linear_residual(normed_out)

    One basic unit of PagedHSTUBlock. Input and output are all JaggedData.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
        kv_cache_config: KVCacheConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._embedding_dim: int = config.hidden_size
        # per head dim
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim
        self._num_heads: int = config.num_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0 / (self._attention_dim_per_head**0.5)
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]
        self._max_seqlen = kv_cache_config.max_seq_len
        self.page_size = kv_cache_config.page_size

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch.device(f"npu:{torch.npu.current_device()}")
        
        self.pos2page, self.pos2entry = build_pos_tables(
            self._max_seqlen,
            self.page_size,
            device=device,
            dtype=torch.long
            )

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
            * self._num_heads,
            bias=True,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_uvqk.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        
        with torch.no_grad():
            self._linear_uvqk.bias = torch.nn.Parameter(
                torch.empty_like(self._linear_uvqk.bias, dtype=torch.float32),
                requires_grad=False
            )
            self._linear_uvqk.bias.uniform_(-0.5, 0.5)
        self._linear_uvqk_weight = self._linear_uvqk.weight.contiguous()

        # input norm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        # output norm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

        for param in self._linear_proj.parameters():
            param.requires_grad = False
            param.copy_(torch.randn_like(param))
        self._linear_proj_weight = self._linear_proj.weight.T.contiguous()

        # output buffer
        max_num_tokens = kv_cache_config.max_batch_size * kv_cache_config.max_seq_len
        self.output_buffer_ = torch.empty(
            (max_num_tokens, config.hidden_size),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        self.uvqk_buffer_ = torch.empty(
            (
                max_num_tokens,
                (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
                * self._num_heads,
            ),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    @torch.inference_mode()
    def forward_naive(
        self,
        batch_size: int,
        num_tokens: int,
        layer_input: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        if fused_enabled("in_linear_silu") and hasattr(torch.ops.mxrec, "distance_in_linear_silu_forward"):
            (user, value, query, key, _) = torch.ops.mxrec.distance_in_linear_silu_forward(
                normed_input, self._linear_uvqk_weight, self._linear_uvqk.bias, self._split_arg_list
            )
        else:
            mixed_uvqk = F.silu(self._linear_uvqk(normed_input))
            (user, value, query, key) = torch.split(
                mixed_uvqk,
                self._split_arg_list,
                dim=-1,
            )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = (kv_cache_table[0], kv_cache_table[1])
        append_kvcache(
            key,
            value,
            kv_cache_metadata.batch_indices,
            kv_cache_metadata.position,
            jd.num_candidates_offsets,
            num_tokens,  # kv_cache_metadata.new_history_nnz
            paged_k_cache,
            paged_v_cache,
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            0,  # NHD layout
            self.pos2page,
            self.pos2entry,
        )

        kv_cache_metadata.onload_history_kv_events[self.layer_idx].wait(
            torch_npu.npu.current_stream()
        )
        jagged_attn_output = torch.ops.mxrec.hstu_paged(q=query,
            k=key,
            v=value,
            kv_cache=kv_cache_table,
            mask=None,
            attn_bias=None,
            mask_type=2,
            max_seq_len=self._max_seqlen,
            max_seq_len_k=jd.seqlen_k,
            silu_scale=1.0 / self._max_seqlen,
            seq_offset=jd.seqlen_offsets,
            seq_offset_k=kv_cache_metadata.total_history_offsets,
            seq_offset_t=jd.num_candidates_offsets,
            page_offsets=kv_cache_metadata.kv_indptr,
            page_ids=kv_cache_metadata.kv_indices,
            last_page_len=kv_cache_metadata.kv_last_page_len,
            num_target=jd.num_candidates,
            target_group_size=1)

        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )

        parallel_input = user * F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )
        layer_output = self._linear_proj(parallel_input)
        if self._residual:
            torch.add(layer_output, layer_input, out=layer_output)
        return layer_output

    @torch.inference_mode()
    def forward_input(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        input_tensor = input_buffer[:num_tokens, ...]
        normed_input = F.layer_norm(
            input_tensor,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        if fused_enabled("in_linear_silu") and hasattr(torch.ops.mxrec, "distance_in_linear_silu_forward"):
            (_, value, _, key, _) = torch.ops.mxrec.distance_in_linear_silu_forward(
                normed_input, self._linear_uvqk_weight, self._linear_uvqk.bias, self._split_arg_list
            )
        else:
            self.uvqk_buffer_[:num_tokens, ...] = F.silu(self._linear_uvqk(normed_input))
            (_, value, _, key) = torch.split(
                self.uvqk_buffer_[:num_tokens, ...],
                self._split_arg_list,
                dim=-1,
            )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = (kv_cache_table[0], kv_cache_table[1])
        append_kvcache(
            key,
            value,
            kv_cache_metadata.batch_indices,
            kv_cache_metadata.position,
            jd.num_candidates_offsets,
            num_tokens,  # kv_cache_metadata.new_history_nnz
            paged_k_cache,
            paged_v_cache,
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            0,  # NHD layout
            self.pos2page,
            self.pos2entry,
        )

        return self.uvqk_buffer_[:num_tokens, ...]

    @torch.inference_mode()
    def forward_output(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        (user, value, query, key) = torch.split(
            self.uvqk_buffer_[:num_tokens, ...],
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        jagged_attn_output = torch.ops.mxrec.hstu_paged(q=query,
            k=key,
            v=value,
            kv_cache=kv_cache_table,
            mask=None,
            attn_bias=None,
            mask_type=2,
            max_seq_len=self._max_seqlen,
            max_seq_len_k=jd.seqlen_k,
            silu_scale=1.0 / self._max_seqlen,
            seq_offset=jd.seqlen_offsets,
            seq_offset_k=kv_cache_metadata.total_history_offsets,
            seq_offset_t=jd.num_candidates_offsets,
            page_offsets=kv_cache_metadata.kv_indptr,
            page_ids=kv_cache_metadata.kv_indices,
            last_page_len=kv_cache_metadata.kv_last_page_len,
            num_target=jd.num_candidates,
            target_group_size=1)

        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )
        parallel_input = user * F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )
        layer_output = self._linear_proj(parallel_input)
        if self._residual:
            torch.add(layer_output, input_buffer[:num_tokens, ...], out=layer_output, )
        else:
            self.output_buffer_[:num_tokens, ...] = layer_output
        return self.output_buffer_[:num_tokens, ...]