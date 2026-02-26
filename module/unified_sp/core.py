# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026.
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

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class SPConfig:
    ulysses_degree: int = 1
    ring_degree: int = 1
    use_ring_overlap: bool = True
    
    @property
    def sp_degree(self):
        return self.ulysses_degree * self.ring_degree


@dataclass
class ForwardParams:
    """Encapsulates forward method parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = False
    softmax_scale: Optional[float] = None
    seq_lens: Optional[int] = None
    per_head_compute: bool = False
    joint_tensor_query: Optional[torch.Tensor] = None
    joint_tensor_key: Optional[torch.Tensor] = None
    joint_tensor_value: Optional[torch.Tensor] = None
    joint_strategy: str = "none"


@dataclass
class AttentionParams:
    """Encapsulates attention computation parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = False
    softmax_scale: Optional[float] = None
    per_head_compute: bool = False
    joint_tensor_key: Optional[torch.Tensor] = None
    joint_tensor_value: Optional[torch.Tensor] = None


@dataclass
class NPUAttentionConfig:
    """NPU attention API parameters configuration."""
    num_heads: int
    scale: float
    input_layout: str = "BNSD"
    num_key_value_heads: int = 0
    pre_tokens: int = 65535
    next_tokens: int = 65535
    sparse_mode: int = 0
    inner_precise: int = 0
    softmax_lse_flag: bool = False


class UnifiedSPAttention:
    
    def __init__(
        self,
        ulysses_group: dist.ProcessGroup,
        ring_group: dist.ProcessGroup,
        use_ring_overlap: bool = True,
    ):
        self.ulysses_pg = ulysses_group
        self.ring_pg = ring_group
        self.use_ring_overlap = use_ring_overlap
        
        self.ulysses_world_size = dist.get_world_size(ulysses_group) if ulysses_group else 1
        self.ring_world_size = dist.get_world_size(ring_group) if ring_group else 1
        
        self.ulysses_rank = dist.get_rank(ulysses_group) if self.ulysses_world_size > 1 else 0
        self.ring_rank = dist.get_rank(ring_group) if self.ring_world_size > 1 else 0
        
        self._other_indices_cache = None
        self._cached_rank = None
        self._cached_ring_size = None
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        seq_lens: Optional[int] = None,
        per_head_compute: bool = False,
        joint_tensor_query: Optional[torch.Tensor] = None,
        joint_tensor_key: Optional[torch.Tensor] = None,
        joint_tensor_value: Optional[torch.Tensor] = None,
        joint_strategy: str = "none",
        **kwargs
    ) -> torch.Tensor:
        """Forward method with original interface for backward compatibility."""
        params = ForwardParams(
            q=q, k=k, v=v,
            causal=causal,
            softmax_scale=softmax_scale,
            seq_lens=seq_lens,
            per_head_compute=per_head_compute,
            joint_tensor_query=joint_tensor_query,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy
        )
        return self._forward_impl(params, **kwargs)
    
    def _forward_impl(
        self,
        params: ForwardParams,
        **kwargs
    ) -> torch.Tensor:
        """Internal forward implementation using ForwardParams."""
        q, k, v = params.q, params.k, params.v
        
        if params.joint_strategy == "rear" and params.joint_tensor_query is not None:
            q = torch.cat([q, params.joint_tensor_query], dim=1)
        elif params.joint_strategy == "front" and params.joint_tensor_query is not None:
            q = torch.cat([params.joint_tensor_query, q], dim=1)
        
        if params.seq_lens is not None and params.seq_lens < q.shape[1]:
            q_main, q_pad = q[:, :params.seq_lens], q[:, params.seq_lens:]
            k_main, k_pad = k[:, :params.seq_lens], k[:, params.seq_lens:]
            v_main, v_pad = v[:, :params.seq_lens], v[:, params.seq_lens:]
            
            attn_params_main = AttentionParams(
                q=q_main, k=k_main, v=v_main,
                causal=params.causal, 
                softmax_scale=params.softmax_scale,
                per_head_compute=params.per_head_compute
            )
            attn_params_pad = AttentionParams(
                q=q_pad, k=k_pad, v=v_pad,
                causal=params.causal, 
                softmax_scale=params.softmax_scale,
                per_head_compute=params.per_head_compute
            )
            
            out_main = self._forward_core(attn_params_main, **kwargs)
            out_pad = self._forward_core(attn_params_pad, **kwargs)
            
            return torch.cat([out_main, out_pad], dim=1)
        
        attn_params = AttentionParams(
            q=q, k=k, v=v,
            causal=params.causal, 
            softmax_scale=params.softmax_scale,
            per_head_compute=params.per_head_compute,
            joint_tensor_key=params.joint_tensor_key,
            joint_tensor_value=params.joint_tensor_value,
        )
        return self._forward_core(attn_params, **kwargs)
    
    def _forward_core(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        
        q, k, v = params.q, params.k, params.v
        original_seq_len = None
        joint_tensor_key = params.joint_tensor_key
        joint_tensor_value = params.joint_tensor_value
        
        if self.ulysses_world_size > 1:
            # Check head divisibility
            num_heads = q.shape[2]
            if num_heads % self.ulysses_world_size != 0:
                raise ValueError(
                    f"Number of heads ({num_heads}) must be divisible by "
                    f"ulysses_world_size ({self.ulysses_world_size}). "
                    f"Please adjust your model configuration or Ulysses degree."
                )
            
            original_seq_len = q.shape[1]
            
            q = self._all_to_all_head_to_seq(q)
            k = self._all_to_all_head_to_seq(k)
            v = self._all_to_all_head_to_seq(v)

            if params.joint_tensor_key is not None:
                ulysses_rank = dist.get_rank(self.ulysses_pg)
                attn_heads_per_ulysses_rank = (
                    params.joint_tensor_key.shape[-2] // self.ulysses_world_size
                )
                start_idx = attn_heads_per_ulysses_rank * ulysses_rank
                end_idx = start_idx + attn_heads_per_ulysses_rank
                joint_tensor_key = joint_tensor_key[..., start_idx:end_idx, :]
                joint_tensor_value = joint_tensor_value[..., start_idx:end_idx, :]

        if self.ring_world_size == 1:
            k = torch.cat([k, joint_tensor_key], dim=1)
            v = torch.cat([v, joint_tensor_value], dim=1)

        # Create updated params with transformed q, k, v
        updated_params = AttentionParams(
            q=q, k=k, v=v,
            causal=params.causal,
            softmax_scale=params.softmax_scale,
            per_head_compute=params.per_head_compute,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value
        )
        
        if self.ring_world_size > 1:
            if self.use_ring_overlap:
                out = self._ring_attention_overlap(updated_params, **kwargs)
            else:
                out = self._ring_attention_allgather(updated_params, **kwargs)
        else:
            out = self._compute_attention(updated_params)
        
        if self.ulysses_world_size > 1:
            out = self._all_to_all_seq_to_head(out)
        
        return out
    
    def _get_other_indices(self, rank, ring_size, device):
        rank_changed = self._cached_rank != rank
        ring_size_changed = self._cached_ring_size != ring_size
        cache_missing = self._other_indices_cache is None
        device_changed = (self._other_indices_cache is not None and 
                        self._other_indices_cache.device != device)
        
        needs_update = (rank_changed or ring_size_changed or 
                        cache_missing or device_changed)
        
        if needs_update:
            indices_list = [i for i in range(ring_size) if i != rank]
            self._other_indices_cache = torch.tensor(
                indices_list, dtype=torch.long, device=device
            )
            self._cached_rank = rank
            self._cached_ring_size = ring_size
        
        return self._other_indices_cache
    
    def _all_to_all_head_to_seq(self, input_: torch.Tensor) -> torch.Tensor:
        if self.ulysses_world_size == 1:
            return input_
        
        bs, shard_s, hc, d = input_.shape
        world_size = self.ulysses_world_size
        
        s = shard_s * world_size
        shard_hc = hc // world_size
        
        input_t = input_.reshape(bs, shard_s, world_size, shard_hc, d)
        input_t = input_t.transpose(0, 2).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
        
        output = output.reshape(s, bs, shard_hc, d)
        output = output.transpose(0, 1).contiguous()
        output = output.reshape(bs, s, shard_hc, d)
        
        return output
    
    def _all_to_all_seq_to_head(self, input_: torch.Tensor) -> torch.Tensor:
        if self.ulysses_world_size == 1:
            return input_
        
        bs, s, shard_hc, d = input_.shape
        world_size = self.ulysses_world_size
        
        hc = shard_hc * world_size
        shard_s = s // world_size
        
        input_t = input_.reshape(bs, world_size, shard_s, shard_hc, d)
        input_t = input_t.transpose(0, 3).transpose(0, 1).contiguous()
        input_t = input_t.reshape(world_size, shard_hc, shard_s, bs, d)
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
        
        output = output.reshape(hc, shard_s, bs, d)
        output = output.transpose(0, 2).contiguous()
        output = output.reshape(bs, shard_s, hc, d)
        
        return output
    
    def _ring_attention_overlap(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        
        q, k, v = params.q, params.k, params.v
        b, s_local, n, d = k.shape
        device = k.device
        
        k_gathered = torch.empty(
            [self.ring_world_size, b, s_local, n, d],
            dtype=k.dtype, device=device
        )
        v_gathered = torch.empty(
            [self.ring_world_size, b, s_local, n, d],
            dtype=v.dtype, device=device
        )
        
        k_handle = dist.all_gather_into_tensor(
            k_gathered, k.contiguous(), 
            group=self.ring_pg, async_op=True
        )
        v_handle = dist.all_gather_into_tensor(
            v_gathered, v.contiguous(), 
            group=self.ring_pg, async_op=True
        )
        
        out_local, lse_local = self._attention_with_lse(params)
        
        k_handle.wait()
        v_handle.wait()
        
        other_indices = self._get_other_indices(
            self.ring_rank, self.ring_world_size, device
        )
        
        k_others = torch.index_select(k_gathered, dim=0, index=other_indices)
        v_others = torch.index_select(v_gathered, dim=0, index=other_indices)

        target_shape = (b, n, d)
        k_others = self._kv_post_process(k_others, params.joint_tensor_key, target_shape)
        v_others = self._kv_post_process(v_others, params.joint_tensor_value, target_shape)
        
        params_others = AttentionParams(
            q=q, k=k_others, v=v_others,
            causal=False,
            softmax_scale=params.softmax_scale,
            per_head_compute=params.per_head_compute
        )
        out_others, lse_others = self._attention_with_lse(params_others)
        
        out_merged = self._merge_two_outputs(
            out_local, lse_local,
            out_others, lse_others
        )
        
        return out_merged
    
    def _ring_attention_allgather(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        
        k, v = params.k, params.v
        b, s_local, n, d = k.shape
        
        k_gathered = torch.empty(
            [self.ring_world_size, b, s_local, n, d],
            dtype=k.dtype, device=k.device
        )
        v_gathered = torch.empty(
            [self.ring_world_size, b, s_local, n, d],
            dtype=v.dtype, device=v.device
        )
        
        dist.all_gather_into_tensor(k_gathered, k.contiguous(), group=self.ring_pg)
        dist.all_gather_into_tensor(v_gathered, v.contiguous(), group=self.ring_pg)
        
        target_shape = (b, n, d)
        k_full = self._kv_post_process(k_gathered, params.joint_tensor_key, target_shape)
        v_full = self._kv_post_process(v_gathered, params.joint_tensor_value, target_shape)
        
        params_full = AttentionParams(
            q=params.q, k=k_full, v=v_full,
            causal=params.causal,
            softmax_scale=params.softmax_scale,
            per_head_compute=params.per_head_compute
        )
        return self._compute_attention(params_full)
    
    @staticmethod
    def _kv_post_process(tensor, joint_tensor, target_shape, dim=1):
        b, n, d = target_shape
        tensor = tensor.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)
        return torch.cat([tensor, joint_tensor], dim=dim)
    
    def _compute_attention(
        self,
        params: AttentionParams
    ) -> torch.Tensor:
        
        q, k, v = params.q, params.k, params.v
        
        if params.per_head_compute:
            output = []
            for i in range(q.shape[2]):
                head_params = AttentionParams(
                    q=q[:, :, i: i + 1, :],
                    k=k[:, :, i: i + 1, :],
                    v=v[:, :, i: i + 1, :],
                    causal=params.causal,
                    softmax_scale=params.softmax_scale,
                    per_head_compute=False
                )
                out_i = self._npu_attention(head_params)
                output.append(out_i)
            return torch.cat(output, dim=2)
        else:
            return self._npu_attention(params)
    
    def _attention_with_lse(
        self,
        params: AttentionParams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        q = params.q
        
        if params.per_head_compute:
            output = []
            lse_output = []
            for i in range(q.shape[2]):
                head_params = AttentionParams(
                    q=q[:, :, i: i + 1, :],
                    k=params.k[:, :, i: i + 1, :],
                    v=params.v[:, :, i: i + 1, :],
                    causal=params.causal,
                    softmax_scale=params.softmax_scale,
                    per_head_compute=False
                )
                out_i, lse_i = self._npu_attention_with_lse(head_params)
                output.append(out_i)
                lse_output.append(lse_i)
            return torch.cat(output, dim=2), torch.cat(lse_output, dim=1)
        else:
            return self._npu_attention_with_lse(params)
    
    @staticmethod
    def _npu_attention(
        params: AttentionParams
    ) -> torch.Tensor:
        
        import torch_npu
        
        q, k, v = params.q, params.k, params.v
        out_dtype = q.dtype
        
        b, s, n, d = q.shape
        softmax_scale = params.softmax_scale
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        num_heads = n
        num_key_value_heads = k.shape[1]
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0
        
        config = NPUAttentionConfig(
            num_heads=num_heads,
            scale=float(softmax_scale),
            num_key_value_heads=num_key_value_heads,
            next_tokens=65535 if not params.causal else 0,
        )
        
        out = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=config.num_heads,
            scale=config.scale,
            input_layout=config.input_layout,
            num_key_value_heads=config.num_key_value_heads,
            pre_tokens=config.pre_tokens,
            next_tokens=config.next_tokens,
            sparse_mode=config.sparse_mode,
            inner_precise=config.inner_precise,
        )[0]
        
        out = out.transpose(1, 2).contiguous()
        
        return out.to(out_dtype)
    
    @staticmethod
    def _npu_attention_with_lse(
        params: AttentionParams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        import torch_npu
        
        q, k, v = params.q, params.k, params.v
        out_dtype = q.dtype
        
        b, s, n, d = q.shape
        softmax_scale = params.softmax_scale
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        num_heads = n
        num_key_value_heads = k.shape[1]
        
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"GQA requires num_heads divisible by num_key_value_heads, "
                f"got num_heads={num_heads}, num_key_value_heads={num_key_value_heads}"
            )
        
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0
        
        config = NPUAttentionConfig(
            num_heads=num_heads,
            scale=float(softmax_scale),
            num_key_value_heads=num_key_value_heads,
            next_tokens=65535 if not params.causal else 0,
            softmax_lse_flag=True,
        )
        
        out, lse = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=config.num_heads,
            scale=config.scale,
            input_layout=config.input_layout,
            num_key_value_heads=config.num_key_value_heads,
            pre_tokens=config.pre_tokens,
            next_tokens=config.next_tokens,
            sparse_mode=config.sparse_mode,
            inner_precise=config.inner_precise,
            softmax_lse_flag=config.softmax_lse_flag,
        )
        
        out = out.transpose(1, 2).contiguous()
        lse = lse.squeeze(-1)
        
        return out.to(out_dtype), lse
    
    @staticmethod
    def _merge_two_outputs(out1, lse1, out2, lse2):
        
        lse1_expanded = lse1.transpose(1, 2).unsqueeze(-1)
        lse2_expanded = lse2.transpose(1, 2).unsqueeze(-1)
        
        max_lse = torch.maximum(lse1_expanded, lse2_expanded)
        lse_new = max_lse + torch.log(
            torch.exp(lse1_expanded - max_lse) + 
            torch.exp(lse2_expanded - max_lse)
        )
        
        weight1 = torch.exp(lse1_expanded - lse_new)
        weight2 = torch.exp(lse2_expanded - lse_new)
        
        out_merged = weight1 * out1 + weight2 * out2
        
        return out_merged