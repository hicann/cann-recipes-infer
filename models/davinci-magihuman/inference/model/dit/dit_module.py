# coding=utf-8
# Adapted from
# https://github.com/GAIR-NLP/daVinci-MagiHuman,
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Copyright (c) 2026 SandAI. All Rights Reserved.
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

import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch_npu
import torch.nn as nn
from einops import rearrange, repeat
from inference.common import Modality, VarlenHandler
from inference.infra.parallelism import ulysses_scheduler
from magi_compiler import magi_compile
from magi_compiler.api import magi_register_custom_op
from magi_compiler.config import CompileConfig
from torch import Tensor
from torch.nn import Parameter


@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float


# Define the MLP activation type
class MLPActivationType(Enum):
    """Enumeration of supported activation functions for MLP"""

    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x_f32 = x.to(torch.float32).contiguous()
    out = torch_npu.npu_clipped_swiglu(x_f32, alpha=alpha, limit=limit, bias=1.0, interleaved=True)
    return out.to(out_dtype)


def gelu7(x, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x.clamp_(max=limit)
    x = torch_npu.npu_fast_gelu(x)
    return x.to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    match activation_type:
        case MLPActivationType.SWIGLU7:
            return swiglu7
        case MLPActivationType.GELU7:
            return gelu7
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


class ModalityDispatcher:
    permuted_modality_mapping: torch.Tensor
    group_size: torch.Tensor
    group_size_cpu: list[int]
    num_modalities: int

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        """
        Initialize dispatcher.
        This runs once during object construction and precomputes all mappings.
        """
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        self.permuted_modality_mapping = self._precompute_permute_mapping(modality_mapping)

        self.group_size = torch.bincount(self.permuted_modality_mapping, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]

    def _precompute_permute_mapping(self, modality_mapping):
        # 1. Compute forward and inverse permutation mappings.
        # argsort is an efficient O(N log N) operation.
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)

        # 2. Compute group size for each modality.
        # bincount is highly efficient for counting.
        permuted_modality_mapping = modality_mapping[self.permute_mapping]

        return permuted_modality_mapping

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        grouped_tensors = torch.split(x, self.group_size_cpu, dim=0)
        return list(grouped_tensors)

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply forward permutation to tensor."""
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply inverse permutation to tensor."""
        return x[inv_permute_mapping]


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: Optional[torch.device] = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    if not interleaved and x.dim() == 4:
        r1 = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2).to(x.dtype)
        r2 = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2).to(x.dtype)
        if ro_dim == x.shape[-1]:
            return torch_npu.npu_rotary_mul(x, r1, r2, rotary_mode='half')
        x_rot = torch_npu.npu_rotary_mul(x[..., :ro_dim], r1, r2, rotary_mode='half')
        return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


def apply_rotary_emb_npu(x, cos, sin):
    """
    Apply rotary embedding using npu_rotary_mul with pre-expanded cos/sin.
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (1, seqlen, 1, rotary_dim) - pre-expanded via cat([d, d], dim=-1)
    """
    ro_dim = cos.shape[-1]
    if ro_dim == x.shape[-1]:
        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='half')
    x_rot = torch_npu.npu_rotary_mul(x[..., :ro_dim], cos, sin, rotary_mode='half')
    return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            dim: Output feature dimension, total channels, must be divisible by 6
            max_res: Max pixel-frequency resolution for pixel-domain bands
            temperature: Temperature in inverse-frequency mode
            in_pixels: True -> pixel-frequency bands, False -> inverse-frequency bands
            linear_bands: Whether pixel-frequency bands are linearly spaced
            learnable: Whether frequency bands are trainable
        """
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        # Make frequency bands trainable or register as buffer
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [L,9], column order (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] element-wise Fourier embedding
        """
        # Use slicing instead of unbind + stack to reduce intermediates
        coords_xyz = coords[:, :3]  # [L,3] -> (t, h, w)
        sizes = coords[:, 3:6]  # [L,3] -> (T, H, W)
        refs = coords[:, 6:9]  # [L,3] -> (ref_T, ref_H, ref_W)

        # Compute scale factors
        scales = (refs - 1) / (sizes - 1)  # [L,3]

        # NOTE: if both ref and size are 1, scale is fixed to 1; otherwise invalid
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        # Center alignment: apply to h,w only (not time)
        centers = (sizes - 1) / 2  # [L,3]
        centers[:, 0] = 0  # Do not center the time dimension
        coords_xyz = coords_xyz - centers  # [L,3]

        # Project to frequency bands in one shot: [L,3,B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands

        # Compute sin & cos and concatenate
        sin_proj = proj.sin()  # [L,3,B]
        cos_proj = proj.cos()

        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=self.device).to(self.dtype)
        return bands


class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]
    dim: int
    eps: float
    num_modality: int

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality

        self.weight = torch.nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.float32))
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward_multi_experts(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher, residual: Optional[torch.Tensor] = None
    ):
        x_list = modality_dispatcher.dispatch(x)
        weight_chunked = self.weight.chunk(self.num_modality, dim=0)
        if residual is not None:
            residual_list = modality_dispatcher.dispatch(residual)
            for i in range(self.num_modality):
                gamma_i = (weight_chunked[i] + 1).to(torch.bfloat16).contiguous()
                x_list[i], _, residual_list[i] = torch_npu.npu_add_rms_norm(
                    residual_list[i].to(gamma_i.dtype), x_list[i].to(gamma_i.dtype), gamma_i, self.eps)
            t = modality_dispatcher.undispatch(*x_list)
            new_residual = modality_dispatcher.undispatch(*residual_list)
            return t, new_residual
        for i in range(self.num_modality):
            gamma_i = (weight_chunked[i] + 1).contiguous()
            x_list[i] = torch_npu.npu_rms_norm(x_list[i], gamma_i, self.eps)[0]
        t = modality_dispatcher.undispatch(*x_list)
        return t

    def forward_single_expert(
        self,
        x: torch.Tensor,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        gamma = (self.weight + 1).contiguous()
        if residual is not None:
            gamma = gamma.to(torch.bfloat16)
            out, _, new_residual = torch_npu.npu_add_rms_norm(
                residual.to(gamma.dtype), x.to(gamma.dtype), gamma, self.eps)
            return out, new_residual
        out = torch_npu.npu_rms_norm(x, gamma, self.eps)[0]
        return out


class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_dtype: Optional[torch.dtype],
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        # Convert input to specified input data type
        input_cast = input.to(compute_dtype)
        # Convert weight to computation data type
        weight_cast = weight.to(compute_dtype)
        # Perform linear operation
        output = torch.matmul(input_cast, weight_cast.t())

        # Add bias if present
        if bias is not None:
            bias_cast = bias.to(compute_dtype)
            output = output + bias_cast
        else:
            bias_cast = None

        # Convert output to specified output data type
        return output.to(output_dtype)


class BaseLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_layers", "num_experts"]
    in_features: int
    out_features: int
    num_layers_for_initialization: int
    num_experts: int
    weight: Tensor

    def __init__(
        self, in_features, out_features, num_layers_for_initialization, num_experts, bias=True, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = Parameter(torch.empty((out_features * num_experts, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features * num_experts, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        return _BF16ComputeLinear.apply(input, self.weight, self.bias, output_dtype, torch.bfloat16)


class NativeMoELinear(BaseLinear):
    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype

        input_list = modality_dispatcher.dispatch(input)  # type: ignore
        weight_chunked = self.weight.chunk(self.num_experts, dim=0)

        if self.bias is not None:
            bias_chunked = self.bias.chunk(self.num_experts, dim=0)

        for i in range(self.num_experts):
            input_list[i] = _BF16ComputeLinear.apply(
                input_list[i],
                weight_chunked[i],
                bias_chunked[i] if self.bias is not None else None,
                output_dtype,
                torch.bfloat16,
            )
        return modality_dispatcher.undispatch(*input_list)  # type: ignore


def create_linear(
    in_features, out_features, num_layers=1, num_experts=1, bias=True, device=None, dtype=None
) -> BaseLinear | NativeMoELinear:
    if num_experts == 1:
        return BaseLinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)
    else:
        return NativeMoELinear(in_features, out_features, num_layers, num_experts, bias, device, dtype)



@magi_register_custom_op(name="infra::flash_attn_func", is_subgraph_boundary=True)
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    b, s, n, d = query.shape
    _, _, kv_n, _ = key.shape
    scale = 1.0 / (d ** 0.5)

    original_dtype = query.dtype

    query = query.to(torch.bfloat16).contiguous()
    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query, key, value,
        num_heads=n,
        num_key_value_heads=kv_n,
        scale=scale,
        input_layout="BSND",
        sparse_mode=0
    )
    return output.to(original_dtype)


_split_q_range_cache = {}


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[List[List[int]], List[List[List[int]]]]:
    q_id = id(q_ranges)
    entry = _split_q_range_cache.get(q_id)
    if entry is not None:
        wr, k_id, result = entry
        if wr() is q_ranges and k_id == id(k_ranges):
            return result
        _split_q_range_cache.pop(q_id, None)
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [[start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])]
    q_ranges_list = q_ranges.tolist()
    k_ranges_list = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges_list, k_ranges_list):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    result = (q_ranges_out, k_ranges_out)

    def _on_release(wr, _q_id=q_id):
        _split_q_range_cache.pop(_q_id, None)

    _split_q_range_cache[q_id] = (weakref.ref(q_ranges, _on_release), id(k_ranges), result)
    return q_ranges_out, k_ranges_out


def _flex_flash_attn_func_infer_output_meta(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> torch.Tensor:
    output = torch.empty_like(query)
    return output


@magi_register_custom_op(
    name="infra::flex_flash_attn_func",
    mutates_args=(),
    infer_output_meta_fn=_flex_flash_attn_func_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> torch.Tensor:
    if query.dim() == 4:
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)

    _, n, d = query.shape
    _, nk, _ = key.shape
    scale = 1.0 / (d ** 0.5)

    original_dtype = query.dtype

    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    q_ranges_split, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)

    output = torch.zeros_like(query)

    for q_range, k_ranges in zip(q_ranges_split, k_range_list):
        q_start, q_end = q_range
        qo_out, qo_lse = None, None
        for k_range in k_ranges:
            k_start, k_end = k_range
            q_seg = query[q_start:q_end].permute(1, 0, 2).unsqueeze(0).contiguous()
            k_seg = key[k_start:k_end].permute(1, 0, 2).unsqueeze(0).contiguous()
            v_seg = value[k_start:k_end].permute(1, 0, 2).unsqueeze(0).contiguous()

            cur_qo_out, cur_qo_lse = torch_npu.npu_fused_infer_attention_score(
                q_seg, k_seg, v_seg,
                num_heads=n,
                num_key_value_heads=nk,
                scale=scale,
                input_layout="BNSD",
                pre_tokens=65535,
                next_tokens=65535,
                sparse_mode=0,
                softmax_lse_flag=True,
            )
            cur_qo_out = cur_qo_out.squeeze(0).permute(1, 0, 2).contiguous()
            cur_qo_lse = cur_qo_lse.squeeze(0).squeeze(-1).float()

            if qo_out is None:
                qo_out = cur_qo_out
                qo_lse = cur_qo_lse
            else:
                qo_lse.masked_fill_(qo_lse == torch.inf, -torch.inf)
                cur_qo_lse.masked_fill_(cur_qo_lse == torch.inf, -torch.inf)
                max_lse = torch.max(qo_lse, cur_qo_lse)
                qo_se, cur_qo_se = torch.exp(qo_lse - max_lse), torch.exp(cur_qo_lse - max_lse)
                sum_se = qo_se + cur_qo_se
                qo_scale, cur_qo_scale = qo_se / sum_se, cur_qo_se / sum_se

                qo_out = (
                    qo_out * qo_scale.permute(1, 0).unsqueeze(-1)
                    + cur_qo_out * cur_qo_scale.permute(1, 0).unsqueeze(-1)
                )
                qo_lse = torch.log(sum_se) + max_lse

        output[q_start:q_end] = qo_out

    return output.to(original_dtype)


def _attention_with_cp_infer_output_meta(q: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.empty_like(q, dtype=torch.bfloat16).squeeze(0)


@magi_register_custom_op(
    name="infra::flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flash_attn_with_cp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_split_sizes: List[int]) -> torch.Tensor:
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    from inference.infra.distributed import get_cp_group, get_cp_world_size
    from inference.infra.parallelism.all_to_all_primitive import (
        batch_scatter_head_gather_seqlen, scatter_seqlen_gather_head,
    )

    if get_cp_world_size() > 1:
        q, k, v = batch_scatter_head_gather_seqlen(
            [q.squeeze(0), k.squeeze(0), v.squeeze(0)], cp_split_sizes, get_cp_group()
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    self_attn_out = torch.ops.infra.flash_attn_func(q, k, v).squeeze(0)

    if get_cp_world_size() > 1:
        self_attn_out = scatter_seqlen_gather_head(self_attn_out, cp_split_sizes, get_cp_group(), async_op=False)
        self_attn_out = rearrange(self_attn_out, "(cp sq) hn hd -> sq (cp hn) hd", cp=get_cp_world_size())

    return self_attn_out


@magi_register_custom_op(
    name="infra::flex_flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_split_sizes: List[int],
) -> torch.Tensor:
    q, k, v = q.to(torch.bfloat16).squeeze(0), k.to(torch.bfloat16).squeeze(0), v.to(torch.bfloat16).squeeze(0)

    from inference.infra.distributed import get_cp_group, get_cp_world_size
    from inference.infra.parallelism.all_to_all_primitive import (
        batch_scatter_head_gather_seqlen, scatter_seqlen_gather_head,
    )

    if get_cp_world_size() > 1:
        q, k, v = batch_scatter_head_gather_seqlen([q, k, v], cp_split_sizes, get_cp_group())

    out = torch.ops.infra.flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges)

    if get_cp_world_size() > 1:
        out = scatter_seqlen_gather_head(out, cp_split_sizes, get_cp_group(), async_op=False)
        out = rearrange(out, "(cp sq) hn hd -> sq (cp hn) hd", cp=get_cp_world_size())

    return out


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, eps=1e-6, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        self.linear_qkv = create_linear(
            config.hidden_size,
            config.num_heads_q * config.head_dim + config.num_heads_kv * config.head_dim * 2 + self.gating_size,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.linear_proj = create_linear(
            config.num_heads_q * config.head_dim,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def reset_parameters(self):
        if hasattr(self.linear_proj, "reset_parameters_output_layer"):
            self.linear_proj.reset_parameters_output_layer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher, residual=residual)
        if residual is not None:
            normed, hidden_states = normed
        qkv: torch.Tensor = self.linear_qkv(normed.to(torch.bfloat16), modality_dispatcher=modality_dispatcher)

        q, k, v, g = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=1)
        q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
        k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
        g = g.view(k.shape[0], self.config.num_heads_q, -1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

        if isinstance(rope, tuple):
            rope_cos, rope_sin = rope
            q = apply_rotary_emb_npu(q, rope_cos, rope_sin)
            k = apply_rotary_emb_npu(k, rope_cos, rope_sin)
        else:
            sin_emb, cos_emb = rope.tensor_split(2, -1)
            q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
            k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        if self.config.use_local_attn:
            self_attn_out = flex_flash_attn_with_cp(
                q, k, v, local_attn_handler.q_ranges, local_attn_handler.k_ranges, cp_split_sizes
            )
        else:
            self_attn_out = flash_attn_with_cp(q, k, v, cp_split_sizes)
        self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

        if self.config.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)

        self_attn_out = self_attn_out.view(-1, self.config.num_heads_q * self.config.head_dim)
        out = self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)
        return out, hidden_states


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        num_experts = config.num_modality
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(
        self,
        x: torch.Tensor,
        modality_dispatcher: ModalityDispatcher,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.pre_norm(x, modality_dispatcher=modality_dispatcher, residual=residual)
        if residual is not None:
            normed, x = normed
        normed = normed.to(torch.bfloat16)
        normed = self.up_gate_proj(normed, modality_dispatcher=modality_dispatcher).to(torch.float32)
        normed = self.activation_func(normed).to(torch.bfloat16)
        out = self.down_proj(normed, modality_dispatcher=modality_dispatcher)
        return out, x

    def extra_repr(self) -> str:
        return f"{self.up_gate_proj.weight.shape=}, {self.down_proj.weight.shape=}"


@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    params_dtype: torch.dtype


class Adapter(torch.nn.Module):
    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=torch.float32)
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads, in_pixels=False, learnable=False
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        rope = self.rope(coords_mapping)
        output_x = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device, dtype=x.dtype)
        output_x[text_mask] = self.text_embedder(x[text_mask, : self.config.text_in_channels])
        output_x[audio_mask] = self.audio_embedder(x[audio_mask, : self.config.audio_in_channels])
        output_x[video_mask] = self.video_embedder(x[video_mask, : self.config.video_in_channels])
        return output_x, rope


class TransFormerLayer(torch.nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            params_dtype=config.params_dtype,
            checkpoint_qk_layernorm_rope=config.checkpoint_qk_layernorm_rope,
            num_modality=num_modality,
            num_layers=config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=config.enable_attn_gating,
        )
        self.attention: Attention = Attention(attention_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            params_dtype=config.params_dtype,
            num_modality=num_modality,
            num_layers=config.num_layers,
            gated_act=gated_act,
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, hidden_states = self.attention(
            hidden_states,
            rope,
            permute_mapping,
            inv_permute_mapping,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
            cp_split_sizes,
            residual=residual,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
            hidden_states = hidden_states + attn_out
            mlp_out, _ = self.mlp(hidden_states, modality_dispatcher)
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
            hidden_states = hidden_states + mlp_out
            return hidden_states, None
        else:
            mlp_out, hidden_states = self.mlp(hidden_states, modality_dispatcher, residual=attn_out)
            return hidden_states, mlp_out


is_base_model = True


def config_patch(compile_config: CompileConfig) -> CompileConfig:
    global is_base_model
    if is_base_model:
        is_base_model = False
    else:
        compile_config.offload_config.gpu_resident_weight_ratio = 0.0
    return compile_config


@magi_compile(config_patch=config_patch)
class TransformerBlock(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers: list[TransFormerLayer] = nn.ModuleList()
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransFormerLayer(model_config, layer_idx))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        residual = None
        for _, layer in enumerate(self.layers):
            x, residual = layer(
                x,
                rope,
                permute_mapping,
                inv_permute_mapping,
                varlen_handler,
                local_attn_handler,
                modality_dispatcher,
                cp_split_sizes,
                residual=residual,
            )
        if residual is not None:
            x = x + residual
        return x


@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    post_process_dtype: torch.dtype


class DiTModel(torch.nn.Module):
    config: TransformerConfig

    def __init__(self, model_config: Any):
        super().__init__()
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            post_process_dtype=torch.float32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            params_dtype=torch.float32,
        )
        self.adapter: Adapter = Adapter(adapter_config)
        self.block: TransformerBlock = TransformerBlock(model_config=model_config)
        self.final_norm_video = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_linear_video = nn.Linear(
            self.config.hidden_size, self.config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            self.config.hidden_size, self.config.audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
    ):
        x = ulysses_scheduler().dispatch(x)
        coords_mapping = ulysses_scheduler().dispatch(coords_mapping)
        modality_mapping = ulysses_scheduler().dispatch(modality_mapping)
        cp_split_sizes = ulysses_scheduler().cp_split_sizes

        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        permute_mapping = modality_dispatcher.permute_mapping
        inv_permute_mapping = modality_dispatcher.inv_permute_mapping
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
        x = x.to(self.config.params_dtype)
        rope = rope.to(torch.bfloat16)
        sin_emb, cos_emb = rope.tensor_split(2, -1)
        rope_cos = torch.cat([cos_emb, cos_emb], dim=-1).unsqueeze(0).unsqueeze(2).contiguous()
        rope_sin = torch.cat([sin_emb, sin_emb], dim=-1).unsqueeze(0).unsqueeze(2).contiguous()
        rope = (rope_cos, rope_sin)
        x = ModalityDispatcher.permute(x, permute_mapping)
        x = self.block(
            x,
            rope,
            permute_mapping=permute_mapping,
            inv_permute_mapping=inv_permute_mapping,
            varlen_handler=varlen_handler,
            local_attn_handler=local_attn_handler,
            modality_dispatcher=modality_dispatcher,
            cp_split_sizes=cp_split_sizes,
        )
        x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=torch.float32,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio
        x_out = ulysses_scheduler().undispatch(x_out)
        return x_out
