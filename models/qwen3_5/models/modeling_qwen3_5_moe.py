# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
import itertools
import math
import os
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union, Iterable, Any

import torch
import torch_npu
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.types import _dtype
import torchair

from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    ModelOutput,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from executor.core.config import InferenceConfig, CommManager
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import align_up
from executor.utils.forward_metadata import ForwardMetaData
from module.fuse_moe_gmm import FusedMoEGMM
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
    VocabParallelEmbedding,
)
from module.utils import set_weight_attrs

from .configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig

logger = logging.getLogger(__name__)

causal_conv1d_update, causal_conv1d_fn = None, None

chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
FusedRMSNormGated = None

local_rank = os.environ.get('LOCAL_RANK', '0')

global_last_recurrent_state = None
global_core_attn_out = None
global_mask = None
global_mask_diagonal = None


def qwen3_5_prefill_mm_all_reduce(
    layer: RowParallelLinear,
    input_: torch.Tensor,
    comm_manager: CommManager,
    group_name: str,
    enable_mm_all_reduce_base: bool,
    forward_metadata: ForwardMetaData | None = None,
) -> torch.Tensor | None:
    if (
        not enable_mm_all_reduce_base
        or forward_metadata is None
        or not forward_metadata.is_prefill
        or layer.tp_size <= 1
        or not layer.input_is_parallel
        or layer.bias is not None
        or layer.skip_bias_add
    ):
        return None
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return None

    hcom = comm_manager.get_group_name(group_name)
    if hcom is None:
        return None
    return torch_npu.npu_mm_all_reduce_base(input_, layer.weight.data, hcom, reduce_op="sum")


class Qwen3_5MoeGatedDeltaNetFusedInProj(MergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        total_key_dim: int,
        total_value_dim: int,
        total_num_v_heads: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        prefix: str = "",
    ):
        self.total_key_dim = total_key_dim
        self.total_value_dim = total_value_dim
        self.total_num_v_heads = total_num_v_heads
        super().__init__(
            hidden_size,
            [
                total_key_dim,
                total_key_dim,
                total_value_dim,
                total_value_dim,
                total_num_v_heads,
                total_num_v_heads,
            ],
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            prefix=prefix,
        )
        self._shard_name_to_id = {
            "q": 0,
            "k": 1,
            "v": 2,
            "z": 3,
            "b": 4,
            "a": 5,
        }

    def _load_output_shard(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        is_per_block_scale = getattr(param, "is_per_block_scale", False)

        if output_dim is not None:
            if is_per_block_scale:
                block_size = self.quant_config.weight_block_size[0]
                shard_offset = math.ceil(sum(self.output_sizes[:loaded_shard_id]) / block_size) // self.tp_size
                shard_size = math.ceil(self.output_sizes[loaded_shard_id] / block_size) // self.tp_size
            else:
                shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
                shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if not use_bitsandbytes_4bit:
                start_idx = self.tp_rank * shard_size
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "Qwen3_5MoeGatedDeltaNetFusedInProj, assume the weight is "
                    "the same for all partitions."
                )

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ) -> None:
        if loaded_shard_id is None:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
            return

        if loaded_shard_id == "qkv":
            shard_offsets = [
                ("q", 0, self.total_key_dim),
                ("k", self.total_key_dim, self.total_key_dim),
                ("v", 2 * self.total_key_dim, self.total_value_dim),
            ]
            output_dim = getattr(param, "output_dim", None)
            if output_dim is None:
                raise RuntimeError("QKV fused loading requires `output_dim` to be set")
            packed_dim = getattr(param, "packed_dim", None)
            for shard_name, shard_offset, shard_size in shard_offsets:
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                qkv_shard = loaded_weight.narrow(output_dim, shard_offset, shard_size)
                self._load_output_shard(param, qkv_shard, self._shard_name_to_id[shard_name])
            return

        shard_id = self._shard_name_to_id.get(loaded_shard_id)
        if shard_id is None:
            raise RuntimeError(f"Unsupported loaded_shard_id: {loaded_shard_id}")
        self._load_output_shard(param, loaded_weight, shard_id)


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.silu(x)


class SiLUAndMul(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        output = torch_npu.npu_swiglu(x)
        return output


def maybe_autocast(
    device_type: str,
    dtype: _dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()



class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config,
        device,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    @torch.no_grad()
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3_5Moe has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        # BSH format: position_ids is 2D [batch, seq_len], expand to 3D [3, batch, seq_len]
        # TND format: position_ids is 1D [total_tokens], expand to 3D [3, 1, total_tokens]
        if position_ids.ndim == 1:
            # TND: [total_tokens] -> [3, 1, total_tokens]
            position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(3, 1, -1)
        elif position_ids.ndim == 2:
            # BSH: [batch, seq_len] -> [3, batch, seq_len]
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # position_ids shape: [3, 1, total_tokens] for TND or [3, batch, seq_len] for BSH
        # inv_freq shape: [32] (for rotary_dim=64, partial_rotary_factor=0.25)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape [3, batch/tokens, 1, seq_len/tokens]

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3_5MoeRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def ln_npu(self, hidden_states):
        return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = self.ln_npu(hidden_states)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    NOTE: In TND format, attention_mask is typically None, so this function returns unchanged.
    """
    # TND format: attention_mask is usually None, return unchanged
    if attention_mask is None:
        return hidden_states
    # Legacy BSH format support
    if attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def ge_safe_softplus(x: torch.Tensor) -> torch.Tensor:
    # Avoid aten::softplus so ge_graph can lower this path.
    return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_solve_triangular=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    global global_mask
    if global_mask is None:
        global_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    mask = global_mask

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    if use_solve_triangular:
        attn = torch.linalg.solve_triangular(
            torch.eye(chunk_size, device=attn.device, dtype=attn.dtype) - attn,
            attn,
            upper=False,
        )
    else:
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    global global_last_recurrent_state
    global global_core_attn_out
    global global_mask_diagonal
    if global_last_recurrent_state is None:
        global_last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        global_core_attn_out = torch.zeros_like(value)
        global_mask_diagonal = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
            diagonal=1
        )
    last_recurrent_state = global_last_recurrent_state
    core_attn_out = global_core_attn_out
    mask = global_mask_diagonal

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.infer_config = infer_config
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_tp_rank = (
            comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        )
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get(
            "enable_mm_all_reduce_base", False
        )
        self.enable_gdn_solve_triangular = infer_config.model_config.custom_params.get(
            "enable_gdn_solve_triangular", False
        )
        self.comm_manager = comm_manager
        if config.linear_num_value_heads % self.attn_tp_size != 0:
            raise ValueError(
                f"linear_num_value_heads ({config.linear_num_value_heads}) must be divisible by "
                f"attn_tp_size ({self.attn_tp_size})."
            )
        if config.linear_num_key_heads % self.attn_tp_size != 0:
            raise ValueError(
                f"linear_num_key_heads ({config.linear_num_key_heads}) must be divisible by "
                f"attn_tp_size ({self.attn_tp_size})."
            )
        self.total_num_v_heads = config.linear_num_value_heads
        self.total_num_k_heads = config.linear_num_key_heads
        self.num_v_heads = self.total_num_v_heads // self.attn_tp_size
        self.num_k_heads = self.total_num_k_heads // self.attn_tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.total_key_dim = self.head_k_dim * self.total_num_k_heads
        self.total_value_dim = self.head_v_dim * self.total_num_v_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = SiLUActivation()
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        set_weight_attrs(self.conv1d.weight, {"weight_loader": self._load_linear_attn_qkv_conv_weight})

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        set_weight_attrs(self.dt_bias, {"weight_loader": self._load_linear_attn_v_head_param})

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        set_weight_attrs(self.A_log, {"weight_loader": self._load_linear_attn_v_head_param})

        self.norm = (
            Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=torch.cuda.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        )

        self.out_proj = RowParallelLinear(
            self.total_value_dim,
            self.hidden_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        platform = torch.npu.get_device_name()
        self.use_npu_chunk_gated_delta_rule = '950' not in platform and hasattr(torch_npu, "npu_chunk_gated_delta_rule")

        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.info(
                "WARNING: The fast path is not available because one of the required library is not installed. "
                "Falling back to torch implementation. To install follow "
                "https://github.com/fla-org/flash-linear-attention#installation and "
                "https://github.com/Dao-AILab/causal-conv1d"
            )

        self.in_proj_fused = Qwen3_5MoeGatedDeltaNetFusedInProj(
            self.hidden_size,
            self.total_key_dim,
            self.total_value_dim,
            self.total_num_v_heads,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            prefix=f"{prefix}.in_proj_fused",
        )

        self.tmp = torch.zeros(
            1,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            dtype=torch.bfloat16,
            device=torch.npu.current_device(),
        )
        self.gdn_actual_seq_lengths = torch.ones((1,), dtype=torch.int32, device=torch.npu.current_device())
        self.gdn_ssm_state_indices = torch.arange(1, dtype=torch.int32, device=torch.npu.current_device())


        self.conv_state = None
        self.recurrent_state = None

    def _slice_linear_attn_qkv_packed_tensor(self, loaded_weight: torch.Tensor, shard_dim: int) -> torch.Tensor:
        shard_specs = [
            (0, self.total_key_dim, self.key_dim),
            (self.total_key_dim, self.total_key_dim, self.key_dim),
            (2 * self.total_key_dim, self.total_value_dim, self.value_dim),
        ]
        shards = []
        for offset, _, local_size in shard_specs:
            start_idx = offset + self.attn_tp_rank * local_size
            shards.append(loaded_weight.narrow(shard_dim, start_idx, local_size))
        return torch.cat(shards, dim=shard_dim).contiguous()

    def _load_linear_attn_qkv_conv_weight(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if self.attn_tp_size > 1:
            loaded_weight = self._slice_linear_attn_qkv_packed_tensor(loaded_weight, shard_dim=0)
        default_weight_loader(param, loaded_weight)

    def _load_linear_attn_v_head_param(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if self.attn_tp_size > 1:
            shard_size = param.shape[0]
            start_idx = self.attn_tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        default_weight_loader(param, loaded_weight)

    def _ensure_recurrent_decode_buffers(self, batch_size: int, seq_len: int, device: torch.device):
        expected_tmp_shape = (batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim)
        if self.tmp.shape != expected_tmp_shape or self.tmp.device != device:
            self.tmp = torch.zeros(
                *expected_tmp_shape,
                dtype=torch.bfloat16,
                device=device,
            )
            torch._dynamo.mark_static(self.tmp)
        else:
            self.tmp.zero_()

        expected_meta_shape = (batch_size,)
        if self.gdn_actual_seq_lengths.shape != expected_meta_shape or self.gdn_actual_seq_lengths.device != device:
            self.gdn_actual_seq_lengths = torch.empty(expected_meta_shape, dtype=torch.int32, device=device)
            torch._dynamo.mark_static(self.gdn_actual_seq_lengths)
        self.gdn_actual_seq_lengths.fill_(seq_len)

        if self.gdn_ssm_state_indices.shape != expected_meta_shape or self.gdn_ssm_state_indices.device != device:
            self.gdn_ssm_state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            torch._dynamo.mark_static(self.gdn_ssm_state_indices)

    def _forward_prefill(
        self,
        fused_proj: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
    ):
        """
        Prefill path.

        Input:
            fused_proj: [total_tokens, proj_dim] (TND)

        Return:
            core_attn_out: [total_tokens * num_v_heads, head_v_dim]
        """
        conv_state = self.conv_state
        recurrent_state = self.recurrent_state

        actual_seq_lens = forward_metadata.actual_seq_lengths_q
        cu_seq_lens = forward_metadata.actual_seq_lengths_cu_q

        # Ensure leading 0
        if cu_seq_lens[0] != 0:
            cu_seq_lens = F.pad(cu_seq_lens, (1, 0))

        num_requests = actual_seq_lens.numel()
        proj_dim = fused_proj.shape[-1]

        # Chunk alignment to avoid internal padding in chunk_gated_delta_rule
        max_seq_len = actual_seq_lens.max().item()
        chunk_size = 64
        aligned_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size

        # === Build flatten indices for index_copy_ ===
        token_idx = torch.arange(fused_proj.shape[0], device=fused_proj.device)
        batch_idx = torch.bucketize(token_idx, cu_seq_lens[1:], right=False)
        seq_start = cu_seq_lens[batch_idx]
        seq_idx = token_idx - seq_start
        flat_idx = batch_idx * aligned_seq_len + seq_idx

        # === TND -> BSH (index_copy_) ===
        fused_proj_flat = fused_proj.new_zeros(num_requests * aligned_seq_len, proj_dim)
        fused_proj_flat.index_copy_(0, flat_idx, fused_proj)
        fused_proj_bsh = fused_proj_flat.view(num_requests, aligned_seq_len, proj_dim)

        # === Split in BSH format ===
        mixed_qkv, z, b, a = torch.split(
            fused_proj_bsh,
            [
                self.key_dim * 2 + self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            dim=-1,
        )

        # === BSH -> BSC for conv1d ===
        mixed_qkv = mixed_qkv.transpose(1, 2)
        z = z.view(num_requests, aligned_seq_len, self.num_v_heads, self.head_v_dim)

        # Update conv_state
        mixed_qkv_valid = mixed_qkv[:, :, :max_seq_len]
        pre_conv_state = F.pad(
            mixed_qkv_valid,
            (self.conv_kernel_size - max_seq_len, 0),
        )
        conv_state.copy_(pre_conv_state)

        # Apply conv1d
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(
                self.conv1d(mixed_qkv)[:, :, :aligned_seq_len]
            )

        # === BSC -> BSH ===
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Split QKV
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # Reshape
        query = query.view(num_requests, aligned_seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(num_requests, aligned_seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(num_requests, aligned_seq_len, self.num_v_heads, self.head_v_dim)

        # Compute beta and g
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * ge_safe_softplus(
            a.float() + self.dt_bias
        )

        # Repeat heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        # === chunk_gated_delta_rule ===
        with torch.autocast(device_type="npu", dtype=torch.bfloat16, enabled=True):
            chunk_rule_kwargs = {
                "g": g,
                "beta": beta,
                "initial_state": None,
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": True,
            }

            chunk_rule_kwargs["use_solve_triangular"] = (
                self.enable_gdn_solve_triangular
            )

            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                **chunk_rule_kwargs,
            )

            if recurrent_state is not None:
                recurrent_state.copy_(
                    last_recurrent_state.transpose(-1, -2).to(torch.bfloat16)
                )

        # === BSH -> TND ===
        core_attn_out = core_attn_out.view(
            num_requests * aligned_seq_len,
            self.num_v_heads,
            self.head_v_dim,
        )
        z = z.view(
            num_requests * aligned_seq_len,
            self.num_v_heads,
            self.head_v_dim,
        )

        core_attn_out = core_attn_out[flat_idx]
        z = z[flat_idx]

        # reshape for norm
        core_attn_out = core_attn_out.view(-1, self.head_v_dim)
        z = z.view(-1, self.head_v_dim)

        return core_attn_out, z

    def _forward_prefill_with_fused_kernel(
        self,
        fused_proj: torch.Tensor,
        forward_metadata: ForwardMetaData,
    ):
        """
        Prefill with fused NPU chunk gated delta rule kernel.

        Flow:
            TND -> BSH(for conv1d) -> TND -> fused chunk_gdr
        """
        conv_state = self.conv_state
        recurrent_state = self.recurrent_state

        actual_seq_lens = forward_metadata.actual_seq_lengths_q
        cu_seq_lens = forward_metadata.actual_seq_lengths_cu_q

        if cu_seq_lens[0] != 0:
            cu_seq_lens = F.pad(cu_seq_lens, (1, 0))

        num_requests = actual_seq_lens.numel()
        proj_dim = fused_proj.shape[-1]
        max_seq_len = actual_seq_lens.max().item()

        # TND -> BSH
        mixed_qkv_tnd, z, b, a = torch.split(
            fused_proj,
            [
                self.key_dim * 2 + self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            dim=-1,
        )

        # z: [total_token, num_v_heads, head_v_dim]
        z = z.view(-1, self.num_v_heads, self.head_v_dim)

        # Build flat idx
        token_idx = torch.arange(mixed_qkv_tnd.shape[0], device=mixed_qkv_tnd.device)
        batch_idx = torch.bucketize(token_idx, cu_seq_lens[1:], right=False)
        seq_idx = token_idx - cu_seq_lens[batch_idx]
        flat_idx = batch_idx * max_seq_len + seq_idx

        # Only mixed_qkv: TND -> BSH
        mixed_dim = mixed_qkv_tnd.shape[-1]

        mixed_qkv_flat = mixed_qkv_tnd.new_zeros(num_requests * max_seq_len, mixed_dim)
        mixed_qkv_flat.index_copy_(0, flat_idx, mixed_qkv_tnd)

        mixed_qkv = mixed_qkv_flat.view(num_requests, max_seq_len, mixed_dim)

        # Conv1d
        mixed_qkv = mixed_qkv.transpose(1, 2)

        mixed_qkv_valid = mixed_qkv[:, :, :max_seq_len]
        pre_conv_state = F.pad(
            mixed_qkv_valid,
            (self.conv_kernel_size - max_seq_len, 0),
        )
        conv_state.copy_(pre_conv_state)

        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :max_seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)

        # QKV
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.view(num_requests, max_seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(num_requests, max_seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(num_requests, max_seq_len, self.num_v_heads, self.head_v_dim)

        # Only QKV: BSH -> TND
        # z / b / a already TND
        query = query.reshape(-1, query.shape[2], query.shape[3])[flat_idx].contiguous()
        key = key.reshape(-1, key.shape[2], key.shape[3])[flat_idx].contiguous()
        value = value.reshape(-1, value.shape[2], value.shape[3])[flat_idx].contiguous()

        # beta / g
        beta = b.sigmoid().contiguous()

        g = -self.A_log.float().exp() * ge_safe_softplus(
            a.float() + self.dt_bias
        )

        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=-2)
            key = key.repeat_interleave(repeat_factor, dim=-2)

        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
        scale = 1.0 / (self.head_k_dim ** 0.5)

        # Fused chunk GDR
        initial_state = torch.zeros(
            num_requests,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            dtype=torch.bfloat16,
            device=query.device,
        )

        core_attn_out, last_recurrent_state = torch_npu.npu_chunk_gated_delta_rule(
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
            value.to(torch.bfloat16),
            beta=beta.to(torch.bfloat16),
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lens.to(torch.int32),
            scale=scale,
            g=g.to(torch.float32),
        )

        if recurrent_state is not None:
            recurrent_state.copy_(
                last_recurrent_state.to(torch.bfloat16)
            )

        # reshape for norm
        core_attn_out = core_attn_out.view(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        return core_attn_out, z

    def _forward_decode(
        self,
        fused_proj: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """
        Decode path.

        Input:
            fused_proj: [batch_size, proj_dim] (TND)

        Return:
            core_attn_out: [batch_size * num_v_heads, head_v_dim]
        """
        conv_state = self.conv_state
        recurrent_state = self.recurrent_state

        batch_size = hidden_states.shape[0]
        seq_len = 1

        # Split in TND format
        mixed_qkv, z, b, a = torch.split(
            fused_proj,
            [
                self.key_dim * 2 + self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            dim=-1,
        )

        # TND -> BSC
        mixed_qkv = mixed_qkv.view(batch_size, self.conv_dim, seq_len)

        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )

        # BSC -> TND
        mixed_qkv = mixed_qkv.view(batch_size, self.conv_dim)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.view(batch_size, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid().view(batch_size, self.num_v_heads)

        g = -self.A_log.float().exp() * ge_safe_softplus(
            a.float() + self.dt_bias
        )
        g = g.view(batch_size, self.num_v_heads)

        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=1)
            key = key.repeat_interleave(repeat_factor, dim=1)

        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        scale = 1.0 / (self.head_k_dim ** 0.5)

        self._ensure_recurrent_decode_buffers(
            batch_size,
            seq_len,
            hidden_states.device,
        )

        last_recurrent_state = (
            self.tmp if recurrent_state is None else recurrent_state
        )

        core_attn_out = torch_npu.npu_recurrent_gated_delta_rule(
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
            value.to(torch.bfloat16),
            last_recurrent_state.to(torch.bfloat16),
            beta=beta.to(torch.bfloat16),
            scale=scale,
            actual_seq_lengths=self.gdn_actual_seq_lengths,
            ssm_state_indices=self.gdn_ssm_state_indices,
            num_accepted_tokens=None,
            g=g.to(torch.float32),
            gk=None,
        )

        # reshape for norm
        core_attn_out = core_attn_out.view(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        return core_attn_out, z

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        forward_metadata: ForwardMetaData = None,
    ):
        """
        Linear Attention forward with TND input/output.
        """
        hidden_states = apply_mask_to_padding_states(
            hidden_states,
            attention_mask,
        )

        is_prefill = forward_metadata.is_prefill

        # === Step 1: in_proj_fused (directly on TND) ===
        fused_proj = self.in_proj_fused(hidden_states)

        if is_prefill and self.use_npu_chunk_gated_delta_rule:
            core_attn_out, z = self._forward_prefill_with_fused_kernel(
                fused_proj,
                forward_metadata,
            )
        elif is_prefill:
            core_attn_out, z = self._forward_prefill(
                fused_proj,
                hidden_states,
                forward_metadata,
            )
        else:
            core_attn_out, z = self._forward_decode(
                fused_proj,
                hidden_states,
            )

        # === Norm and reshape for out_proj ===
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.view(-1, self.value_dim)

        # === Output projection ===
        if is_prefill:
            output = qwen3_5_prefill_mm_all_reduce(
                self.out_proj,
                core_attn_out.unsqueeze(0),
                self.comm_manager,
                "attn_tp_group",
                self.enable_mm_all_reduce_base,
                forward_metadata,
            )

            used_mm_all_reduce_base = output is not None

            if output is None:
                output = self.out_proj(core_attn_out.unsqueeze(0)).squeeze(0)
            else:
                output = output.squeeze(0)

        else:
            output = self.out_proj(core_attn_out)
            used_mm_all_reduce_base = False

        if self.attn_tp_size > 1 and not used_mm_all_reduce_base:
            dist.all_reduce(
                output,
                group=self.comm_manager.get_group("attn_tp_group"),
            )

        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3_5MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.exe_mode = infer_config.model_config.exe_mode
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_tp_rank = comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
        self.comm_manager = comm_manager
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads_per_rank = max(self.num_key_value_heads // self.attn_tp_size, 1)
        self.scale_fa = 1.0 / math.sqrt(self.head_dim)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attn_intermediate_size = self.num_heads * self.head_dim
        self.attn_intermediate_size_per_rank = self.num_heads_per_rank * self.head_dim
        # block_size will be set in init_cache() to cache_seq_len (not PageAttention's block_size)
        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads * 2,
            total_num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.attn_intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            input_is_parallel=True,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.k_cache = self.v_cache = torch.Tensor([])
        self.cache_unit = self.head_dim * self.num_key_value_heads_per_rank
        # PageAttention cache: will be initialized in init_cache()
        # block_size: fixed PA block size (e.g., 128)
        # block_table: [batch_size, cache_len] mapping request to blocks
        # kv_len_offset: [batch_size, 1] each request's cache region start
        # cache_seq_len: max cache length per request
        self.block_size = None
        self.block_table = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_len: torch.IntTensor = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TND format: hidden_states is [total_tokens, hidden_size]
        q_len = hidden_states.size(0)

        qkv_states = self.merged_qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.num_heads_per_rank * self.head_dim * 2,
                self.num_key_value_heads_per_rank * self.head_dim,
                self.num_key_value_heads_per_rank * self.head_dim,
            ],
            dim=-1,
        )

        # Split query and gate
        query_states, gate = torch.chunk(
            query_states.view(q_len, -1, self.head_dim * 2),
            2,
            dim=-1
        )
        gate = gate.reshape(q_len, -1)

        # TND layout: [T, N, D]
        query_shape = (q_len, self.num_heads_per_rank, self.head_dim)
        key_value_shape = (q_len, self.num_key_value_heads_per_rank, self.head_dim)

        query_states = self.q_norm(query_states.view(query_shape))
        key_states = self.k_norm(key_states.view(key_value_shape))
        value_states = value_states.view(key_value_shape)

        cos, sin = position_embeddings
        rotary_dim = cos.shape[-1]  # 64

        # Slice rotary and passthrough parts -- TND layout (T, N, D)
        q_rot = query_states[..., :rotary_dim]   # (T, N, 64)
        q_pass = query_states[..., rotary_dim:]  # (T, N, 192)
        k_rot = key_states[..., :rotary_dim]     # (T, N, 64)
        k_pass = key_states[..., rotary_dim:]    # (T, N, 192)

        q_rot_4d = q_rot.unsqueeze(0)  # (T, N, 64) -> (1, T, N, 64)
        k_rot_4d = k_rot.unsqueeze(0)  # (T, N, 64) -> (1, T, N, 64)
        cos_4d = cos.unsqueeze(2)      # (1, T, 64) -> (1, T, 1, 64)
        sin_4d = sin.unsqueeze(2)      # (1, T, 64) -> (1, T, 1, 64)

        q_rot_4d = torch_npu.npu_rotary_mul(q_rot_4d, cos_4d, sin_4d, rotary_mode='half')
        k_rot_4d = torch_npu.npu_rotary_mul(k_rot_4d, cos_4d, sin_4d, rotary_mode='half')

        # Remove batch dimension: (1, T, N, 64) -> (T, N, 64)
        q_rot = q_rot_4d.squeeze(0)
        k_rot = k_rot_4d.squeeze(0)

        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)

        k_cache, v_cache = self.k_cache, self.v_cache
        if not k_cache.numel() or not v_cache.numel():
            raise RuntimeError("A BUG: k_cache or v_cache are not initialized properly.")

        is_prefill = forward_metadata.is_prefill
        batch_size = forward_metadata.actual_seq_lengths_kv.shape[0]

        if is_prefill:
            # Prefill: forward_metadata.kv_len already contains slot_mapping
            # computed by execution_engine: kv_len = batch_idx * cache_seq_len + position
            slot_mapping = slot_mapping.long()
        else:
            slot_mapping = kv_len

        # KV Cache update using npu_scatter_nd_update_
        # k_cache/v_cache: [kv_cache_num_block, block_size, num_heads, head_dim]
        # Flatten to [total_slots, num_heads, head_dim] for scatter update
        slot_mapping_flat = slot_mapping.view(-1)

        torch_npu.npu_scatter_nd_update_(
            k_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping_flat.view(-1, 1),
            key_states,
        )
        torch_npu.npu_scatter_nd_update_(
            v_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping_flat.view(-1, 1),
            value_states,
        )

        # block_table: [batch_size, cache_len], slice to actual batch_size
        block_table = self.block_table[:batch_size] if self.block_table is not None else None

        fa_ops = torch.ops.npu
        if not is_prefill and self.enable_gegraph:
            fa_ops = torchair.ops

        if not is_prefill and self.enable_npugraph_ex:
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
        else:
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q

        # Use FA v2 with TND layout
        sparse_mode = 3 if is_prefill else 0

        k_cache_fa = k_cache.view(*k_cache.shape[:2], -1)
        v_cache_fa = v_cache.view(*v_cache.shape[:2], -1)
        attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
            query_states,
            k_cache_fa,
            v_cache_fa,
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            softmax_scale=self.scale_fa,
            input_layout="TND",
            sparse_mode=sparse_mode,
            atten_mask=forward_metadata.attention_mask if is_prefill else None,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            block_size=self.block_size,
        )

        attn_output = attn_output.reshape(q_len, self.attn_intermediate_size_per_rank)
        attn_output = attn_output * torch.sigmoid(gate)

        fused_attn_output = qwen3_5_prefill_mm_all_reduce(
            self.o_proj,
            attn_output,
            self.comm_manager,
            "attn_tp_group",
            self.enable_mm_all_reduce_base,
            forward_metadata,
        )
        if fused_attn_output is None:
            attn_output = self.o_proj(attn_output)
            used_mm_all_reduce_base = False
        else:
            attn_output = fused_attn_output
            used_mm_all_reduce_base = True
        if self.attn_tp_size > 1 and not used_mm_all_reduce_base:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))
        return attn_output


class Qwen3_5MoeMLP(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
        self.comm_manager = comm_manager
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiLUAndMul()

    def forward(self, x, forward_metadata: ForwardMetaData = None):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        down_proj = qwen3_5_prefill_mm_all_reduce(
            self.down_proj,
            x,
            self.comm_manager,
            "dense_tp_group",
            self.enable_mm_all_reduce_base,
            forward_metadata,
        )
        used_mm_all_reduce_base = down_proj is not None
        if down_proj is None:
            down_proj = self.down_proj(x)
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if self.dense_tp_size > 1 and not used_mm_all_reduce_base:
            dist.all_reduce(down_proj, group=self.comm_manager.get_group("dense_tp_group"))
        return down_proj


class Qwen3_5MoeExperts(FusedMoEGMM):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = "",
    ):
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.moe_tp_rank = comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0
        self.moe_ep_rank = comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        super().__init__(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            bias=False,
            quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=self.moe_tp_rank,
            ep_size=self.moe_ep_size,
            ep_rank=self.moe_ep_rank,
            prefix=prefix,
        )

    def _map_global_expert_id(self, expert_id: int) -> int | None:
        if self.moe_ep_size <= 1:
            return expert_id

        local_start = self.moe_ep_rank * self.experts_per_rank
        local_end = local_start + self.experts_per_rank
        if expert_id < local_start or expert_id >= local_end:
            return None
        return expert_id - local_start

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        local_expert_id = self._map_global_expert_id(expert_id)
        if local_expert_id is None:
            return

        if self.moe_ep_size <= 1:
            super().weight_loader(param, loaded_weight, weight_name, shard_id, local_expert_id)
            return

        original_ep_size = self.ep_size
        try:
            # FusedMoEGMM only slices experts for ep>1,tp=1. Convert to a local expert id here so mixed tp+ep
            # ranks load the same tensor-parallel shard while keeping only their local expert block.
            self.ep_size = 1
            super().weight_loader(param, loaded_weight, weight_name, shard_id, local_expert_id)
        finally:
            self.ep_size = original_ep_size


class Qwen3_5MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)

        # router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        # router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        # router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        # router_top_value = router_top_value.to(router_logits.dtype)
        # router_scores = router_top_value
        # return router_logits, router_scores, router_indices

        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k_softmax(
            router_logits.to(torch.float32), None, k=self.top_k
        )
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)

        return _, topk_weight, topk_idx



class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.moe_tp_rank = (
            comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0
        )
        self.moe_ep_rank = (
            comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        )
        self.enable_decode_moe_dispatch_combine_v2 = infer_config.model_config.custom_params.get(
            "enable_decode_moe_dispatch_combine_v2", False
        )
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.local_expert_start = self.moe_ep_rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank
        self.comm_manager = comm_manager
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config, infer_config, comm_manager, prefix=f"{prefix}.experts")
        self.shared_expert = Qwen3_5MoeMLP(
            config,
            infer_config,
            comm_manager,
            intermediate_size=config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=selected_experts.to(torch.int32),
            active_num=selected_experts.shape[0] * selected_experts.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )
        return expanded_x, expanded_row_idx, tokens_per_expert

    def _finalize_routing(
        self,
        hidden_states_ordered_by_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
    ) -> torch.Tensor:
        return torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=None,
            skip2=None,
            bias=None,
            scales=routing_weights.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )

    def _dispatch_to_ep_group(
        self,
        tokens_per_expert: torch.Tensor,
        expanded_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)

        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        output_splits = [int(item) for item in combine_tokens[0].cpu().tolist()]
        input_splits = [int(item) for item in combine_tokens[1].cpu().tolist()]
        total_tokens = sum(output_splits)
        gathered_tokens = expanded_x.new_empty((total_tokens, expanded_x.shape[1]))
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, input_splits, output_splits

    def _run_experts_tp_only(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        expanded_x, expanded_row_idx, tokens_per_expert = self._init_routing(hidden_states, selected_experts)
        hidden_states_ordered_by_experts = self.experts(expanded_x, tokens_per_expert, group_list_type=1)
        expert_output = self._finalize_routing(hidden_states_ordered_by_experts, routing_weights, expanded_row_idx)
        if self.moe_tp_size > 1:
            dist.all_reduce(expert_output, group=self.comm_manager.get_group("moe_tp_group"))
        return expert_output

    def _run_experts_ep(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        expanded_x, expanded_row_idx, tokens_per_expert = self._init_routing(hidden_states, selected_experts)
        tokens_per_expert_group, gathered_tokens, input_splits, output_splits = self._dispatch_to_ep_group(
            tokens_per_expert,
            expanded_x,
        )

        hidden_states_ordered_by_experts, _, gathered_ids_unsort, tokens_per_local_expert = (
            torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.moe_ep_size, -1),
            )
        )
        hidden_states_ordered_by_experts = self.experts(
            hidden_states_ordered_by_experts,
            tokens_per_local_expert,
            group_list_type=1,
        )
        new_x = torch.index_select(
            hidden_states_ordered_by_experts,
            0,
            gathered_ids_unsort.float().argsort().int(),
        )

        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        gathered_tokens = new_x.new_empty(expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

        expert_output = self._finalize_routing(gathered_tokens, routing_weights, expanded_row_idx)
        if self.moe_tp_size > 1:
            dist.all_reduce(expert_output, group=self.comm_manager.get_group("moe_tp_group"))
        return expert_output

    def _can_use_decode_dispatch_combine_v2(self, forward_metadata: ForwardMetaData | None) -> bool:
        return (
            self.enable_decode_moe_dispatch_combine_v2
            and forward_metadata is not None
            and not forward_metadata.is_prefill
            and self.moe_ep_size > 1
            and self.moe_tp_size == 1
        )

    def _run_experts_dispatch_combine_v2(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group")
        topk_ids = selected_experts.to(torch.int32)
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "x_active_mask": None,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "scales": None,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": self.moe_ep_rank,
            "group_tp": "",
            "tp_world_size": 0,
            "tp_rank_id": 0,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "quant_mode": 0,
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, _, assist_info_for_combine, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]
        expand_scales = output[6] if len(output) > 6 else None

        hidden_states_ordered_by_experts = self.experts(
            expand_x,
            expert_token_num,
            group_list_type=1,
        )

        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "expert_ids": topk_ids,
            "assist_info_for_combine": assist_info_for_combine,
            "expert_scales": routing_weights.to(torch.float32),
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            "x_active_mask": None,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": self.moe_ep_rank,
            "group_tp": "",
            "tp_world_size": 0,
            "tp_rank_id": 0,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "comm_quant_mode": 0,
        }
        if expand_scales is not None:
            combine_args["expand_scales"] = expand_scales
        return torch_npu.npu_moe_distribute_combine_v2(**combine_args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TND format: hidden_states is [total_tokens, hidden_dim]
        hidden_dim = hidden_states.shape[-1]
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped, forward_metadata=forward_metadata)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        if self._can_use_decode_dispatch_combine_v2(forward_metadata):
            expert_output = self._run_experts_dispatch_combine_v2(
                hidden_states_reshaped,
                routing_weights,
                selected_experts,
            )
        elif self.moe_ep_size > 1:
            expert_output = self._run_experts_ep(hidden_states_reshaped, routing_weights, selected_experts)
        else:
            expert_output = self._run_experts_tp_only(hidden_states_reshaped, routing_weights, selected_experts)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        # TND format: output stays as [total_tokens, hidden_dim]
        return expert_output


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    @property
    def norm_weight(self):
        return 1.0 + self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def ln(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3_5Moe is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * self.norm_weight.float()
        return output.type_as(x)

    def ln_npu(self, x):
        return torch_npu.npu_rms_norm(x, self.norm_weight, self.eps)[0]

    def forward(self, x, *args):
        if len(args) == 0:
            return self.ln_npu(x)
        elif len(args) == 1 and args[0] is None:
            return self.ln_npu(x), x
        elif len(args) == 1:
            residual = args[0]
            y, _, residual = torch_npu.npu_add_rms_norm(residual, x, self.norm_weight, self.eps)
            return y, residual
        else:
            raise NotImplementedError(
                f"insupportable Qwen3_5MoeRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(
                config,
                infer_config,
                comm_manager,
                layer_idx,
                prefix=f"model.layers.{layer_idx}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(
                config,
                infer_config,
                comm_manager,
                layer_idx,
                prefix=f"model.layers.{layer_idx}.self_attn",
            )
        self.mlp = Qwen3_5MoeSparseMoeBlock(
            config,
            infer_config,
            comm_manager,
            prefix=f"model.layers.{layer_idx}.mlp",
        )
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_len: torch.IntTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_residual: torch.Tensor | None = None,
        forward_metadata: ForwardMetaData = None,
        slot_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                forward_metadata=forward_metadata,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_len=kv_len,
                position_embeddings=position_embeddings,
                forward_metadata=forward_metadata,
                slot_mapping=slot_mapping
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, forward_metadata=forward_metadata)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states

        return residual, hidden_states





@dataclass
class Qwen3_5MoeModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our
        [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class Qwen3_5MoeCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our
        [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    router_logits: tuple[torch.FloatTensor] | None = None
    aux_loss: torch.FloatTensor | None = None


class Qwen3_5MoeTextModel(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, infer_config: InferenceConfig, comm_manager: CommManager):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.vocab_size_per_rank = config.vocab_size // self.embed_tp_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            getattr(config, "torch_dtype", torch.bfloat16),
            tp_size=self.embed_tp_size,
            tp_rank=comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0,
        )

        batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        cache_seq_len = self.infer_config.data_config.input_truncated_len + \
            self.infer_config.scheduler_config.max_new_tokens
        self.block_size = self.infer_config.scheduler_config.block_size
        self.pa_max_length = align_up(cache_seq_len, 128)
        self.cache_len = self.pa_max_length // self.block_size
        self.kv_cache_num_block = self.cache_len * batch_size_per_rank
        self.kv_len_offset = torch.arange(
            0,
            batch_size_per_rank * self.pa_max_length,
            self.pa_max_length,
            dtype=torch.int64,
            device="npu",
        ).view(-1, 1)

        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, infer_config, comm_manager, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        forward_metadata: ForwardMetaData = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        position_ids = position_ids.view(-1).long()

        if inputs_embeds is None:
            if self.embed_tp_size > 1:
                embed_rank = self.comm_manager.get_rank("embed_tp_group")
                new_input_ids = input_ids - embed_rank * self.vocab_size_per_rank
                mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
                new_input_ids_per_rank = new_input_ids * mask
                inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
                dist.all_reduce(inputs_embeds, group=self.comm_manager.get_group("embed_tp_group"))
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        residual = None

        seq_len = forward_metadata.actual_seq_lengths_kv
        is_prefill = forward_metadata.is_prefill
        slot_mapping = self.get_slot_mapping(seq_len, is_prefill, position_ids.device)
        kv_len = forward_metadata.kv_len
        if not is_prefill:
            batch_size = seq_len.shape[0]
            kv_len = kv_len.long().view(-1) + self.kv_len_offset[:batch_size].view(-1)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            residual, hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_residual=residual,
                forward_metadata=forward_metadata,
                kv_len=kv_len,
                slot_mapping=slot_mapping,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        # TND format: get last token for each sequence using index_select
        cu_seq_lens_q = forward_metadata.actual_seq_lengths_cu_q if forward_metadata else None
        if cu_seq_lens_q is None:
            raise RuntimeError("actual_seq_lengths_cu_q is required.")

        if forward_metadata.is_prefill:
            seq_index = cu_seq_lens_q - 1
            hidden_states = torch.index_select(hidden_states, 0, seq_index)
            hidden_states = hidden_states.view(seq_index.numel(), 1, hidden_states.size(-1))
        else:
            hidden_states = hidden_states.view(hidden_states.shape[0], 1, hidden_states.shape[-1])

        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states
        )

    def get_slot_mapping(self, kv_len, is_prefill, device):
        if not is_prefill:
            return None
        all_tensors = []
        for i, seq_len in enumerate(kv_len):
            new_index = torch.arange(
                self.pa_max_length * i, seq_len.item() + self.pa_max_length * i,
                dtype=kv_len.dtype, device=device
                )
            all_tensors.append(new_index)
        return torch.cat(all_tensors)

    def _update_linear_attn_mask(self, attention_mask, forward_metadata):
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if (not forward_metadata.is_prefill) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None
        return linear_attn_mask


class Qwen3_5MoeForCausalLM(nn.Module):
    def __init__(self, config, infer_config, comm_manager, prefix: str = ""):
    # def __init__(self, config):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        # Parallel config
        self.world_size = infer_config.parallel_config.world_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size

        # Initialize communication groups before creating model components
        self.init_parallel_comm_group()

        self.model = Qwen3_5MoeTextModel(config, infer_config, comm_manager)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
            prefix="lm_head",
        )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.decode_uses_explicit_metadata = True

        # Initialize weights and apply final processing
        # self.post_init()

    def init_parallel_comm_group(self):
        """Register all communication groups required by the model."""
        # Attention TP group
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )

        # Embedding TP group
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=self.world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
        )

        # LM Head TP group
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=self.world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
        )

        # Dense MLP TP group
        if self.dense_tp_size > 1:
            self.comm_manager.register_group(
                name="dense_tp_group",
                group_num=self.world_size // self.dense_tp_size,
                group_size=self.dense_tp_size,
            )

        # MoE TP group
        if self.moe_tp_size > 1:
            self.comm_manager.register_group(
                name="moe_tp_group",
                group_num=self.world_size // self.moe_tp_size,
                group_size=self.moe_tp_size,
            )

        # MoE EP group
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
            )

        # MoE EP group for MC2 (dispatch/combine fusion)
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            self.comm_manager.register_group(
                name="moe_ep_group_mc2",
                group_num=self.world_size // self.moe_ep_size,
                group_size=self.moe_ep_size,
                group_stride=self.world_size // self.moe_ep_size,
                return_name=True,
                allow_physical_reuse=False,
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,
    ):
        # TND format
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            **kwargs,
        )

        hidden_states = hidden_states.last_hidden_state
        logits = self.lm_head(hidden_states)

        if self.lmhead_tp_size > 1:
            gathered_logits = [torch.empty_like(logits) for _ in range(self.lmhead_tp_size)]
            dist.all_gather(gathered_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            logits = torch.cat(gathered_logits, dim=-1)

        return logits.float()

    def init_cache(self, device):
        """Initialize KV cache using standard PageAttention format (block_size=128)."""
        # batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        # cache_seq_len = self.infer_config.data_config.input_truncated_len + \
        #     self.infer_config.scheduler_config.max_new_tokens
        # dtype = self.config.torch_dtype

        # # Standard PageAttention: fixed block_size (FA optimized size)
        # # block_size can be configured, default 128
        # block_size = self.infer_config.model_config.custom_params.get("pa_block_size", 128)

        # # Each request needs cache_len blocks
        # cache_len = cache_seq_len // block_size
        # if cache_seq_len % block_size != 0:
        #     cache_len += 1  # round up

        # # Total blocks in cache pool
        # kv_cache_num_block = cache_len * batch_size

        # # block_table: [batch_size, cache_len]
        # # Each request occupies cache_len consecutive blocks
        # # request 0: blocks [0, 1, ..., cache_len-1]
        # # request 1: blocks [cache_len, cache_len+1, ..., 2*cache_len-1]
        # block_table = torch.arange(
        #     0, batch_size * cache_len, dtype=torch.int32, device=device
        # ).reshape(batch_size, cache_len)

        # # kv_len_offset: [batch_size, 1]
        # # Each request's cache region starts at: batch_idx * cache_seq_len
        # # Used for decode slot_mapping calculation
        # kv_len_offset = torch.arange(
        #     0, batch_size * cache_seq_len, cache_seq_len,
        #     dtype=torch.int64, device=device
        # ).view(-1, 1)


        batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        cache_seq_len = self.infer_config.data_config.input_truncated_len + \
            self.infer_config.scheduler_config.max_new_tokens
        dtype = self.config.torch_dtype

        # 使用框架配置的 block_size，确保一致性
        block_size = self.infer_config.scheduler_config.block_size  # 与 engine 保持一致

        # 确保 cache_seq_len 是 block_size 的整数倍
        cache_len = (cache_seq_len + block_size - 1) // block_size  # round up
        aligned_cache_seq_len = cache_len * block_size  # 对齐后的值

        # block_table: [batch_size, cache_len]
        block_table = torch.arange(
            0, batch_size * cache_len, dtype=torch.int32, device=device
        ).reshape(batch_size, cache_len)

        # kv_len_offset: [batch_size, 1] - 使用对齐后的 cache_seq_len
        kv_len_offset = torch.arange(
            0, batch_size * aligned_cache_seq_len, aligned_cache_seq_len,
            dtype=torch.int64, device=device
        ).view(-1, 1)

        # k_cache/v_cache: [kv_cache_num_block, block_size, num_heads, head_dim]
        kv_cache_num_block = cache_len * batch_size

        for layer_idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            if layer.layer_type == "linear_attention":
                # LinearAttention state: batch-indexed BSH format
                layer.linear_attn.recurrent_state = torch.zeros(
                    (batch_size, layer.linear_attn.num_v_heads,
                     layer.linear_attn.head_v_dim, layer.linear_attn.head_k_dim),
                    dtype=dtype, device=device
                )
                layer.linear_attn.conv_state = torch.zeros(
                    (batch_size, layer.linear_attn.conv_dim,
                     layer.linear_attn.conv_kernel_size),
                    dtype=dtype, device=device
                )
            else:
                # FullAttention cache: PageAttention format
                # Shape: [kv_cache_num_block, block_size, num_heads, head_dim]
                num_heads = layer.self_attn.num_key_value_heads_per_rank
                head_dim = layer.self_attn.head_dim

                layer.self_attn.k_cache = torch.empty(
                    (kv_cache_num_block, block_size, num_heads, head_dim),
                    dtype=dtype, device=device
                )
                layer.self_attn.v_cache = torch.empty(
                    (kv_cache_num_block, block_size, num_heads, head_dim),
                    dtype=dtype, device=device
                )

                # Set block_table, block_size
                layer.self_attn.block_table = block_table
                layer.self_attn.block_size = block_size


    def load_weights(self, weights):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_fused", "in_proj_qkv", "qkv"),
            ("in_proj_fused", "in_proj_z", "z"),
            ("in_proj_fused", "in_proj_b", "b"),
            ("in_proj_fused", "in_proj_a", "a"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()

        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts
        )

        for name, loaded_weight in weights:
            if name.startswith("mtp.") or name.startswith("model.visual."):
                continue

            # Normalization
            if "model.language_model." in name:
                norm_name = name.replace("model.language_model.", "model.", 1)
            else:
                norm_name = name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if (
                    f".self_attn.{weight_name}." not in norm_name
                    and f".linear_attn.{weight_name}." not in norm_name
                    and f".shared_expert.{weight_name}." not in norm_name
                ):
                    continue

                name_mapped = norm_name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                break
            else:
                if norm_name.endswith("mlp.experts.gate_up_proj"):
                    param_name = norm_name.replace("gate_up_proj", "w13_weight")
                    if param_name not in params_dict:
                        logger.warning(f"[SKIP] No match in params_dict: {norm_name} (Original: {name})")
                        continue
                    param = params_dict[param_name]
                    intermediate_size = loaded_weight.shape[1] // 2
                    for expert_id in range(loaded_weight.shape[0]):
                        param.weight_loader(
                            param,
                            loaded_weight[expert_id, :intermediate_size, :],
                            param_name,
                            shard_id="w1",
                            expert_id=expert_id,
                        )
                        param.weight_loader(
                            param,
                            loaded_weight[expert_id, intermediate_size:, :],
                            param_name,
                            shard_id="w3",
                            expert_id=expert_id,
                        )
                    loaded_params.add(param_name)
                    continue

                if norm_name.endswith("mlp.experts.down_proj"):
                    param_name = norm_name.replace("down_proj", "w2_weight")
                    if param_name not in params_dict:
                        logger.warning(f"[SKIP] No match in params_dict: {norm_name} (Original: {name})")
                        continue
                    param = params_dict[param_name]
                    for expert_id in range(loaded_weight.shape[0]):
                        param.weight_loader(
                            param,
                            loaded_weight[expert_id],
                            param_name,
                            shard_id="w2",
                            expert_id=expert_id,
                        )
                    loaded_params.add(param_name)
                    continue

                # MoE expert 逻辑 (基于 norm_name)
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in norm_name:
                        continue

                    name_mapped = norm_name.replace(weight_name, param_name)

                    if name_mapped not in params_dict:
                        continue
                    # 命中 MoE
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader

                    weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )

                    loaded_params.add(name_mapped)
                    is_expert_weight = True
                    break

                if is_expert_weight:
                    continue

                # 普通权重逻辑 (基于 norm_name)
                if norm_name not in params_dict:
                    logger.warning(f"[SKIP] No match in params_dict: {norm_name} (Original: {name})")
                    continue

                param = params_dict[norm_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

                loaded_params.add(norm_name)

        return loaded_params

    def process_weights_after_loading(self):
        enable_mm_all_reduce_base = self.infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
        mm_all_reduce_suffixes = (
            ".linear_attn.out_proj",
            ".self_attn.o_proj",
            ".mlp.shared_expert.down_proj",
        )
        for name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                use_mm_all_reduce_base = (
                    enable_mm_all_reduce_base
                    and isinstance(module, RowParallelLinear)
                    and isinstance(quant_method, UnquantizedLinearMethod)
                    and module.tp_size > 1
                    and module.input_is_parallel
                    and module.bias is None
                    and not module.skip_bias_add
                    and (
                        name.endswith(mm_all_reduce_suffixes)
                        or name.endswith(".mlp.down_proj")
                    )
                )
                quant_method.process_weights_after_loading(
                    module,
                    is_nz=self.infer_config.model_config.enable_weight_nz and not use_mm_all_reduce_base,
                )

    __all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeTextModel",
]