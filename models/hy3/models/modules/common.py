# Adapted from transformers/models/hy_v3/modeling_hy_v3.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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

"""Common layers and sequence-parallel helpers for HYV3."""

import glob
import os
import sysconfig
from dataclasses import replace
from functools import lru_cache

import torch
import torch.distributed as dist
import torch_npu
from torch import nn

from module.linear import LinearBase

from ..configuration_hy_v3 import HYV3Config


def _load_swiglu_group_quant_op() -> None:
    """Load the installed swiglu group-quant operator when it is available."""
    if hasattr(torch.ops.custom, "npu_swiglu_group_quant"):
        return

    custom_ops_dir = os.path.join(sysconfig.get_paths()["purelib"], "custom_ops")
    shared_objects = glob.glob(os.path.join(custom_ops_dir, "custom_ops_lib*.so"))
    if shared_objects:
        torch.ops.load_library(shared_objects[0])


_load_swiglu_group_quant_op()


def is_sequence_parallel_enabled(infer_config) -> bool:
    """Return the explicit sequence-parallel setting from the model config."""
    if infer_config is None:
        return False
    model_config = getattr(infer_config, "model_config", None)
    custom_params = getattr(model_config, "custom_params", None)
    return bool(custom_params and custom_params.get("enable_sp", False))


def quantize_sequence_parallel_transport(
    tensor: torch.Tensor,
    gmm_quant_mode: str,
    target_linear: LinearBase | None = None,
):
    """Quantize an activation for SP or DP-TP-DP transport.

    MXFP4 uses dynamic MXFP8 values with per-group scales. FP8 uses the target
    linear layer's static per-tensor input scale and therefore transports only
    values.
    """
    if gmm_quant_mode in ("w4a8mxfloat4", "w4a8mx"):
        return torch_npu.npu_dynamic_mx_quant(tensor, dst_type=torch.float8_e4m3fn)
    if gmm_quant_mode == "w8a8float8":
        if target_linear is None:
            raise ValueError("FP8 sequence-parallel transport requires target_linear")
        quantized = torch_npu.npu_quantize(tensor, target_linear.input_scale, None, torch.float8_e4m3fn, -1, True)
        return quantized, None
    raise NotImplementedError(
        "SP transport quantization is not implemented for "
        f"gmm_quant_mode={gmm_quant_mode!r}; supported modes are MXFP4 and FP8"
    )


def equal_all_to_all(
    tensor: torch.Tensor,
    group,
    group_size: int,
) -> torch.Tensor:
    """Run an equal-split all-to-all that is capturable by npugraph_ex."""
    flat_tensor = tensor.reshape(-1)
    chunk_size = flat_tensor.shape[0] // group_size
    split_sizes = [chunk_size] * group_size
    output = torch.empty_like(flat_tensor)
    dist.all_to_all_single(
        output,
        flat_tensor,
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
        group=group,
    )
    return output.view_as(tensor)


def build_pad_aware_prefill_metadata(
    forward_metadata,
    slot_mapping,
    block_table,
    pad_len: int,
    prompt_tokens: int,
):
    """Append an SP alignment pad as an independent dummy request segment.

    The dummy segment writes to null block 0, which is reserved by the block
    pool. Real request metadata remains unchanged and callers discard the dummy
    output after attention.
    """
    cumulative_query_lengths = forward_metadata.actual_seq_lengths_cu_q
    key_value_lengths = forward_metadata.actual_seq_lengths_kv
    padded_query_lengths = torch.cat(
        [
            cumulative_query_lengths,
            cumulative_query_lengths.new_tensor([prompt_tokens + pad_len]),
        ]
    )
    padded_key_value_lengths = torch.cat([key_value_lengths, key_value_lengths.new_tensor([pad_len])])
    padded_metadata = replace(
        forward_metadata,
        actual_seq_lengths_cu_q=padded_query_lengths,
        actual_seq_lengths_kv=padded_key_value_lengths,
    )

    padded_slot_mapping = dict(slot_mapping) if slot_mapping else slot_mapping
    if slot_mapping:
        for attention_type, mapping in slot_mapping.items():
            flat_mapping = mapping.view(-1)
            dummy_slots = torch.arange(
                pad_len,
                device=flat_mapping.device,
                dtype=flat_mapping.dtype,
            )
            padded_slot_mapping[attention_type] = torch.cat([flat_mapping, dummy_slots])

    padded_block_table = dict(block_table) if block_table else block_table
    if block_table:
        for attention_type, table in block_table.items():
            dummy_row = table.new_zeros((1, table.shape[1]))
            padded_block_table[attention_type] = torch.cat([table, dummy_row], dim=0)

    return padded_metadata, padded_slot_mapping, padded_block_table


@lru_cache(maxsize=1)
def ensure_qkv_fused_kscale_registered() -> None:
    """Import the optional fused QKV operator once to trigger registration."""
    from cann_ops_transformer.ops import (
        qkv_rms_norm_rope_cache_with_k_scale as _register_qkv_fused_kscale,  # noqa: F401
    )


class HYV3RMSNorm(nn.Module):
    """HYV3 RMSNorm backed by NPU fused kernels."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, *args):
        """Normalize hidden states, optionally fusing the residual addition."""
        if not args:
            return torch_npu.npu_rms_norm(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )[0]
        if len(args) == 1 and args[0] is None:
            normalized = torch_npu.npu_rms_norm(
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )[0]
            return normalized, hidden_states
        if len(args) == 1:
            normalized, _, residual = torch_npu.npu_add_rms_norm(
                args[0],
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )
            return normalized, residual
        raise NotImplementedError(
            "HYV3RMSNorm accepts hidden_states and at most one residual tensor; "
            f"received {len(args) + 1} positional arguments"
        )


class HYV3RotaryEmbedding(nn.Module):
    """Rotary position embedding tables for packed HYV3 attention."""

    def __init__(
        self,
        config: HYV3Config,
        max_position_embeddings: int = 2048,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = config.rope_parameters.get("rope_theta", config.default_theta)

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        max_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_seq_len is not None and max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=hidden_states.device, dtype=hidden_states.dtype)

        if position_ids.dim() != 1:
            raise RuntimeError("HYV3 expects packed 1D position_ids")

        cosine = self.cos_cached[position_ids]
        sine = self.sin_cached[position_ids]
        if hidden_states.dim() == 2:
            cosine = cosine.unsqueeze(1)
            sine = sine.unsqueeze(1)
        elif hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
            cosine = cosine.view(batch_size, seq_len, 1, self.dim)
            sine = sine.view(batch_size, seq_len, 1, self.dim)
        else:
            raise RuntimeError(f"unsupported HYV3 RoPE input rank: {hidden_states.dim()}")

        return (cosine.to(dtype=hidden_states.dtype), sine.to(dtype=hidden_states.dtype),)

    def get_cos_sin_table(self, max_seq_len: int | None = None) -> torch.Tensor:
        if max_seq_len is not None and max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return self.cos_sin_cached

    def _set_cos_sin_cache(self, seq_len: int, device, dtype) -> None:
        self.max_seq_len_cached = seq_len
        positions = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, self.inv_freq.to(positions.device))
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embeddings.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", embeddings.sin().to(dtype), persistent=False)
        self.register_buffer(
            "cos_sin_cached",
            torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(torch.float32),
            persistent=False,
        )


__all__ = [
    "HYV3RMSNorm",
    "HYV3RotaryEmbedding",
    "build_pad_aware_prefill_metadata",
    "ensure_qkv_fused_kscale_registered",
    "equal_all_to_all",
    "is_sequence_parallel_enabled",
    "quantize_sequence_parallel_transport",
]
