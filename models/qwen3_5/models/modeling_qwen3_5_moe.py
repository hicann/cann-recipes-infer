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
import json
import math
import os
import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import List, Optional, Iterable

import torch
import torch_npu
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.types import _dtype
import torchair
import cann_ops_transformer.ops

from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import (
    CacheEntry,
    LayerCacheInfo,
    MambaCacheEntry,
    ModelCacheInfo,
)
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import calc_moe_hccl_buffer_size
from executor.utils.forward_metadata import ForwardMetaData
from module.fuse_moe_gmm import FusedMoEGMM, UnquantizedFusedMoEGMMMethod
from module.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
    VocabParallelEmbedding,
)
from module.quantization import get_quant_config
from module.quantization.mxfp8 import BLOCK_K, MxFp8Config, MxFp8LinearMethod, MxFp8MoEGMMMethod
from module.utils import set_weight_attrs

from .configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

logger = logging.getLogger(__name__)

local_rank = os.environ.get('LOCAL_RANK', '0')

_PLATFORM_VERSION: str | None = None

qwen3_5_use_aiv_all_reduce = os.environ.get("HCCL_OP_EXPANSION_MODE") == "AIV"


def qwen3_5_all_reduce(tensor: torch.Tensor, group) -> None:
    if qwen3_5_use_aiv_all_reduce and tensor.dtype == torch.bfloat16:
        tensor_fp32 = tensor.to(torch.float32)
        dist.all_reduce(tensor_fp32, group=group)
        tensor.copy_(tensor_fp32.to(tensor.dtype))
    else:
        dist.all_reduce(tensor, group=group)


def _qwen3_5_token_shard_supported(infer_config: InferenceConfig) -> bool:
    """Return whether the configured parallel layout supports token sharding."""
    parallel_config = infer_config.parallel_config
    return (
        parallel_config.attn_tp_size > 1
        and parallel_config.moe_tp_size == 1
        and parallel_config.moe_ep_size == parallel_config.attn_tp_size
        and parallel_config.shared_tp_size == 1
        and not infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
    )


def qwen3_5_attention_reduce(
    output: torch.Tensor,
    attn_tp_size: int,
    group,
    enable_token_shard: bool = False,
):
    """Keep token-sharded activations for EP-only attention/TP layouts."""
    token_shard_active = (
        attn_tp_size > 1
        and enable_token_shard
        and output.shape[0] % attn_tp_size == 0
    )
    if token_shard_active:
        local_shape = (
            output.shape[0] // attn_tp_size,
            output.shape[1],
        )
        if qwen3_5_use_aiv_all_reduce and output.dtype == torch.bfloat16:
            output_fp32 = output.to(torch.float32)
            local_output_fp32 = output_fp32.new_empty(*local_shape)
            dist.reduce_scatter_tensor(
                local_output_fp32,
                output_fp32,
                group=group,
            )
            return local_output_fp32.to(output.dtype)

        local_output = output.new_empty(*local_shape)
        dist.reduce_scatter_tensor(local_output, output, group=group)
        return local_output
    qwen3_5_all_reduce(output, group=group)
    return output


def _build_qwen3_5_pad_prefill_metadata(forward_metadata, pad_len):
    """Add a dummy request so FA sees matching padded Q and KV lengths."""
    def append_length(value, length):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return torch.cat([value, value.new_tensor([length])])
        return [*value, length]

    def append_cumulative(value, length):
        if value is None:
            return None
        last = value[-1].item() if isinstance(value, torch.Tensor) else value[-1]
        return append_length(value, last + length)

    padded_actual_q = append_length(forward_metadata.actual_seq_lengths_q, pad_len)
    padded_actual_kv = append_length(forward_metadata.actual_seq_lengths_kv, pad_len)
    padded_cu_q = append_cumulative(forward_metadata.actual_seq_lengths_cu_q, pad_len)
    padded_cu_kv = append_cumulative(forward_metadata.actual_seq_lengths_cu_kv, pad_len)
    padded_list_q = append_length(forward_metadata.actual_seq_lengths_list_q, pad_len)
    padded_list_kv = append_length(forward_metadata.actual_seq_lengths_list_kv, pad_len)
    padded_cu_list_q = append_cumulative(forward_metadata.actual_seq_lengths_cu_list_q, pad_len)
    padded_cu_list_kv = append_cumulative(forward_metadata.actual_seq_lengths_cu_list_kv, pad_len)
    slot_mapping = forward_metadata.slot_mapping
    new_slot_mapping = dict(slot_mapping) if slot_mapping else slot_mapping
    if slot_mapping:
        for key, mapping in slot_mapping.items():
            mapping = mapping.reshape(-1)
            dummy_slots = torch.arange(pad_len, device=mapping.device, dtype=mapping.dtype)
            new_slot_mapping[key] = torch.cat([mapping, dummy_slots])

    block_table = forward_metadata.block_table
    new_block_table = dict(block_table) if block_table else block_table
    if block_table:
        for key, table in block_table.items():
            dummy_row = table.new_zeros((1, table.shape[1]))
            new_block_table[key] = torch.cat([table, dummy_row], dim=0)
    padded_metadata = replace(
        forward_metadata,
        actual_seq_lengths_q=padded_actual_q,
        actual_seq_lengths_kv=padded_actual_kv,
        actual_seq_lengths_cu_q=padded_cu_q,
        actual_seq_lengths_cu_kv=padded_cu_kv,
        actual_seq_lengths_list_q=padded_list_q,
        actual_seq_lengths_list_kv=padded_list_kv,
        actual_seq_lengths_cu_list_q=padded_cu_list_q,
        actual_seq_lengths_cu_list_kv=padded_cu_list_kv,
        slot_mapping=new_slot_mapping,
        block_table=new_block_table,
    )
    return padded_metadata


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


LINEAR_TARGET_MODULES = (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    LinearBase,
)
GMM_TARGET_NAMES = {"gmm", "moe", "moegmm", "fused_moe_gmm"}


def _normalize_names(values: Iterable[str] | str | None, default: list[str]) -> list[str]:
    if values is None:
        return default
    if isinstance(values, str):
        return [values.lower()]
    return [value.lower() for value in values]


def _is_ignored(name: str, ignored_layers: Iterable[str]) -> bool:
    return any(name == ignored or name.startswith(f"{ignored}.") for ignored in ignored_layers)


def _is_mxfp8_quantized_linear(layer: nn.Module) -> bool:
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    quant_method = getattr(layer, "quant_method", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale is not None
        and not isinstance(quant_method, UnquantizedLinearMethod)
    )


def _is_mxfp8_quantized_gmm(layer: nn.Module) -> bool:
    w13_weight = getattr(layer, "w13_weight", None)
    w2_weight = getattr(layer, "w2_weight", None)
    return (
        w13_weight is not None
        and w2_weight is not None
        and w13_weight.dtype == torch.float8_e4m3fn
        and w2_weight.dtype == torch.float8_e4m3fn
        and getattr(layer, "w13_weight_scale", None) is not None
        and getattr(layer, "w2_weight_scale", None) is not None
    )


def _flatten_mxfp8_dynamic_scale(dynamic_scale: torch.Tensor | None) -> torch.Tensor | None:
    if dynamic_scale is None:
        return None
    if dynamic_scale.dim() >= 3 and dynamic_scale.shape[-1] == 2:
        return dynamic_scale.reshape(-1, dynamic_scale.shape[-2], 2)
    return dynamic_scale.reshape(-1, dynamic_scale.shape[-1])


def _reshape_mxfp8_scale_for_gmm(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None or (scale.dim() >= 3 and scale.shape[-1] == 2):
        return scale
    return scale.view(*scale.shape[:-1], scale.shape[-1] // 2, 2)


def _quantize_weight_last_dim(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = weight.shape
    weight_2d = weight.contiguous().view(-1, original_shape[-1])
    quant_weight, weight_scale = torch_npu.npu_dynamic_mx_quant(
        weight_2d,
        dst_type=torch.float8_e4m3fn,
    )
    quant_weight = quant_weight.reshape(original_shape).contiguous()
    weight_scale = weight_scale.reshape(*original_shape[:-1], -1).contiguous()
    return quant_weight, weight_scale


def _quantize_mxfp8_linear_weight(layer: LinearBase) -> None:
    weight = layer.weight.data
    if weight.dim() != 2:
        return

    quant_weight, weight_scale = _quantize_weight_last_dim(weight)
    layer.weight = torch.nn.Parameter(quant_weight.contiguous(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.quant_method = MxFp8LinearMethod()

    if getattr(layer, "bias", None) is not None:
        layer.bias = torch.nn.Parameter(layer.bias.to(torch.float32).contiguous(), requires_grad=False)


def _quantize_mxfp8_gmm_weight(layer: FusedMoEGMM) -> None:
    w13_quant_weight, w13_weight_scale = _quantize_weight_last_dim(layer.w13_weight.data)
    w2_quant_weight, w2_weight_scale = _quantize_weight_last_dim(layer.w2_weight.data)

    layer.w13_weight = torch.nn.Parameter(w13_quant_weight, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_quant_weight, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
    quant_config = MxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, BLOCK_K],
    )
    layer.quant_method = MxFp8MoEGMMMethod(quant_config)


def _apply_qwen3_5_online_mxfp8_quantization(
    model: torch.nn.Module,
    targets: Iterable[str] | str | None = None,
    ignored_layers: Iterable[str] | None = None,
) -> int:
    target_names = _normalize_names(targets, ["linear"])
    ignored_names = set(ignored_layers or [])
    convert_linear = "linear" in target_names
    convert_gmm = any(target_name in GMM_TARGET_NAMES for target_name in target_names)

    converted_linear = 0
    converted_gmm = 0
    for name, layer in model.named_modules():
        if _is_ignored(name, ignored_names):
            continue

        if convert_gmm and isinstance(layer, FusedMoEGMM):
            if not isinstance(getattr(layer, "quant_method", None), UnquantizedFusedMoEGMMMethod):
                continue
            if layer.w13_weight.dtype == torch.float8_e4m3fn and layer.w2_weight.dtype == torch.float8_e4m3fn:
                continue
            if layer.w13_weight.shape[-1] % BLOCK_K != 0 or layer.w2_weight.shape[-1] % BLOCK_K != 0:
                logger.warning(
                    "Skip online MXFP8 GMM quantization for %s: last dims are %s and %s, not divisible by %s.",
                    name,
                    layer.w13_weight.shape[-1],
                    layer.w2_weight.shape[-1],
                    BLOCK_K,
                )
                continue
            _quantize_mxfp8_gmm_weight(layer)
            converted_gmm += 1
            continue

        if not convert_linear or not isinstance(layer, LINEAR_TARGET_MODULES):
            continue
        if not isinstance(getattr(layer, "quant_method", None), UnquantizedLinearMethod):
            continue
        if layer.weight.dtype == torch.float8_e4m3fn:
            continue
        if layer.weight.shape[-1] % BLOCK_K != 0:
            logger.warning(
                "Skip online MXFP8 quantization for %s: input dimension %s is not divisible by %s.",
                name,
                layer.weight.shape[-1],
                BLOCK_K,
            )
            continue

        _quantize_mxfp8_linear_weight(layer)
        converted_linear += 1

    logger.info(
        "Online MXFP8 quantization converted %s linear layers and %s GMM layers.",
        converted_linear,
        converted_gmm,
    )
    return converted_linear + converted_gmm


def _normalize_quantization_method(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    return quantization.lower().replace("compressed_tensors", "compressed-tensors")


def _get_platform_version(infer_config: InferenceConfig | dict) -> str:
    global _PLATFORM_VERSION
    if _PLATFORM_VERSION is not None:
        return _PLATFORM_VERSION

    platform_version = infer_config.model_config.platform_version

    _PLATFORM_VERSION = getattr(platform_version, "value", platform_version)
    return _PLATFORM_VERSION


def _validate_qwen3_5_quantization_support(
    config,
    infer_config: InferenceConfig,
) -> None:
    model_config = infer_config.model_config
    custom_params = model_config.custom_params or {}

    exe_mode = model_config.exe_mode

    quantization = None
    quant_config = _get_qwen3_5_quantization_config(config, infer_config)
    if isinstance(quant_config, dict):
        quantization = _normalize_quantization_method(quant_config.get("quant_method"))

    if custom_params.get("enable_online_mxfp8_quantization", False):
        quantization = "mxfp8"

    if _PLATFORM_VERSION != "950" and quantization in {"fp8", "mxfp8"}:
        raise ValueError(
            f"The received model config platform_version is {_PLATFORM_VERSION}.\n"
            f"Qwen3.5 {quantization} is not supported on Atlas {_PLATFORM_VERSION}."
        )

    if _PLATFORM_VERSION == "950" and quantization == "mxfp8" and exe_mode == "ge_graph":
        raise ValueError(
            f"The received model config platform_version is {_PLATFORM_VERSION}.\n"
            f"Qwen3.5 mxfp8 is not supported with {exe_mode} on Atlas {_PLATFORM_VERSION}."
        )


def _validate_qwen3_5_mm_all_reduce_base_support(infer_config: InferenceConfig) -> None:
    model_config = infer_config.model_config
    custom_params = model_config.custom_params or {}

    if (
        custom_params.get("enable_mm_all_reduce_base", False)
        and _PLATFORM_VERSION == "A3"
    ):
        raise ValueError(
            f"The received model config platform_version is {_PLATFORM_VERSION}.\n"
            "Qwen3.5 enable_mm_all_reduce_base is only supported on Atlas 950."
        )


def _validate_qwen3_5_moe_parallel_support(infer_config: InferenceConfig) -> None:
    parallel_config = infer_config.parallel_config
    if parallel_config.moe_tp_size > 1 and parallel_config.moe_ep_size > 1:
        raise ValueError(
            "Qwen3.5 MoE does not support mixed tensor parallelism and expert parallelism: "
            f"got moe_tp_size={parallel_config.moe_tp_size}, "
            f"moe_ep_size={parallel_config.moe_ep_size}. "
            "Set moe_tp_size=1 for pure EP or moe_tp_size=world_size for pure TP."
        )


def _configure_qwen3_5_npugraph(infer_config: InferenceConfig) -> None:
    exe_mode = infer_config.model_config.exe_mode

    if _PLATFORM_VERSION != "950" and exe_mode == "npugraph_ex":
        torch.npu.config.allow_internal_format = False
        torch.npu.set_compile_mode(jit_compile=False)


def _normalize_qwen3_5_quantization_config(quant_config: dict | None) -> dict | None:
    if not isinstance(quant_config, dict):
        return quant_config

    normalized = dict(quant_config)
    ignored_layers = normalized.get("ignored_layers")
    if ignored_layers is None:
        ignored_layers = normalized.get("modules_to_not_convert")
    if ignored_layers is not None:
        normalized["ignored_layers"] = [
            layer_name.replace("model.language_model.", "model.", 1)
            for layer_name in ignored_layers
        ]
    return normalized


def _load_quantization_config_from_model_path(model_path: str) -> dict | None:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as config_file:
        model_config = json.load(config_file)
    return model_config.get("quantization_config") or model_config.get("compression_config")


def _get_qwen3_5_quantization_config(config, infer_config: InferenceConfig) -> dict | None:
    custom_params = infer_config.model_config.custom_params or {}
    quant_config = custom_params.get("quantization_config")
    if quant_config is None:
        quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        quant_config = getattr(config, "compression_config", None)
    if quant_config is None:
        quant_config = _load_quantization_config_from_model_path(infer_config.model_config.model_path)
    return _normalize_qwen3_5_quantization_config(quant_config)


def _sync_qwen3_5_ignored_layers(config, infer_config: InferenceConfig) -> None:
    quant_method_config = getattr(config, "quant_config", None)
    if quant_method_config is None or not hasattr(quant_method_config, "ignored_layers"):
        return

    quant_config = _get_qwen3_5_quantization_config(config, infer_config)
    if not quant_config:
        return
    ignored_layers = quant_config.get("ignored_layers")
    if ignored_layers is not None:
        quant_method_config.ignored_layers = ignored_layers


def _register_qwen3_5_packed_modules_mapping(config) -> None:
    quant_method_config = getattr(config, "quant_config", None)
    if quant_method_config is None:
        return

    quant_method_config.packed_modules_mapping.update({
        "merged_qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    })


def _init_qwen3_5_quant_config(config, infer_config: InferenceConfig) -> None:
    if getattr(config, "quant_config", None) is not None:
        _sync_qwen3_5_ignored_layers(config, infer_config)
        _register_qwen3_5_packed_modules_mapping(config)
        return

    quant_config = _get_qwen3_5_quantization_config(config, infer_config)
    if quant_config is not None:
        config.quantization_config = quant_config

    quantization = None
    if isinstance(quant_config, dict):
        quantization = _normalize_quantization_method(quant_config.get("quant_method"))

    if quantization:
        config.quant_config = get_quant_config(
            config,
            quantization,
            infer_config.model_config.model_path,
        )
        _register_qwen3_5_packed_modules_mapping(config)


def _map_quant_scale_name(param_name: str, params_dict: dict[str, torch.nn.Parameter]) -> str:
    candidates = [param_name]
    if ".weight_scale_inv" in param_name:
        candidates.extend([
            param_name.replace(".weight_scale_inv", ".scale"),
            param_name.replace(".weight_scale_inv", ".weight_scale"),
        ])
    if "_weight_scale_inv" in param_name:
        candidates.extend([
            param_name.replace("_weight_scale_inv", "_scale"),
            param_name.replace("_weight_scale_inv", "_weight_scale"),
        ])
    for candidate in candidates:
        if candidate in params_dict:
            return candidate
    return candidates[0]


def _has_fp8_weight(module: nn.Module) -> bool:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return False
    for attr_name in ("weight", "w13_weight", "w2_weight"):
        weight = getattr(module, attr_name, None)
        if weight is not None and weight.dtype == fp8_dtype:
            return True
    return False


def _has_non_mxfp8_fp8_weight(module: nn.Module) -> bool:
    return (
        _has_fp8_weight(module)
        and not _is_mxfp8_quantized_linear(module)
        and not _is_mxfp8_quantized_gmm(module)
    )


class Qwen3_5MoeQKVParallelLinear(QKVParallelLinear):
    def __init__(self, *args, quant_config=None, **kwargs):
        self.quant_config = quant_config
        super().__init__(*args, quant_config=quant_config, **kwargs)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ) -> None:
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        if loaded_shard_id is None:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
            return

        if output_dim is not None:
            is_per_block_scale = getattr(param, "is_per_block_scale", False)
            block_size = self.quant_config.weight_block_size[0] if is_per_block_scale else 1
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            else:
                raise RuntimeError(f"Unsupported loaded_shard_id: {loaded_shard_id}")

            if is_per_block_scale:
                shard_offset = math.ceil(shard_offset / block_size)
                shard_size = math.ceil(shard_size / block_size)

            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            shard_id = self.tp_rank if loaded_shard_id == "q" else self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "Qwen3_5MoeQKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)


class Qwen3_5MoeGatedDeltaNetQKVZProj(MergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        total_key_dim: int,
        total_value_dim: int,
        key_dim: int,
        value_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        quant_config=None,
        prefix: str = "",
    ):
        self.total_key_dim = total_key_dim
        self.total_value_dim = total_value_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.qkv_output_size = total_key_dim * 2 + total_value_dim
        super().__init__(
            hidden_size,
            [self.qkv_output_size, total_value_dim],
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _slice_qkv_packed_tensor(
        self,
        loaded_weight: torch.Tensor,
        output_dim: int,
        is_scale: bool,
    ) -> torch.Tensor:
        if is_scale:
            block_size = self.quant_config.weight_block_size[0]
            shard_specs = [
                (0, math.ceil(self.total_key_dim / block_size), math.ceil(self.key_dim / block_size)),
                (
                    math.ceil(self.total_key_dim / block_size),
                    math.ceil(self.total_key_dim / block_size),
                    math.ceil(self.key_dim / block_size),
                ),
                (
                    math.ceil(2 * self.total_key_dim / block_size),
                    math.ceil(self.total_value_dim / block_size),
                    math.ceil(self.value_dim / block_size),
                ),
            ]
        else:
            shard_specs = [
                (0, self.total_key_dim, self.key_dim),
                (self.total_key_dim, self.total_key_dim, self.key_dim),
                (2 * self.total_key_dim, self.total_value_dim, self.value_dim),
            ]

        shards = []
        for offset, _, local_size in shard_specs:
            start_idx = offset + self.tp_rank * local_size
            shards.append(loaded_weight.narrow(output_dim, start_idx, local_size))
        return torch.cat(shards, dim=output_dim).contiguous()

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ) -> None:
        if loaded_shard_id != 0:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
            return

        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            raise RuntimeError("QKV packed loading requires `output_dim` to be set")

        is_per_block_scale = getattr(param, "is_per_block_scale", False)
        loaded_weight = self._slice_qkv_packed_tensor(loaded_weight, output_dim, is_per_block_scale)

        param_data = param.data
        shard_size = loaded_weight.shape[output_dim]
        param_data = param_data.narrow(output_dim, 0, shard_size)
        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)


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

    def forward_mx(self, x: Tensor) -> tuple[Tensor, Tensor]:
        output, scale = torch_npu.npu_swiglu_mx_quant(
            x,
            group_index=None,
            dst_type=torch_npu.float8_e4m3fn,
            activate_left=True
        )
        return output, scale


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


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


@dataclass(frozen=True)
class CausalConv1dMetaData:
    query_start_loc: torch.Tensor
    has_initial_state: torch.Tensor


@dataclass(frozen=True)
class TndToBcsMetaData:
    num_requests: int
    max_seq_len: int
    flat_idx: torch.Tensor


@dataclass(frozen=True)
class LinearAttentionPrefillMetaData:
    causal_conv1d: CausalConv1dMetaData | None = None
    tnd_to_bcs: TndToBcsMetaData | None = None


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
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.attn_tp_rank = (
            comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        )
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get(
            "enable_mm_all_reduce_base", False
        )
        self.token_shard_supported = _qwen3_5_token_shard_supported(infer_config)
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
        self.causal_conv1d_weight = None

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        set_weight_attrs(self.dt_bias, {"weight_loader": self._load_linear_attn_v_head_param})

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        set_weight_attrs(self.A_log, {"weight_loader": self._load_linear_attn_v_head_param})

        self.norm = Qwen3_5MoeRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )

        self.out_proj = RowParallelLinear(
            self.total_value_dim,
            self.hidden_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            input_is_parallel=True,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.out_proj",
        )

        self.use_fused_causal_conv1d = _PLATFORM_VERSION == "950"

        self.in_proj_qkvz = Qwen3_5MoeGatedDeltaNetQKVZProj(
            self.hidden_size,
            self.total_key_dim,
            self.total_value_dim,
            self.key_dim,
            self.value_dim,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.in_proj_qkvz",
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            self.hidden_size,
            [self.total_num_v_heads, self.total_num_v_heads],
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.in_proj_ba",
        )

        self.attn_type = "Mamba"
        self.conv_state_cache = torch.Tensor([])
        self.ssm_state_cache = torch.Tensor([])
        cache_dtype = torch.bfloat16
        self.cache_entries = [
            MambaCacheEntry(
                cache_name="conv_state_cache",
                dtype=cache_dtype,
                needs_block=True,
                shape=[self.conv_kernel_size - 1, self.conv_dim],
                tensor_setter=lambda tensor, layer=self: setattr(layer, "conv_state_cache", tensor),
            ),
            MambaCacheEntry(
                cache_name="ssm_state_cache",
                dtype=cache_dtype,
                needs_block=True,
                shape=[self.num_v_heads, self.head_v_dim, self.head_k_dim],
                tensor_setter=lambda tensor, layer=self: setattr(layer, "ssm_state_cache", tensor),
            ),
        ]

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

    def _state_block_ids(
        self,
        forward_metadata: ForwardMetaData,
        batch_size: int,
    ) -> torch.Tensor:
        block_table = forward_metadata.block_table[self.attn_type]
        if block_table.shape[0] < batch_size:
            raise RuntimeError(
                f"layer {self.layer_idx}: Mamba block_table covers "
                f"{block_table.shape[0]} requests but this step runs {batch_size}."
            )
        return block_table[:batch_size, 0].to(torch.int32)

    @staticmethod
    def _gather_cache_rows(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=cache.device).view(-1)
        return torch.index_select(cache, 0, indices).contiguous()

    @staticmethod
    def _copy_cache_rows(cache: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
        indices = indices.to(device=cache.device).view(-1)
        values = values.to(device=cache.device, dtype=cache.dtype).contiguous()
        torch_npu.npu_scatter_nd_update_(cache, indices.view(-1, 1), values)

    def _build_conv_state(
        self,
        mixed_qkv: torch.Tensor,
        actual_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Build the operator-native [B, K-1, D] state from padded BCS input."""
        seq_lens = actual_seq_lens.to(device=mixed_qkv.device, dtype=torch.long).view(-1, 1)
        window_offsets = torch.arange(
            1 - self.conv_kernel_size,
            0,
            device=mixed_qkv.device,
            dtype=torch.long,
        ).view(1, -1)
        positions = seq_lens + window_offsets
        valid_positions = positions >= 0
        positions = positions.clamp(min=0, max=mixed_qkv.shape[-1] - 1)
        gather_indices = positions.unsqueeze(1).expand(-1, mixed_qkv.shape[1], -1)
        conv_state = torch.gather(mixed_qkv, dim=2, index=gather_indices)
        conv_state = conv_state.masked_fill(~valid_positions.unsqueeze(1), 0)
        return conv_state.transpose(1, 2).contiguous()

    def _prefill_causal_conv1d(
        self,
        mixed_qkv_tnd: torch.Tensor,
        forward_metadata: ForwardMetaData,
        state_indices: torch.Tensor,
        prefill_metadata: LinearAttentionPrefillMetaData | None,
    ) -> torch.Tensor:
        if self.use_fused_causal_conv1d:
            causal_conv1d_metadata = prefill_metadata.causal_conv1d
            return torch.ops.cann_ops_transformer.causal_conv1d_fn(
                x=mixed_qkv_tnd,
                weight=self.causal_conv1d_weight,
                bias=self.conv1d.bias,
                conv_states=self.conv_state_cache,
                cache_indices=state_indices,
                query_start_loc=causal_conv1d_metadata.query_start_loc,
                has_initial_state=causal_conv1d_metadata.has_initial_state,
                activation="silu",
            )

        actual_seq_lens = forward_metadata.actual_seq_lengths_q
        tnd_to_bcs_metadata = prefill_metadata.tnd_to_bcs

        mixed_qkv_flat = mixed_qkv_tnd.new_zeros(
            tnd_to_bcs_metadata.num_requests * tnd_to_bcs_metadata.max_seq_len,
            mixed_qkv_tnd.shape[-1],
        )
        mixed_qkv_flat.index_copy_(0, tnd_to_bcs_metadata.flat_idx, mixed_qkv_tnd)
        mixed_qkv_bcs = mixed_qkv_flat.view(
            tnd_to_bcs_metadata.num_requests,
            tnd_to_bcs_metadata.max_seq_len,
            self.conv_dim,
        ).transpose(1, 2)

        pre_conv_state = self._build_conv_state(mixed_qkv_bcs, actual_seq_lens)
        self._copy_cache_rows(self.conv_state_cache, state_indices, pre_conv_state)
        mixed_qkv_bcs = F.silu(
            self.conv1d(mixed_qkv_bcs)[:, :, :tnd_to_bcs_metadata.max_seq_len]
        )
        return mixed_qkv_bcs.transpose(1, 2).reshape(
            tnd_to_bcs_metadata.num_requests * tnd_to_bcs_metadata.max_seq_len,
            self.conv_dim,
        )[tnd_to_bcs_metadata.flat_idx].contiguous()

    def _forward_prefill(
        self,
        fused_proj: torch.Tensor,
        forward_metadata: ForwardMetaData,
        prefill_metadata: LinearAttentionPrefillMetaData | None,
    ):
        """
        Prefill with a platform-specific Conv1d and the shared fused chunk GDR.

        Flow:
            TND -> causal_conv1d -> fused chunk_gdr
        """
        actual_seq_lens = forward_metadata.actual_seq_lengths_q
        num_requests = actual_seq_lens.numel()
        state_indices = self._state_block_ids(forward_metadata, num_requests)

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

        mixed_qkv_tnd = self._prefill_causal_conv1d(
            mixed_qkv_tnd,
            forward_metadata,
            state_indices,
            prefill_metadata,
        )

        # QKV
        query, key, value = torch.split(
            mixed_qkv_tnd,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.view(-1, self.num_k_heads, self.head_k_dim)
        key = key.view(-1, self.num_k_heads, self.head_k_dim)
        value = value.view(-1, self.num_v_heads, self.head_v_dim)

        # beta / g
        beta = b.sigmoid().contiguous()

        g = -self.A_log.float().exp() * ge_safe_softplus(
            a.float() + self.dt_bias
        )

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

        self._copy_cache_rows(self.ssm_state_cache, state_indices, last_recurrent_state)

        # reshape for norm
        core_attn_out = core_attn_out.view(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        return core_attn_out, z

    def _decode_causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = mixed_qkv.shape[0]

        if self.use_fused_causal_conv1d:
            mixed_qkv = torch.ops.cann_ops_transformer.causal_conv1d_update(
                x=mixed_qkv.contiguous().view(batch_size, 1, self.conv_dim),
                conv_state=self.conv_state_cache,
                conv_state_indices=state_indices,
                weight=self.causal_conv1d_weight,
                bias=self.conv1d.bias,
                activation="silu",
            )
        else:
            conv_state = self._gather_cache_rows(self.conv_state_cache, state_indices)
            conv_state_bcs = conv_state.transpose(1, 2).contiguous()
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv.contiguous().view(batch_size, self.conv_dim, 1),
                conv_state_bcs,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
            )
            self._copy_cache_rows(
                self.conv_state_cache,
                state_indices,
                conv_state_bcs.transpose(1, 2),
            )
        return mixed_qkv.view(batch_size, self.conv_dim)

    def _forward_decode(
        self,
        fused_proj: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
    ):
        """
        Decode path.

        Input:
            fused_proj: [batch_size, proj_dim] (TND)

        Return:
            core_attn_out: [batch_size * num_v_heads, head_v_dim]
        """
        batch_size = hidden_states.shape[0]
        state_indices = self._state_block_ids(forward_metadata, batch_size)

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

        mixed_qkv = self._decode_causal_conv1d(mixed_qkv, state_indices)

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.view(batch_size, self.num_k_heads, self.head_k_dim)
        key = key.view(batch_size, self.num_k_heads, self.head_k_dim)
        value = value.view(batch_size, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid().view(batch_size, self.num_v_heads)

        g = ge_safe_softplus(
            a.float() + self.dt_bias
        ) * (-self.A_log.float().exp())
        g = g.view(batch_size, self.num_v_heads)

        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        scale = 1.0 / (self.head_k_dim ** 0.5)

        core_attn_out = torch_npu.npu_recurrent_gated_delta_rule(
            query.to(torch.bfloat16),
            key.to(torch.bfloat16),
            value.to(torch.bfloat16),
            self.ssm_state_cache,
            beta=beta.to(torch.bfloat16),
            scale=scale,
            actual_seq_lengths=forward_metadata.actual_seq_lengths_q.to(
                device=query.device,
                dtype=torch.int32,
            ),
            ssm_state_indices=state_indices,
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
        dynamic_scale: torch.Tensor | None = None,
        prefill_metadata: LinearAttentionPrefillMetaData | None = None,
    ):
        """
        Linear Attention forward with TND input/output.
        """
        hidden_states = apply_mask_to_padding_states(
            hidden_states,
            attention_mask,
        )

        is_prefill = forward_metadata.is_prefill

        # === Step 1: projections directly on TND ===
        qkvz_proj = self.in_proj_qkvz(hidden_states, dynamic_scale=dynamic_scale)
        ba_proj = self.in_proj_ba(hidden_states, dynamic_scale=dynamic_scale)
        fused_proj = torch.cat([qkvz_proj, ba_proj], dim=-1)

        if is_prefill:
            core_attn_out, z = self._forward_prefill(
                fused_proj,
                forward_metadata,
                prefill_metadata,
            )
        else:
            core_attn_out, z = self._forward_decode(
                fused_proj,
                hidden_states,
                forward_metadata,
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
            output = qwen3_5_attention_reduce(
                output,
                attn_tp_size=self.attn_tp_size,
                group=self.comm_manager.get_group("attn_tp_group"),
                enable_token_shard=(self.token_shard_supported and forward_metadata.is_prefill),
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
        self.moe_tp_size = infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        self.attn_tp_rank = comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
        self.token_shard_supported = _qwen3_5_token_shard_supported(infer_config)
        self.comm_manager = comm_manager
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads_per_rank = max(self.num_key_value_heads // self.attn_tp_size, 1)
        self.scale_fa = 1.0 / math.sqrt(self.head_dim)
        self.attn_intermediate_size = self.num_heads * self.head_dim
        self.attn_intermediate_size_per_rank = self.num_heads_per_rank * self.head_dim
        self.enable_gegraph = infer_config.model_config.exe_mode == "ge_graph"
        self.enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        self.merged_qkv_proj = Qwen3_5MoeQKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads * 2,
            total_num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=getattr(config, "quant_config", None),
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
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_type = "FullAttention"
        self.block_size = infer_config.scheduler_config.block_size
        self.k_cache = self.v_cache = torch.Tensor([])
        cache_dtype = config.torch_dtype if config.torch_dtype is not None else torch.get_default_dtype()
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
            ),
            CacheEntry(
                cache_name="v_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
            ),
        ]

    def _get_cache_slot_mapping(self, forward_metadata: ForwardMetaData) -> torch.Tensor:
        if self.k_cache.numel() == 0 or self.v_cache.numel() == 0:
            raise RuntimeError(
                f"layer {self.layer_idx}: FullAttention k/v cache is not initialized."
            )
        slot_mappings = forward_metadata.slot_mapping if forward_metadata is not None else None
        if slot_mappings is None or self.attn_type not in slot_mappings:
            raise RuntimeError(
                f"layer {self.layer_idx}: slot_mapping for {self.attn_type} is not initialized."
            )
        return slot_mappings[self.attn_type]

    def _update_cache(
        self,
        slot_mapping: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        num_slots = slot_mapping.numel()
        key_tokens = key_states.shape[0]
        value_tokens = value_states.shape[0]
        if num_slots != key_tokens or num_slots != value_tokens:
            raise RuntimeError(
                f"layer {self.layer_idx}: slot_mapping covers {num_slots} tokens, "
                f"but key_states/value_states contain {key_tokens}/{value_tokens} tokens."
            )
        slot_mapping = slot_mapping.to(device=self.k_cache.device, dtype=torch.long).view(-1, 1)
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping,
            key_states,
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping,
            value_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        forward_metadata: ForwardMetaData = None,
        dynamic_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        q_len = hidden_states.size(0)
        qkv_states = self.merged_qkv_proj(hidden_states, dynamic_scale=dynamic_scale)
        query_states, key_states, value_states = torch.split(
            qkv_states,
            [
                self.num_heads_per_rank * self.head_dim * 2,
                self.num_key_value_heads_per_rank * self.head_dim,
                self.num_key_value_heads_per_rank * self.head_dim,
            ],
            dim=-1,
        )

        query_states, gate = torch.chunk(
            query_states.view(q_len, -1, self.head_dim * 2),
            2,
            dim=-1
        )
        gate = gate.reshape(q_len, -1)

        query_states, _, _ = self.q_norm(query_states.view(q_len, self.num_heads_per_rank, self.head_dim))
        key_states, _, _ = self.k_norm(
            key_states.view(q_len, self.num_key_value_heads_per_rank, self.head_dim)
        )
        value_states = value_states.view(q_len, self.num_key_value_heads_per_rank, self.head_dim)

        cos, sin = position_embeddings
        rotary_dim = cos.shape[-1]
        q_rot = query_states[..., :rotary_dim]
        q_pass = query_states[..., rotary_dim:]
        k_rot = key_states[..., :rotary_dim]
        k_pass = key_states[..., rotary_dim:]

        q_rot = torch_npu.npu_rotary_mul(
            q_rot.unsqueeze(0), cos.unsqueeze(2), sin.unsqueeze(2), rotary_mode='half'
        ).squeeze(0)
        k_rot = torch_npu.npu_rotary_mul(
            k_rot.unsqueeze(0), cos.unsqueeze(2), sin.unsqueeze(2), rotary_mode='half'
        ).squeeze(0)

        query_states = torch.cat([q_rot, q_pass], dim=-1).contiguous()
        key_states = torch.cat([k_rot, k_pass], dim=-1).contiguous()
        value_states = value_states.contiguous()

        slot_mapping = self._get_cache_slot_mapping(forward_metadata)
        is_prefill = forward_metadata.is_prefill
        block_table = None if is_prefill else forward_metadata.block_table[self.attn_type]
        fa_ops = torch.ops.npu
        if not is_prefill and self.enable_gegraph:
            fa_ops = torchair.ops

        if not is_prefill and self.enable_npugraph_ex:
            actual_seq_kvlen = forward_metadata.actual_seq_lengths_list_kv
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_list_q
        else:
            actual_seq_kvlen = (
                forward_metadata.actual_seq_lengths_cu_kv
                if is_prefill
                else forward_metadata.actual_seq_lengths_kv
            )
            actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q

        if is_prefill:
            attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
                query_states,
                key_states,
                value_states,
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="TND",
                softmax_scale=self.scale_fa,
                sparse_mode=3,
                atten_mask=forward_metadata.attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            self._update_cache(slot_mapping, key_states, value_states)
        else:
            self._update_cache(slot_mapping, key_states, value_states)
            attn_output, _ = fa_ops.npu_fused_infer_attention_score_v2(
                query_states,
                self.k_cache.view(*self.k_cache.shape[:2], -1),
                self.v_cache.view(*self.v_cache.shape[:2], -1),
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="TND",
                softmax_scale=self.scale_fa,
                sparse_mode=3,
                atten_mask=forward_metadata.attention_mask,
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
            attn_output = qwen3_5_attention_reduce(
                attn_output,
                attn_tp_size=self.attn_tp_size,
                group=self.comm_manager.get_group("attn_tp_group"),
                    enable_token_shard=(self.token_shard_supported and forward_metadata.is_prefill),
            )
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
        self.shared_tp_size = infer_config.parallel_config.shared_tp_size
        self.enable_mm_all_reduce_base = infer_config.model_config.custom_params.get("enable_mm_all_reduce_base", False)
        self.comm_manager = comm_manager
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            tp_size=self.shared_tp_size,
            tp_rank=comm_manager.get_rank("shared_tp_group") if self.shared_tp_size > 1 else 0,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            tp_size=self.shared_tp_size,
            tp_rank=comm_manager.get_rank("shared_tp_group") if self.shared_tp_size > 1 else 0,
            input_is_parallel=True,
            quant_config=getattr(config, "quant_config", None),
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiLUAndMul()

    def forward(
        self,
        x,
        forward_metadata: ForwardMetaData = None,
        dynamic_scale: torch.Tensor | None = None,
    ):
        gate_up = self.gate_up_proj(x, dynamic_scale=dynamic_scale)
        if _is_mxfp8_quantized_linear(self.down_proj):
            x, swiglu_scale = self.act_fn.forward_mx(gate_up)
        else:
            x = self.act_fn(gate_up)
            swiglu_scale = None
        down_proj = qwen3_5_prefill_mm_all_reduce(
            self.down_proj,
            x,
            self.comm_manager,
            "shared_tp_group",
            self.enable_mm_all_reduce_base,
            forward_metadata,
        )
        used_mm_all_reduce_base = down_proj is not None
        if down_proj is None:
            down_proj = self.down_proj(x, dynamic_scale=swiglu_scale)
        if self.shared_tp_size > 1 and not used_mm_all_reduce_base:
            qwen3_5_all_reduce(down_proj, group=self.comm_manager.get_group("shared_tp_group"))
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
            quant_config=getattr(config, "quant_config", None),
            tp_size=self.moe_tp_size,
            tp_rank=self.moe_tp_rank,
            ep_size=self.moe_ep_size,
            ep_rank=self.moe_ep_rank,
            prefix=prefix,
        )


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
        self.exe_mode = infer_config.model_config.exe_mode
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
        pertoken_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        use_mxfp8_init_routing = (
            pertoken_scale is None
            and hidden_states.dtype != torch.float8_e4m3fn
            and _is_mxfp8_quantized_gmm(self.experts)
        )
        output = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=selected_experts.to(torch.int32),
            active_num=selected_experts.shape[0] * selected_experts.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=3 if use_mxfp8_init_routing else -1,
        )
        expanded_x, expanded_row_idx, tokens_per_expert = output[:3]
        if use_mxfp8_init_routing:
            expanded_pertoken_scale = output[3].view(output[3].shape[0], -1, 2)
        else:
            expanded_pertoken_scale = self._expand_pertoken_scale(pertoken_scale, expanded_row_idx)
        return expanded_x, expanded_row_idx, tokens_per_expert, expanded_pertoken_scale

    @staticmethod
    def _expand_pertoken_scale(
        pertoken_scale: torch.Tensor | None,
        expanded_row_idx: torch.Tensor,
    ) -> torch.Tensor | None:
        if pertoken_scale is None:
            return None
        return torch.index_select(pertoken_scale, 0, expanded_row_idx.to(torch.long))

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
        expanded_pertoken_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, List[int], List[int]]:
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
        gathered_pertoken_scale = None
        if expanded_pertoken_scale is not None:
            scale_for_dispatch = expanded_pertoken_scale
            use_mxfp8_scale_view = scale_for_dispatch.dtype == torch_npu.float8_e8m0fnu
            if use_mxfp8_scale_view:
                scale_for_dispatch = scale_for_dispatch.view(torch.int8)
            gathered_pertoken_scale = scale_for_dispatch.new_empty(
                (total_tokens, *scale_for_dispatch.shape[1:])
            )
            dist.all_to_all_single(
                gathered_pertoken_scale,
                scale_for_dispatch,
                output_splits,
                input_splits,
                group=moe_ep_group,
            )
            if use_mxfp8_scale_view:
                gathered_pertoken_scale = gathered_pertoken_scale.view(torch_npu.float8_e8m0fnu)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def _run_experts_tp_only(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded_x, expanded_row_idx, tokens_per_expert, expanded_pertoken_scale = self._init_routing(
            hidden_states,
            selected_experts,
            pertoken_scale,
        )
        hidden_states_ordered_by_experts = self.experts(
            expanded_x,
            tokens_per_expert,
            group_list_type=1,
            pertoken_scale=expanded_pertoken_scale,
        )
        expert_output = self._finalize_routing(hidden_states_ordered_by_experts, routing_weights, expanded_row_idx)
        if self.moe_tp_size > 1:
            qwen3_5_all_reduce(expert_output, group=self.comm_manager.get_group("moe_tp_group"))
        return expert_output

    def _run_experts_ep(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expanded_x, expanded_row_idx, tokens_per_expert, expanded_pertoken_scale = self._init_routing(
            hidden_states,
            selected_experts,
            pertoken_scale,
        )
        (
            tokens_per_expert_group,
            gathered_tokens,
            gathered_pertoken_scale,
            input_splits,
            output_splits,
        ) = self._dispatch_to_ep_group(
            tokens_per_expert,
            expanded_x,
            expanded_pertoken_scale,
        )

        scale_for_rerouting = gathered_pertoken_scale
        use_mxfp8_scale_view = (
            scale_for_rerouting is not None
            and scale_for_rerouting.dtype == torch_npu.float8_e8m0fnu
        )
        if use_mxfp8_scale_view:
            scale_for_rerouting = scale_for_rerouting.flatten(1)
        if scale_for_rerouting is None:
            rerouting_output = torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.moe_ep_size, -1),
            )
        else:
            rerouting_output = torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=scale_for_rerouting,
            )
        (
            hidden_states_ordered_by_experts,
            local_pertoken_scale,
            gathered_ids_unsort,
            tokens_per_local_expert,
        ) = rerouting_output
        if use_mxfp8_scale_view:
            local_pertoken_scale = _reshape_mxfp8_scale_for_gmm(local_pertoken_scale)
        hidden_states_ordered_by_experts = self.experts(
            hidden_states_ordered_by_experts,
            tokens_per_local_expert,
            group_list_type=1,
            pertoken_scale=local_pertoken_scale,
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
        return expert_output

    def _run_experts_dispatch_combine_v2(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_rank = self.comm_manager.config.global_rank
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        topk_ids = selected_experts.to(torch.int32)
        use_mxfp8 = (
            pertoken_scale is None
            and hidden_states.dtype != torch.float8_e4m3fn
            and _is_mxfp8_quantized_gmm(self.experts)
        )
        # quant_mode/comm_quant_mode: 4 means MXFP8 quantized output in float8_e4m3fn;
        # 0 means no quantization.
        quant_mode = 4 if use_mxfp8 else 0
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "x_active_mask": None,
            "moe_expert_num": self.num_experts,
            "global_bs": 0,
            "scales": None,
            "group_ep": moe_ep_group_name,
            "ep_world_size": self.moe_ep_size,
            "ep_rank_id": global_rank // self.moe_tp_size,
            "group_tp": moe_ep_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": global_rank % self.moe_tp_size,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "quant_mode": quant_mode,
        }
        if use_mxfp8:
            dispatch_args["y_dtype"] = torch.float8_e4m3fn
        if _PLATFORM_VERSION != "950":
            dispatch_args["comm_alg"] = "fullmesh_v2"
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, expand_scales, assist_info_for_combine, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        if use_mxfp8:
            expand_scales = _reshape_mxfp8_scale_for_gmm(expand_scales)

        hidden_states_ordered_by_experts = self.experts(
            expand_x,
            expert_token_num,
            group_list_type=1,
            pertoken_scale=expand_scales if use_mxfp8 else None,
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
            "ep_rank_id": global_rank // self.moe_tp_size,
            "group_tp": moe_ep_group_name,
            "tp_world_size": self.moe_tp_size,
            "tp_rank_id": global_rank % self.moe_tp_size,
            "expert_shard_type": 0,
            "shared_expert_num": 0,
            "shared_expert_rank_num": 0,
            "comm_quant_mode": quant_mode,
        }

        return torch_npu.npu_moe_distribute_combine_v2(**combine_args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData = None,
        dynamic_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TND format: hidden_states is [total_tokens, hidden_dim]
        hidden_dim = hidden_states.shape[-1]
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        pertoken_scale = _flatten_mxfp8_dynamic_scale(dynamic_scale)
        shared_expert_output = self.shared_expert(
            hidden_states_reshaped,
            forward_metadata=forward_metadata,
            dynamic_scale=pertoken_scale,
        )
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        if self.moe_tp_size > 1:
            expert_output = self._run_experts_tp_only(
                hidden_states_reshaped,
                routing_weights,
                selected_experts,
                pertoken_scale=pertoken_scale
            )
        else:
            if forward_metadata.is_prefill:
                expert_output = self._run_experts_ep(
                    hidden_states_reshaped,
                    routing_weights,
                    selected_experts,
                    pertoken_scale=pertoken_scale
                )
            else:
                expert_output = self._run_experts_dispatch_combine_v2(
                    hidden_states_reshaped,
                    routing_weights,
                    selected_experts,
                    pertoken_scale=pertoken_scale,
                )

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        # TND format: output stays as [total_tokens, hidden_dim]
        return expert_output


def _is_mxfp8_rms_norm_enabled(infer_config: InferenceConfig) -> bool:
    model_config = infer_config.model_config
    custom_params = getattr(model_config, "custom_params", {}) or {}
    online_mxfp8_enabled = custom_params.get(
        "enable_online_mxfp8_quantization",
        getattr(model_config, "enable_online_mxfp8_quantization", False),
    )
    target_names = _normalize_names(
        custom_params.get("online_mxfp8_quant_layers", None),
        ["linear"],
    )
    return (
        getattr(model_config, "quantization", None) == "mxfp8"
        or (online_mxfp8_enabled and "linear" in target_names)
    )


class Qwen3_5MoeRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_mxfp8_rms_norm: bool = False):
        super().__init__()
        self.eps = eps
        self.use_mxfp8_rms_norm = use_mxfp8_rms_norm
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
        if self.use_mxfp8_rms_norm:
            return self.rms_norm_mx(x)
        return torch_npu.npu_rms_norm(x, self.norm_weight, self.eps)[0]

    def rms_norm_mx(self, x, dst_type=292):
        """
        dst_type:{
            291: torch.float8_e5m2,
            292: torch.float8_e4m3fn,
            296: float4_e2m1fn_x2,
            297: float4_e1m2fn_x2
        }
        """
        x_quant, x_scale, _ = torch_npu.npu_rms_norm_dynamic_mx_quant(
            x,
            self.norm_weight,
            epsilon=self.eps,
            dst_type=dst_type,
        )
        return x_quant, x_scale

    def add_rms_norm_mx(self, residual, x, dst_type=292):
        """
        dst_type:{
            291: torch.float8_e5m2,
            292: torch.float8_e4m3fn,
            296: float4_e2m1fn_x2,
            297: float4_e1m2fn_x2
        }
        """
        y, x_out, mxscale_out, _ = torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            residual,
            x,
            gamma=self.norm_weight,
            epsilon=self.eps,
            dst_type=dst_type,
        )
        return y, mxscale_out, x_out

    def add_rms_norm_npu(self, residual, x):
        y, _, residual = torch_npu.npu_add_rms_norm(residual, x, self.norm_weight, self.eps)
        return y, residual

    def forward(self, x, *args):
        if len(args) == 0:
            if self.use_mxfp8_rms_norm:
                y, scale = self.ln_npu(x)
                return y, scale, None
            return self.ln_npu(x), None, None
        elif len(args) == 1 and args[0] is None:
            if self.use_mxfp8_rms_norm:
                y, scale = self.ln_npu(x)
                return y, scale, x
            return self.ln_npu(x), None, x
        elif len(args) == 1:
            residual = args[0]
            if self.use_mxfp8_rms_norm:
                return self.add_rms_norm_mx(residual, x)
            y, residual = self.add_rms_norm_npu(residual, x)
            return y, None, residual
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
        use_mxfp8_rms_norm = _is_mxfp8_rms_norm_enabled(infer_config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_mxfp8_rms_norm=use_mxfp8_rms_norm,
        )
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_residual: torch.Tensor | None = None,
        forward_metadata: ForwardMetaData = None,
        input_is_sharded: bool = False,
        prefill_metadata: LinearAttentionPrefillMetaData | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # RMSNorm is token-local, so normalize before restoring the full token sequence.
        # This keeps the residual sharded and avoids transporting it through AllGather.
        hidden_states, hidden_scale, residual = self.input_layernorm(hidden_states, past_residual)

        if input_is_sharded:
            full_tokens = hidden_states.shape[0] * self.mlp.moe_ep_size
            group = self.mlp.comm_manager.get_group("attn_tp_group")
            gathered_hidden_states = hidden_states.new_empty(
                full_tokens, *hidden_states.shape[1:]
            )
            dist.all_gather_into_tensor(
                gathered_hidden_states,
                hidden_states.contiguous(),
                group=group,
            )
            hidden_states = gathered_hidden_states

            if hidden_scale is not None:
                gathered_hidden_scale = hidden_scale.new_empty(
                    full_tokens, *hidden_scale.shape[1:]
                )
                dist.all_gather_into_tensor(
                    gathered_hidden_scale,
                    hidden_scale.contiguous(),
                    group=group,
                )
                hidden_scale = gathered_hidden_scale

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                forward_metadata=forward_metadata,
                dynamic_scale=hidden_scale,
                prefill_metadata=prefill_metadata,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                forward_metadata=forward_metadata,
                dynamic_scale=hidden_scale,
            )

        if hidden_states.shape[0] != residual.shape[0]:
            raise RuntimeError(
                "Qwen3.5 hidden_states and residual have inconsistent token lengths: "
                f"hidden_states.shape[0]={hidden_states.shape[0]}, "
                f"residual.shape[0]={residual.shape[0]}"
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm.add_rms_norm_npu(residual, hidden_states)
        hidden_states = self.mlp(hidden_states, forward_metadata=forward_metadata, dynamic_scale=None)
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


class Qwen3_5MoeTextModel(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, infer_config: InferenceConfig, comm_manager: CommManager):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.vocab_size_per_rank = config.vocab_size // self.embed_tp_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            getattr(config, "torch_dtype", torch.bfloat16),
            tp_size=self.embed_tp_size,
            tp_rank=comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3_5MoeDecoderLayer(config, infer_config, comm_manager, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.token_shard_supported = _qwen3_5_token_shard_supported(infer_config)

    @staticmethod
    def _prepare_linear_attention_prefill_metadata(
        forward_metadata: ForwardMetaData,
        hidden_states: torch.Tensor,
    ) -> LinearAttentionPrefillMetaData:
        if _PLATFORM_VERSION == "950":
            return LinearAttentionPrefillMetaData(
                causal_conv1d=CausalConv1dMetaData(
                    query_start_loc=F.pad(
                        forward_metadata.actual_seq_lengths_cu_q,
                        (1, 0),
                    ).to(torch.int32),
                    has_initial_state=torch.zeros(
                        forward_metadata.actual_seq_lengths_cu_q.numel(),
                        dtype=torch.int32,
                        device=hidden_states.device,
                    ),
                )
            )

        actual_seq_lens = forward_metadata.actual_seq_lengths_q
        cu_seq_lens = F.pad(
            forward_metadata.actual_seq_lengths_cu_q,
            (1, 0),
        )
        max_seq_len = actual_seq_lens.max().item()

        # A3 Conv1d consumes padded BCS input. Precompute the packed TND token
        # positions once, then reuse the mapping in every linear attention layer.
        token_idx = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )
        batch_idx = torch.bucketize(token_idx, cu_seq_lens[1:], right=True)
        seq_idx = token_idx - cu_seq_lens[batch_idx]
        return LinearAttentionPrefillMetaData(
            tnd_to_bcs=TndToBcsMetaData(
                num_requests=actual_seq_lens.numel(),
                max_seq_len=max_seq_len,
                flat_idx=batch_idx * max_seq_len + seq_idx,
            )
        )

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
        original_forward_metadata = forward_metadata
        pad_len = 0
        if forward_metadata.is_prefill and self.token_shard_supported:
            prompt_tokens = int(forward_metadata.prompt_tokens)
            padded_tokens = ((prompt_tokens + self.attn_tp_size - 1) // self.attn_tp_size) * self.attn_tp_size
            pad_len = padded_tokens - prompt_tokens
            if pad_len:
                if input_ids is not None and input_ids.shape[0] < padded_tokens:
                    input_ids = torch.cat([
                        input_ids,
                        input_ids.new_zeros(padded_tokens - input_ids.shape[0]),
                    ])
                if inputs_embeds is not None and inputs_embeds.shape[0] < padded_tokens:
                    inputs_embeds = torch.cat([
                        inputs_embeds,
                        inputs_embeds.new_zeros((padded_tokens - inputs_embeds.shape[0], *inputs_embeds.shape[1:])),
                    ])
                position_ids = torch.cat([position_ids, position_ids.new_zeros(pad_len)])

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

        padded_forward_metadata = forward_metadata
        if pad_len:
            padded_forward_metadata = _build_qwen3_5_pad_prefill_metadata(
                forward_metadata,
                pad_len,
            )
            # Attention/cache paths consume the padded metadata for this pass.
            forward_metadata = padded_forward_metadata

        # RoPE metadata is packed for the full (possibly padded) prompt; build
        # it before token sharding so its sequence dimension matches position_ids.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        prefill_metadata = (
            self._prepare_linear_attention_prefill_metadata(
                forward_metadata,
                hidden_states,
            )
            if forward_metadata.is_prefill
            else None
        )

        # In Prefill, shard the replicated embedding output across the attention
        # TP ranks when the token count is evenly divisible. Complete the
        # vocabulary-TP all-reduce before slicing to keep token positions aligned.
        token_shard_active = self.token_shard_supported and forward_metadata.is_prefill and (
            hidden_states.shape[0] % self.attn_tp_size == 0
        )
        if token_shard_active:
            rank = self.comm_manager.get_rank("attn_tp_group")
            hidden_states = torch.chunk(hidden_states, self.attn_tp_size, dim=0)[rank]

        residual = None

        # The first layer receives either the full embedding output or its local
        # token shard; subsequent layers preserve the same token-shard layout.
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            residual, hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_residual=residual,
                forward_metadata=forward_metadata,
                prefill_metadata=prefill_metadata,
                input_is_sharded=token_shard_active,
                **kwargs,
            )

        # Final RMSNorm is token-local; keep the residual sharded and gather only
        # the normalized states needed by the global last-token selection below.
        hidden_states, _, _ = self.norm(hidden_states, residual)

        if token_shard_active:
            full_hidden_states = hidden_states.new_empty(
                hidden_states.shape[0] * self.attn_tp_size,
                *hidden_states.shape[1:],
            )
            dist.all_gather_into_tensor(
                full_hidden_states,
                hidden_states.contiguous(),
                group=self.comm_manager.get_group("attn_tp_group"),
            )
            hidden_states = full_hidden_states

        # TND format: get last token for each sequence using index_select
        cu_seq_lens_q = original_forward_metadata.actual_seq_lengths_cu_q if original_forward_metadata else None
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
        super().__init__()
        assert infer_config.model_config.next_n == 0, (
            "Qwen3.5 only supports non-speculative decoding (next_n=0)."
        )
        _get_platform_version(infer_config)
        _init_qwen3_5_quant_config(config, infer_config)
        _validate_qwen3_5_quantization_support(config, infer_config)
        _validate_qwen3_5_mm_all_reduce_base_support(infer_config)
        _validate_qwen3_5_moe_parallel_support(infer_config)
        _configure_qwen3_5_npugraph(infer_config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        # Parallel config
        self.world_size = infer_config.parallel_config.world_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = infer_config.parallel_config.attn_dp_size
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        self.shared_tp_size = infer_config.parallel_config.shared_tp_size
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
            quant_config=getattr(config, "quant_config", None),
            prefix="lm_head",
        )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.decode_uses_explicit_metadata = True

    def init_parallel_comm_group(self):
        """Register all communication groups required by the model."""
        # Attention TP group
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
            return_name=True,
        )

        # Embedding TP group
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=self.world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
            return_name=True,
        )

        # LM Head TP group
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=self.world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
            return_name=True,
        )

        # Shared Expert TP group
        if self.shared_tp_size > 1:
            self.comm_manager.register_group(
                name="shared_tp_group",
                group_num=self.world_size // self.shared_tp_size,
                group_size=self.shared_tp_size,
                return_name=True,
            )

        # MoE TP group
        if self.moe_tp_size > 1:
            self.comm_manager.register_group(
                name="moe_tp_group",
                group_num=self.world_size // self.moe_tp_size,
                group_size=self.moe_tp_size,
                return_name=True,
            )

        # MoE EP group
        if self.moe_ep_size > 1:
            moe_ep_group_num = self.world_size // self.moe_ep_size
            moe_ep_group_type = 0 if _PLATFORM_VERSION == "950" else None
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
                group_type=moe_ep_group_type,
            )

        # MoE EP group for MC2 (dispatch/combine fusion)
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            moe_ep_mc2_buffer_size = calc_moe_hccl_buffer_size(
                self.infer_config, self.config, is_full_mesh_v2=_PLATFORM_VERSION != "950"
            )
            self.comm_manager.register_group(
                name="moe_ep_group_mc2",
                group_num=self.world_size // self.moe_ep_size,
                group_size=self.moe_ep_size,
                group_stride=self.world_size // self.moe_ep_size,
                return_name=True,
                allow_physical_reuse=False,
                hccl_buffer_size=moe_ep_mc2_buffer_size,
                group_type=3 if _PLATFORM_VERSION == "950" else None,
            )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
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

    def get_cache_info(self) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(attention.cache_entries),
                )
            )

        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
        )

    def load_weights(self, weights):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj_qkvz", "in_proj_qkv", 0),
            ("in_proj_qkvz", "in_proj_z", 1),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
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

                name_mapped = _map_quant_scale_name(norm_name.replace(weight_name, param_name), params_dict)
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
                    param_name = _map_quant_scale_name(norm_name.replace("gate_up_proj", "w13_weight"), params_dict)
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
                    param_name = _map_quant_scale_name(norm_name.replace("down_proj", "w2_weight"), params_dict)
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

                # MoE expert
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in norm_name:
                        continue

                    name_mapped = _map_quant_scale_name(norm_name.replace(weight_name, param_name), params_dict)

                    if name_mapped not in params_dict:
                        continue

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

                norm_name = _map_quant_scale_name(norm_name, params_dict)
                if norm_name not in params_dict:
                    logger.warning(f"[SKIP] No match in params_dict: {norm_name} (Original: {name})")
                    continue

                param = params_dict[norm_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

                loaded_params.add(norm_name)

        return loaded_params

    def process_weights_after_loading(self):
        custom_params = getattr(self.infer_config.model_config, "custom_params", {}) or {}
        if custom_params.get("enable_online_mxfp8_quantization", False):
            _apply_qwen3_5_online_mxfp8_quantization(
                self,
                targets=custom_params.get("online_mxfp8_quant_layers", ["linear"]),
                ignored_layers=custom_params.get("online_mxfp8_ignored_layers", []),
            )

        enable_mm_all_reduce_base = custom_params.get("enable_mm_all_reduce_base", False)
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
                    is_nz=(
                        self.infer_config.model_config.enable_weight_nz
                        and not use_mm_all_reduce_base
                        and not _has_non_mxfp8_fp8_weight(module)
                    ),
                    scales_dtype={},
                )
        for layer in self.model.layers:
            if layer.layer_type == "linear_attention" and layer.linear_attn.use_fused_causal_conv1d:
                layer.linear_attn.causal_conv1d_weight = (
                    layer.linear_attn.conv1d.weight.squeeze(1).transpose(0, 1).contiguous()
                )
    __all__ = [
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeTextModel",
]
