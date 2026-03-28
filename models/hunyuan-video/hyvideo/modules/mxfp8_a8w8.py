# coding=utf-8
# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanVideo library and the HunyuanVideo
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to HunyuanVideo used by Tencent-Hunyuan team that trained the model.
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

import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch_npu


def mxfp8_linear_forward(layer, original_dtype, x):
    """
    Forward pass for MXFP8 quantized linear layer with A8W8 computation.

    This function performs quantized matrix multiplication using MXFP8 format:
    1. Quantizes input tensor to MXFP8 (E4M3FN format) dynamically
    2. Performs quantized matmul with pre-quantized weights

    Args:
        layer: nn.Linear module with quantized weight and weight_scale attribute.
               layer.weight: MXFP8 quantized weight tensor (torch.float8_e4m3fn)
               layer.weight_scale: Scale tensor for weight quantization
               layer.bias: Bias tensor (converted to float32 for computation)
        original_dtype: torch.dtype, the original data type for output (e.g., torch.bfloat16, torch.float16)
        x: torch.Tensor, input tensor with shape (batch_size, ..., in_features)

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, ..., out_features), dtype matches original_dtype
    """
    if x.dtype not in [torch.bfloat16, torch.float16]:
        x = x.to(torch.bfloat16)

    # Dynamically quantize input to MXFP8 format with per-token scale
    x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)

    if len(x_scale.shape) > 1:
        x_scale = x_scale.squeeze(0)

    linear_weight = layer.weight
    # Perform quantized matmul: A8W8 computation with input quantization and weight quantization
    output = torch_npu.npu_quant_matmul(
        x_quant,
        linear_weight.T,
        pertoken_scale=x_scale,
        bias=layer.bias.to(torch.float32),
        scale=layer.weight_scale.permute(1, 0, 2),
        output_dtype=original_dtype,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, 32]
    )

    return output


def dynamic_convert_mxfp8_linear(module, original_dtype, params_to_keep=None):
    """
    Dynamically convert Linear layers to MXFP8 quantized format for A8W8 computation.

    This function iterates through the module and converts eligible nn.Linear layers:
    1. Quantizes weights to MXFP8 (E4M3FN format) with per-group scales
    2. Replaces forward method with quantized forward pass

    Args:
        module: nn.Module, the root module containing Linear layers to convert
                       (e.g., DiT model with double_blocks and single_blocks)
        dit_weight_path: str, path to DiT weights (reserved for future loading, currently unused)
        original_dtype: torch.dtype, the original data type for model outputs (e.g., torch.bfloat16)
        params_to_keep: dict, optional parameter names to keep in original format (default: empty dict)

    Returns:
        None (modifies module in-place, converting Linear layers to MXFP8)

    Note:
        - Only converts Linear layers in 'double_blocks' or 'single_blocks' submodules
        - Skips layers with 1D weights or 'mod' in layer name (e.g., modulation layers)
        - Stores original forward in 'original_forward' attribute for potential restoration
    """
    
    # Mark the module as having MXFP8 quantization enabled
    setattr(module, "mxfp8_matmul_enabled", True)
    # Add getattr to safe globals for torch serialization compatibility
    torch.serialization.add_safe_globals([getattr])

    mxfp8_layers = []
    for key, layer in module.named_modules():
        # Only convert Linear layers in transformer blocks (double_blocks or single_blocks)
        if isinstance(layer, nn.Linear) and ('double_blocks' in key or 'single_blocks' in key):
            mxfp8_layers.append(key)
            original_forward = layer.forward
            # Freeze weight gradients since we're doing inference-only quantization
            layer.weight = torch.nn.Parameter(layer.weight, requires_grad=False)

            # Dynamic MXFP8 quantization for weights
            bf16_weight = layer.weight.data
            # Skip modulation layers (e.g., adaLN mod projections) and 1D weights
            is_mod_layer = 'mod' in key
            if len(bf16_weight.shape) == 1 or is_mod_layer:
                continue
            # Quantize weight to MXFP8 (E4M3FN) with per-group scale
            mxfp8_weight, mxfp8_layers_scale = torch_npu.npu_dynamic_mx_quant(
                bf16_weight,
                dst_type=torch.float8_e4m3fn
            )

            # Replace weight data with quantized version
            layer.weight.data = mxfp8_weight
            # Store the scale for weight quantization
            setattr(layer, "weight_scale", mxfp8_layers_scale)
            # Store original forward for potential restoration
            setattr(layer, "original_forward", original_forward)
            # Replace forward with quantized MXFP8 forward pass
            setattr(layer, "forward", lambda input, m=layer: mxfp8_linear_forward(m, original_dtype, input))