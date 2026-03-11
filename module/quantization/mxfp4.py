# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/layers/quantization/fp8.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from typing import Callable, Any, Dict, List, Optional, Union

import torch
import torch_npu
import torch.nn.functional as F
from torch.nn import Parameter
from module.quantization import QuantizationMethods, QuantizeMethodBase, QuantizationConfig
from module.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from module.quantization.utils.quant_utils import is_layer_skipped
from module.fuse_moe_gmm import FusedMoEGMM, FusedMoeWeightScaleSupported
from module.utils import set_weight_attrs


ACTIVATION_SCHEMES = ["static", "dynamic"]
BEFORE_INIT = 0
AFTER_INIT = 1
BLOCK_K = 32


def reshape_mx_scale(scale_tensor):
    """
    Reshape the last dimension of 2D/3D tensor into (original_size // 2, 2) for GMM/MM operators.
    """
    # Keep all dims except last, then split last into (n // 2, 2)
    return scale_tensor.view(*scale_tensor.shape[:-1], scale_tensor.size(-1) // 2, 2)


def unpack_mxfloat4_to_fp32(packed_tensor):
    # step 1
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=packed_tensor.device
        )

    # step 2
    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F

    # step 3
    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)

    # step 4
    fp32_tensor = e2m1_values[unpacked.long()]

    # step 5
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2
    return fp32_tensor.view(*new_shape)


class W4A8MxFp4MoEGMMMethod(QuantizeMethodBase):
    def __init__(self):
        super().__init__()
        STORAGE_BITS_NPU = 8
        WEIGHT_BITS = 4
        self.pack_factor = STORAGE_BITS_NPU // WEIGHT_BITS

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_type = torch.uint8
        scale_dtype = torch.uint8

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition,
                                                    hidden_size // self.pack_factor,
                                                    dtype=weight_type),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size,
                                                    intermediate_size_per_partition // self.pack_factor,
                                                    dtype=weight_type),
                                        requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})

        w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  math.ceil(hidden_size / BLOCK_K),
                                                  dtype=scale_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 hidden_size,
                                                 math.ceil(intermediate_size_per_partition / BLOCK_K),
                                                 dtype=scale_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: torch.Tensor = None,
        final_output_dtype: torch.dtype = torch.bfloat16
    ):
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        num_tokens = x.shape[0]
        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [layer.w13_weight.transpose(1, 2)],
            antiquant_scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale.view(-1, x.shape[1] // BLOCK_K)],
            group_list=expert_tokens, split_item=3,
            output_dtype=torch.bfloat16, group_type=0,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        mm1_mm3 = torch_npu.npu_swiglu(mm1_mm3)
        intermediate_h, pertoken_scale = torch_npu.npu_dynamic_mx_quant(mm1_mm3, dst_type=torch.float8_e4m3fn)

        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight.transpose(1, 2)], bias=None,
            antiquant_scale=[layer.w2_weight_scale],
            per_token_scale=[pertoken_scale.view(-1, intermediate_h.shape[1] // BLOCK_K)],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        return out_hidden

    def _weight_transformat_nz(self, weight_data):
        weight_unpacked = torch.zeros(weight_data.shape).to(torch.float32)
        weight_unpacked = unpack_mxfloat4_to_fp32(weight_data)
        weight_data_nz = torch_npu.npu_format_cast(weight_unpacked, 29, torch.float8_e4m3fn)
        weight_data_nz_packed = torch_npu.npu_convert_weight_to_int4pack(weight_data_nz)
        return weight_data_nz_packed

    def process_weights_after_loading(
        self, layer: torch.nn.Module,
        is_transpose: bool = True,
        is_nz: bool = True,
        **kwargs,
    ) -> None:
        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight
        w13_weight_scale = layer.w13_weight_scale
        w2_weight_scale = layer.w2_weight_scale

        w13_weight = self._weight_transformat_nz(w13_weight)
        w2_weight = self._weight_transformat_nz(w2_weight)
        w13_weight_scale.data = w13_weight_scale.data.transpose(1, 2)
        w2_weight_scale.data = w2_weight_scale.data.transpose(1, 2)

        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)