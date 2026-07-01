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

from typing import Callable, Any, Dict, List, Optional, Union

import torch
import torch_npu
import torch.nn.functional as F
from torch.nn import Parameter
import math
from module.quantization import QuantizationMethods, QuantizeMethodBase, QuantizationConfig
from module.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from module.quantization.utils.quant_utils import is_layer_skipped, reshape_mx_scale
from module.fuse_moe_gmm import FusedMoEGMM, FusedMoeWeightScaleSupported
from module.utils import set_weight_attrs


ACTIVATION_SCHEMES = ["static", "dynamic"]
BEFORE_INIT = 0
AFTER_INIT = 1
BLOCK_K = 32
PACK_FACTOR = 2


def transpose_packed_fp4(weight: torch.Tensor) -> torch.Tensor:
    """Convert packed [..., N, K/2] FP4 weights to packed [..., K, N/2]."""
    if weight.dim() < 2:
        raise ValueError(
            "MXFP4 packed weight must have at least 2 dimensions, but got "
            f"{weight.dim()}.")
    if weight.shape[-2] % PACK_FACTOR != 0:
        raise ValueError(
            "MXFP4 output dimension must be even, but got "
            f"{weight.shape[-2]}.")
    low = weight & 0x0F
    high = (weight // 16) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(
        *weight.shape[:-2], weight.shape[-2], weight.shape[-1] * PACK_FACTOR)
    transposed = unpacked.transpose(-2, -1).contiguous()
    return (transposed[..., 0::2] | (transposed[..., 1::2] * 16)).contiguous()


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
        final_output_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        # Keep the torch_npu operator argument name. pertoken_scale does not
        # mean one scale for a whole token; the token dimension is indexed
        # independently, and each token stores MX scales per BLOCK_K hidden values.
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        num_tokens = x.shape[0]
        w13_weight_scale = layer.w13_weight_scale.transpose(1, 2) if self.enable_ge_graph else layer.w13_weight_scale
        # Do not pre-transpose the weights to prevent the weights from becoming attributes
        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [layer.w13_weight.transpose(1, 2)],
            antiquant_scale=[w13_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=torch.bfloat16, group_type=0,
            weight_dtype = torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        swiglu_limit = kwargs.get("swiglu_limit", None)
        enable_custom_ops = kwargs.get("enable_custom_ops", False)
        if enable_custom_ops:
            intermediate_h, pertoken_scale , _ = torch.ops.custom.npu_swiglu_group_quant(mm1_mm3,
                                                                                        dst_type=torch.float8_e4m3fn,
                                                                                        round_scale=True,
                                                                                        quant_mode=1,
                                                                                        clamp_limit=swiglu_limit,
                                                                                        group_index=expert_tokens)
        else:
            intermediate_h, pertoken_scale = torch_npu.npu_swiglu_mx_quant(
                                                mm1_mm3,
                                                group_index=expert_tokens,
                                                dst_type=torch_npu.float8_e4m3fn,
                                                activate_left=True,
                                            )
        w2_weight_scale = layer.w2_weight_scale.transpose(1, 2) if self.enable_ge_graph else layer.w2_weight_scale
        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight.transpose(1, 2)], bias=None,
            antiquant_scale=[w2_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            weight_dtype = torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        return out_hidden

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
        exe_mode = kwargs.get("exe_mode") or "eager"
        self.enable_ge_graph = exe_mode == "ge_graph"

        w13_weight = torch_npu.npu_format_cast(
            w13_weight.data.contiguous(), 29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2
        )
        w2_weight = torch_npu.npu_format_cast(
            w2_weight.data.contiguous(), 29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2
        )
        if self.enable_ge_graph:
            w13_weight_scale.data = reshape_mx_scale(w13_weight_scale.data)
            w2_weight_scale.data = reshape_mx_scale(w2_weight_scale.data)
        else:
            w13_weight_scale.data = reshape_mx_scale(w13_weight_scale.data).transpose(1, 2)
            w2_weight_scale.data = reshape_mx_scale(w2_weight_scale.data).transpose(1, 2)
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)


class W4A4MxFp4MoEGMMMethod(W4A8MxFp4MoEGMMMethod):
    """MXFP4 MoE with FP4 activations for both grouped matmuls."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: torch.Tensor = None,
        final_output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        # Keep the torch_npu operator argument name. pertoken_scale does not
        # mean one scale for a whole token; the token dimension is indexed
        # independently, and each token stores MX scales per BLOCK_K hidden values.
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                x.bfloat16(), dst_type=torch_npu.float4_e2m1fn_x2)

        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.bfloat16,
            group_type=0,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0],
        )[0]

        mm1_mm3 = torch_npu.npu_swiglu(mm1_mm3)
        intermediate_h, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
            mm1_mm3.bfloat16(), dst_type=torch_npu.float4_e2m1fn_x2)

        return torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight],
            scale=[layer.w2_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            split_item=3,
            output_dtype=final_output_dtype,
            group_type=0,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0],
        )[0]

    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
        is_transpose: bool = True,
        is_nz: bool = True,
        **kwargs,
    ) -> None:
        w13_weight = transpose_packed_fp4(
            layer.w13_weight.data.contiguous())
        w2_weight = transpose_packed_fp4(
            layer.w2_weight.data.contiguous())
        w13_weight_scale = reshape_mx_scale(
            layer.w13_weight_scale.data).transpose(1, 2).contiguous()
        w2_weight_scale = reshape_mx_scale(
            layer.w2_weight_scale.data).transpose(1, 2).contiguous()

        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            w13_weight_scale, requires_grad=False)
        layer.w2_weight_scale = Parameter(
            w2_weight_scale, requires_grad=False)


class MxFp4LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        **kwargs,
    ) -> None:
        input_size_per_partition = kwargs["input_size_per_partition"]
        output_partition_sizes = kwargs["output_partition_sizes"]
        weight_loader: Callable = kwargs["weight_loader"]

        if input_size_per_partition % PACK_FACTOR != 0:
            raise ValueError(
                "MXFP4 requires an even input size, but got "
                f"{input_size_per_partition}.")

        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // PACK_FACTOR,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "packed_dim": 1,
            "pack_factor": PACK_FACTOR,
            "weight_loader": weight_loader,
        })
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                math.ceil(input_size_per_partition / BLOCK_K),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": weight_loader,
        })
        layer.register_parameter("weight_scale", weight_scale)
        setattr(layer, "init_state", BEFORE_INIT)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
        dynamic_scale: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        return_scale: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        output_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])

        original_m = x.shape[0]
        pad_m = dynamic_scale is None and original_m == 1
        if pad_m:
            # Pad single-token input to avoid the FP4 quant matmul
            # kernel's m == 1 limitation.
            x = torch.cat([x, x], dim=0)

        if dynamic_scale is None:
            x, x_scale = torch_npu.npu_dynamic_mx_quant(
                x.bfloat16(), dst_type=torch_npu.float4_e2m1fn_x2)
        else:
            x_scale = dynamic_scale

        activation_scale = x_scale[:original_m] if pad_m else x_scale
        if out_dtype == torch.int32:
            return_scale = True
            matmul_out_dtype = torch.bfloat16
        else:
            matmul_out_dtype = out_dtype or torch.bfloat16

        # Keep the torch_npu operator argument name. pertoken_scale does not
        # mean one scale for a whole token; the token dimension is indexed
        # independently, and each token stores MX scales per BLOCK_K hidden values.
        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=matmul_out_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, BLOCK_K],
        )
        if pad_m:
            output = output[:original_m]
        output = output.view(output_shape + (-1,))
        if return_scale:
            return output, activation_scale
        return output

    apply_weights = apply

    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
        is_transpose: bool = True,
        is_nz: bool = True,
        scales_dtype=None,
    ) -> None:
        weight = layer.weight
        weight_scale = layer.weight_scale

        if is_transpose:
            weight.data = transpose_packed_fp4(weight.data)
            weight_scale.data = reshape_mx_scale(
                weight_scale.data).transpose(0, 1).contiguous()
        else:
            weight.data = weight.data.contiguous()
            weight_scale.data = reshape_mx_scale(
                weight_scale.data).contiguous()
        setattr(layer, "init_state", AFTER_INIT)
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)


class UpGateW4A4DownW4A8MxFp4MoEGMMMethod(W4A8MxFp4MoEGMMMethod):
    def __init__(self):
        super().__init__()

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
        final_output_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3, group_type=0,
            output_dtype=torch.bfloat16,
            x_dtype = torch_npu.float4_e2m1fn_x2,
            weight_dtype = torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        swiglu_limit = kwargs.get("swiglu_limit", None)
        enable_custom_ops = kwargs.get("enable_custom_ops", False) 
        if enable_custom_ops:
            intermediate_h, pertoken_scale , _ = torch.ops.custom.npu_swiglu_group_quant(mm1_mm3,
                                                                                        dst_type=torch.float8_e4m3fn,
                                                                                        round_scale=True,
                                                                                        quant_mode=1,
                                                                                        clamp_limit=swiglu_limit,
                                                                                        group_index=expert_tokens)
        else:
            mm1_mm3 = torch_npu.npu_swiglu(mm1_mm3)
            intermediate_h, pertoken_scale = torch_npu.npu_dynamic_mx_quant(mm1_mm3, dst_type=torch.float8_e4m3fn)

        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight.transpose(1, 2)], bias=None,
            antiquant_scale=[layer.w2_weight_scale],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            weight_dtype = torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        return out_hidden

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

        # w13 transpose + nd; w2 nz, transpose when inference
        w13_weight.data = w13_weight.data.transpose(1, 2)
        w2_weight = torch_npu.npu_format_cast(
            w2_weight.data.contiguous(), 29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch_npu.float4_e2m1fn_x2
        )
        w13_weight_scale.data = reshape_mx_scale(w13_weight_scale.data).transpose(1, 2)
        w2_weight_scale.data = reshape_mx_scale(w2_weight_scale.data).transpose(1, 2)

        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)
