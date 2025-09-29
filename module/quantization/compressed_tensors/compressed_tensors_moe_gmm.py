# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import torch
import torch_npu
from torch.nn import Parameter
from module.utils import set_weight_attrs
from module.fuse_moe_gmm import FusedMoeWeightScaleSupported
from module.quantization import QuantizeMethodBase


class CompressedTensorsMoEGMMMethod(QuantizeMethodBase):

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",
        layer: torch.nn.Module,
    ) -> "CompressedTensorsMoEMethod":
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get(
            "input_activations")

        if quant_config.is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorW8A8Int8MoEGMMMethod()
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class CompressedTensorW8A8Int8MoEGMMMethod(QuantizeMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        scale_dtype = torch.float32 if params_dtype == torch.float16 else torch.bfloat16
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition,
                                                    hidden_size,
                                                    dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size,
                                                    intermediate_size_per_partition,
                                                    dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})

        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  dtype=scale_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 hidden_size,
                                                 dtype=scale_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        smooth_scale_1 = Parameter(torch.ones((num_experts, hidden_size), dtype=scale_dtype), requires_grad=False)
        smooth_scale_2 = Parameter(torch.ones((num_experts, intermediate_size_per_partition),
                                              dtype=scale_dtype),
                                   requires_grad=False)
        layer.register_parameter("smooth_scale_1", smooth_scale_1)
        layer.register_parameter("smooth_scale_2", smooth_scale_2)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              expert_tokens: torch.Tensor,
              group_list_type: int,
              pertoken_scale: torch.Tensor = None,
              final_output_dtype: torch.dtype = torch.bfloat16,):
        hidden_size = x.size(-1)

        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            x = x.view(-1, hidden_size)

        mm1_mm3 = torch_npu.npu_grouped_matmul([x], [layer.w13_weight],
                                                group_list=expert_tokens, split_item=3,
                                                output_dtype=torch.int32, group_type=0,
                                                group_list_type=group_list_type,
                                                tuning_config=[0]
                                                )[0]

        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            mm1_mm3, weight_scale=layer.w13_weight_scale,
            quant_scale=layer.smooth_scale_2,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
            activation_scale=pertoken_scale
            )

        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [layer.w2_weight], bias=None,
            scale=[layer.w2_weight_scale], per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            group_list_type=group_list_type,
            tuning_config=[0]
        )[0]

        return out_hidden

    def process_weights_after_loading(self, layer: torch.nn.Module,
                                      is_transpose: bool = True,
                                      is_nz: bool = True,
                                      **kwargs,
                                      ) -> None:
        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight
        if is_transpose:
            w13_weight.data = w13_weight.data.transpose(1, 2).contiguous()
            w2_weight.data = w2_weight.data.transpose(1, 2).contiguous()
        if is_nz:
            w13_weight.data = torch_npu.npu_format_cast(w13_weight.data, 29)  # 29: format nz
            w2_weight.data = torch_npu.npu_format_cast(w2_weight.data, 29)  # 29: format nz
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.to(torch.float)
        layer.smooth_scale_1.data = layer.smooth_scale_1.data.to(torch.float)
        layer.smooth_scale_2.data = layer.smooth_scale_2.data.to(torch.float)
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)
