# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

LINEAR_MODES = {"a8w8", "a4w4"}
CONV2D_MODES = {"a8w8"}


def _validate_mode(mode: Optional[str], valid_modes, module_name: str) -> None:
    if mode is None:
        return
    if mode not in valid_modes:
        raise ValueError(f"Unsupported {module_name} quant mode: {mode}")


class QuantLinear(nn.Module):
    def __init__(self, module: nn.Linear, mode: str = "a8w8"):
        super().__init__()
        _validate_mode(mode, LINEAR_MODES, "linear")
        self.mode = mode
        self.quant_format = "mxfp"
        
        if isinstance(module, nn.Linear):
            self.in_features = module.in_features
            self.out_features = module.out_features
            weight = module.weight.detach()
            bias = module.bias
        elif isinstance(module, nn.Conv2d):
            self.in_features = module.in_channels
            self.out_features = module.out_channels
            weight = module.weight.detach().view(module.out_channels, module.in_channels)
            bias = module.bias
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")

        qweight, w_scale = self.quant_weight(weight)
        
        self.qweight = qweight
        self.register_buffer("w_scale", w_scale)
        if module.bias is not None:
            self.register_buffer("bias", bias.detach().clone().float())
        else:
            self.bias = None

    def quant_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.mode == "a8w8":
            weight_mx, w_scale = torch_npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn)
            weight_mx = weight_mx.transpose(-1, -2)
            w_scale = w_scale.transpose(0, 1)
        elif self.mode == "a4w4":
            weight = weight.transpose(-1, -2)
            weight_mx, w_scale = torch_npu.npu_dynamic_mx_quant(
                weight, dst_type=torch_npu.float4_e2m1fn_x2, axis=0)
        else:
            raise KeyError("not implement")
        
        weight_mx = weight_mx.contiguous()
        w_scale = w_scale.contiguous()
        return weight_mx, w_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "a8w8":
            out = self.forward_a8w8(x, self.bias)
        elif self.mode == "a4w4":
            out = self.forward_a4w4(x, self.bias)
        else:
            raise KeyError("not implement")
    
        return out
    
    def forward_a8w8(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out_shape_dim_0 = x.size()[:-1]
        x = x.view(-1, x.size(-1))

        mx_x, mx_scale = torch_npu.npu_dynamic_mx_quant(x.bfloat16(), dst_type=torch.float8_e4m3fn)
        out = torch_npu.npu_quant_matmul(
            mx_x, 
            self.qweight, 
            self.w_scale, 
            pertoken_scale=mx_scale,
            bias=bias,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu, 
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32])
        out = out.view(out_shape_dim_0 + (-1,))
        return out
    
    def forward_a4w4(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out_shape_dim_0 = x.size()[:-1]
        x = x.view(-1, x.size(-1))
        mx_x, mx_scale = torch_npu.npu_dynamic_mx_quant(x.bfloat16(), dst_type=torch_npu.float4_e2m1fn_x2)
        out = torch_npu.npu_quant_matmul(
            mx_x,
            self.qweight, 
            self.w_scale, 
            pertoken_scale=mx_scale, 
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu, 
            output_dtype=torch.bfloat16, 
            group_sizes=[1, 1, 32], 
            bias=bias, 
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu)
        out = out.view(out_shape_dim_0 + (-1,))
        return out


class QuantConv1x1AsLinear(nn.Module):
    def __init__(self, conv: nn.Conv2d, mode: str = "a8w8"):
        super().__init__()
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"not support {type(conv)} convert to QuantConv1x1AsLinear")

        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.linear = QuantLinear(conv, mode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_trans = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        out = self.linear(x_trans)
        out = out.permute(0, 2, 1).contiguous().view(b, self.out_channels, h, w)
        return out


class QuantConv2d(nn.Module):
    def __init__(self, module: nn.Conv2d, mode: str = "a8w8"):
        super().__init__()
        _validate_mode(mode, CONV2D_MODES, "conv2d")
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = nn.modules.utils._pair(module.kernel_size)
        self.stride = nn.modules.utils._pair(module.stride)
        self.padding = nn.modules.utils._pair(module.padding)
        self.dilation = nn.modules.utils._pair(module.dilation)
        self.groups = module.groups

        self.mode = mode
        self.quant_format = torch_npu.hifloat8
        self.output_dtype = torch.bfloat16

        weight = module.weight.detach()
        bias = module.bias
        qweight, w_scale = self.quant_weight(weight)

        self.register_buffer("w_scale", w_scale)

        self.conv = torch_npu.contrib.module.QuantConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            output_dtype=self.output_dtype,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=(bias is not None),
            input_dtype=self.quant_format,
            weight_dtype=self.quant_format
        )

        self.conv.weight.data = qweight
        self.conv.scale.data = w_scale
        if bias is not None:
            self.conv.bias.data = bias.detach().float()

    def quant_weight(self, weight: torch.Tensor) -> torch.Tensor:
        ori_shape = weight.shape
        weight = weight.view(ori_shape[0], -1)
        qweight, w_scale = torch_npu.npu_dynamic_quant(weight, dst_type=self.quant_format, quant_mode="pertoken")
        qweight = qweight.view(ori_shape)
        return qweight, w_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        qx, x_scale = torch_npu.npu_dynamic_quant(x, dst_type=self.quant_format, quant_mode="pertensor")
        effective_scale = self.w_scale * x_scale
        self.conv.scale.data = effective_scale
        out = self.conv(qx)
        out = out.to(torch.bfloat16)

        return out
    

def is_quant_module(module: nn.Module) -> bool:
    return isinstance(module, (QuantLinear, QuantConv2d))


def replace_quant_module(
    model: nn.Module,
    quant_mode: Optional[str] = "a8w8",
):
    _validate_mode(quant_mode, LINEAR_MODES, "linear")

    for block in model.blocks:
        ## conv2d to linear
        block.mlp.inverted_conv.conv = QuantConv1x1AsLinear(block.mlp.inverted_conv.conv, "a8w8")
        block.mlp.point_conv.conv = QuantConv1x1AsLinear(block.mlp.point_conv.conv, "a8w8")

        ## linear
        block.attn.qkv = QuantLinear(block.attn.qkv, quant_mode)
        block.attn.proj = QuantLinear(block.attn.proj, "a8w8")
        block.cross_attn.q_linear = QuantLinear(block.cross_attn.q_linear, quant_mode)
        block.cross_attn.proj = QuantLinear(block.cross_attn.proj, quant_mode)
        block.cross_attn.kv_linear = QuantLinear(block.cross_attn.kv_linear, quant_mode)
    
    return model