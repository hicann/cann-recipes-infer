# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""W8A8 Linear with canonical checkpoint buffers and offload-safe device views."""
from types import SimpleNamespace

import torch
import torch.nn as nn


def validate_scale(scale, shape, name):
    if list(scale.shape) != list(shape) or scale.dtype != torch.float32:
        raise ValueError(f'{name}: expected FP32 scale {shape}, got {scale.dtype} {list(scale.shape)}')
    if scale.device.type != 'meta' and not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError(f'{name}: persistent scale must be finite and strictly positive')


class Int8Linear(nn.Linear):
    """Keep NK INT8 weight and FP32 channel scale in registered canonical buffers."""

    def __init__(self, in_features, out_features, bias=True, layout='nd_view', device='cpu', prefix=''):
        nn.Module.__init__(self)
        if layout not in ('nd_view', 'nd_contiguous', 'nz'):
            raise ValueError(f'Unsupported INT8 device layout: {layout}')
        self.in_features, self.out_features = in_features, out_features
        self.layout, self.prefix = layout, prefix
        self.register_buffer('weight', torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer('weight_scale', torch.ones(out_features, 1, dtype=torch.float32, device=device))
        self.register_buffer('bias', torch.zeros(out_features, dtype=torch.bfloat16, device=device) if bias else None)
        self._runtime = None
        from compressed_tensors.quantization import QuantizationStrategy
        from module.quantization.compressed_tensors.compressed_tensors_w8a8_int8 import (
            CompressedTensorsW8A8Int8LinearMethod)
        # Delegate all activation quantization and MatMul behavior to the shared
        # compressed-tensors implementation. This class only adapts its canonical
        # checkpoint buffers and offload lifecycle to Wan's nn.Linear modules.
        self.quant_method = CompressedTensorsW8A8Int8LinearMethod(
            strategy=QuantizationStrategy.CHANNEL, is_static_input_scheme=False,
            input_symmetric=True)

    def _apply(self, fn, recurse=True):
        # Module.to()/cpu() recurse through this hook, including expert offload.
        # Keep checkpoint scale/bias precision while moving them with the weight.
        self._runtime = None
        preserved = {name: self._buffers.pop(name) for name in ('weight_scale', 'bias')}
        try:
            result = super()._apply(fn, recurse=recurse)
            for name, tensor in preserved.items():
                if tensor is not None:
                    self._buffers[name] = tensor.to(device=self.weight.device, dtype=tensor.dtype)
                else:
                    self._buffers[name] = None
        finally:
            for name, tensor in preserved.items():
                if name not in self._buffers:
                    self._buffers[name] = tensor
        if self.weight.device.type == 'npu':
            self.prepare_runtime()
        return result

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._runtime = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        self._check_state()
        if self.weight.device.type == 'npu':
            self.prepare_runtime()

    def _check_state(self):
        if self.weight.dtype != torch.int8 or list(self.weight.shape) != [self.out_features, self.in_features]:
            raise ValueError(f'{self.prefix}: invalid canonical INT8 weight shape/dtype')
        validate_scale(self.weight_scale, [self.out_features, 1], self.prefix)
        if self.bias is not None and (
                self.bias.dtype != torch.bfloat16 or list(self.bias.shape) != [self.out_features]):
            raise ValueError(f'{self.prefix}: expected original BF16 bias [{self.out_features}]')

    def prepare_runtime(self):
        """Build device views once after loading or moving an expert to NPU."""
        if self.weight.device.type != 'npu':
            raise ValueError(f'{self.prefix}: move INT8 Linear to NPU before inference')
        weight = self.weight.transpose(0, 1)
        if self.layout != 'nd_view':
            weight = weight.contiguous()
        if self.layout == 'nz':
            import torch_npu
            weight = torch_npu.npu_format_cast(weight, 29)
        self._runtime = SimpleNamespace(
            weight=weight, weight_scale=self.weight_scale, bias=self.bias,
            smooth_scales=torch.ones(self.in_features, dtype=torch.bfloat16, device=self.weight.device))
        return self._runtime

    def forward(self, x):
        if x.shape[-1] != self.in_features or x.device != self.weight.device:
            raise ValueError(f'{self.prefix}: input/device does not match loaded weight')
        if not x.is_floating_point():
            raise ValueError(f'{self.prefix}: expected floating activation before dynamic INT8 quantization')
        if self._runtime is None:
            raise ValueError(f'{self.prefix}: move INT8 Linear to NPU before inference')
        matrix = x.reshape(-1, self.in_features).to(torch.bfloat16)
        result = self.quant_method.apply_weights(self._runtime, matrix, self.bias, out_dtype=torch.bfloat16)
        return result.reshape(*x.shape[:-1], self.out_features)
