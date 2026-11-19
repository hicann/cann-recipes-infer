# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Dynamic token INT8 x channel INT8, with no floating MatMul fallback."""

import torch
from torch import nn


def quantize_rows(value):
    """CPU export/oracle rule: nearest-even, symmetric [-127,127]."""
    value = value.float()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("quantization requires a finite matrix")
    scale = value.abs().amax(-1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (value / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale


class NPUBackend:

    def __call__(self, x, weight, scale, output_dtype):
        if x.device.type != "npu":
            raise RuntimeError("W8A8 production backend requires NPU; no CPU/BF16 fallback")
        import torch_npu

        q, token_scale = torch_npu.npu_dynamic_quant(x.contiguous())
        # Native all-zero token scale=0 is valid; weights always use positive scales.
        return torch_npu.npu_quant_matmul(
            q, weight.t(), scale, pertoken_scale=token_scale.float(), output_dtype=output_dtype
        )


class Int8Linear(nn.Module):

    def __init__(
        self, in_features, out_features, num_experts=1, bias=False, device=None, backend=None, smoothing=False
    ):
        super().__init__()
        self.in_features, self.out_features, self.num_experts = in_features, out_features, num_experts
        self.register_buffer(
            "weight", torch.empty(num_experts * out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "weight_scale", torch.empty(num_experts * out_features, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "bias",
            torch.empty(num_experts * out_features, dtype=torch.bfloat16, device=device) if bias else None,
        )
        self.register_buffer(
            "smooth_scale",
            torch.empty(num_experts, in_features, dtype=torch.float32, device=device) if smoothing else None,
        )
        self.backend = NPUBackend() if backend is None else backend

    def _expert(self, x, expert, dtype):
        shape = (*x.shape[:-1], self.out_features)
        if x.numel() == 0:
            return x.new_empty(shape, dtype=dtype)
        x = x.reshape(-1, self.in_features)
        if self.smooth_scale is not None:
            x = (x.float() / self.smooth_scale[expert]).to(torch.bfloat16)
        return self._project(x, expert, dtype).reshape(shape)

    def _project(self, x, expert, dtype):
        """Project a flattened expert input."""
        start, end = expert * self.out_features, (expert + 1) * self.out_features
        # A3 INT8 kernel supports BF16/FP16 output, not FP32. Bias is separately
        # added in FP32 after this unavoidable kernel-output rounding.
        y = self.backend(
            x,
            self.weight[start:end],
            self.weight_scale[start:end],
            torch.float16 if dtype == torch.float16 else torch.bfloat16,
        )
        if self.bias is not None:
            y = y.float() + self.bias[start:end].float()
        return y.to(dtype)

    def forward(self, inputs, output_dtype=None, modality_dispatcher=None):
        dtype = inputs.dtype if output_dtype is None else output_dtype
        if inputs.ndim < 2 or inputs.shape[-1] != self.in_features:
            raise ValueError("expected [..., in_features] with at least two dimensions")
        if inputs.dtype != torch.bfloat16 or dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("W8A8 requires BF16 input and a floating output dtype")
        if self.num_experts == 1:
            return self._expert(inputs, 0, dtype)
        groups = modality_dispatcher.dispatch(inputs)
        if len(groups) != self.num_experts:
            raise ValueError("modality/expert count mismatch")
        return modality_dispatcher.undispatch(*(self._expert(x, i, dtype) for i, x in enumerate(groups)))
