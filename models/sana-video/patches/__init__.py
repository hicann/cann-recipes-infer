# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025 - 2026. All rights reserved.
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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

import importlib
import sys
import types

import torch

from .npu_patches import apply_npu_optimization_patches


def patch_triton_rms_norm_import() -> None:
    module_name = "diffusion.model.dc_ae.efficientvit.models.nn.triton_rms_norm"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class TritonRMSNorm2dFunc:
        @staticmethod
        def apply(*args, **kwargs):
            raise RuntimeError("triton rms norm is not available in Sana-Video NPU patch mode")

    module.__dict__["TritonRMSNorm2dFunc"] = TritonRMSNorm2dFunc
    sys.modules[module_name] = module


def patch_wan_rotary_npu() -> None:
    sana_blocks = importlib.import_module("diffusion.model.nets.sana_blocks")

    def forward(self, fhw: torch.Tensor, device: torch.device) -> torch.Tensor:
        ppf, pph, ppw = fhw

        freqs = self.freqs.to(dtype=torch.complex64).to(device)
        freqs = freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

    sana_blocks.WanRotaryPosEmbed.forward = forward


def apply_all() -> None:
    # NPU adaptation patches.
    patch_triton_rms_norm_import()
    patch_wan_rotary_npu()
    
    # NPU optimization patches.
    apply_npu_optimization_patches()
