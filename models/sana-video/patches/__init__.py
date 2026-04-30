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

VALID_QUANT_TYPES = {"bf16", "a8w8", "a4w4"}


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
        f_len, h_len, w_len = map(int, fhw)

        freqs = self.freqs.to(device=device, dtype=torch.complex64)

        axis_dim = self.attention_head_dim // 6
        f_dim = self.attention_head_dim // 2 - 2 * axis_dim

        f_freq, h_freq, w_freq = torch.split(freqs, [f_dim, axis_dim, axis_dim], dim=-1)

        f_part = f_freq[:f_len].reshape(f_len, 1, 1, f_dim)
        h_part = h_freq[:h_len].reshape(1, h_len, 1, axis_dim)
        w_part = w_freq[:w_len].reshape(1, 1, w_len, axis_dim)

        out = torch.cat(
            [
                f_part.expand(f_len, h_len, w_len, f_dim),
                h_part.expand(f_len, h_len, w_len, axis_dim),
                w_part.expand(f_len, h_len, w_len, axis_dim),
            ],
            dim=-1,
        )

        return out.flatten(0, 2).unsqueeze(0).unsqueeze(0)

    sana_blocks.WanRotaryPosEmbed.forward = forward


def apply_all(quant_type: str) -> None:

    if quant_type not in VALID_QUANT_TYPES:
        raise ValueError(
            f"Invalid quant_type: {quant_type}. "
            f"Expected one of {VALID_QUANT_TYPES}"
        )

    enable_quant = quant_type != "bf16"
    
    # NPU adaptation patches.
    patch_wan_rotary_npu()
    
    # NPU optimization patches.
    apply_npu_optimization_patches(enable_quant)
