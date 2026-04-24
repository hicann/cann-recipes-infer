# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import torch
import logging
from typing import Tuple, Optional, Literal
from torch import nn


logging.basicConfig(level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from act_quant import act_quant_kernel_int8_optimized

torch.set_default_device('npu')
torch.manual_seed(42)

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


# golden
def act_quant_torch(x: torch.Tensor, round_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        
        if x.dim() == 3:
            batch, seq, N = x.shape
            M = batch * seq
            x_2d = x.view(M, N)
        else:
            x_2d = x
            M, N = x_2d.shape
        
        x_fp32 = x_2d.float()
        
        abs_max = torch.max(torch.abs(x_fp32), dim=1, keepdim=True)[0]  # [M, 1]
        
        abs_max = torch.clamp(abs_max, min=1e-4)
        
        if round_scale:
            scales = 2 ** torch.ceil(torch.log2(abs_max / 127.0))
        else:
            scales = abs_max / 127.0
        
        scaled = x_fp32 / scales
        clipped = torch.clamp(scaled, -127, 127)
        x_int8 = torch.round(clipped).to(torch.float16).to(torch.int8)
        
        if len(original_shape) == 3:
            x_int8 = x_int8.view(original_shape)
            scales = scales.view(batch, seq, 1)
        
        return x_int8, scales

def validate_act_quant_kernel(x_bf16, M, N):
    x = x_bf16
    x_int8_torch, scales_torch = act_quant_torch(x, round_scale=False)

    return x_int8_torch, scales_torch



def test(custom_args=None):
    M, N = 128, 1024

    x_bf16 = torch.randn(M, N, dtype=torch.bfloat16)

    kernel = act_quant_kernel_int8_optimized(N, block_M=16, block_N=N, round_scale=False)
    logging.info("init successful!")

    Y, S = kernel(x_bf16.npu())
    
    x_int8_torch, scales_torch = validate_act_quant_kernel(x_bf16, M, N)
    torch.npu.synchronize()

    torch.testing.assert_close(Y, x_int8_torch, rtol=1e-2, atol=1)
    torch.testing.assert_close(S.reshape(M), scales_torch.reshape(M), rtol=1e-2, atol=1e-2)
    logging.info("Kernel Output Match!")


if __name__ == "__main__":
    test()
