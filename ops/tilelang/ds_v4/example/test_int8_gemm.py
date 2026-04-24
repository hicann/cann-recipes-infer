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
from torch import nn
from typing import Tuple, Optional, Literal


logging.basicConfig(level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from int8_gemm import int8_gemm_kernel_corrected

torch.set_default_device('npu')
torch.manual_seed(42)


FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


# golden
def int8_gemm_torch_optimized(
    a_int8: torch.Tensor,
    a_scales: torch.Tensor,
    b_int8: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    original_shape = a_int8.shape

    if a_int8.dim() == 3:
        batch, M, K = a_int8.shape
        a_int8_2d = a_int8.view(batch * M, K)
        a_scales_2d = a_scales.view(batch * M, 1)
    else:
        M, K = a_int8.shape
        batch = 1
        a_int8_2d = a_int8
        a_scales_2d = a_scales

    N = b_int8.shape[0]


    a_float32 = a_int8_2d.to(torch.float32)  # [M, K]
    b_float32 = b_int8.to(torch.float32)     # [N, K]
    output_float32 = torch.matmul(a_float32, b_float32.T)  # [M, N]

    scale_matrix = torch.outer(a_scales_2d.squeeze(1), b_scales.squeeze(1))  # [M, N]

    output_fp32 = output_float32 * scale_matrix  # [M, N]
    if out_dtype == torch.bfloat16:
        output_fp32 = output_fp32.bfloat16()
    if len(original_shape) == 3:
        output = output_fp32.view(batch, M, N)
    else:
        output = output_fp32

    return output


def test(custom_args=None):
    M, N, K = M, N, K = 1024, 1024, 1024
    torch_dtype_map = {
    "float16": torch.half, "float32": torch.float32, "bfloat16": torch.bfloat16,
    "int8": torch.int8, "int32": torch.int32, "int64": torch.int64, "uint64": torch.uint64
    }
    A = torch.randint(-128, 127, [M, K], dtype=torch_dtype_map["int8"])
    B = torch.randint(-128, 127, [K, N], dtype=torch_dtype_map["int8"])
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16)
    a_fp32 = a_bf16.float()
    a_abs_max = torch.max(torch.abs(a_fp32), dim=1, keepdim=True)[0]
    a_abs_max = torch.clamp(a_abs_max, min=1e-4)
    a_scales = a_abs_max / 127.0  # [M, 1]
    a_scaled = a_fp32 / a_scales
    a_int8 = torch.clamp(a_scaled, -128, 127).round().to(torch.int8)
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    b_fp32 = b_bf16.float()
    b_abs_max = torch.max(torch.abs(b_fp32), dim=1, keepdim=True)[0]
    b_abs_max = torch.clamp(b_abs_max, min=1e-4)
    b_scales = b_abs_max / 127.0  # [N, 1]
    b_scaled = b_fp32 / b_scales
    b_int8 = torch.clamp(b_scaled, -128, 127).round().to(torch.int8)
    a_int8_npu = a_int8.npu()
    a_scales_npu = a_scales.npu()
    b_int8_npu = b_int8.npu()
    b_scales_npu = b_scales.npu()
    output_npu = torch.empty(M, N, dtype=torch.bfloat16, device=a_int8_npu.device)
    kernel = int8_gemm_kernel_corrected(N, K, block_M=64, block_N=64, block_K=64, out_dtype=BF16)
    logging.info("init successful!")

    result = kernel(a_int8_npu, b_int8_npu, a_scales_npu, b_scales_npu)
    torch.npu.synchronize()

    output_torch_ref = int8_gemm_torch_optimized(
        a_int8, a_scales, b_int8, b_scales, out_dtype=torch.bfloat16
    )
    torch.npu.synchronize()

    torch.testing.assert_close(result, output_torch_ref, rtol=1e-2, atol=1e-2)
    logging.info("Kernel Output Match!")


if __name__ == "__main__":
    test()
