# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch_npu
import torchair

import custom_ops
from en_dtypes import hifloat8
from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))


@dataclass
class RopeQuantConfig:
    rotary_mode: str
    partial_slice: list
    scale: float


@dataclass
class TestShapeConfig:
    b: int
    s: int
    n: int
    d: int
    con_b: int
    con_s: int
    con_n: int
    dtype: torch.dtype


def requantize_compare(golden, output):
    """Bit-level comparison for hifloat8: view as int8, allow up to 1-bit difference."""
    output_int8 = output.view(torch.int8)
    golden_int8 = golden.view(torch.int8)

    diff_abs = torch.abs(output_int8.view(-1) - golden_int8.view(-1))
    diff_indices = torch.where(diff_abs > 1)[0]

    # Exclude positions where both are NaN
    output_float = output.view(-1).float()
    golden_float = golden.view(-1).float()
    both_nan = torch.isnan(output_float) & torch.isnan(golden_float)
    both_nan_idx = torch.where(both_nan)[0]
    diff_indices = diff_indices[~torch.isin(diff_indices, both_nan_idx)]

    total_elements = golden.numel()
    good_elements = total_elements - len(diff_indices)
    precision = good_elements / total_elements
    is_pass = (1 - precision) <= 0.001  # <= 0.1% element diff
    return is_pass


def rotate_every_two(x):
    """interleave mode: swap adjacent pairs and negate odd"""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack(((-x_odd, x_even)), dim=-1)
    return stacked.reshape(x.shape)


def half_rope(x, cos, sin):
    """half mode: q_out[0] = q[0]*cos[0] - q[1]*sin[0], q_out[1] = q[1]*cos[1] + q[0]*sin[1]"""
    half_d = x.shape[-1] // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]
    cos1 = cos[..., :half_d]
    cos2 = cos[..., half_d:]
    sin1 = sin[..., :half_d]
    sin2 = sin[..., half_d:]
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2
    return torch.cat([out1, out2], dim=-1)


def cpu_rope_and_quant(x, cos, sin, config):
    """CPU golden: rope on slice + full D quantize to hifloat8."""
    rotary_mode = config.rotary_mode
    partial_slice = config.partial_slice
    scale = config.scale
    slice_start, slice_end = partial_slice[0], partial_slice[1]
    result = x.float().numpy().copy()
    x_slice = result[..., slice_start:slice_end]
    cos_np = cos.float().numpy()
    sin_np = sin.float().numpy()

    if rotary_mode == "interleave":
        x_even = x_slice[..., ::2]
        x_odd = x_slice[..., 1::2]
        stacked = np.stack((-x_odd, x_even), axis=-1)
        rotated = stacked.reshape(x_slice.shape)
        rope_result = cos_np * x_slice + rotated * sin_np
    elif rotary_mode == "half":
        half_d = x_slice.shape[-1] // 2
        x1 = x_slice[..., :half_d]
        x2 = x_slice[..., half_d:]
        cos1 = cos_np[..., :half_d]
        cos2 = cos_np[..., half_d:]
        sin1 = sin_np[..., :half_d]
        sin2 = sin_np[..., half_d:]
        rope_result = np.concatenate((x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2), axis=-1)
    else:
        raise ValueError(f"Unsupported mode: {rotary_mode}")

    result[..., slice_start:slice_end] = rope_result
    # Quantize: multiply by scale in float32 (keep same precision as kernel Muls), cast to hifloat8
    scale_f32 = np.float32(scale)
    result_f32 = (result * scale_f32).astype(np.float32)
    return result_f32.astype(hifloat8)


def partial_rotary_mul_quant_eager(shapes, config, self):
    """Eager mode test with CPU golden comparison"""
    b, s, n, d = shapes.b, shapes.s, shapes.n, shapes.d
    con_b, con_s, con_n = shapes.con_b, shapes.con_s, shapes.con_n
    dtype = shapes.dtype
    rotary_mode = config.rotary_mode
    partial_slice = config.partial_slice
    scale = config.scale

    np.random.seed(0)
    slice_size = partial_slice[1] - partial_slice[0]

    x = torch.tensor(np.random.uniform(1, 10, (b, s, n, d))).to(dtype)
    cos = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)
    sin = torch.tensor(np.random.uniform(-10, 10, (con_b, con_s, con_n, slice_size))).to(dtype)

    # CPU golden
    golden_np = cpu_rope_and_quant(x, cos, sin, config)
    golden = torch.from_numpy(golden_np.view(np.int8)).view(torch.int8)

    # NPU eager
    x_npu = x.to("npu:%s" % DEVICE_ID)
    cos_npu = cos.to("npu:%s" % DEVICE_ID)
    sin_npu = sin.to("npu:%s" % DEVICE_ID)

    y_npu = torch.ops.custom.partial_rotary_mul_quant(
        x_npu, cos_npu, sin_npu,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
        scale=scale
    )

    y_cpu = y_npu.cpu()
    self.assertTrue(requantize_compare(golden, y_cpu),
                    f"hifloat8 precision compare fail for dtype {dtype}")


class TestNpuPartialRotaryMulQuant(TestCase):
    """AB tiling pattern"""
    def test_partial_rotary_mul_quant_eager_ab(self):
        b, s, n, d = 128, 64, 1, 512
        con_b, con_s, con_n = 128, 1, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[448, 512], scale=1.0),
            self)

    """ABA tiling pattern"""
    def test_partial_rotary_mul_quant_eager_aba(self):
        b, s, n, d = 128, 64, 128, 512
        con_b, con_s, con_n = 128, 1, 128
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[484, 512], scale=1.0),
            self)

    """BAB tiling pattern"""
    def test_partial_rotary_mul_quant_eager_bab(self):
        b, s, n, d = 128, 64, 128, 512
        con_b, con_s, con_n = 1, 64, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[448, 512], scale=1.0),
            self)

    """A pattern: cos broadcast (1,1,1,D)"""
    def test_partial_rotary_mul_quant_eager_a(self):
        b, s, n, d = 1, 1, 1, 512
        con_b, con_s, con_n = 1, 1, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[448, 512], scale=1.0),
            self)

    """float16 dtype test"""
    def test_partial_rotary_mul_quant_eager_fp16(self):
        b, s, n, d = 128, 64, 1, 512
        con_b, con_s, con_n = 128, 1, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.float16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[448, 512], scale=1.0),
            self)

    """different scale test"""
    def test_partial_rotary_mul_quant_eager_scale(self):
        b, s, n, d = 128, 64, 1, 512
        con_b, con_s, con_n = 128, 1, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="interleave", partial_slice=[448, 512], scale=0.5),
            self)

    """half mode test"""
    def test_partial_rotary_mul_quant_eager_half(self):
        b, s, n, d = 128, 64, 1, 512
        con_b, con_s, con_n = 128, 1, 1
        partial_rotary_mul_quant_eager(
            TestShapeConfig(b=b, s=s, n=n, d=d, con_b=con_b, con_s=con_s, con_n=con_n, dtype=torch.bfloat16),
            RopeQuantConfig(rotary_mode="half", partial_slice=[448, 512], scale=1.0),
            self)


if __name__ == "__main__":
    run_tests()