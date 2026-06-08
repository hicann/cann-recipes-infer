# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import copy
import random
import re
import unittest
from dataclasses import dataclass

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchair
import torch_npu
from ml_dtypes import bfloat16 as bf16
from ml_dtypes import float8_e4m3fn, float8_e5m2
from torch_npu.testing.testcase import TestCase, run_tests

import custom_ops

np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

DATA_TYPE_INT_TO_STR = {
    1: 'float16',
    27: 'bfloat16',
    35: 'float8_e5m2',
    36: 'float8_e4m3fn',
    40: 'float4_e2m1',
    41: 'float4_e1m2',
}

# torch only exposes a packed fp4 dtype (float4_e2m1fn_x2) from 2.8 onwards. fp4 is now selected purely
# through the standard dst_type ScalarType, so the eager fp4 cases require torch>=2.8 with that dtype.
# On torch<2.8 there is no way to select fp4 from PyTorch, so those cases are skipped (expected).
_TORCH_VER = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
HAS_TORCH_FP4 = _TORCH_VER >= (2, 8) and hasattr(torch, "float4_e2m1fn_x2")
# torch ScalarType used to select FLOAT4_E2M1 output (only defined on torch>=2.8).
TORCH_FP4_E2M1 = getattr(torch, "float4_e2m1fn_x2", None)
FP4_SKIP_MSG = "fp4 requires torch>=2.8 with torch.float4_e2m1fn_x2 (selected via dst_type)"
try:
    from ml_dtypes import float4_e2m1fn as fp4_e2m1
    HAS_ML_FP4 = True
except ImportError:
    HAS_ML_FP4 = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def get_dtype_range(dt):
    if 'bfloat16' in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if 'float8_e5m2' in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if 'float8_e4m3fn' in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in 'iu':
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def swiglu(x, clamp_limit=None):
    last_dim = x.shape[-1] // 2
    x0 = x.reshape(-1, last_dim * 2)[..., 0: last_dim]
    x1 = x.reshape(-1, last_dim * 2)[..., last_dim:]
    if clamp_limit is not None:
        x0 = np.minimum(x0, clamp_limit)
        x1 = np.minimum(clamp_limit, np.maximum(x1, -clamp_limit))
    y = silu(x0) * x1
    y = y.reshape(-1, last_dim)
    return y


def replace_inf_nan_with_zero(x):
    x_cleand = x.copy()
    mask = np.isinf(x) | np.isnan(x)
    x_cleand[mask] = 0
    mask_tensor = np.zeros_like(x, dtype=int)
    mask_tensor[mask] = 1
    return x_cleand, mask_tensor


def block_max_with_padding(x, row_block_size, col_block_size):
    rows, cols = x.shape
    pad_rows = (row_block_size - rows % row_block_size) % row_block_size
    pad_cols = (col_block_size - cols % col_block_size) % col_block_size

    x_padded = np.pad(x, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    padded_rows, padded_cols = x_padded.shape
    row_blocks = padded_rows // row_block_size
    col_blocks = padded_cols // col_block_size

    result = np.zeros((row_blocks, col_blocks))
    for i in range(row_blocks):
        for j in range(col_blocks):
            block = x_padded[i * row_block_size: (i + 1) * row_block_size,
                             j * col_block_size: (j + 1) * col_block_size]
            result[i, j] = np.max(block)

    return result


# quant_mode == 0 ---------------------------------------
def dynamic_block_quant(x, dst_type):
    row_block_size = 1
    col_block_size = 128
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    max_value = 0
    if dst_type_str == 'float8_e5m2':
        max_value = (2 - pow(2, -2)) * pow(2, 15)
    elif dst_type_str == 'float8_e4m3fn':
        max_value = (2 - pow(2, -2)) * pow(2, 8)


    x_cleaned, mask_tensor = replace_inf_nan_with_zero(x)
    x_abs = np.abs(x_cleaned)
    block_max = block_max_with_padding(x_abs, 1, 128)
    block_max_f32 = block_max.astype(np.float32)
    scale = block_max_f32 / max_value

    scale_expanded = np.zeros_like(x_cleaned).astype(np.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            scale_expanded[i * row_block_size: (i + 1) * row_block_size,
                           j * col_block_size: (j + 1) * col_block_size] = scale[i, j]

    x_f32 = x_cleaned.astype(np.float32)
    out_f32 = x_f32 / scale_expanded
    out_f32[mask_tensor==1] = x[mask_tensor==1]
    max_norm = get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)
    output_scale = scale.astype("float32")
    round_data = np.round(out_f32, 8)
    round_data = np.nan_to_num(round_data, nan=0.0, copy=False)
    if dst_type == 35:
        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        round_data = round_data.astype(float8_e4m3fn, copy=False)

    return round_data, output_scale


def swiglu_group_quant(x, dst_type, quant_mode, weight=None, clamp_limit=None):
    y = swiglu(x, clamp_limit)
    if weight is not None:
        y = y * weight
    if quant_mode == 0:
        out, scale = dynamic_block_quant(y, dst_type)
    return out, scale


# quant compare function
def requantize_compare(golden, output, dtype_str):
    if dtype_str in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8'):
        output_0 = output.view(torch.int8)
        golden_0 = golden.view(torch.int8)

    diff_results = torch.abs(torch.subtract(output_0.view(-1), golden_0.view(-1)))
    diff_indices = torch.where(diff_results > 1)[0]

    npu_nan, golden_nan = torch.isnan(output.view([-1])), torch.isnan(golden.view([-1]))
    diff_nan = torch.logical_and(npu_nan, golden_nan)
    both_nan_idx = torch.where(diff_nan)

    diff_indices = torch.where(torch.logical_not(torch.isin(diff_indices, both_nan_idx[0])))[0]
    del diff_results, npu_nan, golden_nan, diff_nan

    golden_size, diff_size = golden.numel(), diff_indices.numel()
    precision = (golden_size - diff_size) / golden_size
    is_pass = (1 - precision) <= 0.001
    return is_pass


# quant_mode 1 : get scale output shape -------------------------------
def get_sf_shape(num_tokens: int, hidden: int, num_per_channels: int, use_ue8m0: bool):
    num_scales = ceil_div(hidden, num_per_channels)
    return (num_tokens, num_scales)


# quant_mode 1 : calculate scale and inv_scale
def get_sf_and_inv(amax, round_sf: bool, use_ue8m0: bool):
    sf = amax / 448.0
    if not round_sf:
        return sf, 448.0 / amax
    bits = sf.view(np.uint32)
    exp = (bits >> 23) & 0xFF
    man_bits = bits & ((1 << 23) - 1)
    exp_scale = (exp - 127 + (man_bits != 0)).view(np.int32)
    if use_ue8m0:
        sf = (exp_scale + 127).astype(np.uint8)
    else:
        sf = ((exp_scale + 127) << 23).view(np.float32)
    return sf, ((127 - exp_scale) << 23).view(np.float32)


# quant_mode 1 : golden entry
def swiglu_fp8_quant_per_token_golden(x, weight=None, clamp_limit=None, round_scale=True, ue8m0_scale=True):
    num_per_channels = 32
    hidden_size = x.shape[-1]

    assert(hidden_size % (2 * num_per_channels) == 0)
    x = x.reshape(-1, hidden_size)

    last_dim = hidden_size // 2
    x0 = x.reshape(-1, last_dim * 2)[..., 0: last_dim]
    x1 = x.reshape(-1, last_dim * 2)[..., last_dim:]

    if weight is not None:
        weight = weight.reshape(-1, 1)

    if clamp_limit is not None:
        x0 = np.minimum(x0, clamp_limit)
        x1 = np.minimum(clamp_limit, np.maximum(x1, -clamp_limit))

    if weight is not None:
        y = silu(x0) * x1 * weight
    else:
        y = silu(x0) * x1

    num_tokens = y.shape[0]
    hidden = y.shape[1]

    absmax = np.max(np.abs(y.reshape(-1, num_per_channels)), axis=-1, keepdims=True)
    absmax = np.maximum(absmax, 1e-4)
    scale_shape = get_sf_shape(num_tokens, hidden, num_per_channels, ue8m0_scale)
    scale_dtype = np.uint8 if ue8m0_scale else np.float32
    scale = np.zeros(scale_shape, dtype=scale_dtype)

    absmax = absmax.reshape(num_tokens, -1)
    sf, inv_sf = get_sf_and_inv(absmax, round_scale, ue8m0_scale)

    y_tmp = y.reshape(-1, num_per_channels)
    y_tmp = y_tmp.astype(np.float32) * inv_sf.reshape(-1, 1)
    y_fp8 = y_tmp.astype(float8_e4m3fn).reshape(num_tokens, hidden)

    for i in range(y.shape[0]):
        for j in range(y.shape[1] // num_per_channels):
            scale[i, j] = sf[i, j]
    return y_fp8, scale, y


# fp4 e1m2 nibble encoder (no ml_dtypes support for this format).
# The e1m2 value grid was reverse-engineered against the kernel (VFComputeDataMXFP4 with
# U=fp4x2_e1m2_t, Cast<...CAST_RINT>) and the constants in swiglu_group_quant_base.h
# (FP4_E1M2_MAX_EXP=0x0000 -> f4Emax exponent 0, SPECIAL_VALUE_E1M2=0x007f). With a 1-bit
# exponent (bias 1) and 2 mantissa bits the representable magnitudes are uniform:
#   subnormals (exp field 0): 0, 0.25, 0.50, 0.75   (codes 0..3)
#   normals    (exp field 1): 1.00, 1.25, 1.50, 1.75 (codes 4..7)
# i.e. magnitude = code * 0.25, max magnitude 1.75 (saturating). The 4-bit nibble layout is
# [sign(1) | exp(1) | mantissa(2)], so nibble = round_half_even(|y|/0.25) clamped to 7, with the
# sign bit 0x8 OR'd in for negative inputs (signed zero kept, matching the hardware cast).
# Empirically this reproduces the kernel nibbles at ~0.998 (the residual is bf16 rounding at
# block boundaries, the same granularity tolerated by the e2m1 path).
E1M2_STEP = 0.25
E1M2_MAX_CODE = 7


def quantize_e1m2_nibble(y_scaled):
    mag = np.rint(np.abs(y_scaled) / E1M2_STEP).astype(np.int32)
    mag = np.clip(mag, 0, E1M2_MAX_CODE).astype(np.uint8)
    nibble = np.where(y_scaled < 0, mag | 0x8, mag).astype(np.uint8)
    return nibble


# quant_mode 1 + dst_type FLOAT4_E2M1/E1M2 : MxFp4 golden
# Mirrors the kernel math (swiglu_group_quant_base.h: VFComputeMaxExpMXFP4 / VFComputeScaleMXFP4 /
# VFComputeDataMXFP4) and the official swiglu_mx_quant reference:
#   per 32-element block: shared exponent comes from the bf16 exponent field of the block amax,
#   e8m0 scale = max(E_amax - f4Emax_exp, 0), data is multiplied by 2^(127 - e8m0) then cast to fp4
#   and packed two values per int8 byte (output last dim = splitD / 2).
def swiglu_mxfp4_quant_golden(x, dst_type, weight=None, clamp_limit=None):
    num_per_channels = 32
    hidden_size = x.shape[-1]
    assert hidden_size % (2 * num_per_channels) == 0
    x = x.reshape(-1, hidden_size).astype(np.float32)

    last_dim = hidden_size // 2
    x0 = x[:, 0:last_dim]
    x1 = x[:, last_dim:]
    if clamp_limit is not None:
        x0 = np.minimum(x0, clamp_limit)
        x1 = np.minimum(clamp_limit, np.maximum(x1, -clamp_limit))
    y = silu(x0) * x1
    if weight is not None:
        y = y * weight.reshape(-1, 1)

    num_tokens = y.shape[0]
    hidden = y.shape[1]
    n_scale = hidden // num_per_channels
    # f4Emax exponent (FP4_E2M1_BF16_MAX_EXP=0x0100 >> 7 = 2; FP4_E1M2_MAX_EXP=0x0000 >> 7 = 0)
    f4emax_exp = 2 if dst_type == 40 else 0

    y_block = y.reshape(num_tokens, n_scale, num_per_channels)
    amax = np.max(np.abs(y_block), axis=-1)  # (num_tokens, n_scale)

    # bf16 exponent field of amax
    amax_bf16 = amax.astype(bf16)
    bits = amax_bf16.view(np.uint16)
    e_amax = ((bits >> 7) & 0xFF).astype(np.int32)
    e8m0 = np.maximum(e_amax - f4emax_exp, 0).astype(np.uint8)  # e8m0 biased scale exponent

    inv_scale = np.exp2(127.0 - e8m0.astype(np.float32))  # 2^(127 - e8m0)
    y_scaled = y_block * inv_scale[..., None]

    if dst_type == 40:
        if not HAS_ML_FP4:
            raise RuntimeError("FLOAT4_E2M1 golden needs ml_dtypes.float4_e2m1fn")
        y_fp4 = y_scaled.astype(np.float32).astype(fp4_e2m1)
        nibble = y_fp4.view(np.uint8).reshape(num_tokens, hidden)
    elif dst_type == 41:
        # e1m2 has no ml_dtypes equivalent; use the reverse-engineered uniform-grid encoder.
        nibble = quantize_e1m2_nibble(y_scaled.astype(np.float32)).reshape(num_tokens, hidden)
    else:
        raise RuntimeError(f"MxFp4 golden only implements FLOAT4_E2M1/E1M2, got {dst_type}")

    # pack two fp4 nibbles per byte (low nibble first), matching the kernel's packed output layout
    packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).astype(np.uint8)  # (num_tokens, hidden/2)
    return packed, e8m0


@dataclass
class SwigluGroupQuantConfig:
    dst_type: torch.dtype
    quant_mode: int
    block_size: int = 0
    round_scale: bool = False
    output_origin: bool = False
    clamp_limit: float = None


# ======================== test start =========================
class TestCustomSwigluGroupQuant(TestCase):
    # ======================== test quant_mode 0 =========================
    def test_mode0_with_group_index(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x, group_index=None):
                npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                    x, group_index=group_index, dst_type=self.npu_config.dst_type,
                    quant_mode=self.npu_config.quant_mode, block_size=self.npu_config.block_size,
                    round_scale=self.npu_config.round_scale, output_origin=self.npu_config.output_origin,
                    clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin_out

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            split_d = d // 2
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 0
            scale_d = (split_d + 127) // 128
            np.random.seed(42)

            # construct input tensor
            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)
            group_index = torch.tensor([b * s // 2, b * s - b * s // 2 - 10]).to(torch.int64)

            # call golden function
            cpu_y_out, cpu_scale_out = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            # eager
            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

            # graph
            npu_config = SwigluGroupQuantConfig(dst_type_torch, quant_mode)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(x_npu, group_index=group_index_npu)

            # to CPU and compare
            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.cpu().float().numpy()

            # group_index filter
            real_bs = group_index.sum().item()
            npu_y_out_cpu = npu_y_out_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_y_out = cpu_y_out.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            npu_scale_cpu = npu_scale_cpu.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            cpu_scale_out = cpu_scale_out.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)

            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")

    def test_mode0(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x):
                npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                    x, dst_type=self.npu_config.dst_type, quant_mode=self.npu_config.quant_mode,
                    block_size=self.npu_config.block_size, round_scale=self.npu_config.round_scale,
                    output_origin=self.npu_config.output_origin, clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin_out

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 0
            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)

            cpu_y_out, cpu_scale_out = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode)

            x_npu = x.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

            npu_config = SwigluGroupQuantConfig(dst_type_torch, quant_mode)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(x_npu)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")

    def test_mode0_with_weight_and_clamp_limit(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x, weight=None):
                npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                    x, weight=weight, dst_type=self.npu_config.dst_type, quant_mode=self.npu_config.quant_mode,
                    block_size=self.npu_config.block_size, round_scale=self.npu_config.round_scale,
                    output_origin=self.npu_config.output_origin, clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin_out

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 0
            clamp_limit = 1.0
            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)
            weight = torch.tensor(np.random.uniform(-2, 2, (b * s, 1))).to(torch.float32)

            cpu_y_out, cpu_scale_out = swiglu_group_quant(
                x.numpy().astype(np.float32), dst_type, quant_mode, weight.numpy(), clamp_limit=clamp_limit)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            weight_npu = weight.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, weight=weight_npu, dst_type=dst_type_torch, quant_mode=quant_mode,
                clamp_limit=clamp_limit)

            npu_config = SwigluGroupQuantConfig(dst_type_torch, quant_mode, clamp_limit=clamp_limit)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(x_npu, weight=weight_npu)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")

    # ======================== test quant_mode 1 =========================
    def test_mode1_all_input(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x, weight=None, group_index=None):
                npu_y_out, npu_scale_out, npu_y_origin = torch.ops.custom.npu_swiglu_group_quant(
                    x, weight=weight, group_index=group_index, dst_type=self.npu_config.dst_type,
                    quant_mode=self.npu_config.quant_mode, block_size=self.npu_config.block_size,
                    round_scale=self.npu_config.round_scale, output_origin=self.npu_config.output_origin,
                    clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin

        bs = 8192
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 1
            clamp_limit = 10
            round_scale = True # mode 1 only support True
            output_origin = True
            block_size = 32

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2]).to(torch.int64)

            cpu_y_out, cpu_scale_out, cpu_y_origin = swiglu_fp8_quant_per_token_golden(
                x.numpy(), weight.numpy(), clamp_limit, round_scale)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            weight_npu = weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, weight=weight_npu, group_index=group_index_npu, dst_type=dst_type_torch,
                quant_mode=quant_mode, block_size=block_size, round_scale=round_scale,
                output_origin=output_origin, clamp_limit=clamp_limit)

            npu_config = SwigluGroupQuantConfig(
                dst_type_torch, quant_mode, block_size, round_scale, output_origin, clamp_limit)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(
                x_npu, weight=weight_npu, group_index=group_index_npu)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.view(torch.uint8).cpu().numpy()
            npu_y_origin_cpu = npu_y_origin_out.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=1, equal_nan=True)
            y_origin_out_close = np.allclose(npu_y_origin_cpu.reshape(-1), cpu_y_origin.reshape(-1),
                                             rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")
            self.assertTrue(y_origin_out_close, f"y_origin_out_close precision compare fail")

    def test_mode1_with_group_index(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x, weight=None, group_index=None):
                npu_y_out, npu_scale_out, npu_y_origin = torch.ops.custom.npu_swiglu_group_quant(
                    x, weight=weight, group_index=group_index, dst_type=self.npu_config.dst_type,
                    quant_mode=self.npu_config.quant_mode, block_size=self.npu_config.block_size,
                    round_scale=self.npu_config.round_scale, output_origin=self.npu_config.output_origin,
                    clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin

        bs = 1024
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 1
            clamp_limit = 10
            round_scale = True # mode 1 only support True
            output_origin = True
            block_size = 32
            split_d = d // 2
            scale_d = (split_d + 127) // 128

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2 - 20]).to(torch.int64)

            cpu_y_out, cpu_scale_out, cpu_y_origin = swiglu_fp8_quant_per_token_golden(
                x.numpy(), weight.numpy(), clamp_limit, round_scale)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            weight_npu = weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, weight=weight_npu, group_index=group_index_npu, dst_type=dst_type_torch,
                quant_mode=quant_mode, block_size=block_size, round_scale=round_scale,
                output_origin=output_origin, clamp_limit=clamp_limit)

            npu_config = SwigluGroupQuantConfig(
                dst_type_torch, quant_mode, block_size, round_scale, output_origin, clamp_limit)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(
                x_npu, weight=weight_npu, group_index=group_index_npu)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.view(torch.uint8).cpu().numpy()
            npu_y_origin_cpu = npu_y_origin_out.cpu().float().numpy()

            # group_index filter
            real_bs = group_index.sum().item()
            npu_y_out_cpu = npu_y_out_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_y_out = cpu_y_out.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            npu_scale_cpu = npu_scale_cpu.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            cpu_scale_out = cpu_scale_out.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            npu_y_origin_cpu = npu_y_origin_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_y_origin = cpu_y_origin.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)

            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=1, equal_nan=True)
            y_origin_out_close = np.allclose(npu_y_origin_cpu.reshape(-1), cpu_y_origin.reshape(-1),
                                             rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")
            self.assertTrue(y_origin_out_close, f"y_origin_out_close precision compare fail")

    def test_mode1_without_y_origin(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        class Network(nn.Module):
            def __init__(self, npu_config):
                super(Network, self).__init__()
                self.npu_config = npu_config

            def forward(self, x, weight=None, group_index=None):
                npu_y_out, npu_scale_out, npu_y_origin = torch.ops.custom.npu_swiglu_group_quant(
                    x, weight=weight, group_index=group_index, dst_type=self.npu_config.dst_type,
                    quant_mode=self.npu_config.quant_mode, block_size=self.npu_config.block_size,
                    round_scale=self.npu_config.round_scale, output_origin=self.npu_config.output_origin,
                    clamp_limit=self.npu_config.clamp_limit)
                return npu_y_out, npu_scale_out, npu_y_origin

        bs = 8192
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 1
            clamp_limit = 10
            round_scale = True # mode 1 only support True
            output_origin = False
            block_size = 32

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2]).to(torch.int64)

            cpu_y_out, cpu_scale_out, cpu_y_origin = swiglu_fp8_quant_per_token_golden(
                x.numpy(), weight.numpy(), clamp_limit, round_scale)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            weight_npu = weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, weight=weight_npu, group_index=group_index_npu, dst_type=dst_type_torch,
                quant_mode=quant_mode, block_size=block_size, round_scale=round_scale,
                output_origin=output_origin, clamp_limit=clamp_limit)

            npu_config = SwigluGroupQuantConfig(
                dst_type_torch, quant_mode, block_size, round_scale, output_origin, clamp_limit)
            npu_mode = Network(npu_config).to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_y_out, npu_scale_out, npu_y_origin_out = npu_mode(
                x_npu, weight=weight_npu, group_index=group_index_npu)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.view(torch.uint8).cpu().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=1, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")

    def test_mode1_ue8m0(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 512
        d_list = [4096, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 1
            round_scale = True # mode 1 only support True
            output_origin = True
            block_size = 32

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)

            cpu_y_out, cpu_scale_out, cpu_y_origin = swiglu_fp8_quant_per_token_golden(
                x.numpy().astype(np.float32), round_scale=round_scale)

            x_npu = x.to("npu:%s" % DEVICE_ID)

            npu_y_out, npu_scale_out, npu_y_origin_out = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=dst_type_torch, quant_mode=quant_mode, block_size=block_size,
                round_scale=round_scale, output_origin=output_origin)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.view(torch.uint8).cpu().numpy()
            npu_y_origin_cpu = npu_y_origin_out.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=1, equal_nan=True)
            y_origin_out_close = np.allclose(npu_y_origin_cpu.reshape(-1), cpu_y_origin.reshape(-1),
                                             rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(y_out_close, f"y_out precision compare fail")
            self.assertTrue(scale_out_close, f"scale_out_close precision compare fail")
            self.assertTrue(y_origin_out_close, f"y_origin_out_close precision compare fail")

    # ======================== test quant_mode 1 : MxFp4 =========================
    @unittest.skipUnless(HAS_TORCH_FP4, FP4_SKIP_MSG)
    def test_mode1_mxfp4_e2m1(self):
        # FLOAT4_E2M1 is selected via the standard dst_type=torch.float4_e2m1fn_x2 (torch>=2.8). The
        # kernel output is packed (two fp4 nibbles per byte); the native packed dtype carries that layout.
        if not HAS_ML_FP4:
            self.skipTest("ml_dtypes lacks float4_e2m1fn; cannot build MxFp4 golden")
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 512
        # d must be divisible by 256; small d=1024 is included for fast iteration.
        d_list = [1024, 4096, 7168]
        dst_type = 40  # FLOAT4_E2M1 (golden integer code)
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        quant_mode = 1
        round_scale = True   # mx_quant only supports True
        block_size = 32
        for d in d_list:
            np.random.seed(42)
            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)

            cpu_y_packed, cpu_scale = swiglu_mxfp4_quant_golden(x.numpy().astype(np.float32), dst_type)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            # fp4 is selected through the standard dst_type ScalarType (no dst_type_code).
            npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=TORCH_FP4_E2M1, quant_mode=quant_mode, block_size=block_size,
                round_scale=round_scale)

            npu_y_packed = npu_y_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)
            npu_scale = npu_scale_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)

            # y compare: packed fp4 bytes, allow small mismatch ratio (fp4 rounding granularity)
            y_diff = np.abs(npu_y_packed.astype(np.int16) - cpu_y_packed.astype(np.int16))
            y_pass_ratio = np.mean(y_diff == 0)
            # e8m0 scale is NOT bit-exact vs golden: a per-block ±1 difference is expected and tolerated.
            # Root cause is an ordering difference in deriving the shared exponent: the kernel takes the
            # bf16 exponent field of each element first and then max-reduces over the 32-elem block
            # (VFComputeMaxExpMXFP4), while the golden takes the block amax first and only then reads its
            # bf16 exponent. The (y, scale) pair stays self-consistent, and atol=1 matches the already
            # published fp8-mx cases' standard (test_mode1_* use atol=1 on the e8m0 scale).
            scale_close = np.allclose(npu_scale.reshape(-1), cpu_scale.reshape(-1),
                                      rtol=0, atol=1, equal_nan=True)

            self.assertTrue(y_pass_ratio > 0.99, f"y_out packed match ratio too low: {y_pass_ratio}")
            self.assertTrue(scale_close, f"scale_out (e8m0) precision compare fail for d={d}")

    @unittest.skip("FLOAT4_E1M2 has no torch ScalarType; only selectable via the aclnn/op-def integer "
                   "dst_type path, not from PyTorch. Kept for documentation/reference only.")
    def test_mode1_mxfp4_e1m2(self):
        # FLOAT4_E1M2 (dst_type code 41) has NO torch dtype (neither torch nor ml_dtypes expose e1m2),
        # so it cannot be selected through the standard dst_type ScalarType. After dropping the
        # dst_type_code parameter this case is unreachable from PyTorch and is skipped unconditionally.
        # The e1m2 sub-type remains available only through the aclnn/op-def integer dst_type path.
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 512
        # d must be divisible by 256.
        d_list = [1024, 4096, 7168]
        dst_type = 41  # FLOAT4_E1M2
        quant_mode = 1
        round_scale = True   # mx_quant only supports True
        block_size = 32
        for d in d_list:
            np.random.seed(42)
            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)

            cpu_y_packed, cpu_scale = swiglu_mxfp4_quant_golden(x.numpy().astype(np.float32), dst_type)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=torch.float8_e4m3fn, quant_mode=quant_mode, block_size=block_size,
                round_scale=round_scale)

            npu_y_packed = npu_y_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)
            npu_scale = npu_scale_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)

            # nibble-level match (unpack both bytes), allow fp4 rounding granularity mismatch
            def unpack(packed):
                lo = (packed & 0xF).astype(np.uint8)
                hi = ((packed >> 4) & 0xF).astype(np.uint8)
                out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
                out[:, 0::2] = lo
                out[:, 1::2] = hi
                return out

            npu_nib = unpack(npu_y_packed)
            cpu_nib = unpack(cpu_y_packed)
            nib_pass_ratio = np.mean(npu_nib == cpu_nib)
            # e8m0 scale is NOT bit-exact vs golden: a per-block ±1 difference is expected and tolerated.
            # Root cause is an ordering difference in deriving the shared exponent: the kernel takes the
            # bf16 exponent field of each element first and then max-reduces over the 32-elem block
            # (VFComputeMaxExpMXFP4), while the golden takes the block amax first and only then reads its
            # bf16 exponent. The (y, scale) pair stays self-consistent, and atol=1 matches the already
            # published fp8-mx cases' standard (test_mode1_* use atol=1 on the e8m0 scale).
            scale_close = np.allclose(npu_scale.reshape(-1), cpu_scale.reshape(-1),
                                      rtol=0, atol=1, equal_nan=True)

            self.assertTrue(nib_pass_ratio > 0.99,
                            f"e1m2 nibble match ratio too low: {nib_pass_ratio} for d={d}")
            self.assertTrue(scale_close, f"scale_out (e8m0) precision compare fail for d={d}")

    @unittest.skipUnless(HAS_TORCH_FP4, FP4_SKIP_MSG)
    def test_mode1_mxfp4_small_d(self):
        # Coverage for small/non-16-aligned scale columns with multi-row (rowFactor>1) full-load
        # tiles. For d=256 -> splitD=128 -> scaleDFactor=splitD/32=4; d=512 -> scaleDFactor=8.
        # Neither is 16-aligned, so this stresses the e8m0 scale CopyOut (PaddingMode::Compact path).
        # bs is large relative to the core count so the tiling packs many rows per loop (rowFactor>1).
        if not HAS_ML_FP4:
            self.skipTest("ml_dtypes lacks float4_e2m1fn; cannot build MxFp4 golden")
        torch_npu.npu.set_device(int(DEVICE_ID))

        quant_mode = 1
        round_scale = True
        block_size = 32
        # only e2m1 (40) is selectable from PyTorch; e1m2 (41) has no torch dtype.
        for dst_type in (40,):
            for d in (256, 512):
                for bs in (64, 512):
                    split_d = d // 2
                    scale_d = split_d // 32
                    np.random.seed(42)
                    x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)

                    cpu_y_packed, cpu_scale = swiglu_mxfp4_quant_golden(
                        x.numpy().astype(np.float32), dst_type)

                    x_npu = x.to("npu:%s" % DEVICE_ID)
                    npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
                        x_npu, dst_type=TORCH_FP4_E2M1, quant_mode=quant_mode,
                        block_size=block_size, round_scale=round_scale)

                    npu_y_packed = npu_y_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)
                    npu_scale = npu_scale_out.view(torch.uint8).cpu().numpy().reshape(bs, scale_d)
                    cpu_scale = cpu_scale.reshape(bs, scale_d)

                    y_diff = np.abs(npu_y_packed.astype(np.int16) - cpu_y_packed.astype(np.int16))
                    y_pass_ratio = np.mean(y_diff == 0)
                    # e8m0 scale is NOT bit-exact vs golden: a per-block ±1 difference is expected and
                    # tolerated. Root cause is an ordering difference in deriving the shared exponent: the
                    # kernel takes the bf16 exponent field of each element first and then max-reduces over
                    # the 32-elem block (VFComputeMaxExpMXFP4), while the golden takes the block amax first
                    # and only then reads its bf16 exponent. The (y, scale) pair stays self-consistent, and
                    # atol=1 matches the already published fp8-mx cases' standard (atol=1 on the e8m0 scale).
                    scale_close = np.allclose(npu_scale.reshape(-1), cpu_scale.reshape(-1),
                                              rtol=0, atol=1, equal_nan=True)

                    self.assertTrue(
                        y_pass_ratio > 0.99,
                        f"small-d y match ratio too low: {y_pass_ratio} dst={dst_type} d={d} bs={bs}")
                    self.assertTrue(
                        scale_close,
                        f"small-d scale mismatch dst={dst_type} d={d} bs={bs} scale_d={scale_d}")


    @unittest.skipUnless(HAS_TORCH_FP4, FP4_SKIP_MSG)
    def test_mode1_mxfp4_multi_dloop(self):
        # Exercises the multi d-loop tiling path (dLoop>1). For splitD=d/2 large enough that a single
        # row cannot be fully loaded into UB, the tiling splits d into chunks. The previous
        # CalcMxFp4QuantOpTiling produced dFactor_=0 (CeilDiv(splitD,0) div-by-zero) on this path; the
        # fix grows dFactor from the smallest mx block and keeps scaleCol_ at the full-row width so the
        # per-chunk e8m0 scale CopyOut lands at the right GM offset. d=49152 -> splitD=24576 > UB budget.
        if not HAS_ML_FP4:
            self.skipTest("ml_dtypes lacks float4_e2m1fn; cannot build MxFp4 golden")
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 64
        d = 49152  # divisible by 256; splitD=24576 forces dLoop>1 on Ascend950 UB
        split_d = d // 2
        scale_d = split_d // 32
        quant_mode = 1
        round_scale = True
        block_size = 32
        # only e2m1 (40) is selectable from PyTorch; e1m2 (41) has no torch dtype.
        for dst_type in (40,):
            np.random.seed(42)
            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)

            cpu_y_packed, cpu_scale = swiglu_mxfp4_quant_golden(x.numpy().astype(np.float32), dst_type)

            x_npu = x.to("npu:%s" % DEVICE_ID)
            npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=TORCH_FP4_E2M1, quant_mode=quant_mode, block_size=block_size,
                round_scale=round_scale)

            npu_y_packed = npu_y_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)
            npu_scale = npu_scale_out.view(torch.uint8).cpu().numpy().reshape(bs, scale_d)
            cpu_scale = cpu_scale.reshape(bs, scale_d)

            y_diff = np.abs(npu_y_packed.astype(np.int16) - cpu_y_packed.astype(np.int16))
            y_pass_ratio = np.mean(y_diff == 0)
            # e8m0 scale is NOT bit-exact vs golden: a per-block ±1 difference is expected and tolerated.
            # Root cause is an ordering difference in deriving the shared exponent: the kernel takes the
            # bf16 exponent field of each element first and then max-reduces over the 32-elem block
            # (VFComputeMaxExpMXFP4), while the golden takes the block amax first and only then reads its
            # bf16 exponent. The (y, scale) pair stays self-consistent, and atol=1 matches the already
            # published fp8-mx cases' standard (test_mode1_* use atol=1 on the e8m0 scale).
            scale_close = np.allclose(npu_scale.reshape(-1), cpu_scale.reshape(-1),
                                      rtol=0, atol=1, equal_nan=True)

            self.assertTrue(y_pass_ratio > 0.99,
                            f"multi-dloop y match ratio too low: {y_pass_ratio} dst={dst_type}")
            self.assertTrue(scale_close, f"multi-dloop scale mismatch dst={dst_type}")

    @unittest.skipUnless(HAS_TORCH_FP4, FP4_SKIP_MSG)
    def test_mode1_mxfp4_bf16_input(self):
        # bf16-input coverage for the MxFp4 path. All other fp4 cases use fp16 input, which exercises the
        # kernel's `T == half` branch (FP16Convert) in VFComputeMaxExpMXFP4 / VFComputeDataMXFP4. bf16
        # input takes the other branch (the `else` path: no FP16Convert, direct bf16 exponent extraction
        # and bf16 mul), which was previously untested. dst_type E2M1(40)/E1M2(41), d in {1024, 4096}.
        # The op def registers bf16-input fp4-output combos (idx 9/11), so a dedicated kernel binary exists.
        if not HAS_ML_FP4:
            self.skipTest("ml_dtypes lacks float4_e2m1fn; cannot build MxFp4 golden")
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 512
        d_list = [1024, 4096]
        quant_mode = 1
        round_scale = True   # mx_quant only supports True
        block_size = 32

        def unpack(packed):
            lo = (packed & 0xF).astype(np.uint8)
            hi = ((packed >> 4) & 0xF).astype(np.uint8)
            out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
            out[:, 0::2] = lo
            out[:, 1::2] = hi
            return out

        # only e2m1 (40) is selectable from PyTorch; e1m2 (41) has no torch dtype.
        for dst_type in (40,):
            for d in d_list:
                np.random.seed(42)
                # Build the input as bf16, then feed the bf16-rounded values (upcast to fp32) to the golden
                # so the reference matches the kernel's bf16 input precision. torch bf16 tensors cannot be
                # converted to numpy directly, hence the .float().numpy() round-trip.
                x_bf16 = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.bfloat16)
                x_np = x_bf16.float().numpy().astype(np.float32)

                cpu_y_packed, cpu_scale = swiglu_mxfp4_quant_golden(x_np, dst_type)

                x_npu = x_bf16.to("npu:%s" % DEVICE_ID)
                npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
                    x_npu, dst_type=TORCH_FP4_E2M1, quant_mode=quant_mode, block_size=block_size,
                    round_scale=round_scale)

                npu_y_packed = npu_y_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)
                npu_scale = npu_scale_out.view(torch.uint8).cpu().numpy().reshape(bs, -1)

                # nibble-level match (unpack both bytes), allow fp4 rounding granularity mismatch; the
                # bf16 input path inherits the same fp4 granularity tolerance as the fp16-input fp4 cases.
                npu_nib = unpack(npu_y_packed)
                cpu_nib = unpack(cpu_y_packed)
                nib_pass_ratio = np.mean(npu_nib == cpu_nib)
                # e8m0 scale is NOT bit-exact vs golden: a per-block ±1 difference is expected and tolerated.
                # Root cause is an ordering difference in deriving the shared exponent: the kernel takes the
                # bf16 exponent field of each element first and then max-reduces over the 32-elem block
                # (VFComputeMaxExpMXFP4), while the golden takes the block amax first and only then reads its
                # bf16 exponent. The (y, scale) pair stays self-consistent, and atol=1 matches the already
                # published fp8-mx cases' standard (test_mode1_* use atol=1 on the e8m0 scale).
                scale_close = np.allclose(npu_scale.reshape(-1), cpu_scale.reshape(-1),
                                          rtol=0, atol=1, equal_nan=True)

                self.assertTrue(
                    nib_pass_ratio > 0.99,
                    f"bf16-input fp4 nibble match ratio too low: {nib_pass_ratio} dst={dst_type} d={d}")
                self.assertTrue(
                    scale_close, f"bf16-input fp4 scale mismatch dst={dst_type} d={d}")

    def test_mode1_fp8mx_multi_dloop(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        bs = 64
        d = 49152  # divisible by 256
        dst_type = 36
        dst_type_torch = torch.float8_e4m3fn
        quant_mode = 1
        round_scale = True
        block_size = 32
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]

        np.random.seed(42)
        x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
        cpu_y_out, cpu_scale_out, cpu_y_origin = swiglu_fp8_quant_per_token_golden(
            x.numpy().astype(np.float32), round_scale=round_scale)
        x_npu = x.to("npu:%s" % DEVICE_ID)

        for output_origin in (False, True):
            npu_y_out, npu_scale_out, npu_y_origin = torch.ops.custom.npu_swiglu_group_quant(
                x_npu, dst_type=dst_type_torch, quant_mode=quant_mode, block_size=block_size,
                round_scale=round_scale, output_origin=output_origin)

            npu_y_out_cpu = npu_y_out.cpu()
            npu_scale_cpu = npu_scale_out.view(torch.uint8).cpu().numpy()
            y_out_close = requantize_compare(
                torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
            # atol=1 on the e8m0 scale: same per-block ±1 ordering tolerance as the published fp8-mx cases.
            scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                          rtol=0.0001, atol=1, equal_nan=True)
            self.assertTrue(y_out_close, f"fp8-mx multi-dloop y_out fail (output_origin={output_origin})")
            self.assertTrue(scale_out_close,
                            f"fp8-mx multi-dloop scale_out fail (output_origin={output_origin})")
            if output_origin:
                npu_y_origin_cpu = npu_y_origin.cpu().float().numpy()
                y_origin_close = np.allclose(npu_y_origin_cpu.reshape(-1), cpu_y_origin.reshape(-1),
                                             rtol=0.01, atol=0.01, equal_nan=True)
                self.assertTrue(y_origin_close, "fp8-mx multi-dloop y_origin fail (output_origin=True)")

    def test_mode0_block_multi_dloop(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        b = 1
        s = 64
        d = 65536  # divisible by 256
        split_d = d // 2
        scale_d = (split_d + 127) // 128
        dst_type = 35
        dst_type_torch = torch.float8_e5m2
        quant_mode = 0
        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)
        cpu_y_out, cpu_scale_out = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode)

        x_npu = x.to("npu:%s" % DEVICE_ID)
        npu_y_out, npu_scale_out, _ = torch.ops.custom.npu_swiglu_group_quant(
            x_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

        npu_y_out_cpu = npu_y_out.cpu()
        npu_scale_cpu = npu_scale_out.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        y_out_close = requantize_compare(torch.from_numpy(cpu_y_out.view(np.int8)), npu_y_out_cpu, dst_type_str)
        scale_out_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scale_out.reshape(-1),
                                      rtol=0.0001, atol=0.0001, equal_nan=True)
        self.assertTrue(y_out_close, "block-quant multi-dloop y_out precision compare fail")
        self.assertTrue(scale_out_close, "block-quant multi-dloop scale_out precision compare fail")


if __name__ == "__main__":
    run_tests()
