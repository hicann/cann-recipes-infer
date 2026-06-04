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
}


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


if __name__ == "__main__":
    run_tests()
