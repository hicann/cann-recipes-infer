# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import numpy
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from ml_dtypes import bfloat16 as bf16
from ml_dtypes import float8_e4m3fn, float8_e5m2
import re
import copy

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

def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e4m3fn dtype")

def numpy_float8_e5m2():
    try:
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e5m2 dtype")

def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed to support float8_e8m0 dtype")

def swiglu(x, clamp_value=0.0):
    last_dim = x.shape[-1] // 2
    x0 = x.reshape(-1, last_dim * 2)[..., 0: last_dim]
    x1 = x.reshape(-1, last_dim * 2)[..., last_dim:]
    if clamp_value != 0.0:
        x0 = np.minimum(x0, clamp_value)
        x1 = np.minimum(clamp_value, np.maximum(x1, -clamp_value))
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
            block = x_padded[i * row_block_size: (i + 1) * row_block_size, j * col_block_size: (j + 1) * col_block_size]
            result[i, j] = np.max(block)
    
    return result

# quant_mode == 1 ---------------------------------------
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
            scale_expanded[i * row_block_size: (i + 1) * row_block_size, j * col_block_size: (j + 1) * col_block_size] = scale[i, j]
    
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

# quant_mode == 2 -------------------------------------------
def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
    fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = numpy.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape

def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int,
                               orig_shape: tuple, padded_shape: tuple):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = numpy.squeeze(fp_array, axis=axis + 1)
    return fp_array

def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(numpy.log2(max_norm))
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
    res = numpy.floor(
        numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_calculate_share_exp_nv(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    import numpy
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(numpy.log2(max_norm))
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True).astype(numpy.float32)
    s_fp32 =  fp_abs_max / max_norm
    binary_ints = numpy.array(s_fp32.view(numpy.uint32))
    exponent_mask = numpy.uint32(0x7F800000)
    mantissa_mask = numpy.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(numpy.int16)
    mantissas = (binary_ints & mantissa_mask)
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    condition_2 = (exponents_int16 == 0) & (mantissas > 2 ** 22)
    exponents_int16 = numpy.where((condition_1|condition_2), exponents_int16 + 1, exponents_int16)
    res = (exponents_int16 - 127).astype(numpy.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res

def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
    if round_mode in ("rint", "even"):
        fp_array = numpy.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = numpy.signbit(fp_array)
        rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
        fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = numpy.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = numpy.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = numpy.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array

def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                    mx_ele_dtype: str, round_mode: str):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r'e(\d+)m(\d+)', mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2 ** share_exp)
    private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
                              ).astype(fp_array.dtype, copy=False)
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    max_norm = get_dtype_range(mx_dtype)[1]
    numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret

def pad_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
    if not isinstance(tensor, numpy.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = numpy.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor

def interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
    if not isinstance(tensor, numpy.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = (
        shape[:axis] +
        [group_length, 2] +
        shape[axis+1:])
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis+1)) +
        list(range(axis + 2, len(new_shape))) +
        [axis+1,])

    transposed = reshaped.transpose(transpose_order)
    return transposed

def mx_quantize(fp_array: numpy.ndarray, mx_ele_dtype: str = "float4_e2m1",
                axis: int = -1, block_size: int = 32, round_mode: str = "rint", scale_alg: int = 0) -> tuple:

    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array,
                                            scale_axis=axis + 1,
                                            mx_ele_dtype=mx_ele_dtype)
    else:
        share_exp = _mx_calculate_share_exp_nv(fp_array,
                                               scale_axis=axis + 1,
                                               mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    
    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)

    return scale_array, ele_array


def swiglu_group_quant(x, dst_type, quant_mode, topk_weight=None, clamp_value=0.0):
    y = swiglu(x, clamp_value)
    if topk_weight is not None:
        y = y * topk_weight
    if quant_mode == 1:
        out, scale = dynamic_block_quant(y, dst_type)
    elif quant_mode == 2:
        y = y.astype(bf16).astype(np.float32)
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        scale, out = mx_quantize(y, mx_ele_dtype=dst_type_str, axis=-1, block_size=32, round_mode="rint", scale_alg=0)
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

# quant_mode 3 : get scale output shape -------------------------------
def get_sf_shape(num_tokens: int, hidden: int, num_per_channels: int, use_ue8m0: bool):
    num_scales = ceil_div(hidden, num_per_channels)
    return (num_tokens, num_scales)

# quant_mode 3 : calculate scale and inv_scale
def get_sf_and_inv(amax: float, round_sf: bool, use_ue8m0: bool):
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

# quant_mode 3 : golden entry
def swiglu_fp8_quant_per_token_golden(x, topk_weight, clamp_value, round_scale, ue8m0_scale):
    num_per_channels = 128
    hidden_size = x.shape[-1]

    assert(hidden_size % (2 * num_per_channels) == 0)
    x = x.reshape(-1, hidden_size)

    last_dim = hidden_size // 2
    x0 = x.reshape(-1, last_dim * 2)[..., 0: last_dim]
    x1 = x.reshape(-1, last_dim * 2)[..., last_dim:]

    topk_weight = topk_weight.reshape(-1, 1)

    if clamp_value != 0.0:
        x0 = np.minimum(x0, clamp_value)
        x1 = np.minimum(clamp_value, np.maximum(x1, -clamp_value))

    y = silu(x0) * x1 * topk_weight

    num_tokens = y.shape[0]
    hidden = y.shape[1]

    absmax = np.max(np.abs(y.reshape(-1, 128)), axis = -1, keepdims = True)
    absmax = np.maximum(absmax, 1e-4)
    scale_shape = get_sf_shape(num_tokens, hidden, num_per_channels, ue8m0_scale)
    scale_dtype = np.uint8 if ue8m0_scale else np.float32
    scale = np.zeros(scale_shape, dtype=scale_dtype)

    absmax = absmax.reshape(num_tokens, -1)
    sf, inv_sf = get_sf_and_inv(absmax, round_scale, ue8m0_scale)

    y_tmp = y.reshape(-1, 128)
    y_tmp = y_tmp.astype(np.float32) * inv_sf.reshape(-1, 1)
    y_fp8 = y_tmp.astype(float8_e4m3fn).reshape(num_tokens, hidden)

    for i in range(y.shape[0]):
        for j in range(y.shape[1] // 128):
            scale[i, j] = sf[i, j]
    return y_fp8, scale, y

# ======================== test start =========================
class TestCustomSwigluGroupQuant(TestCase):
    # ======================== test quant_mode 1 =========================
    def test_mode1_with_group_index(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, dst_type, quant_mode):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, dst_type=dst_type, quant_mode=quant_mode)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            split_d = d // 2
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 1
            scale_d = (split_d + 127) // 128
            np.random.seed(42)

            # construct input tensor
            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)
            group_index = torch.tensor([b * s // 2, b * s - b * s // 2 - 10]).to(torch.int64)

            # call golden function
            cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            # eager
            npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

            # graph
            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, dst_type_torch, quant_mode)

            # to CPU and compare
            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scleOut.cpu().float().numpy()

            # group_index filter
            real_bs = group_index.sum().item()
            npu_yOut_cpu = npu_yOut_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_yOut = cpu_yOut.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            npu_scale_cpu = npu_scale_cpu.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            cpu_scaleOut = cpu_scaleOut.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)

            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    def test_mode1(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, dst_type, quant_mode):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, dst_type=dst_type, quant_mode=quant_mode)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 1
            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)

            cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, dst_type_torch, quant_mode)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scleOut.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    def test_mode1_with_topk_weight_and_clamp_value(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, dst_type, quant_mode, clamp_value):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, topk_weight=topk_weight, dst_type=dst_type, quant_mode=quant_mode, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 8192
        d_list = [4096, 7168, 12288]
        for d in d_list:
            dst_type = 35
            dst_type_torch = torch.float8_e5m2
            quant_mode = 1
            clamp_value = 1.0
            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (b, s, d))).to(torch.float16)
            topk_weight = torch.tensor(np.random.uniform(-2, 2, (b * s, 1))).to(torch.float32)

            cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.numpy().astype(np.float32), dst_type, quant_mode, topk_weight.numpy(), clamp_value=clamp_value)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, topk_weight=topk_weight_npu, dst_type=dst_type_torch, quant_mode=quant_mode, clamp_value=clamp_value)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight_npu, dst_type_torch, quant_mode, clamp_value)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scleOut.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    # ======================== test quant_mode 2 =========================
    def test_mode2(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, dst_type, quant_mode):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, dst_type=dst_type, quant_mode=quant_mode)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 96
        d = 4096
        dst_type = 36
        dst_type_torch = torch.float8_e4m3fn
        quant_mode = 2
        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-100, 10, (b, s, d))).to(torch.bfloat16)

        cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.float().numpy().astype(np.float32), dst_type, quant_mode)
        
        x_npu = x.to("npu:%s" % DEVICE_ID)

        npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, dst_type_torch, quant_mode)

        npu_yOut_cpu = npu_yOut.cpu()
        npu_scale_cpu = npu_scleOut.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
        scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.astype(np.float32).reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

        self.assertTrue(yOut_close, f"yOut precision compare fail")
        self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")
    
    def test_mode2_with_group_index(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, group_index, dst_type, quant_mode):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, group_index=group_index, dst_type=dst_type, quant_mode=quant_mode)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 96
        bs = b * s
        d = 4096
        split_d = d // 2
        scale_d = (split_d + 127) // 128
        dst_type = 36
        dst_type_torch = torch.float8_e4m3fn
        quant_mode = 2
        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-100, 10, (b, s, d))).to(torch.bfloat16)
        group_index = torch.tensor([b * s // 2, b * s - b * s // 2 - 20]).to(torch.int64)

        cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.float().numpy().astype(np.float32), dst_type, quant_mode)
        
        x_npu = x.to("npu:%s" % DEVICE_ID)
        group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

        npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, group_index_npu, dst_type_torch, quant_mode)

        npu_yOut_cpu = npu_yOut.cpu()
        npu_scale_cpu = npu_scleOut.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]

        # group_index filter
        real_bs = group_index.sum().item()
        npu_yOut_cpu = npu_yOut_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
        cpu_yOut = cpu_yOut.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
        npu_scale_cpu = npu_scale_cpu.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
        cpu_scaleOut = cpu_scaleOut.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)

        yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
        scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.astype(np.float32).reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

        self.assertTrue(yOut_close, f"yOut precision compare fail")
        self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    def test_mode2_with_topk_weight_and_clamp_value(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, dst_type, quant_mode, clamp_value):
                npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x, topk_weight=topk_weight, dst_type=dst_type, quant_mode=quant_mode, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOriginOut

        b = 1
        s = 96
        d = 4096
        dst_type = 36
        dst_type_torch = torch.float8_e4m3fn
        quant_mode = 2
        clamp_value = 5.0
        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-100, 10, (b, s, d))).to(torch.bfloat16)
        topk_weight = torch.tensor(np.random.uniform(-2, 2, (b * s, 1))).to(torch.float32)

        cpu_yOut, cpu_scaleOut = swiglu_group_quant(x.float().numpy().astype(np.float32), dst_type, quant_mode, topk_weight.numpy(), clamp_value)
        
        x_npu = x.to("npu:%s" % DEVICE_ID)
        topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)

        npu_yOut, npu_scleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, topk_weight=topk_weight_npu, dst_type=dst_type_torch, quant_mode=quant_mode, clamp_value=clamp_value)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_yOut, npu_scleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight_npu, dst_type_torch, quant_mode, clamp_value)

        npu_yOut_cpu = npu_yOut.cpu()
        npu_scale_cpu = npu_scleOut.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
        scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.astype(np.float32).reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

        self.assertTrue(yOut_close, f"yOut precision compare fail")
        self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    # ======================== test quant_mode 3 =========================
    def test_mode3_all_input(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, group_index, dst_type, quant_mode, group_size, round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value):
                npu_yOut, npu_scaleOut, npu_yOrigin = torch.ops.custom.npu_swiglu_group_quant(x, topk_weight=topk_weight, group_index=group_index, dst_type=dst_type, quant_mode=quant_mode, group_size=group_size, \
                    round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOrigin

        bs = 8192
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 3
            clamp_value = 10
            round_scale = False
            ue8m0_scale = False
            output_origin = True
            group_list_type = 0
            group_size = 128

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            topk_weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2]).to(torch.int64)

            cpu_yOut, cpu_scaleOut, cpu_yOrigin = swiglu_fp8_quant_per_token_golden(x.numpy(), topk_weight.numpy(), clamp_value, round_scale, ue8m0_scale)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scaleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scaleOut.cpu().float().numpy()
            npu_yOrigin_cpu = npu_yOriginOut.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)
            yOriginOut_close = np.allclose(npu_yOrigin_cpu.reshape(-1), cpu_yOrigin.reshape(-1), rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")
            self.assertTrue(yOriginOut_close, f"yOriginOut_close precision compare fail")

    def test_mode3_with_group_index(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, group_index, dst_type, quant_mode, group_size, round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value):
                npu_yOut, npu_scaleOut, npu_yOrigin = torch.ops.custom.npu_swiglu_group_quant(x, topk_weight=topk_weight, group_index=group_index, dst_type=dst_type, quant_mode=quant_mode, group_size=group_size, \
                    round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOrigin

        bs = 1024
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 3
            clamp_value = 10
            round_scale = False
            ue8m0_scale = False
            output_origin = True
            group_list_type = 0
            group_size = 128
            split_d = d // 2
            scale_d = (split_d + 127) // 128

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            topk_weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2 - 20]).to(torch.int64)

            cpu_yOut, cpu_scaleOut, cpu_yOrigin = swiglu_fp8_quant_per_token_golden(x.numpy(), topk_weight.numpy(), clamp_value, round_scale, ue8m0_scale)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scaleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scaleOut.cpu().float().numpy()
            npu_yOrigin_cpu = npu_yOriginOut.cpu().float().numpy()

            # group_index filter
            real_bs = group_index.sum().item()
            npu_yOut_cpu = npu_yOut_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_yOut = cpu_yOut.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            npu_scale_cpu = npu_scale_cpu.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            cpu_scaleOut = cpu_scaleOut.reshape(-1, scale_d)[:real_bs, :].reshape(1, real_bs, scale_d)
            npu_yOrigin_cpu = npu_yOrigin_cpu.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)
            cpu_yOrigin = cpu_yOrigin.reshape(-1, split_d)[:real_bs, :].reshape(1, real_bs, split_d)

            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)
            yOriginOut_close = np.allclose(npu_yOrigin_cpu.reshape(-1), cpu_yOrigin.reshape(-1), rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")
            self.assertTrue(yOriginOut_close, f"yOriginOut_close precision compare fail")

    def test_mode3_without_y_origin(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, group_index, dst_type, quant_mode, group_size, round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value):
                npu_yOut, npu_scaleOut, npu_yOrigin = torch.ops.custom.npu_swiglu_group_quant(x, topk_weight=topk_weight, group_index=group_index, dst_type=dst_type, quant_mode=quant_mode, group_size=group_size, \
                    round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOrigin

        bs = 8192
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 3
            clamp_value = 10
            round_scale = False
            ue8m0_scale = False
            output_origin = False
            group_list_type = 0
            group_size = 128

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            topk_weight = torch.tensor(np.random.uniform(-2, 2, (bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2]).to(torch.int64)

            cpu_yOut, cpu_scaleOut, cpu_yOrigin = swiglu_fp8_quant_per_token_golden(x.numpy(), topk_weight.numpy(), clamp_value, round_scale, ue8m0_scale)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scaleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scaleOut.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=0.0001, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")

    def test_mode3_ue8m0(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, topk_weight, group_index, dst_type, quant_mode, group_size, round_scale, ue8m0_scale, output_origin, group_list_type, clamp_value):
                # no topk_weight case
                npu_yOut, npu_scaleOut, npu_yOrigin = torch.ops.custom.npu_swiglu_group_quant(x, dst_type=dst_type, quant_mode=quant_mode, group_size=group_size, \
                    round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)
                return npu_yOut, npu_scaleOut, npu_yOrigin

        bs = 1024
        d_list = [1024, 7168, 12288]
        for d in d_list:
            dst_type = 36
            dst_type_torch = torch.float8_e4m3fn
            quant_mode = 3
            clamp_value = 0.0
            round_scale = True
            ue8m0_scale = True
            output_origin = True
            group_list_type = 0
            group_size = 128

            np.random.seed(42)

            x = torch.tensor(np.random.uniform(-2, 2, (bs, d))).to(torch.float16)
            topk_weight = torch.tensor(np.ones((bs, 1))).to(torch.float32)
            group_index = torch.tensor([bs // 2, bs - bs // 2]).to(torch.int64)

            cpu_yOut, cpu_scaleOut, cpu_yOrigin = swiglu_fp8_quant_per_token_golden(x.numpy(), topk_weight.numpy(), clamp_value, round_scale, ue8m0_scale)
            
            x_npu = x.to("npu:%s" % DEVICE_ID)
            topk_weight_npu = topk_weight.to("npu:%s" % DEVICE_ID)
            group_index_npu = group_index.to("npu:%s" % DEVICE_ID)

            npu_yOut, npu_scaleOut, npu_yOriginOut = torch.ops.custom.npu_swiglu_group_quant(x_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_mode = Network().to("npu:%s" % DEVICE_ID)
            from torchair.configs.compiler_config import CompilerConfig
            config = CompilerConfig()
            config.mode = "reduce-overhead"
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            npu_yOut, npu_scaleOut, npu_yOriginOut = npu_mode(x_npu, topk_weight=topk_weight_npu, group_index=group_index_npu, dst_type=dst_type_torch, quant_mode=quant_mode, group_size=group_size, \
                        round_scale=round_scale, ue8m0_scale=ue8m0_scale, output_origin=output_origin, group_list_type=group_list_type, clamp_value=clamp_value)

            npu_yOut_cpu = npu_yOut.cpu()
            npu_scale_cpu = npu_scaleOut.view(torch.uint8).cpu().numpy()
            npu_yOrigin_cpu = npu_yOriginOut.cpu().float().numpy()
            dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
            yOut_close = requantize_compare(torch.from_numpy(cpu_yOut.view(np.int8)), npu_yOut_cpu, dst_type_str)
            scaleOut_close = np.allclose(npu_scale_cpu.reshape(-1), cpu_scaleOut.reshape(-1), rtol=0.0001, atol=1, equal_nan=True)
            yOriginOut_close = np.allclose(npu_yOrigin_cpu.reshape(-1), cpu_yOrigin.reshape(-1), rtol=0.01, atol=0.01, equal_nan=True)

            self.assertTrue(yOut_close, f"yOut precision compare fail")
            self.assertTrue(scaleOut_close, f"scaleOut_close precision compare fail")
            self.assertTrue(yOriginOut_close, f"yOriginOut_close precision compare fail")

if __name__ == "__main__":
    run_tests()
