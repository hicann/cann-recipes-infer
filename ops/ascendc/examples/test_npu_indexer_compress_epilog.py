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
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from ml_dtypes import bfloat16 as bf16
from ml_dtypes import float8_e4m3fn, float8_e5m2

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

def fast_log2_ceil(x):
    bits_x = x.view(np.uint32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((np.uint32(1) << np.uint32(23)) - np.uint32(1))
    if man_bits != 0:
        base = 1
    else:
        base = 0
    return np.array([exp_x - 127 + base], dtype=np.uint32).view(np.int32)[0]

def fast_pow2(x):
    bits_x = (x + 127) << 23
    return np.array([bits_x], dtype=np.int32).view(np.float32)[0]

def fast_round_scale(amax):
    tmp = fast_log2_ceil(amax)
    tmp1 = fast_pow2(tmp)
    return tmp1

def act_quant(x, dst_type, round_scale):
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

    scale_shape = scale.shape
    if round_scale == True:
        scale = scale.reshape(-1)
        for i in range(len(scale)):
            scale[i] = fast_round_scale(scale[i])

    scale = scale.reshape(scale_shape)
    scale_expanded = np.zeros_like(x_cleaned).astype(np.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            scale_expanded[i * row_block_size: (i + 1) * row_block_size, j * col_block_size: (j + 1) * col_block_size] = scale[i, j]
    
    x_f32 = x_cleaned.astype(np.float32)
    out_f32 = x_f32 / scale_expanded
    out_f32[mask_tensor==1] = x[mask_tensor==1]
    max_norm = get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)

    result = scale.view(np.uint32) >> 23
    output_scale = result.astype(np.uint8)

    round_data = np.round(out_f32, 8)
    round_data = np.nan_to_num(round_data, nan=0.0, copy=False)
    if dst_type == 35:
        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        round_data = round_data.astype(float8_e4m3fn, copy=False)
    
    return round_data, output_scale

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

def index_compress_epilog_(indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, dst_type, quant_mode, round_scale):
    index = np.where(slot_mapping != -1)
    x = x[index]
    slot_mapping = slot_mapping[index]
    if quant_mode == 0:
        y, scale = act_quant(x, dst_type, round_scale)
    elif quant_mode == 1:
        y, scale = dynamic_block_quant(x, dst_type)
    indexer_compress_cache = indexer_compress_cache.astype(float8_e5m2)
    indexer_compress_cache[slot_mapping] = y
    indexer_compress_cache_scale[slot_mapping] = scale
    return indexer_compress_cache, indexer_compress_cache_scale

class TestCustomIndexerCompressEpilog(TestCase):
    def test_indexer_compress_epilog_normal_quant(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, quant_mode, round_scale):
                torch.ops.custom.indexer_compress_epilog(indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, quant_mode=quant_mode, round_scale=round_scale)

        b = 1
        s = 8192
        d = 4096
        dst_type = 35
        scale_factor = int((d + 127) / 128)
        quant_mode = 1
        round_scale = True

        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float16)
        # 构造slot_mapping
        slot_mapping = np.arange(b * s) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序
        slot_mapping = torch.tensor(slot_mapping).to(torch.int32)

        # 1: 'float16',
        # 27: 'bfloat16',
        # 35: 'float8_e5m2',
        # 36: 'float8_e4m3fn',
        dst_type = 35
        indexer_compress_cache = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float8_e5m2)
        indexer_compress_cache_scale = torch.tensor(np.random.uniform(-2, 2, (b * s, scale_factor))).to(torch.float32)

        indexer_compress_cache_golden, indexer_compress_cache_scale_golden = index_compress_epilog_(indexer_compress_cache.view(torch.int8).numpy().copy(), indexer_compress_cache_scale.numpy().copy().astype(np.float32), \
            x.numpy().copy().astype(np.float16), slot_mapping.numpy().copy().astype(np.int32), dst_type, quant_mode, round_scale)
        
        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = indexer_compress_cache.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_scale_npu = indexer_compress_cache_scale.to("npu:%s" % DEVICE_ID)

        torch.ops.custom.indexer_compress_epilog(indexer_compress_cache_npu, indexer_compress_cache_scale_npu, x_npu, slot_mapping_npu, quant_mode=quant_mode, round_scale=round_scale)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_mode(indexer_compress_cache_npu, indexer_compress_cache_scale_npu, x_npu, slot_mapping_npu, quant_mode, round_scale)

        indexer_compress_cache_cpu = indexer_compress_cache_npu.cpu()
        indexer_compress_cache_scale_cpu = indexer_compress_cache_scale_npu.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(indexer_compress_cache_golden.view(np.int8)), indexer_compress_cache_cpu, dst_type_str)
        cache_scale_close = np.allclose(indexer_compress_cache_scale_golden.reshape(-1), indexer_compress_cache_scale_cpu.reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)

        self.assertTrue(cache_close, f"indexer_compress_cache precision compare fail")
        self.assertTrue(cache_scale_close, f"indexer_compress_cache_scale precision compare fail")

    def test_indexer_compress_epilog_mx_fp8_quant(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, quant_mode, round_scale):
                torch.ops.custom.indexer_compress_epilog(indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping, quant_mode=quant_mode, round_scale=round_scale)

        b = 1
        s = 8192
        d = 4096
        dst_type = 35
        scale_factor = int((d + 127) / 128)
        quant_mode = 0
        round_scale = True

        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float16)
        # 构造slot_mapping
        slot_mapping = np.arange(b * s) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序
        slot_mapping = torch.tensor(slot_mapping).to(torch.int32)

        # 1: 'float16',
        # 27: 'bfloat16',
        # 35: 'float8_e5m2',
        # 36: 'float8_e4m3fn',
        dst_type = 35
        indexer_compress_cache = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float8_e5m2)
        indexer_compress_cache_scale = torch.tensor(np.random.uniform(-2, 2, (b * s, scale_factor))).to(torch.float8_e8m0fnu)

        indexer_compress_cache_golden, indexer_compress_cache_scale_golden = index_compress_epilog_(indexer_compress_cache.view(torch.int8).numpy().copy(), \
            indexer_compress_cache_scale.view(torch.int8).numpy().copy().astype(np.int8), x.numpy().copy().astype(np.float16), slot_mapping.numpy().copy().astype(np.int32), dst_type, quant_mode, round_scale)
        
        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = indexer_compress_cache.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_scale_npu = indexer_compress_cache_scale.to("npu:%s" % DEVICE_ID)

        torch.ops.custom.indexer_compress_epilog(indexer_compress_cache_npu, indexer_compress_cache_scale_npu, x_npu, slot_mapping_npu, quant_mode=quant_mode, round_scale=round_scale)

        indexer_compress_cache_cpu = indexer_compress_cache_npu.cpu()
        indexer_compress_cache_scale_cpu = indexer_compress_cache_scale_npu.cpu().view(torch.int8).numpy().astype(np.int8)
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(indexer_compress_cache_golden.view(np.int8)), indexer_compress_cache_cpu, dst_type_str)
        cache_scale_close = np.allclose(indexer_compress_cache_scale_golden.reshape(-1), indexer_compress_cache_scale_cpu.reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)

        self.assertTrue(cache_close, f"indexer_compress_cache precision compare fail")
        self.assertTrue(cache_scale_close, f"indexer_compress_cache_scale precision compare fail")

if __name__ == "__main__":
    run_tests()
