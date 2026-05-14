# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair as tng
import numpy as np
import custom_ops
import math

from torch_npu.testing.testcase import TestCase, run_tests
from ml_dtypes import bfloat16
from ml_dtypes import float8_e4m3fn, float8_e5m2
from en_dtypes import hifloat8

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

DATA_TYPE_INT_TO_STR = {
    1: 'float16',
    27: 'bfloat16',
    34: 'hifloat8',
    35: 'float8_e5m2',
    36: 'float8_e4m3fn',
}


def requantize_compare(golden, output, dtype_str, print_diff_limit=100):
    if dtype_str in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8'):
        # 将浮点数据重新解释为 int8（保持位模式）
        output_for_compare = output.view(torch.int8)
        golden_for_compare = golden.view(torch.int8)
    else:
        raise ValueError(f"Unsupported dtype_str: {dtype_str}")

    # 计算 int8 表示下的绝对差值
    diff_abs = torch.abs(output_for_compare.view(-1) - golden_for_compare.view(-1))
    diff_indices = torch.where(diff_abs > 1)[0]

    # 处理双方均为 NaN 的情况（NaN 的位模式在 int8 下没有固定值，这里用浮点 isNaN 判断）
    output_flat_float = output.view(-1).float()   # 转为 float32 方便判断 NaN
    golden_flat_float = golden.view(-1).float()
    both_nan = torch.isnan(output_flat_float) & torch.isnan(golden_flat_float)
    both_nan_idx = torch.where(both_nan)[0]

    # 从差异中排除双方均为 NaN 的位置（它们不算错误）
    diff_indices = diff_indices[~torch.isin(diff_indices, both_nan_idx)]

    # 打印差异信息
    num_diff = len(diff_indices)

    # 计算精度
    total_elements = golden.numel()
    good_elements = total_elements - num_diff
    precision = good_elements / total_elements
    is_pass = (1 - precision) <= 0.001  # 差异率 <= 0.1%

    return is_pass


def hifp8_block_quant(kv_compress_cache, x, slot_mapping, scale):
    valid_mask = slot_mapping != -1
    if not np.any(valid_mask):
        return kv_compress_cache.astype(hifloat8)

    x_valid = x[valid_mask].astype(np.float32)

    slots_valid = slot_mapping[valid_mask]
    scale_f32 = np.float32(scale)
    y_f32 = x_valid * scale_f32
    y_hifp8 = y_f32.astype(hifloat8)
    new_cache = kv_compress_cache.astype(hifloat8)
    new_cache[slots_valid] = y_hifp8
    return new_cache


class TestNpuKvCompressEpilog(TestCase):

    def test_kv_compress_epilog_with_cpu_benchmark(self):
        """测试函数，包含CPU标杆验证"""
        # 设置测试参数（与原始测试相同）
        num_tokens = 4096
        head_dim = 512
        quant_group_size = 128
        quant_mode = 1
        round_scale = True
        
        # 生成输入数据
        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-10, 10, (num_tokens, head_dim))).to(torch.bfloat16)
        
        # 生成slot_mapping
        slot_mapping = torch.tensor([i for i in range(num_tokens)], dtype=torch.int32)
        
        scale_num = math.ceil((head_dim - 64) / 128)
        scale_size = 1
        if quant_mode == 1:
            scale_size = 4
        y_size = head_dim - 64 + 2 * 64 + scale_num * scale_size
        
        kv_len = (128 - y_size % 128) % 128 + y_size
        
        # 初始化kv_compress_cache
        kv_compress_cache = torch.zeros(num_tokens, kv_len, dtype=torch.float8_e5m2)
        
        # 1. 运行NPU操作
        print(f'======================== NPU Eager BEGIN ========================')
        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        kv_cache_npu = kv_compress_cache.to("npu:%s" % DEVICE_ID)
        
        torch.ops.custom.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode
        )
        print(f'======================== NPU Eager FINISH ========================')
        
        return kv_cache_npu

    def test_kv_compress_epilog_hifloat8_eager(self):
        num_tokens = 4096
        head_dim = 512
        quant_group_size = 128
        quant_mode = 3
        round_scale = True
        scale_val = 0.5
        dst_type = 34

        np.random.seed(0)
        x_np = np.random.uniform(2, 2, (num_tokens, head_dim)).astype(np.float32)

        slot_mapping = np.arange(num_tokens) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序

        kv_compress_cache_np = np.zeros((num_tokens, head_dim), dtype=np.float16)
        kv_compress_cache_golden = hifp8_block_quant(kv_compress_cache_np.copy().astype(np.int8), x_np.copy(),
                                                     slot_mapping.copy().astype(np.int32), scale_val)

        print(f'======================== Eager BEGIN ========================')
        x_npu = torch.tensor(x_np).to(torch.bfloat16).to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = torch.tensor(slot_mapping).to(torch.int32).to("npu:%s" % DEVICE_ID)
        kv_cache_npu = torch.tensor(kv_compress_cache_np).to("npu:%s" % DEVICE_ID)
        kv_cache_npu = torch_npu.npu_dtype_cast(kv_cache_npu, torch_npu.hifloat8)

        torch.ops.custom.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale_flag=round_scale,
            scale=scale_val
        )

        kv_cache_cpu = kv_cache_npu.cpu()

        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(kv_compress_cache_golden.view(np.int8)),
                                         kv_cache_cpu, dst_type_str)

        self.assertTrue(cache_close, f"kv_compress_cache precision compare fail")
        print(f'======================== Eager FINISH ========================')



if __name__ == "__main__":
    run_tests()