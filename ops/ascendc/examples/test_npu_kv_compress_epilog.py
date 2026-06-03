# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from dataclasses import dataclass

import custom_ops
import numpy as np
import torch
import torch_npu
import torchair as tng

from en_dtypes import hifloat8
from ml_dtypes import float8_e4m3fn, float8_e5m2
from torch_npu.testing.testcase import TestCase, run_tests

IMPORT_SIDE_EFFECT_MODULES = (custom_ops, tng)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

DATA_TYPE_INT_TO_STR = {
    1: "float16",
    27: "bfloat16",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
}

SLICE_SIZE = 64
FP8_MAX_VALUE = {
    "float8_e5m2": np.float32(57344.0),
    "float8_e4m3fn": np.float32(448.0),
}
FP8_MIN_VALUE = {
    "float8_e5m2": np.float32(-57344.0),
    "float8_e4m3fn": np.float32(-448.0),
}
TORCH_FP8_DTYPE = {
    "float8_e5m2": torch.float8_e5m2,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


@dataclass(frozen=True)
class Layout1CompareInput:
    golden_u8: np.ndarray
    actual_u8: np.ndarray
    slot_mapping: torch.Tensor
    params: dict
    quant_mode: int
    dtype_str: str


def requantize_compare(golden, output, dtype_str):
    if dtype_str in ("float8_e5m2", "float8_e4m3fn", "hifloat8"):
        # 将浮点数据重新解释为 int8（保持位模式）
        output_for_compare = output.view(torch.int8)
        golden_for_compare = golden.view(torch.int8)
    else:
        raise ValueError(f"Unsupported dtype_str: {dtype_str}")

    # 计算 int8 表示下的绝对差值
    diff_abs = torch.abs(
        output_for_compare.view(-1) - golden_for_compare.view(-1)
    )
    diff_indices = torch.where(diff_abs > 1)[0]

    # 处理双方均为 NaN 的情况（NaN 的位模式在 int8 下没有固定值，这里用浮点 isNaN 判断）
    output_flat_float = output.view(-1).float()  # 转为 float32 方便判断 NaN
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


def f32_to_fp8_bytes(arr, dtype_str):
    if dtype_str == "float8_e5m2":
        return arr.astype(float8_e5m2).view(np.uint8)
    if dtype_str == "float8_e4m3fn":
        return arr.astype(float8_e4m3fn).view(np.uint8)
    raise ValueError(f"Unsupported fp8 dtype: {dtype_str}")


def fp8_bytes_to_f32(arr, dtype_str):
    if dtype_str == "float8_e5m2":
        return arr.view(float8_e5m2).astype(np.float32)
    if dtype_str == "float8_e4m3fn":
        return arr.view(float8_e4m3fn).astype(np.float32)
    raise ValueError(f"Unsupported fp8 dtype: {dtype_str}")


def calc_kv_compress_layout(head_dim, quant_mode, quant_group_size):
    quant_col = head_dim - SLICE_SIZE
    if quant_mode == 2:
        scale_col = math.ceil(quant_col / 64)
        scale_bytes = 1
    else:
        scale_col = math.ceil(quant_col / quant_group_size)
        scale_bytes = 4
    rope_bytes = SLICE_SIZE * 2
    concat_col = quant_col + rope_bytes + scale_col * scale_bytes
    kv_cache_col = math.ceil(concat_col / 128) * 128
    return {
        "quant_col": quant_col,
        "scale_col": scale_col,
        "scale_bytes": scale_bytes,
        "rope_bytes": rope_bytes,
        "concat_col": concat_col,
        "kv_cache_col": kv_cache_col,
    }


def golden_group_fp8_row(x_row, quant_col, quant_group_size, dtype_str):
    scale_col = math.ceil(quant_col / quant_group_size)
    fp8_bytes = np.zeros(quant_col, dtype=np.uint8)
    scale_bytes = []
    fp8_max = FP8_MAX_VALUE[dtype_str]

    for group_idx in range(scale_col):
        start = group_idx * quant_group_size
        end = min(start + quant_group_size, quant_col)
        group = x_row[start:end].copy()

        abs_vals = np.where(np.isnan(group), np.float32(0.0), np.abs(group))
        max_abs = np.float32(np.max(abs_vals))
        scale = (
            np.float32(np.inf)
            if max_abs == 0
            else np.float32(max_abs / fp8_max)
        )

        x_scaled = group / scale
        x_scaled = np.where(np.isnan(x_scaled), np.float32(0.0), x_scaled)
        fp8_bytes[start:end] = f32_to_fp8_bytes(x_scaled, dtype_str)
        scale_bytes.append(np.asarray([scale], dtype="<f4").view(np.uint8))

    return fp8_bytes, scale_bytes


def golden_mxfp8_row(x_row, quant_col, dtype_str, round_scale):
    group_size = 64
    fp8_bytes = np.zeros(quant_col, dtype=np.uint8)
    scale_bytes = []
    fp8_max = FP8_MAX_VALUE[dtype_str]
    fp8_min = FP8_MIN_VALUE[dtype_str]
    coeff = np.float32(1.0) / fp8_max

    for start in range(0, quant_col, group_size):
        end = min(start + group_size, quant_col)
        group = x_row[start:end].copy()
        max_abs = np.float32(max(np.max(np.abs(group)), 1e-4))
        scale = np.float32(max_abs * coeff)
        scale_u32 = np.asarray([scale], dtype=np.float32).view(np.uint32)[0]

        if round_scale:
            exp_bits = int((scale_u32 >> 23) & 0xFF)
            mantissa = int(scale_u32 & 0x7FFFFF)
            new_exp = exp_bits + (1 if mantissa else 0)
            scale = np.asarray(
                [np.uint32(new_exp << 23)],
                dtype=np.uint32,
            ).view(np.float32)[0]
            scale_u8 = new_exp & 0xFF
        else:
            scale_u8 = int((scale_u32 >> 23) & 0xFF)

        x_scaled = np.clip(group / scale, fp8_min, fp8_max)
        fp8_bytes[start:end] = f32_to_fp8_bytes(x_scaled, dtype_str)
        scale_bytes.append(np.uint8(scale_u8))

    return fp8_bytes, scale_bytes


def kv_compress_epilog_layout1_golden(
    x, slot_mapping, quant_mode, quant_group_size, dtype_str
):
    params = calc_kv_compress_layout(x.shape[1], quant_mode, quant_group_size)
    quant_col = params["quant_col"]
    rope_bytes = params["rope_bytes"]
    kv_cache_col = params["kv_cache_col"]

    x_bf16 = x.view(torch.uint16).numpy()
    x_f32 = (x_bf16.astype(np.uint32) << 16).view(np.float32)
    slot_np = slot_mapping.numpy().astype(np.int32)
    valid_slots = slot_np[slot_np >= 0]
    cache_rows = int(np.max(valid_slots)) + 1 if len(valid_slots) > 0 else 1
    golden = np.zeros((cache_rows, kv_cache_col), dtype=np.uint8)

    for row_idx, slot in enumerate(slot_np):
        if slot == -1:
            continue

        rope_raw = x_bf16[row_idx, quant_col:].view(np.uint8)

        if quant_mode == 2:
            fp8_bytes, scale_bytes = golden_mxfp8_row(
                x_f32[row_idx, :quant_col], quant_col, dtype_str, True
            )
            golden[slot, :rope_bytes] = rope_raw
            golden[slot, rope_bytes:rope_bytes + quant_col] = fp8_bytes
            scale_start = rope_bytes + quant_col
            for group_idx, scale_raw in enumerate(scale_bytes):
                golden[slot, scale_start + group_idx] = scale_raw
        else:
            fp8_bytes, scale_bytes = golden_group_fp8_row(
                x_f32[row_idx, :quant_col],
                quant_col,
                quant_group_size,
                dtype_str,
            )
            golden[slot, :quant_col] = fp8_bytes
            golden[slot, quant_col:quant_col + rope_bytes] = rope_raw
            for group_idx, scale_raw in enumerate(scale_bytes):
                offset = quant_col + rope_bytes + group_idx * 4
                golden[slot, offset:offset + 4] = scale_raw

    return golden, params


def cosine_similarity(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == 0 and norm_b == 0 else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compare_kv_compress_layout1(compare_input):
    golden_u8 = compare_input.golden_u8
    actual_u8 = compare_input.actual_u8
    slot_mapping = compare_input.slot_mapping
    params = compare_input.params
    quant_mode = compare_input.quant_mode
    dtype_str = compare_input.dtype_str

    quant_col = params["quant_col"]
    rope_bytes = params["rope_bytes"]
    scale_col = params["scale_col"]
    scale_bytes = params["scale_bytes"]

    total_bytes = 0
    total_match = 0
    cos_sims = []
    slot_np = slot_mapping.numpy().astype(np.int32)

    for slot in slot_np:
        if slot == -1:
            continue

        golden_row = golden_u8[slot]
        actual_row = actual_u8[slot]
        if quant_mode == 2:
            golden_rope = golden_row[:rope_bytes]
            actual_rope = actual_row[:rope_bytes]
            golden_quant = golden_row[rope_bytes:rope_bytes + quant_col]
            actual_quant = actual_row[rope_bytes:rope_bytes + quant_col]
        else:
            golden_quant = golden_row[:quant_col]
            actual_quant = actual_row[:quant_col]
            golden_rope = golden_row[quant_col:quant_col + rope_bytes]
            actual_rope = actual_row[quant_col:quant_col + rope_bytes]

        scale_start = quant_col + rope_bytes
        scale_end = scale_start + scale_col * scale_bytes
        golden_scale = golden_row[scale_start:scale_end]
        actual_scale = actual_row[scale_start:scale_end]

        total_match += int(np.sum(golden_quant == actual_quant))
        total_match += int(np.sum(golden_rope == actual_rope))
        total_match += int(np.sum(golden_scale == actual_scale))
        total_bytes += quant_col + rope_bytes + scale_col * scale_bytes

        cos_sims.append(
            cosine_similarity(
                fp8_bytes_to_f32(golden_quant, dtype_str),
                fp8_bytes_to_f32(actual_quant, dtype_str),
            )
        )

    match_rate = total_match / total_bytes if total_bytes > 0 else 1.0
    avg_cos = float(np.mean(cos_sims)) if cos_sims else 1.0
    return match_rate, avg_cos


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
    def _make_slot_mapping(self, num_tokens, slot_mode, slot_dtype):
        if slot_mode == "shuffled":
            slot_mapping_np = np.random.permutation(num_tokens)
            slot_mapping_np = slot_mapping_np.astype(np.int32)
        elif slot_mode == "with_skip":
            slot_mapping_np = np.arange(num_tokens, dtype=np.int32)
            slot_mapping_np[np.random.rand(num_tokens) < 0.2] = -1
        elif slot_mode == "sparse":
            slot_mapping_np = np.arange(num_tokens, dtype=np.int32) * 2
        else:
            slot_mapping_np = np.arange(num_tokens, dtype=np.int32)

        dtype = torch.int64 if slot_dtype == "int64" else torch.int32
        return torch.tensor(slot_mapping_np, dtype=dtype)

    def run_layout1_case(self, case):
        num_tokens = case["num_tokens"]
        head_dim = case["head_dim"]
        quant_mode = case["quant_mode"]
        quant_group_size = case["quant_group_size"]
        dst_type_str = case["dst_type_str"]
        round_scale = case.get("round_scale", True)
        seed = case.get("seed", 0)

        np.random.seed(seed)
        x_np = np.random.uniform(
            -10,
            10,
            (num_tokens, head_dim),
        ).astype(np.float32)
        x = torch.tensor(x_np).to(torch.bfloat16)
        slot_mapping = self._make_slot_mapping(
            num_tokens,
            case.get("slot_mode", "sequential"),
            case.get("slot_dtype", "int32"),
        )

        kv_compress_cache_golden, params = kv_compress_epilog_layout1_golden(
            x,
            slot_mapping,
            quant_mode,
            quant_group_size,
            dst_type_str,
        )
        cache_rows = kv_compress_cache_golden.shape[0]
        kv_len = params["kv_cache_col"]

        kv_compress_cache = torch.zeros(
            cache_rows,
            kv_len,
            dtype=TORCH_FP8_DTYPE[dst_type_str],
        )

        print(
            f"======================== {case['name']} BEGIN "
            "========================"
        )
        device = f"npu:{DEVICE_ID}"
        x_npu = x.to(device)
        slot_mapping_npu = slot_mapping.to(device)
        kv_cache_npu = kv_compress_cache.to(device)

        torch.ops.custom.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale_flag=round_scale,
        )

        kv_cache_cpu = (
            kv_cache_npu.cpu()
            .view(torch.uint8)
            .numpy()
            .reshape(cache_rows, kv_len)
        )
        match_rate, avg_cos = compare_kv_compress_layout1(
            Layout1CompareInput(
                golden_u8=kv_compress_cache_golden,
                actual_u8=kv_cache_cpu,
                slot_mapping=slot_mapping,
                params=params,
                quant_mode=quant_mode,
                dtype_str=dst_type_str,
            )
        )
        print(
            f"{case['name']} golden compare: match={match_rate:.6f}, "
            f"cos={avg_cos:.6f}"
        )
        print(
            f"======================== {case['name']} FINISH "
            "========================"
        )

        self.assertTrue(
            match_rate >= 0.99 and avg_cos >= 0.999,
            f"{case['name']} golden compare fail: match={match_rate}, "
            f"cos={avg_cos}",
        )

    def test_kv_compress_epilog_with_cpu_benchmark(self):
        self.run_layout1_case(
            {
                "name": "layout1_group_fp8_e5m2_bs4096_d512",
                "num_tokens": 4096,
                "head_dim": 512,
                "quant_mode": 1,
                "quant_group_size": 128,
                "dst_type_str": "float8_e5m2",
            },
        )

    def test_kv_compress_epilog_hifloat8_eager(self):
        num_tokens = 4096
        head_dim = 512
        quant_group_size = 128
        quant_mode = 3
        round_scale = True
        scale_val = 0.5
        dst_type = 34

        np.random.seed(0)
        x_np = np.random.uniform(2, 2, (num_tokens, head_dim))
        x_np = x_np.astype(np.float32)

        slot_mapping = np.arange(num_tokens)  # 生成下标序列
        np.random.shuffle(slot_mapping)  # 随机打乱顺序

        kv_compress_cache_np = np.zeros(
            (num_tokens, head_dim),
            dtype=np.float16,
        )
        kv_compress_cache_golden = hifp8_block_quant(
            kv_compress_cache_np.copy().astype(np.int8),
            x_np.copy(),
            slot_mapping.copy().astype(np.int32),
            scale_val,
        )

        print("======================== Eager BEGIN ========================")
        device = f"npu:{DEVICE_ID}"
        x_npu = torch.tensor(x_np).to(torch.bfloat16).to(device)
        slot_mapping_npu = (
            torch.tensor(slot_mapping).to(torch.int32).to(device)
        )
        kv_cache_npu = torch.tensor(kv_compress_cache_np).to(device)
        kv_cache_npu = torch_npu.npu_dtype_cast(
            kv_cache_npu,
            torch_npu.hifloat8,
        )

        torch.ops.custom.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale_flag=round_scale,
            scale=scale_val,
        )

        kv_cache_cpu = kv_cache_npu.cpu()

        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(
            torch.from_numpy(kv_compress_cache_golden.view(np.int8)),
            kv_cache_cpu,
            dst_type_str,
        )

        self.assertTrue(
            cache_close,
            "kv_compress_cache precision compare fail",
        )
        print("======================== Eager FINISH ========================")


KV_COMPRESS_LAYOUT1_CASES = [
    {
        "name": "layout1_group_fp8_e5m2_bs1_d512",
        "num_tokens": 1,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
    },
    {
        "name": "layout1_group_fp8_e5m2_bs64_d512",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
    },
    {
        "name": "layout1_group_fp8_e4m3fn_bs64_d512",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e4m3fn",
    },
    {
        "name": "layout1_group_fp8_e5m2_bs32_d192",
        "num_tokens": 32,
        "head_dim": 192,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
    },
    {
        "name": "layout1_group_fp8_e4m3fn_bs32_d384",
        "num_tokens": 32,
        "head_dim": 384,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e4m3fn",
    },
    {
        "name": "layout1_group_fp8_e5m2_shuffled",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
        "slot_mode": "shuffled",
    },
    {
        "name": "layout1_group_fp8_e5m2_with_skip",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
        "slot_mode": "with_skip",
    },
    {
        "name": "layout1_group_fp8_e5m2_sparse",
        "num_tokens": 32,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
        "slot_mode": "sparse",
    },
    {
        "name": "layout1_group_fp8_e5m2_int64_slots",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 1,
        "quant_group_size": 128,
        "dst_type_str": "float8_e5m2",
        "slot_dtype": "int64",
    },
    {
        "name": "layout1_mxfp8_e4m3fn_bs64_d512",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 2,
        "quant_group_size": 64,
        "dst_type_str": "float8_e4m3fn",
    },
    {
        "name": "layout1_mxfp8_e5m2_bs64_d512",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 2,
        "quant_group_size": 64,
        "dst_type_str": "float8_e5m2",
    },
    {
        "name": "layout1_mxfp8_e4m3fn_bs32_d256",
        "num_tokens": 32,
        "head_dim": 256,
        "quant_mode": 2,
        "quant_group_size": 64,
        "dst_type_str": "float8_e4m3fn",
    },
    {
        "name": "layout1_mxfp8_e4m3fn_bs32_d768",
        "num_tokens": 32,
        "head_dim": 768,
        "quant_mode": 2,
        "quant_group_size": 64,
        "dst_type_str": "float8_e4m3fn",
    },
    {
        "name": "layout1_mxfp8_e4m3fn_with_skip",
        "num_tokens": 64,
        "head_dim": 512,
        "quant_mode": 2,
        "quant_group_size": 64,
        "dst_type_str": "float8_e4m3fn",
        "slot_mode": "with_skip",
    },
]


def _make_layout1_test(case):
    def test(self):
        self.run_layout1_case(case)

    test.__name__ = f"test_kv_compress_epilog_{case['name']}"
    return test


for _case in KV_COMPRESS_LAYOUT1_CASES:
    setattr(
        TestNpuKvCompressEpilog,
        f"test_kv_compress_epilog_{_case['name']}",
        _make_layout1_test(_case),
    )


if __name__ == "__main__":
    run_tests()
