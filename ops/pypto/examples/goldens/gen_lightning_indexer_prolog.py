#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import math
import sys
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch


if __name__ == "__main__":
    """单独执行时配置"""
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/st/operator/src/deepseek_DSA")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    sys.path.append(str(g_src_root))
    # from tests.cmake.scripts.golden_register import (
    #     GoldenRegister,
    # )  # 单独执行 import 失败, 需确认上文中 '系统 import 路径' 配置正确
# else:
#     from golden_register import GoldenRegister


def tensor_tofile(t: torch.Tensor, output: Path, dtype: torch.dtype):
    with open(str(output), "wb") as f:
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).numpy().tobytes())


def inputs_tofile(inputs: dict, output: Path):
    for name, tensor in inputs.items():
        print("Input", name, "=", tensor.shape, "dtype=", tensor.dtype)
        tensor_tofile(tensor, Path(output, f"{name}.bin"), tensor.dtype)


def golden_tofile(golden: dict, output: Path):
    for name, tensor in golden.items():
        print("Output", name, "=", tensor.shape, "dtype=", tensor.dtype)
        tensor_tofile(tensor, Path(output, f"{name}_golden.bin"), tensor.dtype)


def precompute_freqs_cis(args: dict):
    dim = args["qk_rope_head_dim"]
    seqlen = args["max_seq_len"]
    beta_fast = args["beta_fast"]
    beta_slow = args["beta_slow"]
    base = args["rope_theta"]
    factor = args["rope_factor"]
    original_seq_len = args["original_seq_len"]

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    start_pos, s = args["start_pos"], args["s"]
    end_pos = start_pos + s
    return cos[start_pos:end_pos], sin[start_pos:end_pos]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def mlp_single_rope(x, cos_in, sin_in):
    logging.info("Entering into mlp_single_rope!")
    # x: (b, s, n, d), cos_in: (b, s, d), sin_in: (b, s, d)
    x_dtype = x.dtype
    b, s, n, d = x.shape
    x_cast = x.to(torch.float32)
    cos_cast = cos_in.to(torch.float32)
    sin_cast = sin_in.to(torch.float32)
    cos_re = cos_cast.unsqueeze(2)  # (b, s, 1, d)
    sin_re = sin_cast.unsqueeze(2)  # (b, s, 1, d)
    x_re = x_cast.reshape(b, s, n, d // 2, 2)
    x_trans = x_re.permute(0, 1, 2, 4, 3)  # (b, s, n, 2, d // 2)
    x_re1 = x_trans.reshape(b, s, n, d)
    res = x_re1 * cos_re + rotate_half(x_re1) * sin_re  # (b, s, n, d)
    return res.to(x_dtype)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps=1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return x * weight.to(torch.float32) + bias.to(torch.float32)


def quant_int8(x: torch.Tensor, block_size=128):
    x_fp32 = x.to(torch.float32)
    x_per_block = x_fp32.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    max_value = torch.max(torch.abs(x_per_block), dim=-1, keepdim=True).values
    scale_quant = 127 / max_value
    y_fp32 = x_per_block * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_int32 = torch.round(y_fp32).to(torch.int32)
    y_int8 = torch.trunc(y_int32.to(torch.bfloat16)).to(torch.int8)
    scale_dequant = 1 / scale_quant
    return y_int8, scale_dequant


def int8_index(
    q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor
) -> torch.Tensor:
    # (b, s, n, d) @ (b, 1, d, s) -> (b, s, n, s)
    logits = torch.matmul(q, k[:, None, ...].transpose(-1, -2)).to(dtype=torch.float32)
    logits = torch.relu(logits) * q_s
    logits_sum = torch.sum(logits, dim=-2)
    index_score = logits_sum * k_s
    return index_score


def gen_block_table(act_seq, block_size, s1, need_indices=False):
    b = act_seq.shape[0]
    block_num = 0
    block_num_each = []
    max_kv = max(act_seq)
    for cur_s in act_seq:
        cur_block_num = math.ceil(cur_s / block_size)
        block_num_each.append(cur_block_num)
        block_num += cur_block_num
    block_table_shape = [b, math.ceil(max_kv / block_size)]
    block_idx_list = torch.arange(0, block_num, 1)
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(
        torch.int32
    )

    block_idx = 0
    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int32)
        for i in range(b):
            cur_act = act_seq[i]
            for j in range(s1):
                pos = cur_act - s1 + j
                block_idx_in_seq = pos // block_size
                global_block_id = block_table[i, block_idx_in_seq]

                offset_in_block = pos % block_size
                global_index = global_block_id * block_size + offset_in_block
                cache_index[i, j] = global_index
    else:
        cache_index = None

    if need_indices:
        return block_num, block_table, cache_index
    else:
        return block_num, block_table


def gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size):
    dtype = k_cache_bsnd.dtype
    b, s2, n_kv, d = k_cache_bsnd.shape
    k_cache = torch.zeros((block_num, block_size, n_kv, d), dtype=dtype)
    s2_new = (
        (s2 - 1) // block_size + 1
    ) * block_size  # ceil to multiplicity of block_size
    k_cache_raw = torch.zeros((b, s2_new, n_kv, d), dtype=dtype)
    k_cache_raw[:, :s2, :, :] = k_cache_bsnd

    for b_idx in range(b):
        for block_idx, cache_block_idx in enumerate(block_table[b_idx]):
            block_offset = block_idx * block_size
            if cache_block_idx == -1:
                continue
            else:
                k_cache[cache_block_idx, :, :, :] = k_cache_raw[
                    b_idx, block_offset : (block_offset + block_size), :, :
                ]

    return k_cache


def scatter_update_pa_bsnd(cache, k_bsnd, cache_index, axis):
    block_number, block_size, n_kv, d = cache.shape
    res = cache.reshape(block_number * block_size * n_kv, d)
    b, s1 = cache_index.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = cache_index[b_i][s1_i]
                res[index_value, :] = k_bsnd[b_i, s1_i, :, :]

    return res.reshape(block_number, block_size, n_kv, d)


def indexer_prolog(inputs: dict, dims: dict):
    b, s, n, d = dims["b"], dims["seq"], dims["idx_n_heads"], dims["idx_head_dim"]
    rope_head_dim = dims["rope_head_dim"]
    x = inputs["token_x"]  # (b, s, dim)
    qr = inputs["qr"]  # (b, s, q_lora_rank)
    wq_b = inputs["wq_b"]  # (q_lora_rank, n_heads * head_dim)
    wk = inputs["wk"]  # (dim, head_dim)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    # freqs_cis = torch.view_as_complex(torch.stack([cos, sin], dim=-1))
    w_layernorm = inputs["weight_layer_norm"]
    b_layernorm = inputs["bias_layer_norm"]

    qtype = qr.dtype
    q = torch.matmul(qr.to(torch.float32), wq_b.to(torch.float32)).to(qtype)  # (b, s, n * d)
    q = q.reshape(b, s, n, d)
    q_pe, q_nope = torch.split(q, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_pe = mlp_single_rope(q_pe, cos, sin)
    q = torch.cat([q_pe, q_nope], dim=-1)

    xtype = x.dtype
    k = torch.matmul(x.to(torch.float32), wk.to(torch.float32))  # (b, s, d)
    print("k:", k)
    k = layer_norm(k, w_layernorm, b_layernorm).to(xtype)
    k_pe, k_nope = torch.split(k, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_pe = mlp_single_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # q = rotate_activation(q)
    # k = rotate_activation(k)

    k_cache = inputs["idx_k_cache"].clone()  # (block_num, block_size, n_kv, d)
    index = inputs["idx_k_cache_index"]  # (b, s)
    scatter_update_pa_bsnd(k_cache, k.reshape(b, s, 1, d), index, -2)
    # k_scale_cache[:b, start_pos:end_pos] = k_scale

    weights_proj = inputs["weights_proj"]
    weights = torch.matmul(x.to(torch.float32), weights_proj.to(torch.float32)).to(xtype)  # (b, s, n, 1)

    outputs = {"query": q, "idx_k_cache_out": k_cache, "weights": weights}
    return outputs


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["seq"] = params["s1"]
    dims["dim"] = 7168
    dims["q_lora_rank"] = 1536
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 64
    dims["rope_head_dim"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_indexer_prolog_inputs(dims, dtype=torch.bfloat16):
    b, s, n, d = dims["b"], dims["seq"], dims["idx_n_heads"], dims["idx_head_dim"]
    dim = dims["dim"]
    q_lora_rank = dims["q_lora_rank"]
    block_num = dims["block_num"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]

    x = torch.empty((b, s, dim), dtype=dtype).uniform_(-1, 1)
    qr = torch.empty((b, s, q_lora_rank), dtype=dtype).uniform_(-1, 1)
    wq_b = torch.empty((q_lora_rank, n * d), dtype=dtype).uniform_(-1, 1)
    wq_b_nz = wq_b.reshape(q_lora_rank, n * d // 16, 16).permute(1, 0, 2)

    wk = torch.empty((dim, d), dtype=dtype).uniform_(-1, 1)
    wk_nz = wk.reshape(dim, d // 16, 16).permute(1, 0, 2)

    weight_proj = torch.empty((dim, n), dtype=dtype).uniform_(-1, 1)
    weight_proj_nz = weight_proj.reshape(dim, n // 16, 16).permute(1, 0, 2)

    weight_ln = torch.ones((d), dtype=dtype)
    bias_ln = torch.zeros((d), dtype=dtype)
    # cos, sin = precompute_freqs_cis(dims)
    random_angles = (
        torch.rand(b, dims["seq"], dims["rope_head_dim"], dtype=torch.float32)
        * 2
        * torch.pi
    )
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)
    act_seq = torch.tensor([s2] * b)
    k_cache_bsnd = torch.empty((b, s2, n_kv, d), dtype=dtype).uniform_(-1, 1)
    # k_cache_index (b, s)
    block_num, block_table, k_cache_index = gen_block_table(
        act_seq, block_size, s, need_indices=True
    )
    k_cache = gen_cache_tensor(
        k_cache_bsnd, block_table, block_num, block_size
    )  # (block_num, block_size, n_kv, d)

    return {
        "token_x": x,
        "qr": qr,
        "wq_b": wq_b,
        "wq_b_nz": wq_b_nz,
        "wk": wk,
        "wk_nz": wk_nz,
        "weights_proj": weight_proj,
        "weights_proj_nz": weight_proj_nz,
        "weight_layer_norm": weight_ln,
        "bias_layer_norm": bias_ln,
        "cos_idx_rope": cos,
        "sin_idx_rope": sin,
        "idx_k_cache": k_cache,
        "idx_k_cache_index": k_cache_index,
        "idx_block_table": block_table,
    }


def gen_indexer_golden(params, output):
    seed = 0
    # NumPy 随机数生成器
    np.random.seed(seed)
    # PyTorch 随机数生成器
    torch.manual_seed(seed)

    dims = gen_dims(params)
    dim_tensor = torch.tensor(list(dims.values()), dtype=torch.int32)
    print("params:", dim_tensor, flush=True)
    tensor_tofile(dim_tensor, Path(output, f"input_param.bin"), torch.int32)

    inputs = gen_indexer_prolog_inputs(dims, torch.bfloat16)
    outputs = indexer_prolog(inputs, dims)
    inputs_tofile(inputs, output)
    golden_tofile(outputs, output)


# @GoldenRegister.reg_golden_func(
#     case_names=[
#         "LightningIndexerPrologSTest.bf16_indexer_prolog",
#         "LightningIndexerPrologSTest.b48_s1_1_s2_8k",
#         "LightningIndexerPrologSTest.b2_s1_2_s2_2k",
#         "LightningIndexerPrologSTest.b35_s1_2_s2_8k",
#         "LightningIndexerPrologSTest.b40_s1_4_s2_8k",
#         "LightningIndexerPrologSTest.b4_s1_1_s2_64k"
#     ]
# )
def indexer_test(case_name: str, output: Path) -> bool:
    if case_name.startswith("LightningIndexerPrologSTest.bf16_indexer_prolog"):
        params = {
            "b": 28,
            "s1": 1,
            "s2": 1024 * 2
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b48_s1_1_s2_8k"):
        params = {
            "b": 48,
            "s1": 1,
            "s2": 1024 * 8
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b2_s1_2_s2_2k"):
        params = {
            "b": 2,
            "s1": 2,
            "s2": 1024 * 2
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b35_s1_2_s2_8k"):
        params = {
            "b": 35,
            "s1": 2,
            "s2": 1024 * 8
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b40_s1_4_s2_8k"):
        params = {
            "b": 40,
            "s1": 4,
            "s2": 1024 * 8
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b4_s1_1_s2_64k"):
        params = {
            "b": 4,
            "s1": 1,
            "s2": 1024 * 64
        }
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    gen_indexer_golden(params, output)
    return True


def main() -> bool:
    """
    单独执行 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "LightningIndexerPrologSTest.bf16_indexer_prolog",
        "LightningIndexerPrologSTest.b48_s1_2_s2_8k",
        "LightningIndexerPrologSTest.b2_s1_2_s2_2k",
        "LightningIndexerPrologSTest.b48_s1_1_s2_8k",
        "LightningIndexerPrologSTest.b48_s1_4_s2_8k"
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = indexer_test(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
