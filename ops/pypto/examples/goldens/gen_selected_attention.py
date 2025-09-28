
#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
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
from bfloat16 import bfloat16

if __name__ == "__main__":
    """ 单独执行时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
#     from golden_register import GoldenRegister  # 单独执行 import 失败, 需确认上文中 '系统 import 路径' 配置正确
# else:
#     from golden_register import GoldenRegister


def dump_file(data, data_path, dtype):
    """将PyTorch张量保存到文件，支持BFloat16类型转换"""
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.int32:
        np_dtype = np.int32
    elif dtype == torch.bfloat16:
        np_dtype = bfloat16
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")

    if isinstance(data, torch.Tensor):
        # 处理BFloat16类型：转换为float32后再转NumPy（NumPy不支持BFloat16）
        if data.dtype == torch.bfloat16:
            data_np = data.cpu().to(torch.float32).numpy()
        else:
            data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)

    # 确保最终类型与指定dtype一致
    data_np = data_np.astype(np_dtype)
    data_np.tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)

    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)

    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)

    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def softmax(x):
    """PyTorch实现的softmax函数"""
    x = x.float()
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = torch.sum(y, dim=-1, keepdim=True)
    ans = y
    return ans, x_sum, x_max


def compute_attention(q, k, v, actualSeq, scalar, topk, atten_out_shape):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    # 提取维度信息
    b, s_q, n_q, d_q = q.shape
    _, _, s_max, d_k = k.shape
    _, _, _, d_v = v.shape

    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=torch.float32)

    # 遍历每个批次
    for i in range(b):
        # 遍历每个s_q
        for j in range(s_q):
            # 获取当前批次的实际序列长度
            if isinstance(actualSeq, torch.Tensor):
                kv_seq_len = actualSeq[i].item()
            else:
                kv_seq_len = actualSeq[i]

            # s_q!=1 MTP场景下的casual计算
            seq_len = min(max(kv_seq_len - s_q + 1 + j, 0), topk)
            print("==============cur s1 seq_len: ",  seq_len)

            # 获取当前批次和s_q的q [n_q, d_q]
            q_bs = q[i, j]

            # 获取当前批次的[seq_len, d_k/d_v]
            k_bs = k[i, j, :seq_len]
            v_bs = v[i, j, :seq_len]

            # MM1: 矩阵乘法
            qk_bmm_res = torch.matmul(q_bs.float(), k_bs.transpose(1, 0).float())
            qk_ele_res = qk_bmm_res * scalar

            # Softmax计算
            softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

            # MM2: 矩阵乘法
            bmm2_res = torch.matmul(softmax_res / softmax_sum, v_bs.float())

            # 存储结果
            attention_output[i, j] = bmm2_res

    return attention_output


def gen_dsa_sa_entry(dtype, bn1n2s1, kv_slc_actual_seq, output):
    torch.manual_seed(42)

    b, n_q, n_kv, s_q = bn1n2s1

    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    selectBlockSize = 1

    np.random.seed(None)

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank

    s_max = topk * selectBlockSize

    scalar = d_q ** -0.5

    if isinstance(kv_slc_actual_seq, int):
        actual_seq = [kv_slc_actual_seq] * b
    elif isinstance(kv_slc_actual_seq, list):
        if len(kv_slc_actual_seq) == b:
            actual_seq = kv_slc_actual_seq
        else:
            raise RuntimeError("unsupported kv_slc_actual_seq list length")
    else:
        raise RuntimeError("unsupported kv_slc_actual_seq data type")

    # 1. 定义shape
    shape_q = [b, s_q, n_q, d_q]
    shape_k = [b, s_q, s_max, d_k]
    shape_v = [b, s_q, s_max, d_v]
    atten_out_shape = [b, s_q, n_q, d_v]

    # 2. 生成数据
    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_bsnd = gen_uniform_data(shape_k, -1, 1, dtype)
    v_bsnd = k_bsnd[:, :, :, :kv_lora_rank]

    # 3. 计算attention
    atten_out = compute_attention(q_bsnd, k_bsnd, v_bsnd, actual_seq, scalar, topk, atten_out_shape)

    # 4.dump 数据
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    # input params
    input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, s_max, s_max] # 保留一位

    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    k_slc_path = Path(output, 'k_slc.bin')
    v_slc_path = Path(output, 'v_slc.bin')
    actual_seq_path = Path(output, 'actual_seq.bin')
    atten_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')

    # dump golden file
    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)
    dump_file(actual_seq, actual_seq_path, torch.int32)
    dump_file(atten_out, atten_out_path, dtype)
    dump_file(input_params, input_param_path, torch.int32)

    return True


# @GoldenRegister.reg_golden_func(
#     case_names=[
#         "DynamicSlcFlashAttnDSASTest.dsa_slc_attn_bf16_b48_s1",
#         "DynamicSlcFlashAttnDSASTest.dsa_slc_attn_bf16_b32_s2",
#     ],
#     version=0,
#     timeout=0
# )
def dsa_sa_func(case_name: str, output: Path) -> bool:
    print("========================r2 sa golden")
    if case_name == "DynamicSlcFlashAttnDSASTest.dsa_slc_attn_bf16_b48_s1":
        # b, n_q, n_kv, s_q = bn1n2s1
        gen_dsa_sa_entry(torch.bfloat16, (48, 128, 1, 1), 4096, output)
    elif case_name == "DynamicSlcFlashAttnDSASTest.dsa_slc_attn_bf16_b32_s2":
        gen_dsa_sa_entry(torch.bfloat16, (32, 128, 1, 2), 2048, output)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    return True


def main() -> bool:
    """
    单独执行 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicSlcFlashAttnDSASTest.dsa_slc_attn_bf16_b32_s2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = dsa_sa_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
