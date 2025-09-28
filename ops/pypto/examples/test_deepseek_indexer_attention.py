# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import math
import numpy as np
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig

import torch._dynamo
import sys
import os
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"
import logging
import torchair
from bfloat16 import bfloat16
import custom_pypto
torchair.logger.setLevel(logging.DEBUG)
np.random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), 'goldens'))
from goldens import gen_deepseek_indexer_attention

RED = "\033[1;31m"
GREEN = "\033[1;32m"
RESET = "\033[0m"
YELLOW = "\033[93m"

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache,
                block_table, act_seqs, w_idx_qb, w_idx_k, w_idx_proj, in_gamma_k, in_beta_k, index_k_cache):
        res = torch.ops.custom_pypto.npu_sparse_attention_pto(
            x=x,
            w_dq=w_dq,
            w_uq_qr=w_uq_qr,
            w_uk=w_uk,
            w_dkv_kr=w_dkv_kr,
            gamma_cq=gamma_cq,
            gamma_ckv=gamma_ckv,
            sin=sin,
            cos=cos,
            cache_index=cache_index,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            block_table=block_table,
            act_seqs=act_seqs,
            w_idx_qb=w_idx_qb,
            w_idx_k=w_idx_k,
            w_idx_proj=w_idx_proj,
            in_gamma_k=in_gamma_k,
            in_beta_k=in_beta_k,
            index_k_cache=index_k_cache)
        return res 

def precision_compare_float(output_data, golden_data):
    standard = 0.03
    diff_result = np.isclose(output_data.astype(np.float32), golden_data.astype(np.float32), rtol=standard, atol=0.0001, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    print(f'\nerr_idx: {err_idx}')
    split_count = golden_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    pct_thd = (1 - standard) * 100.0
    result = "pass" if (fulfill_percent >= pct_thd) else "failed"
    color = GREEN if (fulfill_percent >= pct_thd) else RED
    for idx in err_idx:
        cpu = golden_data[idx]
        npu = output_data[idx]
        diff = abs(cpu-npu)
        diff_rate = abs(diff / cpu)
        print(f'diff threshold: {standard}, idx: {idx}, cpu->{cpu}, npu->{npu}, diff->{diff}, diff_rate->{diff_rate} {RESET}')
    print(f"\npass percent is: {fulfill_percent}, error percent threshold is: {pct_thd}", flush=True)
    print(f'\n{color}================ case deepseek indexer attention result {result} ================{RESET} \n')
    return result, fulfill_percent

def read_array_from_bin(filename, dtype):
    """从二进制文件读取NumPy数组"""
    try:
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {filename}")
        return None
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {e}")
        return None

def build_npu_graph():
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_device(0)
    compiler_config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)

    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

    case_name = 'DecodeSparseAttentionSTest.mini'
    base = str(os.getcwd()) + "/gen_golden_data/"
    gen_deepseek_indexer_attention.attention_entry(case_name, base)

    b = 4
    s1 = 1
    h = 7168
    qLoraRank = 1536
    qkNopeHeadDim = 128
    qkRopeHeadDim = 64
    qHeadDim = qkNopeHeadDim + qkRopeHeadDim
    kv_lora_rank = 512
    vHeadDim = 512
    blockSize = 128
    n1 = 128
    n2 = 1
    idx_n_heads = 64
    idx_head_dim = 128
    idx_block_num = 512
    s2 = 128*1024
    dn = kv_lora_rank
    topk = 2048

    nz = True
    dType = torch.bfloat16
    
    case_golden_path = base + case_name + '/'

    act_seqs = read_array_from_bin(case_golden_path + 'kv_cache_actual_seq_len.bin', np.int32).reshape(b)
    
    block_size = 128

    blockNum = sum([(s + block_size - 1) // block_size for s in act_seqs])
    maxBlockNumPerBatch = (max(act_seqs) + block_size - 1) // block_size
    
    nz_prefix = "nz_" if nz else ""
    x = torch.from_file(case_golden_path + 'x.bin', size=b*s1*h, dtype=dType).reshape(b, s1, h).npu()
    w_dq = torch.from_file(case_golden_path + nz_prefix + 'wDq.bin', size=h*qLoraRank, dtype=dType).reshape(h, qLoraRank).npu()
    w_uq_qr = torch.from_file(case_golden_path + nz_prefix + 'wUqQr.bin', size=qLoraRank* n1 * qHeadDim, dtype=dType).reshape(qLoraRank, n1 * qHeadDim).npu()
    w_uk = torch.from_file(case_golden_path + 'wUk.bin', size=n1* qkNopeHeadDim* dn, dtype=dType).reshape(n1, qkNopeHeadDim, dn).npu()
    w_dkv_kr = torch.from_file(case_golden_path + nz_prefix + 'wDkvKr.bin', size=h* (dn + qkRopeHeadDim), dtype=dType).reshape(h, dn + qkRopeHeadDim).npu()
    gamma_cq = torch.from_file(case_golden_path + 'gamma_cq.bin', size=qLoraRank, dtype=dType).reshape(qLoraRank).npu()
    gamma_ckv = torch.from_file(case_golden_path + 'gamma_ckv.bin', size=dn, dtype=dType).reshape(dn).npu()
    sin = torch.from_file(case_golden_path + 'sin.bin', size=b* s1* qkRopeHeadDim, dtype=dType).reshape(b, s1, qkRopeHeadDim).npu()
    cos = torch.from_file(case_golden_path + 'cos.bin', size=b* s1* qkRopeHeadDim, dtype=dType).reshape(b, s1, qkRopeHeadDim).npu()
    cache_index = torch.from_file(case_golden_path + 'k_cache_index.bin', size=b* s1, dtype=torch.int32).reshape(b, s1).npu()
    kv_cache = torch.from_file(case_golden_path + 'kv_cache.bin', size=blockNum* blockSize* n2* dn, dtype=dType).reshape(blockNum, blockSize, n2, dn).npu()
    kr_cache = torch.from_file(case_golden_path + 'kr_cache.bin', size=blockNum* blockSize* n2* qkRopeHeadDim, dtype=dType).reshape(blockNum, blockSize, n2, qkRopeHeadDim).npu()
    block_table = torch.from_file(case_golden_path + 'block_table.bin', size=b* maxBlockNumPerBatch, dtype=torch.int32).reshape(b, maxBlockNumPerBatch).npu()
    act_seqs = torch.from_file(case_golden_path + 'kv_cache_actual_seq_len.bin', size=b, dtype=torch.int32).reshape(b).npu()
    w_idx_qb = torch.from_file(case_golden_path + 'wq_b_nz.bin', size=qLoraRank* idx_n_heads * idx_head_dim, dtype=dType).reshape(qLoraRank, idx_n_heads * idx_head_dim).npu()
    w_idx_k = torch.from_file(case_golden_path + 'wk_nz.bin', size=h* idx_head_dim, dtype=dType).reshape(h, idx_head_dim).npu()
    w_idx_proj = torch.from_file(case_golden_path + 'weights_proj_nz.bin', size=h* idx_n_heads, dtype=dType).reshape(h, idx_n_heads).npu()
    in_gamma_k = torch.from_file(case_golden_path + 'weight_layer_norm.bin', size=idx_head_dim, dtype=dType).reshape(idx_head_dim).npu()
    in_beta_k = torch.from_file(case_golden_path + 'bias_layer_norm.bin', size=idx_head_dim, dtype=dType).reshape(idx_head_dim).npu()
    index_k_cache = torch.from_file(case_golden_path + 'idx_k_cache.bin', size=blockNum* blockSize* n2* idx_head_dim, dtype=dType).reshape(blockNum, blockSize, n2, idx_head_dim).npu()

    # out golden
    dsa_golden = torch.from_file(case_golden_path + 'atten_out.bin', size=b* s1* n1* dn, dtype=dType).reshape(b* s1* n1* dn)

    print("===mmm cache_index", cache_index.cpu(), flush=True)
    print("===mmm block_table", block_table.cpu(), flush=True)
    print("===mmm act_seqs", act_seqs.cpu(), flush=True)
    res = model(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache,
                block_table, act_seqs, w_idx_qb, w_idx_k, w_idx_proj, in_gamma_k, in_beta_k, index_k_cache)
    print(f'{YELLOW}case {case_name} actual seqs shape {act_seqs.shape} s1 {s1} act_seqs {act_seqs} blockNum {blockNum} maxBlockNumPerBatch {maxBlockNumPerBatch} {RESET}')  
    print("===================================res npu===================================")
    print(f"{res[0].cpu()}", flush=True)
    npu_out = res.view(-1).cpu()
    print("res npu shape: ", npu_out.shape)
    precision_compare_float(npu_out.to(torch.float32).numpy(), dsa_golden.to(torch.float32).numpy())

if __name__ == "__main__":
    build_npu_graph()
    print("===execute end ", flush=True)