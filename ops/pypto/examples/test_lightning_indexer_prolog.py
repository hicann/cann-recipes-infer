# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.nn import Parameter
import torch_npu
import math
import numpy as np
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
import torch._dynamo
import sys
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"
import logging
import torchair
import custom_pypto
from pathlib import Path
from goldens import gen_lightning_indexer_prolog as gen_prolog
from bfloat16 import bfloat16

torch.npu.config.allow_internal_format = True

RED = "\033[1;31m"
GREEN = "\033[1;32m"
RESET = "\033[0m"
YELLOW = "\033[93m"

# torchair.logger.setLevel(logging.DEBUG)
torch.manual_seed(0)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, token_x, q_norm, q_norm_scale, wq_b, wq_b_scale, wk, weights_proj, ln_gamma_k, ln_beta_k,
                cos_idx_rope, sin_idx_rope,hadamard_q, hadamard_k, idx_k_cache, idx_k_scale_cache, idx_k_cache_index,
                layernorm_epsilon_k, layout_query="TND", layout_key="PA_BSND"):
        res = torch.ops.custom_pypto.npu_lightning_indexer_prolog_pto(
            token_x=token_x,
            q_norm=q_norm,
            q_norm_scale=q_norm_scale,
            wq_b=wq_b,
            wk=wk,
            wq_b_scale=wq_b_scale,
            weights_proj=weights_proj,
            ln_gamma_k=ln_gamma_k,
            ln_beta_k=ln_beta_k,
            cos_idx_rope=cos_idx_rope,
            sin_idx_rope=sin_idx_rope,
            hadamard_q=hadamard_q,
            hadamard_k=hadamard_k,
            idx_k_cache=idx_k_cache,
            idx_k_scale_cache=idx_k_scale_cache,
            idx_k_cache_index=idx_k_cache_index,
            layernorm_epsilon_k=layernorm_epsilon_k,
            layout_query=layout_query,
            layout_key=layout_key)
        res1 = torch.add(idx_k_cache, 0)
        res2 = torch.add(idx_k_scale_cache, 0)
        return res, res1, res2


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
    logging.info("Entering into single_rope")
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


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x_dtype = x.dtype
    if x_dtype != torch.float32:
        x = x.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return (x * gamma.to(torch.float32) + beta.to(torch.float32)).to(x_dtype)


def quant_int8(x: torch.Tensor):
    # pertoken
    x_dtype = x.dtype  # bf16, (b, s, n, d)
    x_fp32 = x.to(torch.float32)
    max_value = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    y_fp32 = x_fp32 * scale_quant
    y_fp32 = y_fp32.view(x.shape)
    y_int32 = torch.round(y_fp32).to(torch.int32)  # rint mode
    y_int8 = torch.trunc(y_int32.to(x_dtype)).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    # (b, s, n, d) int8, (b, s, n, 1) fp32
    return y_int8, scale_dequant


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
    block_idx_list = block_idx_list[torch.randperm(block_idx_list.size(0))].to(torch.int32)

    block_table = -torch.ones(block_table_shape, dtype=torch.int32)

    block_idx = 0
    block_table_bidx = 0
    for cur_block in block_num_each:
        for j in range(cur_block):
            block_table[block_table_bidx, j] = block_idx_list[block_idx]
            block_idx += 1
        block_table_bidx += 1

    if need_indices:
        cache_index = -torch.ones((b, s1), dtype=torch.int64)
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
    s2_new = ((s2 + block_size - 1) // block_size) * block_size  # ceil to block_size
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


# IndexerProlog quant
def indexer_prolog(inputs: dict, dims: dict):
    # input
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    rope_head_dim = dims["rope_head_dim"]
    x = inputs["token_x"]  # (b, s, h)
    q_norm = inputs["q_norm"]  # (b, s, q_lora_rank), int8
    q_norm_scale = inputs["q_norm_scale"]  # (b, s, 1), fp32
    w_idx_qb = inputs["w_idx_qb"]  # (q_lora_rank, n * d), int8
    w_idx_qb_scale = inputs["w_idx_qb_scale"]  # (1, n * d), fp32
    w_idx_k = inputs["w_idx_k"]  # (h, d)
    w_idx_proj = inputs["w_idx_proj"]  # (h, n)
    layer_norm_gamma = inputs["layer_norm_gamma"]  # (d,)
    layer_norm_beta = inputs["layer_norm_beta"]  # (d,)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    hadamard_q = inputs["hadamard_q"]  # (d, d)
    hadamard_k = inputs["hadamard_k"]  # (d, d)
    idx_k_cache = inputs["idx_k_cache"]  # input13, int8
    idx_k_scale_cache = inputs["idx_k_scale_cache"]  # input14, fp16
    cache_index = inputs["idx_k_cache_index"]  # (b, s), int64
    x_dtype = x.dtype

    # calculate
    q = torch.matmul(q_norm.to(torch.int32), w_idx_qb.to(torch.int32))  # (b, s, n * d)
    q_fp32 = q.to(torch.float32)
    q_fp32 = q_fp32 * q_norm_scale
    q_fp32 = q_fp32 * w_idx_qb_scale.reshape(1, n * d)
    q_bf16 = q_fp32.reshape(b, s, n, d).to(torch.bfloat16)
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q = torch.cat([q_rope, q_nope], dim=-1)
    # hadamard
    q = torch.matmul(q, hadamard_q)  # (b, s, n, d)
    q_int8, q_scale = quant_int8(q)  # (b, s, n, d) int8, (b, s, n, 1) fp32
    q_scale = q_scale.to(torch.float16)

    k = torch.matmul(x.to(torch.float32), w_idx_k.to(torch.float32))  # (b, s, d)
    k = layer_norm(k, layer_norm_gamma, layer_norm_beta).to(x_dtype)
    k_rope, k_nope = torch.split(k, [rope_head_dim, d - rope_head_dim], dim=-1)
    k_rope = single_rope(k_rope.unsqueeze(2), cos, sin).squeeze(2)
    k = torch.cat([k_rope, k_nope], dim=-1)
    # hadamard
    k = torch.matmul(k.to(torch.float32), hadamard_k.to(torch.float32)).to(x_dtype)  # (b, s, d)
    k_int8, k_scale = quant_int8(k)  # (b, s, d) int8, (b, s, 1) fp32
    k_scale = k_scale.to(torch.float16)
    # cache update
    k_cache = idx_k_cache.clone()  # (block_num, block_size, n_kv, d)
    k_scale_cache = idx_k_scale_cache.clone()  # (block_num, block_size, n_kv, 1)
    scatter_update_pa_bsnd(k_cache, k_int8.reshape(b, s, 1, d), cache_index, -2)
    scatter_update_pa_bsnd(k_scale_cache, k_scale.reshape(b, s, 1, 1), cache_index, -2)

    weights = torch.matmul(x, w_idx_proj).to(torch.float32)  # (b, s, n)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.to(torch.float16)

    # output dtype: int8, fp16, int8, fp16, fp16
    outputs = {"query": q_int8, "query_scale": q_scale,
               "idx_k_cache_out": k_cache, "idx_k_scale_cache_out": k_scale_cache,
               "weights": weights}
    return outputs


def gen_dims(params):
    dims = {}
    dims["s2"] = params["s2"]
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 7168
    dims["q_lora_rank"] = 1536
    dims["idx_head_dim"] = 128
    dims["rope_head_dim"] = 64
    dims["idx_n_heads"] = 64
    dims["block_size"] = 128
    dims["block_num"] = dims["b"] * dims["s2"] // dims["block_size"]
    dims["n_kv"] = 1
    return dims


def gen_indexer_prolog_inputs(dims, dtype=torch.bfloat16, qunat_dtype=torch.int8, eps=1e-6):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    block_num = dims["block_num"]
    block_size = dims["block_size"]
    n_kv = dims["n_kv"]
    s2 = dims["s2"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1)
    q_norm = torch.randint(low=-128, high=128, size=(b, s, q_lora_rank), dtype=qunat_dtype)
    q_norm_scale = torch.empty((b, s, 1), dtype=torch.float32).uniform_(-1, 1)

    w_idx_qb = torch.randint(low=-128, high=128, size=(q_lora_rank, n * d), dtype=qunat_dtype)
    w_idx_qb_nz = torch_npu.npu_format_cast(w_idx_qb.npu(), 29)

    w_idx_qb_scale = torch.empty((n * d, 1), dtype=torch.float32).uniform_(-1, 1)

    w_idx_k = torch.empty((h, d), dtype=dtype).uniform_(-1, 1)
    w_idx_k_nz = torch_npu.npu_format_cast(w_idx_k.npu(), 29)

    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1)
    w_idx_proj_nz = torch_npu.npu_format_cast(w_idx_proj.npu(), 29)

    ln_gamma = torch.ones((d,), dtype=dtype)
    ln_beta = torch.zeros((d,), dtype=dtype)

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)  # (128, 128)
    hadamard_k = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)

    act_seq = torch.tensor([s2] * b)  # (b,)
    k_cache_bsnd = torch.randint(low=-128, high=128, size=(b, s2, n_kv, d), dtype=qunat_dtype)
    k_scale_cache_bsnd = torch.empty((b, s2, n_kv, 1), dtype=torch.float16).uniform_(-1, 1)
    # k_cache_index (b, s)
    block_num, block_table, k_cache_index = gen_block_table(act_seq, block_size, s, need_indices=True)
    # (block_num, block_size, n_kv, d), (block_num, block_size, n_kv, 1)
    k_cache = gen_cache_tensor(k_cache_bsnd, block_table, block_num, block_size)
    k_scale_cache = gen_cache_tensor(k_scale_cache_bsnd, block_table, block_num, block_size)

    return {
        "token_x": x,  # input0, bf16
        "q_norm": q_norm,  # input1, int8
        "q_norm_scale": q_norm_scale,  # input2, fp32
        "w_idx_qb": w_idx_qb,  # input3, int8
        "w_idx_qb_nz": w_idx_qb_nz,  # input3 nz, int8
        "w_idx_qb_scale": w_idx_qb_scale,  # input4, fp32
        "w_idx_k": w_idx_k,  # input5, bf16
        "w_idx_k_nz": w_idx_k_nz,  # input5 nz, bf16
        "w_idx_proj": w_idx_proj,  # input6, bf16
        "weights_proj_nz": w_idx_proj_nz,  # input6 nz, bf16
        "layer_norm_gamma": ln_gamma,  # input7, bf16
        "layer_norm_beta": ln_beta,  # input8, bf16
        "cos_idx_rope": cos,  # input9, bf16
        "sin_idx_rope": sin,  # input10, bf16
        "hadamard_q": hadamard_q,  # input11, bf16
        "hadamard_k": hadamard_k,  # input12, bf16
        "idx_k_cache": k_cache,              # input13, int8  # (block_num, block_size, n_kv, d)
        "idx_k_scale_cache": k_scale_cache,  # input14, fp16  # (block_num, block_size, n_kv, 1)
        "idx_k_cache_index": k_cache_index,  # input15, int32  (b, s)/（t,)
        "idx_block_table": block_table,  # input16, int32  (b, ceil(s2, block_size))
        "act_seq": act_seq,  # input17, int32
        "layernorm_epsilon_k": eps,  # attr0, fp32
    }


def precision_compare_float(output_name, output_data, golden_data, standard=0.005):
    diff_result = np.isclose(output_data.astype(np.float32), golden_data.astype(np.float32), rtol=standard, atol=0.0001, equal_nan=True)
    err_idx = np.where(diff_result != np.array((True,)))[0]
    print(f'\noutput [{output_name}] compare {RED}err_idx{RESET}: {err_idx}')
    split_count = golden_data.size
    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0
    pct_thd = (1 - standard) * 100.0
    result = "PASS" if (fulfill_percent >= pct_thd) else "FAILED"
    color = GREEN if (fulfill_percent >= pct_thd) else RED
    for idx in err_idx:
        cpu = golden_data[idx]
        npu = output_data[idx]
        diff = abs(cpu - npu)
        diff_rate = abs(diff / cpu)

        # diff在 正负1之内 正常
        if (diff > 1.0):
            print(f'diff threshold: {standard}, idx: {idx}, cpu->{cpu}, npu->{npu}, diff->{diff}, diff_rate->{diff_rate} {RESET}')

    print(f"\n{output_name} pass percent is: {fulfill_percent}, error percent threshold is: {pct_thd}, result is {color}{result}. {RESET}", flush=True)
    return result, fulfill_percent


def build_npu_graph():
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_device(0)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    model = Model()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

    print("=========== start gen lightning indexer golden=============:")
    ########################  gen golden
    ### param
    dtype = torch.bfloat16

    b = 4
    s1 = 2
    t = b * s1
    s2 = 1024 * 64 # 64k
    n_kv = 1
    block_size = 128

    params = {}
    params["b"] = b
    params["s1"] = s1
    params["s2"] = s2

    dims = gen_dims(params)
    dim_tensor = torch.tensor(list(dims.values()), dtype=torch.int32)
    print("params:", dim_tensor, flush=True)

    input_data_map = gen_indexer_prolog_inputs(dims, torch.bfloat16)
    outputs = indexer_prolog(input_data_map, dims)

    token_x = input_data_map["token_x"]
    q_norm = input_data_map["q_norm"]
    q_norm_scale = input_data_map["q_norm_scale"]

    headNum, headDim = dims["idx_n_heads"], dims["idx_head_dim"]
    headMul = headNum * headDim
    # 使用wq_b_nz
    wq_b_nz = input_data_map["w_idx_qb_nz"]

    wq_b_scale = input_data_map["w_idx_qb_scale"].npu()

    h = dims["h"]
    # 使用wk_nz
    wk_nz = input_data_map["w_idx_k_nz"]

    # 使用weights_proj_nz
    weights_proj_nz = input_data_map["weights_proj_nz"]
    weights_proj = weights_proj_nz

    ln_gamma_k = input_data_map["layer_norm_gamma"].npu()
    ln_beta_k = input_data_map["layer_norm_beta"].npu()
    cos_idx_rope = input_data_map["cos_idx_rope"].npu()
    sin_idx_rope = input_data_map["sin_idx_rope"].npu()
    hadamard_q = input_data_map["hadamard_q"].npu()
    hadamard_k = input_data_map["hadamard_k"].npu()
    idx_k_cache = input_data_map["idx_k_cache"].npu()
    idx_k_cache_index = input_data_map["idx_k_cache_index"].npu()
    idx_k_scale_cache = input_data_map["idx_k_scale_cache"].reshape(dims["block_num"], block_size, n_kv).npu()
    idx_block_table = input_data_map["idx_block_table"].npu()
    act_seq = input_data_map["act_seq"].npu()
    layernorm_epsilon_k = input_data_map["layernorm_epsilon_k"]

    # b/s 合轴成 t
    token_x = token_x.reshape(t, token_x.shape[2]).npu()
    q_norm = q_norm.reshape(t, q_norm.shape[2]).npu()
    q_norm_scale = q_norm_scale.reshape(t, q_norm_scale.shape[2]).npu()
    cos_idx_rope = cos_idx_rope.reshape(t, cos_idx_rope.shape[2]).npu()
    sin_idx_rope = sin_idx_rope.reshape(t, sin_idx_rope.shape[2]).npu()
    idx_k_cache_index = idx_k_cache_index.reshape(t).npu()

    prolog_res, res1, res2 = model(
        token_x, q_norm, q_norm_scale, wq_b_nz, wq_b_scale, wk_nz, weights_proj, ln_gamma_k, ln_beta_k,
        cos_idx_rope, sin_idx_rope, hadamard_q, hadamard_k, idx_k_cache, idx_k_scale_cache, idx_k_cache_index,
        layernorm_epsilon_k)

    # tQuery = t * headMul
    output_query = outputs["query"].view(-1)
    res_query = prolog_res[0].view(-1).cpu()
    precision_compare_float("query", res_query.to(torch.float32).numpy(), output_query.to(torch.float32).numpy())

    print(f" ")

    # tQueryScale = t * headNum
    output_key = outputs["query_scale"].view(-1)
    res_key = prolog_res[1].view(-1).cpu()
    precision_compare_float("query_scale", res_key.to(torch.float32).numpy(), output_key.to(torch.float32).numpy())

    print(f" ")

    # tWeight = t * headNum
    output_weight = outputs["weights"].view(-1)
    res_weight = prolog_res[2].view(-1).cpu()
    precision_compare_float("weights", res_weight.to(torch.float32).numpy(), output_weight.to(torch.float32).numpy())

    print(f" ")

    output_k_cache = outputs["idx_k_cache_out"].view(-1)
    res1_out = res1.view(-1).cpu()
    precision_compare_float("idx_k_cache_out", res1_out.to(torch.float32).numpy(), output_k_cache.to(torch.float32).numpy())

    print(f" ")

    output_k_scale_cache = outputs["idx_k_scale_cache_out"].view(-1)
    res2_out = res2.view(-1).cpu()
    precision_compare_float("idx_k_scale_cache_out", res2_out.to(torch.float32).numpy(), output_k_scale_cache.to(torch.float32).numpy())

    print(f" ")
    print(f"{GREEN}================case lightning indexer prolog run completed================{RESET} \n")

    return prolog_res, res1, res2


if __name__ == "__main__":
    res, res1, res2 = build_npu_graph()
    print("===execute end ", flush=True)