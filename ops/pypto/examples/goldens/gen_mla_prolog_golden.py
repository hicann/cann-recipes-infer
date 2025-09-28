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

import sys
import math
import time
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np
from bfloat16 import bfloat16


if __name__ == "__main__":
    """ 单独执行时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
#     from golden_register import GoldenRegister  # 单独执行 import 失败, 需确认上文中 '系统 import 路径' 配置正确
# else:
#     from golden_register import GoldenRegister

fp32 = np.float32


def rms_norm(x, gamma, eps):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.astype(fp32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = np.sum(mean_res, axis=-1, keepdims=True) + eps
    reduce_sqrt = np.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != fp32:
        res = res.astype(x_dtype)
    return res


def scatter_update_bnsd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [b, 1, s2, d]
    # key_states shape: [b, 1, s1, d]
    # indices shape: [b, s1]
    cache, key_states, indices = inputs
    b, n2, s2, d = cache.shape  # n2=1
    s1 = indices.shape[1]
    res = cache

    if axis == -2:
        for b_i in range(b):
            for s2_i in range(s2):
                for s1_i in range(s1):
                    index_value = indices[b_i][s1_i]
                    if s2_i == index_value:
                        logging.debug("find the index value and to replace!")
                        res[b_i][0][s2_i][:] = key_states[b_i][0][s1_i][:]

    return res


def scatter_update_pa_bsnd(inputs, axis):
    # inputs: cache, key_states, indices
    # cache shape: [block_number,block_size,n2,d], n2=1
    # key_states shape: [b*s1*1, d]
    # indices shape: [b, s1], s1=1
    cache, key_states, indices = inputs
    block_number, block_size, n2, d = cache.shape
    res = cache.reshape(block_number * block_size * n2, d)
    b, s1 = indices.shape

    if axis == -2:
        for b_i in range(b):
            for s1_i in range(s1):
                index_value = indices[b_i][s1_i]
                res[index_value][:] = key_states[b_i * s1 + s1_i][:]

    return res.reshape(block_number, block_size, n2, d)


def scatter_update(inputs, axis, cache_mode="BNSD"):
    if cache_mode != "BNSD":
        return scatter_update_pa_bsnd(inputs, axis)
    else:
        return scatter_update_bnsd(inputs, axis)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=2):
    input_dtype = q.dtype
    if input_dtype != fp32:
        q = q.astype(fp32)
        k = k.astype(fp32)
    if cos.dtype != fp32:
        cos = cos.astype(fp32)
        sin = sin.astype(fp32)

    cos = np.expand_dims(cos, axis=unsqueeze_dim)  # [b,s,1,qk_d]
    sin = np.expand_dims(sin, axis=unsqueeze_dim)  # [b,s,1,qk_d]
    logging.debug("expand sin.shape: %s", sin.shape)
    logging.debug("expand cos.shape: %s", cos.shape)

    b, s, h, d = q.shape
    q = q.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,n,qk_d]

    b, s, h, d = k.shape
    k = k.reshape(b, s, h, d // 2, 2).transpose(0, 1, 2, 4, 3).reshape(b, s, h, d)  # [b,s,1,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != fp32:
        q_embed, k_embed = q_embed.astype(input_dtype), k_embed.astype(input_dtype)
    return q_embed, k_embed


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.astype(fp32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def to_file(data, dir, name):
    bin_path = Path(dir, f'{name}')
    data.tofile(bin_path)


def mla_prolog_compute(inputs):
    dtype = inputs.get("dtype")
    is_quant_a = inputs.get("is_quant_a")
    is_quant_b = inputs.get("is_quant_b")
    has_smooth = inputs.get("has_smooth")
    cache_mode = inputs.get("cache_mode")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    epsilon = inputs.get("epsilon")
    x = inputs.get("x")
    w_dq = inputs.get("w_dq")
    w_uqqr = inputs.get("w_uqqr")
    w_uk = inputs.get("w_uk")
    w_dkvkr = inputs.get("w_dkvkr")
    cos = inputs.get("cos")
    sin = inputs.get("sin")
    kv_cache = inputs.get("kv_cache")
    kr_cache = inputs.get("kr_cache")
    cache_index = inputs.get("cache_index")
    if is_quant_a:
        w_qa_scale = inputs.get("w_qa_scale")
        w_kva_scale = inputs.get("w_kva_scale")
    if is_quant_b:
        w_qb_scale = inputs.get("w_qb_scale")
        if has_smooth:
            smooth_cq = inputs.get("smooth_cq")

    b, s, h = x.shape
    qk_rope_head_dim = cos.shape[2]
    n, qk_nope_head_dim, kv_lora_rank = w_uk.shape
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    """ q """
    x_2d = x.reshape(b * s, h)
    # shape is: [b * s, h] @ [h, q_lora_rank] -> [b * s, q_lora_rank]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        q_a_proj = np.matmul(x_2d_quant.astype(np.int32), w_dq.astype(np.int32))

        """ dequant """
        q_a_proj_fp32 = q_a_proj.astype(fp32)
        q_a_proj_fp32_dequant = q_a_proj_fp32 * x_2d_scale_dequant
        q_a_proj = q_a_proj_fp32_dequant * w_qa_scale
    else:
        q_a_proj = np.matmul(x_2d.astype(fp32), w_dq.astype(fp32))  # [b * s, q_lora_rank]

    q_a_proj = q_a_proj.astype(dtype)

    q_a_layernorm = rms_norm(q_a_proj, gamma_cq, epsilon)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    # shape is: [b * s, q_lora_rank] @ [q_lora_rank, n * q_head_dim] -> [b * s, n * q_head_dim]
    if is_quant_b:
        if has_smooth:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True, True, smooth_cq)
        else:
            q_a_layernorm, q_a_layernorm_scale_dequant = quant(q_a_layernorm, True)  # scale: [b*s,1]
        q_b_proj = np.matmul(q_a_layernorm.astype(np.int32), w_uqqr.astype(np.int32))  # q_b_proj

        """ dequant """
        q_b_proj_fp32 = q_b_proj.astype(fp32)
        q_b_proj_fp32_dequant = q_b_proj_fp32 * q_a_layernorm_scale_dequant
        q_b_proj = q_b_proj_fp32_dequant * w_qb_scale
    else:
        q_b_proj = np.matmul(q_a_layernorm.astype(fp32), w_uqqr.astype(fp32))  # [b * s, n * q_head_dim]

    q_b_proj = q_b_proj.astype(dtype)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)

    q_reshape = q_b_proj.reshape(b, s, n, q_head_dim)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    q_nope = q_reshape[:, :, :, 0:qk_nope_head_dim]  # [b, s, n, qk_nope_head_dim]
    q_nope_r = q_nope.reshape(b * s, n, qk_nope_head_dim)
    q_nope_t = q_nope_r.transpose(1, 0, 2)  # [n, b*s, qk_nope_head_dim]
    # shape is: [n, b*s, qk_nope_head_dim] @ [n, qk_nope_head_dim, kv_lora_rank] -> [n, b*s, kv_lora_rank]
    q_nope_new = np.matmul(q_nope_t.astype(fp32), w_uk.astype(fp32))
    q_nope_new = q_nope_new.astype(dtype)
    q_nope_new_t = q_nope_new.transpose(1, 0, 2)  # [b*s, n, kv_lora_rank]
    q_out = q_nope_new_t.reshape(b, s, n, kv_lora_rank)  # [b, s, n, kv_lora_rank]

    """ kv """
    # shape is: [b*s, h] @ [h, kv_lora_rank + qk_rope_head_dim] -> [b*s, kv_lora_rank + qk_rope_head_dim]
    if is_quant_a:
        # no smooth
        x_2d_quant, x_2d_scale_dequant = quant(x_2d, True)
        kv_a_proj = np.matmul(x_2d_quant.astype(np.int32), w_dkvkr.astype(np.int32))
        """ dequant """
        kv_a_proj_fp32 = kv_a_proj.astype(fp32)
        kv_a_proj_fp32_dequant = kv_a_proj_fp32 * x_2d_scale_dequant
        kv_a_proj = kv_a_proj_fp32_dequant * w_kva_scale
    else:
        kv_a_proj = np.matmul(x_2d.astype(fp32), w_dkvkr.astype(fp32))  # [b*s, kv_lora_rank + qk_rope_head_dim]

    kv_a_proj = kv_a_proj.astype(dtype)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj.reshape(b, s, kv_lora_rank + qk_rope_head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    compressed_kv = kv_reshape[:, :, 0:kv_lora_rank]  # [b, s, kv_lora_rank]
    compressed_kv_norm = rms_norm(compressed_kv, gamma_ckv, epsilon)
    compressed_kv_r = compressed_kv_norm.reshape(b, s, 1, kv_lora_rank)
    if cache_mode != "BNSD":
        k_nope = compressed_kv_r.reshape(b * s * 1, kv_lora_rank)
    else:
        k_nope = compressed_kv_r.transpose(0, 2, 1, 3)  # [b, 1, s, kv_lora_rank]

    """ RoPE """
    q_pe = q_reshape[:, :, :, qk_nope_head_dim:]  # [b, s, n, qk_rope_head_dim]

    k_pe = kv_reshape[:, :, kv_lora_rank:]  # [b, s, qk_rope_head_dim]
    k_pe_r = k_pe.reshape(b, s, 1, qk_rope_head_dim)

    # q_embed: [b, s, n, qk_rope_head_dim], k_embed: [b, s, 1, qk_rope_head_dim]
    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 2)
    if cache_mode != "BNSD":
        k_embed_r = k_embed.reshape(b * 1 * s, qk_rope_head_dim)
    else:
        k_embed_r = k_embed.reshape(b, 1, s, qk_rope_head_dim)

    """ kv_cache output, [b,1,s2,kv_lora_rank] """
    kv_cache_out = scatter_update([kv_cache, k_nope, cache_index], -2, cache_mode)

    """ kr_cache output, [b,1,s2,qk_rope_head_dim] """
    kr_cache_out = scatter_update([kr_cache, k_embed_r, cache_index], -2, cache_mode)

    return q_out, q_embed, kv_cache_out, kr_cache_out, q_a_layernorm


def gen_block_table(b, actual_seq_len, block_size):
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    slc_s_max = max(actual_seq_len)
    # gen block table [b, slc_s_max/block_size]
    block_table_shape = [b, math.ceil(slc_s_max / block_size)]
    block_num = block_num_min

    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    return block_num, block_table


def gen_block_input_data(b, s2, block_size):
    if isinstance(s2, int):
        kv_cache_actual_seq = [s2] * b
    elif isinstance(s2, list):
        if len(s2) == b:
            kv_cache_actual_seq = s2
        else:
            raise RuntimeError("unsupported this kv_cache_actual_seq")
    else:
        raise RuntimeError("unsupported kv_cache_actual_seq data type")
    skv_max = max(kv_cache_actual_seq)
    block_num, block_table = gen_block_table(b, kv_cache_actual_seq, block_size)
    return skv_max, block_num, block_table


def gen_prolog_input_data(params, dtypes, epsilon, output_dir: Path, is_quant=(False, False), is_nz=False,
                          has_smooth=False, block_size=128, cache_mode="BNSD"):
    dtype, w_dtype = dtypes
    logging.debug(f"gen_prolog_input_data  dtype:{dtype}, w_dtype:{w_dtype}")
    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s = params.get("s")  # s=1 or 2
    s2 = params.get("s2")  # s2=4k
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    v_head_dim = params.get("v_head_dim")
    param_block_num = params.get("block_num", None)
    block_table = params.get("block_table", None)
    skv_max = params.get("skv_max", None)
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    NzFrac = 16 if dtype != float else 8
    x_shape = [b, s, h]
    w_qa_shape = [h, q_lora_rank]
    w_qb_shape = [q_lora_rank, n * q_head_dim]
    w_kv_a_shape = [h, kv_lora_rank + qk_rope_head_dim]
    w_kv_b_k_shape = [n, qk_nope_head_dim, kv_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [kv_lora_rank]
    cos_shape = [b, s, qk_rope_head_dim]
    kv_len_shape = [b, s]
    kv_cache_shape = [b, 1, s2, kv_lora_rank]
    kr_cache_shape = [b, 1, s2, qk_rope_head_dim]
    index_value_max = s2
    if cache_mode != "BNSD":
        if not param_block_num:
            block_num = b * (math.ceil(s2 / block_size))
        else:
            block_num = param_block_num
            kv_bsnd_shape = [b, skv_max, 1, kv_lora_rank + qk_rope_head_dim]
        kv_cache_shape = [block_num, block_size, 1, kv_lora_rank]
        kr_cache_shape = [block_num, block_size, 1, qk_rope_head_dim]
        index_value_max = block_num * block_size
    smooth_cq_shape = [1, q_lora_rank]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_dq shape is %s", w_qa_shape)
    logging.debug("w_uqqr shape is %s", w_qb_shape)
    logging.debug("w_dkvkr shape is %s", w_kv_a_shape)
    logging.debug("w_uk shape is %s", w_kv_b_k_shape)
    logging.debug("cos sin shape is %s", cos_shape)
    logging.debug("cgamma_cq shape is %s", gamma_cq_shape)
    logging.debug("cgamma_ckv shape is %s", gamma_ckv_shape)
    logging.debug("kv_len shape is %s", kv_len_shape)
    logging.debug("kv_cache shape is %s", kv_cache_shape)
    logging.debug("kr_cache shape is %s", kr_cache_shape)

    nz_prefix = 'nz_'
    x_path = Path(output_dir, 'x.bin')
    w_dq_path = Path(output_dir, 'wDq.bin')
    w_dq_nz_path = Path(output_dir, nz_prefix + 'wDq.bin')
    w_qa_scale_path = Path(output_dir, 'w_qa_scale.bin')
    w_uqqr_path = Path(output_dir, 'wUqQr.bin')
    w_uqqr_nz_path = Path(output_dir, nz_prefix + 'wUqQr.bin')
    w_qb_scale_path = Path(output_dir, 'w_qb_scale.bin')
    w_dkvkr_path = Path(output_dir, 'wDkvKr.bin')
    w_dkvkr_nz_path = Path(output_dir, nz_prefix + 'wDkvKr.bin')
    w_kva_scale_path = Path(output_dir, 'w_kva_scale.bin')
    w_uk_path = Path(output_dir, 'wUk.bin')  # kv_b_proj_w_k
    w_uk_nz_path = Path(output_dir, nz_prefix + 'wUk.bin')
    gamma_cq_path = Path(output_dir, 'gamma_cq.bin')
    gamma_ckv_path = Path(output_dir, 'gamma_ckv.bin')
    cos_path = Path(output_dir, 'cos.bin')
    sin_path = Path(output_dir, 'sin.bin')
    kv_len_path = Path(output_dir, 'kv_len.bin')
    kv_cache_path = Path(output_dir, 'kv_cache.bin')
    kr_cache_path = Path(output_dir, 'kr_cache.bin')
    smooth_cq_path = Path(output_dir, 'smooth_cq.bin')

    res = [None] * 14
    x = np.random.uniform(-1, 1, x_shape).astype(dtype)
    x.tofile(x_path)
    res[0] = x
    w_dq = np.random.uniform(-0.1, 0.1, w_qa_shape).astype(w_dtype)
    w_uqqr = np.random.uniform(-0.1, 0.1, w_qb_shape).astype(w_dtype)
    w_dkvkr = np.random.uniform(-0.1, 0.1, w_kv_a_shape).astype(w_dtype)
    res[4] = dict()

    if is_quant_a:
        w_dq, w_qa_scale = quant(w_dq, False)
        w_dkvkr, w_kva_scale = quant(w_dkvkr, False)
        w_qa_scale.tofile(w_qa_scale_path)
        w_kva_scale.tofile(w_kva_scale_path)
        res[4]["w_dq"] = w_qa_scale
        res[4]["w_dkvkr"] = w_kva_scale
        w_dq.reshape(h, q_lora_rank // 32, 32).transpose(1, 0, 2).tofile(w_dq_nz_path)
        w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 32, 32).transpose(1, 0, 2).tofile(w_dkvkr_nz_path)
        w_dq.tofile(w_dq_path)
        w_dkvkr.tofile(w_dkvkr_path)
    else:
        w_dq.reshape(h, q_lora_rank // 16, 16).transpose(1, 0, 2).tofile(w_dq_nz_path)
        w_dkvkr.reshape(h, (kv_lora_rank + qk_rope_head_dim) // 16, 16).transpose(1, 0, 2).tofile(w_dkvkr_nz_path)
        w_dq.tofile(w_dq_path)
        w_dkvkr.tofile(w_dkvkr_path)

    if is_quant_b:
        w_uqqr, w_qb_scale = quant(w_uqqr, False)
        w_qb_scale.tofile(w_qb_scale_path)
        res[4]["w_uqqr"] = w_qb_scale
        # smooth_data
        if has_smooth:
            smooth_cq = np.random.uniform(-1, 1, smooth_cq_shape).astype(np.float32)
            smooth_cq.tofile(smooth_cq_path)
            res[3] = smooth_cq
        w_uqqr.reshape(q_lora_rank, n * q_head_dim // 32, 32).transpose(1, 0, 2).tofile(w_uqqr_nz_path)
        w_uqqr.tofile(w_uqqr_path)
    else:
        w_uqqr.reshape(q_lora_rank, n * q_head_dim // 16, 16).transpose(1, 0, 2).tofile(w_uqqr_nz_path)
        w_uqqr.tofile(w_uqqr_path)

    res[1] = w_dq
    res[2] = w_uqqr
    res[5] = w_dkvkr

    w_uk = np.random.uniform(-0.1, 0.1, w_kv_b_k_shape).astype(w_dtype)
    w_uk.reshape(n * qk_nope_head_dim, kv_lora_rank // 16, 16).transpose(1, 0, 2).tofile(w_uk_nz_path)
    w_uk.tofile(w_uk_path)
    res[6] = w_uk
    gamma_cq = np.random.uniform(-1, 1, gamma_cq_shape).astype(dtype)  # [q_lora_rank]
    gamma_ckv = np.random.uniform(-1, 1, gamma_ckv_shape).astype(dtype)  # [kv_lora_rank]
    gamma_cq.tofile(gamma_cq_path)
    gamma_ckv.tofile(gamma_ckv_path)
    res[7] = gamma_cq
    res[8] = gamma_ckv
    cos = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [b, s, qk_rope_head_dim]
    sin = np.random.uniform(-0.1, 0.1, cos_shape).astype(dtype)  # [b, s, qk_rope_head_dim]
    cos.tofile(cos_path)
    sin.tofile(sin_path)
    res[9] = cos
    res[10] = sin
    kv_len = np.random.choice(np.arange(0, index_value_max), size=kv_len_shape, replace=False).astype(np.int64)
    kv_len.tofile(kv_len_path)
    res[11] = kv_len
    kv_cache = np.random.uniform(-1, 1, kv_cache_shape).astype(dtype)
    kr_cache = np.random.uniform(-1, 1, kr_cache_shape).astype(dtype)
    if param_block_num:
        k_bsnd = np.random.uniform(-1, 1, kv_bsnd_shape).astype(dtype)
        v_bsnd = k_bsnd[:, :, :, : kv_lora_rank]
        # kv paddIng
        per_batch_max_num = math.ceil(skv_max / block_size)
        k_tensor_bsnd = np.zeros((b, per_batch_max_num * block_size, 1, kv_lora_rank + qk_rope_head_dim)).astype(dtype)
        k_tensor_bsnd[:, :k_bsnd.shape[1], :, :] = k_bsnd[:, :, :, :]
        # kv_cache
        k_cache_tensor = np.zeros([block_num, block_size, 1, kv_lora_rank + qk_rope_head_dim]).astype(dtype)
        for b_idx in range(b):
            for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
                block_offset = block_i * block_size
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache_tensor[kv_cache_blk_id, 0:block_size, :, :] = k_tensor_bsnd[
                                                                b_idx, block_offset:(block_offset + block_size), :, :]
        kv_cache = k_cache_tensor[:, :, :, : kv_lora_rank]
        kr_cache = k_cache_tensor[:, :, :, kv_lora_rank :]
    if cache_mode == "PA_NZ":
        kr_cache.reshape((block_num, block_size, qk_rope_head_dim // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kr_cache_path)
        kv_cache.reshape((block_num, block_size, kv_lora_rank // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kv_cache_path)
    else:
        kr_cache.tofile(kr_cache_path)  # kr_cache in
        kv_cache.tofile(kv_cache_path)  # kv_cache in
    res[12] = kv_cache
    res[13] = kr_cache

    return res


def gen_mla_prolog_data(params, dtypes, epsilon, output_dir: Path, is_quant=(False, False), is_nz=False,
                        has_smooth=False, block_size=128, cache_mode="BNSD"):
    np.random.seed(int(time.time()))
    dtype, w_dtype = dtypes
    logging.debug(f"gen_mla_prolog_data  dtype:{dtype}, w_dtype:{w_dtype}")
    x, w_dq, w_uqqr, smooth_cq, scale_data, w_dkvkr, w_uk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache = \
        gen_prolog_input_data(params, dtypes, epsilon, output_dir, is_quant, is_nz, has_smooth, block_size, cache_mode)
    is_quant_a, is_quant_b = is_quant
    b = params.get("b")
    s2 = params.get("s2")  # s2=4k
    qk_rope_head_dim = params.get("qk_rope_head_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    NzFrac = 16 if dtype != float else 8
    if cache_mode != "BNSD":
        block_num = b * (s2 // block_size)
    # output
    q_golden_path = Path(output_dir, 'q_golden.bin')
    q_rope_golden_path = Path(output_dir, 'q_rope_golden.bin')
    kv_golden_path = Path(output_dir, 'kv_cache_golden.bin')
    kr_golden_path = Path(output_dir, 'kr_cache_golden.bin')

    inputs = {"dtype": dtype, "is_quant_a": is_quant_a, "is_quant_b": is_quant_b, "has_smooth": has_smooth}
    inputs["cache_mode"] = cache_mode
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv
    inputs["epsilon"] = epsilon
    inputs["x"] = x
    inputs["w_dq"] = w_dq
    inputs["w_uqqr"] = w_uqqr
    inputs["w_uk"] = w_uk
    inputs["w_dkvkr"] = w_dkvkr
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["kv_cache"] = kv_cache
    inputs["kr_cache"] = kr_cache
    inputs["cache_index"] = kv_len
    if is_quant_a:
        inputs["w_qa_scale"] = scale_data["w_dq"]
        inputs["w_kva_scale"] = scale_data["w_dkvkr"]
    if is_quant_b:
        inputs["w_qb_scale"] = scale_data["w_uqqr"]
        if has_smooth:
            inputs["smooth_cq"] = smooth_cq

    q_out, q_embed, kv_cache_out, kr_cache_out, _ = mla_prolog_compute(inputs)

    q_out.tofile(q_golden_path)  # [b,s,n,kv_lora_rank]
    q_embed.tofile(q_rope_golden_path)  # [b,s,n,qk_rope_head_dim]

    if cache_mode == "PA_NZ":
        kr_cache_out.reshape((block_num, block_size, qk_rope_head_dim // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kr_golden_path)
        kv_cache_out.reshape((block_num, block_size, kv_lora_rank // NzFrac, NzFrac)).transpose(0, 2, 1, 3).tofile(kv_golden_path)
        kr_cache_out = kr_cache_out.transpose(0, 2, 1, 3)
        kv_cache_out = kv_cache_out.transpose(0, 2, 1, 3)
    else:
        kr_cache_out.tofile(kr_golden_path)
        kv_cache_out.tofile(kv_golden_path)

    return q_out, q_embed, kv_cache_out, kr_cache_out


def gen_mla_prolog_test1(dtypes, bns2, epsilon, output_dir: Path, is_quant=False):
    b, n, s2 = bns2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": 1,
        "s2": s2,
        "h": 256,
        "num_heads": n,
        "q_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice)


def gen_mla_prolog_test_net(dtypes, bns2, epsilon, output_dir: Path, is_quant=False, is_nz=False, is_smooth=False,
                            block_size=128, cache_mode="BNSD"):
    b, n, s2 = bns2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": 1,
        "s2": s2,
        "h": 7168,
        "num_heads": n,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)


def gen_nsa_prolog_test(dtypes, bns1s2, epsilon, output_dir: Path, quant_a=False, quant_b=True, is_smooth=True,
                        is_nz=True, block_size=128, cache_mode="PA_BSND"):
    b, n, s1, s2 = bns1s2
    quant_choice = (quant_a, quant_b)
    skv_max, block_num, block_table = gen_block_input_data(b, s2, block_size)
    params = {
        "b": b,
        "s": s1,
        "s2": s2,   # 128K
        "h": 7168,
        "num_heads": n,     # n1
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,     # rope_dim
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "block_num": block_num,
        "block_table": block_table,
        "skv_max": skv_max,
    }
    gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)


def gen_prolog_data_small(dtypes, bn1s1s2, epsilon, output_dir: Path, is_quant=False, is_nz=False,
                          is_smooth=False, block_size=128, cache_mode="BNSD"):
    b, n, s1, s2 = bn1s1s2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "h": 256,
        "num_heads": n,
        "q_lora_rank": 256,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)


def gen_prolog_data(dtypes, bn1s1s2, epsilon, output_dir: Path, is_quant=False, is_nz=False,
                    is_smooth=False, block_size=128, cache_mode="BNSD"):
    b, n, s1, s2 = bn1s1s2
    quant_choice = (False, is_quant)
    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "h": 7168,
        "num_heads": n,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
    }
    gen_mla_prolog_data(params, dtypes, epsilon, output_dir, quant_choice, is_nz, is_smooth, block_size, cache_mode)


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == 'fp16':
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == 'fp32':
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == 'fp64':
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == 'int32':
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == 'int64':
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == 'uint8':
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == 'uint16':
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == 'uint32':
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == 'uint64':
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == 'complex64':
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == 'complex128':
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == 'bool':
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == 'bf16':
        np.array(data_pool).astype(bfloat16).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def trans_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        b = shape[0]
        n = shape[1]
        s = shape[2]
        d = shape[3]
        h = n * d
        return tensor.transpose(0, 2, 1, 3).reshape(b, s, h)
    else:
        return tensor


def split_tensor_shape_by_b(input_list):
    # [[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    list_len = input_list[0]
    list_new = []
    for _ in range(0, list_len):
        list_new_item = [1, input_list[1], input_list[2], input_list[3]]
        list_new.append(list_new_item)
    return list_new


def split_tensor_by_b(input_tensor):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    return split_data


def softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_sum, x_max


def ifa_pa_func(params, output: Path, q_res, kv_res):
    b, n_q, skv, block_size, a_q_no, a_q_ro, a_kv_no, a_kv_ro = params

    dtype = bfloat16
    n_kv = 1
    kv_lora_rank = 512
    qk_rope_dim = 64

    # q head dim
    d_q = kv_lora_rank + qk_rope_dim

    # k head dim
    d_k = kv_lora_rank + qk_rope_dim

    # v head dim
    d_v = kv_lora_rank

    sq = 1
    scalar = 0.8  # 临时
    actual_seq_len = [skv] * b
    s_max = max(actual_seq_len)

    shape_q = [b, n_q, sq, d_q]
    shape_k = [b, n_kv, s_max, d_k]
    shape_v = [b, n_kv, s_max, d_v]

    atten_out_shape = [b, n_q, sq, d_v]

    block_num_per_block = []
    block_num_min = 0
    block_num = 0

    # gen q k v data
    q_bnsd = gen_uniform_data(shape_q, -1, 1, dtype)
    k_bnsd = gen_uniform_data(shape_k, -1, 1, dtype)
    v_bnsd = gen_uniform_data(shape_v, -1, 1, dtype)

    if q_res.max() != 0 and kv_res.max() != 0:
        q_bnsd = q_res
        k_bnsd = kv_res
        v_bnsd = kv_res[:, :, :, :512]  # k0

    for actual_seq in actual_seq_len:
        block_num_per_block.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    # 处理pageatten场景（block table, kv cache处理不涉及cpu、真值计算，仅为npu生成输入）：
    # 1、生成随机的block_table，并覆写原有bin文件
    # 2、将kv shape 统一转换成bsh后处理
    # 3、生成kv cache
    # 4、将kv cache dump成新的bin文件，供aclnn接口调用

    # gen block table [b, s_max/block_size]
    block_table_shape = [b, math.ceil(s_max / block_size)]
    block_num = block_num_min

    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    # invalid block_id set as -1
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_block:
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1
    # input_t = np.arange(0, b *n *s *d, 1).reshape(qkv_shape).astype(dtype_f32)
    block_table = np.arange(0, b * skv // block_size, 1).reshape(b, skv // block_size).astype(np.int32)
    logging.debug(f"block_table : {block_table}")

    # gen kv cache. [block_num , block_size, H]
    k_cache = np.zeros([block_num, block_size, n_kv * d_k]).astype(dtype)
    v_cache = np.zeros([block_num, block_size, n_kv * d_v]).astype(dtype)

    logging.debug(f"dtype {type(k_bnsd)}, shape {k_bnsd.shape}")

    k_tensor_bsh_raw = trans_bnsd_to_bsh(k_bnsd, shape_k)
    v_tensor_bsh_raw = trans_bnsd_to_bsh(v_bnsd, shape_v)

    # kv paddIng
    k_tensor_bsh = np.zeros((b, block_table_shape[1] * block_size, n_kv * d_k)).astype(dtype)
    v_tensor_bsh = np.zeros((b, block_table_shape[1] * block_size, n_kv * d_v)).astype(dtype)

    k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
    v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]

    for b_idx in range(b):
        for block_i, kv_cache_blk_id in enumerate(block_table[b_idx]):
            block_offset = block_i * block_size
            if kv_cache_blk_id == -1:
                continue
            else:
                k_cache[kv_cache_blk_id, 0:block_size, :] = k_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]
                v_cache[kv_cache_blk_id, 0:block_size, :] = v_tensor_bsh[
                                                            b_idx, block_offset:(block_offset + block_size), :]

    # calculate result
    attent_out = np.zeros(atten_out_shape, dtype=np.float32)

    # 处理连续场景：将单个tensor依据B值拆成列表

    k_tensor_list = split_tensor_by_b(k_bnsd)
    v_tensor_list = split_tensor_by_b(v_bnsd)

    for b_index in range(b):
        matmul_dtype = np.float32

        act_seq = actual_seq_len[b_index]

        k_sub_tensor = k_tensor_list[b_index]
        v_sub_tensor = v_tensor_list[b_index]

        q_tensor_cur = q_bnsd[b_index:(b_index + 1), :, :, :]
        k_cur = k_sub_tensor[:, :, :act_seq, :]
        v_cur = v_sub_tensor[:, :, :act_seq, :]

        # MM1
        qk_bmm_res = np.matmul(q_tensor_cur, k_cur.transpose(0, 1, 3, 2), dtype=matmul_dtype)
        qk_ele_res = qk_bmm_res * scalar
        softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)

        # MM2
        bmm2_res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
        attent_out[b_index:(b_index + 1), :, :, :] = bmm2_res

    # data split to [nope + rope]
    q_nope = q_bnsd[:, :, :, : kv_lora_rank]
    q_rope = q_bnsd[:, :, :, kv_lora_rank:]

    # BBH split [B B kv_lora_rank]  + [B B rope]
    k_cache_nope_h = kv_lora_rank * n_kv
    k_cache_nope = k_cache[:, :, : k_cache_nope_h]
    k_cache_rope = k_cache[:, :, k_cache_nope_h:]

    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    k_cache_nope_path = Path(output, 'k_cache_nope.bin')
    k_cache_rope_path = Path(output, 'k_cache_rope.bin')
    v_cache_path = Path(output, 'v_cache.bin')
    block_table_path = Path(output, 'block_table.bin')
    actual_seq_len_path = Path(output, 'actual_seq_len.bin')
    block_size_path = Path(output, 'block_size.bin')
    attent_out_path = Path(output, 'atten_out.bin')

    # dump golden file
    dump_file(q_nope, q_nope_path, "bf16")
    dump_file(q_rope, q_rope_path, "bf16")
    dump_file(k_cache_nope, k_cache_nope_path, "bf16")
    dump_file(k_cache_rope, k_cache_rope_path, "bf16")
    dump_file(v_cache, v_cache_path, "bf16")
    dump_file(block_table, block_table_path, "int32")
    dump_file(actual_seq_len, actual_seq_len_path, "int32")
    dump_file(block_size, block_size_path, "int64")
    dump_file(attent_out, attent_out_path, "fp32")

    # a_q_no,a_q_ro,a_kv_no,a_kv_ro
    a_q_no_path = Path(output, 'a_q_no.bin')
    a_q_ro_path = Path(output, 'a_q_ro.bin')
    a_kv_no_path = Path(output, 'a_kv_no.bin')
    a_kv_ro_path = Path(output, 'a_kv_ro.bin')

    dump_file(a_q_no, a_q_no_path, "bf16")
    dump_file(a_q_ro, a_q_ro_path, "bf16")
    dump_file(a_kv_no, a_kv_no_path, "bf16")
    dump_file(a_kv_ro, a_kv_ro_path, "bf16")
    return attent_out


def quantize_torch(input_fp32):
    abs_res = torch.abs(input_fp32)
    max_value, _ = torch.max(abs_res, dim=-1, keepdim=True)
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = torch.round(out_fp32).to(torch.int32)
    out_int8 = torch.clamp(out_int32, -128, 127).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_int8, scale_dequant


def gen_quant_mm_torch(a, w, scale_w):
    a_fp32 = a.to(torch.float32)
    quantized_a, scale_dequant_a = quantize_torch(a_fp32)

    a_int32 = quantized_a.to(torch.int32)
    w_int32 = w.to(torch.int32)
    res_int32 = torch.matmul(a_int32, w_int32)
    res = res_int32.to(torch.float32)
    res = res * scale_dequant_a
    res = res * scale_w
    return res.to(a.dtype)


def tensor_bf16_tofile(t: torch.Tensor, output: Path):
    input_file_bin = open(str(output), "wb")
    for each in t:
        if t.dtype == torch.bfloat16:
            input_file_bin.write(each.view(torch.int16).numpy().tobytes())
        elif t.dtype == torch.float32:
            input_file_bin.write(each.view(torch.int32).numpy().tobytes())
        elif t.dtype == torch.int32:
            input_file_bin.write(each.numpy().tobytes())
        elif t.dtype == torch.int8:
            input_file_bin.write(each.numpy().tobytes())
        else:
            raise ValueError(f"Unsupported dtype: {t.dtype}")
    input_file_bin.close()


def gen_data_func_bf16_quant_onlymm5(shape_size, dtype, case_name: str, output: Path) -> bool:
    input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, attention_out = shape_size
    params_path = Path(output, 'params.bin')
    input_path = Path(output, 'input.bin')
    t1_path = Path(output, 't1.bin')
    r1_path = Path(output, 'r1.bin')
    t2_path = Path(output, 't2.bin')
    w_uv_path = Path(output, 'w_uv.bin')
    w_uv_scale_w_path = Path(output, 'w_uv_scale_w.bin')
    cast0_out_path = Path(output, 'cast0_out.bin')
    abs_out_path = Path(output, 'abs_out.bin')
    mul0_out_path = Path(output, 'mul0_out.bin')
    rms_out_path = Path(output, 'rms_out.bin')
    quant1_int8_path = Path(output, 'quant0_int8.bin')
    quant1_fp32_path = Path(output, 'quant0_fp32.bin')
    bmm4_int32_path = Path(output, 'bmm4_int32.bin')
    bmm4_path = Path(output, 'bmm4.bin')
    t3_path = Path(output, 't3.bin')
    r2_path = Path(output, 'r2.bin')
    w_o_path = Path(output, 'w_o.bin')
    w_o_scale_w_path = Path(output, 'w_o_scale_w.bin')
    bmm5_path = Path(output, 'bmm5.bin')
    attn_output_path = Path(output, 'attn_output.bin')
    complete = (params_path.exists() and input_path.exists() and t1_path.exists() and r1_path.exists() and
                t2_path.exists() and w_uv_path.exists() and bmm4_path.exists() and t3_path.exists() and
                r2_path.exists() and w_o_path.exists() and attn_output_path.exists() and bmm5_path.exists() and
                w_uv_scale_w_path.exists() and w_o_scale_w_path.exists())
    if False:
        logging.debug("Case(%s), Golden complete.", case_name)
    else:
        dtype_num = 0
        if dtype == torch.float32:
            dtype_num = 0
        elif dtype == torch.float16:
            dtype_num = 1
        elif dtype == torch.bfloat16:
            dtype_num = 2
        params = torch.tensor([input_b, input_s, input_n, input_h, kv_lora_rank, v_head_dim, dtype_num],
                              dtype=torch.int64)

        input_t = torch.randn([input_b, input_n, input_s, kv_lora_rank], dtype=dtype)
        input_t = torch.from_numpy(attention_out.reshape(input_b, input_n, input_s, kv_lora_rank)).to(dtype)
        w_uv = torch.randn([input_n, kv_lora_rank, v_head_dim], dtype=dtype)
        w_uv_scale_w = torch.randn([input_n, 1, v_head_dim], dtype=torch.float32) * 0.001

        w_o = torch.randint(size=(input_n * v_head_dim, input_h), low=-128, high=128, dtype=torch.int8)
        w_o_scale_w = torch.randn([input_h], dtype=torch.float32) * 0.001

        params.numpy().tofile(params_path)
        tensor_bf16_tofile(input_t, input_path)
        tensor_bf16_tofile(w_uv, w_uv_path)
        tensor_bf16_tofile(w_uv_scale_w, w_uv_scale_w_path)
        tensor_bf16_tofile(w_o, w_o_path)
        tensor_bf16_tofile(w_o_scale_w, w_o_scale_w_path)

        t1 = input_t.transpose(1, 2)
        tensor_bf16_tofile(t1, t1_path)
        r1 = t1.reshape(input_b * input_s, input_n, kv_lora_rank)
        tensor_bf16_tofile(r1, r1_path)
        t2 = r1.transpose(0, 1)
        tensor_bf16_tofile(t2, t2_path)
        calc_input = t2
        bmm4 = torch.matmul(calc_input.to(torch.float32), w_uv.to(torch.float32))
        if dtype != torch.float32:
            bmm4 = bmm4.to(dtype)

        # 原bmm4 = gen_quant_mm_torch(calc_input, w_uv, w_uv_scale_w)
        tensor_bf16_tofile(bmm4, bmm4_path)

        t3 = bmm4.transpose(0, 1)
        tensor_bf16_tofile(t3, t3_path)
        r2 = t3.reshape(input_b * input_s, input_n * v_head_dim)
        tensor_bf16_tofile(r2, r2_path)
        bmm5_i = r2

        bmm5 = gen_quant_mm_torch(bmm5_i, w_o, w_o_scale_w)
        tensor_bf16_tofile(bmm5, bmm5_path)

        bmm5 = bmm5.reshape(input_b, input_s, input_h)
        tensor_bf16_tofile(bmm5, attn_output_path)
    return True


def gen_attention_test_net(dtypes, params, epsilon, output_dir: Path, is_quant=False):
    b, s2, h, n, q_lora_rank = params
    kv_lora_rank = 512
    v_head_dim = 128
    params = {
        "b": b,
        "s": 1,
        "s2": s2,
        "h": h,
        "num_heads": n,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": kv_lora_rank,
        "v_head_dim": v_head_dim,
    }
    q_out, q_embed, kv_cache_out, kr_cache_out = gen_mla_prolog_data(params, dtypes, epsilon, output_dir, is_quant)
    a_q_no = q_out.transpose(0, 2, 1, 3)
    a_q_ro = q_embed.transpose(0, 2, 1, 3)
    a_k_no = kv_cache_out
    a_k_ro = kr_cache_out
    q_res = np.concatenate((q_out, q_embed), axis=-1).transpose(0, 2, 1, 3)
    key_states = np.concatenate((kv_cache_out, kr_cache_out), axis=-1)
    attention_out = ifa_pa_func((b, n, s2, 256, a_q_no, a_q_ro, a_k_no, a_k_ro), output_dir, q_res, key_states)

    dtype = torch.bfloat16
    # 原torch.float32 torch.float16   torch.bfloat16
    gen_data_func_bf16_quant_onlymm5((b, 1, n, h, kv_lora_rank, v_head_dim, attention_out),
                                     dtype, "", output_dir)


def gen_mla_prolog_data_nsa(case_name: str, output: Path):
    # fp16, quant, weight nz, "PA_NZ"
    if case_name == "MlaPrologSTest.b16_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (16, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b16_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (16, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b32_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (32, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b32_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (32, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b64_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (64, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b64_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (64, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b24_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (24, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b24_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (24, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b48_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (48, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b48_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (48, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b96_s1_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (96, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b96_s2_pa_nz_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (96, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    # bf16, quant, weight nz, "PA_NZ"
    if case_name == "MlaPrologSTest.b16_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (16, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b16_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (16, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b32_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (32, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b32_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (32, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b64_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (64, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b64_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (64, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b24_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (24, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b24_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (24, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b48_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (48, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b48_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (48, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b96_s1_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (96, 128, 1, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b96_s2_pa_nz_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (96, 128, 2, 8192), 1e-5, output, True, True, True, 128, "PA_NZ")
    # fp16, quant, weight nd, "PA_BSND"
    elif case_name == "MlaPrologSTest.b32_s1_pa_nd_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (32, 128, 1, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b32_s2_pa_nd_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (32, 128, 2, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b48_s1_pa_nd_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (48, 128, 1, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b48_s2_pa_nd_fp16_quant":
        gen_prolog_data((np.float16, np.float16), (48, 128, 2, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    # bf16, quant, weight nd, "PA_BSND"
    elif case_name == "MlaPrologSTest.b64_s1_pa_nd_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (64, 128, 1, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b64_s2_pa_nd_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (64, 128, 2, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b96_s1_pa_nd_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (96, 128, 1, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b96_s2_pa_nd_bf16_quant":
        gen_prolog_data((bfloat16, bfloat16), (96, 128, 2, 8192), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b16_s1_pa_nd_bf16_allquant":
        gen_nsa_prolog_test((bfloat16, bfloat16), (16, 128, 1, 8192), 1e-5, output, True, True, True, True, 128)
    # fp16, no quant, weight nz, "PA_NZ"
    elif case_name == "MlaPrologSTest.b32_s1_pa_nz_fp16":
        gen_prolog_data((np.float16, np.float16), (32, 128, 1, 8192), 1e-5, output, False, True, False, 128, "PA_NZ")
    elif case_name == "MlaPrologSTest.b32_s2_pa_nz_fp16":
        gen_prolog_data((np.float16, np.float16), (32, 128, 2, 8192), 1e-5, output, False, True, False, 128, "PA_NZ")
    # fp16, no quant, weight nd, "PA_BSND"
    elif case_name == "MlaPrologSTest.b32_s1_pa_nd_fp16":
        gen_prolog_data((np.float16, np.float16), (32, 128, 1, 8192), 1e-5, output, False, False, False, 128, "PA_BSND")
    elif case_name == "MlaPrologSTest.b32_s2_pa_nd_fp16":
        gen_prolog_data((np.float16, np.float16), (32, 128, 2, 8192), 1e-5, output, False, False, False, 128, "PA_BSND")
    # small shape
    elif case_name == "MlaPrologSTest.b16_s2_pa_nd_fp16_small":
        gen_prolog_data_small((np.float16, np.float16), (16, 128, 2, 256), 1e-5, output, False, False, False, 128,
                              "PA_BSND")


# NSA MlaProlog
nsa_case_names = [
    # fp16, quant, weight nz, "PA_NZ"
    "MlaPrologSTest.b16_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b16_s2_pa_nz_fp16_quant",
    "MlaPrologSTest.b32_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b32_s2_pa_nz_fp16_quant",
    "MlaPrologSTest.b64_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b64_s2_pa_nz_fp16_quant",
    "MlaPrologSTest.b24_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b24_s2_pa_nz_fp16_quant",
    "MlaPrologSTest.b48_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b48_s2_pa_nz_fp16_quant",
    "MlaPrologSTest.b96_s1_pa_nz_fp16_quant",
    "MlaPrologSTest.b96_s2_pa_nz_fp16_quant",
    # bf16, quant, weight nz, "PA_NZ"
    "MlaPrologSTest.b16_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b16_s2_pa_nz_bf16_quant",
    "MlaPrologSTest.b32_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b32_s2_pa_nz_bf16_quant",
    "MlaPrologSTest.b64_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b64_s2_pa_nz_bf16_quant",
    "MlaPrologSTest.b24_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b24_s2_pa_nz_bf16_quant",
    "MlaPrologSTest.b48_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b48_s2_pa_nz_bf16_quant",
    "MlaPrologSTest.b96_s1_pa_nz_bf16_quant",
    "MlaPrologSTest.b96_s2_pa_nz_bf16_quant",
    # fp16, quant, weight nd, "PA_BSND"
    "MlaPrologSTest.b32_s1_pa_nd_fp16_quant",
    "MlaPrologSTest.b32_s2_pa_nd_fp16_quant",
    "MlaPrologSTest.b48_s1_pa_nd_fp16_quant",
    "MlaPrologSTest.b48_s2_pa_nd_fp16_quant",
    # bf16, quant, weight nd, "PA_BSND"
    "MlaPrologSTest.b64_s1_pa_nd_bf16_quant",
    "MlaPrologSTest.b64_s2_pa_nd_bf16_quant",
    "MlaPrologSTest.b96_s1_pa_nd_bf16_quant",
    "MlaPrologSTest.b96_s2_pa_nd_bf16_quant",
    "MlaPrologSTest.b16_s1_pa_nd_bf16_allquant",
    # fp16, no quant, weight nz, "PA_NZ"
    "MlaPrologSTest.b32_s1_pa_nz_fp16",
    "MlaPrologSTest.b32_s2_pa_nz_fp16",
    # fp16, no quant, weight nd, "PA_BSND"
    "MlaPrologSTest.b32_s1_pa_nd_fp16",
    "MlaPrologSTest.b32_s2_pa_nd_fp16",
    # small shape
    "MlaPrologSTest.b16_s2_pa_nd_fp16_small"
]


# @GoldenRegister.reg_golden_func(
#     case_names = [
#         # NSA MlaProlog
#         *nsa_case_names,
#         # MLA_prolog v2
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_2_1_256_256_512",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_32_1_256_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_32_1_256_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_128_1_4096_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_128_1_4096_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_128_1_256_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_4096_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_128_1_4096_7168_1536",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_128_1_256_7168_1536",
#         "MlaPrologV2OnBoardTest.test_mla_bf16_low_quant_smooth",
#         "MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_4_32_1_256_7168_1536_quant",
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_32_1_256_7168_1536_quant",
#         "MlaPrologV2OnBoardTest.test_mla_fp16_high_quant_smooth",
#         "MlaPrologV2OnBoardCostTest.test_mla_bf16_low_quant_smooth",
#         "MlaPrologV2OnBoardCostTest.test_mla_bf16_high_quant_smooth",
#         "DynamicMLATest.dynamic_mla_low",
#         "DyMla.low_bf",
#         "DyMla.low_PA_BSND",
#         "DyMla.low_PA_NZ",
#         "DyMla.high_PA_NZ",
#         "DyMla.low",
#         "DyMla.high",
#         "MlaPrologV2OnBoardTest.attentionV2_bf16_4_1024_1024_32_256",
#         "MlaPrologV2OnBoardTest.attentionV2_bf16_low",
#         # PA_BSND
#         "MlaPrologV2OnBoardTest.test_mla_fp16_low_quant_smooth_pa_bsnd",
#         "MlaPrologV2OnBoardTest.test_mla_fp16_high_quant_smooth_pa_bsnd",
#         "MlaPrologV2OnBoardTest.test_mla_fp16_low_quant_smooth_nz_pa_bsnd",
#         "MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth_nz_pa_bsnd",
#         "MlaPrologV2OnBoardTest.test_mla_bf16_high48_quant_smooth_nz_pa_bsnd",
#         # CodeGen by Json
#         "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536_codegen_by_json",
#         "DyMla.AB_low",
#     ]
# )


def gen_mla_prolog_date_v2(case_name: str, output: Path) -> bool:
    # b_n_s_s2_h_q_lora_rank
    # NSA MlaProlog
    if case_name in nsa_case_names:
        gen_mla_prolog_data_nsa(case_name, output)
        return True

    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_2_1_256_256_512":
        gen_mla_prolog_test1((np.float16, np.float16), (32, 2, 256), 1e-5, output)  # (b, n, s2)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_32_1_256_7168_1536":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 32, 256), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_32_1_256_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 32, 256), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_128_1_4096_7168_1536":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_128_1_4096_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 128, 4096), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_32_128_1_256_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 128, 256), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_4096_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 32, 4096), 1e-5, output)
    elif (case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536"
          or case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_32_1_256_7168_1536_codegen_by_json"):
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 32, 256), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_128_1_4096_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 128, 4096), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_bfloat16_4_128_1_256_7168_1536":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 128, 256), 1e-5, output)
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_bf16_low_quant_smooth":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 32, 256), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 128, 4096), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_4_32_1_256_7168_1536_quant":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardTest.test_MlaPrologV2_float16_32_32_1_256_7168_1536_quant":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 32, 256), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_fp16_high_quant_smooth":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output, True, True, True)

    # PA_BSND
    # dtypes, bns2, epsilon, output_dir, is_quant, is_nz, is_smooth, block_size, cache_mode
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_fp16_low_quant_smooth_pa_bsnd":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_fp16_high_quant_smooth_pa_bsnd":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output, True, False, True, 128, "PA_BSND")
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_fp16_low_quant_smooth_nz_pa_bsnd":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True, True, True, 128, "PA_BSND")
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_bf16_high_quant_smooth_nz_pa_bsnd":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 128, 4096), 1e-5, output, True, True, True, 128, "PA_BSND")
    elif case_name == "MlaPrologV2OnBoardTest.test_mla_bf16_high48_quant_smooth_nz_pa_bsnd":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (48, 128, 4096), 1e-5, output, True, True, True, 128, "PA_BSND")

    # cost case
    elif case_name == "MlaPrologV2OnBoardCostTest.test_mla_bf16_low_quant_smooth":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 32, 256), 1e-5, output, True, True, True)
    elif case_name == "MlaPrologV2OnBoardTest.attentionV2_bf16_low":
        gen_attention_test_net((bfloat16, bfloat16), (4, 256, 7168, 32, 1536), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardTest.attentionV2_bf16_4_1024_1024_32_256":
        gen_attention_test_net((bfloat16, bfloat16), (4, 256, 256, 32, 256), 1e-5, output, True)
    elif case_name == "MlaPrologV2OnBoardCostTest.test_mla_bf16_high_quant_smooth":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (32, 128, 4096), 1e-5, output, True, True, True)
    elif case_name == "DyMla.dynamic_mla_low":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, False, False, False)
    elif case_name == "DyMla.dynamic_mla_high":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output, False, False, False)
    elif case_name == "DyMla.low":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True, True, True)
    elif case_name == "DyMla.low_PA_BSND":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True, True, True, 128, "PA_BSND")
    elif case_name == "DyMla.low_PA_NZ":
        gen_mla_prolog_test_net((np.float16, np.float16), (4, 32, 256), 1e-5, output, True, True, True, 128, "PA_NZ")
    elif case_name == "DyMla.low_bf":
        gen_mla_prolog_test_net((bfloat16, bfloat16), (4, 32, 256), 1e-5, output, True, True, True)
    elif case_name == "DyMla.high":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output, True, True, True)
    elif case_name == "DyMla.high_PA_NZ":
        gen_mla_prolog_test_net((np.float16, np.float16), (32, 128, 4096), 1e-5, output, True, True, True, 128, "PA_NZ")
    # elif case_name == "DyMla.TestD_low":
    #     gen_mla_prolog_subgraphD_test_data((np.float16, np.float16), (4, 32, 256), 1e-5, output)
    # elif case_name == "DyMla.TestD_mini_low":
    #     gen_mla_prolog_subgraphD_test_data((np.float16, np.float16), (4, 32, 256), 1e-5, output)
    # elif case_name == "DyMla.TestD_high":
    #     gen_mla_prolog_subgraphD_test_data((np.float16, np.float16), (32, 128, 4096), 1e-5, output)
    # elif case_name == "DyMla.TestC_low":
    #     gen_mla_prolog_subgraphC_test_data((np.float16, np.float16), (4, 32, 256), output)
    # elif case_name == "DyMla.TestC_high":
    #     gen_mla_prolog_subgraphC_test_data((np.float16, np.float16), (32, 128, 4096), output)
    # elif case_name == "DyMla.TestCD_low":
    #     gen_mla_prolog_subgraphCD_test_data((np.float16, np.float16), (4, 32, 256), 1e-5, output)
    # elif case_name == "DyMla.TestCD_high":
    #     gen_mla_prolog_subgraphCD_test_data((np.float16, np.float16), (32, 128, 4096), 1e-5, output)
    # elif case_name == "DyMla.TestKvAndSubgraphCD_low":
    #     gen_mla_prolog_kvAndSubgraphCD_test_data((np.float16, np.float16), (4, 32, 256), 1e-5, output)
    # elif case_name == "DyMla.TestKvAndSubgraphCD_high":
    #     gen_mla_prolog_kvAndSubgraphCD_test_data((np.float16, np.float16), (32, 128, 4096), 1e-5, output)

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
        "DyMla.low_PA_NZ",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_mla_prolog_date_v2(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    # TestTranspose_DEBUG_ABC_BAC(1,1,128)
    # TestTranspose_DEBUG_BNDS_BNSD(2,1,1,128)
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    exit(0 if main() else 1)
