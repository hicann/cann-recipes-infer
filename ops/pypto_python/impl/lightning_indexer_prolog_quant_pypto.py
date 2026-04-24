#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Lightning Indexer Prolog Quantization Module

This module implements Lightning Indexer Prolog quantization computation
for deepseek_v4 model. It handles:
- Query computation with dynamic quantization
- Weight computation for indexer attention

Main Functions:
    - lightning_indexer_prolog_compute: Main computation function

Example:
    See test_lightning_indexer_prolog_quant.py for usage examples.
"""

import pypto
import math
import torch
from dataclasses import dataclass


from common import inverse_rope_3d, quant_tensor


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "quant_lightning_indexer_prolog(Tensor qr, Tensor idx_wq_b, Tensor x, Tensor weights_proj, Tensor cos, Tensor sin, Tensor hadamard, Tensor qr_scale, Tensor idx_wq_b_scale) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(pyptolib, "quant_lightning_indexer_prolog", "Meta")
def quant_lightning_indexer_prolog(
    qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
):
    q = torch.empty(
        [qr.size(0), weights_proj.size(1), hadamard.size(0)],
        dtype=qr.dtype,
        device=qr.device,
    )
    weights = torch.empty(
        [qr.size(0), hadamard.size(0)],
        dtype=weights_proj.dtype,
        device=weights_proj.device,
    )
    q_scale = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )
    return q, weights, q_scale


@torch.library.impl(pyptolib, "quant_lightning_indexer_prolog", "NPU")
def quant_lightning_indexer_prolog(
    qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
):
    return npu_quant_lightning_indexer_prolog(
        qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
    )


@dataclass
class IndexerPrologQuantConfig:
    unroll_list: list


def quant_lightning_indexer_prolog_compute(
    qr: pypto.Tensor,
    idx_wq_b: pypto.Tensor,
    x: pypto.Tensor,
    weights_proj: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    hadamard: pypto.Tensor,
    qr_scale: pypto.Tensor,
    idx_wq_b_scale: pypto.Tensor,
    q: pypto.Tensor,
    weights: pypto.Tensor,
    q_scale: pypto.Tensor,
    tile_config: IndexerPrologQuantConfig,
):
    """Compute Lightning Indexer Prolog with quantization.

    Main computation function for Lightning Indexer Prolog Quantization.
    This function processes input tokens to generate quantized-q and weights
    for the indexer attention mechanism. The computation includes:
    Args:
        group       name           dtype     shape                               format
        INPUT 0	    qr	           DT_INT8	 (t, q_lora_rank)                    ND
        INPUT 1	    idx_wq_b	   DT_INT8	 (q_lora_rank, idx_nq * head_dim)	 ND
        INPUT 2	    x	           DT_BF16	 (t, h)	                             ND
        INPUT 3	    weights_proj   DT_BF16	 (h, idx_nq)	                     ND
        INPUT 4	    cos	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 5	    sin	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 6     hadamard       DT_BF16   (head_dim, head_dim)                ND
        INPUT 7     qr_scale       DT_FP32   (t, 1)                              ND
        INPUT 8     idx_wq_b_scale DT_FP32   (idx_nq * head_dim, 1)              ND
        OUTPUT 0	q              DT_BF16	 (t, idx_nq * head_dim)	             ND
        OUTPUT 1    weights        DT_FP16   (t, idx_nq)                         ND
        OUTPUT 2    q_scale        DT_FP16   (t, idx_nq)                         ND
        CONFIGS     tile_config    /          /                                  /
    1. Query Path:
       - Dequantize qr (INT8) to FP32
       - Apply linear transformation with w_qb
       - Apply RoPE (Rotary Position Embedding)
       - Apply Hadamard transformation
       - Quantize to INT8 with per-token-head scale

    2. Weights Path:
       - Linear transformation with w_proj
       - Normalize by sqrt(idx_nq * head_dim)
       - Convert to FP16
    Note:
        - The function processes tokens in tiles using loop_unroll for optimization
        - All outputs are written in-place using pypto.assemble or scatter_update
        - The computation uses dynamic tiling based on configs.unroll_list
    """
    x_dtype = x.dtype
    # dynamic axis
    t = qr.shape[0]
    # static axes
    q_lora_rank = qr.shape[1]
    h = x.shape[1]
    idx_nq = weights_proj.shape[1]
    head_dim = hadamard.shape[0]
    rope_dim = cos.shape[1]

    # Reshape inplace will not generate data move
    w_qb_scale = pypto.reshape(idx_wq_b_scale, [1, idx_nq * head_dim], inplace=True)
    hadamard_q = pypto.reshape(hadamard, [1, head_dim, head_dim], inplace=True)

    unroll_list = tile_config.unroll_list
    for t_idx, unrollLength in pypto.loop_unroll(
        0,
        t,
        1,
        name="IndexerPrologLoop",
        idx_name="t_idx",
        unroll_list=unroll_list,
    ):
        # use for perf optimization
        pypto.experimental.set_operation_config(combine_axis=True)
        t_tile = unrollLength
        qr_in = pypto.view(qr, [t_tile, q_lora_rank], [t_idx, 0])
        qs_in = pypto.view(qr_scale, [t_tile, 1], [t_idx, 0])
        pypto.set_semantic_label("Query-Linear")
        # (t_tile, q_lora_rank) @ (q_lora_rank, idx_nq * head_dim) --> (t_tile, idx_nq * head_dim)
        pypto.set_cube_tile_shapes([128, 128], [256, 1024], [256, 256], enable_multi_data_load=True)
        q_s32 = pypto.matmul(qr_in, idx_wq_b, pypto.DT_INT32)

        pypto.set_semantic_label("Query-Dequant")
        pypto.set_vec_tile_shapes(1, idx_nq * head_dim)
        # (t_tile, idx_nq * head_dim), fp32
        q_f32 = pypto.cast(q_s32, pypto.DT_FP32)
        # (t_tile, idx_nq * head_dim), fp32, last dim brc
        q_f32 = q_f32 * qs_in
        # (t_tile, idx_nq * head_dim), fp32, first dim brc
        q_f32 = q_f32 * w_qb_scale
        q_cast = pypto.cast(q_f32, x_dtype)
        q_re = pypto.reshape(q_cast, [t_tile, idx_nq, head_dim])

        # UB view
        q_nope = pypto.view(q_re, [t_tile, idx_nq, head_dim - rope_dim], [0, 0, 0])
        q_rope = pypto.view(
            q_re, [t_tile, idx_nq, rope_dim], [0, 0, head_dim - rope_dim]
        )

        rope_cos = pypto.view(cos, [t_tile, rope_dim], [t_idx, 0])
        rope_sin = pypto.view(sin, [t_tile, rope_dim], [t_idx, 0])

        q_roped = inverse_rope_3d(q_rope, rope_cos, rope_sin)

        pypto.set_vec_tile_shapes(1, idx_nq, head_dim)
        q_assemble = pypto.tensor([t_tile, idx_nq, head_dim], x_dtype, "q_assemble")
        pypto.assemble(pypto.clone(q_nope), [0, 0, 0], q_assemble)
        pypto.assemble(q_roped, [0, 0, head_dim - rope_dim], q_assemble)

        pypto.set_semantic_label("Hadamard-Compute")
        # (t_tile, idx_nq, head_dim) @ (1, head_dim, head_dim) -> (t_tile, idx_nq, head_dim)
        pypto.set_cube_tile_shapes([idx_nq, idx_nq], [head_dim, head_dim], [head_dim, head_dim])
        q_hadamard = pypto.matmul(q_assemble, hadamard_q, x_dtype)  # (t_tile, idx_nq, head_dim)
        pypto.set_vec_tile_shapes(1, idx_nq, head_dim)
        # (t_tile, idx_nq, head_dim), (t_tile, idx_nq, 1)
        q_res, q_scale_res = quant_tensor(q_hadamard)
        q_scale_out = pypto.reshape(q_scale_res, [t_tile, idx_nq])
        pypto.set_vec_tile_shapes(t_tile, idx_nq)
        q_scale_cast = pypto.cast(q_scale_out, pypto.DT_FP16)

        pypto.assemble(q_res, [t_idx, 0, 0], q)
        pypto.assemble(q_scale_cast, [t_idx, 0], q_scale)

        pypto.set_semantic_label("Weight-Compute")
        x_in = pypto.view(x, [t_tile, h], [t_idx, 0])
        # (t_tile, h) @ (h, idx_nq) --> (t_tile, idx_nq)
        pypto.set_cube_tile_shapes([32, 64], [h // 4, h], [idx_nq // 4, idx_nq // 4], enable_multi_data_load=True)
        pypto.set_vec_tile_shapes(t_tile, idx_nq)
        weights_fp32 = pypto.cast(pypto.matmul(x_in, weights_proj, x_dtype), pypto.DT_FP32)
        weights_mul = pypto.mul(weights_fp32, 1.0 / (math.sqrt(idx_nq) * math.sqrt(head_dim)))
        weights_fp16 = pypto.cast(weights_mul, pypto.DT_FP16)
        pypto.assemble(weights_fp16, [t_idx, 0], weights)


@pypto.jit(
    pass_options={
        "cube_nbuffer_mode": 2,
        "vec_nbuffer_mode": 2,
        "cube_l1_reuse_setting": {-1: 2, 1: 0},
        "vec_nbuffer_setting": {1: 2},
    },
    runtime_options={
        "stitch_function_inner_memory": 128,
        "stitch_function_outcast_memory": 128,
        "stitch_cfgcache_size": 2500000,
        "device_sched_mode": 1,
    },
)
def quant_lightning_indexer_prolog_kernel(
    qr: pypto.Tensor,
    idx_wq_b: pypto.Tensor,
    x: pypto.Tensor,
    weights_proj: pypto.Tensor,
    cos: pypto.Tensor,
    sin: pypto.Tensor,
    hadamard: pypto.Tensor,
    qr_scale: pypto.Tensor,
    idx_wq_b_scale: pypto.Tensor,
    q: pypto.Tensor,
    weights: pypto.Tensor,
    q_scale: pypto.Tensor,
    tile_config: IndexerPrologQuantConfig,
):
    """JIT-compiled wrapper for Lightning Indexer Prolog Quantization computation.

    This is the main entry point for the Lightning Indexer Prolog Quantization operator.
    It sets up optimization passes and runtime options before calling the core
    computation function in JIT decorator.

    Args:
        group       name           dtype     shape                               format
        INPUT 0	    qr	           DT_INT8	 (t, q_lora_rank)                    ND
        INPUT 1	    idx_wq_b	   DT_INT8	 (q_lora_rank, idx_nq * head_dim)	 ND
        INPUT 2	    x	           DT_BF16	 (t, h)	                             ND
        INPUT 3	    weights_proj   DT_BF16	 (h, idx_nq)	                     ND
        INPUT 4	    cos	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 5	    sin	           DT_BF16	 (t, rope_dim)                       ND
        INPUT 6     hadamard       DT_BF16   (head_dim, head_dim)                ND
        INPUT 7     qr_scale       DT_FP32   (t, 1)                              ND
        INPUT 8     idx_wq_b_scale DT_FP32   (idx_nq * head_dim, 1)              ND
        OUTPUT 0	q              DT_BF16	 (t, idx_nq * head_dim)	             ND
        OUTPUT 1    weights        DT_FP16   (t, idx_nq)                         ND
        OUTPUT 2    q_scale        DT_FP16   (t, idx_nq)                         ND
        CONFIGS     tile_config    /          /                                  /
    Note:
        This function is decorated with @pypto.jit for JIT compilation.
        It configures pass options for memory optimization and calls the core
        computation function.
    """

    quant_lightning_indexer_prolog_compute(
        qr,
        idx_wq_b,
        x,
        weights_proj,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
        q,
        weights,
        q_scale,
        tile_config,
    )
