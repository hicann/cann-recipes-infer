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
"""
from dataclasses import dataclass
import math
from typing import List, Tuple
import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor
import torch_npu
import pypto

pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("mla_prolog(Tensor token_x, Tensor wq_a, Tensor wq_b, Tensor wkv, Tensor rope_cos, Tensor rope_sin, Tensor gamma_cq, Tensor gamma_ckv) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(pyptolib, "mla_prolog", "Meta")
def mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    q_out_shape = torch.empty([token_x.size(0), wq_b.size(1) // gamma_ckv.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=token_x.device)
    kv_out_shape = torch.empty([token_x.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=token_x.device)
    qr_out_shape = torch.empty([token_x.size(0), gamma_cq.size(0)], dtype=token_x.dtype, device=token_x.device)

    return q_out_shape, kv_out_shape, qr_out_shape

@torch.library.impl(pyptolib, "mla_prolog", "NPU")
def mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    return mla_prolog_v4_in(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv)

def mla_prolog_pypto(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    return torch.ops.pypto.mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv)

"""
MLA Prolog Quantization Module

This module implements MLA (Multi-head Latent Attention) Prolog quantization
for deepseek_v4 model. It converts hidden states to query, key, and value
projections with support for quantization and RoPE (Rotary Position Embedding).

Main Functions:
    - mla_prolog_quant_compute: Core MLA prolog computation with quantization
    - pre_compute_2d: Pre-computation for query and key-value projections
    - rms_norm: RMS normalization implementation
    - quant: Quantization function with symmetry and smooth factor support
    - dequant: Dequantization function
    - rope_v2: 2D RoPE implementation
    - rope_3d_v2: 3D RoPE implementation
    - k_nope_quant: Key quantization function

Example:
    See testdsv32_mla_prolog_quant.py for usage examples.
"""

SHAPE_DIM_2 = 2
SHAPE_DIM_3 = 3

NUM_0 = 0
NUM_1 = 1
NUM_2 = 2
NUM_3 = 3
NUM_4096 = 4096
NUM_512 = 512

TILE_CUBE_DIM = 6
Q_PARAM_DIM = 2
NZ_DIM = 4
COS_SIN_DIM = 2
L0M_INDEX = 0
L1M_INDEX = 1
L0K_INDEX = 2
L1K_INDEX = 3
L0N_INDEX = 4
L1N_INDEX = 5
SCATTER_DIM = -2
NZ_FIRST_DIM = 16
NZ_B8_C0 = 32
NZ_B16_C0 = 16

VEC_TILE_256 = 256
VEC_TILE_128 = 128
VEC_TILE_64 = 64
VEC_TILE_8 = 8
VEC_TILE_4 = 4
VEC_TILE_32 = 32


@dataclass
class MlaPrologV4Output:
    x: torch.tensor  # BF16, (t, h)
    wq_a: torch.tensor  # BF16, (h, q_lora_rank)
    wq_b: torch.tensor # BF16, (q_lora_rank, n_q*head_dim)
    wkv: torch.tensor # BF16, (h, head_dim)
    rmsnorm_gamma_cq: torch.tensor # BF16, (q_lora_rank, )
    rmsnorm_gamma_ckv: torch.tensor  # BF16, (head_dim, )
    cos: torch.tensor # BF16, (t, rope_dim)
    sin: torch.tensor # BF16, (t, rope_dim)


@dataclass
class MlaPrologV4Output:
    q: torch.tensor  # BF16, (t, n_q, head_dim)
    kv: torch.tensor  # BF16, (t, head_dim)
    qr: torch.tensor  # BF16, (t, q_lora_rank)


@dataclass
class MlaPrologV4Attrs:
    eps: float
    layout_query: str
    layout_key: str


@dataclass
class MlaPrologV4Configs:
    unroll_list: List[int]
    cube_l1_reuse_setting: dict[int, int]
    mg_copyin_upper_bound: int
    pg_upper_bound: int
    block_size: int
    t_sub_tile: int
    chunk_size: int
    vec_nbuffer_mode: int


def check_input_output_shape_dtype(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv,
                                    output_q_data, output_kv_data, output_qr_data):
    assert token_x.size(1) == 4096 and token_x.dim() == 2, f"expected token_x dim num 2, token_x axis1 4096"
    assert wq_a.dim() == 2 and wq_a.size(0) == 4096 and wq_a.size(1) == 1024, \
            f"expected wq_a dim num 2 residual axis0 4096, wq_a axis1 1024"
    assert wq_b.dim() == 2 and wq_b.size(0) == 1024 and wq_b.size(1) == 32768, \
            f"expected wq_b dim num 2, wq_b axis0 1024, wq_b axis1 32768"
    assert wkv.dim() == 2 and wkv.size(0) == 4096 and wkv.size(1) == 512, \
            f"expected wkv dim num 2, wkv axis0 4096, wkv axis1 512"
    assert rope_cos.dim() == 2 and rope_cos.size(1) == 64, \
            f"expected rope_cos dim num 2, rope_cos axis1 64"
    assert rope_sin.dim() == 2 and rope_sin.size(1) == 64, \
            f"expected rope_sin dim num 2, rope_sin axis1 64"
    assert gamma_cq.dim() == 1 and gamma_cq.size(0) == 1024, \
            f"expected gamma_cq dim num 1, gamma_cq axis0 1024"
    assert gamma_ckv.dim() == 1 and gamma_ckv.size(0) == 512, \
            f"expected gamma_ckv dim num 1, gamma_ckv axis0 512"
    assert output_q_data.dim() == 3 and output_q_data.size(1) == 64 and output_q_data.size(2) == 512, \
            f"expected output_q_data dim num 3, output_q_data axis1 64, output_q_data axis2 512"
    assert output_kv_data.dim() == 2 and output_kv_data.size(1) == 512, \
            f"expected output_kv_data dim num 2, output_kv_data axis1 512"
    assert output_qr_data.dim() == 2 and output_qr_data.size(1) == 1024, \
            f"expected output_qr_data dim num 2, output_qr_data axis1 4096"

    assert token_x.dtype == torch.bfloat16, f"token_x.dtype is {token_x.dtype}, expected torch.bfloat16"
    assert wq_a.dtype == torch.bfloat16, f"wq_a.dtype is {wq_a.dtype}, expected torch.bfloat16"
    assert wq_b.dtype == torch.bfloat16, f"wq_b.dtype is {wq_b.dtype}, expected torch.bfloat16"
    assert wkv.dtype == torch.bfloat16, f"wkv.dtype is {wkv.dtype}, expected torch.bfloat16"
    assert rope_cos.dtype == torch.bfloat16, f"rope_cos.dtype is {rope_cos.dtype}, expected torch.bfloat16"
    assert rope_sin.dtype == torch.bfloat16, f"rope_sin.dtype is {rope_sin.dtype}, expected torch.bfloat16"
    assert gamma_cq.dtype == torch.bfloat16, f"gamma_cq.dtype is {gamma_cq.dtype}, expected torch.bfloat16"
    assert gamma_ckv.dtype == torch.bfloat16, f"gamma_ckv.dtype is {gamma_ckv.dtype}, expected torch.bfloat16"
    assert output_q_data.dtype == torch.bfloat16, f"output_q_data.dtype is {output_q_data.dtype}, \
                                                    expected torch.bfloat16"
    assert output_kv_data.dtype == torch.bfloat16, f"output_kv_data.dtype is {output_kv_data.dtype}, \
                                                    expected torch.bfloat16"
    assert output_qr_data.dtype == torch.bfloat16, f"output_qr_data.dtype is {output_qr_data.dtype}, \
                                                    expected torch.bfloat16"


def rms_norm(input_tensor: pypto.Tensor, epsilon: float) -> pypto.Tensor:
    """Compute RMS (Root Mean Square) normalization.

    Applies RMS normalization to the input tensor. RMS normalization is similar
    to LayerNorm but uses root mean square instead of standard deviation.

    Formula: output = gamma * input / sqrt(mean(input^2) + epsilon)

    Args:
        input_tensor: Input tensor to normalize
        epsilon: Small constant added to variance to avoid division by zero

    Returns:
        Normalized tensor with the same shape as input

    Note:
        The normalization is performed along the last dimension.
        Computation is done in FP32 for numerical stability.
    """
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    dim = len(input_tensor.shape)
    y = pypto.mul(input_fp32, input_fp32)
    y = pypto.mul(y, 1.0 / input_tensor.shape[dim - 1])
    y = pypto.sum(y, -1, keepdim=True)
    y = pypto.add(y, epsilon)
    y = pypto.sqrt(y)
    ones_vector = pypto.full(y.shape, 1.0, pypto.DT_FP32)
    y = pypto.div(ones_vector, y)
    y = pypto.mul(input_fp32, y)
    return y


def rotate_half(input_tensor: pypto.Tensor) -> pypto.Tensor:
    """Rotate half of the tensor dimensions for RoPE computation.

    Splits the last dimension in half and applies rotation transformation:
    [-x2, x1] where x1 is the first half and x2 is the second half.
    This is a key component of RoPE (Rotary Position Embedding).

    Args:
        input_tensor: Input tensor with last dimension divisible by 2

    Returns:
        Rotated tensor with same shape as input

    Raises:
        AssertionError: If input dimension is less than 1 or last dimension
                       is not divisible by 2
    """
    shape = input_tensor.shape
    shape_size = len(shape)

    new_shape = list(shape)
    new_shape[shape_size - 1] //= 2

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def rope_2d(
    x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply 2D Rotary Position Embedding (RoPE) version 2.

    Implements RoPE transformation for 2D tensors with optimized tiling.
    The function reshapes and transposes the input before applying rotation.

    Args:
        x: Input tensor of shape (seq_size, d_r)
        cos: Cosine values for RoPE, shape (seq_size, d_r)
        sin: Sine values for RoPE, shape (seq_size, d_r)
        tile_config: RopeTileShapeConfig object containing tiling parameters:
            - two_dim: Tile shape for 2D operations
            - three_dim: Tile shape for 3D reshape operations

    Returns:
        Tensor with RoPE applied, same shape as input x

    Note:
        The function performs reshape and transpose operations before applying
        rotation to optimize memory access patterns.
    """
    assert len(x.shape) == 2 and len(cos.shape) == 2 and len(sin.shape) == 2
    input_dtype = x.dtype
    pypto.set_vec_tile_shapes(1, 64, 128)
    y = pypto.clone(x)
    y_cast = pypto.cast(y, pypto.DT_FP32)
    x_view = pypto.reshape(x, [x.shape[0], x.shape[1]//2, 2])
    x_trans = pypto.transpose(x_view, 1, 2)
    x_re_second = pypto.reshape(x_trans, x.shape)
    x_t = rotate_half(x_re_second)
    x_new = pypto.reshape(x_t, [x.shape[0], 2, x.shape[1]//2])
    x_new_trans = pypto.transpose(x_new, 1, 2)
    x_new_r = pypto.reshape(x_new_trans, x.shape)
    x_new_cast = pypto.cast(x_new_r, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, 64)
    cast_cos = pypto.cast(cos, pypto.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DT_FP32)

    pypto.set_vec_tile_shapes(1, 64, 64)
    x_embed = y_cast * cast_cos + x_new_cast * cast_sin
    return pypto.cast(x_embed, input_dtype)


def rope_3d(x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor) -> pypto.Tensor:
    """Apply inverse 3D Rotary Position Embedding.
    """
    assert (len(x.shape) == SHAPE_DIM_3 and len(cos.shape) == SHAPE_DIM_2 and len(sin.shape) == SHAPE_DIM_2)

    pypto.set_vec_tile_shapes(1, 64)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)

    pypto.set_vec_tile_shapes(1, 64, 64)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [x.shape[0], 1, x.shape[2]])
    cast_sin = pypto.reshape(cast_sin, [x.shape[0], 1, x.shape[2]])

    x_view = pypto.reshape(cast_x, [x.shape[0], x.shape[1], x.shape[2] // 2, 2])
    pypto.set_vec_tile_shapes(1, 64, 64, 64)
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    pypto.set_vec_tile_shapes(1, 64, 64)
    x_rotate = rotate_half(x_re_second)

    # add two extra transpose to avoid last axis unalign transpose
    # origin calc flow: reshape(1,64,2,32)->transpose(1,64,32,2)->reshape(1,64,64)
    # new calc flow: transpose(1,64,64)->reshape(1,2,32,64)->transpose(1,32,2,64)->reshape(1,64,64)->transpose(1,64,64)
    x_rotate_trs_1 = pypto.transpose(x_rotate, 1, 2) # [1, 64.., 64]
    x_rotate_reshape_1 = pypto.reshape(x_rotate_trs_1, [
        x_rotate_trs_1.shape[0], 2, x_rotate_trs_1.shape[1] // 2, x_rotate_trs_1.shape[2]]) # [1, 2, 32, 64]
    pypto.set_vec_tile_shapes(1, 64, 64, 64)
    x_rotate_trs_2 = pypto.transpose(x_rotate_reshape_1, 1, 2) # [1, 32, 2, 64]
    x_rotate_reshape_2 = pypto.reshape(x_rotate_trs_2, x_rotate.shape) # [1, 64.., 64]
    pypto.set_vec_tile_shapes(1, 64, 64)
    x_rotate_res = pypto.transpose(x_rotate_reshape_2, 1, 2) # [1, 64, 64..]

    pypto.set_vec_tile_shapes(1, 64, 64)
    x_embed = cast_x * cast_cos + x_rotate_res * cast_sin
    x_embed_cast = pypto.cast(x_embed, x.dtype)

    return x_embed_cast


def mla_prolog_v4_compute(x, wq_a, wq_b, wkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, cos, sin, q_out, kv_out, qr_out, attrs, configs):
    t = x.shape[0]
    h = x.shape[1]
    q_lora_rank = rmsnorm_gamma_cq.shape[0]
    head_dim = rmsnorm_gamma_ckv.shape[0]
    head_num = wq_b.shape[1] // head_dim
    rope_dim = cos.shape[1]
    gamma_cq_2d = pypto.reshape(rmsnorm_gamma_cq, [1, rmsnorm_gamma_cq.shape[0]], inplace=True)
    gamma_ckv_2d = pypto.reshape(rmsnorm_gamma_ckv, [1, rmsnorm_gamma_ckv.shape[0]], inplace=True)
    pypto.set_vec_tile_shapes(4, q_lora_rank)
    gamma_cq_2d_fp32 = pypto.cast(gamma_cq_2d, pypto.DataType.DT_FP32)
    gamma_ckv_2d_fp32 = pypto.cast(gamma_ckv_2d, pypto.DataType.DT_FP32)

    unroll_list = configs.unroll_list
    for tIdx, unrollLength in pypto.loop_unroll(0, t, 1, name="MLA_BS_LOOP", idx_name="bs_offset",
                                                unroll_list=unroll_list, ):
        t_tile = unrollLength
        x_tile = pypto.view(x, [t_tile, h], [tIdx, 0], valid_shape=[t_tile, h])
        pypto.set_semantic_label("wqa-linear")
        pypto.set_cube_tile_shapes([32, 32], [512, 512], [64, 64], True)
        q = pypto.matmul(x_tile, wq_a, pypto.DataType.DT_BF16)
        pypto.set_semantic_label("q-rmsnorm with weight")
        pypto.set_vec_tile_shapes(8, q_lora_rank)
        qr = rms_norm(q, attrs.eps)
        qr = pypto.mul(qr, gamma_cq_2d_fp32)
        qr = pypto.cast(qr, pypto.DataType.DT_BF16)
        pypto.assemble(qr, [tIdx, 0], qr_out)

        pypto.set_semantic_label("wqb-linear")
        pypto.set_cube_tile_shapes([32, 32], [128, 128], [256, 256], True)
        q = pypto.matmul(qr, wq_b, pypto.DataType.DT_BF16)
        q_3d = pypto.reshape(q, [t_tile, head_num, head_dim])
        pypto.set_vec_tile_shapes(4, 64, 64)
        qr2_3d = rms_norm(q_3d, attrs.eps)
        qr2_3d = pypto.cast(qr2_3d, pypto.DataType.DT_BF16)
        pypto.set_vec_tile_shapes(4, 64)
        cos_2d = pypto.view(cos, [t_tile, rope_dim], [tIdx, 0], valid_shape=[t_tile, rope_dim])
        sin_2d = pypto.view(sin, [t_tile, rope_dim], [tIdx, 0], valid_shape=[t_tile, rope_dim])
        pypto.set_vec_tile_shapes(4, 64, 64)
        qr2_3d_nope = pypto.view(qr2_3d, [t_tile, head_num, head_dim-rope_dim], [0, 0, 0], valid_shape=[t_tile, head_num, head_dim-rope_dim])
        qr2_3d_rope = pypto.view(qr2_3d, [t_tile, head_num, rope_dim], [0, 0, head_dim-rope_dim], valid_shape=[t_tile, head_num, rope_dim])
        qr2_3d_rope = rope_3d(qr2_3d_rope, cos_2d, sin_2d)
        qr2_3d = pypto.concat([qr2_3d_nope, qr2_3d_rope], -1)
        pypto.assemble(qr2_3d, [tIdx, 0, 0], q_out)

        pypto.set_semantic_label("wkv-linear")
        pypto.set_cube_tile_shapes([32, 32], [256, 256], [128, 128], True)
        kv = pypto.matmul(x_tile, wkv, pypto.DataType.DT_BF16)
        pypto.set_vec_tile_shapes(4, 64)
        kv_norm = rms_norm(kv, attrs.eps)
        kv_norm = pypto.mul(kv_norm, gamma_ckv_2d_fp32)
        kv_norm = pypto.cast(kv_norm, pypto.DataType.DT_BF16)

        kv_norm_nope = pypto.view(kv_norm, [t_tile, head_dim-rope_dim], [0, 0], valid_shape=[t_tile, head_dim-rope_dim])
        kv_norm_rope = pypto.view(kv_norm, [t_tile, rope_dim], [0, head_dim-rope_dim], valid_shape=[t_tile, rope_dim])
        kv_norm_rope = rope_2d(kv_norm_rope, cos_2d, sin_2d)
        kv_norm = pypto.concat([kv_norm_nope, kv_norm_rope], -1)
        pypto.assemble(kv_norm, [tIdx, 0], kv_out)


class MLAKernelMAnager:
    def __init__(self):
        self.vec_all_shape = {}
        self.t_vec = [4096, 128, 64, 32, 16, 1]
        self.wq_a_shape = [4096, 1024]
        self.wq_b_shape = [1024, 64 * 512]
        self.wkv_shape = [4096, 512]
        self.rmsnorm_gamma_cq_shape = [1024]
        self.rmsnorm_gamma_ckv_shape = [512]

        for t in self.t_vec:
            x_shape = [t, 4096]
            rops_cos_shape = [t, 64]
            q_out_shape = [t, 64, 512]
            kv_out_shape = [t, 512]
            qr_out_shape = [t, 1024]
            self.vec_all_shape[t] = [x_shape, self.wq_a_shape, self.wq_b_shape, self.wkv_shape, self.rmsnorm_gamma_cq_shape, \
                                self.rmsnorm_gamma_ckv_shape,  rops_cos_shape, rops_cos_shape, q_out_shape, kv_out_shape, qr_out_shape]

    def infer_controlflow_shape(self, *args):
        global vec_all_shape, t_vec
        if not args:
            return [v for v in self.vec_all_shape.values()]
        x_shape = args[0]
        for t in self.t_vec:
            if x_shape[0]>=t:
                return self.vec_all_shape[t]


manager = MLAKernelMAnager()


@pypto.jit(runtime_options={
        "stitch_function_inner_memory": 1024,
        "stitch_function_outcast_memory": 1024,
        "stitch_cfgcache_size": 3000000,
        "triple_stream_sched": True
    },
    infer_controlflow_shape=manager.infer_controlflow_shape,)
def mla_prolog_v4(x, wq_a, wq_b, wkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, cos, sin, q_out, kv_out, qr_out, attrs, configs):
    pypto.experimental.set_operation_config(combine_axis=True)
    mla_prolog_v4_compute(x, wq_a, wq_b, wkv, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, cos, sin, q_out, kv_out, qr_out, attrs, configs)


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define("mla_prolog(Tensor token_x, Tensor wq_a, Tensor wq_b, Tensor wkv, Tensor rope_cos, Tensor rope_sin, Tensor gamma_cq, Tensor gamma_ckv) -> (Tensor, Tensor, Tensor)")

@torch.library.impl(pyptolib, "mla_prolog", "Meta")
def mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    q_out_shape = torch.empty([token_x.size(0), wq_b.size(1) // gamma_ckv.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=token_x.device)
    kv_out_shape = torch.empty([token_x.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=token_x.device)
    qr_out_shape = torch.empty([token_x.size(0), gamma_cq.size(0)], dtype=token_x.dtype, device=token_x.device)

    return q_out_shape, kv_out_shape, qr_out_shape

@torch.library.impl(pyptolib, "mla_prolog", "NPU")
def mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    return mla_prolog_v4_in(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv)

def mla_prolog_pypto(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    return torch.ops.pypto.mla_prolog(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv)


@allow_in_graph
def mla_prolog_v4_in(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv):
    output_q_data = torch.empty([token_x.size(0), wq_b.size(1) // gamma_ckv.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=f'{token_x.device}')
    output_kv_data = torch.empty([token_x.size(0), gamma_ckv.size(0)], dtype=token_x.dtype, device=f'{token_x.device}')
    output_qr_data = torch.empty([token_x.size(0), gamma_cq.size(0)], dtype=token_x.dtype, device=f'{token_x.device}')

    check_input_output_shape_dtype(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv,
                                    output_q_data, output_kv_data, output_qr_data)
    if not isinstance(token_x, FakeTensor):
        out_q = pypto.from_torch(output_q_data, dynamic_axis=[0], name="output_q")
        out_kv = pypto.from_torch(output_kv_data, dynamic_axis=[0], name="output_kv")
        out_qr = pypto.from_torch(output_qr_data, dynamic_axis=[0], name="output_qr")

    token_x_data = pypto.from_torch(token_x, dynamic_axis=[0], name="token_x")
    wq_a_data = pypto.from_torch(wq_a, name="wq_a")
    wq_b_data = pypto.from_torch(wq_b, name="wq_a")
    wkv_data = pypto.from_torch(wkv, name="w_kv")
    rope_cos_data = pypto.from_torch(rope_cos, dynamic_axis=[0], name="rope_cos")
    rope_sin_data = pypto.from_torch(rope_sin, dynamic_axis=[0], name="rope_sin")
    gamma_cq_data = pypto.from_torch(gamma_cq, name="gamma_cq")
    gamma_ckv_data = pypto.from_torch(gamma_ckv, name="gamma_ckv")

    input_data = [token_x_data, wq_a_data, wq_b_data, wkv_data, gamma_cq_data, gamma_ckv_data, rope_cos_data, rope_sin_data]
    output_data = [out_q, out_kv, out_qr]
    attrs = MlaPrologV4Attrs(eps=1e-6, layout_query="TND", layout_key="PA_BSND")
    configs = MlaPrologV4Configs(unroll_list=[128, 64, 32, 16, 1],
                                cube_l1_reuse_setting={2: 4},
                                mg_copyin_upper_bound=2 * 1024 * 1024,
                                pg_upper_bound=8192,
                                block_size=128,
                                t_sub_tile=1,
                                chunk_size=2,
                                vec_nbuffer_mode=1)
    mla_prolog_v4(*input_data, *output_data, attrs, configs)

    return output_q_data, output_kv_data, output_qr_data
