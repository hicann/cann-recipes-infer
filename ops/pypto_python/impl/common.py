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
deepseek_v4 common functions Module

This module implements common functions of deepseek_v4

Main Functions:
    - interleaved_rope_3d: attention common RoPE computation
    - rotate_half: rotate main function of rope
"""
import pypto
from typing import Tuple
from pypto import pypto_impl
from pypto.operation import op_wrapper

ROPE_DIM_2 = 2
ROPE_DIM_3 = 3
ROPE_CHUNK = 2


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
    assert shape_size >= 1, "rope rotate_half input dim less than 1"
    assert (
        shape[shape_size - 1] % ROPE_CHUNK == 0
    ), "rope rotate_half last dim shape is even"

    new_shape = list(shape)
    new_shape[shape_size - 1] //= ROPE_CHUNK

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = new_shape[shape_size - 1]

    x1 = pypto.view(input_tensor, new_shape, offset1)
    x2 = pypto.view(input_tensor, new_shape, offset2)

    return pypto.concat([x2 * (-1.0), x1 + 0.0], -1)


def inverse_rope_3d(
    x: pypto.Tensor, cos: pypto.Tensor, sin: pypto.Tensor
) -> pypto.Tensor:
    """Apply inverse 3D Rotary Position Embedding."""
    assert (
        len(x.shape) == ROPE_DIM_3
        and len(cos.shape) == ROPE_DIM_2
        and len(sin.shape) == ROPE_DIM_2
    )
    assert x.shape[2] == cos.shape[1] and cos.shape[1] == sin.shape[1]
    t = x.shape[0]
    n_q = x.shape[1]
    rope_dim = x.shape[2]

    pypto.set_vec_tile_shapes(1, rope_dim)
    cast_cos = pypto.cast(cos, pypto.DataType.DT_FP32)
    cast_sin = pypto.cast(sin, pypto.DataType.DT_FP32)
    cast_sin = cast_sin * (-1.0)

    pypto.set_vec_tile_shapes(1, n_q, rope_dim)
    cast_x = pypto.cast(x, pypto.DataType.DT_FP32)
    cast_cos = pypto.reshape(cast_cos, [t, 1, rope_dim])
    cast_sin = pypto.reshape(cast_sin, [t, 1, rope_dim])

    x_view = pypto.reshape(cast_x, [t, n_q, rope_dim // 2, 2])
    pypto.set_vec_tile_shapes(1, n_q, rope_dim, rope_dim)
    x_trans = pypto.transpose(x_view, 2, 3)
    x_re_second = pypto.reshape(x_trans, x.shape)
    pypto.set_vec_tile_shapes(1, n_q, rope_dim)
    x_rotate = rotate_half(x_re_second)

    # add two extra transpose to avoid last axis unalign transpose
    # origin calc flow: reshape(1,n_q,2,rope_dim // 2)->transpose(1,n_q,rope_dim // 2,2)->reshape(1,n_q,rope_dim)
    # new calc flow: transpose(1,n_q,rope_dim)->reshape(1,2,rope_dim // 2,n_q)->transpose(1,rope_dim // 2,2,n_q)->reshape(1,rope_dim,n_q)->transpose(1,n_q,rope_dim)
    x_rotate_trs_1 = pypto.transpose(x_rotate, 1, 2)
    x_rotate_reshape_1 = pypto.reshape(
        x_rotate_trs_1,
        [
            x_rotate_trs_1.shape[0],
            2,
            x_rotate_trs_1.shape[1] // 2,
            x_rotate_trs_1.shape[2],
        ],
    )
    pypto.set_vec_tile_shapes(1, rope_dim, rope_dim, n_q)
    x_rotate_trs_2 = pypto.transpose(x_rotate_reshape_1, 1, 2)
    x_rotate_add = pypto.add(x_rotate_trs_2, 0.0)
    x_rotate_reshape_2 = pypto.reshape(x_rotate_add, [t, rope_dim, n_q])
    pypto.set_vec_tile_shapes(1, rope_dim, n_q)
    x_rotate_res = pypto.transpose(x_rotate_reshape_2, 1, 2)
    pypto.set_vec_tile_shapes(1, n_q, rope_dim)
    x_embed = cast_x * cast_cos + x_rotate_res * cast_sin
    x_embed_cast = pypto.cast(x_embed, x.dtype)

    return x_embed_cast


@op_wrapper
def scalar_div(tensor, other, is_reserve=False):
    """Scalar division operation wrapper.

    Performs element-wise division of input tensor by a scalar value.

    Args:
        tensor: Input tensor
        other: Scalar divisor value
        is_reserve: Whether to reserve (inverse) the operation

    Returns:
        Result tensor after scalar division
    """
    return pypto_impl.ScalarDivS(
        tensor, pypto_impl.Element(tensor.dtype, other), is_reserve
    )


def quant(
    input_tensor: pypto.Tensor,
    is_symmetry: bool = True,
    has_smooth_factor: bool = False,
    smooth_factor: pypto.Tensor = None,
) -> Tuple[pypto.Tensor, pypto.Tensor]:
    """Quantize input tensor to INT8 with optional symmetry and smooth factor.

    Performs quantization to INT8 format with support for:
    - Symmetric quantization (centered around zero)
    - Asymmetric quantization (with offset)
    - Smooth quantization factor (for improved quantization quality)

    Args:
        input_tensor: Input tensor to quantize
        is_symmetry: If True, use symmetric quantization (range: [-127, 127])
                    If False, use asymmetric quantization (range: [0, 255])
        has_smooth_factor: Whether to apply smooth quantization factor
        smooth_factor: Smooth factor tensor to multiply before quantization

    Returns:
        Tuple of (quantized_tensor, dequant_scale):
            - quantized_tensor: INT8 quantized tensor
            - dequant_scale: FP32 scale factor for dequantization

    Note:
        For symmetric quantization, scale = max(|input|) / 127.0
        For asymmetric quantization, scale = (max - min) / 255.0
    """
    input_fp32 = pypto.cast(input_tensor, pypto.DT_FP32)
    if has_smooth_factor:
        input_fp32 = pypto.mul(input_fp32, smooth_factor)
    if is_symmetry:
        abs_res = pypto.abs(input_fp32)
        max_value = pypto.amax(abs_res, -1, keepdim=True)
        scale_quant = scalar_div(max_value, 127.0, True)
        out_fp32 = pypto.mul(input_fp32, scale_quant)
        out_int32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC)
        scale_de_quant = scalar_div(scale_quant, 1.0, True)
        return out_int8, scale_de_quant
    else:
        max_value = pypto.amax(input_fp32, -1, keepdim=True)
        min_value = pypto.amin(input_fp32, -1, keepdim=True)
        scale_de_quant = pypto.max(
            pypto.div(pypto.sub(max_value, min_value), 255.0), 1e-12
        )
        scale_quant = scalar_div(max_value, 1.0, True)
        out_fp32 = pypto.mul(input_fp32, scale_quant)
        out_int32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
        out_half = pypto.cast(out_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
        out_int8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC)
        return out_int8, scale_de_quant


def quant_tensor(x: pypto.Tensor):
    """Perform per-token quantization to INT8.
    Quantizes the input tensor to INT8 format using dynamic quantization.
    The quantization scale is computed per-token based on the maximum absolute
    value, ensuring the full INT8 range [-127, 127] is utilized.
    Args:
        input: Input tensor to quantize, can be any shape. Quantization is
               performed along the last dimension per token.
    Returns:
        Tuple of (quantized_tensor, dequant_scale):
            - quantized_tensor: INT8 quantized tensor, same shape as input
            - dequant_scale: FP32 scale factor for dequantization, shape matches
                            input with last dimension reduced to 1
    Note:
        The quantization process:
        1. Find per-token maximum absolute value
        2. Compute scale = 127.0 / max_value
        3. Quantize: int8 = round(input * scale)
        4. Return dequantization scale = 1.0 / scale
    """
    assert (
        len(pypto.get_vec_tile_shapes()) > 0
    ), f"expected set vec tile shape before call function, but not set."
    s8_max_value = 127.0
    s8_one_value = 1.0
    input_fp32 = pypto.cast(x, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

    abs_res = pypto.abs(input_fp32)
    max_value = pypto.amax(abs_res, dim=-1, keepdim=True)
    temp127 = pypto.full(max_value.shape, s8_max_value, pypto.DT_FP32)
    scale_quant = temp127 / max_value

    out_fp32 = input_fp32 * scale_quant
    out_int32 = pypto.cast(out_fp32, pypto.DT_INT32, pypto.CastMode.CAST_RINT)
    out_half = pypto.cast(out_int32, pypto.DT_FP16, pypto.CastMode.CAST_ROUND)
    out_int8 = pypto.cast(out_half, pypto.DT_INT8, pypto.CastMode.CAST_TRUNC)
    temp1 = pypto.full(scale_quant.shape, s8_one_value, pypto.DT_FP32)
    scale_dequant = temp1 / scale_quant
    return out_int8, scale_dequant
