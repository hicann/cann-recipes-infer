# coding=utf-8
# Adapted from
# https://github.com/microsoft/microxcaling/blob/main/mx/formats.py
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import shutil
import collections
import traceback
from argparse import ArgumentParser
from glob import glob
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
import torch
from torch import Tensor
from safetensors.torch import load_file, save_file
from tqdm import tqdm


# ============================================================================
# Inlined from microxcaling_official.mx.formats
# ============================================================================

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)


class RoundingMode(IntEnum):
    nearest = 0
    floor = 1
    even = 2

    @staticmethod
    def string_enums():
        return [s.name for s in list(RoundingMode)]


class ElemFormat(Enum):
    int8 = 1
    int4 = 2
    int2 = 3
    fp8_e5m2 = 4
    fp8_e4m3 = 5
    fp6_e3m2 = 6
    fp6_e2m3 = 7
    fp4 = 8
    fp4_e2m1 = 8
    float16 = 9
    fp16 = 9
    bfloat16 = 10
    bf16 = 10

    @staticmethod
    def from_str(s):
        assert s is not None, "String elem_format == None"
        s = s.lower()
        if hasattr(ElemFormat, s):
            return getattr(ElemFormat, s)
        else:
            raise Exception("Undefined elem format", s)


def _get_min_norm(ebits):
    emin = 2 - (2 ** (ebits - 1))
    return 0 if ebits == 0 else 2 ** emin


def _get_max_norm(ebits, mbits):
    assert(ebits >= 5), "invalid for floats that don't define NaN"
    emax = 0 if ebits == 0 else 2 ** (ebits - 1) - 1
    return 2 ** emax * float(2 ** (mbits - 1) - 1) / 2 ** (mbits - 2)


_FORMAT_CACHE = {}


def _get_format_params(fmt):
    if isinstance(fmt, str):
        fmt = ElemFormat.from_str(fmt)

    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]

    if fmt == ElemFormat.int8:
        ebits, mbits = 0, 8
        emax = 0
    elif fmt == ElemFormat.int4:
        ebits, mbits = 0, 4
        emax = 0
    elif fmt == ElemFormat.int2:
        ebits, mbits = 0, 2
        emax = 0
    elif fmt == ElemFormat.fp8_e5m2:
        ebits, mbits = 5, 4
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.fp8_e4m3:
        ebits, mbits = 4, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e3m2:
        ebits, mbits = 3, 4
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp6_e2m3:
        ebits, mbits = 2, 5
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.fp4:
        ebits, mbits = 2, 3
        emax = 2**(ebits - 1)
    elif fmt == ElemFormat.float16:
        ebits, mbits = 5, 12
        emax = 2**(ebits - 1) - 1
    elif fmt == ElemFormat.bfloat16:
        ebits, mbits = 8, 9
        emax = 2**(ebits - 1) - 1
    else:
        raise Exception("Unknown element format %s" % fmt)

    if fmt != ElemFormat.fp8_e4m3:
        max_norm = 2**emax * float(2**(mbits - 1) - 1) / 2**(mbits - 2)
    else:
        max_norm = 2**emax * 1.75

    min_norm = _get_min_norm(ebits)

    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)

    return ebits, mbits, emax, max_norm, min_norm


# ============================================================================
# Inlined from microxcaling_official.mx.specs
# ============================================================================

_ASSERT_MODE = os.environ.get('MX_ASSERT', 'False')


class MxSpecs(collections.UserDict):
    def __init__(self, *args, **kwargs):
        super(MxSpecs, self).__init__(*args, **kwargs)

        defaults = {
            "scale_bits": 0,
            "w_elem_format": None,
            "a_elem_format": None,
            "w_elem_format_bp": None,
            "a_elem_format_bp": None,
            "a_elem_format_bp_ex": None,
            "a_elem_format_bp_os": None,
            "mx_flush_fp32_subnorms": False,
            "shared_exp_method": "max",
            "block_size": 0,
            "bfloat": 0,
            "fp": 0,
            "bfloat_subnorms": True,
            "quantize_backprop": True,
            "round": "nearest",
            "round_m": "nearest",
            "round_weight": "nearest",
            "round_output": "nearest",
            "round_grad_weight": "nearest",
            "round_grad_input": "nearest",
            "round_mx_output": "nearest",
            "round_mx_input_grad_input": "nearest",
            "round_mx_weight_grad_input": "nearest",
            "round_mx_grad_output_grad_input": "nearest",
            "round_mx_input_grad_weight": "nearest",
            "round_mx_grad_output_grad_weight": "nearest",
            "softmax_exp2": False,
            "vec_use_exp2": False,
            "vec_use_recip": False,
            "custom_cuda": False,
        }

        self.help_strings = {
            "scale_bits": "Bits (sign + magnitude) to use for shared exponent/scale",
            "w_elem_format": "Weight MX elem format",
            "a_elem_format": "Activation MX elem format",
            "w_elem_format_bp": "Backpass weight MX elem format",
            "a_elem_format_bp": "Backpass stashed activation MX elem format",
            "a_elem_format_bp_ex": "Backpass act (grad) MX elem format",
            "a_elem_format_bp_os": "Backpass act (grad) MX elem format",
            "mx_flush_fp32_subnorms": "MX quantization flushes blocks with subnormal shared scale to zero",
            "shared_exp_method": "Shared exponent calculation method",
            "block_size": "mx shared exponent block size",
            "bfloat": "BfloatX format",
            "fp": "fpX format",
            "bfloat_subnorms": "Bfloat/FP supports subnorms",
            "quantize_backprop": "Enable mx/bfloat quantization on backward pass",
            "round": "Global rounding mode",
            "round_m": "ADAM optimizer m and v rounding mode",
            "round_weight": "Weight bfloat rounding mode",
            "round_output": "Activation bfloat rounding mode",
            "round_grad_weight": "Weight update rounding mode",
            "round_grad_input": "Error gradient rounding mode",
            "round_mx_output": "Forward pass mx rounding mode",
            "round_mx_input_grad_input": "",
            "round_mx_weight_grad_input": "",
            "round_mx_grad_output_grad_input": "",
            "round_mx_input_grad_weight": "",
            "round_mx_grad_output_grad_weight": "",
            "softmax_exp2": "Softmax uses 2^x instead of e^x",
            "vec_use_exp2": "Use 2^x to compute e^x",
            "vec_use_recip": "Use 1/x to compute division",
            "custom_cuda": "Enable custom CUDA kernels for quantization",
        }

        for k in defaults:
            if k not in self.data.keys():
                self.data[k] = defaults[k]

        for k in self.data.keys():
            assert(k in self.help_strings.keys())


def get_default_mx_specs():
    specs = MxSpecs()
    return specs


def apply_mx_specs(mx_specs, default_mx_specs=None):
    if not default_mx_specs:
        default_mx_specs = get_default_mx_specs()

    if not mx_specs:
        return default_mx_specs

    for k in mx_specs:
        if mx_specs[k] is not None:
            if k not in default_mx_specs:
                raise KeyError(f"Unknown key '{k}' passed to mx specs")
            default_mx_specs[k] = mx_specs[k]

    return default_mx_specs


def finalize_mx_specs(specs, early_exit=True):
    format_keys = [
        "w_elem_format", "a_elem_format",
        "w_elem_format_bp", "a_elem_format_bp",
        "a_elem_format_bp_os", "a_elem_format_bp_ex",
        "bfloat", "fp",
    ]
    if early_exit and not any(specs.get(k, 0) for k in format_keys):
        return None

    if specs.get('custom_cuda'):
        assert torch.cuda.is_available(), f"'custom_cuda' is only supported on CUDA devices."

    def assign_if_none(f1, f2):
        if (f1 not in specs or specs[f1] is None) and f2 in specs:
            specs[f1] = specs[f2]

    assign_if_none("w_elem_format_bp", "w_elem_format")
    assign_if_none("a_elem_format_bp", "a_elem_format")
    assign_if_none("a_elem_format_bp_os", "a_elem_format")
    assign_if_none("a_elem_format_bp_ex", "a_elem_format")

    assign_if_none("round_m", "round")
    assign_if_none("round_output", "round")
    assign_if_none("round_grad_weight", "round")
    assign_if_none("round_grad_input", "round")
    assign_if_none("round_weight", "round")
    assign_if_none("round_mx_output", "round")

    assign_if_none("round_mx_input_grad_input", "round_grad_input")
    assign_if_none("round_mx_weight_grad_input", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")
    assign_if_none("round_mx_input_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_weight", "round_grad_input")
    assign_if_none("round_mx_grad_output_grad_input", "round_grad_input")

    specs = apply_mx_specs(specs, get_default_mx_specs())
    return specs


def mx_assert_test(mx_specs):
    if _ASSERT_MODE == "True" and mx_specs is None:
        stack = traceback.extract_stack()
        f1 = stack[-2]
        f2 = stack[-3]
        msg = (
            "MX assert test failed!\n"
            + f"mx_specs is None in function {f1.name}\n"
            + f"Called from {f2.filename}, line {f2.lineno}\n"
            + f"  {f2.line}"
        )
        raise ValueError(msg)


# ============================================================================
# MX Quantization configuration
# ============================================================================

@dataclass
class MxQuantConfig:
    scale_bits: int = 8
    elem_format: object = None
    shared_exp_method: str = "max"
    axes: object = None
    block_size: int = 0
    round_mode: str = "nearest"
    flush_fp32_subnorms: bool = False
    custom_cuda: bool = False
    shared_exp_round_method: str = "floor"
    return_fp: bool = True
    fp_scale: bool = False
    pack: bool = False
    saturate_normals: bool = False
    allow_denorm: bool = True

    @classmethod
    def from_mx_specs(cls, mx_specs, **overrides):
        return cls(
            scale_bits=mx_specs.get("scale_bits", 8) or 8,
            shared_exp_method=mx_specs.get("shared_exp_method", "max"),
            flush_fp32_subnorms=mx_specs.get("mx_flush_fp32_subnorms", False),
            custom_cuda=mx_specs.get("custom_cuda", False),
            block_size=mx_specs.get("block_size", 0),
            **overrides,
        )


# ============================================================================
# Inlined from microxcaling_official.mx.elemwise_ops
# ============================================================================

def _safe_lshift(x, bits, exp):
    if exp is None:
        return x * (2**bits)
    else:
        return x / (2 ** exp) * (2**bits)


def _safe_rshift(x, bits, exp):
    if exp is None:
        return x / (2**bits)
    else:
        return x / (2**bits) * (2 ** exp)


def _round_mantissa(a, bits, round_mode, clamp=False):
    if round_mode == "dither":
        rand_a = torch.rand_like(a, requires_grad=False)
        a = torch.sign(a) * torch.floor(torch.abs(a) + rand_a)
    elif round_mode == "floor":
        a = torch.sign(a) * torch.floor(torch.abs(a))
    elif round_mode == "nearest":
        a = torch.sign(a) * torch.floor(torch.abs(a) + 0.5)
    elif round_mode == "even":
        abs_a = torch.abs(a)
        mask_a = ((abs_a - 0.5) % 2 == torch.zeros_like(a)).type(a.dtype)
        a = torch.sign(a) * (torch.floor(abs_a + 0.5) - mask_a)
    else:
        raise Exception("Unrecognized round method %s" % (round_mode))

    if clamp:
        max_mantissa = 2 ** (bits - 1) - 1
        a = torch.clamp(a, -max_mantissa, max_mantissa)
    return a


def _quantize_elemwise_core(a, bits, exp_bits, max_norm, config=None):
    if config is None:
        config = MxQuantConfig()

    round_mode = config.round_mode
    saturate_normals = config.saturate_normals
    allow_denorm = config.allow_denorm
    custom_cuda = config.custom_cuda

    a_is_sparse = a.is_sparse
    if a_is_sparse:
        if a.layout != torch.sparse_coo:
            raise NotImplementedError("Only COO layout sparse tensors are currently supported.")
        sparse_a = a.coalesce()
        a = sparse_a.values().clone()

    custom_cuda = custom_cuda and round_mode in RoundingMode.string_enums()

    if custom_cuda:
        raise NotImplementedError("custom_cuda is not supported in this standalone script")

    if not allow_denorm and exp_bits > 0:
        min_norm = _get_min_norm(exp_bits)
        out = (torch.abs(a) >= min_norm).type(a.dtype) * a
    else:
        out = a

    if exp_bits != 0:
        private_exp = torch.floor(torch.log2(
            torch.abs(a) + (a == 0).type(a.dtype)))
        min_exp = -(2**(exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    out = _safe_lshift(out, bits - 2, private_exp)
    out = _round_mantissa(out, bits, round_mode, clamp=False)
    out = _safe_rshift(out, bits - 2, private_exp)

    if saturate_normals or exp_bits == 0:
        out = torch.clamp(out, min=-max_norm, max=max_norm)
    else:
        out = torch.where((torch.abs(out) > max_norm),
                           torch.sign(out) * float("Inf"), out)

    if not custom_cuda:
        out[a == float("Inf")] = float("Inf")
        out[a == -float("Inf")] = -float("Inf")
        out[a == float("NaN")] = float("NaN")

    return out


# ============================================================================
# Inlined from microxcaling_official.mx.pack_utils
# ============================================================================

def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


def f32_to_f4_unpacked(x):
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)


def _f32_to_floatx_unpacked(x: Tensor, ebits: int, mbits: int) -> Tensor:
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        (F32_EXP_BIAS - exp_bias)
        + (MBITS_F32 - mbits)
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4_npu(uint8_data) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(_down_size(shape))


# ============================================================================
# Inlined from microxcaling_official.mx.mx_ops
# ============================================================================

class SteMxExponent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        abs_x = torch.abs(x)
        exponent = torch.floor(torch.log2(abs_x))
        mantissa = abs_x / (2 ** exponent)
        exponent = torch.where(mantissa > 1.75, exponent + 1, exponent)
        return exponent

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_to_2nd_decimal(x):
    exponent = SteMxExponent.apply(x)
    return exponent


def _shared_exponents(a, method="max", axes=None, ebits=0, shared_exp_round_method="floor"):
    if method == "max":
        if axes is None:
            shared_exp = torch.max(torch.abs(a))
        else:
            shared_exp = a
            for axis in axes:
                shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
    elif method == "none":
        shared_exp = torch.abs(a)
    else:
        raise Exception("Unrecognized shared exponent selection method %s" % (method))
    if shared_exp_round_method == "floor":
        shared_exp = torch.floor(
            torch.log2(
                shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
            )
        )
    elif shared_exp_round_method == "round2decimal":
        shared_exp = round_to_2nd_decimal(shared_exp)
    else:
        raise ValueError("unexpected round method!")
    if ebits > 0:
        emax = 2 ** (ebits - 1) - 1
        shared_exp[shared_exp > emax] = float("NaN")
        shared_exp[shared_exp < -emax] = -emax

    return shared_exp


def _reshape_to_blocks(a, axes, block_size):
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    axes = [(x + len(a.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    for i, axis_val in enumerate(axes):
        axes[i] = axis_val + i
        a = torch.unsqueeze(a, dim=axes[i] + 1)

    orig_shape = a.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        a = torch.nn.functional.pad(a, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    padded_shape = a.size()
    reshape = _reshape(list(padded_shape), block_size)

    a = a.view(reshape)
    return a, axes, orig_shape, padded_shape


def _undo_reshape_to_blocks(a, padded_shape, orig_shape, axes):
    a = a.view(padded_shape)
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        a = a[slices]
    for axis in reversed(axes):
        a = torch.squeeze(a, dim=axis + 1)
    return a


def _quantize_mx(a, config):
    elem_format = config.elem_format
    if elem_format is None:
        return a

    scale_bits = config.scale_bits
    assert(scale_bits > 0)

    axes = config.axes
    axes = [axes] if type(axes) == int else axes
    axes = [x + a.ndim if x < 0 else x for x in axes]

    custom_cuda = config.custom_cuda
    round_mode = config.round_mode
    custom_cuda = custom_cuda and round_mode in RoundingMode.string_enums()

    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)

    if a.device.type == "cuda" and custom_cuda and len(axes) == 1:
        raise NotImplementedError("custom_cuda is not supported in this standalone script")

    block_size = config.block_size
    if block_size > 0:
        a, axes, orig_shape, padded_shape = _reshape_to_blocks(
            a, axes, block_size
        )

    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes

    if custom_cuda:
        raise NotImplementedError("custom_cuda is not supported in this standalone script")
    else:
        shared_exp = _shared_exponents(
            a, method=config.shared_exp_method, axes=shared_exp_axes, ebits=0,
            shared_exp_round_method=config.shared_exp_round_method
        )

        if config.flush_fp32_subnorms:
            a = a * (shared_exp > -FP32_EXPONENT_BIAS).type(a.dtype)

        shared_exp = shared_exp - emax

        scale_emax = 2 ** (scale_bits - 1) - 1
        shared_exp[shared_exp > scale_emax] = float("NaN")
        shared_exp[shared_exp < -scale_emax] = -scale_emax

        scale_e8m0_biased = shared_exp + 127
        assert (scale_e8m0_biased >= 0).all()

        a = a / (2**shared_exp)

        core_config = MxQuantConfig(
            round_mode=config.round_mode,
            custom_cuda=custom_cuda,
            saturate_normals=True,
            allow_denorm=True,
        )
        a = _quantize_elemwise_core(a, mbits, ebits, max_norm, config=core_config)

        if config.pack:
            assert elem_format.name == 'fp4'
            if block_size:
                a = _undo_reshape_to_blocks(a, padded_shape, orig_shape, axes)

            a_uint8 = f32_to_f4_unpacked(a)
            a = pack_uint4_npu(a_uint8)

            return a, scale_e8m0_biased

        if not config.return_fp:
            if block_size:
                a = _undo_reshape_to_blocks(a, padded_shape, orig_shape, axes)

            if config.fp_scale:
                return a, 2**shared_exp
            else:
                return a, scale_e8m0_biased

        a = a * (2**shared_exp)

    if block_size:
        a = _undo_reshape_to_blocks(a, padded_shape, orig_shape, axes)

    return a


def quantize_mx_op(a, config):
    elem_format = config.elem_format
    if elem_format is None:
        return a
    elif isinstance(elem_format, str):
        config.elem_format = ElemFormat.from_str(elem_format)

    return _quantize_mx(a, config)


# ============================================================================
# Original conversion script functions
# ============================================================================

def mx_quantize_real(x, bit=8):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if bit == 4:
        elem_fmt = 'fp4_e2m1'
        pack = True
    elif bit == 8:
        elem_fmt = 'fp8_e4m3'
        pack = False

    mx_specs = MxSpecs()
    mx_specs['scale_bits'] = 8
    mx_specs['w_elem_format'] = elem_fmt
    mx_specs['a_elem_format'] = elem_fmt
    mx_specs['block_size'] = 32
    mx_specs['bfloat'] = 16
    mx_specs['custom_cuda'] = False

    mx_specs = finalize_mx_specs(mx_specs)
    mx_specs = apply_mx_specs(mx_specs)

    config = MxQuantConfig.from_mx_specs(
        mx_specs,
        elem_format=elem_fmt,
        axes=[-1],
        shared_exp_round_method="round2decimal",
        return_fp=False,
        fp_scale=False,
        pack=pack,
    )

    x_mx, scale_e8m0 = quantize_mx_op(x, config)
    if bit == 8:
        x_mx = x_mx.to(torch.float8_e4m3fn)
    else:
        assert x_mx.dtype == (torch.uint8)
        pass

    scale_e8m0 = scale_e8m0.to(torch.uint8)
    scale_e8m0 = scale_e8m0.squeeze(-1)
    return x_mx, scale_e8m0


def get_had_pow2(n, norm=True):
    if not ((n & (n - 1) == 0) and (n > 0)):
        raise ValueError(f"n must be a positive power of 2, got{n}")
    had = torch.ones(1, 1)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1),
                        torch.cat([had, -had], 1)), 0)
        if norm:
            had /= math.sqrt(2)
    return had


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    M, N = weight.shape
    scale_m, scale_n = scale.shape
    assert scale_m == (
        M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (
        N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    weight = weight.to(torch.float32)
    scale_expanded = scale.repeat_interleave(
        block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:M, :N]
    dequantized_weight = weight * scale_expanded
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())
    return dequantized_weight


def scale_fp32_to_u64(weight_scale):
    k, n = weight_scale.shape
    scale_np = weight_scale.float().cpu().numpy()
    scale_uint32 = scale_np.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((k, n * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.uint64
    scale_uint64 = torch.from_numpy(scale_uint64).to(torch.uint64)
    return scale_uint64


def pack_4bit(x: torch.Tensor):
    assert x.dtype == torch.int8
    x = x.T.contiguous()
    shape = x.shape
    x = x.view(-1, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def int_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    qmax = 2 ** (bits - 1) - 1
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    if weight_clip_factor is not None:
        abs_max = abs_max * weight_clip_factor
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    if bits == 4:
        quantized = quantized.to(torch.int8)
        bias = int4_assistance_bias(quantized, scale)
        quantized = pack_4bit(quantized)
        scale = scale_fp32_to_u64(scale)
        return quantized, scale, bias
    else:
        return quantized.to(torch.int8), scale.to(torch.float32), None


def mxfp_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    weight_fp, scale_e8m0 = mx_quantize_real(tensor, bits)
    return weight_fp, scale_e8m0


def int4_assistance_bias(weight, weight_scale):
    repeat_times = weight.shape[1] // weight_scale.shape[1]
    expanded_scale = weight_scale.repeat_interleave(repeat_times, dim=1)
    weight_assistant_matrix = (expanded_scale * weight * 8).sum(dim=1).float()
    return weight_assistant_matrix


def is_match_layer_name(weight_name, layer_names):
    is_match_weight_name = False
    for layer in layer_names:
        if layer in weight_name:
            is_match_weight_name = True
            break
    return is_match_weight_name


def generate_ignore_item_back(num_layers):
    ignore = []
    for i in range(0, num_layers):
        ignore.append(f'model.layers.{i}.input_layernorm.0')
        ignore.append(f'model.layers.{i}.input_layernorm.1')
        ignore.append(f'model.layers.{i}.mlp.router.classifier')
        ignore.append(f'model.layers.{i}.mlp.router.e_score_correction_bias')
        ignore.append(f'model.layers.{i}.post_attention_layernorm.0')
        ignore.append(f'model.layers.{i}.post_attention_layernorm.1')
        ignore.append(f'model.layers.{i}.self_attn.0.kv_a_layernorm')
        ignore.append(f'model.layers.{i}.self_attn.0.q_a_layernorm')
        ignore.append(f'model.layers.{i}.self_attn.1.kv_a_layernorm')
        ignore.append(f'model.layers.{i}.self_attn.1.q_a_layernorm')
        ignore.append(f'model.layers.{i}.self_attn.0.kv_b_proj')
        ignore.append(f'model.layers.{i}.self_attn.1.kv_b_proj')

    ignore.append(f'model.mtp.embed_tokens')
    ignore.append(f'model.mtp.layers.0.eh_proj')
    ignore.append(f'model.mtp.layers.0.enorm.m')
    ignore.append(f'model.mtp.layers.0.hnorm.m')
    ignore.append(f'model.mtp.layers.0.input_layernorm')
    ignore.append(f'model.mtp.layers.0.post_attention_layernorm')
    ignore.append(f'model.mtp.layers.0.self_attn.kv_a_layernorm')
    ignore.append(f'model.mtp.layers.0.self_attn.kv_b_proj')
    ignore.append(f'model.mtp.layers.0.self_attn.q_a_layernorm')

    ignore.append('lm_head')
    ignore.append('model.embed_tokens')
    ignore.append('model.mtp.norm')
    ignore.append('model.norm')

    return ignore


def generate_ignore_item(num_layers):
    ignore = []
    for i in range(0, num_layers):
        ignore.append(f'model.layers.{i}.input_layernorm')
        ignore.append(f'model.layers.{i}.post_attention_layernorm')
        ignore.append(f'model.layers.{i}.mlp.gate')
        ignore.append(f'model.layers.{i}.self_attn.k_norm')
        ignore.append(f'model.layers.{i}.self_attn.q_norm')

    ignore.append('lm_head')
    ignore.append('model.embed_tokens')
    ignore.append('model.norm')

    return ignore


def generate_w4a8_quant(num_layers):
    quant_w4a8_layers = []
    for i in range(0, num_layers):
        quant_w4a8_layers.append(f'model.layers.{i}.mlp.experts')
    return quant_w4a8_layers


def generate_quant_group(a_num_bits=8, w_num_bits=8, targets=None, activation_use_clip=False):
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": 32, "num_bits": a_num_bits,
                                         "observer": "minmax", "observer_kwargs": {},
                                         "strategy": "group", "symmetric": True, "type": "float"},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "targets": targets,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": 32, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": "group", "symmetric": True, "type": "float"}}
    return quant_group


def generate_quant_config_bakc(c8, ignores, w4a8=False, clip=False):
    kv_cache_scheme = {"num_bits": 8,
                       "type": 'int',
                       "strategy": 'group',
                       "group_size": 128,
                       "dynamic": 'true',
                       "symmetric": 'true'} if c8 else None
    config_groups = {"group_0": {}}
    if w4a8:
        config_groups.update({"group_1": {}})
    quant_config = {"config_groups": config_groups,
                    "format": "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "kv_cache_scheme": kv_cache_scheme,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed"}
    targets = ["Linear"]
    quant_config["config_groups"]["group_0"] = generate_quant_group(
        a_num_bits=8, w_num_bits=8, targets=targets)
    if w4a8:
        quant_config["config_groups"]["group_1"] = generate_quant_group(
            a_num_bits=8, w_num_bits=4, targets=["MoEGMM"], activation_use_clip=clip)
    return quant_config


def generate_quant_config(c8, ignores, w4a8=False, clip=False):
    kv_cache_scheme = {"num_bits": 8,
                       "type": 'float',
                       "strategy": 'token',
                       "group_size": -1,
                       "dynamic": 'true',
                       "symmetric": 'true'} if c8 else None
    config_groups = {"group_0": {}}
    if w4a8:
        config_groups.update({"group_1": {}})
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized",
                    "global_compression_ratio": "null",
                    "ignore": ignores,
                    "kv_cache_scheme": kv_cache_scheme,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed",
                    "weight_block_size": [1, 32]}
    targets = ["Linear"]
    quant_config["config_groups"]["group_0"] = generate_quant_group(
        a_num_bits=8, w_num_bits=8, targets=targets)
    if w4a8:
        quant_config["config_groups"]["group_1"] = generate_quant_group(
            a_num_bits=8, w_num_bits=4, targets=["MoEGMM"], activation_use_clip=clip)
    return quant_config


def generate_li_hadamard_matrix(quant_param_path, num_layers, dim=128):
    hadamard_matrixs = {}
    for layer_idx in range(0, num_layers):
        key = f'model.layers.{layer_idx}.self_attn.indexer.hadamard_matrix'
        if quant_param_path is None:
            hadamard_matrixs[key] = get_had_pow2(
                dim, norm=True).to(torch.bfloat16)
        else:
            hadamard_path = os.path.join(
                quant_param_path, f'quant_parameters_{layer_idx}.pth')
            if not os.path.exists(hadamard_path):
                hadamard_matrix = get_had_pow2(
                    dim, norm=True).to(torch.bfloat16)
            else:
                quant_params = torch.load(hadamard_path)
                if key not in quant_params:
                    hadamard_matrix = get_had_pow2(
                        dim, norm=True).to(torch.bfloat16)
                else:
                    hadamard_matrix = quant_params[key].bfloat16()
            hadamard_matrixs[key] = hadamard_matrix
    return hadamard_matrixs


def copy_py_json(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def load_clip_params(num_hidden_layers, num_nextn_predict_layers, clip_param_path):
    num_layers = num_hidden_layers + num_nextn_predict_layers
    kv_clip_params = {}
    act_clip_params = {}
    weight_clip_params = {}
    clip_param_files = list(glob(os.path.join(clip_param_path, "*.pth")))
    clip_param_files.sort()
    for layer_idx in range(0, num_layers):
        expected_file = os.path.join(
            clip_param_path, f'quant_parameters_{layer_idx}.pth')
        if not os.path.exists(expected_file):
            if layer_idx < num_hidden_layers:
                raise ValueError(
                    f"{expected_file} not found, please check the {clip_param_path}")
            else:
                expected_file = os.path.join(
                    clip_param_path, f'quant_parameters_{num_hidden_layers - 1}.pth')
                old_quant_params = torch.load(expected_file)
                quant_params = {
                    k.replace(f'layers.{num_hidden_layers - 1}', f'layers.{layer_idx}'): torch.tensor(1.0).to(v.dtype)
                    for k, v in old_quant_params.items()}
        else:
            quant_params = torch.load(expected_file)
        for name, factor in quant_params.items():
            complete_name = f"model.layers.{layer_idx}.{name}"
            if complete_name.endswith("w_alpha"):
                weight_clip_params[complete_name] = factor
            elif complete_name.endswith("ckv_a_alpha"):
                kv_clip_params[complete_name] = factor
            elif complete_name.endswith("alpha"):
                act_clip_params[complete_name] = factor
    return kv_clip_params, act_clip_params, weight_clip_params


def load_c8_params(param_path, num_layers):
    clip_param_files = list(glob(os.path.join(param_path, "*.pth")))
    assert len(clip_param_files) == num_layers * 2
    kv_c8_params = {}
    scale_nums = 2
    for layer_idx in range(0, num_layers):
        for mla_idx in range(scale_nums):
            expected_file = os.path.join(param_path, f'quant_parameters_{mla_idx}_{layer_idx}.pth')
            mla_c8_scale = torch.load(expected_file)['self_attn.k_cache_quantizer.scale']
            c8_scale_name = f"model.layers.{layer_idx}.self_attn.{mla_idx}.c8_scale"
            kv_c8_params[c8_scale_name] = mla_c8_scale
    return kv_c8_params


def main(fp8_path, output_path, **kwargs):
    quant_type = kwargs.get("quant_type")
    clip = kwargs.get("clip")
    quant_param_path = kwargs.get("quant_param_path")
    model_name = kwargs.get("model_name")
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    assert quant_type in [
        "bfloat16", "w8a8c16", "w8a8c8", "w4a8c16", "w4a8c8"], f"Unsupported quant_type: {quant_type}"

    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, 'config.json')
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)
    if 'quantization_config' in config:
        config.pop('quantization_config')

    weight_map = model_index["weight_map"]
    new_weight_map = {}
    num_layers = config['num_hidden_layers']
    quant_ignore_layers = []
    quant_w4a8_layers = []
    c8 = quant_type.endswith('c8')
    w4a8 = quant_type.startswith("w4a8")
    w8a8 = quant_type.startswith("w8a8")

    if w8a8 or w4a8:
        quant_ignore_layers = generate_ignore_item(num_layers)
        c8_fake = True if model_name == "qwen3_moe" else False
        quantization_config = generate_quant_config(
            c8=c8 if not c8_fake else c8_fake, 
            ignores=quant_ignore_layers, 
            w4a8=w4a8, 
            clip=clip)
        config['quantization_config'] = quantization_config

    if w4a8:
        quant_w4a8_layers = generate_w4a8_quant(
            num_layers)

    loaded_files = {}

    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)
                    bf16_weight = weight_dequant(weight, scale_inv)
                    if w8a8 or w4a8:
                        is_ignore_layer = is_match_layer_name(
                            weight_name, quant_ignore_layers)
                        if is_ignore_layer:
                            print(f'Ignore quantization {weight_name}')
                        if not is_ignore_layer:
                            weight_clip_factor = None
                            bits = 8
                            if is_match_layer_name(weight_name, quant_w4a8_layers):
                                bits = 4
                                print(f" {weight_name} - bits {bits}")
                            mxfp_weight, scale_inv = mxfp_weight_quant(bf16_weight, bits=bits)
                            new_scale_name = scale_inv_name.replace(
                                '_scale_inv', '_scale')

                            new_state_dict[weight_name] = mxfp_weight
                            new_state_dict[new_scale_name] = scale_inv

                            new_weight_map[weight_name] = file_name
                            new_weight_map[new_scale_name] = file_name
                        else:
                            new_state_dict[weight_name] = bf16_weight
                            new_weight_map[weight_name] = file_name
                    else:
                        new_state_dict[weight_name] = bf16_weight
                        new_weight_map[weight_name] = file_name
                except KeyError:
                    print(
                        f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
            else:
                is_ignore_layer = is_match_layer_name(weight_name, quant_ignore_layers)

                if is_ignore_layer:
                    print(f'Ignore quantization {weight_name}')
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
                else:
                    bits = 8
                    if is_match_layer_name(weight_name, quant_w4a8_layers):
                        bits = 4
                        print(f" {weight_name} - bits {bits}")

                    mxfp_weight, scale_inv = mxfp_weight_quant(weight, bits=bits)
                    new_scale_name = weight_name.replace(
                        'weight', 'weight_scale')

                    new_state_dict[weight_name] = mxfp_weight
                    new_state_dict[new_scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    if c8:
        assert quant_param_path is not None, "Please pass the quant_param_path"
        kv_c8_params = load_c8_params(quant_param_path, num_layers)

        safetensor_files = list(
            glob(os.path.join(output_path, "*.safetensors")))
        safetensor_files.sort()
        first_safetensor_file = safetensor_files[-1]
        file_name = os.path.basename(first_safetensor_file)
        first_safetensor_dict = load_file(first_safetensor_file, device="cpu")

        first_safetensor_dict.update(kv_c8_params)
        for weight_name in kv_c8_params.keys():
            new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(first_safetensor_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

    copy_py_json(fp8_path, output_path)

    new_model_index_file = os.path.join(
        output_path, "model.safetensors.index.json")
    new_config_file = os.path.join(output_path, "config.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    with open(new_config_file, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    quant_type = "w4a8c16"
    clip = False
    quant_param_path = "None"
    model_name = "qwen3_moe"
    parser = ArgumentParser()
    parser.add_argument("--input_bf16_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    args = parser.parse_args()

    main(args.input_bf16_hf_path,
         args.output_hf_path,
         quant_type="w4a8c16", 
         clip=False,
         quant_param_path="None",
         model_name="qwen3_moe"
         )
