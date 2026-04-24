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
""" """
import torch
import torch_npu
import pypto
import os
import sys
import logging
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
impl_path = os.path.join(parent_dir, 'impl')
sys.path.insert(0, impl_path)
from lightning_indexer_prolog_quant_pypto import *
from common_func import (
    apply_rotary_pos_emb,
    gen_uniform_data,
    quant_golden,
)
from compare import compare

def compute_quant_lightning_indexer_prolog(inputs, params):
    qr = inputs[0]
    idx_wq_b = inputs[1]
    x = inputs[2]
    weights_proj = inputs[3]
    cos = inputs[4]
    sin = inputs[5]
    hadamard = inputs[6]
    qr_scale = inputs[7]
    idx_wq_b_scale = inputs[8]

    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")
    rope_dim = params.get("rope_dim")
    calc_dtype = torch.bfloat16

    # q_linear & q_rope
    q_fp32 = torch.matmul(qr.to(torch.int32), idx_wq_b.to(torch.int32)).to(
        torch.float32
    )  # (t, idx_nq * head_dim)
    q_fp32 = q_fp32 * qr_scale
    q_fp32 = q_fp32 * idx_wq_b_scale.reshape(1, idx_nq * head_dim)
    q_re = q_fp32.to(calc_dtype).reshape(t, idx_nq, head_dim)
    q_nope, q_rope = torch.split(q_re, [head_dim - rope_dim, rope_dim], dim=-1)
    q_roped = apply_rotary_pos_emb(q_rope, cos, sin)
    q = torch.cat([q_nope, q_roped], dim=-1)

    # hadamard
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    # (t, idx_nq, head_dim) @ (1, head_dim, head_dim) -> (t, idx_nq, head_dim)
    q_hadamard = torch.matmul(
        q.to(torch.float32), hadamard.reshape(1, head_dim, head_dim).to(torch.float32)
    ).to(calc_dtype)
    # (t, idx_nq, head_dim), (t, idx_nq, 1)
    q_int8, q_scale = quant_golden(q_hadamard)
    q_scale = q_scale.to(torch.float16).reshape(t, idx_nq)

    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    weights = (
        torch.matmul(x.to(torch.float32), weights_proj.to(torch.float32))
        .to(calc_dtype)
        .to(torch.float32)
    )
    weights = weights * (idx_nq**-0.5) * (head_dim**-0.5)
    weights = weights.to(torch.float16)

    return q_int8, weights, q_scale


class QuantLightningIndexerProlog(torch.nn.Module):
    def forward(
        self,
        qr,
        idx_wq_b,
        x,
        weights_proj,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    ):
        for _ in range(20):
            torch.add(qr_scale, 0)
        return torch.ops.pypto.quant_lightning_indexer_prolog(
            qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
        )


def gen_quant_attention_post_golden(params):
    torch.manual_seed(42)
    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")
    rope_dim = params.get("rope_dim")
    q_lora_rank = params.get("q_lora_rank")
    h = params.get("h")

    # construct inputs
    qr_ori = gen_uniform_data([t, q_lora_rank], -1, 1, torch.int8)
    idx_wq_b_ori = gen_uniform_data([q_lora_rank, idx_nq * head_dim], -1, 1, torch.int8)
    x = gen_uniform_data([t, h], -1, 1, torch.bfloat16)
    weights_proj = gen_uniform_data([h, idx_nq], -0.1, 0.1, torch.bfloat16)
    cos = gen_uniform_data([t, rope_dim], -1, 1, torch.bfloat16)
    sin = gen_uniform_data([t, rope_dim], -1, 1, torch.bfloat16)
    hadamard = gen_uniform_data([head_dim, head_dim], -1, 1, torch.bfloat16)
    qr, qr_scale = quant_golden(qr_ori)
    idx_wq_b, idx_wq_b_scale = quant_golden(idx_wq_b_ori.t())
    idx_wq_b = idx_wq_b.t()

    # generate golden outputs
    inputs = [
        qr,
        idx_wq_b,
        x,
        weights_proj,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    ]
    q_golden, weights_golden, q_scale_golden = compute_quant_lightning_indexer_prolog(
        inputs, params
    )
    return inputs, q_golden, weights_golden, q_scale_golden


def check_input_shape_dtype(
    qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
):
    q_lora_rank = 1024
    idx_nq = 64
    head_dim = 128
    rope_dim = 64
    h = 4096
    assert (
        len(qr.shape) == 2 and qr.size(1) == q_lora_rank
    ), f"qr shape need to be: (t, f{q_lora_rank}), but got: f{qr.shape}"
    assert (
        len(idx_wq_b.shape) == 2
        and idx_wq_b.size(0) == q_lora_rank
        and idx_wq_b.size(1) == idx_nq * head_dim
    ), f"idx_wq_b shape need to be: (f{q_lora_rank, idx_nq * head_dim}), but got: f{idx_wq_b.shape}"
    assert (
        len(x.shape) == 2 and x.size(1) == h
    ), f"x shape need to be: (t, f{h}), but got: f{x.shape}"
    assert (
        len(weights_proj.shape) == 2
        and weights_proj.size(0) == h
        and weights_proj.size(1) == idx_nq
    ), f"weights_proj shape need to be: (f{h, idx_nq}), but got: f{weights_proj.shape}"
    assert (
        len(cos.shape) == 2 and cos.size(1) == rope_dim
    ), f"cos shape need to be: (t, f{rope_dim}), but got: f{cos.shape}"
    assert (
        len(sin.shape) == 2 and sin.size(1) == rope_dim
    ), f"sin shape need to be: (t, f{rope_dim}), but got: f{sin.shape}"
    assert (
        len(hadamard.shape) == 2
        and hadamard.size(0) == head_dim
        and hadamard.size(1) == head_dim
    ), f"hadamard shape need to be: (f{head_dim, head_dim}), but got: f{hadamard.shape}"
    assert (
        len(qr_scale.shape) == 2 and qr_scale.size(1) == 1
    ), f"qr_scale shape need to be: (t, f{1}), but got: f{qr_scale.shape}"
    assert (
        len(idx_wq_b_scale.shape) == 2
        and idx_wq_b_scale.size(0) == idx_nq * head_dim
        and idx_wq_b_scale.size(1) == 1
    ), f"idx_wq_b_scale shape need to be: (f{idx_nq * head_dim, 1}), but got: f{idx_wq_b_scale.shape}"

    assert (
        qr.dtype == idx_wq_b.dtype == torch.int8
    ), f"expected qr and idx_wq_b dtype to be torch.int8, but got: f{qr.dtype} and f{idx_wq_b.dtype}"
    assert (
        x.dtype
        == weights_proj.dtype
        == cos.dtype
        == sin.dtype
        == hadamard.dtype
        == torch.bfloat16
    ), f"expected x, weights_proj, cos, sin and hadamard dtype to be torch.bfloat16 but got: f{x.dtype}, f{weights_proj.dtype}, f{cos.dtype}, f{sin.dtype} and f{hadamard.dtype}"
    assert (
        qr_scale.dtype == idx_wq_b_scale.dtype == torch.float32
    ), f"expected qr_scale and idx_wq_b_scale dtype to be torch.float32, but got: f{qr_scale.dtype} and f{idx_wq_b_scale.dtype}"


@allow_in_graph
def npu_quant_lightning_indexer_prolog(
    qr: torch.Tensor,
    idx_wq_b: torch.Tensor,
    x: torch.Tensor,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    hadamard: torch.Tensor,
    qr_scale: torch.Tensor,
    idx_wq_b_scale: torch.Tensor,
):
    """
    torch npu graph interface

    """

    q = torch.empty(
        [qr.size(0), weights_proj.size(1), hadamard.size(0)],
        dtype=qr.dtype,
        device=qr.device,
    )
    weights = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )
    q_scale = torch.empty(
        [qr.size(0), weights_proj.size(1)],
        dtype=torch.float16,
        device=weights_proj.device,
    )

    check_input_shape_dtype(
        qr, idx_wq_b, x, weights_proj, cos, sin, hadamard, qr_scale, idx_wq_b_scale
    )
    # gen pto tensor from torch
    qr_pto = pypto.from_torch(qr, dynamic_axis=[0], name="qr")
    idx_wq_b_pto = pypto.from_torch(idx_wq_b, name="idx_wq_b")
    idx_wq_b_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x_pto = pypto.from_torch(x, dynamic_axis=[0], name="x")
    weights_proj_pto = pypto.from_torch(weights_proj, name="weights_proj")
    weights_proj_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    hadamard_pto = pypto.from_torch(hadamard, name="hadamard")
    hadamard_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    qr_scale_pto = pypto.from_torch(qr_scale, dynamic_axis=[0], name="qr_scale")
    idx_wq_b_scale_pto = pypto.from_torch(idx_wq_b_scale, name="idx_wq_b_scale")

    q_pto = pypto.from_torch(q, dynamic_axis=[0], name="q")
    weights_pto = pypto.from_torch(weights, dynamic_axis=[0], name="weights")
    q_scale_pto = pypto.from_torch(q_scale, dynamic_axis=[0], name="q_scale")

    # tiling
    tile_config = IndexerPrologQuantConfig(unroll_list=[128, 64, 32, 16, 8, 1])

    # kernel
    if not isinstance(qr, FakeTensor):
        pto_inputs = [
            qr_pto,
            idx_wq_b_pto,
            x_pto,
            weights_proj_pto,
            cos_pto,
            sin_pto,
            hadamard_pto,
            qr_scale_pto,
            idx_wq_b_scale_pto,
        ]
        pto_outputs = [q_pto, weights_pto, q_scale_pto]
        quant_lightning_indexer_prolog_kernel(*pto_inputs, *pto_outputs, tile_config)

    return q, weights, q_scale


def do_indexer_prolog_quant_func(inputs, params, golden_list):
    """
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
    """
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    qr = inputs[0].npu()
    idx_wq_b = inputs[1].npu()
    idx_wq_b_nz = torch_npu.npu_format_cast(idx_wq_b, torch_npu.Format.FRACTAL_NZ)
    x = inputs[2].npu()
    weights_proj = inputs[3].npu()
    weights_proj_nz = torch_npu.npu_format_cast(weights_proj, torch_npu.Format.FRACTAL_NZ)
    cos = inputs[4].npu()
    sin = inputs[5].npu()
    hadamard = inputs[6].npu()
    qr_scale = inputs[7].npu()
    idx_wq_b_scale = inputs[8].npu()

    t = params.get("t")
    idx_nq = params.get("idx_nq")
    head_dim = params.get("head_dim")

    # define npu outputs
    q = torch.zeros([t, idx_nq, head_dim]).to(torch.int8).npu()
    weights = torch.zeros([t, idx_nq]).to(torch.float16).npu()
    q_scale = torch.zeros([t, idx_nq]).to(torch.float16).npu()

    # gen pto tensor from torch
    qr_pto = pypto.from_torch(qr, dynamic_axis=[0], name="qr")
    idx_wq_b_pto = pypto.from_torch(idx_wq_b_nz, name="idx_wq_b")
    idx_wq_b_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    x_pto = pypto.from_torch(x, dynamic_axis=[0], name="x")
    weights_proj_pto = pypto.from_torch(weights_proj_nz, name="weights_proj")
    weights_proj_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    hadamard_pto = pypto.from_torch(hadamard, name="hadamard")
    hadamard_pto.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    qr_scale_pto = pypto.from_torch(qr_scale, dynamic_axis=[0], name="qr_scale")
    idx_wq_b_scale_pto = pypto.from_torch(idx_wq_b_scale, name="idx_wq_b_scale")

    q_pto = pypto.from_torch(q, dynamic_axis=[0], name="q")
    weights_pto = pypto.from_torch(weights, dynamic_axis=[0], name="weights")
    q_scale_pto = pypto.from_torch(q_scale, dynamic_axis=[0], name="q_scale")

    pto_inputs = [
        qr_pto,
        idx_wq_b_pto,
        x_pto,
        weights_proj_pto,
        cos_pto,
        sin_pto,
        hadamard_pto,
        qr_scale_pto,
        idx_wq_b_scale_pto,
    ]
    pto_outputs = [q_pto, weights_pto, q_scale_pto]

    tile_config = IndexerPrologQuantConfig(unroll_list=[128, 64, 32, 16, 8, 1])

    # call main function
    quant_lightning_indexer_prolog_kernel(*pto_inputs, *pto_outputs, tile_config)

    pypto.runtime._device_synchronize()
    compare(q.cpu(), golden_list[0], "q", atol=0.0001, rtol=0.005)
    compare(weights.cpu(), golden_list[1], "weights", atol=0.0001, rtol=0.007825)
    compare(q_scale.cpu(), golden_list[2], "q_scale", atol=0.0001, rtol=0.007825)


def do_indexer_prolog_quant_torch_graph(inputs, golden_list):
    """
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
    """
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # define npu inputs
    qr = inputs[0].npu()
    idx_wq_b = inputs[1].npu()
    idx_wq_b_nz = torch_npu.npu_format_cast(idx_wq_b, torch_npu.Format.FRACTAL_NZ)
    x = inputs[2].npu()
    weights_proj = inputs[3].npu()
    weights_proj_nz = torch_npu.npu_format_cast(weights_proj, torch_npu.Format.FRACTAL_NZ)
    cos = inputs[4].npu()
    sin = inputs[5].npu()
    hadamard = inputs[6].npu()
    qr_scale = inputs[7].npu()
    idx_wq_b_scale = inputs[8].npu()

    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(
        QuantLightningIndexerProlog(),
        dynamic=False,
        fullgraph=True,
        backend=npu_backend,
    )

    q, weights, q_scale = model(
        qr,
        idx_wq_b_nz,
        x,
        weights_proj_nz,
        cos,
        sin,
        hadamard,
        qr_scale,
        idx_wq_b_scale,
    )

    pypto.runtime._device_synchronize()
    compare(q.cpu(), golden_list[0], "q", atol=0.0001, rtol=0.005)
    compare(weights.cpu(), golden_list[1], "weights", atol=0.0001, rtol=0.007825)
    compare(q_scale.cpu(), golden_list[2], "q_scale", atol=0.0001, rtol=0.007825)


def get_indexer_prolog_quant_config(case_name: str):
    test_case_config = {
        "test_indexer_prolog_quant_b64_s1": 64,
        "test_indexer_prolog_quant_b64_s2": 128,
        "test_indexer_prolog_quant_b64_s4": 256,
        "test_indexer_prolog_quant_graph": 511,
    }
    return test_case_config.get(case_name)


def do_indexer_prolog_quant_entry(case_name: str, is_torch_graph: bool = False):
    params = {
        "idx_nq": 64,
        "head_dim": 128,
        "rope_dim": 64,
        "q_lora_rank": 1024,
        "h": 4096,
    }
    params["t"] = get_indexer_prolog_quant_config(case_name)
    if not params:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    inputs, q_golden, weights_golden, q_scale_golden = gen_quant_attention_post_golden(
        params
    )

    if is_torch_graph:
        print("\n =============== torch graph ====================")
        do_indexer_prolog_quant_torch_graph(
            inputs, [q_golden, weights_golden, q_scale_golden]
        )
    else:
        print("\n =============== st ====================")
        do_indexer_prolog_quant_func(
            inputs, params, [q_golden, weights_golden, q_scale_golden]
        )

    return True


def test_indexer_prolog_quant_b64_s1():
    """
    lightning_indexer_prolog quant decode mtp=0 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s1", is_torch_graph=False
    )


def test_indexer_prolog_quant_b64_s2():
    """
    lightning_indexer_prolog quant decode mtp=1 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s2", is_torch_graph=False
    )


def test_indexer_prolog_quant_b64_s4():
    """
    lightning_indexer_prolog quant decode mtp=3 case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_b64_s4", is_torch_graph=False
    )


def test_indexer_prolog_quant_graph():
    """
    lightning_indexer_prolog quant graph typical case
    """
    do_indexer_prolog_quant_entry(
        "test_indexer_prolog_quant_graph", is_torch_graph=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    test_indexer_prolog_quant_b64_s1()
