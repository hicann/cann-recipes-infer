# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests


np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

_NPU_BACKEND = None

def _get_npu_backend():
    global _NPU_BACKEND
    if _NPU_BACKEND is None:
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        _NPU_BACKEND = torchair.get_npu_backend(compiler_config=config)
    return _NPU_BACKEND


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.005, pct_thd=0.005, max_diff_hd=0.001):
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        return result, 0.0, max_error
    
    split_count = int(end - start + 1) if end != start else 1
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)
    
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    # if len(err_diff) > 0:
    #     max_error = max(err_diff)
    #     if max(err_diff) >= max_diff_hd:
    #         result = "Failed"

    return result, fulfill_percent, max_error

def hc_split_sinkhorn_torch(
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    # base/scale按mixes维度自适应广播（2D mixes即3D输入时为(1,hc)，3D mixes即4D输入时为(1,1,hc)），
    # 避免双重unsqueeze引入虚假前导维，导致y的sum轴错位（golden形状/数值错误）
    lead = (1,) * (mixes.dim() - 1)
    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].reshape(lead + (hc_mult,))) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].reshape(lead + (hc_mult,)))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].reshape(lead + (hc_mult, hc_mult))

    comb = comb.softmax(-1) + eps
    col_sum = comb.sum(-2, keepdim=True)
    comb = comb / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


def to_hf32(t: torch.Tensor) -> torch.Tensor:
    # Model the Cube Matmul HF32 mode used by the HcPre kernel (SetHF32Mode(1) + SetHF32TransMode(1)):
    # before the multiply, each fp32 operand in L0A/L0B is rounded to HF32 (1 sign + 8 exp + 10 mantissa).
    # SetHF32TransMode(1) selects round-toward-zero, which is exactly clearing the low 13 mantissa bits of
    # the fp32 bit pattern (23 -> 10). Accumulation stays fp32, so only the inputs are truncated here.
    # (x is bf16-valued -> 7 mantissa bits -> already representable in HF32, so this is a no-op for x; it
    #  matters for the fp32 hc_fn weight.)
    hf32_mantissa_bits = 10
    drop = 23 - hf32_mantissa_bits
    bits = t.contiguous().view(torch.int32)
    return (bits & (~((1 << drop) - 1))).view(torch.float32)

def _hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int, hc_sinkhorn_iters: int, norm_eps: float, hc_eps: float):
    # x: [b, s, hc, d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b, s, d]
    shape, dtype = x.size(), x.dtype
    if x.dim() == 4:
        x = x.flatten(2).float()
    elif x.dim() == 3:
        x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    # The kernel computes mixes via the Cube Matmul in HF32 mode; mirror that in the golden so the
    # comparison reflects true algorithmic error rather than the (expected) HF32 vs fp32 gap.
    mixes = F.linear(to_hf32(x), to_hf32(hc_fn)) * rsqrt

    pre, post, comb = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    # y = Σ_hc mix_hc * x[hc]：求和轴是 hc 轴（4D 为 dim=2，3D 为 dim=1），统一取倒数第二维
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=len(shape) - 2)
    return y.to(dtype), post, comb

def _hc_pre_v2(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
               pre_mix: torch.Tensor, hc_mult: int, hc_sinkhorn_iters: int, norm_eps: float, hc_eps: float):
    # v2 golden: when pre_mix is provided, y is computed from pre_mix instead of the kernel-computed pre;
    # post/comb are always derived from mixes. pre always follows the kernel formula.
    shape, dtype = x.size(), x.dtype
    if x.dim() == 4:
        x = x.flatten(2).float()
    elif x.dim() == 3:
        x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(to_hf32(x), to_hf32(hc_fn)) * rsqrt

    pre, post, comb = hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    mix_for_y = pre_mix.float() if pre_mix is not None else pre
    # y = Σ_hc mix_hc * x[hc]：求和轴是 hc 轴（4D 为 dim=2，3D 为 dim=1），统一取倒数第二维
    y = torch.sum(mix_for_y.unsqueeze(-1) * x.view(shape), dim=len(shape) - 2)
    return y.to(dtype), post, comb, pre

def run_hc_pre_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, graph_mode=False):
    hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
    hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
    hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
    x_npu = x.to("npu:%s" % DEVICE_ID)

    golden_y_out, golden_post_out, golden_comb_frag_out = _hc_pre(
        x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    if graph_mode:
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
                npu_y_out, npu_post_out, npu_comb_frag_out = torch.ops.custom.npu_hc_pre(
                    x, hc_fn, hc_scale, hc_base, hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters,
                    norm_eps=norm_eps, hc_eps=hc_eps
                )
                return npu_y_out, npu_post_out, npu_comb_frag_out

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        npu_backend = _get_npu_backend()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_y_out, npu_post_out, npu_comb_frag_out = npu_mode(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
    else:
        npu_y_out, npu_post_out, npu_comb_frag_out = torch.ops.custom.npu_hc_pre(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps)

    # Precision targets: y is bfloat16 -> 4e-3 (= bf16 ULP). post & comb_frag are float32 -> 1e-4.
    # y keeps a looser element pass-rate (pct_thd) because the bf16 output rounding plus the kernel's
    # vector-unit sigmoid approximation put a small fraction of elements right on a bf16 rounding
    # boundary (max observed ~2 bf16 ULP); post/comb_frag are exact-enough to hold the tight 99.5%.
    compare_y = data_compare(golden_y_out.cpu().float().numpy(), npu_y_out.cpu().float().numpy(),
                             diff_thd=0.004, pct_thd=0.02)
    compare_post = data_compare(golden_post_out.cpu().numpy(), npu_post_out.cpu().float().numpy(),
                                diff_thd=0.0001, pct_thd=0.005)
    compare_comb_frag = data_compare(golden_comb_frag_out.cpu().numpy(), npu_comb_frag_out.cpu().float().numpy(),
                                     diff_thd=0.0001, pct_thd=0.005)

    return compare_y, compare_post, compare_comb_frag


def run_hc_pre_v2_case(x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps,
                       hc_eps, graph_mode=False):
    hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
    hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
    hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
    x_npu = x.to("npu:%s" % DEVICE_ID)
    pre_mix_npu = pre_mix.to("npu:%s" % DEVICE_ID) if pre_mix is not None else None

    golden_y_out, golden_post_out, golden_comb_frag_out, golden_pre_out = _hc_pre_v2(
        x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    if graph_mode:
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
                npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out = torch.ops.custom.npu_hc_pre_v2(
                    x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult=hc_mult,
                    hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps
                )
                return npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        npu_backend = _get_npu_backend()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out = npu_mode(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, pre_mix_npu, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
    else:
        npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out = torch.ops.custom.npu_hc_pre_v2(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, pre_mix_npu,
            hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps)

    # Precision targets identical to npu_hc_pre: y is bfloat16 -> 4e-3; post/comb/pre are float32 -> 1e-4.
    compare_y = data_compare(golden_y_out.cpu().float().numpy(), npu_y_out.cpu().float().numpy(),
                             diff_thd=0.004, pct_thd=0.02)
    compare_post = data_compare(golden_post_out.cpu().numpy(), npu_post_out.cpu().float().numpy(),
                                diff_thd=0.0001, pct_thd=0.005)
    compare_comb_frag = data_compare(golden_comb_frag_out.cpu().numpy(), npu_comb_frag_out.cpu().float().numpy(),
                                     diff_thd=0.0001, pct_thd=0.005)
    compare_pre = data_compare(golden_pre_out.cpu().numpy(), npu_pre_out.cpu().float().numpy(),
                               diff_thd=0.0001, pct_thd=0.005)
    return compare_y, compare_post, compare_comb_frag, compare_pre


def create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=42):
    np.random.seed(seed)
    hc_scale = torch.tensor(np.random.uniform(0, 2, (3))).to(torch.float32)
    hc_base = torch.tensor(np.random.uniform(0, 2, (hc_mix))).to(torch.float32)
    # hc_fn (the [mix_hc, hc_mult*d] projection weight) is kept positive but scaled by 1/(hc_mult*d) so the
    # matmul output `mixes = x @ hc_fn^T` stays O(1) instead of O(hc_mult*d). With the original U(0,2) the
    # sum over hc_mult*d all-positive terms drives the comb softmax logits to ~1e3, making it effectively
    # one-hot and the 20-iter Sinkhorn numerically singular (tiny perturbations blow up after iteration) —
    # no finite-precision kernel can match an fp32 golden to 1e-4 there. Keeping mixes O(1) makes the
    # Sinkhorn path well-conditioned (comb error drops to ~1e-6) while preserving the all-positive
    # (no catastrophic-cancellation) regime that keeps y/post accurate. Range stays data-dependent/meaningful.
    fan_in = hc_mult * shape[-1]
    hc_fn_hi = 1.0 / fan_in
    hc_fn = torch.tensor(np.random.uniform(0, hc_fn_hi, (hc_mix, fan_in))).to(torch.float32)
    # x直接以bf16生成：避免numpy float64中间态（大bs*d时可到数GB）叠加golden的fp32峰值，
    # 触发容器cgroup内存上限被OOM Killed
    gen = torch.Generator()
    gen.manual_seed(seed)
    x = torch.empty(shape, dtype=torch.bfloat16).uniform_(0, 2, generator=gen)
    return x, hc_fn, hc_scale, hc_base


class TestCustomHcPre(TestCase):
    def _run_and_check_case(self, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, graph_mode=False):
        compare_y, compare_post, compare_comb_frag = run_hc_pre_case(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, graph_mode=graph_mode)
        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")

    def test_hc_pre_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 5120, 7168]
        for d in d_list:
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((b, s, hc_mult, d), hc_mix, hc_mult)
            print(f'======================== PTA eager test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_graph(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        d_list = [4096, 5120, 7168]
        for d in d_list:
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((b, s, hc_mult, d), hc_mix, hc_mult)
            print(f'======================== PTA graph test ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, graph_mode=True)

    def test_hc_pre_ascend950_large_bs_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        for d in [4096, 5120, 7168]:
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((8193, hc_mult, d), hc_mix, hc_mult, seed=d)
            print(f'======================== PTA large-bs eager regression test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_small_bs_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        for d in [4096, 5120, 7168]:
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((3079, hc_mult, d), hc_mix, hc_mult, seed=d)
            print(f'======================== PTA small-bs eager regression test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_fusion_bs_eager(self):
        """Ascend950: bs > 512 and divisible by 256, routed to HcPre fused op."""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        # bs values: <=512 range + >=8192 and divisible by 256
        bs_list = [
            # bs <= 512 range (always fused on 950)
            1, 16, 32, 64, 128, 256, 512,
            # bs >= 8192, divisible by 256 (fused)
            8192,      # power of 2, minimum fused
            10240,     # 20 * 512
            12288,     # 24 * 512
            16384,     # power of 2
            20480,     # 40 * 512
            24576,     # 48 * 512
            32768,     # power of 2
            40960,     # 80 * 512
            49152,     # 96 * 512
            64000,     # upper bound, 250 * 256
        ]

        import gc
        for bs in bs_list:
            seed = bs
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                (bs, hc_mult, 4096), hc_mix, hc_mult, seed=seed)
            print(f'=== Ascend950 fusion bs eager: bs={bs}, d=4096 ===')
            self._run_and_check_case(
                x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
            del x, hc_fn, hc_scale, hc_base
            gc.collect()
            torch_npu.npu.empty_cache()

    def test_hc_pre_ascend950_fusion_bs_extend_d_eager(self):
        """Ascend950: bs > 512 and divisible by 256, with extended D (d=5120/7168)."""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        bs_list = [
            1, 128, 256, 512, 8192, 16384, 32768, 64000,
        ]

        import gc
        for d in [5120, 7168]:
            for bs in bs_list:
                seed = bs + d
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                    (bs, hc_mult, d), hc_mix, hc_mult, seed=seed)
                print(f'=== Ascend950 fusion bs eager (d={d}): bs={bs} ===', flush=True)
                self._run_and_check_case(
                    x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                del x, hc_fn, hc_scale, hc_base
                gc.collect()
                torch_npu.npu.empty_cache()

    def test_hc_pre_generalized_d_eager(self):
        """测试 d=4096/5120/7168，所有芯片通用"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 5120, 7168]

        rng = random.Random(2025)
        bs_candidates = []
        for _ in range(8):
            bs_candidates.append(rng.randint(1, 32768))
        bs_candidates += [1, 128, 1024, 4096, 32768]
        seen = set()
        bs_list = []
        for v in bs_candidates:
            if v not in seen:
                seen.add(v)
                bs_list.append(v)

        for d in d_list:
            for bs in bs_list:
                seed = d * 100 + bs
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((bs, hc_mult, d), hc_mix, hc_mult, seed=seed)
                print(f'=== hc_pre generalized D eager: d={d}, bs={bs} ===')
                self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_generalized_bs_eager(self):
        """泛化 bs 精度验证：覆盖 Ascend950 两条路由(融合算子与小算子拼接)，d=4096/7168。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This generalized-bs case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        import gc
        bs_list = [
            # fused: bs <= 512
            1, 2, 7, 16, 31, 64, 128, 256, 511, 512,
            # composite: 512 < bs, bs % 8192 != 0
            513, 768, 1000, 1536, 3079, 4096, 6000, 8193, 10000, 12288, 20480, 30000,
            # fused: bs % 8192 == 0
            8192, 16384, 24576, 32768,
        ]
        fails = []
        for d in [4096, 5120, 7168]:
            for bs in bs_list:
                seed = d * 131 + bs
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((bs, hc_mult, d), hc_mix, hc_mult, seed=seed)
                route = "fused" if (bs <= 512 or bs % 8192 == 0) else "composite"
                cy, cp, cc = run_hc_pre_case(
                    x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                ok = (cy[0] == "Pass" and cp[0] == "Pass" and cc[0] == "Pass")
                print(f'=== hc_pre generalized bs: d={d} bs={bs:>6} [{route:9}] '
                      f'y={cy[1]:.3f}% post={cp[1]:.3f}% comb={cc[1]:.3f}% -> {"PASS" if ok else "FAIL"}', flush=True)
                if not ok:
                    fails.append((d, bs, route, cy, cp, cc))
                del x, hc_fn, hc_scale, hc_base
                gc.collect()
                torch_npu.npu.empty_cache()
        if fails:
            msg = "; ".join(f"d{d}-bs{bs}({r}):y{cy[1]:.2f}/post{cp[1]:.2f}/comb{cc[1]:.2f}"
                            for d, bs, r, cy, cp, cc in fails)
            self.fail(f"{len(fails)} generalized-bs case(s) failed: {msg}")


class TestCustomHcPreV2(TestCase):
    """npu_hc_pre_v2：pre_mix 可选输入，总是返回 pre。shape 生成方式与 v1 保持一致。"""

    # 4D BS 泛化扫描（范围 [1, 2048]）：极小值、8 对齐各余数（Brcb 按 8 float 广播，行数 mod 8 敏感）、
    # 16/32/64/128 对齐边界、中值、大值与上边界
    GEN_BS_LIST_4D = [
        1, 2, 3, 5, 7, 8, 9,
        15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129,
        192, 255, 257,
        511, 512, 513, 1023, 1024, 1025, 2047, 2048,
    ]
    # 3D 与 4D(b=1) 数学等价，用精简子集覆盖 kernel/converter 的 3D 分支（余数 + 边界 + 大值）
    GEN_BS_LIST_3D = [
        1, 2, 3, 7, 8, 15, 16, 17, 32, 63, 64, 65, 128, 192, 255, 512, 1024, 2047, 2048,
    ]

    def _run_and_check_case(self, x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters,
                            norm_eps, hc_eps):
        compare_y, compare_post, compare_comb_frag, compare_pre = run_hc_pre_v2_case(
            x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")
        assert(compare_pre[0] == "Pass")

    def _run_v2_bs_sweep(self, dim_mode, bs_list, d_list, hc_mix, hc_mult, hc_sinkhorn_iters,
                         norm_eps, hc_eps, tag):
        """BS 扫描：dim_mode='4d' 时 shape=(1, bs, hc_mult, d)，'3d' 时 shape=(bs, hc_mult, d)。
        d 按 bs 序号轮转 [4096, 5120, 7168]。收集失败明细统一汇报，不逐 case 断言。"""
        import gc
        fails = []
        for i, bs in enumerate(bs_list):
            d = d_list[i % len(d_list)]
            if dim_mode == '4d':
                shape = (1, bs, hc_mult, d)
            else:
                shape = (bs, hc_mult, d)
            seed = d * 1000 + bs
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=seed)
            # pre_mix 故意取与公式 pre 不同的随机值，确保 y 确实由 pre_mix 计算而非 pre
            pre_mix = torch.tensor(np.random.uniform(0, 2, shape[:-1])).to(torch.float32)
            cy, cp, cc, cpre = run_hc_pre_v2_case(
                x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
            ok = all(c[0] == "Pass" for c in (cy, cp, cc, cpre))
            print(f'=== v2 {tag}: bs={bs:>5} d={d} y={cy[1]:.3f}% post={cp[1]:.3f}% '
                  f'comb={cc[1]:.3f}% pre={cpre[1]:.3f}% -> {"PASS" if ok else "FAIL"}', flush=True)
            if not ok:
                fails.append(f"{tag}-bs{bs}-d{d}: "
                             f"y{cy[1]:.2f}/post{cp[1]:.2f}/comb{cc[1]:.2f}/pre{cpre[1]:.2f}")
            del x, hc_fn, hc_scale, hc_base, pre_mix
            gc.collect()
            torch_npu.npu.empty_cache()
        return fails

    def test_hc_pre_v2_eager(self):
        """pre_mix 输入：y 用 pre_mix 加权求和，pre 输出对比公式 golden。shape 生成与 v1 一致。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 5120, 7168]
        for d in d_list:
            shape = (b, s, hc_mult, d)
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult)
            # pre_mix 故意取与公式 pre 不同的随机值，确保 y 确实由 pre_mix 计算而非 pre
            pre_mix = torch.tensor(np.random.uniform(0, 2, shape[:-1])).to(torch.float32)
            print(f'======================== v2 eager pre_mix+pre test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, pre_mix,
                                     hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_v2_no_optional_eager(self):
        """v2 缺省可选参数（pre_mix=None），y 由公式 pre 计算（与原 npu_hc_pre 一致），pre 输出对比公式 golden。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 5120, 7168]
        for d in d_list:
            shape = (b, s, hc_mult, d)
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult)
            print(f'======================== v2 eager no-optional test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, None,
                                     hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_v2_generalization_4d_eager(self):
        """v2 泛化 4D：(b, s, 4, d)，BS=b*s∈[1,2048] 大小值/对齐非对齐覆盖，d 轮转 [4096, 5120, 7168]。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        import gc
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        d_list = [4096, 5120, 7168]

        fails = self._run_v2_bs_sweep('4d', self.GEN_BS_LIST_4D, d_list, hc_mix, hc_mult,
                                      hc_sinkhorn_iters, norm_eps, hc_eps, '4D-gen')
        # 补充 b>1 组合：覆盖 4D 的 b 维展开路径（含非对齐 s）
        for (b, s, d) in [(2, 97, 5120), (7, 293, 4096), (16, 128, 7168)]:
            shape = (b, s, hc_mult, d)
            seed = d * 1000 + b * 100 + s
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=seed)
            pre_mix = torch.tensor(np.random.uniform(0, 2, shape[:-1])).to(torch.float32)
            cy, cp, cc, cpre = run_hc_pre_v2_case(
                x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
            ok = all(c[0] == "Pass" for c in (cy, cp, cc, cpre))
            print(f'=== v2 4D-gen b>1: b={b} s={s} (bs={b * s}) d={d} y={cy[1]:.3f}% post={cp[1]:.3f}% '
                  f'comb={cc[1]:.3f}% pre={cpre[1]:.3f}% -> {"PASS" if ok else "FAIL"}', flush=True)
            if not ok:
                fails.append(f"4D-gen-b{b}s{s}-d{d}: "
                             f"y{cy[1]:.2f}/post{cp[1]:.2f}/comb{cc[1]:.2f}/pre{cpre[1]:.2f}")
            del x, hc_fn, hc_scale, hc_base, pre_mix
            gc.collect()
            torch_npu.npu.empty_cache()
        if fails:
            self.fail(f"{len(fails)} v2 4D generalization case(s) failed: {'; '.join(fails)}")

    def test_hc_pre_v2_generalization_3d_eager(self):
        """v2 泛化 3D：(bs, 4, d)，BS∈[1,2048] 覆盖，d 轮转 [4096, 5120, 7168]。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        d_list = [4096, 5120, 7168]

        fails = self._run_v2_bs_sweep('3d', self.GEN_BS_LIST_3D, d_list, hc_mix, hc_mult,
                                      hc_sinkhorn_iters, norm_eps, hc_eps, '3D-gen')
        if fails:
            self.fail(f"{len(fails)} v2 3D generalization case(s) failed: {'; '.join(fails)}")

    def test_hc_pre_v2_ascend950_tiling_paths_eager(self):
        """Ascend950(A5): v2 覆盖 arch35 两条 tiling 路径（小bs=1001 / 大bs=1000），pre_mix 传/不传。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This v2 A5 tiling-path case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        # 小bs走tilingKey=1001，大bs走tilingKey=1000
        bs_list = [192, 3079, 8193, 16384]
        import gc
        for d in [4096, 5120, 7168]:
            for bs in bs_list:
                shape = (bs, hc_mult, d)
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=bs + d)
                # pre_mix 故意取与公式 pre 不同的随机值，确保 y 确实由 pre_mix 计算而非 pre
                pre_mix = torch.tensor(np.random.uniform(0.5, 1.5, shape[:-1])).to(torch.float32)
                for use_pre_mix in [True, False]:
                    cur_pre_mix = pre_mix if use_pre_mix else None
                    print(f'=== v2 A5 tiling-path eager: d={d}, bs={bs}, use_pre_mix={use_pre_mix} ===')
                    self._run_and_check_case(x, hc_fn, hc_scale, hc_base, cur_pre_mix,
                                             hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                del x, hc_fn, hc_scale, hc_base, pre_mix
                gc.collect()
                torch_npu.npu.empty_cache()

    def test_hc_pre_v2_ascend950_graph(self):
        """Ascend950(A5): v2 图模式（torchair成图验证converter），pre_mix 传/不传。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This v2 A5 graph case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        for d in [4096, 5120, 7168]:
            shape = (192, hc_mult, d)
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=d)
            pre_mix = torch.tensor(np.random.uniform(0.5, 1.5, shape[:-1])).to(torch.float32)
            for use_pre_mix in [True, False]:
                cur_pre_mix = pre_mix if use_pre_mix else None
                print(f'=== v2 A5 graph: d={d}, use_pre_mix={use_pre_mix} ===')
                compare_y, compare_post, compare_comb_frag, compare_pre = run_hc_pre_v2_case(
                    x, hc_fn, hc_scale, hc_base, cur_pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps,
                    hc_eps, graph_mode=True)
                assert(compare_y[0] == "Pass")
                assert(compare_post[0] == "Pass")
                assert(compare_comb_frag[0] == "Pass")
                assert(compare_pre[0] == "Pass")

    def test_hc_pre_v2_ascend950_graph_aclgraph(self):
        """Ascend950(A5): v2 aclgraph 模式（npugraph_ex 后端），验证 converter 在 aclgraph 路径下的成图。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This aclgraph case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        for d in [4096, 5120, 7168]:
            shape = (bs := 192, hc_mult, d)
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=d)
            pre_mix = torch.tensor(np.random.uniform(0.5, 1.5, shape[:-1])).to(torch.float32)
            for use_pre_mix in [True, False]:
                cur_pre_mix = pre_mix if use_pre_mix else None

                class Network(nn.Module):
                    def forward(self, x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
                        npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out = torch.ops.custom.npu_hc_pre_v2(
                            x, hc_fn, hc_scale, hc_base, pre_mix, hc_mult=hc_mult,
                            hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps
                        )
                        return npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out

                npu_mode = Network().to("npu:%s" % DEVICE_ID)
                npu_mode = torch.compile(npu_mode, fullgraph=True, backend="npugraph_ex", dynamic=False)
                print(f'=== v2 A5 aclgraph: d={d}, use_pre_mix={use_pre_mix} ===')
                npu_y_out, npu_post_out, npu_comb_frag_out, npu_pre_out = npu_mode(
                    x.to("npu:%s" % DEVICE_ID), hc_fn.to("npu:%s" % DEVICE_ID),
                    hc_scale.to("npu:%s" % DEVICE_ID), hc_base.to("npu:%s" % DEVICE_ID),
                    cur_pre_mix.to("npu:%s" % DEVICE_ID) if cur_pre_mix is not None else None,
                    hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                golden_y, golden_post, golden_comb, golden_pre = _hc_pre_v2(
                    x, hc_fn, hc_scale, hc_base, cur_pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                compare_y = data_compare(golden_y.cpu().float().numpy(), npu_y_out.cpu().float().numpy(),
                                         diff_thd=0.004, pct_thd=0.02)
                compare_post = data_compare(golden_post.cpu().numpy(), npu_post_out.cpu().float().numpy(),
                                            diff_thd=0.0001, pct_thd=0.005)
                compare_comb_frag = data_compare(golden_comb.cpu().numpy(), npu_comb_frag_out.cpu().float().numpy(),
                                                 diff_thd=0.0001, pct_thd=0.005)
                compare_pre = data_compare(golden_pre.cpu().numpy(), npu_pre_out.cpu().float().numpy(),
                                           diff_thd=0.0001, pct_thd=0.005)
                assert(compare_y[0] == "Pass")
                assert(compare_post[0] == "Pass")
                assert(compare_comb_frag[0] == "Pass")
                assert(compare_pre[0] == "Pass")

    def test_hc_pre_v2_generalized_d_eager(self):
        """v2 泛化 d（所有芯片通用）：d=4096/5120/7168，bs 随机+固定值，pre_mix 传/不传。"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 5120, 7168]

        rng = random.Random(2025)
        bs_candidates = []
        for _ in range(8):
            bs_candidates.append(rng.randint(1, 32768))
        bs_candidates += [1, 128, 1024, 4096, 32768]
        seen = set()
        bs_list = []
        for v in bs_candidates:
            if v not in seen:
                seen.add(v)
                bs_list.append(v)

        import gc
        fails = []
        for d in d_list:
            for bs in bs_list:
                seed = d * 100 + bs
                shape = (bs, hc_mult, d)
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=seed)
                # pre_mix 故意取与公式 pre 不同的随机值，确保 y 确实由 pre_mix 计算而非 pre
                pre_mix = torch.tensor(np.random.uniform(0.5, 1.5, shape[:-1])).to(torch.float32)
                for use_pre_mix in [True, False]:
                    cur_pre_mix = pre_mix if use_pre_mix else None
                    print(f'=== v2 generalized D eager: d={d}, bs={bs}, use_pre_mix={use_pre_mix} ===', flush=True)
                    cy, cp, cc, cpre = run_hc_pre_v2_case(
                        x, hc_fn, hc_scale, hc_base, cur_pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                    ok = all(c[0] == "Pass" for c in (cy, cp, cc, cpre))
                    if not ok:
                        fails.append(f"D-gen-d{d}-bs{bs}-premix{use_pre_mix}: "
                                     f"y{cy[1]:.2f}/post{cp[1]:.2f}/comb{cc[1]:.2f}/pre{cpre[1]:.2f}")
                del x, hc_fn, hc_scale, hc_base, pre_mix
                gc.collect()
                torch_npu.npu.empty_cache()
        if fails:
            self.fail(f"{len(fails)} v2 generalized D case(s) failed: {'; '.join(fails)}")

    def test_hc_pre_v2_generalized_bs_eager(self):
        """v2 泛化 bs（Ascend950）：覆盖 fused/composite 两条路由，d=4096/5120/7168，pre_mix 传/不传。"""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This generalized-bs case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        bs_list = [
            # fused: bs <= 512
            1, 2, 7, 16, 31, 64, 128, 256, 511, 512,
            # composite: 512 < bs, bs % 8192 != 0
            513, 768, 1000, 1536, 3079, 4096, 6000, 8193, 10000, 12288, 20480, 30000,
            # fused: bs % 8192 == 0
            8192, 16384, 24576, 32768,
        ]
        import gc
        fails = []
        for d in [4096, 5120, 7168]:
            for bs in bs_list:
                seed = d * 131 + bs
                shape = (bs, hc_mult, d)
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(shape, hc_mix, hc_mult, seed=seed)
                route = "fused" if (bs <= 512 or bs % 8192 == 0) else "composite"
                # pre_mix 故意取与公式 pre 不同的随机值，确保 y 确实由 pre_mix 计算而非 pre
                pre_mix = torch.tensor(np.random.uniform(0.5, 1.5, shape[:-1])).to(torch.float32)
                for use_pre_mix in [True, False]:
                    cur_pre_mix = pre_mix if use_pre_mix else None
                    cy, cp, cc, cpre = run_hc_pre_v2_case(
                        x, hc_fn, hc_scale, hc_base, cur_pre_mix, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
                    ok = all(c[0] == "Pass" for c in (cy, cp, cc, cpre))
                    print(f'=== v2 generalized bs: d={d} bs={bs:>6} [{route:9}] premix={use_pre_mix} '
                          f'y={cy[1]:.3f}% post={cp[1]:.3f}% comb={cc[1]:.3f}% pre={cpre[1]:.3f}% '
                          f'-> {"PASS" if ok else "FAIL"}', flush=True)
                    if not ok:
                        fails.append(f"bs-gen-d{d}-bs{bs}-{route}-premix{use_pre_mix}: "
                                     f"y{cy[1]:.2f}/post{cp[1]:.2f}/comb{cc[1]:.2f}/pre{cpre[1]:.2f}")
                del x, hc_fn, hc_scale, hc_base, pre_mix
                gc.collect()
                torch_npu.npu.empty_cache()
        if fails:
            self.fail(f"{len(fails)} v2 generalized bs case(s) failed: {'; '.join(fails)}")


if __name__ == "__main__":
    run_tests()
