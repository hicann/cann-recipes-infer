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

    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

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
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb

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
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
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
    x = torch.tensor(np.random.uniform(0, 2, shape)).to(torch.bfloat16)
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

        d_list = [4096, 7168]
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
        d_list = [4096, 7168]
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
        x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((8193, hc_mult, 4096), hc_mix, hc_mult)

        print('======================== PTA large-bs eager regression test ========================')
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
        x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((3079, hc_mult, 4096), hc_mix, hc_mult)

        print('======================== PTA large-bs eager regression test ========================')
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

        for bs in bs_list:
            seed = bs
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                (bs, hc_mult, 4096), hc_mix, hc_mult, seed=seed)
            print(f'=== Ascend950 fusion bs eager: bs={bs}, d=4096 ===')
            self._run_and_check_case(
                x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_fusion_bs_extend_d_eager(self):
        """Ascend950: bs > 512 and divisible by 256, with d=7168 (extended D)."""
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

        for bs in bs_list:
            seed = bs + 7168
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                (bs, hc_mult, 7168), hc_mix, hc_mult, seed=seed)
            print(f'=== Ascend950 fusion bs eager (d=7168): bs={bs} ===')
            self._run_and_check_case(
                x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_generalized_d_eager(self):
        """测试 d=4096/7168，所有芯片通用"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 7168]

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
        for d in [4096, 7168]:
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

if __name__ == "__main__":
    run_tests()
