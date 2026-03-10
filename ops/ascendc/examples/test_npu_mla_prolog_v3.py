# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math, copy
import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))
torch.npu.config.allow_internal_format = True


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.concatenate((-x2, x1), dim=-1)


def dynamic_quant(inputdata, smooth_scale=None):
    if len(inputdata.shape) == 3:
        inputdata = inputdata.reshape(-1, inputdata.size(-1))
    T, H = inputdata.shape
    y = torch.zeros(T, H).to(torch.int32)
    scale = torch.zeros(T).to(torch.float32)

    inputdata = inputdata.reshape(T, H).to(torch.float32)
    if smooth_scale != None:
        if len(smooth_scale.shape) != 2 or smooth_scale.shape[1] != H:
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputdata[bs_index, :] * smooth_scale[0, :])
            scale_bs = abs_bs_tensor.max() / 127
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(
                inputdata[bs_index:] * smooth_scale[0, :] / scale_bs
            )
    else:
        for bs_index in range(T):
            abs_bs_tensor = torch.abs(inputdata[bs_index, :])
            scale_bs = abs_bs_tensor.max() / 127
            scale[bs_index] = scale_bs
            y[bs_index:] = torch.round(inputdata[bs_index:] / scale_bs)
    return y, scale


def dynamic_quant_ckv_with_amax(inputdata, amax, smooth_scale=None):
    T, N, H = inputdata.shape
    inputdata = inputdata.to(torch.float32)
    amax = amax.to(torch.float32).clamp(min=1e-8)

    if smooth_scale is None:
        scaled_inputdata = inputdata
    else:
        if len(smooth_scale.shape) != 3 or smooth_scale.shape[1:] != (N, H):
            return None, None
        smooth_scale = smooth_scale.to(torch.float32)
        scaled_inputdata = inputdata * smooth_scale

    scale = amax / 127.0
    y = torch.round(scaled_inputdata / scale)
    return y, scale


def cpu_mla_prolog_v3(
    token_x,
    weight_dq,
    weight_uq_qr,
    weight_uk,
    weight_dkv_kr,
    rmsnorm_gamma_cq,
    rmsnorm_gamma_ckv,
    rope_sin,
    rope_cos,
    kv_cache,
    kr_cache,
    cache_index,
    rmsnorm_epsilon_cq,
    rmsnorm_epsilon_ckv,
    cache_mode="PA_BSND",
    dequant_scale_x=None,
    dequant_scale_w_dq=None,
    dequant_scale_w_uq_qr=None,
    dequant_scale_w_dkvkr=None,
    quant_scale_ckv=None,
    smooth_scale_cq=None,
    query_norm_flag=True,
    weight_quant_mode=1,
    kv_cache_quant_mode=3,
    query_quant_mode=0,
    ckvkr_repo_mode=1,
    quant_scale_repo_mode=1,
    tile_size=128,
    k_nope_clip_alpha=None,
    qc_qr_scale=1,
    kc_scale=1,
    mla_param=None,
):

    token_x = token_x.cpu()
    weight_dq = weight_dq.cpu()
    weight_uq_qr = weight_uq_qr.cpu()
    weight_dkv_kr = weight_dkv_kr.cpu()
    weight_uk = weight_uk.cpu()
    rmsnorm_gamma_cq = rmsnorm_gamma_cq.cpu()
    rmsnorm_gamma_ckv = rmsnorm_gamma_ckv.cpu()
    rope_sin = rope_sin.cpu()
    rope_cos = rope_cos.cpu()
    cache_index = cache_index.cpu()
    kv_cache = kv_cache.cpu()
    kr_cache = kr_cache.cpu()
    dequant_scale_w_uq_qr = dequant_scale_w_uq_qr.cpu()
    if k_nope_clip_alpha is not None:
        k_nope_clip_alpha = k_nope_clip_alpha.cpu()

    B = mla_param["B"]
    S1 = mla_param["S1"]
    S2 = mla_param["S2"]
    D = mla_param["D"]
    Dr = mla_param["Dr"]
    N1 = mla_param["N1"]
    N2 = mla_param["N2"]
    He = mla_param["He"]
    Hckv = mla_param["Hckv"]
    Hcq = mla_param["Hcq"]
    BlockNum = mla_param["BlockNum"]
    BlockSize = mla_param["BlockSize"]
    T = mla_param["T"]
    index_table = cache_index

    cos = rope_cos.to(torch.float32)
    sin = rope_sin.to(torch.float32)
    if cache_mode in ["PA_BSND", "PA_NZ"]:
        index_table = index_table.reshape(T)

    if not mla_param["t_flag"]:
        T = B * S1
        token_x = token_x.reshape(T, He).to(torch.float32)
        cos = cos.reshape(T, Dr)
        sin = sin.reshape(T, Dr)
        if cache_mode in ["PA_BSND", "PA_NZ"]:
            index_table = index_table.reshape(T)

    # matmul1 : token_x(B*S1,He) * w_dq (He,Hcq) -> matmul1_res(B*S1,Hcq)
    w_dq = weight_dq.to(torch.float32)
    token_x = token_x.to(torch.float32)
    matmul1_res = torch.matmul(token_x, w_dq).to(torch.float32)
    matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)

    # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
    ep1 = float(rmsnorm_epsilon_cq)
    gamma1 = rmsnorm_gamma_cq.to(torch.float32)
    norm1_res = matmul1_res / torch.sqrt(
        torch.mean(matmul1_res**2, dim=-1, keepdim=True) + ep1
    )
    norm1_res *= gamma1

    # matmul2 预处理
    w_uq_qr = weight_uq_qr.to(torch.int32)
    norm1_res, dequant_scale_qcqr = dynamic_quant(norm1_res, smooth_scale_cq)
    matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(torch.int32)
    out_qnorm = None
    out_deq_qnorm = None
    if query_norm_flag:
        out_qnorm = norm1_res
        out_deq_qnorm = dequant_scale_qcqr

    # matmul2 后处理
    matmul2_res = matmul2_res.to(torch.float32)
    matmul2_res = matmul2_res * dequant_scale_qcqr.unsqueeze(-1)
    matmul2_res = matmul2_res * dequant_scale_w_uq_qr

    matmul2_res = matmul2_res.reshape(T, N1, D + Dr)

    # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
    splitd1_res1 = matmul2_res[:, :, :D]  # 取前 D 维度
    splitd1_res2 = matmul2_res[:, :, D:]  # 取剩余的 Dr 维度

    # matmul3 : -> splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> query_mla(B,S1,N,Hckv)
    w_uk = weight_uk.to(torch.float32)
    splitd1_res1 = splitd1_res1.transpose(0, 1)
    splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
    query_mla = torch.zeros((N1, T, Hckv))
    query_mla = torch.matmul(splitd1_res1, w_uk).to(torch.float32)
    query_mla = query_mla.transpose(0, 1)
    query_mla = query_mla if mla_param["t_flag"] else query_mla.reshape(B, S1, N1, Hckv)
    query_mla = query_mla.to(torch.bfloat16).to(torch.float32)

    # rotary1 : -> splitd1_res2(B*S1,N,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> query_rope_mla(B,S1,N,Dr)
    expanded_cos = cos.unsqueeze(1).repeat(1, N1, 1)
    expanded_sin = sin.unsqueeze(1).repeat(1, N1, 1)
    q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
    query_rope_mla = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
    query_rope_mla = (
        query_rope_mla if mla_param["t_flag"] else query_rope_mla.reshape(B, S1, N1, Dr)
    )
    query_rope_mla = query_rope_mla.to(torch.bfloat16).to(torch.float32)

    # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
    w_kv_kr = weight_dkv_kr.to(torch.float32)
    matmul4_res = torch.matmul(token_x, w_kv_kr).to(torch.float32)

    # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
    splitd2_res1 = matmul4_res[:, :Hckv]  # 取前 Hckv 维度
    splitd2_res2 = matmul4_res[:, Hckv:]  # 取剩余的 Dr 维度

    # rotary2 : splitd2_res2(B*S1,Dr) * cos(B*S1,Dr) * sin(B*S1,Dr) -> rotary2_res(B*S1,Dr)
    k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
    rotary2_res = (k * cos) + (rotate_half(k) * sin)

    # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
    ep2 = float(rmsnorm_epsilon_ckv)
    gamma2 = rmsnorm_gamma_ckv
    norm2_res = splitd2_res1 / torch.sqrt(
        torch.mean(splitd2_res1**2, dim=-1, keepdim=True) + ep2
    )
    norm2_res *= gamma2

    # rmsnorm2 后处理
    Dtile = Hckv
    if kv_cache_quant_mode == 3:
        norm2_res = norm2_res.reshape(T, Hckv // tile_size, tile_size)
        eps = 1e-8
        amax = torch.max(torch.abs(norm2_res), dim=-1, keepdim=True)[0]
        amax = torch.clamp(amax, min=eps) * k_nope_clip_alpha
        clip_res = torch.clamp(norm2_res, min=-amax, max=amax)
        norm2_res, deq_scale_ckv = dynamic_quant_ckv_with_amax(clip_res, amax)
        deq_scale_ckv = deq_scale_ckv.reshape(T, -1)
        norm2_res = norm2_res.reshape(T, Hckv).to(torch.int8)
        rotary2_res_bf16 = rotary2_res.to(torch.bfloat16)
        if ckvkr_repo_mode == 1:
            norm2_res = torch.cat(
                (norm2_res, rotary2_res_bf16.view(torch.int8)), dim=-1
            )
            Dtile = Dtile + Dr * 2
        if quant_scale_repo_mode == 1:
            norm2_res = torch.cat((norm2_res, deq_scale_ckv.view(torch.int8)), dim=-1)
            Dtile = Dtile + Hckv // tile_size * 4

    # scatter1 : norm2_res(B*S1,Hckv) * kv_cache(B,N2,S2,Hckv/B,B,N2,Hckv) -> kv_cache_out_mla(B,N2,S2,Hckv/B,B,N2,Hckv)
    kv_cache = copy.deepcopy(kv_cache)
    kv_cache_out_mla_shape = kv_cache.shape
    kv_cache = kv_cache.reshape(BlockNum * BlockSize, N2, Dtile)
    for i in range(T):
        for j in range(N2):
            if cache_mode in ["PA_BSND", "PA_NZ"]:
                kv_cache[index_table[i], j, :] = norm2_res[i, :]
            else:
                kv_cache[i, j, :] = norm2_res[i, :]
    kv_cache_out_mla = kv_cache.reshape(kv_cache_out_mla_shape)

    # scatter2 : rotary2_res(B*S1,Dr) * kr_cache(B,N2,S2,Dr/B,B,N2,Dr) -> kr_cache_out_mla(B,N2,S2,Dr/B,B,N2,Dr)
    kr_cache = copy.deepcopy(kr_cache)
    kr_cache_out_mla_shape = kr_cache.shape
    kr_cache = kr_cache.reshape(BlockNum * BlockSize, N2, Dr)
    for i in range(T):
        for j in range(N2):
            if cache_mode in ["PA_BSND", "PA_NZ"]:
                kr_cache[index_table[i], j, :] = rotary2_res[i, :]
            else:
                kr_cache[i, j, :] = rotary2_res[i, :]

    kr_cache_out_mla = kr_cache.reshape(kr_cache_out_mla_shape)

    return (
        query_mla,
        query_rope_mla,
        out_qnorm,
        out_deq_qnorm,
        kv_cache_out_mla,
        kr_cache_out_mla,
    )


def compare(cpu_out, npu_out, info):
    res = np.isclose(cpu_out, npu_out, rtol=0.005, atol=0.0001, equal_nan=False)
    true_ratio = np.mean(res)
    if true_ratio < 0.99:
        print(f"{info['name']}_cpu output:\n{cpu_out.shape}\n{cpu_out}")
        print(f"{info['name']}_npu output:\n{npu_out.shape}\n{npu_out}")
        print(
            f"{info['mode']}-{info['name']}:correct ratio of cpu vs npu is:",
            true_ratio * 100,
            "%",
        )
        return False
    return True


class TestCustomMlaPrologV3(TestCase):
    def test_mla_prolog_v3_pertile_eager(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 128
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        ckvkr_repo_mode = 1
        quant_scale_repo_mode = 1
        tile_size = 128
        Dtile = (
            Hckv
            + quant_scale_repo_mode * (Hckv // tile_size) * 4
            + ckvkr_repo_mode * Dr * 2
        )

        token_x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(1, 2, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(T).to(torch.int64).npu()
        kv_cache = torch.randint(
            -1, 1, (1, BlockNum * BlockSize * Nkv * Dtile), dtype=torch.int8
        ).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Dtile)
        kv_cache_input = kv_cache.clone().detach()
        kr_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16
        ).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        kr_cache_input = kr_cache.clone().detach()
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        dequant_scale_w_uq_qr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()
        query_norm_flag = True
        weight_quant_mode = 1
        kv_cache_quant_mode = 3
        query_quant_mode = 0
        k_nope_clip_alpha = torch.tensor([1], dtype=torch.float32).npu()
        qc_qr_scale = 1
        kc_scale = 1

        mla_param = {
            "B": B,
            "He": He,
            "Hcq": Hcq,
            "Hckv": Hckv,
            "N1": N,
            "D": D,
            "Dr": Dr,
            "S2": Skv,
            "S1": S,
            "N2": Nkv,
            "BlockNum": BlockNum,
            "BlockSize": BlockSize,
            "t_flag": True,
            "T": T,
        }

        print(f"======================== PTA eager BEGIN ========================")
        # start run custom ops
        query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
            torch.ops.custom.npu_mla_prolog_v3(
                token_x,
                w_dq_cast,
                w_uq_qr_cast,
                w_uk,
                w_dkv_kr_cast,
                rmsnorm_gamma_cq,
                rmsnorm_gamma_ckv,
                rope_sin,
                rope_cos,
                kv_cache,
                kr_cache,
                cache_index=cache_index,
                rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
                rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
                cache_mode=cache_mode,
                dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
                query_norm_flag=query_norm_flag,
                weight_quant_mode=weight_quant_mode,
                kv_cache_quant_mode=kv_cache_quant_mode,
                query_quant_mode=query_quant_mode,
                ckvkr_repo_mode=ckvkr_repo_mode,
                quant_scale_repo_mode=quant_scale_repo_mode,
                tile_size=tile_size,
                k_nope_clip_alpha=k_nope_clip_alpha,
                qc_qr_scale=qc_qr_scale,
                kc_scale=kc_scale,
            )
        )

        # compare result
        (
            query_cpu,
            query_rope_cpu,
            query_norm_cpu,
            dequant_scale_q_norm_cpu,
            kv_cache_output,
            kr_cache_output,
        ) = cpu_mla_prolog_v3(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache_input,
            kr_cache_input,
            cache_index,
            rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            k_nope_clip_alpha=k_nope_clip_alpha,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            mla_param=mla_param,
        )

        query = query.cpu().to(torch.float32).numpy()
        query_rope = query_rope.cpu().to(torch.float32).numpy()
        query_norm = query_norm.cpu().to(torch.float32).numpy()
        dequant_scale_q_norm = dequant_scale_q_norm.cpu().to(torch.float32).numpy()
        kv_cache = kv_cache.cpu().to(torch.float32).numpy()
        kr_cache = kr_cache.cpu().to(torch.float32).numpy()
        kv_cache_output = kv_cache_output.cpu().to(torch.float32).numpy()
        kr_cache_output = kr_cache_output.cpu().to(torch.float32).numpy()

        mode = "pertile-eager"
        self.assertTrue(
            compare(query_cpu, query, {"mode": mode, "name": "query"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_rope_cpu, query_rope, {"mode": mode, "name": "query_rope"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_norm_cpu, query_norm, {"mode": mode, "name": "query_norm"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(
                dequant_scale_q_norm_cpu,
                dequant_scale_q_norm,
                {"mode": mode, "name": "dequant_scale_q_norm"},
            ),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kv_cache_output, kv_cache, {"mode": mode, "name": "kv_cache"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kr_cache_output, kr_cache, {"mode": mode, "name": "kr_cache"}),
            "precision compare fail",
        )
        print(f"======================== PTA eager FINISH ========================")

    def test_mla_prolog_v3_pertile_graph(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 128
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8
        ckvkr_repo_mode = 1
        quant_scale_repo_mode = 1
        tile_size = 128
        Dtile = (
            Hckv
            + quant_scale_repo_mode * (Hckv // tile_size) * 4
            + ckvkr_repo_mode * Dr * 2
        )

        token_x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(1, 2, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(T).to(torch.int64).npu()
        kv_cache = torch.randint(
            -1, 1, (1, BlockNum * BlockSize * Nkv * Dtile), dtype=torch.int8
        ).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Dtile)
        kv_cache_input = kv_cache.clone().detach()
        kr_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16
        ).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        kr_cache_input = kr_cache.clone().detach()
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        dequant_scale_w_uq_qr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()
        query_norm_flag = True
        weight_quant_mode = 1
        kv_cache_quant_mode = 3
        query_quant_mode = 0
        k_nope_clip_alpha = torch.tensor([1], dtype=torch.float32).npu()
        qc_qr_scale = 1
        kc_scale = 1

        mla_param = {
            "B": B,
            "He": He,
            "Hcq": Hcq,
            "Hckv": Hckv,
            "N1": N,
            "D": D,
            "Dr": Dr,
            "S2": Skv,
            "S1": S,
            "N2": Nkv,
            "BlockNum": BlockNum,
            "BlockSize": BlockSize,
            "t_flag": True,
            "T": T,
        }

        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        class NetworkV3(nn.Module):
            def __init__(self):
                super(NetworkV3, self).__init__()

            def forward(
                self,
                x,
                w_dq,
                w_uq_qr,
                w_uk,
                w_dkv_kr,
                gamma_cq,
                gamma_ckv,
                sin,
                cos,
                kv_cache,
                kr_cache,
                cache_index=None,
                k_nope_clip_alpha=None,
                rmsnorm_epsilon_cq=1e-5,
                rmsnorm_epsilon_ckv=1e-5,
                cache_mode="PA_BSND",
                dequant_scale_x=None,
                dequant_scale_w_dq=None,
                dequant_scale_w_uq_qr=None,
                dequant_scale_w_dkvkr=None,
                quant_scale_ckv=None,
                smooth_scale_cq=None,
                query_norm_flag=True,
                weight_quant_mode=1,
                kv_cache_quant_mode=3,
                query_quant_mode=0,
                ckvkr_repo_mode=1,
                quant_scale_repo_mode=1,
                tile_size=128,
                qc_qr_scale=1,
                kc_scale=1,
            ):

                out0, out1, out2, out3, out4 = torch.ops.custom.npu_mla_prolog_v3(
                    x,
                    w_dq,
                    w_uq_qr,
                    w_uk,
                    w_dkv_kr,
                    gamma_cq,
                    gamma_ckv,
                    sin,
                    cos,
                    kv_cache,
                    kr_cache,
                    cache_index=cache_index,
                    k_nope_clip_alpha=k_nope_clip_alpha,
                    rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
                    rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
                    cache_mode=cache_mode,
                    dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
                    query_norm_flag=query_norm_flag,
                    weight_quant_mode=weight_quant_mode,
                    kv_cache_quant_mode=kv_cache_quant_mode,
                    query_quant_mode=query_quant_mode,
                    ckvkr_repo_mode=ckvkr_repo_mode,
                    quant_scale_repo_mode=quant_scale_repo_mode,
                    tile_size=tile_size,
                    qc_qr_scale=qc_qr_scale,
                    kc_scale=kc_scale,
                )
                return out0, out1, out2, out3, out4

        mod = torch.compile(NetworkV3().npu(), backend=npu_backend, fullgraph=True)
        print(f"======================== PTA graph BEGIN ========================")
        # calculate on npu compile
        query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = mod(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache,
            kr_cache,
            cache_index=cache_index,
            k_nope_clip_alpha=k_nope_clip_alpha,
            rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
        )

        # compare result
        (
            query_cpu,
            query_rope_cpu,
            query_norm_cpu,
            dequant_scale_q_norm_cpu,
            kv_cache_output,
            kr_cache_output,
        ) = cpu_mla_prolog_v3(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache_input,
            kr_cache_input,
            cache_index,
            rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            k_nope_clip_alpha=k_nope_clip_alpha,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            mla_param=mla_param,
        )

        query = query.cpu().to(torch.float32).numpy()
        query_rope = query_rope.cpu().to(torch.float32).numpy()
        query_norm = query_norm.cpu().to(torch.float32).numpy()
        dequant_scale_q_norm = dequant_scale_q_norm.cpu().to(torch.float32).numpy()
        kv_cache = kv_cache.cpu().to(torch.float32).numpy()
        kr_cache = kr_cache.cpu().to(torch.float32).numpy()
        kv_cache_output = kv_cache_output.cpu().to(torch.float32).numpy()
        kr_cache_output = kr_cache_output.cpu().to(torch.float32).numpy()

        mode = "pertile-graph"
        self.assertTrue(
            compare(query_cpu, query, {"mode": mode, "name": "query"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_rope_cpu, query_rope, {"mode": mode, "name": "query_rope"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_norm_cpu, query_norm, {"mode": mode, "name": "query_norm"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(
                dequant_scale_q_norm_cpu,
                dequant_scale_q_norm,
                {"mode": mode, "name": "dequant_scale_q_norm"},
            ),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kv_cache_output, kv_cache, {"mode": mode, "name": "kv_cache"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kr_cache_output, kr_cache, {"mode": mode, "name": "kr_cache"}),
            "precision compare fail",
        )
        print(f"======================== PTA graph FINISH ========================")

    def test_mla_prolog_v3_eager(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 128
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8

        token_x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(1, 2, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(T).to(torch.int64).npu()
        kv_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Hckv, dtype=torch.bfloat16
        ).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kv_cache_input = kv_cache.clone().detach()
        kr_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16
        ).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        kr_cache_input = kr_cache.clone().detach()
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        dequant_scale_w_uq_qr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()
        query_norm_flag = True
        weight_quant_mode = 1
        kv_cache_quant_mode = 0
        query_quant_mode = 0
        ckvkr_repo_mode = 0
        quant_scale_repo_mode = 0
        tile_size = 128
        k_nope_clip_alpha = None
        qc_qr_scale = 1
        kc_scale = 1

        mla_param = {
            "B": B,
            "He": He,
            "Hcq": Hcq,
            "Hckv": Hckv,
            "N1": N,
            "D": D,
            "Dr": Dr,
            "S2": Skv,
            "S1": S,
            "N2": Nkv,
            "BlockNum": BlockNum,
            "BlockSize": BlockSize,
            "t_flag": True,
            "T": T,
        }

        print(f"======================== PTA eager BEGIN ========================")
        # start run custom ops
        query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
            torch.ops.custom.npu_mla_prolog_v3(
                token_x,
                w_dq_cast,
                w_uq_qr_cast,
                w_uk,
                w_dkv_kr_cast,
                rmsnorm_gamma_cq,
                rmsnorm_gamma_ckv,
                rope_sin,
                rope_cos,
                kv_cache,
                kr_cache,
                cache_index=cache_index,
                k_nope_clip_alpha=k_nope_clip_alpha,
                rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
                rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
                cache_mode=cache_mode,
                dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
                query_norm_flag=query_norm_flag,
                weight_quant_mode=weight_quant_mode,
                kv_cache_quant_mode=kv_cache_quant_mode,
                query_quant_mode=query_quant_mode,
                ckvkr_repo_mode=ckvkr_repo_mode,
                quant_scale_repo_mode=quant_scale_repo_mode,
                tile_size=tile_size,
                qc_qr_scale=qc_qr_scale,
                kc_scale=kc_scale,
            )
        )

        # compare result
        (
            query_cpu,
            query_rope_cpu,
            query_norm_cpu,
            dequant_scale_q_norm_cpu,
            kv_cache_output,
            kr_cache_output,
        ) = cpu_mla_prolog_v3(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache_input,
            kr_cache_input,
            cache_index,
            rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            k_nope_clip_alpha=k_nope_clip_alpha,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            mla_param=mla_param,
        )

        query = query.cpu().to(torch.float32).numpy()
        query_rope = query_rope.cpu().to(torch.float32).numpy()
        query_norm = query_norm.cpu().to(torch.float32).numpy()
        dequant_scale_q_norm = dequant_scale_q_norm.cpu().to(torch.float32).numpy()
        kv_cache = kv_cache.cpu().to(torch.float32).numpy()
        kr_cache = kr_cache.cpu().to(torch.float32).numpy()
        kv_cache_output = kv_cache_output.cpu().to(torch.float32).numpy()
        kr_cache_output = kr_cache_output.cpu().to(torch.float32).numpy()

        mode = "eager"
        self.assertTrue(
            compare(query_cpu, query, {"mode": mode, "name": "query"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_rope_cpu, query_rope, {"mode": mode, "name": "query_rope"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_norm_cpu, query_norm, {"mode": mode, "name": "query_norm"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(
                dequant_scale_q_norm_cpu,
                dequant_scale_q_norm,
                {"mode": mode, "name": "dequant_scale_q_norm"},
            ),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kv_cache_output, kv_cache, {"mode": mode, "name": "kv_cache"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kr_cache_output, kr_cache, {"mode": mode, "name": "kr_cache"}),
            "precision compare fail",
        )
        print(f"======================== PTA eager FINISH ========================")

    def test_mla_prolog_v3_graph(self):
        B = 8
        He = 7168
        Hcq = 1536
        Hckv = 512
        N = 128
        D = 128
        Dr = 64
        Skv = 1024
        S = 1
        Nkv = 1
        BlockSize = 128
        BlockNum = math.ceil(B * Skv / BlockSize)
        T = 8

        token_x = torch.rand(T, He, dtype=torch.bfloat16).npu()
        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
        w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
        w_uq_qr = torch.randint(1, 2, (Hcq, N * (D + Dr)), dtype=torch.int8).npu()
        w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
        w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
        rope_sin = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        rope_cos = torch.rand(T, Dr, dtype=torch.bfloat16).npu()
        cache_index = torch.rand(T).to(torch.int64).npu()
        kv_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Hckv, dtype=torch.bfloat16
        ).npu()
        kv_cache = kv_cache.view(BlockNum, BlockSize, Nkv, Hckv)
        kv_cache_input = kv_cache.clone().detach()
        kr_cache = torch.rand(
            1, BlockNum * BlockSize * Nkv * Dr, dtype=torch.bfloat16
        ).npu()
        kr_cache = kr_cache.view(BlockNum, BlockSize, Nkv, Dr)
        kr_cache_input = kr_cache.clone().detach()
        rmsnorm_epsilon_cq = 1.0e-5
        rmsnorm_epsilon_ckv = 1.0e-5
        cache_mode = "PA_BSND"
        dequant_scale_w_uq_qr = torch.rand(1, N * (D + Dr), dtype=torch.float32).npu()
        smooth_scale_cq = torch.ones(1, Hcq, dtype=torch.float32).npu()
        query_norm_flag = True
        weight_quant_mode = 1
        kv_cache_quant_mode = 0
        query_quant_mode = 0
        ckvkr_repo_mode = 0
        quant_scale_repo_mode = 0
        tile_size = 128
        k_nope_clip_alpha = None
        qc_qr_scale = 1
        kc_scale = 1

        mla_param = {
            "B": B,
            "He": He,
            "Hcq": Hcq,
            "Hckv": Hckv,
            "N1": N,
            "D": D,
            "Dr": Dr,
            "S2": Skv,
            "S1": S,
            "N2": Nkv,
            "BlockNum": BlockNum,
            "BlockSize": BlockSize,
            "t_flag": True,
            "T": T,
        }

        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        class NetworkV3(nn.Module):
            def __init__(self):
                super(NetworkV3, self).__init__()

            def forward(
                self,
                x,
                w_dq,
                w_uq_qr,
                w_uk,
                w_dkv_kr,
                gamma_cq,
                gamma_ckv,
                sin,
                cos,
                kv_cache,
                kr_cache,
                cache_index=None,
                rmsnorm_epsilon_cq=1e-5,
                rmsnorm_epsilon_ckv=1e-5,
                cache_mode="PA_BSND",
                dequant_scale_x=None,
                dequant_scale_w_dq=None,
                dequant_scale_w_uq_qr=None,
                dequant_scale_w_dkvkr=None,
                quant_scale_ckv=None,
                smooth_scale_cq=None,
                k_nope_clip_alpha=None,
                query_norm_flag=True,
                weight_quant_mode=1,
                kv_cache_quant_mode=3,
                query_quant_mode=0,
                ckvkr_repo_mode=1,
                quant_scale_repo_mode=1,
                tile_size=128,
                qc_qr_scale=1,
                kc_scale=1,
            ):

                out0, out1, out2, out3, out4 = torch.ops.custom.npu_mla_prolog_v3(
                    x,
                    w_dq,
                    w_uq_qr,
                    w_uk,
                    w_dkv_kr,
                    gamma_cq,
                    gamma_ckv,
                    sin,
                    cos,
                    kv_cache,
                    kr_cache,
                    cache_index=cache_index,
                    rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
                    rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
                    cache_mode=cache_mode,
                    dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
                    k_nope_clip_alpha=k_nope_clip_alpha,
                    query_norm_flag=query_norm_flag,
                    weight_quant_mode=weight_quant_mode,
                    kv_cache_quant_mode=kv_cache_quant_mode,
                    query_quant_mode=query_quant_mode,
                    ckvkr_repo_mode=ckvkr_repo_mode,
                    quant_scale_repo_mode=quant_scale_repo_mode,
                    tile_size=tile_size,
                    qc_qr_scale=qc_qr_scale,
                    kc_scale=kc_scale,
                )
                return out0, out1, out2, out3, out4

        mod = torch.compile(NetworkV3().npu(), backend=npu_backend, fullgraph=True)
        print(f"======================== PTA graph BEGIN ========================")
        # calculate on npu compile
        query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = mod(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache,
            kr_cache,
            cache_index=cache_index,
            k_nope_clip_alpha=k_nope_clip_alpha,
            rmsnorm_epsilon_cq=rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
        )

        # compare result
        (
            query_cpu,
            query_rope_cpu,
            query_norm_cpu,
            dequant_scale_q_norm_cpu,
            kv_cache_output,
            kr_cache_output,
        ) = cpu_mla_prolog_v3(
            token_x,
            w_dq_cast,
            w_uq_qr_cast,
            w_uk,
            w_dkv_kr_cast,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            kv_cache_input,
            kr_cache_input,
            cache_index,
            rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv,
            cache_mode=cache_mode,
            dequant_scale_w_uq_qr=dequant_scale_w_uq_qr,
            query_norm_flag=query_norm_flag,
            weight_quant_mode=weight_quant_mode,
            kv_cache_quant_mode=kv_cache_quant_mode,
            query_quant_mode=query_quant_mode,
            ckvkr_repo_mode=ckvkr_repo_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            qc_qr_scale=qc_qr_scale,
            kc_scale=kc_scale,
            mla_param=mla_param,
        )

        query = query.cpu().to(torch.float32).numpy()
        query_rope = query_rope.cpu().to(torch.float32).numpy()
        query_norm = query_norm.cpu().to(torch.float32).numpy()
        dequant_scale_q_norm = dequant_scale_q_norm.cpu().to(torch.float32).numpy()
        kv_cache = kv_cache.cpu().to(torch.float32).numpy()
        kr_cache = kr_cache.cpu().to(torch.float32).numpy()
        kv_cache_output = kv_cache_output.cpu().to(torch.float32).numpy()
        kr_cache_output = kr_cache_output.cpu().to(torch.float32).numpy()

        mode = "graph"
        self.assertTrue(
            compare(query_cpu, query, {"mode": mode, "name": "query"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_rope_cpu, query_rope, {"mode": mode, "name": "query_rope"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(query_norm_cpu, query_norm, {"mode": mode, "name": "query_norm"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(
                dequant_scale_q_norm_cpu,
                dequant_scale_q_norm,
                {"mode": mode, "name": "dequant_scale_q_norm"},
            ),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kv_cache_output, kv_cache, {"mode": mode, "name": "kv_cache"}),
            "precision compare fail",
        )
        self.assertTrue(
            compare(kr_cache_output, kr_cache, {"mode": mode, "name": "kr_cache"}),
            "precision compare fail",
        )
        print(f"======================== PTA graph FINISH ========================")


if __name__ == "__main__":
    run_tests()
