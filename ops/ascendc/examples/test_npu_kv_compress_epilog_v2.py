# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""KvCompressEpilogV2 最小调用样例（pytest）。

覆盖 torch eager 直调与 torchair ACL Graph 图模式（capture / 换值 replay），
并与内置 CPU 参考实现做逐字节比对。

算子契约：
  - cache: 二维 [N, kvCacheCol]，原地更新（输入即输出）；x: [T, d] BF16；
  - slot_mapping: [T] INT32/INT64，-1 表示跳过该行，越界 slot 静默跳过；
  - quant_mode: "mxfp8_bf16"（默认，cache 为 E4M3FN/E5M2）/ "mxfp4_bf16"
    （packed E2M1，cache 为 UINT8）；
  - quant_group_size: MXFP8 固定 32；MXFP4 支持 16 或 32；
  - 行布局: [量化数据区 | BF16 scale 区 | 0 填充至 32B 对齐]，
    G = d/quant_group_size, dataCol = d(mxfp8) 或 d/2(mxfp4),
    concatCol = dataCol + 2*G, kvCacheCol = RoundUp(concatCol, 32)。

前置条件：
  1. 编译并安装自定义算子 run 包（见 README.md「自定义融合算子安装」）；
  2. 编译并安装 torch_ops_extension wheel（见 README.md
     「torch_ops_extension算子包编译与安装」）；
  3. source CANN 环境与 vendor 环境：
     source <ascend-toolkit>/set_env.sh
     source <ascend-toolkit>/opp/vendors/customize/bin/set_env.bash

运行方式：
  cd <repo>/ops/ascendc/examples
  pytest test_npu_kv_compress_epilog_v2.py -v
"""

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  (注册 npu device)
import custom_ops  # noqa: F401  (注册 torch.ops.custom.* 与 torchair converter)
import torchair
from custom_ops.converter.npu_kv_compress_epilog_v2 import (
    _validate_mode_and_group_size,
)

OP = torch.ops.custom.kv_compress_epilog_v2.default
DEFAULT_GROUP = 32

FP8_DTYPE_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}
# 与 kernel 一致的 f32 预计算倒数（用乘法而非除法，避免 1 ulp 差异）
_FP8_COEFF = {
    dt: float(np.float32(1.0) / np.float32(mx))
    for dt, mx in FP8_DTYPE_MAX.items()
}


def _layout(d, quant_mode, group_size=DEFAULT_GROUP):
    """返回 (G, dataCol, concatCol, kvCacheCol)。"""
    g = d // group_size
    data_col = d if quant_mode == 2 else d // 2
    concat_col = data_col + 2 * g
    kv_col = (concat_col + 31) // 32 * 32
    return g, data_col, concat_col, kv_col


def _pow2_round_up(s):
    """f32 位运算的 2 的幂上取整（复刻 kernel roundScale）。"""
    bits = s.view(torch.int32)
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    u = e - 127 + (m != 0).to(torch.int32)
    return ((u + 127) << 23).view(torch.float32)


def _encode_mxfp8_row(x_f32_row, fp8_dtype, fp8_max, round_scale=True):
    """一行 MXFP8：返回 (data bytes[d], scale bits[G] uint16)。"""
    d = x_f32_row.numel()
    g = d // DEFAULT_GROUP
    data = torch.empty(d, dtype=torch.uint8)
    scales = torch.empty(g, dtype=torch.uint16)
    for gi in range(g):
        blk = x_f32_row[gi * DEFAULT_GROUP:(gi + 1) * DEFAULT_GROUP]
        s = torch.clamp(blk.abs().amax(), min=1e-4) * _FP8_COEFF[fp8_dtype]
        if round_scale:
            s = _pow2_round_up(s)
        q = torch.clamp(blk / s, -fp8_max, fp8_max).to(fp8_dtype)
        data[gi * DEFAULT_GROUP:(gi + 1) * DEFAULT_GROUP] = q.view(torch.uint8)
        scales[gi] = s.to(torch.bfloat16).view(torch.uint16)
    return data, scales


_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _fp4_rne_codes(q):
    """E2M1 RNE 编码 (0..15)：平局取偶码，|q|>6 饱和到 6，-0 保留符号。"""
    sign = (q.view(torch.int32) < 0).to(torch.uint8)
    a = q.abs()
    idx = torch.searchsorted(_FP4_MIDS, a).to(torch.uint8)
    tie = (a.unsqueeze(1) == _FP4_MIDS.unsqueeze(0)).any(dim=1)
    code = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    code = torch.clamp(code, max=7)
    return code | (sign << 3)


def _encode_mxfp4_row(x_bf16_row, group_size):
    """一行 packed MXFP4：返回 (packed bytes[d/2], scale bits[G] uint16)。"""
    d = x_bf16_row.numel()
    g = d // group_size
    bits = x_bf16_row.contiguous().view(torch.uint16).to(torch.int32)
    nib = torch.zeros(d, dtype=torch.uint8)
    scales = torch.zeros(g, dtype=torch.uint16)
    for gi in range(g):
        blk_bits = bits[gi * group_size:(gi + 1) * group_size]
        e_max = int(((blk_bits >> 7) & 0xFF).max())
        if e_max == 0xFF:  # Inf/NaN 组：scale=NaN，数据按 CAST_RINT 规则为 +0
            scales[gi] = 0x7F81
            continue
        if e_max == 0:  # 全零组：scale=+0，数据全 0
            scales[gi] = 0x0000
            continue
        shared_field = max(e_max, 2) - 2
        scales[gi] = shared_field << 7
        hs_bits = 0x7F00 - (shared_field << 7)
        hs_val = torch.tensor([hs_bits << 16], dtype=torch.int32).view(torch.float32)
        q = x_bf16_row[gi * group_size:(gi + 1) * group_size].float() * hs_val
        nib[gi * group_size:(gi + 1) * group_size] = _fp4_rne_codes(q)
    packed = nib[0::2] | (nib[1::2] << 4)
    return packed, scales


def kv_compress_epilog_v2_cpu(cache_u8, x_bf16, slot_np, quant_mode,
                              group_size=DEFAULT_GROUP, round_scale=True, fp8_dtype=None):
    """CPU 参考实现：返回原地更新后的 cache（uint8 视图）。

    仅覆盖有限值输入（本样例的随机输入不含 NaN/Inf）。
    mode 2 需通过 fp8_dtype 指定 cache 的 fp8 类型。
    """
    n, stride = cache_u8.shape
    t, d = x_bf16.shape
    _, data_col, concat_col, kv_col = _layout(d, quant_mode, group_size)
    fp8_max = FP8_DTYPE_MAX.get(fp8_dtype)
    out = cache_u8.clone()
    for i in range(t):
        slot = int(slot_np[i])
        if slot < 0 or slot >= n:
            continue
        if quant_mode == 2:
            data, scales = _encode_mxfp8_row(
                x_bf16[i].float(), fp8_dtype, fp8_max, round_scale)
        else:
            data, scales = _encode_mxfp4_row(x_bf16[i], group_size)
        row = out[slot]
        row[:data_col] = data
        row[data_col:concat_col] = scales.view(torch.uint8)
        row[concat_col:kv_col] = 0
    return out


def _make_inputs(quant_mode, d, t, n, seed, wide=0, slot_np=None,
                 cache_dtype=None, group_size=DEFAULT_GROUP,
                 slot_dtype=np.int32):
    """构造 NPU 输入与 CPU golden。slot 默认为无重复随机槽位且首行 -1。"""
    _, _, _, kv_col = _layout(d, quant_mode, group_size)
    col = kv_col + wide
    rng = np.random.RandomState(seed)
    if slot_np is None:
        slot_np = rng.permutation(n)[:t].astype(slot_dtype)
        slot_np[0] = -1
    cache_np = rng.uniform(1, 255, (n, col)).astype(np.uint8)
    x_bf16 = torch.from_numpy(
        rng.uniform(-1, 1, (t, d)).astype(np.float32)).to(torch.bfloat16)
    if cache_dtype is None:
        cache_dtype = torch.uint8 if quant_mode == 4 else torch.float8_e4m3fn
    golden = kv_compress_epilog_v2_cpu(
        torch.from_numpy(cache_np.copy()), x_bf16, slot_np, quant_mode,
        group_size=group_size, fp8_dtype=cache_dtype)
    cache = torch.from_numpy(cache_np.copy()).view(cache_dtype).npu()
    x = x_bf16.clone().npu()
    slot = torch.from_numpy(np.ascontiguousarray(slot_np)).npu()
    return cache, x, slot, golden


def _to_u8(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy()


def _meta_inputs(d, quant_mode_name, group_size, cache_col=None):
    quant_mode = 2 if quant_mode_name == "mxfp8_bf16" else 4
    _, _, _, required_col = _layout(d, quant_mode, group_size)
    dtype = torch.float8_e4m3fn if quant_mode == 2 else torch.uint8
    cache = torch.empty((8, cache_col or required_col), dtype=dtype, device="meta")
    x = torch.empty((4, d), dtype=torch.bfloat16, device="meta")
    slot = torch.empty((4,), dtype=torch.int32, device="meta")
    return cache, x, slot


# ==================== 公共检查 ====================

# 公共检查：cache 行宽公式（纯 CPU，不依赖 NPU）
def test_layout_formula():
    """行宽公式 sanity：d=128/mxfp4 与 d=512/mxfp8。"""
    assert _layout(128, 4, 32) == (4, 64, 72, 96)
    assert _layout(128, 4, 16) == (8, 64, 80, 96)
    assert _layout(512, 2, 32) == (16, 512, 544, 544)


# 公共检查：算子注册与 quant_mode str 版 ABI
def test_op_registered_with_str_abi():
    schema = str(OP._schema)
    assert 'str quant_mode="mxfp8_bf16"' in schema, schema
    assert "int quant_group_size=32" in schema, schema


@pytest.mark.parametrize(
    "quant_mode_name,group_size,d",
    [
        ("mxfp8_bf16", 32, 512),
        ("mxfp4_bf16", 32, 512),
        ("mxfp4_bf16", 16, 512),
    ],
)
def test_meta_accepts_supported_mode_group_combinations(quant_mode_name, group_size, d):
    cache, x, slot = _meta_inputs(d, quant_mode_name, group_size)
    OP(cache, x, slot, quant_group_size=group_size, quant_mode=quant_mode_name)


@pytest.mark.parametrize(
    "quant_mode_name,group_size",
    [
        ("mxfp8_bf16", 16),
        ("mxfp8_bf16", 64),
        ("mxfp4_bf16", 64),
    ],
)
def test_meta_rejects_unsupported_mode_group_combinations(quant_mode_name, group_size):
    cache, x, slot = _meta_inputs(512, quant_mode_name, 32)
    with pytest.raises(RuntimeError, match="supported combinations"):
        OP(cache, x, slot, quant_group_size=group_size, quant_mode=quant_mode_name)


def test_meta_rejects_dimension_not_divisible_by_group_size():
    cache, _, slot = _meta_inputs(48, "mxfp4_bf16", 16)
    x = torch.empty((4, 40), dtype=torch.bfloat16, device="meta")
    with pytest.raises(RuntimeError, match="divisible by quant_group_size"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_does_not_validate_cache_size():
    cache, x, slot = _meta_inputs(512, "mxfp4_bf16", 16, cache_col=1)
    OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_cache_dtype_mismatch():
    _, x, slot = _meta_inputs(512, "mxfp4_bf16", 16)
    cache = torch.empty((8, 320), dtype=torch.float8_e4m3fn, device="meta")
    with pytest.raises(RuntimeError, match="cache dtype does not match"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_cache_rank():
    _, x, slot = _meta_inputs(512, "mxfp4_bf16", 16)
    cache = torch.empty((320,), dtype=torch.uint8, device="meta")
    with pytest.raises(RuntimeError, match="cache must be 2D"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_slot_length_mismatch():
    cache, x, _ = _meta_inputs(512, "mxfp4_bf16", 16)
    slot = torch.empty((5,), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError, match="slot_mapping length"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_x_dtype_mismatch():
    cache, _, slot = _meta_inputs(512, "mxfp4_bf16", 16)
    x = torch.empty((4, 512), dtype=torch.float16, device="meta")
    with pytest.raises(RuntimeError, match="x dtype must be bfloat16"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_noncontiguous_input():
    cache, _, slot = _meta_inputs(512, "mxfp4_bf16", 16)
    x = torch.empty((512, 4), dtype=torch.bfloat16, device="meta").transpose(0, 1)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")


def test_meta_rejects_reserved_x_scale():
    cache, x, slot = _meta_inputs(512, "mxfp4_bf16", 16)
    with pytest.raises(RuntimeError, match="x_scale is reserved"):
        OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16", x_scale=2.0)


def test_converter_rejects_unsupported_group_before_graph_construction():
    with pytest.raises(ValueError, match="supported quant_mode/quant_group_size"):
        _validate_mode_and_group_size("mxfp8_bf16", 16)


# ==================== torch eager 直调用例（不经图编译） ====================

# torch 直调：默认 mxfp8_bf16（quant_mode 缺省），验证原地更新与 slot=-1 行保持
def test_eager_mxfp8_default_mode_inplace():
    """默认 quant_mode（缺省即 mxfp8_bf16）eager 直调：原地更新、逐字节对齐、
    slot=-1 行保持不变。"""
    cache, x, slot, golden = _make_inputs(2, 512, 16, 32, seed=11)
    before = _to_u8(cache).copy()
    OP(cache, x, slot, quant_group_size=32, round_scale=True, x_scale=1.0)
    torch.npu.synchronize()
    got = _to_u8(cache)  # 原地语义：cache 自身即输出
    assert np.array_equal(got, golden.numpy())
    assert not np.array_equal(got, before)
    assert np.array_equal(got[0], before[0])  # slot=-1 行不动


# torch 直调：显式 mxfp4_bf16（d=128 为 F2 修复敏感形状），验证加宽 cache 尾部保持与越界 slot 静默跳过
def test_eager_mxfp4_d128_wide_cache_and_oob_slot():
    """显式 mxfp4_bf16、F2 敏感形状 d=128：加宽 cache 尾部保持、
    越界 slot 静默跳过无副作用。"""
    _, _, _, kv_col = _layout(128, 4)
    rng = np.random.RandomState(23)
    slot_np = rng.permutation(32)[:16].astype(np.int32)
    slot_np[0] = -1
    slot_np[3] = 32  # 越界：静默跳过
    cache, x, slot, golden = _make_inputs(4, 128, 16, 32, seed=23, wide=64,
                                          slot_np=slot_np)
    assert cache.shape[1] == kv_col + 64
    OP(cache, x, slot, quant_group_size=32, quant_mode="mxfp4_bf16",
       round_scale=True, x_scale=1.0)
    torch.npu.synchronize()
    got = _to_u8(cache)
    assert np.array_equal(got, golden.numpy())
    # 越界 slot 行与 32B 对齐之外的加宽区不被写坏
    assert np.array_equal(got[:, kv_col:], golden[:, kv_col:].numpy())


@pytest.mark.parametrize(
    "d,t,n,wide,slot_dtype,seed",
    [
        (16, 4, 8, 0, np.int32, 31),
        (48, 8, 16, 32, np.int32, 33),
        (96, 16, 32, 0, np.int64, 35),
        (128, 16, 32, 64, np.int32, 37),
        (512, 64, 128, 32, np.int32, 39),
        (2048, 64, 128, 64, np.int64, 41),
    ],
)
def test_eager_mxfp4_group16_random(d, t, n, wide, slot_dtype, seed):
    """MXFP4 group16 随机输入与 CPU golden 逐字节一致。"""
    cache, x, slot, golden = _make_inputs(
        4, d, t, n, seed=seed, wide=wide, group_size=16,
        slot_dtype=slot_dtype)
    OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16",
       round_scale=True, x_scale=1.0)
    torch.npu.synchronize()
    assert np.array_equal(_to_u8(cache), golden.numpy())


def test_eager_mxfp4_group16_uses_32b_layout():
    d = 128
    _, data_col, concat_col, cache_col = _layout(d, 4, 16)
    cache = torch.full((4, cache_col), 0x5A, dtype=torch.uint8, device="npu")
    x = torch.ones((2, d), dtype=torch.bfloat16, device="npu")
    slot = torch.tensor([1, -1], dtype=torch.int32, device="npu")

    OP(cache, x, slot, quant_group_size=16, quant_mode="mxfp4_bf16")
    torch.npu.synchronize()
    result = cache.cpu()

    assert torch.all(result[1, :data_col] == 0x66)
    scale_bits = result[1, data_col:concat_col].contiguous().view(torch.uint16)
    assert torch.all(scale_bits == 0x3E80)
    assert torch.all(result[1, concat_col:cache_col] == 0)
    assert torch.all(result[0] == 0x5A)


# torch 直调（负例）：非法裸名 "mxfp4" 被拒绝，且 cache 无任何写入
def test_eager_invalid_quant_mode_rejected():
    """裸名 "mxfp4" 必须被拒绝，且 cache 不发生任何写入。"""
    cache, x, slot, _ = _make_inputs(4, 128, 4, 8, seed=5)
    before = _to_u8(cache).copy()
    with pytest.raises((ValueError, RuntimeError)):
        OP(cache, x, slot, quant_group_size=32, quant_mode="mxfp4",
           round_scale=True, x_scale=1.0)
    torch.npu.synchronize()
    assert np.array_equal(_to_u8(cache), before)


# ==================== torchair ACL Graph 图模式用例（capture + 换值 replay） ====================

# ACL Graph 图模式的被编译模块：forward 内直调算子，cache 原地更新后返回
class _Kcev2Module(torch.nn.Module):
    def __init__(self, quant_mode, group_size):
        super().__init__()
        self._qm = quant_mode
        self._group_size = group_size

    def forward(self, cache, x, slot_mapping):
        OP(cache, x, slot_mapping, quant_group_size=self._group_size, quant_mode=self._qm,
           round_scale=True, x_scale=1.0)
        return cache


def _run_aclgraph(backend_mode, quant_mode_str, quant_mode, d, group_size=32):
    """ACL Graph capture + 换值 replay：均为逐字节对齐，且 replay 结果
    不等于首次捕获（无 stale 捕获，原地语义在静态地址约束下正确）。"""
    config = torchair.CompilerConfig()
    config.mode = backend_mode  # reduce-overhead(legacy) / npugraph_ex(现行)
    config.debug.aclgraph.clone_input = False
    backend = torchair.get_npu_backend(compiler_config=config)
    compiled = torch.compile(_Kcev2Module(quant_mode_str, group_size), backend=backend,
                             fullgraph=True, dynamic=False)

    cache, x, slot, golden = _make_inputs(
        quant_mode, d, 16, 32, seed=41, group_size=group_size)
    out = compiled(cache, x, slot)
    torch.npu.synchronize()
    first = _to_u8(out)
    assert np.array_equal(first, golden.numpy()), "capture run != golden"

    cache2, x2, slot2, golden2 = _make_inputs(
        quant_mode, d, 16, 32, seed=97, group_size=group_size)
    cache.copy_(cache2)
    x.copy_(x2)
    slot.copy_(slot2)
    out2 = compiled(cache, x, slot)
    torch.npu.synchronize()
    second = _to_u8(out2)
    assert np.array_equal(second, golden2.numpy()), "replay != new golden"
    assert not np.array_equal(second, first), "stale capture detected"


# torch ACL Graph 图模式：npugraph_ex（现行后端）× mxfp8_bf16，capture 与换值 replay 均逐字节对齐
def test_aclgraph_mxfp8_npugraph_ex():
    _run_aclgraph("npugraph_ex", "mxfp8_bf16", 2, 512)


# torch ACL Graph 图模式：reduce-overhead（legacy 后端）× mxfp4_bf16，覆盖另一后端与另一量化族
def test_aclgraph_mxfp4_reduce_overhead():
    _run_aclgraph("reduce-overhead", "mxfp4_bf16", 4, 128, group_size=16)
