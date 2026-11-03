#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Validate FlashAttention call parameters at the call site.

Insert debug_fa_all_params() right before the FA call, or call the individual
checks. Findings are graded: ERROR marks a contradiction the checker can prove
from the arguments alone, WARN marks a deviation from the recommended value for
the declared path, which several shipped models legitimately depart from, or a
value outside the range the checker covers.

The mask content checks index a 2048x2048 tensor and reduce it, which forces a
device-to-host sync on every call. That sync moves the symptom of memory-class
defects, so do not insert this checker while investigating those.

Usage:
    import sys; sys.path.append("<skill>/scripts")
    from fa_debug_utils import debug_fa_all_params

    debug_fa_all_params(query, key, value, path="std_pa", fa_version="v2",
                        input_layout="TND", sparse_mode=3, atten_mask=mask,
                        actual_seq_qlen=cu_q, actual_seq_kvlen=kv_len,
                        num_heads=n, scale=scale, is_prefill=True, layer_idx=i)

    python fa_debug_utils.py --self-test
"""

import argparse
import sys

import torch

# path -> (recommended prefill sparse_mode, recommended decode sparse_mode, legitimate deviations)
PATH_RECOMMENDED = {
    "std_pa": (3, 3, "单 token Decode 有用 0 + mask=None 的实现，按该模型的分支条件核对"),
    "sliding": (4, 4, "窗口由 pre_tokens / next_tokens 表达，本校验器不接收这两个参数"),
    "mla_absorb": (3, 0, "q_len>1 的 Decode 用 3 并保留 mask；也有实现两阶段共用一份入参恒用 3"),
}

# layout -> (query sequence axis, key sequence axis); None means packed, read from actual_seq_*
SEQ_AXIS = {
    "BSH": (1, 1),
    "BNSD": (2, 2),
    "BSND": (1, 1),
    "TND": (None, None),
    "TND_NTD": (None, None),
    "NTD_TND": (None, None),
}

# packed layout -> axis of the query token dimension
PACKED_TOKEN_AXIS = {"TND": 0, "TND_NTD": 0, "NTD_TND": 1}

CAUSAL_MASK_SHAPES = [(2048, 2048), (1, 2048, 2048), (1, 1, 2048, 2048)]
MASK_DTYPES = (torch.bool, torch.int8, torch.uint8)


def to_list(value):
    """Normalize a sequence-length argument to a plain list, or None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        listed = value.tolist()
        return list(listed) if isinstance(listed, (list, tuple)) else None
    return None


def longest_span(cumsum):
    """Longest per-sequence length encoded in a cumulative-sum list."""
    lens = to_list(cumsum)
    if not lens:
        return None
    longest, prev = 0, 0
    for cur in lens:
        longest = max(longest, cur - prev)
        prev = cur
    return longest


def sequence_lengths(query, key, input_layout, actual_seq_qlen, actual_seq_kvlen,
                     is_paged=False):
    """Longest per-sequence Q and KV lengths.

    Packed layouts carry no per-sequence axis, and a paged KV cache carries a
    block axis rather than a sequence axis, so both read the value from
    actual_seq_*. Taking it from the tensor shape is only valid otherwise.
    """
    q_axis, k_axis = SEQ_AXIS.get(input_layout, (None, None))
    known = input_layout in SEQ_AXIS

    if q_axis is not None and len(query.shape) > q_axis:
        q_len = query.shape[q_axis]
    else:
        # actual_seq_qlen is always cumulative, so the longest span is the per-sequence length.
        q_len = longest_span(actual_seq_qlen) if known else None

    if not is_paged and k_axis is not None and len(key.shape) > k_axis:
        kv_len = key.shape[k_axis]
    else:
        # Paged and packed layouts carry no sequence axis. actual_seq_kvlen is cumulative in some
        # paths and raw per-request in others, and the values alone do not say which. Both
        # readings bound the longest request from below, so take the smaller one.
        lens = to_list(actual_seq_kvlen)
        kv_len = min(longest_span(lens), max(lens)) if lens else None

    return q_len, kv_len


def debug_atten_mask(atten_mask, sparse_mode, is_prefill, path=None,
                     q_seq_len=None, kv_seq_len=None, layer_idx=0):
    """Check that atten_mask and sparse_mode agree, and compare against the declared path."""
    stage = "Prefill" if is_prefill else "Decode"
    print(f"--- Layer {layer_idx} atten_mask Debug ({path or '未声明路径'}, {stage}) ---")
    print(f"  sparse_mode: {sparse_mode}, q_seq_len: {q_seq_len}, kv_seq_len: {kv_seq_len}")

    if path is not None:
        if path not in PATH_RECOMMENDED:
            print(f"  [WARN] 未知 path '{path}', 可选 {list(PATH_RECOMMENDED)}，或传 None 跳过路径对照")
        else:
            prefill_sm, decode_sm, note = PATH_RECOMMENDED[path]
            expected = prefill_sm if is_prefill else decode_sm
            if sparse_mode != expected:
                print(f"  [WARN] {path} {stage} 推荐 sparse_mode={expected}, 当前 {sparse_mode}；"
                      f"合理偏离：{note}")
            else:
                print(f"  [OK] sparse_mode 与 {path} {stage} 推荐值一致")

    if not isinstance(sparse_mode, int) or isinstance(sparse_mode, bool):
        print(f"  [ERROR] sparse_mode 应为整数, 当前 {sparse_mode!r}, 后续 mask 校验跳过")
        return
    if sparse_mode not in (0, 1, 2, 3, 4):
        print(f"  [WARN] sparse_mode={sparse_mode} 超出本校验器覆盖的 0-4, "
              f"该值的 mask 要求按算子 API 文档确认, 后续 mask 校验跳过")
        return
    if sparse_mode in (1, 2, 3, 4) and atten_mask is None:
        print(f"  [ERROR] sparse_mode={sparse_mode} 要求传入 atten_mask, 但收到 None")
        return
    if sparse_mode == 0:
        if atten_mask is not None:
            print(f"  [WARN] sparse_mode=0 通常不需要 atten_mask, "
                  f"当前传入 shape={tuple(atten_mask.shape)}")
        else:
            print(f"  [OK] sparse_mode=0 + mask=None")
        return

    if not hasattr(atten_mask, "shape") or not hasattr(atten_mask, "dtype"):
        print(f"  [ERROR] atten_mask 应为张量, 当前 {type(atten_mask).__name__}")
        return
    print(f"  mask shape: {tuple(atten_mask.shape)}, dtype: {atten_mask.dtype}")
    if atten_mask.dtype not in MASK_DTYPES:
        print(f"  [ERROR] atten_mask dtype 必须为 bool/int8/uint8, 当前为 {atten_mask.dtype}")
        return

    if sparse_mode in (2, 3, 4):
        if tuple(atten_mask.shape) not in CAUSAL_MASK_SHAPES:
            print(f"  [ERROR] sparse_mode={sparse_mode} 要求 mask shape 为 "
                  f"(2048,2048)/(1,2048,2048)/(1,1,2048,2048), 当前为 {tuple(atten_mask.shape)}")
        else:
            print(f"  [OK] mask shape 符合 sparse_mode={sparse_mode} 要求")
            if sparse_mode == 3:
                mask_2d = atten_mask.reshape(2048, 2048) != 0
                ones = torch.ones(2048, 2048, dtype=torch.bool, device=atten_mask.device)
                upper_tri = torch.triu(ones, diagonal=1)
                lower_tri = torch.tril(ones, diagonal=0)
                if not mask_2d[upper_tri].all():
                    print(f"  [ERROR] sparse_mode=3 要求上三角为遮蔽位, 当前 mask 不是下三角形态")
                elif mask_2d[lower_tri].any():
                    print(f"  [ERROR] sparse_mode=3 要求含对角线的下三角为可见位, "
                          f"当前 mask 遮蔽过度")
    elif sparse_mode == 1 and len(atten_mask.shape) >= 2:
        mask_q_s, mask_kv_s = atten_mask.shape[-2:]
        if q_seq_len and mask_q_s < q_seq_len:
            print(f"  [ERROR] mask Q_S 维度 ({mask_q_s}) 小于 query 实际长度 ({q_seq_len})")
        if kv_seq_len and mask_kv_s < kv_seq_len:
            print(f"  [ERROR] mask KV_S 维度 ({mask_kv_s}) 小于 KV 实际长度 ({kv_seq_len})")


def debug_fa_version_params(fa_call_kwargs, fa_version):
    """Check that the keyword names used in the call belong to the declared FA version."""
    v1_only = {"actual_seq_lengths", "actual_seq_lengths_kv", "scale", "num_heads",
               "antiquant_scale", "antiquant_offset", "antiquant_mode",
               "quant_scale1", "quant_scale2", "quant_offset2",
               "dequant_scale1", "dequant_scale2", "softmax_lse_flag"}
    v2_only = {"actual_seq_qlen", "actual_seq_kvlen", "softmax_scale",
               "num_query_heads", "dequant_scale_key", "dequant_scale_value",
               "dequant_offset_key", "dequant_offset_value",
               "key_quant_mode", "value_quant_mode",
               "quant_scale_out", "quant_offset_out",
               "return_softmax_lse", "learnable_sink", "dequant_scale_query"}

    if not fa_call_kwargs:
        print(f"  [INFO] 未传 fa_call_kwargs, 跳过 FA {fa_version} 参数名检查")
        return

    if fa_version not in ("v1", "v2"):
        print(f"  [WARN] fa_version 应为 'v1' 或 'v2', 当前 {fa_version!r}, 参数名检查跳过")
        return
    wrong = v1_only if fa_version == "v2" else v2_only
    other = "v1" if fa_version == "v2" else "v2"
    for param in fa_call_kwargs:
        if param in wrong:
            print(f"  [ERROR] 参数 '{param}' 属于 FA {other}，但当前使用 FA {fa_version}")

    required = ("softmax_scale", "num_query_heads") if fa_version == "v2" else ("scale", "num_heads")
    defaults = ("1.0", "1")
    for name, default in zip(required, defaults):
        if name not in fa_call_kwargs:
            print(f"  [WARN] FA {fa_version} 未传 {name}，将使用默认值 {default}")


def debug_fa_layout_shape(query, key, value, input_layout, num_query_heads,
                          block_table=None, layer_idx=0):
    """Check the query rank and head axis against input_layout; print K/V shapes only.

    K/V are left unchecked: under paging they are cache views whose shape is not
    determined by the layout name.
    """
    print(f"--- Layer {layer_idx} FA Layout Debug ---")
    print(f"  input_layout: {input_layout}")
    print(f"  Q: {tuple(query.shape)}, K: {tuple(key.shape)}, V: {tuple(value.shape)}")

    # layout -> (expected query rank, query head axis)
    rank_and_head_axis = {
        "BSH": (3, None), "BNSD": (4, 1), "BSND": (4, 2),
        "TND": (3, 1), "TND_NTD": (3, 1), "NTD_TND": (3, 0),
    }
    if input_layout not in rank_and_head_axis:
        print(f"  [WARN] layout '{input_layout}' 未纳入 shape 校验, 请人工核对")
    else:
        expected_rank, head_axis = rank_and_head_axis[input_layout]
        if len(query.shape) != expected_rank:
            print(f"  [ERROR] {input_layout} 要求 Q 为 {expected_rank}D, 当前 {len(query.shape)}D")
        elif head_axis is not None and query.shape[head_axis] != num_query_heads:
            print(f"  [ERROR] {input_layout} Q N轴 ({query.shape[head_axis]}) "
                  f"!= num_query_heads ({num_query_heads})")

    if block_table is not None:
        print(f"  [INFO] PA 模式, KV 按 block_table 索引")
        if isinstance(block_table, dict):
            print(f"  [ERROR] block_table 是 Dict[attn_type → Tensor], "
                  f"需先按 attn_type 取出对应 tensor 再传入")
        elif getattr(block_table, "ndim", None) != 2:
            print(f"  [ERROR] block_table 必须为 2D, 当前 {getattr(block_table, 'ndim', '未知')}D")
        elif getattr(block_table, "dtype", None) != torch.int32:
            print(f"  [ERROR] block_table dtype 必须为 int32, "
                  f"当前 {getattr(block_table, 'dtype', '未知')}")


def debug_actual_seq_lengths(actual_seq_qlen, actual_seq_kvlen, input_layout,
                             kv_seq_len=None, layer_idx=0, total_q_tokens=None,
                             kv_capacity=None, is_prefill=None, is_paged=False):
    """Check how actual_seq_lengths was built.

    total_q_tokens is the length of the query token axis under a packed layout;
    kv_capacity is the KV upper bound derived from the block table under paging.
    Both feed the length invariants. Under paging the KV length is independent of
    Q, so the Prefill equality below only holds without it.
    """
    if not isinstance(input_layout, str):
        print(f"  [ERROR] Layer {layer_idx}: input_layout 应为字符串, 当前 {input_layout!r}")
        return
    if input_layout not in SEQ_AXIS:
        print(f"  [ERROR] Layer {layer_idx}: 未知 input_layout {input_layout!r}, "
              f"actual_seq_lengths 校验跳过")
        return
    if input_layout.split("_")[0] == "TND" or input_layout == "NTD_TND":
        qlen = to_list(actual_seq_qlen)
        if not qlen:
            print(f"  [ERROR] Layer {layer_idx}: TND 系列 layout 必须传入 actual_seq_qlen")
            return
        if qlen and qlen[0] <= 0:
            print(f"  [ERROR] actual_seq_qlen 首项为 {qlen[0]}, 应为首个序列的正长度")
        for i in range(1, len(qlen)):
            if qlen[i] < qlen[i - 1]:
                print(f"  [ERROR] TND actual_seq_qlen 必须单调递增 (cumsum), "
                      f"但 [{i-1}]={qlen[i-1]} > [{i}]={qlen[i]}")
            elif qlen[i] == qlen[i - 1]:
                print(f"  [WARN] actual_seq_qlen [{i-1}] 与 [{i}] 相等 ({qlen[i]}), "
                      f"确认是否误传原始长度而非累积和")
        if total_q_tokens is not None and qlen[-1] != total_q_tokens:
            print(f"  [ERROR] actual_seq_qlen 末值 {qlen[-1]} != Q 的 token 总数 "
                  f"{total_q_tokens}, 累积和与实际 packed 长度不符")
        kvlen = to_list(actual_seq_kvlen)
        if not kvlen:
            print(f"  [ERROR] Layer {layer_idx}: TND 系列 layout 必须传入 actual_seq_kvlen")
        elif len(kvlen) != len(qlen):
            print(f"  [ERROR] actual_seq_kvlen {len(kvlen)} 项与 actual_seq_qlen {len(qlen)} 项 "
                  f"batch 数不一致")
        elif is_prefill and not is_paged and kvlen != qlen:
            print(f"  [ERROR] 非分页 TND Prefill 的 K/V 是本次 packed 的完整张量, "
                  f"actual_seq_kvlen 应与 qlen 同为累积和, 当前 kvlen={kvlen} qlen={qlen}")
        check_kv_capacity(actual_seq_kvlen, kv_capacity, layer_idx)
        return

    kvlen = to_list(actual_seq_kvlen)
    if kvlen and kv_seq_len and kv_capacity is None:
        # kv_seq_len only carries independent information when it came from the tensor shape.
        # Under paging it is derived from kvlen itself, so comparing them proves nothing.
        for i, kv_l in enumerate(kvlen):
            if kv_l > kv_seq_len:
                print(f"  [ERROR] actual_seq_kvlen[{i}]={kv_l} > KV 序列长度 ({kv_seq_len})")
    check_kv_capacity(actual_seq_kvlen, kv_capacity, layer_idx)


def check_kv_capacity(actual_seq_kvlen, kv_capacity, layer_idx=0):
    """Under paging, no request may exceed what its block table can address.

    Only capacity is checked. The framework pads block tables with a null block,
    so duplicate block ids are expected and are not an error.
    """
    if kv_capacity is None:
        return
    for i, kv_l in enumerate(to_list(actual_seq_kvlen) or []):
        if kv_l > kv_capacity:
            print(f"  [ERROR] Layer {layer_idx}: actual_seq_kvlen[{i}]={kv_l} 超出 "
                  f"block table 容量 {kv_capacity} (block_table 第二维 × block_size)")


def debug_inner_precise(atten_mask, inner_precise, q_seq_len, sparse_mode, layer_idx=0):
    """Check whether fully masked rows require the row-invalid correction mode."""
    if not q_seq_len or q_seq_len <= 1 or sparse_mode not in (0, 1) or atten_mask is None:
        return
    all_masked_rows = (atten_mask != 0).all(dim=-1)
    num_invalid_rows = int(all_masked_rows.sum().item())
    if num_invalid_rows > 0 and inner_precise in (0, 1):
        print(f"  [WARN] Layer {layer_idx}: {num_invalid_rows} 个全遮蔽行, "
              f"建议设置 inner_precise=2 或 3")


def debug_mla_rope(query, query_rope, key_rope, sparse_mode, layer_idx=0):
    """Check the MLA rope arguments."""
    if (query_rope is None) != (key_rope is None):
        print(f"  [ERROR] Layer {layer_idx}: query_rope 和 key_rope 必须同时配置或同时不配置")
        return
    if query_rope is None:
        return
    q_d = query.shape[-1]
    if q_d not in (512, 128):
        print(f"  [ERROR] MLA query D 仅支持 512/128, 当前 {q_d}")
    # Only query_rope carries a plain [..., 64] layout. key_rope is a paged NZ cache view in the
    # absorb path, whose last axis is the NZ dimension, so its D is not checkable here.
    if query_rope.shape[-1] != 64:
        print(f"  [ERROR] query_rope 的 rope D 必须为 64, 当前 {query_rope.shape[-1]}")
    if q_d == 512 and sparse_mode in (1, 2):
        print(f"  [ERROR] MLA D=512 不支持 sparse_mode={sparse_mode}, 可选 0/3/4")


def debug_fa_all_params(query, key, value, *, path=None, fa_version="v1",
                        input_layout="TND", sparse_mode=None, atten_mask=None,
                        actual_seq_qlen=None, actual_seq_kvlen=None,
                        num_heads=1, scale=1.0, block_table=None, block_size=None,
                        query_rope=None, key_rope=None, inner_precise=0,
                        is_prefill=True, fa_call_kwargs=None, layer_idx=0):
    """Run every check against one FA call site."""
    if sparse_mode is None:
        raise ValueError("sparse_mode 必须显式传入：FA 调用中省略该形参等同于传 0，"
                         "校验器不替你猜")
    if not getattr(query, "shape", ()):
        raise ValueError("query 必须是带非空 shape 的张量")
    stage = "Prefill" if is_prefill else "Decode"
    print(f"\n{'='*60}")
    print(f"FA {fa_version} 综合校验 - Layer {layer_idx} ({path or '未声明路径'}, {stage})")
    print(f"{'='*60}")

    is_paged = block_table is not None
    q_seq_len, kv_seq_len = sequence_lengths(
        query, key, input_layout, actual_seq_qlen, actual_seq_kvlen, is_paged=is_paged)

    token_axis = PACKED_TOKEN_AXIS.get(input_layout)
    total_q_tokens = query.shape[token_axis] if token_axis is not None else None
    kv_capacity = None
    if is_paged and block_size and getattr(block_table, "ndim", None) == 2:
        kv_capacity = block_table.shape[1] * block_size

    if scale == 1.0:
        print(f"  [WARN] scale = 1.0, 确认是否忘记设置 1/sqrt(head_dim)?")

    debug_fa_layout_shape(query, key, value, input_layout, num_heads, block_table, layer_idx)
    debug_fa_version_params(fa_call_kwargs, fa_version)
    debug_atten_mask(atten_mask, sparse_mode, is_prefill, path, q_seq_len, kv_seq_len, layer_idx)
    debug_actual_seq_lengths(actual_seq_qlen, actual_seq_kvlen, input_layout, kv_seq_len,
                             layer_idx, total_q_tokens, kv_capacity, is_prefill, is_paged)
    debug_inner_precise(atten_mask, inner_precise, q_seq_len, sparse_mode, layer_idx)
    debug_mla_rope(query, query_rope, key_rope, sparse_mode, layer_idx)
    print(f"{'='*60}\n")


# --------------------------------------------------------------------------------------------------
# Self-test: configurations taken from models already adapted in this repository must not raise
# ERROR, and the traps listed in references/kvcache_fa_checklist.md must be caught.
# --------------------------------------------------------------------------------------------------

def _meta(shape):
    return torch.zeros(shape, dtype=torch.bfloat16, device="meta")


def _causal_mask(dtype=torch.bool):
    return torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(dtype)


def _shipped_cases():
    """Synthetic configurations built from call sites of models adapted in this repository.

    Shapes are fixtures; the parameter combinations are the ones the models actually pass.
    This is not a run of the models themselves.
    """
    causal = _causal_mask()
    return [
        ("qwen std_pa Prefill", dict(
            query=_meta((256, 32, 128)), key=_meta((256, 8, 128)), value=_meta((256, 8, 128)),
            path="std_pa", fa_version="v2", input_layout="TND", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[128, 256], actual_seq_kvlen=[128, 256],
            num_heads=32, scale=0.088, is_prefill=True)),
        ("qwen std_pa Decode PA", dict(
            query=_meta((2, 32, 128)), key=_meta((64, 128, 8, 128)), value=_meta((64, 128, 8, 128)),
            path="std_pa", fa_version="v2", input_layout="TND", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[1, 2], actual_seq_kvlen=[129, 257],
            num_heads=32, scale=0.088, is_prefill=False,
            block_table=torch.zeros(2, 8, dtype=torch.int32))),
        ("qwen3_moe float8 KV Decode", dict(
            query=_meta((2, 32, 128)), key=_meta((64, 128, 8, 128)), value=_meta((64, 128, 8, 128)),
            path="std_pa", fa_version="v2", input_layout="TND", sparse_mode=0,
            atten_mask=None, actual_seq_qlen=[1, 2], actual_seq_kvlen=[129, 257],
            num_heads=32, scale=0.088, is_prefill=False,
            block_table=torch.zeros(2, 8, dtype=torch.int32))),
        ("gpt_oss sliding Prefill", dict(
            query=_meta((256, 64, 64)), key=_meta((256, 8, 64)), value=_meta((256, 8, 64)),
            path="sliding", fa_version="v2", input_layout="TND", sparse_mode=4,
            atten_mask=causal, actual_seq_qlen=[128, 256], actual_seq_kvlen=[128, 256],
            num_heads=64, scale=0.125, is_prefill=True)),
        ("deepseek_r1 mla_absorb Decode q_len=1", dict(
            query=_meta((2, 128, 512)), key=_meta((128, 64, 576)), value=_meta((128, 64, 576)),
            path="mla_absorb", fa_version="v2", input_layout="TND_NTD", sparse_mode=0,
            atten_mask=None, actual_seq_qlen=[1, 2], actual_seq_kvlen=[129, 257],
            num_heads=128, scale=0.045, is_prefill=False,
            query_rope=_meta((2, 128, 64)), key_rope=_meta((64, 1, 4, 128, 16)),
            block_table=torch.zeros(2, 8, dtype=torch.int32))),
        ("deepseek_r1 mla_absorb MTP Decode q_len>1", dict(
            query=_meta((4, 128, 512)), key=_meta((128, 64, 576)), value=_meta((128, 64, 576)),
            path="mla_absorb", fa_version="v2", input_layout="TND_NTD", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[2, 4], actual_seq_kvlen=[130, 258],
            num_heads=128, scale=0.045, is_prefill=False,
            query_rope=_meta((4, 128, 64)), key_rope=_meta((64, 1, 4, 128, 16)),
            block_table=torch.zeros(2, 8, dtype=torch.int32))),
        ("kimi_k2_thinking mla_absorb Decode sparse_mode=3", dict(
            query=_meta((2, 64, 512)), key=_meta((64, 64, 576)), value=_meta((64, 64, 576)),
            path="mla_absorb", fa_version="v1", input_layout="TND_NTD", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[1, 2], actual_seq_kvlen=[129, 257],
            num_heads=64, scale=0.045, is_prefill=False,
            query_rope=_meta((2, 64, 64)), key_rope=_meta((64, 1, 4, 128, 16)),
            block_table=torch.zeros(2, 8, dtype=torch.int32), block_size=128)),
        ("pangu_7b BSH legacy Prefill sparse_mode=2", dict(
            query=_meta((2, 128, 4096)), key=_meta((2, 128, 4096)), value=_meta((2, 128, 4096)),
            path=None, fa_version="v1", input_layout="BSH", sparse_mode=2,
            atten_mask=causal, actual_seq_kvlen=[128, 128],
            num_heads=32, scale=0.088, is_prefill=True)),
        ("pangu_7b BSH legacy Decode，省略 sparse_mode 等同 0", dict(
            query=_meta((2, 1, 4096)), key=_meta((2, 512, 4096)), value=_meta((2, 512, 4096)),
            path=None, fa_version="v1", input_layout="BSH", sparse_mode=0,
            atten_mask=None, actual_seq_kvlen=[129, 257],
            num_heads=32, scale=0.088, is_prefill=False)),
        ("kimi_k3 DSpark 非因果 TND PA Decode，q_len=7", dict(
            query=_meta((14, 32, 128)), key=_meta((64, 128, 8, 128)), value=_meta((64, 128, 8, 128)),
            path=None, fa_version="v2", input_layout="TND", sparse_mode=0,
            atten_mask=None, actual_seq_qlen=[7, 14], actual_seq_kvlen=[135, 263],
            num_heads=32, scale=0.088, is_prefill=False,
            block_table=torch.zeros(2, 8, dtype=torch.int32), block_size=128)),
        ("gemma_4 全注意力 Prefill BNSD", dict(
            query=_meta((2, 8, 1024, 128)), key=_meta((2, 8, 1024, 128)),
            value=_meta((2, 8, 1024, 128)),
            path="std_pa", fa_version="v1", input_layout="BNSD", sparse_mode=3,
            atten_mask=causal, actual_seq_kvlen=[1024, 1024],
            num_heads=8, scale=0.088, is_prefill=True)),
        ("deepseek_r1 MLA 非 absorb Prefill NTD_TND", dict(
            query=_meta((128, 256, 192)), key=_meta((256, 128, 192)),
            value=_meta((256, 128, 192)),
            path=None, fa_version="v1", input_layout="NTD_TND", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[128, 256], actual_seq_kvlen=[128, 256],
            num_heads=128, scale=0.045, is_prefill=True)),
        ("kimi_k3 MLA 分页 chunked Prefill，kvlen 为原始值", dict(
            query=_meta((256, 64, 512)), key=_meta((64, 1, 4, 128, 16)),
            value=_meta((64, 1, 4, 128, 16)),
            path=None, fa_version="v2", input_layout="TND_NTD", sparse_mode=3,
            atten_mask=causal, actual_seq_qlen=[128, 256], actual_seq_kvlen=[512, 768],
            num_heads=64, scale=0.045, is_prefill=True,
            query_rope=_meta((256, 64, 64)), key_rope=_meta((64, 1, 4, 128, 16)),
            block_table=torch.zeros(2, 8, dtype=torch.int32), block_size=128)),
        ("gemma_4 分页 BSH Decode，kvlen 为原始值", dict(
            query=_meta((2, 1, 4096)), key=_meta((64, 128, 2048)), value=_meta((64, 128, 2048)),
            path=None, fa_version="v1", input_layout="BSH", sparse_mode=0,
            atten_mask=None, actual_seq_kvlen=[513, 1025],
            num_heads=8, scale=0.088, is_prefill=False,
            block_table=torch.zeros(2, 16, dtype=torch.int32), block_size=128)),
    ]


def _trap_cases():
    """Traps from references/kvcache_fa_checklist.md that must be reported."""
    causal = _causal_mask()
    base = dict(query=_meta((256, 32, 128)), key=_meta((256, 8, 128)),
                value=_meta((256, 8, 128)), fa_version="v2", input_layout="TND",
                num_heads=32, scale=0.088, is_prefill=True,
                actual_seq_qlen=[128, 256], actual_seq_kvlen=[128, 256])
    return [
        ("sparse_mode=3 却未传 mask", {**base, "sparse_mode": 3, "atten_mask": None}),
        ("mask dtype 为 float", {**base, "sparse_mode": 3,
                                 "atten_mask": _causal_mask(torch.float32)}),
        ("mask shape 非 2048 系列", {**base, "sparse_mode": 3,
                                     "atten_mask": torch.zeros(512, 512, dtype=torch.bool)}),
        ("sparse_mode=3 mask 非下三角", {**base, "sparse_mode": 3,
                                        "atten_mask": torch.tril(
                                            torch.ones(2048, 2048, dtype=torch.bool), -1)}),
        ("TND 未传 actual_seq_qlen", {**base, "sparse_mode": 3, "atten_mask": causal,
                                      "actual_seq_qlen": None}),
        ("TND Prefill 的 kvlen 误传原始值", {**base, "sparse_mode": 3, "atten_mask": causal,
                                             "actual_seq_kvlen": [128, 128]}),
        ("非张量 atten_mask", {**base, "sparse_mode": 3, "atten_mask": [[0, 1], [1, 0]]}),
        ("input_layout 拼写错误", {**base, "sparse_mode": 3, "atten_mask": causal,
                                   "input_layout": "tnd"}),
        ("actual_seq_qlen 首项为 0", {**base, "sparse_mode": 3, "atten_mask": causal,
                                      "actual_seq_qlen": [0, 128]}),
        ("actual_seq_qlen 非单调递增", {**base, "sparse_mode": 3, "atten_mask": causal,
                                        "actual_seq_qlen": [256, 128]}),
        ("FA v2 混入 v1 参数名", {**base, "sparse_mode": 3, "atten_mask": causal,
                                  "fa_call_kwargs": {"scale": 0.088, "num_heads": 32}}),
        ("TND Q N轴与 num_heads 不符", {**base, "sparse_mode": 3, "atten_mask": causal,
                                        "num_heads": 16}),
        ("block_table 传了原始 Dict", {**base, "sparse_mode": 3, "atten_mask": causal,
                                       "block_table": {"full": torch.zeros(2, 8)}}),
        ("TND 累计末值与 Q token 数不符", {**base, "sparse_mode": 3, "atten_mask": causal,
                                          "actual_seq_qlen": [128, 200]}),
        ("分页 KV 长度超出 block table 容量", {
            **base, "sparse_mode": 3, "atten_mask": causal,
            "block_table": torch.zeros(2, 8, dtype=torch.int32), "block_size": 128,
            "actual_seq_kvlen": [999999, 999999]}),
        ("block_table dtype 非 int32", {**base, "sparse_mode": 3, "atten_mask": causal,
                                        "block_table": torch.zeros(2, 8, dtype=torch.int64)}),
        ("sparse_mode=3 mask 全遮蔽", {**base, "sparse_mode": 3,
                                       "atten_mask": torch.ones(2048, 2048, dtype=torch.bool)}),
        ("sparse_mode 非整数", {**base, "sparse_mode": "3", "atten_mask": causal}),
        ("MLA rope 只传一半", {**base, "sparse_mode": 3, "atten_mask": causal,
                               "query_rope": _meta((256, 32, 64))}),
        ("MLA rope D 不是 64", {**base, "sparse_mode": 3, "atten_mask": causal,
                                "query_rope": _meta((256, 32, 128)),
                                "key_rope": _meta((256, 8, 128))}),
        ("MLA D=512 配 sparse_mode=1", {**base, "sparse_mode": 1, "atten_mask": causal,
                                        "query": _meta((256, 32, 512)),
                                        "query_rope": _meta((256, 32, 64)),
                                        "key_rope": _meta((256, 8, 64))}),
    ]


def _capture(kwargs):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        debug_fa_all_params(**kwargs)
    return buf.getvalue()


def run_self_test():
    """Assert no false positive on shipped configurations and no miss on known traps."""
    failures = 0

    print("已适配模型的真实配置（不应报 ERROR）")
    for name, kwargs in _shipped_cases():
        errors = [line.strip() for line in _capture(kwargs).splitlines() if "[ERROR]" in line]
        print(f"  {'OK  ' if not errors else 'FAIL'} {name}")
        for err in errors:
            print(f"         {err}")
        failures += bool(errors)

    print("\n已知陷阱（必须报 ERROR）")
    for name, kwargs in _trap_cases():
        errors = [line.strip() for line in _capture(kwargs).splitlines() if "[ERROR]" in line]
        print(f"  {'OK  ' if errors else 'FAIL'} {name}")
        failures += not errors

    print(f"\n{'全部通过' if not failures else f'{failures} 项未通过'}")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Validate FlashAttention call parameters at the call site.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the built-in regression cases and exit")
    args = parser.parse_args()
    if args.self_test:
        sys.exit(1 if run_self_test() else 0)
    parser.print_help()


if __name__ == "__main__":
    main()
