# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K3 Phase-2 ``block_attn_res_update`` partial update and online-softmax merge.

This operator is the fused CANNDSL counterpart of K3's
``_update_attn_res_online_softmax`` helper.  For one caller-sliced slot it:

1. computes ``updated_partial = partial_block + partial_delta`` in fp32;
2. computes the RMS-normalized partial score with the selected effective query;
3. merges that one-candidate partial state with the Phase-1 inter-block state;
4. returns both the final normalized output and ``updated_partial``.

The Phase-1 state uses the paper's stable representation ``(o_tilde, m, l)``.
For the partial candidate, ``m2=score``, ``l2=1`` and ``o2=partial_block``.
"""

import cannbotdsl
import torch

from cannbotdsl import dtypes
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.core.frontend.compiler import compile_function
from cannbotdsl.dtypes import (
    bfloat16 as BFloat16,
    float16 as Float16,
    float32 as Float32,
)
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.raw_reg import (
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp_sub,
    vload,
    vload_brc,
    vload_unpack,
    vmax,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore,
)
from cannbotdsl.runtime import from_torch_npu
from cannbotdsl.tensor import local_slice, mem_copy, tile_view
from cannbotdsl.typing.types import MemLoc, Tensor
from cannbotdsl.vf import vf


VL = 64
DEFAULT_BLOCK_NUM = 64

_TORCH_TO_DSL = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
}
_COMPILED_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _launch_block_num(tokens: int) -> int:
    """Update owns token rows, so launching more blocks cannot add work."""
    return max(1, min(DEFAULT_BLOCK_NUM, int(tokens)))


def _block_attn_res_update_body(
        partial_ub, delta_ub, inter_ub, query_ub,
        stats_ub, partial_out_ub, h_out_ub,
        num_seg, d, avg, epsilon):
    """fp32 partial plus bf16/fp16 delta; all arithmetic/output is fp32."""
    with vf(mode="raw"):
        full = full_mask(32)

        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_ub, off)
            delta = vcast(
                vload_unpack(delta_ub, off, mode=UnpackMode.B16_TO_B32),
                Float32, mask=mask,
            )
            updated = vadd(partial, delta, mask=mask)
            vstore(partial_out_ub, off, updated, mask)
        vmem_bar(mode="vst_vld")

        dot = vdup_scalar(0.0, Float32, mask=full)
        ssq = vdup_scalar(0.0, Float32, mask=full)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_out_ub, off)
            query = vload(query_ub, off)
            dot = vadd(dot, vmul(partial, query, mask=mask), mask=full)
            ssq = vadd(ssq, vmul(partial, partial, mask=mask), mask=full)

        denom = vsqrt(
            vadds(
                vmuls(vreduce_sum(ssq, mask=full), avg, mask=full),
                epsilon, mask=full,
            ),
            mask=full,
        )
        partial_score = vdup_lane0(
            vdiv(vreduce_sum(dot, mask=full), denom, mask=full),
            mask=full,
        )

        inter_max = vload_brc(stats_ub, 0)
        inter_sum = vload_brc(stats_ub, 8)
        merged_max = vmax(inter_max, partial_score, mask=full)
        history_scale = vexp_sub(inter_max, merged_max, mask=full)
        partial_scale = vexp_sub(partial_score, merged_max, mask=full)
        merged_sum = vadd(
            vmul(history_scale, inter_sum, mask=full),
            partial_scale,
            mask=full,
        )

        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(d) - off, elem_bits=32)
            partial = vload(partial_out_ub, off)
            inter = vload(inter_ub, off)
            numerator = vadd(
                vmul(inter, history_scale, mask=mask),
                vmul(partial, partial_scale, mask=mask),
                mask=mask,
            )
            merged = vdiv(numerator, merged_sum, mask=mask)
            vstore(h_out_ub, off, merged, mask)
        vmem_bar(mode="vst_vld")


_block_attn_res_update_body = compile_function(
    _block_attn_res_update_body, enable_preprocessor=True,
).function


class BlockAttnResUpdateVector:
    """Per-core UB state for the fused Phase-2 update."""

    def __init__(self, d: int, delta_dtype=Float16):
        self.d = int(d)
        self.delta_dtype = delta_dtype
        self.w = _ceil_div(self.d, VL) * VL
        self.num_seg = _ceil_div(self.d, VL)
        self.avg = 1.0 / self.d

        self.partial_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        self.delta_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=delta_dtype, depth=1,
        )
        self.inter_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        self.query_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        # m and l occupy separate 32-byte aligned regions.
        self.stats_ub = Channel(
            MemLoc.UB, shape=(1, VL), dtype=Float32, depth=1,
        )
        self.partial_out_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        self.h_out_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
    def load_query(self, gm_effective_query):
        slot = self.query_ub.acquire()
        mem_copy(
            local_slice(slot, (1, self.d), offset=0),
            tile_view(
                gm_effective_query, (1, self.d),
                (Int64(0), Int64(0)),
            ),
        )
        self.query_ub.commit(slot)

    def load_row(self, gm_partial, gm_delta, gm_inter_numerator,
                 gm_inter_max, gm_inter_sum, row):
        partial_slot = self.partial_ub.acquire()
        mem_copy(
            local_slice(partial_slot, (1, self.d), offset=0),
            tile_view(gm_partial, (1, self.d), (row, Int64(0))),
        )
        self.partial_ub.commit(partial_slot)

        delta_slot = self.delta_ub.acquire()
        mem_copy(
            local_slice(delta_slot, (1, self.d), offset=0),
            tile_view(gm_delta, (1, self.d), (row, Int64(0))),
        )
        self.delta_ub.commit(delta_slot)

        inter_slot = self.inter_ub.acquire()
        mem_copy(
            local_slice(inter_slot, (1, self.d), offset=0),
            tile_view(
                gm_inter_numerator, (1, self.d),
                (row, Int64(0)),
            ),
        )
        self.inter_ub.commit(inter_slot)

        stats_slot = self.stats_ub.acquire()
        mem_copy(
            local_slice(stats_slot, (1, 1), offset=0),
            tile_view(gm_inter_max, (1, 1), (row, Int64(0))),
        )
        mem_copy(
            local_slice(stats_slot, (1, 1), offset=32),
            tile_view(gm_inter_sum, (1, 1), (row, Int64(0))),
        )
        self.stats_ub.commit(stats_slot)

    def compute_row(self, query_cur, epsilon):
        partial_cur = self.partial_ub.wait()
        delta_cur = self.delta_ub.wait()
        inter_cur = self.inter_ub.wait()
        stats_cur = self.stats_ub.wait()
        partial_out_slot = self.partial_out_ub.acquire()
        h_out_slot = self.h_out_ub.acquire()

        _block_attn_res_update_body(
            partial_cur, delta_cur, inter_cur, query_cur,
            stats_cur, partial_out_slot, h_out_slot,
            self.num_seg, self.d, self.avg, epsilon,
        )

        self.partial_ub.release(partial_cur)
        self.delta_ub.release(delta_cur)
        self.inter_ub.release(inter_cur)
        self.stats_ub.release(stats_cur)
        self.partial_out_ub.commit(partial_out_slot)
        self.h_out_ub.commit(h_out_slot)

    def store_row(self, gm_h, gm_partial_out, row):
        partial_cur = self.partial_out_ub.wait()
        mem_copy(
            tile_view(gm_partial_out, (1, self.d), (row, Int64(0))),
            local_slice(partial_cur, (1, self.d), offset=0),
        )
        self.partial_out_ub.release(partial_cur)

        h_cur = self.h_out_ub.wait()
        mem_copy(
            tile_view(gm_h, (1, self.d), (row, Int64(0))),
            local_slice(h_cur, (1, self.d), offset=0),
        )
        self.h_out_ub.release(h_cur)


@kernel
class block_attn_res_update_kernel:
    """Fused K3 Phase-2 update; each core owns a contiguous token range."""

    def __init__(self, num_tokens: int, d: int, delta_dtype=Float16):
        self.num_tokens = int(num_tokens)
        self.d = int(d)
        self.delta_dtype = delta_dtype

    def __call__(self, gm_partial: Tensor, gm_delta: Tensor,
                 gm_effective_query: Tensor,
                 gm_inter_max: Tensor, gm_inter_sum: Tensor,
                 gm_inter_numerator: Tensor, gm_epsilon: Tensor,
                 gm_h: Tensor, gm_partial_out: Tensor):
        vector = BlockAttnResUpdateVector(
            self.d, self.delta_dtype,
        )
        block_idx = get_block_idx()
        block_num = get_block_num()
        num_tokens = Int64(self.num_tokens)
        logical = block_num
        if logical > num_tokens:
            logical = num_tokens
        rows_per_core = (num_tokens + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_tokens:
            row_end = num_tokens

        epsilon = gm_epsilon[(Int64(0),)]

        if block_idx < logical:
            vector.load_query(gm_effective_query)
            query_cur = vector.query_ub.wait()
            for row in range(row_start, row_end):
                vector.load_row(
                    gm_partial, gm_delta, gm_inter_numerator,
                    gm_inter_max, gm_inter_sum, row,
                )
                vector.compute_row(query_cur, epsilon)
                vector.store_row(gm_h, gm_partial_out, row)
            vector.query_ub.release(query_cur)


class BlockAttnResUpdate:
    def __init__(self, num_tokens: int, d: int, delta_dtype=Float16,
                 block_num: int = DEFAULT_BLOCK_NUM):
        self.num_tokens = int(num_tokens)
        self.d = int(d)
        self.delta_dtype = delta_dtype
        self.block_num = int(block_num)

    @jit
    def run(self, gm_partial, gm_delta, gm_effective_query,
            gm_inter_max, gm_inter_sum, gm_inter_numerator, gm_epsilon,
            gm_h, gm_partial_out):
        op = block_attn_res_update_kernel(
            self.num_tokens, self.d, self.delta_dtype,
        )
        op[self.block_num](
            gm_partial, gm_delta, gm_effective_query,
            gm_inter_max, gm_inter_sum, gm_inter_numerator, gm_epsilon,
            gm_h, gm_partial_out,
        )


def _compiled_update_kernel(
    tokens: int,
    d: int,
    delta_dtype: torch.dtype,
    block_num: int,
):
    """Compile once per static Update shape and reuse it across all slots."""
    key = (int(tokens), int(d), delta_dtype, int(block_num))
    compiled = _COMPILED_KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled

    op = BlockAttnResUpdate(
        tokens, d, delta_dtype=_TORCH_TO_DSL[delta_dtype], block_num=block_num,
    )
    fake = cannbotdsl.TensorSpec
    compiled = op.run.compile(
        fake((tokens, d), dtypes.float32),
        fake((tokens, d), _TORCH_TO_DSL[delta_dtype]),
        fake((1, d), dtypes.float32),
        fake((tokens, 1), dtypes.float32),
        fake((tokens, 1), dtypes.float32),
        fake((tokens, d), dtypes.float32),
        fake((1,), dtypes.float32),
        fake((tokens, d), dtypes.float32),
        fake((tokens, d), dtypes.float32),
    )
    _COMPILED_KERNEL_CACHE[key] = compiled
    return compiled


def _block_attn_res_update_eager(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: float | torch.Tensor = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused equivalent of K3 ``_update_attn_res_online_softmax``.

    Args:
        partial_block: read-only ``[tokens,D]`` fp32 accumulated residual.
        partial_delta: ``[tokens,D]`` bf16/fp16 delta on the same device.
        effective_query: externally sliced ``[D]`` fp32 query for this slot.
        inter_max/inter_exp_sum: externally sliced ``[tokens]`` fp32 Phase-1
            statistics for this slot.
        inter_numerator: externally sliced ``[tokens,D]`` fp32 Phase-1
            numerator.
        epsilon: positive Python float or scalar fp32 tensor.
    Returns:
        ``(h, partial_blocks)``; both are ``[tokens,D]`` fp32. The second
        output is ``partial_block + partial_delta.float()``.
    """
    assert partial_block.dtype == torch.float32, \
        f"partial_block 必须为 fp32,实际 {partial_block.dtype}"
    assert partial_block.dim() == 2 and int(partial_block.shape[1]) >= 1, \
        "partial_block 必须为 [tokens,D] 且 D>=1"
    assert partial_delta.dtype in (torch.bfloat16, torch.float16), \
        f"partial_delta 必须为 bf16/fp16,实际 {partial_delta.dtype}"
    assert tuple(partial_delta.shape) == tuple(partial_block.shape), \
        "partial_delta 必须与 partial_block 同 shape"
    assert partial_block.is_contiguous() and partial_delta.is_contiguous(), \
        "partial_block / partial_delta 必须 contiguous"
    assert partial_block.device == partial_delta.device, \
        "partial_block / partial_delta 必须同 device"

    tokens, d = (int(size) for size in partial_block.shape)
    assert effective_query.dtype == torch.float32 \
        and effective_query.dim() == 1 \
        and int(effective_query.shape[0]) == d, \
        f"effective_query 必须为 [{d}] fp32"
    assert effective_query.is_contiguous() \
        and effective_query.device == partial_block.device, \
        "effective_query 必须 contiguous 且与 partial_block 同 device"

    stats_shape = (tokens,)
    for name, value in (
        ("inter_max", inter_max),
        ("inter_exp_sum", inter_exp_sum),
    ):
        assert value.dtype == torch.float32 \
            and tuple(value.shape) == stats_shape, \
            f"{name} 必须为 [tokens] fp32"
        assert value.is_contiguous() and value.device == partial_block.device, \
            f"{name} 必须 contiguous 且与 partial_block 同 device"

    assert inter_numerator.dtype == torch.float32, \
        "inter_numerator 必须为 fp32"
    assert tuple(inter_numerator.shape) == (tokens, d), \
        "inter_numerator 必须为 [tokens,D]"
    assert inter_numerator.is_contiguous() \
        and inter_numerator.device == partial_block.device, \
        "inter_numerator 必须 contiguous 且与 partial_block 同 device"

    if isinstance(epsilon, torch.Tensor):
        assert epsilon.dtype == torch.float32 and epsilon.numel() == 1, \
            "Tensor epsilon 必须为标量 fp32 Tensor"
        assert epsilon.is_contiguous() and epsilon.device == partial_block.device, \
            "Tensor epsilon 必须 contiguous 且与 partial_block 同 device"
        epsilon_tensor = epsilon.reshape(1)
    else:
        assert isinstance(epsilon, (float, int)) and float(epsilon) > 0.0, \
            "epsilon 必须为正 float 或标量 fp32 Tensor"
        epsilon_tensor = None

    h = torch.empty_like(partial_block)
    partial_blocks = torch.empty_like(partial_block)

    if tokens == 0:
        return h, partial_blocks

    if epsilon_tensor is None:
        epsilon_tensor = torch.tensor(
            float(epsilon), dtype=torch.float32,
            device=partial_block.device,
        ).reshape(1)

    block_num = _launch_block_num(tokens)
    compiled = _compiled_update_kernel(
        tokens, d, partial_delta.dtype, block_num,
    )
    compiled(
        from_torch_npu(partial_block.reshape(tokens, d)),
        from_torch_npu(partial_delta.reshape(tokens, d)),
        from_torch_npu(effective_query.reshape(1, d)),
        from_torch_npu(inter_max.reshape(tokens, 1)),
        from_torch_npu(inter_exp_sum.reshape(tokens, 1)),
        from_torch_npu(inter_numerator.reshape(tokens, d)),
        from_torch_npu(epsilon_tensor),
        from_torch_npu(h.reshape(tokens, d)),
        from_torch_npu(partial_blocks.reshape(tokens, d)),
    )
    return h, partial_blocks


# The eager launcher below materializes CANNBotDSL runtime tensors from raw
# NPU pointers.  Keep that pointer work outside TorchDynamo by representing the
# functional whole-network call as one dispatcher node.
_UPDATE_TORCH_LIBRARY = torch.library.Library("cannbot_attn_res", "FRAGMENT")
_UPDATE_TORCH_LIBRARY.define(
    "block_attn_res_update(Tensor partial_block, Tensor partial_delta, "
    "Tensor effective_query, Tensor inter_max, Tensor inter_exp_sum, "
    "Tensor inter_numerator, Tensor epsilon) -> (Tensor, Tensor)"
)


@torch.library.impl(_UPDATE_TORCH_LIBRARY, "block_attn_res_update", "Meta")
def _block_attn_res_update_meta(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    )
    return (
        partial_block.new_empty(partial_block.shape),
        partial_block.new_empty(partial_block.shape),
    )


@torch.library.impl(
    _UPDATE_TORCH_LIBRARY, "block_attn_res_update", "PrivateUse1",
)
def _block_attn_res_update_npu(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _block_attn_res_update_eager(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon,
    )


def block_attn_res_update(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    effective_query: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    inter_numerator: torch.Tensor,
    epsilon: float | torch.Tensor = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-safe public entry for the fused online-softmax update."""
    if partial_block.device.type != "npu":
        return _block_attn_res_update_eager(
            partial_block,
            partial_delta,
            effective_query,
            inter_max,
            inter_exp_sum,
            inter_numerator,
            epsilon,
        )

    if isinstance(epsilon, torch.Tensor):
        epsilon_tensor = epsilon.reshape(1)
    else:
        epsilon_tensor = torch.scalar_tensor(
            float(epsilon), dtype=torch.float32,
            device=partial_block.device,
        ).reshape(1)
    return torch.ops.cannbot_attn_res.block_attn_res_update.default(
        partial_block,
        partial_delta,
        effective_query,
        inter_max,
        inter_exp_sum,
        inter_numerator,
        epsilon_tensor,
    )
