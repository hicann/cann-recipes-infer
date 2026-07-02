#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""CPU MoE 数值对账（MXFP4）：KTMoEWrapper(LLAMAFILE, MXFP4 GGUF) vs torch 参考。

参考权重直接 dequant **同一份原生 MXFP4 权重**（不是 W8A8），cand 与 ref 用完全相同的母权重，
唯一数值损失源是 kernel 内部把激活量化到 Q8_0（ggml_vec_dot_mxfp4_q8_0 的 vec_dot_type=Q8_0），
因此阈值 cosine >= 0.999。

dequant 语义见 ``verify_mxfp4_layer.dequant_native``：value = FP4_TABLE[nibble] *
2^(e-127)，byte i -> Kpos 2i(lo),2i+1(hi)（原生 consecutive 排布，与转换器一致）。

用法::
  python tools/cpu_moe_reference_check_mxfp4.py \
    --model-dir <原生 MXFP4 模型目录> \
    --gguf <GGUF 缓存目录>/dsv4_layer16_mxfp4.gguf \
    --layer-idx 16 --batch 4 --seed 1
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F

# Side-effect import: loading torch_npu registers the Ascend NPU device/ops. Bound to
# None when unavailable so callers can give a clear error (see _resolve_device_and_stream).
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

# Offline single-layer check: force the synchronous submit/sync path so the CPU MoE output
# actually lands (the NPU stream-callback path returns all-zero in this isolated context;
# production serving uses the real graph stream path and is unaffected). The flag is read at
# runtime by kt-kernel, so setting it here at import time (before main) is early enough.
os.environ.setdefault("KT_FORCE_SYNC_SUBMIT", "1")

logger = logging.getLogger("kt.tools.cpu_moe_reference_check_mxfp4")

# The three per-expert projections (gate=w1, up=w3, down=w2), grouped to keep the
# reference-forward signatures under the arg-count limit. Fields hold either stacked
# tensors (E,I,H)/(E,H,I) or per-expert {eid: tensor} dicts.
Experts = namedtuple("Experts", ["w1", "w3", "w2"])

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    # tools dir on sys.path for sibling-module imports (convert_*/verify_*); insert(0) intentional
    sys.path.insert(0, str(_HERE))  # pylint: disable=no-use-sys-path-insert

# The imports below depend on the tools dir being added to sys.path above, so they come after
# that setup (E402 is expected, not a real violation).
from convert_mxfp4_layer_to_gguf import (  # noqa: E402
    _load_weight_map,
    _detect_experts_prefix,
    _open_shard,
    _as_u8,
)
from verify_mxfp4_layer import dequant_native  # noqa: E402


def _setup_logging() -> None:
    """INFO -> stdout, WARNING+ -> stderr (preserves the original print stream split)."""
    out = logging.StreamHandler(sys.stdout)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[out, err])


# --- 通用 helper（pin_memory patch / reference forward / cosine / device·stream）---
def _ensure_pin_memory_or_patch() -> None:
    """探测 host pin_memory 分配器；不可用则把 torch.empty 的 pin_memory=True 降级为 False
    （KTMoEWrapper 会请求 pin memory，但本离线对账不真正需要 pin）。"""
    try:
        _ = torch.empty(1, dtype=torch.bool, device="cpu", pin_memory=True)
        return
    except RuntimeError as e:
        if "pin_memory" not in str(e).lower():
            raise
    # Intentional monkey-patch of torch.empty (drop pin_memory when the NPU has no pin allocator;
    # offline reference-check only, numerics unchanged). Reassigning a stdlib function + the
    # untyped wrapper make mypy complain, hence the type: ignore below.
    _orig_empty = torch.empty

    def _empty_no_pin(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("pin_memory", False)
        return _orig_empty(*args, **kwargs)

    torch.empty = _empty_no_pin  # type: ignore[assignment]
    logger.error("[warn] pin_memory allocator 不可用，已把 pin_memory=True 降级为 False。")


def reference_moe_forward(hidden, topk_ids, topk_weights, experts: Experts):
    """Pure-PyTorch DSv4 MoE forward (fp32 累加，慢但精确)。

    experts.w1/.w3=(E,I,H)、experts.w2=(E,H,I)。每 token i 对每个 topk k：
    out_i += weight * (SiLU(h@w1[e].T) * (h@w3[e].T)) @ w2[e].T。
    """
    w1, w3, w2 = experts
    n_tokens, hidden_dim = hidden.shape
    top_k = topk_ids.shape[1]
    h = hidden.float()
    out = torch.zeros(n_tokens, hidden_dim, dtype=torch.float32)
    for i in range(n_tokens):
        for k in range(top_k):
            e = int(topk_ids[i, k].item())
            w_eff = float(topk_weights[i, k].item())
            gate = h[i] @ w1[e].t()
            up = h[i] @ w3[e].t()
            act = F.silu(gate) * up
            d = act @ w2[e].t()
            out[i].add_(w_eff * d)
    return out.to(hidden.dtype)


def cosine_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


def _resolve_device_and_stream(want: str, npu_id: int):
    """为 cpu_infer 选 device + 匹配 stream handle（非 cpu 设备 handle=0 会让 CPU 任务不触发→输出全零）。"""
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available()=False")
        device = torch.device("cuda", 0)
        return device, int(torch.cuda.current_stream(device).cuda_stream)
    if want == "npu":
        if torch_npu is None:
            raise RuntimeError("--device npu but `import torch_npu` failed（torch/torch_npu 版本须一致）。")
        if not torch.npu.is_available():
            raise RuntimeError("--device npu but torch.npu.is_available()=False")
        device = torch.device("npu", npu_id)
        return device, int(torch.npu.current_stream(device).npu_stream)
    return torch.device("cpu"), 0


_ensure_pin_memory_or_patch()


def load_routed_experts_fp32(model_dir: Path, layer_idx: int, expert_ids):
    """Native MXFP4 -> fp32 dicts {eid: tensor} for only the routed experts (fast)."""
    weight_map = _load_weight_map(model_dir)
    prefix = _detect_experts_prefix(weight_map, layer_idx)
    cache: dict = {}
    w1d, w3d, w2d = {}, {}, {}
    for e in sorted(set(int(x) for x in expert_ids)):
        for proj, dst in (("w1", w1d), ("w3", w3d), ("w2", w2d)):
            wk = f"{prefix}.{e}.{proj}.weight"
            sk = f"{prefix}.{e}.{proj}.scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            dst[e] = torch.from_numpy(dequant_native(_as_u8(h.get_tensor(wk)), _as_u8(h.get_tensor(sk))))
    return w1d, w3d, w2d


def reference_moe_forward_dict(hidden, topk_ids, topk_weights, experts: Experts):
    """Pure-PyTorch MoE forward using per-expert dicts (only routed experts present)."""
    w1d, w3d, w2d = experts
    n_tokens, hidden_dim = hidden.shape
    top_k = topk_ids.shape[1]
    h = hidden.float()
    out = torch.zeros(n_tokens, hidden_dim, dtype=torch.float32)
    for i in range(n_tokens):
        for k in range(top_k):
            e = int(topk_ids[i, k].item())
            wf = float(topk_weights[i, k].item())
            gate = h[i] @ w1d[e].t()
            up = h[i] @ w3d[e].t()
            out[i].add_(wf * ((F.silu(gate) * up) @ w2d[e].t()))
    return out


def main() -> int:
    _setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-experts-per-tok", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cpuinfer-threads", type=int, default=24)
    ap.add_argument("--threadpool-count", type=int, default=8)
    ap.add_argument("--chunked-prefill-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="npu", choices=("npu", "cuda", "cpu"))
    ap.add_argument("--npu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--rel-tol", type=float, default=0.03)
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    gguf_file = args.gguf.expanduser().resolve()
    if not gguf_file.is_file():
        logger.error(f"ERROR: GGUF 不存在: {gguf_file}")
        return 2

    device, stream_handle = _resolve_device_and_stream(args.device, args.npu_id)
    logger.info(f"[env] device={device} stream_handle={stream_handle}")
    if stream_handle == 0 and device.type != "cpu":
        logger.error("ERROR: stream_handle=0 非 cpu device → task 不触发")
        return 2

    torch.manual_seed(args.seed)
    hidden_cpu = torch.randn(args.batch, args.hidden_size, dtype=torch.bfloat16)
    topk_ids_cpu = torch.randint(0, args.num_experts, (args.batch, args.num_experts_per_tok), dtype=torch.long)
    topk_weights_cpu = torch.softmax(torch.randn(args.batch, args.num_experts_per_tok, dtype=torch.float32), dim=-1)
    routed = sorted(set(int(x) for x in topk_ids_cpu.flatten().tolist()))
    logger.info(f"[run] hidden={tuple(hidden_cpu.shape)} routing[0]={topk_ids_cpu[0].tolist()} "
                f"routed_experts={len(routed)}")

    logger.info(f"[ref] dequant native MXFP4 layer {args.layer_idx} (routed {len(routed)} experts)...")
    w1d, w3d, w2d = load_routed_experts_fp32(model_dir, args.layer_idx, routed)

    try:
        from kt_kernel import KTMoEWrapper
    except ImportError as e:
        logger.error(f"ERROR: import kt_kernel 失败: {e}")
        return 2

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    wrapper = KTMoEWrapper(
        layer_idx=args.layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=str(gguf_file),
        chunked_prefill_size=args.chunked_prefill_size,
        method="LLAMAFILE",
        numa_nodes=None,
    )
    wrapper.load_weights()

    logger.info("[ref]  forward (pure pytorch fp32)...")
    ref_out = reference_moe_forward_dict(hidden_cpu, topk_ids_cpu, topk_weights_cpu, Experts(w1d, w3d, w2d))

    logger.info(f"[cand] forward KTMoEWrapper(MXFP4) on {device} stream={stream_handle}...")
    cand_out = wrapper.forward(hidden_cpu.to(device), topk_ids_cpu.to(device),
                               topk_weights_cpu.to(device), stream_handle)
    if isinstance(cand_out, (list, tuple)):
        cand_out = cand_out[0]
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

    a = cand_out.detach().cpu().float()
    b = ref_out.float()
    if a.shape != b.shape:
        logger.error(f"FAIL: shape cand={tuple(a.shape)} ref={tuple(b.shape)}")
        return 1

    cand_finite = bool(torch.isfinite(a).all().item())
    cos = float(cosine_sim(a, b).item())
    max_abs = float((a - b).abs().max().item())
    max_rel = max_abs / (float(b.abs().max().item()) + 1e-12)

    logger.info("\n=== 数值对账 (MXFP4) ===")
    logger.info(f"  cosine_sim   = {cos:.6f}     (期望 >= {args.cos_min:.4f})")
    logger.info(f"  max_abs_err  = {max_abs:.4e}")
    logger.info(f"  max_rel_err  = {max_rel:.4%}   (期望 <= {args.rel_tol:.2%})")
    logger.info(f"  ref  L2/max  = {b.norm():.4e} / {b.abs().max():.4e}")
    logger.info(f"  cand L2/max  = {a.norm():.4e} / {a.abs().max():.4e}")
    logger.info(f"  finite(cand) = {cand_finite}")
    logger.info(f"  ref [0,:6]   = {[round(float(x),5) for x in b[0,:6].tolist()]}")
    logger.info(f"  cand[0,:6]   = {[round(float(x),5) for x in a[0,:6].tolist()]}")
    logger.info(f"  token0 cand nonzero={(a[0].abs()>1e-7).sum().item()}/{a[0].numel()}\n")

    mismatch = (not cand_finite) or math.isnan(cos) or cos < args.cos_min or max_rel > args.rel_tol
    if mismatch:
        logger.error("RESULT: FAIL — MXFP4 CPU MoE 与参考不一致（kernel/转换器/nibble 序 之一有误）。")
        return 1
    logger.info("RESULT: PASS — MXFP4 kernel 数值正确（唯一损失源为激活 Q8 量化）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
