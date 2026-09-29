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
"""DSV4.1-Flash MXFP4 CPU-offload numerics check (M0b). Pure CPU, never opens an NPU.

Not a pytest module on purpose (no ``test_`` prefix): it needs the 476 GB
DeepSeek-V4.1-Flash checkpoint.

What it checks, per layer:
  1. E8M0 byte round trip: scales read back out of the kt BufferB
     (``write_weight_scale_to_buffer``, compact 1-byte scale mode) are
     byte-identical to the safetensors bytes. A positive control flips one
     byte of the expected side and must report exactly one mismatch.
  2. Forward: ``KTMoEWrapper(method="MXFP4", swiglu_limit=10)`` driven through
     ``forward()`` (pinned-buffer path, sync submit) with a ``gpu_experts_mask``
     that leaves only a subset on the CPU, vs an independent torch reference
     (E2M1 LUT x 2^(byte-127), DSV4.1 SwiGLU clamps, sqrtsoftplus-normalised
     top-k weights x route_scale 1.5). GPU-side experts contribute 0.
     Tokens: 1, 4, 32; routing mixes CPU and GPU experts inside one token's
     top-k, plus one all-CPU and one all-GPU token.
  3. Floor first, then threshold, two tiers, both must hold:
     A (vs fp32 reference): floor = bf16-emulated vs fp32 reference on the same
       data; tau = 2 * max(floor, 2^-8).  Catches gross errors only.
     B (vs bf16-emulated reference): floor = fp32- vs fp64-accumulated
       emulations (accumulation-order noise); tau = 2 * max(floor, 2^-12) for
       rel_l2 and 2 * max(floor, 2^-8) for the worst output channel.
       Sees sub-bf16 defects such as one scale byte.
  4. Mutation arms (applied to the reference side) must go red: nibble order,
     one expert's whole w2 scale +1, and (clamp-driving inputs) no clamp.
     A single scale byte +1 is a detection-limit probe: reported, not required.

Run (private site dir, no global install):
  env -u LD_PRELOAD PYTHONPATH=<kt_site> python3 dsv41_mxfp4_offload_check.py
"""

from __future__ import annotations

import argparse
import builtins
import json
import logging
import os
import sys
import time
from collections import namedtuple

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
LOGGER = logging.getLogger("dsv41_mxfp4_offload_check")


def report(message):
    """Single exit point for the report lines."""
    LOGGER.info("%s", message)


# ---- NPU lockout: before torch / kt_kernel import ----------------------------
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"  # do not auto-import torch_npu
os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"  # forward(): submit()+sync(), no ACL callback
os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"  # never start kt's ACL poller

# Record every attempted sglang import (experts_base._sglang_is_capture_mode).
_SGLANG_IMPORTS = []
_orig_import = builtins.__import__


def _tracking_import(name, *a, **kw):
    if name == "sglang" or name.startswith("sglang."):
        _SGLANG_IMPORTS.append(name)
    return _orig_import(name, *a, **kw)


builtins.__import__ = _tracking_import

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Drop pin_memory from the torch factory calls.
for _name in ("zeros", "empty", "full"):
    _orig = getattr(torch, _name)

    def _nopin(*a, __orig=_orig, **kw):
        kw.pop("pin_memory", None)
        return __orig(*a, **kw)

    setattr(torch, _name, _nopin)

import kt_kernel  # noqa: E402
import kt_kernel.experts_base as _eb  # noqa: E402


def _no_ascend_callback_worker():
    """Stand-in for kt's ACL callback worker starter; this run never opens an NPU."""


setattr(_eb, "_ensure_ascend_callback_worker", _no_ascend_callback_worker)
from kt_kernel.experts import KTMoEWrapper  # noqa: E402
from kt_kernel.utils.amx import _select_mxfp4_backend  # noqa: E402
from safetensors import safe_open  # noqa: E402

if "torch_npu" in sys.modules:
    raise RuntimeError("torch_npu got imported; refusing to run")

CKPT = os.environ.get("MODEL", "/workspace/models/DeepSeek-V4.1-Flash")
HIDDEN, INTER, N_EXP, TOPK = 5120, 2304, 384, 6
LIMIT, ROUTE_SCALE = 10.0, 1.5
PROBE_ARM = "1 scale byte+1"  # single E8M0 byte of one expert's w2: detection-limit probe
BF16_ULP = 2.0**-8  # relative spacing of bf16 at 1.0 = 3.9e-3

RefMode = namedtuple("RefMode", "limit emulate_bf16 acc64")

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class Ckpt:
    def __init__(self, path):
        with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as fp:
            idx = json.load(fp)["weight_map"]
        self.path, self.idx, self.h = path, idx, {}

    def raw(self, key):
        fn = self.idx[key]
        if fn not in self.h:
            self.h[fn] = safe_open(os.path.join(self.path, fn), framework="pt")
        return self.h[fn].get_tensor(key)

    def bytes_(self, key):
        """Byte view -- never a numeric conversion."""
        return self.raw(key).contiguous().view(torch.uint8)


def dequant(w_u8, s_u8, swap_nibbles=False):
    lo, hi = (w_u8 & 0x0F).long(), (w_u8 >> 4).long()
    if swap_nibbles:
        lo, hi = hi, lo
    n, kh = w_u8.shape
    codes = torch.empty(n, kh * 2, dtype=torch.long)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    scale = torch.exp2(s_u8.to(torch.float32) - 127.0)  # integer byte -> exponent
    return E2M1[codes] * scale.repeat_interleave(32, dim=1)


def bf(t):
    return t.to(torch.bfloat16).to(torch.float32)


def expert_out(x, weights, mode):
    """
    One routed expert. weights is the (w1, w3, w2) triple.

    mode.emulate_bf16 rounds at kt's bf16 storage points (gate/up GEMM outputs, activation,
    down output; fp4-moe.hpp keeps them as ggml_bf16_t); mode.acc64 runs the GEMMs in float64
    (changes accumulation only).
    """
    w1, w3, w2 = weights
    if mode.acc64:
        x, w1, w3, w2 = x.double(), w1.double(), w3.double(), w2.double()
    g, u = (x @ w1.T).float(), (x @ w3.T).float()
    if mode.emulate_bf16:
        g, u = bf(g), bf(u)
    if mode.limit > 0:
        g = torch.clamp(g, max=mode.limit)  # gate: one-sided (inference/model.py:847)
        u = torch.clamp(u, min=-mode.limit, max=mode.limit)  # up: symmetric (inference/model.py:846)
    h = torch.nn.functional.silu(g) * u
    if mode.emulate_bf16:
        h = bf(h)
    y = (h.double() @ w2.T).float() if mode.acc64 else h @ w2.T
    return bf(y) if mode.emulate_bf16 else y


def moe_ref(x, routing, ref_w, gpu_set, mode):
    ids, wts = routing
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32)
    for e in sorted({int(v) for v in ids.flatten()} - gpu_set):
        tok, pos = torch.where(ids == e)
        y = expert_out(x[tok], ref_w[e], mode)
        out.index_add_(0, tok, wts[tok, pos, None] * y)
    return bf(out) if mode.emulate_bf16 else out


def rel_l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def col_rel_max(a, b):
    """Worst output channel: localises single-row (single scale byte) defects."""
    d = (a.double() - b.double()).norm(dim=0)
    n = b.double().norm(dim=0)
    keep = n > 1e-3 * n.max()
    return float((d[keep] / n[keep]).max())


def judge(got, target_fp32, target_emu, taus):
    """
    Compare kt's output with both reference tiers.

    Returns the four metrics (tier A rel/col, tier B rel/col) and whether each is within its tau.
    """
    metrics = (rel_l2(got, target_fp32), col_rel_max(got, target_fp32),
               rel_l2(got, target_emu), col_rel_max(got, target_emu))
    return metrics, all(m <= t for m, t in zip(metrics, taus))


def make_routing(rng, n_tok, cpu_ids, gpu_pool, e_force):
    pool = np.array(cpu_ids + gpu_pool)
    ids = np.stack([rng.choice(pool, TOPK, replace=False) for _ in range(n_tok)])
    if n_tok >= 4:
        ids[0] = rng.choice(cpu_ids, TOPK, replace=False)  # all-CPU token
        ids[1] = rng.choice(gpu_pool, TOPK, replace=False)  # all-GPU token -> exact 0
        ids[2, :3] = rng.choice(cpu_ids, 3, replace=False)  # 3 CPU + 3 GPU
        ids[2, 3:] = rng.choice(gpu_pool, 3, replace=False)
    else:
        ids[0, :3] = rng.choice(cpu_ids, 3, replace=False)
        ids[0, 3:] = rng.choice(gpu_pool, 3, replace=False)
    if e_force not in ids[0]:
        ids[0, 0] = e_force
    if any(len(set(r)) != TOPK for r in ids.tolist()):
        raise RuntimeError(f"routing produced duplicate experts inside one token's top-{TOPK}")
    logits = torch.from_numpy(rng.standard_normal((n_tok, TOPK)).astype(np.float32))
    s = torch.nn.functional.softplus(logits).sqrt()  # score_func sqrtsoftplus
    w = s / (s.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE  # norm_topk_prob, route_scale
    return torch.from_numpy(ids.astype(np.int64)), w.contiguous()


def scale_roundtrip(wrapper, ck, layer, cpu_ids):
    """Read scales back out of kt's BufferB and compare bytes with safetensors."""
    res = {"experts": 0, "bytes": 0, "mismatch": 0, "bad_bits": 0, "distinct": set(),
           "w_bytes": 0, "w_mismatch": 0}
    for e in cpu_ids:
        w13w = torch.zeros(2 * INTER * HIDDEN // 2, dtype=torch.uint8)
        w13s = torch.zeros(2 * INTER * HIDDEN // 32, dtype=torch.bfloat16)
        w2w = torch.zeros(HIDDEN * INTER // 2, dtype=torch.uint8)
        w2s = torch.zeros(HIDDEN * INTER // 32, dtype=torch.bfloat16)
        wrapper.submit_write_weight_scale_to_buffer(
            1, e, [w13w.data_ptr()], [w13s.data_ptr()], [w2w.data_ptr()], [w2s.data_ptr()]
        )
        wrapper.sync_write_weight_scale_to_buffer()
        base = f"layers.{layer}.ffn.experts.{e}"
        exp_s = torch.cat([ck.bytes_(f"{base}.w1.scale").flatten(), ck.bytes_(f"{base}.w3.scale").flatten(),
                           ck.bytes_(f"{base}.w2.scale").flatten()])
        got_bits = torch.cat([w13s.view(torch.int16), w2s.view(torch.int16)]).to(torch.int32) & 0xFFFF
        res["bad_bits"] += int(((got_bits & 0x807F) != 0).sum())  # sign / mantissa must be 0
        got_s = ((got_bits >> 7) & 0xFF).to(torch.uint8)
        res["mismatch"] += int((got_s != exp_s).sum())
        res["bytes"] += exp_s.numel()
        res["distinct"] |= set(torch.unique(got_s).tolist())
        exp_w = torch.cat([ck.bytes_(f"{base}.w1.weight").flatten(), ck.bytes_(f"{base}.w3.weight").flatten(),
                           ck.bytes_(f"{base}.w2.weight").flatten()])
        got_w = torch.cat([w13w, w2w])
        res["w_mismatch"] += int((got_w != exp_w).sum())
        res["w_bytes"] += exp_w.numel()
        res["experts"] += 1
        if res["experts"] == 1:  # positive control: one flipped byte
            ctl = exp_s.clone()
            ctl[12345] ^= 0x01
            res["control_mismatch"] = int((got_s != ctl).sum())
            # weight-side positive control: another expert's bytes must differ
            other = [v for v in cpu_ids if v != e][0]
            ob = f"layers.{layer}.ffn.experts.{other}"
            ctl_w = torch.cat([ck.bytes_(f"{ob}.w1.weight").flatten(),
                               ck.bytes_(f"{ob}.w3.weight").flatten(),
                               ck.bytes_(f"{ob}.w2.weight").flatten()])
            res["w_control_mismatch"] = int((got_w != ctl_w).sum())
            res["w_control_expert"] = other
    res["distinct"] = sorted(res["distinct"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,3")
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1, help="threadpool_count")
    ap.add_argument("--n-cpu", type=int, default=0, help="use the first N experts instead")
    ap.add_argument("--cpu-ids", default="")
    ap.add_argument("--nodes", default="", help="comma separated; overrides --numa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x-std", type=float, default=1.0)
    ap.add_argument("--roundtrip-only", action="store_true")
    args = ap.parse_args()

    report(f"kt_kernel {kt_kernel.__version__} from {kt_kernel.__file__}")
    report(f"__cpu_variant__={kt_kernel.__cpu_variant__}  backend={_select_mxfp4_backend().__name__}")
    report(f"KT_MXFP4_COMPACT_SCALES={os.environ.get('KT_MXFP4_COMPACT_SCALES', '1 (default)')}")
    ok = kt_kernel.__cpu_variant__ == "avx512_bf16"
    ck = Ckpt(CKPT)
    rng = np.random.default_rng(args.seed)
    # CPU-resident experts: spread over the id range, include both ends.
    cpu_ids = ([int(v) for v in args.cpu_ids.split(",")] if args.cpu_ids
               else list(range(args.n_cpu)) if args.n_cpu
               else [0, 5, 37, 100, 128, 191, 255, 256, 300, 333, 370, 383])
    if len(cpu_ids) < 2:
        sys.exit(f"need at least 2 CPU experts, the weight-side control compares against "
                 f"another one; got cpu_ids={cpu_ids}")
    if not args.roundtrip_only and len(cpu_ids) < 4:
        sys.exit(f"full mode needs at least 4 CPU experts, the scale arm uses cpu_ids[3]; "
                 f"got cpu_ids={cpu_ids}. For the byte round-trip audit alone add --roundtrip-only")
    gpu_pool = [1, 2, 64, 150, 200, 222, 260, 280, 310, 350, 377, 382]
    mask = torch.ones(N_EXP, dtype=torch.bool)
    mask[cpu_ids] = False
    gpu_set = set(range(N_EXP)) - set(cpu_ids)
    summary, probe_hits, n_required = [], [], 0

    _nodes = [int(v) for v in args.nodes.split(",")] if args.nodes else [args.numa]
    if len(_nodes) != args.tp:
        sys.exit(f"threadpool_count={args.tp} needs {args.tp} nodes, got {_nodes}")
    report(f"[cfg] threadpool_count={args.tp} numa_nodes={_nodes}")

    for layer in [int(v) for v in args.layers.split(",")]:
        report(f"\n===== layer {layer}: CPU experts {cpu_ids} ({len(cpu_ids)}/{N_EXP}); "
              f"GPU pool used in routing {gpu_pool}")
        wr = KTMoEWrapper(
            layer_idx=layer, num_experts=N_EXP, num_experts_per_tok=TOPK, hidden_size=HIDDEN,
            moe_intermediate_size=INTER, gpu_experts_mask=mask, cpuinfer_threads=args.threads,
            threadpool_count=args.tp, weight_path=CKPT, chunked_prefill_size=64, method="MXFP4",
            numa_nodes=_nodes, swiglu_limit=LIMIT,
        )
        wr.load_weights(torch.arange(N_EXP, dtype=torch.int64))

        _t0 = time.perf_counter()
        rt = scale_roundtrip(wr, ck, layer, cpu_ids)
        _dt = time.perf_counter() - _t0
        _gib = (rt["bytes"] + rt["w_bytes"]) / (1 << 30)
        report(f"[bw] kt BufferB -> host: {_gib:.3f} GiB in {_dt:.2f}s = {_gib / _dt:.2f} GiB/s")
        rt_ok = (rt["mismatch"] == 0 and rt["bad_bits"] == 0 and rt["control_mismatch"] == 1
                 and len(rt["distinct"]) > 1 and rt["bytes"] > 0
                 and rt["w_mismatch"] == 0 and rt["w_bytes"] > 0
                 and rt["w_control_mismatch"] > 0)
        report(f"[roundtrip] experts={rt['experts']} scale_bytes={rt['bytes']} mismatch={rt['mismatch']} "
              f"sign/mantissa_bits_set={rt['bad_bits']} distinct={rt['distinct']} "
              f"control(1 flipped)->{rt['control_mismatch']}  weight_bytes={rt['w_bytes']} "
              f"weight_mismatch={rt['w_mismatch']} "
              f"w_control(expert {rt['w_control_expert']})->{rt['w_control_mismatch']} "
              f" [{'PASS' if rt_ok else 'FAIL'}]")
        ok &= rt_ok
        if args.roundtrip_only:
            continue

        ref_w, mut_w = {}, {}
        for e in cpu_ids:
            b = f"layers.{layer}.ffn.experts.{e}"
            ws = [(ck.bytes_(f"{b}.{p}.weight"), ck.bytes_(f"{b}.{p}.scale")) for p in ("w1", "w3", "w2")]
            ref_w[e] = tuple(dequant(w, s) for w, s in ws)
            mut_w[e] = {"nibble": tuple(dequant(w, s, swap_nibbles=True) for w, s in ws)}
        # scale arms on one expert's w2 (down): whole tensor +1 (x2) / single byte +1
        e0 = cpu_ids[3]
        e0_base = f"layers.{layer}.ffn.experts.{e0}"
        w2b, s2b = ck.bytes_(f"{e0_base}.w2.weight"), ck.bytes_(f"{e0_base}.w2.scale")
        s_all = s2b.clone() + 1
        s_one = s2b.clone()
        row = 777
        s_one[row, 5] += 1
        w1r, w3r, _ = ref_w[e0]
        arm_expert = {"scale+1(expert w2)": (w1r, w3r, dequant(w2b, s_all)),
                      "1 scale byte+1": (w1r, w3r, dequant(w2b, s_one))}

        for n_tok in [int(v) for v in args.tokens.split(",")]:
            for xs, tag in ((args.x_std, f"x~N(0,{args.x_std:g})"), (8.0, "x~N(0,8) clamp-driving")):
                if n_tok == 1 and xs != args.x_std:
                    continue
                x = (torch.from_numpy(rng.standard_normal((n_tok, HIDDEN)).astype(np.float32))
                     * xs).to(torch.bfloat16)
                routing = make_routing(rng, n_tok, cpu_ids, gpu_pool, e0)
                ids, wts = routing
                got = wr.forward(x, ids, wts, 0).float().clone()
                xf = x.float()
                ref = moe_ref(xf, routing, ref_w, gpu_set, RefMode(LIMIT, False, False))
                emu = moe_ref(xf, routing, ref_w, gpu_set, RefMode(LIMIT, True, False))
                emu64 = moe_ref(xf, routing, ref_w, gpu_set, RefMode(LIMIT, True, True))
                # Tier A (vs fp32 reference): floor = what an ideal bf16 pipeline loses.
                floor_a, floor_a_col = rel_l2(emu, ref), col_rel_max(emu, ref)
                tau_a, tau_a_col = 2 * max(floor_a, BF16_ULP), 2 * max(floor_a_col, BF16_ULP)
                # Tier B (vs bf16-emulated reference): floor = accumulation-order noise,
                # measured as fp32- vs fp64-accumulated emulations; never below 1 bf16 ULP.
                floor_b, floor_b_col = rel_l2(emu, emu64), col_rel_max(emu, emu64)
                tau_b, tau_b_col = 2 * max(floor_b, BF16_ULP / 16), 2 * max(floor_b_col, BF16_ULP)
                taus = (tau_a, tau_a_col, tau_b, tau_b_col)

                ncl = tot = 0  # how much of the clamp is actually exercised
                for e in {int(v) for v in ids.flatten()} - gpu_set:
                    w1, w3, _ = ref_w[e]
                    g, u = xf @ w1.T, xf @ w3.T
                    ncl += int((g > LIMIT).sum() + (u.abs() > LIMIT).sum())
                    tot += g.numel() + u.numel()
                v, main_ok = judge(got, ref, emu, taus)
                zero_tok = ""
                if n_tok >= 4:
                    z = got[1].abs().max().item()
                    zero_tok = f" allGPU-token max|y|={z:.1e}"
                    main_ok &= z == 0.0
                ok &= main_ok
                report(f"T={n_tok:<2d} {tag:22s} clamped={ncl}/{tot}{zero_tok}")
                report(f"      floors  A: rel={floor_a:.3e} col={floor_a_col:.3e} "
                       f"-> tau {tau_a:.3e}/{tau_a_col:.3e}   "
                       f"B: rel={floor_b:.3e} col={floor_b_col:.3e} -> tau {tau_b:.3e}/{tau_b_col:.3e}")
                report(f"      kt      A: rel={v[0]:.3e} col={v[1]:.3e}   B: rel={v[2]:.3e} col={v[3]:.3e}"
                       f"   [{'PASS' if main_ok else 'FAIL'}]")
                summary.append({"layer": layer, "T": n_tok, "x": tag, "fA": floor_a,
                                "tA": tau_a, "fB": floor_b, "tB": tau_b, "v": v})

                arms = {"nibble swap": ({e: mut_w[e]["nibble"] for e in cpu_ids}, LIMIT)}
                for name, trip in arm_expert.items():
                    arms[name] = ({**ref_w, e0: trip}, LIMIT)
                if xs != args.x_std:
                    arms = {"no clamp": (ref_w, 0.0)}
                for name, (wset, lim) in arms.items():
                    mv, m_ok = judge(got,
                                     moe_ref(xf, routing, wset, gpu_set, RefMode(lim, False, False)),
                                     moe_ref(xf, routing, wset, gpu_set, RefMode(lim, True, False)),
                                     taus)
                    red = not m_ok
                    probe = name == PROBE_ARM
                    report(f"      {'probe   ' if probe else 'mutation'} {name:20s} A: rel={mv[0]:.3e} "
                           f"col={mv[1]:.3e}   B: rel={mv[2]:.3e} col={mv[3]:.3e} "
                           f"-> {'RED' if red else 'green'}")
                    if probe:  # reported, not required
                        probe_hits.append(red)
                    else:
                        ok &= red
                        n_required += 1
        del wr

    if args.roundtrip_only:
        report("\n--roundtrip-only: forward and mutation arms skipped; "
              "the verdict is the per-layer roundtrip only")
    else:
        report(f"\nrequired mutation arms: {n_required} (all must be RED); "
              f"probe '{PROBE_ARM}': RED in {sum(probe_hits)}/{len(probe_hits)}")
        ok &= n_required > 0
    report(f"sglang import attempts during run: {len(_SGLANG_IMPORTS)} {sorted(set(_SGLANG_IMPORTS))}")
    report(f"torch_npu imported: {'torch_npu' in sys.modules}")
    report(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
