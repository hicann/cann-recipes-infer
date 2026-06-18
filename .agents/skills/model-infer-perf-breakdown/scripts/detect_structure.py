#!/usr/bin/env python3
# coding=utf-8
"""Step 2 — Layer detection (sample-driven primary + explore fallback).

Architecture-neutral terminology: "block" / "layer" / "component" denote one
repeating model unit (decoder layer, encoder layer, ViT block, hybrid attention
layer, MTP head…). The script makes no commitment to what a unit represents.

Modes:

  sample (primary): triggered by --structure-spec. Takes the user's Phase 0a
    model structure (phases × scheduled layer_compositions × component types) +
    stream_sample_ack.v1 samples (Phase 0b), extracts per-stream fingerprints,
    runs stream-local sliding-window matching, and emits structure_draft.json
    (mode=stream_sample_driven, schema_version=structure_draft.stream.v1,
    components[].op_indices / op_to_component / unmatched_op_indices /
    warnings / validation). Hard warnings block by default; after user review,
    --accept-warnings explicitly releases them. The matching algorithm lives in
    sample_matching.run_stream_sample_mode.

  explore: triggered by --explore. Runs candidates + a default landmark sweep
    without producing a structure_draft. Used when the user can't yet supply
    samples — gives them a kernel-frequency reference to construct 0a/0b.

Range scope: --op-start/--op-end restrict explore to a sub-range. Sample mode
ignores op-range flags (it always scans the full sequence — samples already
encode where each component lives).

No hardcoded model-specific constants — all thresholds are derived from the
data (max period explored = max(2, count // 4)).
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter

try:
    from scripts import sample_matching
except ImportError:
    import sample_matching  # type: ignore[no-redef]  # invoked directly from skill root


HARD_WARNING_CODES = {
    "primary_stream_missing",
    "sample_ack_mismatch",
    "stream_shape_mismatch",
    "composition_mismatch",
    "composition_schedule_missing",
    "auxiliary_stream_temporally_displaced",
}

AMBIGUOUS_WARNING_CODES = {
    "ambiguous_match",
    "op_membership_conflict",
    "auxiliary_stream_ambiguous",
    "stream_role_ambiguous",
}


# Auxiliary kernels: pure-format / pure-shape ops that rarely carry semantic
# block boundaries. Used to demote (not eliminate) them from anchor rankings.
# Match is exact-name after stripping a trailing version suffix (V2/V3/D),
# so fused kernels like TransposeBatchMatMul or DequantSwigluQuant — which
# start with one of these names but do real compute — are NOT misclassified.
AUX_KERNEL_NAMES = frozenset({
    "Cast", "Reshape", "Transpose", "DynamicQuant", "Dequant",
    "Squeeze", "Unsqueeze", "Slice", "Split", "View", "Contiguous",
    "Identity", "BroadcastTo", "ExpandDims", "Fill", "ZerosLike",
    "Concat", "ScatterNdUpdate", "RotaryMul", "AivKernel",
})
_VERSION_SUFFIX_RE = re.compile(r"(V\d+|D)$")


def load_ops(raw_path):
    with open(raw_path, "r") as f:
        data = json.load(f)
    return data["operators"], data.get("step_id")


def load_structure_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    if "phases" not in spec or not isinstance(spec["phases"], list):
        raise SystemExit(f"structure_spec {path} 缺 phases 列表")
    if "expected_components" not in spec:
        types = set()
        for ph in spec["phases"]:
            for comp in ph.get("layer_compositions", []):
                types.update(comp.get("components", []))
        spec["expected_components"] = sorted(types)
    return spec


def validate_inputs_json(path: str) -> dict:
    """Step 0 物料校验：prof_dir + model_script_paths 必须存在且非空。"""
    import os
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"[Phase 0 物料缺失] 找不到 {path}。\n"
            f"回 SKILL.md 的「前置物料 checklist」段：必须先和用户敲齐 prof 路径 + "
            f"模型脚本路径，写到 <run_dir>/inputs.json。"
        )
    except json.JSONDecodeError as e:
        raise SystemExit(f"[Phase 0 物料损坏] {path} 不是合法 JSON: {e}")
    paths = data.get("model_script_paths") or []
    if not isinstance(paths, list) or not paths:
        raise SystemExit(
            f"[Phase 0 物料缺失] {path} 的 model_script_paths 为空。\n"
            f"模型脚本路径是硬约束，不允许「没有/不方便」——agent 必须先要到，"
            f"否则 Phase 0b 的 stream sample ack 无法做。"
        )
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"[Phase 0 物料缺失] inputs.json 里的模型脚本路径不存在：{missing}"
        )
    if data.get("phase") in (None, ""):
        raise SystemExit(
            f"[Phase 0 物料缺失] {path} 的 phase 字段未填。多卡/多 phase 场景下必须先和"
            f"用户敲定 (phase, rank) 组合（prefill / decode 各一），不允许 agent 自作主张默认。"
        )
    if data.get("rank") is None:
        raise SystemExit(
            f"[Phase 0 物料缺失] {path} 的 rank 字段未填。必须和用户敲定具体 rank。"
        )
    return data


def validate_sample_ack(path: str, expected_components: list) -> dict:
    """Phase 0b ack 校验：每个 expected_component 都必须有 stream sample。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            ack = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"[Phase 0b 未 ack] 找不到 {path}。\n"
            f"回 SKILL.md 的「Phase 0b」段：必须先和用户敲定每个 component 的 "
            f"stream sample 范围，写到 <run_dir>/sample_ack.json 后再跑 detect。"
        )
    except json.JSONDecodeError as e:
        raise SystemExit(f"[Phase 0b ack 文件损坏] {path}: {e}")
    comps = ack.get("components") or {}
    if not isinstance(comps, dict):
        raise SystemExit(f"[Phase 0b ack 格式错] {path} 的 components 应为 dict")
    if ack.get("schema_version") != "stream_sample_ack.v1":
        raise SystemExit(
            f"[Phase 0b ack 格式错] {path} 必须使用 "
            f"schema_version='stream_sample_ack.v1'"
        )
    problems = []
    for name in expected_components:
        entry = comps.get(name)
        if not entry:
            problems.append(f"  - {name}: 完全缺失 stream sample 条目")
            continue
        stream_samples = entry.get("stream_samples") or []
        primary = [s for s in stream_samples if s.get("role") == "primary"]
        if len(primary) != 1:
            problems.append(f"  - {name}: 必须且只能有一个 primary stream sample")
        for i, sample in enumerate(stream_samples):
            if not sample.get("stream_id"):
                problems.append(f"  - {name}[{i}]: 缺 stream_id")
            if not sample.get("op_indices"):
                problems.append(f"  - {name}[{i}]: 缺 op_indices")
    if problems:
        raise SystemExit(
            "[Phase 0b 未 ack] sample_ack.json 不全，必须每个 expected_component 都有 "
            "用户 ack 过的 stream sample：\n" + "\n".join(problems)
        )
    return ack


def blocking_warnings(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w.get("code") in HARD_WARNING_CODES]


def ambiguous_warnings(warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if w.get("code") in AMBIGUOUS_WARNING_CODES]


def is_auxiliary(name):
    base = _VERSION_SUFFIX_RE.sub("", name)
    return base in AUX_KERNEL_NAMES


def resolve_op_range(args, ops):
    """Compute (start, end) inclusive op-index window from --op-start/--op-end.

    Returns (start, end, source_description) where description is logged.
    """
    n = len(ops)
    start = args.op_start if args.op_start is not None else 0
    end = args.op_end if args.op_end is not None else n - 1
    if start < 0 or end >= n or start > end:
        raise SystemExit(f"Invalid op range [{start}, {end}] for {n} ops")
    if args.op_start is None and args.op_end is None:
        return start, end, "full range"
    return start, end, f"--op-start={start} --op-end={end}"


def block_multiset(ops, start, end):
    """Shape-aware fingerprint: (normalized_name, input_shapes) -> count."""
    cnt = Counter()
    for i in range(start, end + 1):
        name = ops[i]["normalized_name"]
        shape = ops[i].get("input_shapes", "")
        cnt[(name, shape)] += 1
    return tuple(sorted(cnt.items()))


def segment_blocks(positions, n_anchors_per_block, range_end):
    """Segment anchor positions into N-tuples; last block extends to range_end."""
    blocks = []
    for li in range(0, len(positions), n_anchors_per_block):
        anchors = positions[li : li + n_anchors_per_block]
        if len(anchors) < n_anchors_per_block:
            break
        start = anchors[0]
        next_idx = li + n_anchors_per_block
        if next_idx < len(positions):
            end = positions[next_idx] - 1
        else:
            end = range_end
        blocks.append({
            "block_idx": len(blocks),
            "anchors": anchors,
            "start": start,
            "end": end,
        })
    return blocks


def gap_cv(positions, p):
    """Mean within-phase coefficient-of-variation of inter-anchor gaps.

    For TRUE anchors appearing P times per block, partitioning gaps by i mod P
    yields uniform groups (gap_cv ≈ 0). For non-structural anchors, gap_cv
    stays high at every period. Used as a period-discovery signal.
    """
    if len(positions) < 2 * p + 1:
        return None
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    cvs = []
    for k in range(p):
        grp = gaps[k::p]
        if len(grp) < 2:
            continue
        mean = sum(grp) / len(grp)
        if mean <= 0:
            continue
        cvs.append(statistics.pstdev(grp) / mean)
    return sum(cvs) / len(cvs) if cvs else None


def per_period_stats(ops, positions, max_p, range_total=None, range_end=None):
    """For p in 1..max_p, run segmentation and report block/cluster stats."""
    if range_total is None:
        range_total = len(ops)
    if range_end is None:
        range_end = len(ops) - 1
    rows = []
    for p in range(1, max_p + 1):
        if len(positions) // p < 2:
            break
        blocks = segment_blocks(positions, p, range_end)
        if len(blocks) < 2:
            continue
        sigs = {block_multiset(ops, b["start"], b["end"]) for b in blocks}
        sizes = [b["end"] - b["start"] + 1 for b in blocks]
        cov = sum(sizes) / range_total
        median = statistics.median(sizes)
        cv_size = statistics.pstdev(sizes) / median if median else float("inf")
        rows.append({
            "p": p,
            "num_blocks": len(blocks),
            "clusters": len(sigs),
            "compression": len(sigs) / len(blocks),
            "coverage": cov,
            "size_cv": cv_size,
            "gap_cv": gap_cv(positions, p),
        })
    return rows


def next_kernel_distribution(ops, positions):
    """Distinct kernels immediately following each anchor position."""
    cnt = Counter()
    for p in positions:
        if p + 1 < len(ops):
            cnt[ops[p + 1]["normalized_name"]] += 1
    return cnt


def build_candidates(ops, op_range, min_count=4, top_k=40):
    """Per-kernel candidacy report. No 'best' is chosen — AI picks in Step 2."""
    start, end = op_range
    by_name = {}
    for i in range(start, end + 1):
        by_name.setdefault(ops[i]["normalized_name"], []).append(i)

    range_total = end - start + 1
    candidates = []
    for name, positions in by_name.items():
        if len(positions) < min_count:
            continue
        aux = is_auxiliary(name)
        # max_p is data-driven: require ≥4 samples per phase group.
        max_p = max(2, len(positions) // 4)
        per_p = per_period_stats(ops, positions, max_p, range_total, end)
        if not per_p:
            continue
        nxt = next_kernel_distribution(ops, positions)
        candidates.append({
            "name": name,
            "count": len(positions),
            "auxiliary": aux,
            "first_pos": positions[0],
            "last_pos": positions[-1],
            "next_kernels_distinct": len(nxt),
            "next_kernels": [{"name": n, "count": c}
                             for n, c in nxt.most_common(5)],
            "per_period": per_p,
        })

    # Sort by structural-anchor quality: non-aux first (aux kernels are
    # format-only and rarely good semantic anchors), then low next-kernel
    # diversity (anchors that lead into a fixed template are structural),
    # then count desc (more occurrences = finer granularity).
    candidates.sort(key=lambda c: (
        c["auxiliary"], c["next_kernels_distinct"], -c["count"], c["name"]
    ))
    return candidates[:top_k]


def build_landmark_candidates(ops, op_range, landmark_name, top_k=40):
    """Find anchor candidates by reference to a high-signal landmark kernel.

    Strategy: divide the analysis window into intervals delimited by
    consecutive landmark occurrences. A kernel X is a strong block-anchor
    candidate iff it appears EXACTLY ONCE in every interval and its offset
    to the trailing landmark is highly stable across all intervals.
    """
    start, end = op_range
    landmark_positions = [
        i for i in range(start, end + 1)
        if ops[i]["normalized_name"] == landmark_name
    ]
    if len(landmark_positions) < 2:
        return [], landmark_positions

    n_landmarks = len(landmark_positions)
    by_name = {}
    for i in range(start, end + 1):
        nm = ops[i]["normalized_name"]
        by_name.setdefault(nm, []).append(i)

    intervals = []
    prev = start - 1
    for li in landmark_positions:
        intervals.append((prev, li))  # (exclusive_lo, exclusive_hi_landmark)
        prev = li

    candidates = []
    for name, positions in by_name.items():
        if name == landmark_name:
            continue
        aux = is_auxiliary(name)
        if len(positions) % n_landmarks != 0:
            continue
        k = len(positions) // n_landmarks
        if k < 1 or k > 4:
            continue
        ok = True
        pi = 0
        slot_offsets = [[] for _ in range(k)]
        for lo, hi in intervals:
            occ_positions = []
            while pi < len(positions) and positions[pi] < hi:
                if positions[pi] > lo:
                    occ_positions.append(positions[pi])
                pi += 1
            if len(occ_positions) != k:
                ok = False
                break
            occ_positions.sort()
            for s, p in enumerate(occ_positions):
                slot_offsets[s].append(hi - p)
        if not ok:
            continue
        slot_stats = []
        for s, offs in enumerate(slot_offsets):
            mean = sum(offs) / len(offs)
            sd = statistics.pstdev(offs) if len(offs) > 1 else 0.0
            slot_stats.append({
                "slot": s,
                "mean_offset": round(mean, 2),
                "stdev_offset": round(sd, 3),
                "min_offset": min(offs),
                "max_offset": max(offs),
            })
        max_stdev = max(s["stdev_offset"] for s in slot_stats)
        earliest_off = slot_stats[0]["mean_offset"]
        candidates.append({
            "name": name,
            "count": len(positions),
            "auxiliary": aux,
            "repeats_per_interval": k,
            "earliest_offset_from_landmark": earliest_off,
            "max_slot_stdev": max_stdev,
            "slots": slot_stats,
        })

    candidates.sort(
        key=lambda c: (c["auxiliary"], c["max_slot_stdev"],
                       -c["earliest_offset_from_landmark"], c["name"])
    )
    return candidates[:top_k], landmark_positions


def _best_period_row(per_period):
    """Pick the period with the lowest gap_cv (most stable cadence)."""
    scored = [r for r in per_period if r.get("gap_cv") is not None]
    if not scored:
        return None, None, None, None
    r = min(scored, key=lambda x: x["gap_cv"])
    return r["p"], r["gap_cv"], r["size_cv"], r["coverage"]


def print_candidates_table(candidates, total_ops):
    print(f"Total ops: {total_ops}")
    print(f"{'name':<36} {'aux':>3} {'count':>6} {'nxt-div':>7} "
          f"{'bestP':>5} {'gap_cv':>7} {'sizeCv':>7} {'cov':>5} "
          f"{'next-top':<22} "
          f"{'p=1 B/C':>9} {'p=2 B/C':>9} {'p=3 B/C':>9} {'p=4 B/C':>9}")
    for c in candidates:
        per_p = {r["p"]: r for r in c["per_period"]}

        def bc(p):
            r = per_p.get(p)
            return f"{r['num_blocks']}/{r['clusters']}" if r else "-"

        best_p, gap_cv_v, size_cv, cov = _best_period_row(c["per_period"])
        aux = "Y" if c.get("auxiliary") else "N"
        bp = str(best_p) if best_p is not None else "-"
        gcv = f"{gap_cv_v:.3f}" if gap_cv_v is not None else "-"
        scv = f"{size_cv:.3f}" if size_cv is not None else "-"
        cvs = f"{cov:.2f}" if cov is not None else "-"
        top_next = ",".join(f"{n['name']}:{n['count']}" for n in c["next_kernels"][:2])
        if len(top_next) > 22:
            top_next = top_next[:21] + "…"
        print(f"{c['name']:<36} {aux:>3} {c['count']:>6} "
              f"{c['next_kernels_distinct']:>7} {bp:>5} {gcv:>7} {scv:>7} "
              f"{cvs:>5} {top_next:<22} "
              f"{bc(1):>9} {bc(2):>9} {bc(3):>9} {bc(4):>9}")


def print_landmark_table(candidates, landmark_name, n_landmarks):
    print(f"landmark: {landmark_name} (×{n_landmarks})")
    print(f"{'name':<38} {'aux':>3} {'count':>6} {'k':>3} {'early_off':>10} "
          f"{'max_sd':>8} {'slot_offsets':<30}")
    for c in candidates:
        slot_str = ", ".join(f"{s['mean_offset']:.0f}" for s in c["slots"])
        aux = "Y" if c.get("auxiliary") else "N"
        print(f"{c['name']:<38} {aux:>3} {c['count']:>6} "
              f"{c['repeats_per_interval']:>3} "
              f"{c['earliest_offset_from_landmark']:>10.2f} "
              f"{c['max_slot_stdev']:>8.3f} {slot_str:<30}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-r", "--raw_ops", required=True, help="raw_ops.json path")
    p.add_argument("-o", "--output", required=True,
                   help="stream sample mode → structure_draft.json; explore mode → "
                        "candidates + landmark_candidates report")
    p.add_argument("--structure-spec", default=None,
                   help="(sample mode) 0a 解析结果 JSON 路径")
    p.add_argument("--inputs", default=None,
                   help="(sample mode 必传) <run_dir>/inputs.json，Step 0 物料 "
                        "(prof_dir + model_script_paths)；缺则 exit 1")
    p.add_argument("--sample-ack", default=None,
                   help="(sample mode 必传) <run_dir>/sample_ack.json，Phase 0b "
                        "用户 ack 过的每个 component 的 stream samples；"
                        "缺则 exit 1")
    p.add_argument("--accept-warnings", action="store_true",
                   help="(sample mode) 默认 hard/ambiguous warnings 会 exit 1 "
                        "强迫复读给用户；用户明确接受后加此 flag 放行")
    p.add_argument("--explore", action="store_true",
                   help="生成 candidates + landmark_candidates；用户拿不出 sample 时探索用")
    p.add_argument("--min-count", type=int, default=4,
                   help="(explore mode) 候选最小出现次数 (default 4)")
    p.add_argument("--top-k", type=int, default=40,
                   help="(explore mode) 最多产 N 个候选 (default 40)")
    p.add_argument("--print-table", action="store_true",
                   help="(explore mode) 在 stdout 打印 summary 表")
    p.add_argument("--op-start", type=int, default=None,
                   help="(explore mode) 限定分析范围起点")
    p.add_argument("--op-end", type=int, default=None,
                   help="(explore mode) 限定分析范围终点（含）")
    args = p.parse_args()

    sample_mode = bool(args.structure_spec)
    if sample_mode and args.explore:
        raise SystemExit("sample mode (--structure-spec) 不能与 --explore 同时使用")
    if not sample_mode and not args.explore:
        raise SystemExit("必须指定模式：sample (--structure-spec) 或 --explore")

    if sample_mode:
        if not args.structure_spec:
            raise SystemExit("sample mode 需要给 --structure-spec")
        if not args.inputs:
            raise SystemExit(
                "[Phase 0 物料缺失] sample mode 必传 --inputs <run_dir>/inputs.json "
                "(prof_dir + model_script_paths)。"
            )
        if not args.sample_ack:
            raise SystemExit(
                "[Phase 0b 未 ack] sample mode 必传 --sample-ack <run_dir>/sample_ack.json "
                "(用户 ack 过的每个 component stream sample)。"
            )
        ops_raw = load_ops(args.raw_ops)
        operators = ops_raw[0] if isinstance(ops_raw, tuple) else ops_raw
        spec = load_structure_spec(args.structure_spec)
        validate_inputs_json(args.inputs)
        ack = validate_sample_ack(args.sample_ack, spec["expected_components"])
        ack_components = set(ack["components"].keys())
        spec_components = set(spec["expected_components"])
        if spec_components - ack_components:
            raise SystemExit(
                f"[Phase 0b 未 ack] structure_spec 的 expected_components 包含 "
                f"{spec_components - ack_components}，但 sample_ack.json 里没对应条目"
            )

        draft = sample_matching.run_stream_sample_mode(
            operators, spec, ack,
        )
        out = sample_matching.stream_draft_to_dict(draft, spec)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"stream_sample_driven structure_draft → {args.output}", flush=True)
        print(f"  components: {len(out['components'])}", flush=True)
        print(f"  warnings:   {len(out['warnings'])}", flush=True)

        hard = blocking_warnings(out["warnings"])
        ambiguous = ambiguous_warnings(out["warnings"])
        blocked = hard + ambiguous
        if blocked and not args.accept_warnings:
            print(
                f"\n[Phase 1.5 未 ack] {len(blocked)} 条 hard/ambiguous 警告，"
                f"必须复读给用户：",
                file=sys.stderr,
            )
            for w in blocked[:10]:
                print(f"  - [{w.get('code')}] {w.get('message')}", file=sys.stderr)
            if len(blocked) > 10:
                print(f"  ... 还有 {len(blocked) - 10} 条", file=sys.stderr)
            print(
                "用户 ack 完每条后重跑并加 --accept-warnings；不要直接传 flag 跳过。",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    # explore mode
    ops, step_id = load_ops(args.raw_ops)
    range_start, range_end, range_desc = resolve_op_range(args, ops)
    op_range = (range_start, range_end)
    candidates = build_candidates(ops, op_range,
                                  min_count=args.min_count, top_k=args.top_k)
    landmarks = {}
    for lm in ("FlashAttentionScore", "FusedInferAttentionScore",
               "RmsNorm", "LayerNorm", "MoeGatingTopKHash"):
        try:
            lm_cands, _lm_positions = build_landmark_candidates(
                ops, op_range, lm, top_k=args.top_k)
            if lm_cands:
                landmarks[lm] = lm_cands
        except Exception:
            pass
    payload = {
        "total_ops": len(ops),
        "step_id": step_id,
        "op_range": {"start": range_start, "end": range_end,
                     "ops_count": range_end - range_start + 1,
                     "source": range_desc},
        "candidates_count": len(candidates),
        "candidates": candidates,
        "landmark_candidates": landmarks,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"explore output → {args.output}", flush=True)
    print(f"  range: {range_desc} ({range_end - range_start + 1} ops)")
    print(f"  candidates: {len(candidates)}")
    if args.print_table:
        print_candidates_table(candidates, range_end - range_start + 1)
        for lm, cands in landmarks.items():
            print()
            print_landmark_table(cands, lm, sum(1 for op in ops
                if op.get("normalized_name") == lm))


if __name__ == "__main__":
    main()
