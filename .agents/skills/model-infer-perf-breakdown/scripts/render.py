#!/usr/bin/env python3
# coding=utf-8
"""Render per-component sub-item dashboard from a sample-driven draft.

Inputs:
  -d structure_draft.json   Step 2 stream_sample_driven output
                            (structure_draft.stream.v1, op_indices schema).
  -r raw_ops.json           Step 1 single-step op list (operators[]).
  -s network_spec.json      Per-network clustering rules. See schema below.
  --raw-ops-details         raw_ops_details.json from Step 1. If it has been
                            enriched by merge_theoretical_columns.py, theory
                            median columns are computed directly from it.
  --insight-annotations     Optional agent-authored high/medium annotations for
                            the final insight column. Omit on initial render.
  -o index.html             Output (single-page HTML).

Network spec schema:

    {
      "model_name": "<network>",
      "outlier": {"method": "iqr", "k": 1.5},     // or {"method": "z", "k": 2}
      "component_clusters": {
        "csa": [
          {
            "cluster": "input_norm",
            "description": "Pre-norm before Q projection",
            "rules": [{"op_name": "RmsNorm"}]
          },
          {
            "cluster": "q_compress",
            "rules": [
              {"op_name": "MatMulV3", "input_shapes_contains": "<dim>"}
            ]
          },
          {"cluster": "other", "rules": [{"catch_all": true}]}
        ],
        "moe": [...]
      }
    }

Rule fields (all listed conditions must hold; first matching cluster wins):
  op_name                exact match on normalized_name
  op_name_regex          regex on normalized_name
  input_shapes_contains  substring match on input_shapes
  input_shapes_regex     regex on input_shapes
  output_shapes_contains substring match on output_shapes
  output_shapes_regex    regex on output_shapes
  catch_all              true → matches any op (use as last fallback)

Reported sub-items per component_type:
  - one row per cluster: wall_ms = union duration of that cluster's ops in the
    instance; outliers detected on wall_ms across instances.
  - one synthetic row 'bubble': layer-level idle gap, computed as
    span_ms − union_wall_ms (gap between adjacent ops on the layer timeline).
    Outliers detected on the bubble series across instances.
  - TOTAL row: layer end-to-end span = max_op_end − min_op_start, i.e. the
    wall-clock interval a user sees when framing the layer in trace_view.
    Equals union_wall + bubble. Outliers on that span series.

Cross-cluster overlap (overlap_summary): the absolute gap is
Σ cluster_wall − union_wall (time double-counted across buckets); the
reported pct is gap / TOTAL (span) so it lines up with the layer
end-to-end time a user sees in trace_view.

Outlier method: IQR (k=1.5 default) or z-score; configured by spec.outlier.
"""

import argparse
import json
import os
import re
import html
import statistics
import sys
import time
from collections import Counter, defaultdict


UNMATCHED_PCT_HARD_LIMIT = 0.05    # > 5% unmatched ops 视为 Phase 1.5 漏 ack
CLUSTER_COVERAGE_MIN_PCT = 80.0    # Σ cluster_wall.median / TOTAL.median 低于此值时红字提示


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_draft_schema(draft):
    if (draft.get("mode") != "stream_sample_driven"
            or draft.get("schema_version") != "structure_draft.stream.v1"):
        raise SystemExit(
            f"draft mode={draft.get('mode')!r}, "
            f"schema_version={draft.get('schema_version')!r} not supported — "
            f"render.py only accepts stream_sample_driven / "
            f"structure_draft.stream.v1 drafts."
        )
    for i, component in enumerate(draft.get("components", [])):
        if "op_indices" not in component:
            raise SystemExit(f"draft component[{i}] missing op_indices")


def op_indices_from_component(component):
    return [int(i) for i in component.get("op_indices") or []]


def op_indices_from_instance(instance):
    return [int(i) for i in instance.get("op_indices") or []]


def instances_by_type_from_draft(draft):
    by_type = defaultdict(list)
    for c in draft.get("components", []):
        by_type[c["type"]].append({
            "layer_idx": c["layer_idx"],
            "phase": c["phase"],
            "op_indices": op_indices_from_component(c),
            "displaced_op_indices": [int(i) for i in c.get("displaced_op_indices") or []],
        })
    return by_type


def build_match_predicate(rule):
    """Compile a single rule dict into a fn(op) -> bool."""
    if rule.get("catch_all"):
        return lambda _op: True

    op_name = rule.get("op_name")
    op_name_re = re.compile(rule["op_name_regex"]) if rule.get("op_name_regex") else None
    in_contains = rule.get("input_shapes_contains")
    in_re = re.compile(rule["input_shapes_regex"]) if rule.get("input_shapes_regex") else None
    out_contains = rule.get("output_shapes_contains")
    out_re = re.compile(rule["output_shapes_regex"]) if rule.get("output_shapes_regex") else None

    def match(op):
        name = op.get("normalized_name", "")
        if op_name is not None and name != op_name:
            return False
        if op_name_re is not None and not op_name_re.search(name):
            return False
        ish = op.get("input_shapes", "") or ""
        if in_contains is not None and in_contains not in ish:
            return False
        if in_re is not None and not in_re.search(ish):
            return False
        osh = op.get("output_shapes", "") or ""
        if out_contains is not None and out_contains not in osh:
            return False
        if out_re is not None and not out_re.search(osh):
            return False
        return True

    return match


def compile_clusters(cluster_defs):
    """Return [(cluster_name, description, [predicate, ...])] preserving order."""
    out = []
    for entry in cluster_defs:
        name = entry["cluster"]
        desc = entry.get("description", "")
        rules = entry.get("rules", [])
        if not rules:
            raise SystemExit(f"cluster {name!r} has no rules")
        preds = [build_match_predicate(r) for r in rules]
        out.append((name, desc, preds))
    return out


def classify(op, compiled):
    """Return cluster_name (first match) or None."""
    for name, _desc, preds in compiled:
        if any(p(op) for p in preds):
            return name
    return None


def interval_union_us(intervals):
    """intervals: iterable of (start_us, end_us). Return total covered µs."""
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cur_lo = None
    cur_hi = None
    for lo, hi in sorted_iv:
        if hi <= lo:
            continue
        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
            continue
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            if hi > cur_hi:
                cur_hi = hi
    if cur_lo is not None:
        total += cur_hi - cur_lo
    return total


def wall_and_span_us(intervals):
    """Return (wall_us, span_us). wall = union; span = max_end - min_start."""
    ivs = [(lo, hi) for lo, hi in intervals if hi > lo]
    if not ivs:
        return 0.0, 0.0
    span = max(hi for _, hi in ivs) - min(lo for lo, _ in ivs)
    wall = interval_union_us(ivs)
    return wall, span


def pair_overlap_us(intervals_a, intervals_b):
    """Total intersection length between two interval lists (each already raw,
    not yet merged). Returns micro-seconds."""
    if not intervals_a or not intervals_b:
        return 0.0

    def merge(iv):
        iv = sorted(iv)
        m = []
        for a, b in iv:
            if m and a <= m[-1][1]:
                m[-1] = (m[-1][0], max(m[-1][1], b))
            else:
                m.append((a, b))
        return m
    a = merge(intervals_a)
    b = merge(intervals_b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def detect_outliers(values, method="iqr", k=1.5, small_n_ratio=1.5):
    """Return set of indices in `values` that are outliers.

    n >= 4: IQR (default) or z-score, unchanged.
    2 <= n <= 3: statistical detectors are unreliable, so fall back to a
      conservative relative-spike rule — flag any value strictly above
      median * small_n_ratio. Model-agnostic (no absolute unit assumption);
      surfaces obvious single spikes (e.g. one layer 3x slower) that IQR
      cannot see at small n. Set small_n_ratio<=0 to disable the fallback.
    n < 2: never flags.
    """
    n = len(values)
    if n < 2:
        return set()
    if n < 4:
        if not small_n_ratio or small_n_ratio <= 0:
            return set()
        med = statistics.median(values)
        if med <= 0:
            return set()
        return {i for i, v in enumerate(values) if v > med * small_n_ratio}
    if method == "z":
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values)
        if sd <= 0:
            return set()
        return {i for i, v in enumerate(values) if abs(v - mean) / sd > k}
    sorted_v = sorted(values)
    q1 = sorted_v[n // 4]
    q3 = sorted_v[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return {i for i, v in enumerate(values) if v < q1 or v > q3}
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return {i for i, v in enumerate(values) if v < lo or v > hi}


def percentile(values, p):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    rank = (p / 100) * (len(sorted_v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = rank - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def series_metrics(per_instance_us):
    """median / mean / std / p95 / min / max for a series of µs values, in ms."""
    vals_ms = [v / 1000.0 for v in per_instance_us]
    return {
        "count": len(vals_ms),
        "median_ms": statistics.median(vals_ms) if vals_ms else 0.0,
        "mean_ms": statistics.fmean(vals_ms) if vals_ms else 0.0,
        "std_ms": statistics.pstdev(vals_ms) if len(vals_ms) > 1 else 0.0,
        "p95_ms": percentile(vals_ms, 95),
        "min_ms": min(vals_ms) if vals_ms else 0.0,
        "max_ms": max(vals_ms) if vals_ms else 0.0,
    }


def outliers_for(series_us, instances, method, k):
    """Build (outlier_index_set, outlier_list) for a per-instance µs series."""
    idx = detect_outliers(series_us, method=method, k=k)
    outs = [
        {
            "phase": instances[i]["phase"],
            "layer_idx": instances[i]["layer_idx"],
            "value_ms": series_us[i] / 1000.0,
        }
        for i in sorted(idx)
    ]
    return idx, outs


def analyze_component_type(
    component_type,
    instances,
    operators,
    compiled,
    outlier_cfg,
    details_by_idx=None,
    theory_decisions=None,
    theory_review_ratio=0.8,
):
    """For each instance: classify ops, compute per-cluster wall + layer bubble.

    Returns list of sub-items (clusters in spec order, then synthetic 'bubble')
    plus a 'total' dict (layer wall) and an 'unmatched' counter.
    Each sub-item: {name, kind, description, metric, outliers, per_instance_ms,
                    outlier_idx, kernel_outliers}
      outlier_idx flags instance indices whose cluster-wall is an outlier.
      kernel_outliers flags individual kernels: each cluster collects every op,
      slotted by (op_name, occurrence_within_instance). Each slot's durations
      across instances run through detect_outliers; flagged kernels list their
      phase/layer + duration vs slot median.
      kernel_count_anomalies flags instances whose op-name count deviates from
      the modal count across the cluster (extras / missing ops per layer).
    """
    method = outlier_cfg.get("method", "iqr")
    k = outlier_cfg.get("k", 1.5)
    n_inst = len(instances)
    theory_present = _has_theory_columns(details_by_idx)   # 理论性能可选；缺则列留空
    cluster_order = [name for name, _d, _p in compiled]
    per_inst_wall_cluster = {c: [] for c in cluster_order}
    # cluster_slot_durs[cluster][(op_name, occurrence)] = [dur_us per instance,
    #   None if absent in that instance].
    cluster_slot_durs = {c: {} for c in cluster_order}
    # cluster_op_counts[cluster][inst_idx][op_name] = count of that op_name in
    #   that instance's slice of the cluster. Used to flag structural drift.
    cluster_op_counts = {c: [defaultdict(int) for _ in range(n_inst)]
                         for c in cluster_order}
    # cluster_op_records[cluster] = list of {inst_idx, op_idx, phase, layer_idx}
    # in trace order. Lets the renderer pull full CSV rows for that cluster.
    cluster_op_records = {c: [] for c in cluster_order}
    cluster_theory_candidates = {c: [] for c in cluster_order}
    total_theory_candidates = []
    total_wall = []   # interval union of all ops (active region, no bubble)
    total_span = []   # max_end − min_start (end-to-end, with bubble)
    total_bubble = []
    unmatched = defaultdict(int)
    # displaced (time-disjoint) aux streams: excluded from cluster/bubble/TOTAL,
    # summarized separately. stream_id -> [union wall per instance]; + op counts.
    displaced_wall_by_stream = defaultdict(list)
    displaced_count_by_stream = defaultdict(int)
    # cross-cluster overlap accounting (against union_wall, not span)
    per_inst_sum_minus_total = []  # us: Σ cluster_wall − union_wall (≥ 0)
    pair_keys = [(cluster_order[i], cluster_order[j])
                 for i in range(len(cluster_order))
                 for j in range(i + 1, len(cluster_order))]
    per_inst_pair_overlap = {p: [] for p in pair_keys}

    for inst_idx, inst in enumerate(instances):
        # Split off displaced aux ops up front: the main metrics (cluster/bubble/
        # TOTAL) run on the remainder unchanged; displaced ops are summarized below.
        displaced_idx = {int(i) for i in inst.get("displaced_op_indices") or []}
        inst_op_indices = [i for i in op_indices_from_instance(inst)
                           if i not in displaced_idx]
        inst_disp_ivs = defaultdict(list)
        for i in displaced_idx:
            op = operators[i]
            st, dur = op.get("start_time_us"), op.get("duration_us", 0.0)
            if st is None or dur is None:
                continue
            sid = str(op.get("stream_id"))
            inst_disp_ivs[sid].append((st, st + dur))
            displaced_count_by_stream[sid] += 1
        for sid, ivs in inst_disp_ivs.items():
            wall, _ = wall_and_span_us(ivs)
            displaced_wall_by_stream[sid].append(wall)
        by_cluster = defaultdict(list)
        by_cluster_op_indices = defaultdict(list)
        all_intervals = []
        all_op_indices = []
        occ_counter = defaultdict(lambda: defaultdict(int))  # cluster -> op_name -> next occ
        for i in inst_op_indices:
            op = operators[i]
            st = op.get("start_time_us")
            dur = op.get("duration_us", 0.0)
            if st is None or dur is None:
                continue
            iv = (st, st + dur)
            all_intervals.append(iv)
            all_op_indices.append(i)
            cname = classify(op, compiled)
            if cname is None:
                unmatched[op.get("normalized_name", "?")] += 1
                continue
            by_cluster[cname].append(iv)
            by_cluster_op_indices[cname].append(i)
            op_name = op.get("normalized_name", "?")
            cluster_op_counts[cname][inst_idx][op_name] += 1
            occ = occ_counter[cname][op_name]
            occ_counter[cname][op_name] += 1
            slot = (op_name, occ)
            if slot not in cluster_slot_durs[cname]:
                cluster_slot_durs[cname][slot] = [None] * n_inst
            cluster_slot_durs[cname][slot][inst_idx] = dur
            cluster_op_records[cname].append({
                "inst_idx": inst_idx,
                "op_idx": op.get("index", i),
                "phase": inst["phase"],
                "layer_idx": inst["layer_idx"],
                "op_name": op_name,
                "occurrence": occ,
            })
        cluster_walls = {}
        sum_walls = 0.0
        for c in cluster_order:
            wall_c, _ = wall_and_span_us(by_cluster.get(c, []))
            per_inst_wall_cluster[c].append(wall_c)
            cluster_walls[c] = wall_c
            sum_walls += wall_c
        wall_total, span_total = wall_and_span_us(all_intervals)
        total_wall.append(wall_total)
        total_span.append(span_total)
        total_bubble.append(max(0.0, span_total - wall_total))
        per_inst_sum_minus_total.append(max(0.0, sum_walls - wall_total))
        for (a, b) in pair_keys:
            per_inst_pair_overlap[(a, b)].append(
                pair_overlap_us(by_cluster.get(a, []), by_cluster.get(b, []))
            )
        if theory_present:
            for c in cluster_order:
                cluster_theory_candidates[c].append(build_theory_candidate(
                    component_type, c, inst_idx, inst, by_cluster_op_indices.get(c, []),
                    operators, details_by_idx, theory_decisions or {},
                    theory_review_ratio,
                ))
            total_theory_candidates.append(build_theory_candidate(
                component_type, "TOTAL", inst_idx, inst, all_op_indices, operators,
                details_by_idx, theory_decisions or {}, theory_review_ratio,
            ))

    sub_items = []
    for cname, desc, _preds in compiled:
        series = per_inst_wall_cluster[cname]
        idx, outs = outliers_for(series, instances, method, k)
        kernel_outliers = []
        for (op_name, occ), durs in cluster_slot_durs[cname].items():
            valid_idx = [i for i, d in enumerate(durs) if d is not None]
            if len(valid_idx) < 4:
                continue
            valid_vals = [durs[i] for i in valid_idx]
            slot_out_pos = detect_outliers(valid_vals, method=method, k=k)
            if not slot_out_pos:
                continue
            baseline = statistics.median(valid_vals)
            for pos in sorted(slot_out_pos):
                ii = valid_idx[pos]
                kernel_outliers.append({
                    "op_name": op_name,
                    "occurrence": occ,
                    "instance_idx": ii,
                    "phase": instances[ii]["phase"],
                    "layer_idx": instances[ii]["layer_idx"],
                    "duration_us": durs[ii],
                    "baseline_median_us": baseline,
                })
        kernel_outliers.sort(key=lambda x: -abs(x["duration_us"] - x["baseline_median_us"]))
        kernel_count_anomalies = []
        counts_per_inst = cluster_op_counts[cname]
        all_names = set()
        for d in counts_per_inst:
            all_names.update(d.keys())
        for op_name in sorted(all_names):
            vec = [d.get(op_name, 0) for d in counts_per_inst]
            ctr = Counter(vec)
            modal_count, modal_freq = ctr.most_common(1)[0]
            if modal_freq == n_inst:
                continue
            for ii, cnt in enumerate(vec):
                if cnt == modal_count:
                    continue
                kernel_count_anomalies.append({
                    "op_name": op_name,
                    "instance_idx": ii,
                    "phase": instances[ii]["phase"],
                    "layer_idx": instances[ii]["layer_idx"],
                    "count": cnt,
                    "modal_count": modal_count,
                })
        kernel_count_anomalies.sort(
            key=lambda x: (-abs(x["count"] - x["modal_count"]),
                           x["op_name"], x["instance_idx"])
        )
        sub_items.append({
            "name": cname,
            "kind": "cluster",
            "description": desc,
            "metric": series_metrics(series),
            "outliers": outs,
            "outlier_idx": sorted(idx),
            "per_instance_ms": [v / 1000.0 for v in series],
            "kernel_outliers": kernel_outliers,
            "kernel_count_anomalies": kernel_count_anomalies,
            "op_records": cluster_op_records[cname],
            "theoretical": summarize_theory(cluster_theory_candidates[cname]),
        })
    idx, outs = outliers_for(total_bubble, instances, method, k)
    sub_items.append({
        "name": "bubble",
        "kind": "bubble",
        "description": "layer idle gap (span − wall)",
        "metric": series_metrics(total_bubble),
        "outliers": outs,
        "outlier_idx": sorted(idx),
        "per_instance_ms": [v / 1000.0 for v in total_bubble],
    })

    idx, outs = outliers_for(total_span, instances, method, k)
    total = {
        "description": "layer end-to-end span (max_end − min_start, 含 bubble)",
        "metric": series_metrics(total_span),
        "outliers": outs,
        "outlier_idx": sorted(idx),
        "per_instance_ms": [v / 1000.0 for v in total_span],
        "theoretical": summarize_theory(total_theory_candidates),
    }

    # Cross-cluster overlap summary. median (Σ cluster_wall − TOTAL wall) as
    # absolute ms and as % of median TOTAL wall, plus top pairs by median
    # pairwise overlap. Lets the renderer / metrics consumer flag "buckets
    # overlap so cluster walls double-count the timeline."
    med_gap = statistics.median(per_inst_sum_minus_total) if per_inst_sum_minus_total else 0.0
    max_gap = max(per_inst_sum_minus_total) if per_inst_sum_minus_total else 0.0
    med_total = statistics.median(total_span) if total_span else 0.0
    pair_summaries = []
    for (a, b), vals in per_inst_pair_overlap.items():
        if not vals:
            continue
        med = statistics.median(vals)
        if med <= 0:
            continue
        pair_summaries.append({
            "cluster_a": a,
            "cluster_b": b,
            "median_overlap_ms": med / 1000.0,
            "median_overlap_pct": (med / med_total * 100) if med_total else 0.0,
            "max_overlap_ms": max(vals) / 1000.0,
        })
    pair_summaries.sort(key=lambda p: -p["median_overlap_ms"])
    overlap_summary = {
        "median_gap_ms": med_gap / 1000.0,
        "median_gap_pct": (med_gap / med_total * 100) if med_total else 0.0,
        "max_gap_ms": max_gap / 1000.0,
        "top_pairs": pair_summaries[:8],
    }
    med_total_span = statistics.median(total_span) if total_span else 0.0
    displaced_summary = [
        {
            "stream_id": sid,
            "op_count": displaced_count_by_stream[sid],
            "median_wall_ms": (statistics.median(walls) if walls else 0.0) / 1000.0,
            "pct_of_total": ((statistics.median(walls) / med_total_span * 100)
                             if walls and med_total_span else 0.0),
        }
        for sid, walls in sorted(displaced_wall_by_stream.items())
    ]
    return sub_items, total, dict(unmatched), overlap_summary, displaced_summary


def fmt_duration_us(value_us):
    if value_us is None:
        return "—"
    v = float(value_us)
    return f"{v:.1f} µs" if abs(v) < 1000.0 else f"{v / 1000.0:.3f} ms"


def fmt_duration_ms(value_ms):
    if value_ms is None:
        return "—"
    return fmt_duration_us(float(value_ms) * 1000.0)


def fmt_outliers(outs):
    return ", ".join(
        f"{o['phase']}#{o['layer_idx']} ({fmt_duration_ms(o['value_ms'])})"
        for o in outs
    ) or "—"


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\t", "")
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _has_theory_columns(details_by_idx):
    """理论列是否存在（Step 1.5 是否跑过）。理论性能是可选的——外部
    operator-theoretical-perf skill 可能不存在；没跑则 render 把理论列留空，不阻塞。"""
    if not details_by_idx:
        return False
    return any("theoretical_operator_time_us" in d for d in details_by_idx.values())


def _detail_theory_us(detail):
    if not detail:
        return None
    supported = detail.get("theory_supported")
    if isinstance(supported, str) and supported.strip().lower() in {
        "false", "0", "no", "n/a", "unsupported",
    }:
        return None
    if supported is False:
        return None
    return _to_float(detail.get("theoretical_operator_time_us"))


def load_theory_decisions(path):
    if not path:
        return {}
    data = load_json(path)
    decisions = data if isinstance(data, list) else data.get("decisions", [])
    out = {}
    for item in decisions:
        # Omit BOTH phase and layer_idx (→ None) for a slot-wide wildcard that
        # covers every instance of (component_type, sub_item); lookup tries the
        # exact (phase,layer) key first, then the (None,None) wildcard. Only
        # component_type + sub_item are mandatory.
        key = (
            item.get("component_type"),
            item.get("sub_item"),
            item.get("phase"),
            item.get("layer_idx"),
        )
        if key[0] is None or key[1] is None:
            continue
        decision = {
            "reason": item.get("reason", "agent annotation"),
            "semantic_note": item.get("semantic_note"),
            "stream_semantics": item.get("stream_semantics") or {},
        }
        if item.get("selected_stream") is not None:
            decision["selected_stream"] = str(item.get("selected_stream"))
        out[key] = decision
    return out


def build_theory_candidate(
    component_type,
    sub_item,
    inst_idx,
    inst,
    op_indices,
    operators,
    details_by_idx,
    decisions,
    review_ratio,
):
    if not details_by_idx:
        return None
    streams = defaultdict(lambda: {
        "op_count": 0,
        "supported_count": 0,
        "unsupported_count": 0,
        "theoretical_sum_us": 0.0,
        "intervals": [],
    })
    missing_theory = 0
    for i in op_indices:
        op = operators[i]
        stream = str(op.get("stream_id", "unknown"))
        bucket = streams[stream]
        bucket["op_count"] += 1
        st = op.get("start_time_us")
        dur = op.get("duration_us")
        if st is not None and dur is not None:
            bucket["intervals"].append((float(st), float(st) + float(dur)))
        detail = details_by_idx.get(op.get("index", i))
        theory_us = _detail_theory_us(detail)
        if theory_us is not None:
            bucket["supported_count"] += 1
            bucket["theoretical_sum_us"] += theory_us
        else:
            bucket["unsupported_count"] += 1
            missing_theory += 1

    stream_rows = []
    for stream, row in streams.items():
        op_count = row["op_count"]
        supported = row["supported_count"]
        stream_rows.append({
            "stream_id": stream,
            "op_count": op_count,
            "supported_count": supported,
            "unsupported_count": row["unsupported_count"],
            "supported_pct": supported / op_count * 100.0 if op_count else 0.0,
            "observed_union_us": interval_union_us(row["intervals"]),
            "theoretical_sum_us": row["theoretical_sum_us"] if supported else None,
        })
    stream_rows.sort(key=lambda x: x["observed_union_us"], reverse=True)

    # exact (ct, sub, phase, layer) → full wildcard (ct, sub, None, None).
    # The wildcard lets one decision cover all instances of a slot.
    decision = (
        decisions.get((component_type, sub_item, inst.get("phase"), inst.get("layer_idx")))
        or decisions.get((component_type, sub_item, None, None))
    )
    selected = stream_rows[0] if stream_rows else None
    selection_source = "script_max_observed_union_us"
    selection_warning = None
    if decision and decision.get("selected_stream") is not None:
        requested = decision["selected_stream"]
        match = next((r for r in stream_rows if r["stream_id"] == requested), None)
        if match:
            selected = match
            selection_source = "agent_decision"
        else:
            selection_warning = f"agent selected missing stream {requested!r}"

    needs_agent_review = False
    if len(stream_rows) >= 2 and stream_rows[0]["observed_union_us"] > 0:
        ratio = stream_rows[1]["observed_union_us"] / stream_rows[0]["observed_union_us"]
        needs_agent_review = ratio >= review_ratio and selection_source != "agent_decision"

    theory_us = selected.get("theoretical_sum_us") if selected else None
    return {
        "instance_idx": inst_idx,
        "phase": inst.get("phase"),
        "layer_idx": inst.get("layer_idx"),
        "selected_stream": selected.get("stream_id") if selected else None,
        "selection_source": selection_source,
        "selection_reason": (
            decision.get("reason") if decision else "largest observed timeline union"
        ),
        "selection_warning": selection_warning,
        "needs_agent_review": needs_agent_review,
        "multi_stream": len(stream_rows) > 1,
        "semantic_note": decision.get("semantic_note") if decision else None,
        "stream_semantics": decision.get("stream_semantics") if decision else {},
        "effective_theoretical_us": theory_us,
        "missing_theory_count": missing_theory,
        "stream_candidates": stream_rows,
    }


def summarize_theory(candidates):
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    theory_values = [
        c["effective_theoretical_us"]
        for c in candidates
        if c.get("effective_theoretical_us") is not None
    ]
    return {
        "median_theoretical_ms": (
            statistics.median(theory_values) / 1000.0 if theory_values else None
        ),
        "theoretical_metric": series_metrics(theory_values) if theory_values else None,
        "supported_instance_count": len(theory_values),
        "instance_count": len(candidates),
        "needs_agent_review_count": sum(
            1 for c in candidates if c.get("needs_agent_review")
        ),
        "multi_stream_count": sum(1 for c in candidates if c.get("multi_stream")),
        "missing_semantic_note_count": sum(
            1 for c in candidates
            if c.get("multi_stream") and not c.get("semantic_note")
        ),
        "semantic_notes": sorted({
            c.get("semantic_note")
            for c in candidates
            if c.get("multi_stream") and c.get("semantic_note")
        }),
        "missing_theory_count": sum(c.get("missing_theory_count", 0) for c in candidates),
        "per_instance": candidates,
    }


def render_theory_cell(summary, actual_median_ms):
    if not summary:
        return "—"
    med = summary.get("median_theoretical_ms")
    if med is None:
        return "<span class='muted'>N/A</span>"
    ratio = None
    if actual_median_ms and med > 0:
        ratio = actual_median_ms / med
    elif summary.get("actual_over_theoretical_median") is not None:
        ratio = summary["actual_over_theoretical_median"]

    notes = []
    inst_n = summary.get("instance_count")
    supported_n = summary.get("supported_instance_count")
    if inst_n is not None and supported_n is not None and supported_n < inst_n:
        notes.append(f"{supported_n}/{inst_n} inst")
    review_n = summary.get("needs_agent_review_count") or 0
    if review_n:
        notes.append(f"{review_n} review")
    multi_n = summary.get("multi_stream_count") or 0
    if multi_n:
        notes.append(f"multi-stream {multi_n} inst")
        semantic_notes = summary.get("semantic_notes") or []
        if semantic_notes:
            notes.extend(semantic_notes[:2])
            if len(semantic_notes) > 2:
                notes.append(f"+{len(semantic_notes) - 2} notes")
        missing_sem_n = summary.get("missing_semantic_note_count") or 0
        if missing_sem_n:
            notes.append(f"{missing_sem_n} semantic note required")
    miss_n = summary.get("missing_theory_count") or 0
    if miss_n:
        notes.append(f"{miss_n} unsupported")

    ratio_html = f"<span class='theory-ratio'>wall/theory {ratio:.2f}x</span>" if ratio else ""
    note_html = f"<span class='theory-note'>{html.escape('; '.join(notes))}</span>" if notes else ""
    return (
        f"<span class='theory-main'>{fmt_duration_ms(med)}</span>"
        f"{ratio_html}{note_html}"
    )


def _insight_confidence(review_or_record):
    if not review_or_record:
        return None
    conf = review_or_record.get("confidence")
    if conf is None and isinstance(review_or_record.get("agent_review"), dict):
        conf = review_or_record["agent_review"].get("confidence")
    if conf is None:
        return None
    text = str(conf).strip().lower()
    return text if text in {"high", "medium"} else None


def _insight_summary(record):
    for key in ("summary", "semantic_summary", "reason"):
        val = record.get(key)
        if val:
            return str(val)
    return ""


def _row_insight_targets(record):
    raw_targets = []
    if isinstance(record.get("target"), dict):
        raw_targets.append(record["target"])
    elif isinstance(record.get("targets"), list):
        raw_targets.extend(t for t in record["targets"] if isinstance(t, dict))
    else:
        raw_targets.append(record)

    targets = []
    for target in raw_targets:
        component_type = (
            target.get("component_type")
            or target.get("primary_component_type")
            or record.get("component_type")
            or record.get("primary_component_type")
        )
        sub_item = (
            target.get("sub_item")
            or target.get("cluster")
            or target.get("primary_sub_item")
            or record.get("sub_item")
            or record.get("cluster")
            or record.get("primary_sub_item")
        )
        if not component_type or not sub_item:
            continue
        targets.append({
            "component_type": component_type,
            "sub_item": sub_item,
            "mapping_type": (
                target.get("mapping_type")
                or record.get("mapping_type")
                or "direct"
            ),
            "mapping_note": target.get("mapping_note") or record.get("mapping_note") or "",
            "related_targets": (
                target.get("related_targets")
                or record.get("related_targets")
                or []
            ),
        })
    return targets


def _format_related_targets(targets):
    if not isinstance(targets, list):
        return ""
    labels = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        component_type = target.get("component_type")
        sub_item = target.get("sub_item") or target.get("cluster")
        if component_type and sub_item:
            labels.append(f"{component_type}/{sub_item}")
    return ", ".join(labels)


def _add_row_insight(lookup, record):
    conf = _insight_confidence(record)
    if conf not in {"high", "medium"}:
        return
    targets = _row_insight_targets(record)
    if not targets:
        return
    summary = _insight_summary(record)
    if not summary:
        return
    evidence = record.get("evidence")
    if isinstance(evidence, (dict, list)):
        evidence = json.dumps(evidence, ensure_ascii=False)
    source = record.get("source", "agent")
    category = (
        record.get("category")
        or record.get("insight_category")
        or (os.path.splitext(os.path.basename(str(source)))[0] if source else "")
    )
    for target in targets:
        lookup[(target["component_type"], target["sub_item"])].append({
            "confidence": conf,
            "summary": summary,
            "category": str(category) if category else "",
            "source": source,
            "evidence": str(evidence) if evidence else "",
            "mapping_type": target.get("mapping_type") or "direct",
            "mapping_note": target.get("mapping_note") or "",
            "related_targets": _format_related_targets(target.get("related_targets")),
        })


def load_agent_insight_annotations(path):
    """Load agent-authored main report insight annotations.

    The renderer does not infer insight semantics from Step 5 JSONs. The agent
    explicitly decides which rows deserve a high/medium note and writes:

      {"items": [{"target": {"component_type": "moe", "sub_item": "expert",
                             "mapping_type": "direct"},
                  "category": "operator_jitter", "confidence": "high",
                  "summary": "...", "evidence": "..."}]}

    Initial Step 3 render passes no path, so the final column remains blank.
    """
    lookup = defaultdict(list)
    if not path:
        return lookup
    data = load_json(path)
    records = data if isinstance(data, list) else data.get("items", [])
    for rec in records:
        _add_row_insight(lookup, rec)

    order = {"high": 0, "medium": 1}
    for key, vals in lookup.items():
        vals.sort(key=lambda x: (
            order.get(x["confidence"], 9),
            x.get("category") or "",
            x["summary"],
        ))
        lookup[key] = vals
    return lookup


def render_insight_cell(items):
    if not items:
        return ""
    parts = []
    for item in items:
        conf = html.escape(item["confidence"])
        summary = html.escape(item["summary"])
        category = html.escape(item.get("category") or "")
        source = html.escape(item.get("source") or "")
        evidence = html.escape(item.get("evidence") or "")
        mapping_type = item.get("mapping_type") or "direct"
        mapping_note = item.get("mapping_note") or ""
        related_targets = item.get("related_targets") or ""
        category_html = (
            f"<span class='insight-category'>{category}</span> " if category else ""
        )
        evidence_html = f"<span class='insight-evidence'>{evidence}</span>" if evidence else ""
        mapping_bits = []
        if mapping_type != "direct":
            mapping_bits.append(f"mapping: {mapping_type}")
        if mapping_note:
            mapping_bits.append(mapping_note)
        if related_targets:
            mapping_bits.append(f"related: {related_targets}")
        mapping_html = (
            f"<span class='insight-mapping'>{html.escape('; '.join(mapping_bits))}</span>"
            if mapping_bits else ""
        )
        parts.append(
            f"<div class='insight-note {conf}'>"
            f"<span class='insight-conf'>{conf}</span> {category_html}{summary}"
            f"<span class='insight-source'>{source}</span>{mapping_html}{evidence_html}"
            f"</div>"
        )
    return "".join(parts)


def render_inst_op_table(op_records, details_by_idx, dur_anom_map, count_anom_map):
    """Per-instance op table with a leading status column.

    op_records: list of {op_idx, op_name, occurrence, ...} for one instance.
    dur_anom_map: {(op_name, occurrence): {duration_us, baseline_us}} flagged
        as duration outliers within this cluster, scoped to this instance.
    count_anom_map: {op_name: {count, modal_count}} for this instance — covers
        both 多出 (count > modal) and 缺失/少出 (count < modal). Missing ops
        are not in op_records, so we emit synthetic rows at the end.
    """
    if not op_records or not details_by_idx:
        return ""
    rows = []
    columns = None
    for rec in op_records:
        op = details_by_idx.get(rec["op_idx"])
        if op is None:
            continue
        if columns is None:
            columns = list(op.keys())
        rows.append((rec, op))
    if not rows or not columns:
        return ""

    header_cells = "<th class='status-col'>状态</th>" + "".join(
        f"<th>{html.escape(c)}</th>" for c in columns
    )

    body_rows = []
    for rec, op in rows:
        tags = []
        row_cls = ""
        key = (rec["op_name"], rec["occurrence"])
        if key in dur_anom_map:
            da = dur_anom_map[key]
            base = da["baseline_us"]
            dur = da["duration_us"]
            is_slow = dur > base
            verb = "偏慢" if is_slow else "偏快"
            pct = ((dur - base) / base * 100.0) if base > 0 else 0.0
            tag_cls = "status slow" if is_slow else "status fast"
            tags.append(
                f"<span class='{tag_cls}'>{verb} ({pct:+.0f}% vs 中位 "
                f"{fmt_duration_us(base)})</span>"
            )
            row_cls = "row-slow" if is_slow else "row-fast"
        ca = count_anom_map.get(rec["op_name"])
        if ca and ca["count"] > ca["modal_count"] and \
                rec["occurrence"] >= ca["modal_count"]:
            tags.append(
                f"<span class='status extra'>多出 "
                f"({ca['count']} vs 多数层 {ca['modal_count']})</span>"
            )
            if not row_cls:
                row_cls = "row-extra"
        status_cell = " ".join(tags) if tags else "—"
        cells = [f"<td class='status-cell'>{status_cell}</td>"]
        for c in columns:
            v = op.get(c, "")
            if v is None:
                v = ""
            elif isinstance(v, float):
                v = f"{v:.6g}"
            else:
                v = str(v)
            cells.append(f"<td>{html.escape(v)}</td>")
        row_attr = f" class='{row_cls}'" if row_cls else ""
        body_rows.append(f"<tr{row_attr}>" + "".join(cells) + "</tr>")

    for op_name, ca in count_anom_map.items():
        if ca["count"] >= ca["modal_count"]:
            continue
        missing_n = ca["modal_count"] - ca["count"]
        verb = "缺失" if ca["count"] == 0 else "少出"
        tag = (
            f"<span class='status missing'>{verb} {missing_n} 个 "
            f"(本层 {ca['count']} / 多数层 {ca['modal_count']})</span>"
        )
        cells = [f"<td class='status-cell'>{tag}</td>"]
        for c in columns:
            if c == "name":
                cells.append(f"<td>{html.escape(op_name)}</td>")
            else:
                cells.append("<td class='missing-cell'>—</td>")
        body_rows.append("<tr class='row-missing'>" + "".join(cells) + "</tr>")

    return (
        "<div class='subwrap'><table class='subtbl'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def render_html(model_name, sections, outlier_cfg, details_by_idx=None, row_insights=None):
    row_insights = row_insights or {}
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           color: #222; max-width: 1400px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; font-size: 13px; margin-bottom: 24px; }
    .muted { color: #888; }
    h2 { margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
    h2 small { color: #888; font-weight: normal; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px;
            table-layout: fixed; }
    col.sub { width: 130px; }
    col.desc { width: 230px; }
    col.med { width: 80px; }
    col.theory { width: 110px; }
    col.insight { width: 190px; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }
    th { background: #f5f5f5; text-align: left; }
    td.med { text-align: right; }
    td.theory { text-align: right; font-variant-numeric: tabular-nums; }
    .theory-main { display: block; font-weight: 600; }
    .theory-ratio, .theory-note { display: block; color: #666; font-size: 11px; }
    .theory-note { color: #8a5810; }
    td.insight { font-size: 12px; }
    .insight-note { margin: 0 0 6px; line-height: 1.35; }
    .insight-conf { display: inline-block; min-width: 48px; margin-right: 4px;
                    border-radius: 999px; padding: 1px 6px; font-size: 10px;
                    font-weight: 700; text-transform: uppercase; }
    .insight-note.high .insight-conf { background: #fee2e2; color: #991b1b; }
    .insight-note.medium .insight-conf { background: #fef3c7; color: #92400e; }
    .insight-category { display: inline-block; margin-right: 4px;
                        border-radius: 4px; padding: 1px 5px;
                        background: #e8eef8; color: #23436d;
                        font-size: 10px; font-weight: 700; }
    .insight-source, .insight-evidence, .insight-mapping {
        display: block; color: #777; font-size: 10.5px; margin-top: 2px; }
    .insight-mapping { color: #8a5810; }
    td.sub { font-weight: 600; }
    td.desc { color: #666; font-size: 12px; }
    tr.bubble td.sub { font-style: italic; }
    tr.total { background: #f9f9f9; font-weight: bold; }
    .chips { display: flex; flex-wrap: wrap; gap: 4px; align-items: stretch; }
    .chip { position: relative; display: inline-block; padding: 2px 6px;
            border-radius: 3px; background: #eef2f7; font-size: 11px;
            font-family: ui-monospace, "SF Mono", Consolas, monospace;
            white-space: nowrap; overflow: hidden; min-width: 70px; }
    .chip > .bar { position: absolute; left: 0; top: 0; bottom: 0;
                   background: #5a8fb8; opacity: 0.22; z-index: 0; }
    .chip > .tag, .chip > .lbl, .chip > .val { position: relative; z-index: 1; }
    .chip .lbl { color: #667; margin-right: 4px; }
    .chip.outlier-slow { font-weight: 600; }
    .chip.outlier-slow > .bar { background: #a32020; opacity: 0.28; }
    .chip.outlier-slow.tier-1 { background: #fdecec; border: 1px solid #f5b8b8;
                                color: #a32020; }
    .chip.outlier-slow.tier-1 .lbl { color: #a32020; }
    .chip.outlier-slow.tier-2 { background: #fbcaca; border: 1px solid #e07070;
                                color: #8a1818; }
    .chip.outlier-slow.tier-2 .lbl { color: #8a1818; }
    .chip.outlier-slow.tier-3 { background: #f59090; border: 1px solid #c84040;
                                color: #6f0e0e; }
    .chip.outlier-slow.tier-3 .lbl { color: #6f0e0e; }
    .chip.outlier-fast { font-weight: 600; }
    .chip.outlier-fast > .bar { background: #2d6a30; opacity: 0.28; }
    .chip.outlier-fast.tier-1 { background: #eaf3ea; border: 1px solid #b5d7b5;
                                color: #2d6a30; }
    .chip.outlier-fast.tier-1 .lbl { color: #2d6a30; }
    .chip.outlier-fast.tier-2 { background: #c8e3c8; border: 1px solid #88c08a;
                                color: #1f5022; }
    .chip.outlier-fast.tier-2 .lbl { color: #1f5022; }
    .chip.outlier-fast.tier-3 { background: #95cf99; border: 1px solid #5fa666;
                                color: #0f3a12; }
    .chip.outlier-fast.tier-3 .lbl { color: #0f3a12; }
    .chip .tag { font-weight: 600; margin-right: 4px;
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .phasebrk { border-left: 1px dashed #bbb; margin: 0 2px; align-self: stretch; }
    .legend { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
              padding: 8px 12px; margin: 8px 0 18px; font-size: 12px;
              line-height: 1.7; }
    .legend .key { display: inline-block; padding: 1px 6px; border-radius: 3px;
                   border: 1px solid #ccc; margin: 0 4px; font-family: ui-monospace,
                   "SF Mono", Consolas, monospace; font-size: 11px; }
    .legend .key.slow { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    .legend .key.fast { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    .unmatched { background: #fff8e7; padding: 8px 12px; border-left: 3px solid #e0a020;
                 margin: 8px 0; font-size: 13px; }
    .unmatched code { background: #f0e8d8; padding: 1px 4px; border-radius: 3px; }
    details.chip-d { display: block; padding: 0; margin: 0; min-width: 0; }
    details.chip-d[open] { flex: 1 1 100%; }
    details.chip-d > summary { list-style: none; cursor: pointer; outline: none;
                               user-select: none; }
    details.chip-d > summary::-webkit-details-marker { display: none; }
    details.chip-d > summary::marker { content: ""; }
    details.chip-d > .panel { display: none; }
    details.chip-d[open] > .panel { display: block; margin: 6px 0 4px; }
    details.chip-d[open] > summary.chip { box-shadow: 0 0 0 2px #5a8fb8; }
    details.chip-d[open] > summary.chip.outlier-slow { box-shadow: 0 0 0 2px #a32020; }
    details.chip-d[open] > summary.chip.outlier-fast { box-shadow: 0 0 0 2px #2d6a30; }
    .chip .med-note { color: #555; font-size: 10.5px; margin-left: 6px;
                      font-family: ui-monospace, "SF Mono", Consolas, monospace; }
    .chip.outlier-slow .med-note { color: #7a3030; }
    .chip.outlier-fast .med-note { color: #2d6a30; }
    .chip .anom-note { position: relative; z-index: 1; font-size: 10.5px;
                       margin-left: 6px; font-family: ui-monospace, "SF Mono",
                       Consolas, monospace; }
    .chip .anom-note .hot { color: #a32020; font-weight: 600; margin-right: 3px; }
    .chip .anom-note .cool { color: #2d6a30; font-weight: 600; margin-right: 3px; }
    .subwrap { overflow-x: auto; max-height: 320px; overflow-y: auto;
               border: 1px solid #e0e0e0; border-radius: 3px; margin-top: 4px; }
    table.subtbl { width: max-content; min-width: 100%; border-collapse: collapse;
                   font-size: 10.5px; font-family: ui-monospace, "SF Mono", Consolas, monospace;
                   table-layout: auto; margin: 0; }
    table.subtbl th, table.subtbl td { padding: 2px 6px; border: 1px solid #e8e8e8;
                                       white-space: nowrap; vertical-align: top; }
    table.subtbl th { background: #f0f4f8; position: sticky; top: 0;
                      font-weight: 600; font-family: -apple-system, BlinkMacSystemFont,
                      "Segoe UI", sans-serif; font-size: 10.5px; }
    table.subtbl td.where { font-weight: 600; background: #fff; }
    table.subtbl tr:nth-child(even) td { background: #f8f8f8; }
    table.subtbl tr:nth-child(even) td.where { background: #f4f4f4; }
    table.subtbl th.status-col, table.subtbl td.status-cell {
        min-width: 140px; max-width: 240px; white-space: normal;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    table.subtbl .status { display: inline-block; padding: 1px 5px; margin: 0 3px 2px 0;
                           border-radius: 3px; border: 1px solid #ccc; font-weight: 600;
                           font-size: 10px; white-space: nowrap; }
    table.subtbl .status.slow { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    table.subtbl .status.fast { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    table.subtbl .status.extra { background: #fde2e2; border-color: #f0a0a0; color: #a32020; }
    table.subtbl .status.missing { background: #e2f0e2; border-color: #a0d0a0; color: #2d6a30; }
    table.subtbl tr.row-slow td, table.subtbl tr.row-slow:nth-child(even) td {
        background: #fceaea; }
    table.subtbl tr.row-fast td, table.subtbl tr.row-fast:nth-child(even) td {
        background: #e8f3e8; }
    table.subtbl tr.row-extra td, table.subtbl tr.row-extra:nth-child(even) td {
        background: #fdeede; }
    table.subtbl tr.row-missing td, table.subtbl tr.row-missing:nth-child(even) td {
        background: #eef5ee; font-style: italic; }
    table.subtbl td.missing-cell { color: #aaa; }
    .coverage-warn { background: #fde8e8; border-left: 3px solid #c83030;
                     padding: 8px 12px; margin: 8px 0; font-size: 13px;
                     color: #6e0e0e; }
    .coverage-warn .head { font-weight: 600; margin-bottom: 4px; }
    .overlap-warn { background: #fff4e0; border-left: 3px solid #d48820;
                    padding: 8px 12px; margin: 8px 0; font-size: 13px; }
    .overlap-warn .head { font-weight: 600; color: #8a5810; margin-bottom: 4px; }
    .overlap-warn .pairs { margin-top: 4px; font-family: ui-monospace,
                           "SF Mono", Consolas, monospace; font-size: 12px;
                           color: #6a4408; }
    .overlap-warn .pairs .pair { display: inline-block; margin: 2px 6px 2px 0;
                                 background: #f8e8c8; padding: 1px 6px;
                                 border-radius: 3px; }
    .overlap-info { background: #eef4fb; border-left: 3px solid #3b73a8;
                    padding: 8px 12px; margin: 8px 0; font-size: 13px;
                    color: #2a4a66; }
    .overlap-info .head { font-weight: 600; color: #2a4a66; margin-bottom: 4px; }
    .overlap-info .pairs { margin-top: 4px; font-family: ui-monospace,
                           "SF Mono", Consolas, monospace; font-size: 12px;
                           color: #2a4a66; }
    .overlap-info .pairs .pair { display: inline-block; margin: 2px 6px 2px 0;
                                 background: #dde8f4; padding: 1px 6px;
                                 border-radius: 3px; }
    """

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(model_name)} cluster breakdown</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>{html.escape(model_name)}</h1>",
        f"<div class='meta'>outlier rule: <code>{html.escape(outlier_cfg.get('method', 'iqr'))}</code> "
        f"k={outlier_cfg.get('k', 1.5)} &middot; "
        f"cluster wall = 该 cluster 算子的时间并集 &middot; "
        f"bubble = 层内算子间空隙（span − 总 wall 并集） &middot; "
        f"TOTAL = 层 end-to-end span（max_end − min_start, 含 bubble）"
        f"；HTML 自动按量级显示 µs / ms</div>",
        "<div class='legend'>"
        "<b>读图说明</b><br>"
        "每行 cluster 是一个 chip 列表，每个 chip 表示一个实例的 wall time。"
        "<span class='key slow'>红 chip = 比中位慢</span>、"
        "<span class='key fast'>绿 chip = 比中位快</span>（均为 IQR 异常），"
        "其旁直接标 <code>vs 中位 X µs/ms (±N%)</code>。<br>"
        "chip 标签后用紧凑标注汇总该实例算子级异常计数："
        "<code><span style='color:#a32020;font-weight:600'>N慢</span></code>/"
        "<code><span style='color:#2d6a30;font-weight:600'>N快</span></code>"
        "（同名同序号算子耗时偏离多数层）、"
        "<code><span style='color:#a32020;font-weight:600'>N多</span></code>/"
        "<code><span style='color:#2d6a30;font-weight:600'>N缺</span></code>"
        "（算子出现次数偏离多数层）。<br>"
        "若 <code>raw_ops_details.json</code> 已由 <code>operator_analysis.csv</code> 合并理论列，"
        "<code>theory median</code> 表示该 sub-item 在 critical stream 上的理论耗时中位数；"
        "多流时默认取 observed union 最大流，标 <code>review</code> 的项需 agent 判定流选择。<br>"
        "<b>点击 chip</b> 展开该实例的算子明细子表（bubble/TOTAL 不可展）。"
        "子表第一列「状态」与行底色给出每条异常的具体算子与数值。"
        "</div>",
    ]

    def chip_cell(per_inst, meta, outlier_idx, kernel_outliers=None,
                  kernel_count_anomalies=None, op_records=None,
                  median_ms=None):
        outlier_set = set(outlier_idx or [])
        max_v = max(per_inst) if per_inst else 0.0
        expandable = bool(op_records) and bool(details_by_idx)
        records_by_inst = defaultdict(list)
        dur_anom_by_inst = defaultdict(dict)
        count_anom_by_inst = defaultdict(dict)
        if expandable:
            for rec in op_records:
                records_by_inst[rec["inst_idx"]].append(rec)
        for ko in kernel_outliers or []:
            dur_anom_by_inst[ko["instance_idx"]][
                (ko["op_name"], ko["occurrence"])
            ] = {
                "duration_us": ko["duration_us"],
                "baseline_us": ko["baseline_median_us"],
            }
        for ka in kernel_count_anomalies or []:
            count_anom_by_inst[ka["instance_idx"]][ka["op_name"]] = {
                "count": ka["count"],
                "modal_count": ka["modal_count"],
            }
        ref_median = median_ms
        if ref_median is None or ref_median <= 0:
            nz = sorted(v for v in per_inst if v > 0)
            if nz:
                mid = len(nz) // 2
                ref_median = nz[mid] if len(nz) % 2 else (nz[mid - 1] + nz[mid]) / 2
        chips = []
        prev_phase = None
        for i, v in enumerate(per_inst):
            m = meta[i]
            if prev_phase is not None and m["phase"] != prev_phase:
                chips.append("<span class='phasebrk'></span>")
            prev_phase = m["phase"]
            is_out = i in outlier_set
            if is_out:
                if ref_median is not None and ref_median > 0:
                    dev = abs(v - ref_median) / ref_median
                else:
                    dev = 0.0
                if dev >= 0.5:
                    tier = "tier-3"
                elif dev >= 0.2:
                    tier = "tier-2"
                else:
                    tier = "tier-1"
                if ref_median is not None and ref_median > 0 and v < ref_median:
                    cls = f"chip outlier-fast {tier}"
                    tag = "<span class='tag'>异常·快</span>"
                else:
                    cls = f"chip outlier-slow {tier}"
                    tag = "<span class='tag'>异常·慢</span>"
            else:
                cls = "chip"
                tag = ""
            bar_pct = (v / max_v * 100.0) if max_v > 0 else 0.0
            bar = f"<span class='bar' style='width:{bar_pct:.1f}%'></span>"
            med_note = ""
            if is_out and ref_median is not None and ref_median > 0:
                pct = (v - ref_median) / ref_median * 100.0
                med_note = (
                    f"<span class='med-note'>vs 中位 {fmt_duration_ms(ref_median)}"
                    f" ({pct:+.0f}%)</span>"
                )
            anom_parts = []
            slow_n = sum(1 for da in dur_anom_by_inst.get(i, {}).values()
                         if da["duration_us"] > da["baseline_us"])
            fast_n = sum(1 for da in dur_anom_by_inst.get(i, {}).values()
                         if da["duration_us"] < da["baseline_us"])
            extra_n = sum(1 for ca in count_anom_by_inst.get(i, {}).values()
                          if ca["count"] > ca["modal_count"])
            missing_n = sum(1 for ca in count_anom_by_inst.get(i, {}).values()
                            if ca["count"] < ca["modal_count"])
            if slow_n:
                anom_parts.append(f"<span class='hot'>{slow_n}慢</span>")
            if fast_n:
                anom_parts.append(f"<span class='cool'>{fast_n}快</span>")
            if extra_n:
                anom_parts.append(f"<span class='hot'>{extra_n}多</span>")
            if missing_n:
                anom_parts.append(f"<span class='cool'>{missing_n}缺</span>")
            anom_note = (
                f"<span class='anom-note'>{' '.join(anom_parts)}</span>"
                if anom_parts else ""
            )
            summary_inner = (
                f"{bar}{tag}"
                f"<span class='lbl'>{html.escape(m['phase'])}#{m['layer_idx']}</span>"
                f"<span class='val'>{fmt_duration_ms(v)}</span>{med_note}{anom_note}"
            )
            inst_records = records_by_inst.get(i) if expandable else None
            inst_has_anom = expandable and (
                i in dur_anom_by_inst or i in count_anom_by_inst
            )
            if inst_records or inst_has_anom:
                panel = render_inst_op_table(
                    inst_records or [], details_by_idx,
                    dur_anom_by_inst.get(i, {}),
                    count_anom_by_inst.get(i, {}),
                )
                chips.append(
                    f"<details class='chip-d'>"
                    f"<summary class='{cls}'>{summary_inner}</summary>"
                    f"<div class='panel'>{panel}</div>"
                    f"</details>"
                )
            else:
                chips.append(f"<span class='{cls}'>{summary_inner}</span>")
        return "<div class='chips'>" + "".join(chips) + "</div>"

    for sec in sections:
        ct = sec["component_type"]
        ic = sec["instance_count"]
        meta = sec.get("instances_meta", [])
        parts.append(f"<h2>{html.escape(ct)} <small>({ic} instances)</small></h2>")
        csp = sec.get("cluster_sum_pct_of_total")
        if csp is not None and csp < CLUSTER_COVERAGE_MIN_PCT:
            parts.append(
                f"<div class='coverage-warn'>"
                f"<div class='head'>⚠ cluster wall 中位之和 = {csp:.1f}% TOTAL "
                f"(&lt; {CLUSTER_COVERAGE_MIN_PCT:.0f}%)</div>"
                f"<div>该 component 大量算子落进 bubble / 未匹配 / 漏算子。"
                f"查下方 Unmatched 面板与 cluster 规则；可能是 catch_all 把真正语义"
                f"算子吞了，或某段 op 完全没被规则覆盖。</div>"
                f"</div>"
            )
        for d in sec.get("displaced") or []:
            parts.append(
                f"<div class='overlap-info'>"
                f"<div class='head'>⟂ 辅流 stream {html.escape(str(d['stream_id']))} "
                f"时间脱节，已从 TOTAL/bubble 排除</div>"
                f"<div>{d['op_count']} op；median wall "
                f"{fmt_duration_ms(d['median_wall_ms'])}/实例 "
                f"({d['pct_of_total']:.1f}% of TOTAL)。op 仍 matched 在该 component，"
                f"逐层明细见 splits/ 对应文件夹。</div>"
                f"</div>"
            )
        ov = sec.get("overlap") or {}
        gap_pct = ov.get("median_gap_pct", 0.0)
        if gap_pct >= 5.0:
            pairs = ov.get("top_pairs", []) or []
            pair_chips = "".join(
                f"<span class='pair'>{html.escape(p['cluster_a'])} ↔ "
                f"{html.escape(p['cluster_b'])}: "
                f"{fmt_duration_ms(p['median_overlap_ms'])} "
                f"({p['median_overlap_pct']:.1f}% of TOTAL)</span>"
                for p in pairs
            )
            parts.append(
                f"<div class='overlap-warn'>"
                f"<div class='head'>⚠ cluster 桶之间存在显著时间轴 overlap，"
                f"cluster wall 之和重复统计 ≈ {fmt_duration_ms(ov.get('median_gap_ms', 0))} "
                f"({gap_pct:.1f}% of TOTAL, 中位)；这种分桶不能直接反映真实"
                f"性能分布。</div>"
                f"<div>主要重叠对（按中位 overlap 降序，可能来自跨 stream 并发）"
                f"</div>"
                f"<div class='pairs'>{pair_chips or '—'}</div>"
                f"</div>"
            )
        elif gap_pct > 0.0:
            # 少量 overlap：过滤掉 < 1µs 的 floating-point 噪声项后，
            # 若仍有 pair 则插一个浅蓝 info 面板提示是哪些桶在跨 stream 并发。
            pairs = [p for p in (ov.get("top_pairs", []) or [])
                     if p.get("median_overlap_ms", 0.0) >= 0.001][:8]
            if pairs:
                pair_chips = "".join(
                    f"<span class='pair'>{html.escape(p['cluster_a'])} ↔ "
                    f"{html.escape(p['cluster_b'])}: "
                    f"{fmt_duration_ms(p['median_overlap_ms'])} "
                    f"({p['median_overlap_pct']:.1f}% of TOTAL)</span>"
                    for p in pairs
                )
                parts.append(
                    f"<div class='overlap-info'>"
                    f"<div class='head'>ℹ cluster 桶之间存在少量 timeline overlap"
                    f"（中位 {fmt_duration_ms(ov.get('median_gap_ms', 0))}, "
                    f"{gap_pct:.1f}% of TOTAL）；占比低于警告阈值，但下列桶对仍"
                    f"有跨 stream 并发，cluster wall 之和会略微重复统计这部分。</div>"
                    f"<div class='pairs'>{pair_chips}</div>"
                    f"</div>"
                )
        parts.append(
            "<table>"
            "<colgroup>"
            "<col class='sub'><col class='desc'><col class='med'><col class='theory'><col><col class='insight'>"
            "</colgroup>"
            "<thead><tr>"
            "<th>sub-item</th><th>description</th>"
            "<th style='text-align:right'>median</th>"
            "<th style='text-align:right'>theory median</th>"
            "<th>per-instance &mdash; outliers highlighted</th>"
            "<th>insight</th>"
            "</tr></thead><tbody>"
        )
        for item in sec["sub_items"]:
            m = item["metric"]
            row_cls = " class='bubble'" if item["kind"] == "bubble" else ""
            is_cluster = item["kind"] == "cluster"
            chip_html = chip_cell(
                item["per_instance_ms"], meta,
                item.get("outlier_idx", []),
                item.get("kernel_outliers"),
                item.get("kernel_count_anomalies"),
                op_records=item.get("op_records") if is_cluster else None,
                median_ms=m["median_ms"],
            )
            parts.append(
                f"<tr{row_cls}>"
                f"<td class='sub'>{html.escape(item['name'])}</td>"
                f"<td class='desc'>{html.escape(item['description'])}</td>"
                f"<td class='med'>{fmt_duration_ms(m['median_ms'])}</td>"
                f"<td class='theory'>{render_theory_cell(item.get('theoretical'), m['median_ms'])}</td>"
                f"<td>{chip_html}</td>"
                f"<td class='insight'>{render_insight_cell(row_insights.get((ct, item['name'])))}</td>"
                f"</tr>"
            )
        tot = sec["total"]
        m = tot["metric"]
        parts.append(
            f"<tr class='total'>"
            f"<td class='sub'>TOTAL</td>"
            f"<td class='desc'>{html.escape(tot['description'])}</td>"
            f"<td class='med'>{fmt_duration_ms(m['median_ms'])}</td>"
            f"<td class='theory'>{render_theory_cell(tot.get('theoretical'), m['median_ms'])}</td>"
            f"<td>{chip_cell(tot['per_instance_ms'], meta, tot.get('outlier_idx', []), median_ms=m['median_ms'])}</td>"
            f"<td class='insight'>{render_insight_cell(row_insights.get((ct, 'TOTAL')))}</td>"
            f"</tr>"
        )
        parts.append("</tbody></table>")
        if sec["unmatched"]:
            top = sorted(sec["unmatched"].items(), key=lambda kv: -kv[1])[:20]
            items = " ".join(f"<code>{html.escape(n)}</code>:{c}" for n, c in top)
            parts.append(
                f"<div class='unmatched'><b>Unmatched ops</b> "
                f"({sum(sec['unmatched'].values())} total): {items}</div>"
            )
    parts.append("</body></html>")
    return "\n".join(parts)


def print_history_reminder(run_dir):
    """Hook #6：<run_dir>/HISTORY.md 缺失时打 stdout 软提示。不 exit 1，
    HISTORY 是收尾物，硬卡反而打断 agent 的写日志节奏。返回是否打印过。"""
    history_path = os.path.join(run_dir, "HISTORY.md")
    if os.path.exists(history_path):
        return False
    print(
        f"\nℹ HISTORY.md 缺失：{history_path}\n"
        f"  本次 run 完成后请 append 一条条目（label / prof / 复用-新建 / "
        f"模型结构 / sample 锁定 / cluster 规则要点 / 渲染产物 / 关键观察）。\n"
        f"  模板见 references/history_template.md。",
        flush=True,
    )
    return True


# Communication cores never live inside an attn/ffn component, so they are
# excluded from the gate denominator on every architecture. Ops without a core
# field degrade naturally to "UNKNOWN" → counted → raw %, so no separate
# fallback branch is needed. (AI_CPU is already dropped upstream in sample mode.)
NON_LAYER_CORES = {"COMMUNICATION", "HCCL"}


def enforce_unmatched_gates(draft, operators, *, accept=False,
                            limit=UNMATCHED_PCT_HARD_LIMIT):
    """Gate when too many *compute* ops fell outside every component — the
    signal that a component/sample was mis-declared.

    The gate metric excludes communication because those are expected to be
    unmatched everywhere; gating on raw % mis-fires on comm/sampling-heavy
    workloads (e.g. decode + MoE all-to-all). The message prints the per-core
    breakdown so the agent can tell "expected IO/sampling" from "missed a real
    compute component". --accept-unmatched releases the gate.
    """
    unmatched_set = {int(i) for i in (draft.get("unmatched_op_indices") or [])}
    total_ops = len(operators)
    if total_ops <= 0:
        return

    by_idx = {int(o.get("index", n)): o for n, o in enumerate(operators)}
    core_counts = {}
    for i in unmatched_set:
        core = (by_idx.get(i, {}) or {}).get("accelerator_core") or "UNKNOWN"
        core_counts[core] = core_counts.get(core, 0) + 1

    def is_gated(op):  # comm excluded; missing core → "UNKNOWN" → gated (= raw %)
        return (op.get("accelerator_core") or "UNKNOWN") not in NON_LAYER_CORES
    gated_total = sum(1 for o in operators if is_gated(o)) or total_ops
    gated_unmatched = sum(1 for i in unmatched_set if is_gated(by_idx.get(i, {}) or {}))
    pct = gated_unmatched / gated_total

    if pct <= limit:
        return

    print(
        f"\n[Step 3 unmatched gate] {gated_unmatched} compute-core unmatched "
        f"(通信类已排除) = {pct * 100:.1f}% (阈值 {limit * 100:.0f}%; 总 unmatched "
        f"{len(unmatched_set)}/{total_ops})。",
        file=sys.stderr,
    )
    if core_counts:
        brk = ", ".join(f"{c}×{n}" for c, n in
                        sorted(core_counts.items(), key=lambda kv: -kv[1]))
        print(f"  - unmatched 按 accelerator_core: {brk}", file=sys.stderr)
    sample = sorted(unmatched_set)[:20]
    print(f"  - unmatched_op_indices sample: {sample}", file=sys.stderr)
    print(
        "  - 判断：若 unmatched 主要是通信/采样/embedding/IO 等非 layer flow，"
        "属正常，加 --accept-unmatched 放行；若有大段 compute 算子漏匹配，"
        "回 Phase 0a/0b 补 sample 或声明 component。可用 --unmatched-limit 调阈值。",
        file=sys.stderr,
    )

    if accept:
        print("[--accept-unmatched] 用户已知情放行，继续渲染。", file=sys.stderr)
        return
    raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--draft", required=True,
                   help="structure_draft.json from Step 2 stream_sample_driven mode")
    p.add_argument("-r", "--raw_ops", required=True,
                   help="raw_ops.json from Step 1")
    p.add_argument("--raw-ops-details", dest="raw_ops_details",
                   help="raw_ops_details.json from Step 1 (full CSV rows). "
                        "When provided, each cluster row gets a collapsible "
                        "sub-table with every clustered op's full CSV row. "
                        "If enriched by merge_theoretical_columns.py, theory "
                        "median columns are computed from these per-kernel fields.")
    p.add_argument("--theory-decisions", dest="theory_decisions",
                   help="optional JSON with agent stream-selection overrides "
                        "for ambiguous multi-stream theory aggregation.")
    p.add_argument("--theory-review-ratio", type=float, default=0.80,
                   help="mark theory stream choice for review when second "
                        "largest observed stream union >= this ratio of the "
                        "largest (default: 0.80).")
    p.add_argument("--insight-annotations", dest="insight_annotations",
                   help="optional agent-authored JSON for the final insight "
                        "column. Only high/medium items are rendered. Omit on "
                        "the first Step 3 render to leave the column blank.")
    p.add_argument("-s", "--spec", required=True,
                   help="network_spec.json (per-network clustering rules)")
    p.add_argument("-o", "--output", required=True,
                   help="output HTML path")
    p.add_argument("--label",
                   help="run label embedded in metrics.json (default: parent "
                        "dir name of -o). compare_runs.py keys runs by label.")
    p.add_argument("--accept-unmatched", action="store_true",
                   help="跳过 compute-core unmatched%% > 阈值 的硬阻断。"
                        "默认 exit 1 强迫 agent 看 per-core 分解后再决定是否放行"
                        "（通信类已自动排除出 gate 指标）。")
    p.add_argument("--unmatched-limit", type=float,
                   default=UNMATCHED_PCT_HARD_LIMIT,
                   help=f"compute-core unmatched 比例硬阈值 (默认 "
                        f"{UNMATCHED_PCT_HARD_LIMIT}); 通信/AICPU 等非 layer "
                        f"core 不计入。重 IO/采样负载可调高。")
    args = p.parse_args()

    draft = load_json(args.draft)
    validate_draft_schema(draft)
    ops_data = load_json(args.raw_ops)
    operators = ops_data["operators"]
    enforce_unmatched_gates(draft, operators, accept=args.accept_unmatched,
                            limit=args.unmatched_limit)
    spec = load_json(args.spec)
    details_by_idx = None
    details_payload = None
    if args.raw_ops_details:
        details_payload = load_json(args.raw_ops_details)
        details_ops = details_payload.get("operators", [])
        details_by_idx = {o.get("index"): o for o in details_ops
                          if o.get("index") is not None}
    theory_decisions = load_theory_decisions(args.theory_decisions)

    model_name = spec.get("model_name", "model")
    outlier_cfg = spec.get("outlier", {"method": "iqr", "k": 1.5})
    cluster_defs = spec.get("component_clusters") or {}
    if not cluster_defs:
        raise SystemExit("network_spec.json 缺 component_clusters")

    by_type = instances_by_type_from_draft(draft)

    sections = []
    for comp_type, instances in by_type.items():
        if comp_type not in cluster_defs:
            # No rules → only synthetic bubble + TOTAL (span).
            total_span = []
            total_bubble = []
            total_theory_candidates = []
            for inst_idx, inst in enumerate(instances):
                ivs = []
                op_indices = []
                for i in op_indices_from_instance(inst):
                    st = operators[i].get("start_time_us")
                    dur = operators[i].get("duration_us")
                    if st is None or dur is None:
                        continue
                    ivs.append((st, st + dur))
                    op_indices.append(i)
                w, span = wall_and_span_us(ivs)
                total_span.append(span)
                total_bubble.append(max(0.0, span - w))
                if details_by_idx:
                    total_theory_candidates.append(build_theory_candidate(
                        comp_type, "TOTAL", inst_idx, inst, op_indices,
                        operators, details_by_idx, theory_decisions,
                        args.theory_review_ratio,
                    ))
            method = outlier_cfg.get("method", "iqr")
            k = outlier_cfg.get("k", 1.5)
            b_idx, b_outs = outliers_for(total_bubble, instances, method, k)
            t_idx, t_outs = outliers_for(total_span, instances, method, k)
            sections.append({
                "component_type": comp_type,
                "instance_count": len(instances),
                "instances_meta": [
                    {"phase": inst["phase"], "layer_idx": inst["layer_idx"]}
                    for inst in instances
                ],
                "sub_items": [{
                    "name": "bubble",
                    "kind": "bubble",
                    "description": "layer idle gap (span − wall)",
                    "metric": series_metrics(total_bubble),
                    "outliers": b_outs,
                    "outlier_idx": sorted(b_idx),
                    "per_instance_ms": [v / 1000.0 for v in total_bubble],
                }],
                "total": {
                    "description": "layer end-to-end span (max_end − min_start, 含 bubble)",
                    "metric": series_metrics(total_span),
                    "outliers": t_outs,
                    "outlier_idx": sorted(t_idx),
                    "per_instance_ms": [v / 1000.0 for v in total_span],
                    "theoretical": summarize_theory(total_theory_candidates),
                },
                "unmatched": {"<no rules defined for this component type>":
                              sum(len(op_indices_from_instance(inst)) for inst in instances)},
                "overlap": {"median_gap_ms": 0.0, "median_gap_pct": 0.0,
                            "max_gap_ms": 0.0, "top_pairs": []},
            })
            continue
        compiled = compile_clusters(cluster_defs[comp_type])
        sub_items, total, unmatched, overlap_summary, displaced_summary = analyze_component_type(
            comp_type, instances, operators, compiled, outlier_cfg,
            details_by_idx=details_by_idx,
            theory_decisions=theory_decisions,
            theory_review_ratio=args.theory_review_ratio,
        )
        cluster_medians = [s["metric"]["median_ms"] for s in sub_items
                           if s.get("kind") == "cluster"]
        total_median = total["metric"]["median_ms"]
        cluster_sum_pct = (sum(cluster_medians) / total_median * 100
                           if cluster_medians and total_median > 0 else None)
        sections.append({
            "component_type": comp_type,
            "instance_count": len(instances),
            "instances_meta": [
                {"phase": inst["phase"], "layer_idx": inst["layer_idx"]}
                for inst in instances
            ],
            "sub_items": sub_items,
            "total": total,
            "unmatched": unmatched,
            "overlap": overlap_summary,
            "cluster_sum_pct_of_total": cluster_sum_pct,
            "displaced": displaced_summary,
        })

    spec_order = list(cluster_defs.keys())
    sections.sort(key=lambda s: (
        spec_order.index(s["component_type"]) if s["component_type"] in spec_order
        else len(spec_order),
        s["component_type"],
    ))

    row_insights = load_agent_insight_annotations(args.insight_annotations)
    html_out = render_html(
        model_name, sections, outlier_cfg, details_by_idx, row_insights
    )
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"dashboard → {args.output}", flush=True)

    label = args.label or os.path.basename(out_dir) or "run"
    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics_payload = {
        "label": label,
        "model_name": model_name,
        "outlier": outlier_cfg,
        "generated_at": int(time.time()),
        "theoretical_perf_source": (
            details_payload.get("theoretical_perf_source") if details_payload else None
        ),
        "theory_decisions": os.path.abspath(args.theory_decisions)
        if args.theory_decisions else None,
        "insight_annotations": os.path.abspath(args.insight_annotations)
        if args.insight_annotations else None,
        "sections": [
            {
                "component_type": sec["component_type"],
                "instance_count": sec["instance_count"],
                "instances_meta": sec.get("instances_meta", []),
                "sub_items": sec["sub_items"],
                "total": sec["total"],
                "unmatched": sec["unmatched"],
                "overlap": sec.get("overlap", {}),
                "cluster_sum_pct_of_total": sec.get("cluster_sum_pct_of_total"),
                "displaced": sec.get("displaced", []),
            }
            for sec in sections
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"metrics  → {metrics_path} (label={label})", flush=True)

    for sec in sections:
        ct = sec["component_type"]
        ic = sec["instance_count"]
        unmatched_n = sum(sec["unmatched"].values())
        clusters_n = sum(1 for s in sec["sub_items"] if s["kind"] == "cluster")
        print(f"  {ct}: {ic} instances, {clusters_n} clusters + bubble, "
              f"unmatched ops: {unmatched_n}", flush=True)

    run_dir = os.path.dirname(os.path.abspath(args.draft)) or "."
    print_history_reminder(run_dir)


if __name__ == "__main__":
    main()
