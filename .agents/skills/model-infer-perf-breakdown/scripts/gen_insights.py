#!/usr/bin/env python3
# coding=utf-8
"""Step 5 — 洞察事实提取（脚本角色，纯机械，不做语义判断）。

执行者边界（见 references/insight_workflow.md §1）：
  脚本（本文件）：join / group-by / 排序 / 区间扫描 / 统计 / 选 Top-K 代表 / 落 JSON。
  agent：定义 movement taxonomy、审阅可信度、写 agent_review（summary/confidence/
         reason/fusion_candidate/elimination_direction）、写 final_conclusions。

因此本脚本**绝不**内置：模型/component/cluster/op/stream 名、搬运或融合 op 规则、
“理论不可靠”算子名单、性能原因、confidence。这些一律来自 agent：
  --movement-taxonomy <json>   agent 定义的 movement family（Insight 5 必需，缺省则只
                               产 op-frequency 候选清单 _taxonomy_candidates.json）。
  --annotations <json>         agent 审阅后填的 agent_review，按 stable key 合并。
缺省时 agent_review 字段留空，并产 _review_stub.json 列出所有待审 key。

排序口径：偏离/抖动一律按**绝对增量**（gap / delta）为主键——绝对量对不同模型/shape
更稳健，且天然压低“理论值极小→ratio 虚高”的小算子噪声（是否可信仍由 agent 用
confidence 标注）。展示层（render_insights.py）再按 confidence 重排。

用法:
  python gen_insights.py \
    --metrics <run>/runs/<label>/metrics.json \
    --raw-ops <run>/raw_ops.json \
    --structure-draft <run>/structure_draft.json \
    --raw-ops-details <run>/raw_ops_details.json \
    --out-dir <run>/runs/<label>/insights \
    [--movement-taxonomy <agent.json>] [--annotations <agent.json>] \
    [--top-k 5] [--small-n-ratio 1.5]

输入事实来源：metrics.json（cluster/bubble/TOTAL/outlier/theoretical 摘要、op_records
的 op→cluster 归属）、raw_ops.json（核类型/时间/stream/连续 vector 扫描/搬运邻接）、
raw_ops_details.json（per-kernel duration_over_theoretical 等理论列）、structure_draft.json
（op_to_component / component 边界）。
"""
import argparse
import json
import os
import statistics
from collections import defaultdict, Counter


def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def median(xs):
    return statistics.median(xs) if xs else 0.0


def us(ms):
    return round(ms * 1000, 1) if isinstance(ms, (int, float)) else None


def small_n_outlier_idx(values, ratio):
    """与 render.detect_outliers 同口径：n>=4 用 IQR，2<=n<=3 用相对尖峰兜底。"""
    n = len(values)
    if n < 2:
        return set()
    if n < 4:
        med = median(values)
        if med <= 0 or not ratio or ratio <= 0:
            return set()
        return {i for i, v in enumerate(values) if v > med * ratio}
    sv = sorted(values)
    q1, q3 = sv[n // 4], sv[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return {i for i, v in enumerate(values) if v < q1 or v > q3}
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {i for i, v in enumerate(values) if v < lo or v > hi}


EMPTY_REVIEW = {"summary": "", "confidence": "", "reason": "", "needs_followup": None}


def review_for(stub_registry, key, annotations, extra_fields=None):
    """登记一个待审 key，返回合并后的 agent_review（缺省全空，等 agent 填）。"""
    base = dict(EMPTY_REVIEW)
    if extra_fields:
        base.update(extra_fields)
    stub_registry[key] = dict(base)
    if annotations and key in annotations:
        merged = dict(base)
        merged.update({k: v for k, v in annotations[key].items() if v is not None})
        return merged
    return base


def topk_with_annotated(cands, k, key_fn, annotations):
    """取前 k（已按事实排序），但任何被 agent 注解过的候选强制保留（agent 可越过
    脚本排序选要展示的项）。"""
    if not annotations:
        return cands[:k]
    annotated = [c for c in cands if key_fn(c) in annotations]
    head = cands[:k]
    seen = {id(c) for c in head}
    for c in annotated:
        if id(c) not in seen:
            head.append(c)
            seen.add(id(c))
    return head


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--raw-ops", required=True)
    ap.add_argument("--structure-draft", required=True)
    ap.add_argument("--raw-ops-details", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--movement-taxonomy", default=None,
                    help="agent 定义的 movement family JSON（Insight 5）。缺省只产候选清单。")
    ap.add_argument("--annotations", default=None,
                    help="agent agent_review 注解 JSON（按 stable key 合并）。")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--small-n-ratio", type=float, default=1.5)
    args = ap.parse_args()

    K = args.top_k
    m = load(args.metrics)
    raw = load(args.raw_ops)
    draft = load(args.structure_draft)
    details = load(args.raw_ops_details)
    annotations = load(args.annotations) if args.annotations else None
    if annotations and "annotations" in annotations:
        annotations = annotations["annotations"]
    taxonomy = load(args.movement_taxonomy) if args.movement_taxonomy else None

    os.makedirs(args.out_dir, exist_ok=True)
    ops = sorted(raw["operators"], key=lambda o: o["index"])
    byidx = {o["index"]: o for o in ops}
    o2c = draft.get("op_to_component", {})
    comp_by_id = {c["component_id"]: c for c in draft.get("components", [])}
    det_by_idx = {d["index"]: d for d in details.get("operators", [])
                  if d.get("index") is not None}

    def sec(ct):
        return next((s for s in m["sections"] if s["component_type"] == ct), None)

    stub = {}  # stable key -> empty review (for _review_stub.json)

    # op_idx -> (component_type, cluster) 来自 metrics op_records（不重评 network_spec 规则）
    op_cluster = {}
    for s in m["sections"]:
        ct = s["component_type"]
        for si in s["sub_items"]:
            if si.get("kind") != "cluster":
                continue
            for rec in si.get("op_records", []):
                op_cluster[rec["op_idx"]] = (ct, si["name"])

    def find_comp(phase, layer, ctype):
        return next((c for c in draft.get("components", [])
                     if c["phase"] == phase and c["layer_idx"] == layer and c["type"] == ctype), None)

    def env_of(cobj):
        return [min(cobj["op_indices"]), max(cobj["op_indices"])] if cobj else None

    # ================= Insight 1: module_bubble =================
    persistent = []
    inst_out = []
    for s in m["sections"]:
        ct = s["component_type"]
        bub = next((si for si in s["sub_items"] if si.get("name") == "bubble"), None)
        if not bub:
            continue
        bpi = bub.get("per_instance_ms", [])
        tpi = s["total"].get("per_instance_ms", [])
        if not bpi:
            continue
        med = median(bpi)
        pct = [(b / t * 100) for b, t in zip(bpi, tpi) if t]
        meta = s.get("instances_meta", [])
        key = f"persistent:{ct}"
        rep = []
        for i in sorted(range(len(bpi)), key=lambda i: -bpi[i])[:K]:
            mi = meta[i] if i < len(meta) else {}
            env = env_of(find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
            rep.append({"phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                        "bubble_us": us(bpi[i]), "op_range_envelope": env})
        persistent.append({
            "component_type": ct, "instance_count": len(bpi),
            "median_bubble_us": us(med),
            "p90_bubble_us": us(sorted(bpi)[int(0.9 * (len(bpi) - 1))]),
            "max_bubble_us": us(max(bpi)),
            "median_bubble_pct_of_total": round(median(pct), 2) if pct else None,
            "high_bubble_instance_ratio": round(sum(1 for p in pct if p > 5) / len(pct), 2) if pct else 0,
            "representative_instances": rep,
            "agent_review": review_for(stub, key, annotations),
            "_sort": med,
        })
        # instance outliers within type (same-type baseline)
        for i in small_n_outlier_idx(bpi, args.small_n_ratio):
            mi = meta[i] if i < len(meta) else {}
            env = env_of(find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
            ikey = f"instance_bubble:{mi.get('phase')}:{mi.get('layer_idx')}:{ct}"
            inst_out.append({
                "phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                "component_type": ct, "bubble_us": us(bpi[i]),
                "bubble_pct_of_total": round(pct[i], 2) if i < len(pct) else None,
                "delta_vs_type_median_us": us(bpi[i] - med),
                "ratio_vs_type_median": round(bpi[i] / med, 2) if med else None,
                "evidence": {"op_range_envelope": env},
                "agent_review": review_for(stub, ikey, annotations),
                "_sort": bpi[i] - med,
            })
    persistent.sort(key=lambda x: -x["_sort"])
    inst_out.sort(key=lambda x: -x["_sort"])
    for x in persistent + inst_out:
        x.pop("_sort", None)
    json.dump({"schema_version": "insight.module_bubble.v1",
               "selection_policy": {"top_k": K, "baseline": "same_component_type",
                                    "sort": "median_bubble_us / delta_vs_type_median_us"},
               "persistent_module_bubbles": persistent[:K],
               "instance_bubble_outliers": inst_out[:K]},
              open(f"{args.out_dir}/module_bubble.json", "w"), ensure_ascii=False, indent=2)

    # ================= Insight 2: operator_jitter =================
    cluster_cand = []
    for s in m["sections"]:
        ct = s["component_type"]
        meta = s.get("instances_meta", [])
        for si in s["sub_items"]:
            if si.get("kind") != "cluster":
                continue
            pi = si.get("per_instance_ms", [])  # cluster wall = interval union (per doc)
            if len(pi) < 2:
                continue
            med = median(pi)
            for i in small_n_outlier_idx(pi, args.small_n_ratio):
                mi = meta[i] if i < len(meta) else {}
                env = env_of(find_comp(mi.get("phase"), mi.get("layer_idx"), ct))
                key = f"cluster_jitter:{ct}:{si['name']}:{mi.get('phase')}:{mi.get('layer_idx')}"
                cluster_cand.append({
                    "component_type": ct, "cluster": si["name"],
                    "phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                    "duration_us": us(pi[i]), "baseline_median_us": us(med),
                    "delta_us": us(pi[i] - med),
                    "ratio_vs_baseline": round(pi[i] / med, 2) if med else None,
                    "evidence": {"op_range_envelope": env},
                    "agent_review": review_for(stub, key, annotations),
                    "_sort": pi[i] - med,
                })
    cluster_cand.sort(key=lambda x: -(x["_sort"] or 0))

    # operator level: consume kernel_outliers from each cluster sub_item. render.py
    # already slots kernels by (op_name, occurrence) within a cluster instance, so the
    # same op name at different semantic positions stays separate (occurrence). Per
    # insight_workflow doc: prefer kernel_outliers; a cluster without kernel_outliers
    # yields no operator-level candidate (do not re-aggregate raw duration nor fabricate
    # operator granularity by collapsing occurrences under normalized_name).
    op_slot = defaultdict(list)
    for s in m["sections"]:
        ct = s["component_type"]
        for si in s["sub_items"]:
            if si.get("kind") != "cluster":
                continue
            for ko in si.get("kernel_outliers", []):
                op_slot[(ct, si["name"], ko.get("op_name"), ko.get("occurrence"))].append(ko)
    op_cand = []
    for (ct, cl, nm, occ), kos in op_slot.items():
        worst = max(kos, key=lambda k: abs((k.get("duration_us") or 0) - (k.get("baseline_median_us") or 0)))
        dur = worst.get("duration_us") or 0
        base = worst.get("baseline_median_us") or 0
        key = f"op_jitter:{ct}:{cl}:{nm}:{occ}"
        op_cand.append({
            "component_type": ct, "cluster": cl, "op_name": nm, "occurrence": occ,
            "phase": worst.get("phase"), "layer_idx": worst.get("layer_idx"),
            "duration_us": round(dur, 1), "baseline_median_us": round(base, 1),
            "delta_us": round(dur - base, 1),
            "ratio_vs_baseline": round(dur / base, 2) if base else None,
            "agent_review": review_for(stub, key, annotations),
            "_sort": abs(dur - base),
        })
    op_cand.sort(key=lambda x: -x["_sort"])
    cc = topk_with_annotated(
        cluster_cand, K,
        lambda c: f"cluster_jitter:{c['component_type']}:{c['cluster']}:{c['phase']}:{c['layer_idx']}",
        annotations)
    oc = topk_with_annotated(
        op_cand, K,
        lambda c: f"op_jitter:{c['component_type']}:{c['cluster']}:{c['op_name']}:{c['occurrence']}",
        annotations)
    for x in cluster_cand + op_cand:
        x.pop("_sort", None)
    json.dump({"schema_version": "insight.operator_jitter.v1",
               "selection_policy": {"baseline": "same_component_type_and_cluster", "top_k": K,
                                    "cluster_metric_source": "metrics.sections[].sub_items[kind=cluster].per_instance_ms",
                                    "cluster_metric_semantics": "cluster wall = interval union, not sum(duration_us)",
                                    "operator_metric_source": "metrics.sections[].sub_items[kind=cluster].kernel_outliers",
                                    "operator_slot": "(component_type, cluster, op_name, occurrence)",
                                    "sort": "delta_us (absolute)"},
               "cluster_jitter_candidates": cc,
               "operator_jitter_candidates": oc},
              open(f"{args.out_dir}/operator_jitter.json", "w"), ensure_ascii=False, indent=2)

    # ================= Insight 3: theoretical_deviation =================
    data_limits = []
    if not det_by_idx or not any(d.get("theory_supported") for d in det_by_idx.values()):
        data_limits.append(
            "理论性能未启用（Step 1.5 可选、默认不开），已跳过实测/理论 gap 分析；"
            "如需对比再开 Step 1.5。")
    sub_dev = []
    for s in m["sections"]:
        ct = s["component_type"]
        items = [(si["name"], si) for si in s["sub_items"] if si["name"] != "other"]
        items.append(("TOTAL", s["total"]))
        for name, si in items:
            th = si.get("theoretical") or {}
            amed = si["metric"].get("median_ms")
            tmed = th.get("median_theoretical_ms")
            if not (amed and tmed):
                continue
            gap = (amed - tmed) * 1000
            if abs(gap) < 1:
                continue
            meta = s.get("instances_meta", [])
            pi = si.get("per_instance_ms", [])
            tpi = th.get("per_instance", [])
            rep = []
            for i in sorted(range(len(pi)), key=lambda i: -pi[i])[:K]:
                mi = meta[i] if i < len(meta) else {}
                cobj_r = find_comp(mi.get("phase"), mi.get("layer_idx"), ct)
                rep_ops = [oi for oi in (cobj_r.get("op_indices", []) if cobj_r else [])
                           if name == "TOTAL" or op_cluster.get(oi, (None, None))[1] == name]
                rep.append({"phase": mi.get("phase"), "layer_idx": mi.get("layer_idx"),
                            "actual_ms": round(pi[i], 4),
                            "theoretical_ms": round(tpi[i]["effective_theoretical_us"] / 1000, 4)
                            if i < len(tpi) and tpi[i].get("effective_theoretical_us") is not None else None,
                            "op_indices": rep_ops,
                            "op_range_envelope": [min(rep_ops), max(rep_ops)] if rep_ops else None})
            key = f"sub_dev:{ct}:{name}"
            sub_dev.append({
                "component_type": ct, "sub_item": name,
                "actual_median_ms": round(amed, 4), "theoretical_median_ms": round(tmed, 4),
                "wall_over_theoretical_median": round(amed / tmed, 2) if tmed else None,
                "absolute_gap_us": round(gap, 1),
                # 理论量级——agent 据此判可信度（极小理论值→ratio 不可信），脚本不替它判
                "theoretical_magnitude_us": round(tmed * 1000, 1),
                "support_ratio": round(th.get("supported_instance_count", 0) / max(th.get("instance_count", 1), 1), 2),
                "representative_instances": rep,
                "agent_review": review_for(stub, key, annotations),
                "_sort": gap,
            })
    sub_dev.sort(key=lambda x: -x["_sort"])
    # operator slot deviation via raw_ops_details duration_over_theoretical.
    # occurrence = Nth appearance of op_name within (component instance, cluster),
    # mirroring render.py's per-instance occ_counter so the same semantic position is
    # compared across instances (insight_workflow doc: slot = (ct, cluster, op, occurrence)).
    occ_of = {}
    occ_counter = defaultdict(lambda: defaultdict(int))  # (component_id, cluster) -> op_name -> next occ
    for opidx in sorted(op_cluster):
        o = byidx.get(opidx)
        if not o:
            continue
        _ct, _cl = op_cluster[opidx]
        scope = (o2c.get(str(opidx)), _cl)
        nm = o["normalized_name"]
        occ_of[opidx] = occ_counter[scope][nm]
        occ_counter[scope][nm] += 1
    op_slot = defaultdict(list)
    for opidx, (ct, cl) in op_cluster.items():
        o = byidx.get(opidx)
        d = det_by_idx.get(opidx)
        if not o or not d or not d.get("theory_supported"):
            continue
        dot = d.get("duration_over_theoretical")
        if dot is None:
            continue
        op_slot[(ct, cl, o["normalized_name"], occ_of.get(opidx))].append(
            (dot, opidx, o.get("org_index"), o.get("duration_us"),
             d.get("theoretical_operator_time_us"), d.get("bound_type")))
    op_dev = []
    for (ct, cl, nm, occ), lst in op_slot.items():
        if len(lst) < 3:
            continue
        dots = [x[0] for x in lst]
        gaps = [(x[3] - x[4]) for x in lst if x[3] is not None and x[4] is not None]
        if not gaps:
            continue
        mgap = median(gaps)
        if mgap <= 1:
            continue
        top = sorted(lst, key=lambda x: -x[0])[:K]
        locs = []
        for dot, opidx, oi, dur, th_us, bt in top:
            cobj = comp_by_id.get(o2c.get(str(opidx)), {})
            locs.append({"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                         "org_index": oi, "duration_us": round(dur, 1) if dur else None,
                         "theoretical_us": round(th_us, 1) if th_us else None,
                         "bound_type": bt})  # 可能为 None（理论 skill 未填）
        key = f"op_dev:{ct}:{cl}:{nm}:{occ}"
        op_dev.append({
            "component_type": ct, "cluster": cl, "op_name": nm, "occurrence": occ,
            "duration_over_theoretical_median": round(median(dots), 2),
            "absolute_gap_us_median": round(mgap, 1),
            "max_duration_over_theoretical": round(max(dots), 2),
            "supported_count": len(lst),
            "top_locations": locs,
            "agent_review": review_for(stub, key, annotations),
            "_sort": mgap,
        })
    op_dev.sort(key=lambda x: -x["_sort"])
    sd = topk_with_annotated(sub_dev, K, lambda c: f"sub_dev:{c['component_type']}:{c['sub_item']}", annotations)
    od = topk_with_annotated(
        op_dev, K,
        lambda c: f"op_dev:{c['component_type']}:{c['cluster']}:{c['op_name']}:{c['occurrence']}",
        annotations)
    for x in sub_dev + op_dev:
        x.pop("_sort", None)
    json.dump({"schema_version": "insight.theoretical_deviation.v1",
               "selection_policy": {"top_k": K, "ranking_object": "logical_slot",
                                    "singleton_instances": "evidence_only",
                                    "sub_item_sort": ["absolute_gap_us"],
                                    "operator_slot_sort": ["absolute_gap_us_median"],
                                    "note": "按绝对 gap 排序（稳健，压低理论值极小的虚高 ratio）；可信度由 agent 用 confidence 标"},
               "data_limits": data_limits,
               "sub_item_deviation_candidates": sd,
               "operator_slot_deviation_candidates": od,
               "unsupported_summary": []},
              open(f"{args.out_dir}/theoretical_deviation.json", "w"), ensure_ascii=False, indent=2)

    # ================= Insight 4: vector_sequence_candidates =================
    runs = []
    cur = []
    for o in ops:
        if o.get("accelerator_core") == "AI_VECTOR_CORE":
            cur.append(o)
        else:
            if len(cur) >= 5:
                runs.append(cur)
            cur = []
    if len(cur) >= 5:
        runs.append(cur)

    def sig(run):
        names = [o["normalized_name"] for o in run]
        out, i = [], 0
        while i < len(names):
            j = i
            while j < len(names) and names[j] == names[i]:
                j += 1
            out.append(f"{names[i]} x{j - i}" if j - i > 1 else names[i])
            i = j
        return " -> ".join(out)

    patt = defaultdict(list)
    for r in runs:
        patt[sig(r)].append(r)

    def cl_of(idx):
        return op_cluster.get(idx, (None, "unknown"))[1] if idx in op_cluster else "unknown"
    pats = []
    for s_sig, rs in patt.items():
        tot = sum(o["duration_us"] for r in rs for o in r)
        comps = sorted({(comp_by_id.get(o2c.get(str(o["index"]), ""), {}) or {}).get("type", "unmatched")
                        for o in rs[0]})
        clusters = sorted({cl_of(o["index"]) for o in rs[0]})
        samples = []
        for r in sorted(rs, key=lambda r: -sum(o["duration_us"] for o in r))[:3]:
            i0 = r[0]["index"]
            cobj = comp_by_id.get(o2c.get(str(i0), ""), {}) or {}
            samples.append({"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                            "component_type": cobj.get("type", "unmatched"),
                            "cluster": cl_of(i0),
                            "op_indices": [o["index"] for o in r],
                            "op_range_envelope": [i0, r[-1]["index"]],
                            "duration_us": round(sum(o["duration_us"] for o in r), 1)})
        key = f"vec:{s_sig}"
        pats.append({"pattern_signature": s_sig, "occurrences": len(rs),
                     "total_duration_us": round(tot, 1),
                     "typical_duration_us": round(median([sum(o["duration_us"] for o in r) for r in rs]), 1),
                     "components": comps,
                     "clusters": clusters,
                     "representative_samples": samples,
                     "agent_review": review_for(stub, key, annotations,
                                                {"semantic_summary": "", "fusion_candidate": "", "confidence": ""}),
                     "_sort": tot})
    pats.sort(key=lambda x: -x["_sort"])
    pk = topk_with_annotated(pats, K, lambda c: f"vec:{c['pattern_signature']}", annotations)
    for x in pats:
        x.pop("_sort", None)
    json.dump({"schema_version": "insight.vector_sequence_candidates.v1",
               "selection_policy": {"min_ops": 5, "min_total_us": 20.0, "top_k": K,
                                    "group_by": "pattern_signature", "sort": "total_duration_us"},
               "patterns": pk},
              open(f"{args.out_dir}/vector_sequence_candidates.json", "w"), ensure_ascii=False, indent=2)

    # taxonomy candidates helper (op-name × core freq) — 给 agent 定 movement family 用
    op_freq = defaultdict(lambda: {"count": 0, "total_us": 0.0, "cores": Counter()})
    for o in ops:
        e = op_freq[o["normalized_name"]]
        e["count"] += 1
        e["total_us"] += o.get("duration_us", 0)
        e["cores"][o.get("accelerator_core")] += 1
    tax_cand = [{"op_name": k, "count": v["count"], "total_duration_us": round(v["total_us"], 1),
                 "cores": dict(v["cores"])}
                for k, v in sorted(op_freq.items(), key=lambda kv: -kv[1]["total_us"])]
    json.dump({"note": "agent 据此定义 movement taxonomy（哪些是 layout/shape/indexing/"
                       "cache_update/communication_prep 搬运），写成 --movement-taxonomy JSON。"
                       "脚本不替你判哪些是搬运。",
               "op_frequency": tax_cand},
              open(f"{args.out_dir}/_taxonomy_candidates.json", "w"), ensure_ascii=False, indent=2)

    # ================= Insight 5: data_movement_ops =================
    families_out = []
    if not taxonomy:
        # 没有 agent taxonomy → 只产候选清单，families 留空
        json.dump({"schema_version": "insight.data_movement_ops.v1",
                   "agent_taxonomy": {"families": [], "needs_user_review": []},
                   "families": [],
                   "_note": "未提供 --movement-taxonomy：请 agent 先用 _taxonomy_candidates.json "
                            "定义 movement family，再带 --movement-taxonomy 重跑。"},
                  open(f"{args.out_dir}/data_movement_ops.json", "w"), ensure_ascii=False, indent=2)
    else:
        # 兼容三种输入：顶层 {"families":[...]}、直接 [...]、或误用输出 schema 的嵌套
        # {"agent_taxonomy":{"families":[...]}}（常见踩坑：照 insight_workflow.md 的输出
        # schema 写输入文件）。
        if isinstance(taxonomy, list):
            tax_families = taxonomy
        else:
            tax_families = (taxonomy.get("families")
                            or taxonomy.get("agent_taxonomy", {}).get("families")
                            or [])
        if not tax_families:
            print('⚠ [gen_insights] 提供了 --movement-taxonomy 但解析出 0 个 family：'
                  '输入应为顶层 {"families": [...]}，不要套用输出 schema 的 '
                  'agent_taxonomy.families 嵌套。data_movement_ops 将为空。')
        op2fam = {op: f["family"] for f in tax_families for op in f.get("ops", [])}

        def neighbors(i):
            o = byidx[i]
            sid = str(o.get("stream_id"))
            same = sorted([x for x in ops if str(x.get("stream_id")) == sid],
                          key=lambda x: x.get("start_time_us", 0))
            pos = next((k for k, x in enumerate(same) if x["index"] == i), None)
            return {
                "prev_same_stream": same[pos - 1]["normalized_name"] if pos and pos > 0 else None,
                "next_same_stream": (
                    same[pos + 1]["normalized_name"]
                    if pos is not None and pos + 1 < len(same) else None
                ),
                "prev_global": byidx.get(i - 1, {}).get("normalized_name"),
                "next_global": byidx.get(i + 1, {}).get("normalized_name"),
            }
        fam_agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "dur": 0.0, "locs": []}))
        fam_tot = defaultdict(lambda: {"n": 0, "dur": 0.0})
        for o in ops:
            fam = op2fam.get(o["normalized_name"])
            if not fam:
                continue
            i = o["index"]
            cid = o2c.get(str(i), "unmatched")
            cobj = comp_by_id.get(cid, {})
            ct = cobj.get("type", "unmatched")
            cl = op_cluster.get(i, (ct, "unmatched"))[1] if i in op_cluster else "unmatched"
            fam_agg[fam][(ct, cl)]["n"] += 1
            fam_agg[fam][(ct, cl)]["dur"] += o.get("duration_us", 0)
            fam_agg[fam][(ct, cl)]["locs"].append((o.get("duration_us", 0), i, o.get("org_index"), cobj))
            fam_tot[fam]["n"] += 1
            fam_tot[fam]["dur"] += o.get("duration_us", 0)
        for fam in sorted(fam_agg, key=lambda f: -fam_tot[f]["dur"]):
            mods = []
            for (ct, cl), v in sorted(fam_agg[fam].items(), key=lambda kv: -kv[1]["dur"])[:K]:
                top = sorted(v["locs"], key=lambda x: -x[0])[:K]
                locs = [{"phase": cobj.get("phase"), "layer_idx": cobj.get("layer_idx"),
                         "duration_us": round(dur, 1), "op_indices": [i],
                         "op_range_envelope": [i, i], "op_idx": i, "org_index": oi,
                         "neighbor_context": neighbors(i)}
                        for dur, i, oi, cobj in top]
                key = f"move:{fam}:{ct}:{cl}"
                rev = review_for(stub, key, annotations,
                                 {"semantic_summary": "", "elimination_direction": "", "confidence": ""})
                mods.append({"component_type": ct, "cluster": cl,
                             "total_duration_us": round(v["dur"], 1), "occurrences": v["n"],
                             "top_locations": locs, "agent_review": rev})
            fam_ops = next((f.get("ops", []) for f in tax_families if f["family"] == fam), [])
            families_out.append({"family": fam, "ops": fam_ops,
                                 "total_duration_us": round(fam_tot[fam]["dur"], 1),
                                 "occurrences": fam_tot[fam]["n"], "by_module": mods})
        json.dump({"schema_version": "insight.data_movement_ops.v1",
                   "agent_taxonomy": {"families": tax_families,
                                      "needs_user_review": taxonomy.get("needs_user_review", [])},
                   "families": families_out[:K]},
                  open(f"{args.out_dir}/data_movement_ops.json", "w"), ensure_ascii=False, indent=2)

    # ================= review stub =================
    json.dump({"note": "agent 填 agent_review 后改名/复制为 annotations.json，带 --annotations 重跑 gen。"
                       "key 稳定，summary/confidence(high|medium|low)/reason 等留给 agent。",
               "annotations": {k: v for k, v in sorted(stub.items())}},
              open(f"{args.out_dir}/_review_stub.json", "w"), ensure_ascii=False, indent=2)

    annotated_n = len([k for k in stub if annotations and k in annotations]) if annotations else 0
    print(f"gen_insights → {args.out_dir}")
    print(f"  candidates: bubble {len(persistent)}+{len(inst_out)}, "
          f"jitter {len(cluster_cand)}+{len(op_cand)}, "
          f"theory {len(sub_dev)}+{len(op_dev)}, vector {len(pats)}, "
          f"movement {'(no taxonomy)' if not taxonomy else len(families_out)}")
    print(f"  review keys: {len(stub)}  ·  annotated: {annotated_n}"
          f"{'  (空 agent_review，待 agent 填 _review_stub.json)' if not annotations else ''}")
    if not taxonomy:
        print("  ⚠ Insight 5 待 movement taxonomy：见 _taxonomy_candidates.json，定义后带 --movement-taxonomy 重跑")


if __name__ == "__main__":
    main()
