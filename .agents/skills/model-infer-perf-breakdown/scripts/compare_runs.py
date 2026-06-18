#!/usr/bin/env python3
# coding=utf-8
"""Aggregate multiple render.py metrics.json files into a comparison HTML.

Each row is one sub-item under a component_type (one cluster, the synthetic
`bubble`, or the TOTAL layer-wall). Each column is one run. Cells show
`median_ms (Δ%)` where Δ is relative to the baseline (first column unless
overridden). Outliers per (sub-item, run) are listed inline so the reader
can see how the outlier set shifted between versions.

Inputs:
  -r runs_dir                 directory containing N */metrics.json files;
                              auto-discovered, sorted by mtime ascending.
  --runs path1 path2 …        explicit list of metrics.json paths (overrides
                              -r ordering).
  --baseline LABEL            label of the run to use as Δ baseline. Default:
                              first run (by order).
  --metric NAME               metric to compare. One of:
                                  median_ms (default) / mean_ms / p95_ms /
                                  max_ms / std_ms
  -o output                   output HTML path.
"""

import argparse
import glob
import html
import json
import os


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    data["__path"] = path
    return data


def pick_metric(metric_dict, name):
    return float(metric_dict.get(name, 0.0) or 0.0)


def delta_pct(new, base):
    """Return (Δ% as float, formatted string). None when baseline is 0."""
    if base == 0:
        if new == 0:
            return 0.0, "0.0%"
        return None, "new"
    pct = (new - base) / base * 100.0
    sign = "+" if pct > 0 else ""
    return pct, f"{sign}{pct:.1f}%"


def collect_rows(runs, metric):
    """Build flat row table.

    Returns list[dict], each:
      { component_type, sub_item, kind, description,
        per_run: {label -> {value, outliers}} }
    Rows are in first-seen order: spec/cluster order within each section,
    then 'bubble', then 'TOTAL'.
    """
    rows = {}
    order = []

    def ensure(key, **fields):
        if key not in rows:
            rows[key] = {**fields, "per_run": {}}
            order.append(key)
        return rows[key]

    for run in runs:
        label = run["label"]
        for sec in run.get("sections", []):
            ct = sec["component_type"]
            for item in sec.get("sub_items", []):
                key = (ct, item["name"])
                row = ensure(key,
                             component_type=ct,
                             sub_item=item["name"],
                             kind=item["kind"],
                             description=item.get("description", ""))
                row["per_run"][label] = {
                    "value": pick_metric(item["metric"], metric),
                    "outliers": item.get("outliers", []),
                }
            tot = sec.get("total") or {}
            key = (ct, "TOTAL")
            row = ensure(key,
                         component_type=ct,
                         sub_item="TOTAL",
                         kind="total",
                         description=tot.get("description",
                                             "layer wall (union of every op)"))
            row["per_run"][label] = {
                "value": pick_metric(tot.get("metric", {}), metric),
                "outliers": tot.get("outliers", []),
            }
    return [rows[k] for k in order]


def fmt_outlier_set(outs):
    """Compact rendering of an outlier list."""
    if not outs:
        return "—"
    return ", ".join(f"{o['phase']}#{o['layer_idx']}" for o in outs)


def classify_outlier_changes(base_outs, new_outs):
    """Return (added, removed, kept) lists of 'phase#layer' strings."""
    base_set = {(o["phase"], o["layer_idx"]) for o in base_outs}
    new_set = {(o["phase"], o["layer_idx"]) for o in new_outs}
    added = sorted(new_set - base_set)
    removed = sorted(base_set - new_set)
    kept = sorted(new_set & base_set)

    def f(items):
        return [f"{p}#{i}" for p, i in items]
    return f(added), f(removed), f(kept)


def render_html(runs, rows, baseline_label, metric):
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           color: #222; max-width: 1500px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .meta { color: #666; font-size: 13px; margin-bottom: 18px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 5px 8px; vertical-align: top; }
    th { background: #f5f5f5; }
    th.num, td.num { text-align: right; }
    th.label, td.label, td.desc { text-align: left; }
    td.desc { color: #888; font-size: 11px; }
    tr.section-head td { background: #eef3f8; font-weight: bold; text-align: left;
                         font-size: 12px; padding: 5px 8px; }
    tr.bubble td.label { font-style: italic; }
    tr.total { background: #f9f9f9; font-weight: bold; }
    .pos { color: #b04040; }
    .neg { color: #2d8a4f; }
    .zero { color: #777; }
    .cell { display: block; min-width: 100px; }
    .cell .ms { font-weight: 500; }
    .cell .delta { font-size: 11px; margin-left: 4px; }
    .cell .outliers { display: block; color: #888; font-size: 10.5px; margin-top: 2px;
                       max-width: 220px; word-break: break-word; line-height: 1.35; }
    .cell .outliers .tag { display: inline-block; margin-right: 6px; white-space: nowrap; }
    .cell .outliers .added { color: #b04040; }
    .cell .outliers .removed { color: #2d8a4f; }
    .cell .outliers .kept { color: #999; }
    .cell .outliers .label { color: #aaa; margin-right: 4px; font-style: italic; }
    .footnote { color: #777; font-size: 11px; margin-top: 16px; }
    """
    model_name = runs[0].get("model_name", "model")
    labels = [r["label"] for r in runs]
    metric_label = metric.replace("_ms", "") + " (ms)"

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(model_name)} run history</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>{html.escape(model_name)} — run history</h1>",
        f"<div class='meta'>metric: <b>{html.escape(metric_label)}</b> &middot; "
        f"baseline: <b>{html.escape(baseline_label)}</b> &middot; "
        f"runs: {len(runs)} &middot; "
        f"cells: value · Δ% vs baseline · outlier change tags "
        f"(<span class='pos'>+new</span> / <span class='neg'>−gone</span> / "
        f"<span style='color:#999'>=still</span>)</div>",
    ]
    parts.append("<table><thead><tr>")
    parts.append("<th class='label'>component / sub-item</th>")
    parts.append("<th class='label'>description</th>")
    for lab in labels:
        marker = " (baseline)" if lab == baseline_label else ""
        parts.append(f"<th class='num'>{html.escape(lab + marker)}</th>")
    parts.append("</tr></thead><tbody>")

    current_ct = None
    for row in rows:
        if row["component_type"] != current_ct:
            current_ct = row["component_type"]
            parts.append(
                f"<tr class='section-head'><td colspan='{2 + len(labels)}'>"
                f"{html.escape(current_ct)}</td></tr>"
            )
        kind = row["kind"]
        if kind == "total":
            row_cls = "total"
        elif kind == "bubble":
            row_cls = "bubble"
        else:
            row_cls = ""
        cls_attr = f" class='{row_cls}'" if row_cls else ""
        parts.append(f"<tr{cls_attr}>")
        parts.append(f"<td class='label'><b>{html.escape(row['sub_item'])}</b></td>")
        parts.append(f"<td class='desc'>{html.escape(row['description'])}</td>")
        base = row["per_run"].get(baseline_label)
        for lab in labels:
            cell = row["per_run"].get(lab)
            if cell is None:
                parts.append("<td class='num'>—</td>")
                continue
            value = cell["value"]
            outs = cell.get("outliers", [])
            if lab == baseline_label or base is None:
                if outs:
                    body = (f"<span class='label'>outliers:</span>"
                            f"<span class='kept'>{html.escape(fmt_outlier_set(outs))}</span>")
                else:
                    body = "<span class='label'>outliers:</span>—"
                parts.append(
                    f"<td class='num'><span class='cell'>"
                    f"<span class='ms'>{value:.3f}</span>"
                    f"<span class='outliers'>{body}</span></span></td>"
                )
                continue
            _, d_str = delta_pct(value, base["value"])
            dcls = "pos" if d_str.startswith("+") else "neg" if d_str.startswith("-") else "zero"
            added, removed, kept = classify_outlier_changes(
                base.get("outliers", []), outs
            )
            tags = []
            for s in added:
                tags.append(f"<span class='tag added'>+{html.escape(s)}</span>")
            for s in removed:
                tags.append(f"<span class='tag removed'>−{html.escape(s)}</span>")
            for s in kept:
                tags.append(f"<span class='tag kept'>={html.escape(s)}</span>")
            outs_html = (
                "<span class='outliers'>" + "".join(tags) + "</span>"
                if tags else ""
            )
            parts.append(
                f"<td class='num'><span class='cell'>"
                f"<span class='ms'>{value:.3f}</span>"
                f"<span class='delta {dcls}'>{html.escape(d_str)}</span>"
                f"{outs_html}</span></td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    parts.append(
        "<div class='footnote'>Outlier method: IQR (k=1.5) by default. "
        "Baseline cell lists this run's full outlier set; comparison cells "
        "list only the changes vs baseline.</div>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("-r", "--runs-dir",
                     help="directory holding N */metrics.json files")
    src.add_argument("--runs", nargs="+",
                     help="explicit list of metrics.json paths in order")
    p.add_argument("--baseline",
                   help="label of baseline run (default: first run by order)")
    p.add_argument("--metric", default="median_ms",
                   choices=["median_ms", "mean_ms", "p95_ms", "max_ms", "std_ms"])
    p.add_argument("-o", "--output", required=True, help="output HTML path")
    args = p.parse_args()

    if args.runs_dir:
        paths = sorted(
            glob.glob(os.path.join(args.runs_dir, "*", "metrics.json")),
            key=os.path.getmtime,
        )
        if not paths:
            raise SystemExit(
                f"no */metrics.json found under {args.runs_dir!r}"
            )
    else:
        paths = args.runs

    runs = [load_metrics(p_) for p_ in paths]
    labels = [r["label"] for r in runs]
    if len(set(labels)) != len(labels):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise SystemExit(f"duplicate labels in runs: {dupes}")

    baseline = args.baseline or labels[0]
    if baseline not in labels:
        raise SystemExit(
            f"baseline {baseline!r} not in run labels {labels}"
        )

    rows = collect_rows(runs, args.metric)
    out_html = render_html(runs, rows, baseline, args.metric)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(out_html)
    print(f"history → {args.output} ({len(runs)} runs, baseline={baseline}, "
          f"metric={args.metric})", flush=True)
    for r in runs:
        print(f"  {r['label']:<20} ← {r['__path']}", flush=True)


if __name__ == "__main__":
    main()
