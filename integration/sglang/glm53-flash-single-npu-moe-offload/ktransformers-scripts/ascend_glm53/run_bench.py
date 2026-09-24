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
"""GSM8K and GPQA-Diamond against the running server, through evalscope.

Defaults come from glm53_env.sh (GLM53_HOST / GLM53_PORT / GLM53_MODEL_PATH /
GLM53_MAX_TOTAL_TOKENS / GLM53_MAX_RUNNING_REQUESTS / GLM53_ARTIFACT_ROOT),
so after sourcing it only --dataset is usually needed:

    source glm53_env.sh && "$GLM53_ENV_ROOT/.venv-eval/bin/python" run_bench.py \
        --dataset gsm8k

⚠ The interpreter above is NOT $GLM53_PYTHON. evalscope is deliberately kept out of
the project venv: it pulls a large dependency set that would fight the pinned
torch/torch_npu, so it lives in its own venv and this script runs with that one.
The guide's accuracy step creates it at $GLM53_ENV_ROOT/.venv-eval.

Why this exists alongside run_ppl.py, and what it can and cannot settle:

  * Perplexity is the numerical criterion. Its run-to-run noise floor on this path
    is 0 ULP, so it resolves any real change in the token distribution.
  * These two benchmarks resolve something perplexity cannot name: whether the
    deployment still answers questions correctly end to end. GSM8K's own
    resolution is coarse -- single-round binomial SE is about +/-0.47pp at
    p=0.97 over its 1319 questions, so about +/-1.3pp at 2 sigma. Treat a
    difference smaller than that as noise, and do not use either benchmark to
    clear a change that perplexity flags.
  * Whether either benchmark reaches the 512-token streaming gate depends on the
    harness, not on the benchmark name. Measured on A3 with evalscope 1.11.1:
    gsm8k is few-shot and its prompts run 557-639 tokens, so it does cross the gate
    (10 requests produced 10 `inline resident` commits and zero fallbacks);
    gpqa_diamond runs 156-479 tokens and never crosses it. The 60-120 token figure
    sometimes quoted for GSM8K comes from a zero-shot runner, not from evalscope.
    run_ppl.py, at 4096 tokens per window, crosses it unconditionally.

Reasoning effort is passed through the chat template rather than as a sampling
parameter, so it rides in extra_body:

    extra_body.chat_template_kwargs.reasoning_effort = "low" | "high" | "max"

The template guards it as

    reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low','high'] else 'max'

so "max" is the default the template falls back to rather than a value it matches.
Passing "max" and passing nothing land in the same branch; --effort max is spelled
out here so the run records which level it meant.

Sampling and generation defaults follow the vendor recipe:

    --seed 42
    max_tokens=16384, temperature=1.0, top_p=0.95, timeout=3600

The recipe also names the endpoint as port 30003 and the model as the short label
"GLM-5.3-Flash-W8A8". Both belong to the 8-die TP8 box it was written for. Here the
port comes from GLM53_PORT and the model defaults to GLM53_MODEL_PATH, because this
server is started without --served-model-name, so its served name is the path.
Pass --model to override if your server advertises the short label.

Concurrency is capped on purpose. The KV pool on a single die is
GLM53_MAX_TOTAL_TOKENS (40960 by default), and exhausting it does not queue and
does not slow down -- it takes the device out with a vector core exception. The
guard below refuses a batch size whose worst case does not fit.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DATASETS = {
    "gsm8k": "gsm8k",
    "gpqa": "gpqa_diamond",
    "gpqa_diamond": "gpqa_diamond",
}

DEFAULT_HOST = os.environ.get("GLM53_HOST") or "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("GLM53_PORT") or 30013)
DEFAULT_MODEL = os.environ.get("GLM53_MODEL_PATH") or ""
DEFAULT_KV = int(os.environ.get("GLM53_MAX_TOTAL_TOKENS") or 40960)
DEFAULT_RUNNING = int(os.environ.get("GLM53_MAX_RUNNING_REQUESTS") or 1)
DEFAULT_OUT = Path(os.environ.get("GLM53_ARTIFACT_ROOT") or "/var/tmp/glm53") / "bench"


def url_host(host: str) -> str:
    """Return a connectable URL host for wildcard and IPv6 bind addresses."""
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in {"::", "[::]"}:
        return "[::1]"
    if host.startswith("[") and host.endswith("]"):
        return host
    return f"[{host}]" if ":" in host else host


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                   help="gsm8k, or gpqa / gpqa_diamond")
    p.add_argument("--effort", default="max", choices=["low", "high", "max"],
                   help="reasoning_effort passed to the chat template; max is the "
                        "template's own default (default max)")
    p.add_argument("--limit", type=int, default=None,
                   help="questions per subset; omit for the full set")
    p.add_argument("--repeats", type=int, default=1, help="sampling rounds per question")
    p.add_argument("--batch", type=int, default=None,
                   help="concurrent requests; defaults to GLM53_MAX_RUNNING_REQUESTS")
    p.add_argument("--max-tokens", type=int, default=16384, help="generation cap")
    p.add_argument("--timeout", type=int, default=3600, help="per-request timeout, seconds")
    p.add_argument("--seed", type=int, default=42, help="evalscope seed")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="model field sent to the server; defaults to GLM53_MODEL_PATH")
    p.add_argument("--out", default=None, help="work-dir; defaults under GLM53_ARTIFACT_ROOT")
    p.add_argument("--resume", default=None, metavar="DIR",
                   help="reuse the predictions already in DIR (a previous run's work-dir) "
                        "and send only the questions still missing; results go back into DIR")
    p.add_argument("--ms-cache", default=os.environ.get("MODELSCOPE_CACHE"),
                   help="ModelScope cache root holding the corpus")
    return p.parse_args(argv)


def check_budget(batch: int, max_tokens: int, kv: int) -> None:
    """Refuse a batch whose worst case oversubscribes the KV pool.

    Exhausting the pool is not a slowdown and not a queue -- it is a device-side
    vector core exception that takes the card down, so this is a hard stop rather
    than a warning.
    """
    worst = batch * max_tokens
    if worst > kv:
        raise ValueError(
            f"!! {batch} concurrent x {max_tokens} max_tokens = {worst} token worst case, "
            f"KV pool is {kv}.\n"
            f"   Lower --batch (<= {max(1, kv // max_tokens)}) or --max-tokens, or raise "
            f"GLM53_MAX_TOTAL_TOKENS and restart the server.\n"
            f"   Exhausting the pool takes the device out; this is refused, not warned."
        )
    LOGGER.info(f"[bench] KV budget ok: {batch} x {max_tokens} = {worst} of {kv} token "
                f"({100.0 * worst / kv:.1f}%)")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = parse_args(argv)
    dataset = DATASETS[a.dataset]
    batch = a.batch if a.batch is not None else DEFAULT_RUNNING

    if not a.model:
        sys.exit("!! no model -- source glm53_env.sh, or pass --model")
    if batch > DEFAULT_RUNNING:
        LOGGER.info(f"[bench] warn: --batch {batch} exceeds the server's "
                    f"--max-running-requests {DEFAULT_RUNNING}; the extra requests only queue")
    try:
        check_budget(batch, a.max_tokens, DEFAULT_KV)
    except ValueError as exc:
        LOGGER.error(exc)
        return 1

    if a.ms_cache:
        os.environ["MODELSCOPE_CACHE"] = a.ms_cache
    ms_cache = os.environ.get("MODELSCOPE_CACHE")

    out = Path(a.out) if a.out else DEFAULT_OUT / f"{dataset}_{a.effort}"
    out.mkdir(parents=True, exist_ok=True)

    try:
        from evalscope import run_task
        from evalscope.config import TaskConfig
    except ImportError:
        sys.exit(
            "!! evalscope is not importable by this interpreter.\n"
            "   It is deliberately not in the project venv. Create a separate one:\n"
            "     <python> -m venv <eval-venv> && <eval-venv>/bin/pip install evalscope==1.11.1\n"
            "   then run this script with <eval-venv>/bin/python."
        )

    cfg = dict(
        model=a.model,
        seed=a.seed,
        api_url=f"http://{url_host(DEFAULT_HOST)}:{DEFAULT_PORT}/v1",
        api_key="EMPTY",
        eval_type="openai_api",
        datasets=[dataset],
        dataset_hub="modelscope",
        work_dir=str(out),
        eval_batch_size=batch,
        repeats=a.repeats,
        timeout=a.timeout,
        generation_config={
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": a.max_tokens,
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": a.effort}},
        },
    )
    if a.limit is not None:
        cfg["limit"] = a.limit
    # evalscope matches cached predictions by sample_id and re-sends only what is missing.
    # It also overwrites work_dir with this path, so the report lands one level shallower.
    if a.resume:
        cfg["use_cache"] = a.resume
    if ms_cache:
        cfg["dataset_dir"] = str(Path(ms_cache) / "datasets")

    LOGGER.info(f"[bench] dataset={dataset} effort={a.effort} batch={batch} "
                f"repeats={a.repeats} limit={a.limit if a.limit is not None else 'full'}")
    LOGGER.info(f"[bench] seed={a.seed} max_tokens={a.max_tokens} temperature=1.0 top_p=0.95 "
                f"timeout={a.timeout}")
    LOGGER.info(f"[bench] model={a.model}")
    LOGGER.info(f"[bench] endpoint={cfg['api_url']}  work-dir={out}")

    t0_ns = time.time_ns()
    t0 = time.time()
    run_task(task_cfg=TaskConfig(**cfg))
    wall = time.time() - t0

    root = Path(a.resume) if a.resume else out
    hits = []
    for pattern in (root / "*" / "reports" / "*" / f"{dataset}.json",
                    root / "reports" / "*" / f"{dataset}.json"):
        hits.extend(glob.glob(str(pattern)))
    # A normal run must never report a file left by an earlier invocation.  Reusing
    # pre-existing output is allowed only when the operator explicitly passed --resume.
    if not a.resume:
        hits = [h for h in hits if os.stat(h).st_mtime_ns >= t0_ns]
    hits.sort(key=os.path.getmtime)
    if not hits:
        LOGGER.error(f"[bench] finished in {wall:.0f}s but no new report was written under {root}")
        return 1
    with open(hits[-1], encoding="utf-8") as report_file:
        m = json.load(report_file)["metrics"]
    # evalscope 1.11.1 stores metrics as a list with one item per metric.
    if isinstance(m, list):
        m = m[0] if m else {}
    score = m.get("score")
    pct = f"  ({100 * score:.1f}%)" if isinstance(score, (int, float)) else ""
    LOGGER.info(f"\n{dataset}  effort={a.effort}")
    LOGGER.info(f"  scored   {m.get('num')}")
    LOGGER.info(f"  score    {score}{pct}")
    LOGGER.info(f"  wall     {wall:.0f}s")
    LOGGER.info(f"  report   {hits[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
