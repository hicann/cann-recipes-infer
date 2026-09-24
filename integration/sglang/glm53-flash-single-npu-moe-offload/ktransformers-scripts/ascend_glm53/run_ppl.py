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
"""Held-out perplexity against the running server, in long windows.

Defaults come from glm53_env.sh (GLM53_HOST / GLM53_PORT / GLM53_MODEL_PATH /
GLM53_EVAL_DIR), so after sourcing glm53_env.sh only --limit is usually needed.

⚠ Run it with $GLM53_PYTHON, not by ./ -- the shebang finds whatever python is on PATH,
and transformers/pyarrow live in the project venv:

    source glm53_env.sh && "$GLM53_PYTHON" run_ppl.py --limit 12 --out ppl.json

Why this exists: GSM8K absorbs distributional error -- a wrong token gets
recovered over 250 tokens of reasoning -- so it cannot resolve a change that
moves the token distribution by a few multiples of the deployment's own floor.
Teacher-forced NLL absorbs nothing, and it is the only criterion here that
gets one number per token instead of one per question.

Two deliberate choices:

  * Windows default to 4096 tokens, above ``index_topk=2048``. Below that the DSA
    indexer selects everything and the sparse path is never taken, which is the
    trap the handoff doc flags for the smoke test -- it applies just as much to
    the 11-24 token prompts the logprob check uses.
  * Requests go out serially. The prefill grouping of concurrent requests is not
    reproducible across runs, and this is a logprob measurement, so the
    one thing it must not have is batch-shape noise.

Per-window NLL is written out, not just the total, so two runs can be compared
as a paired sample rather than as two scalars.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)
HTTP = requests.Session()
# Download proxies belong to setup. Inference traffic must go directly to the
# configured service even when GLM53_HOST is not the default loopback address.
HTTP.trust_env = False

# The corpus is not vendored -- it is too big and it is shared between checkouts --
# so its location comes from the environment. This used to fall back to two absolute
# paths under two different people's home directories, which made the tool silently
# unrunnable for anyone else and silently wrong for anyone who had a stale copy at
# one of them. glm53_env.sh exports GLM53_EVAL_DIR; --data overrides it.
_EVAL_DIR = os.environ.get("GLM53_EVAL_DIR") or ""
DEFAULT_DATA = Path(_EVAL_DIR) / "wikitext" / "test.parquet" if _EVAL_DIR else None
DEFAULT_MODEL = os.environ.get("GLM53_MODEL_PATH") or ""
DEFAULT_PORT = int(os.environ.get("GLM53_PORT") or 30013)
DEFAULT_HOST = os.environ.get("GLM53_HOST") or "127.0.0.1"


def url_host(host: str) -> str:
    """Return a connectable URL host for wildcard and IPv6 bind addresses."""
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in {"::", "[::]"}:
        return "[::1]"
    if host.startswith("[") and host.endswith("]"):
        return host
    return f"[{host}]" if ":" in host else host


def load_text(path: Path) -> str:
    import pyarrow.parquet as pq

    rows = pq.read_table(str(path)).to_pylist()
    return "".join(r["text"] for r in rows)


def window_nll(host: str, port: int, ids: list[int], timeout: float) -> tuple[float, int]:
    """Return (sum of -logprob, number of scored positions) for one window."""
    r = HTTP.post(
        f"http://{url_host(host)}:{port}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    entries = r.json()["meta_info"]["input_token_logprobs"]
    # The first position has no predecessor and carries a null logprob.
    lps = [e[0] for e in entries if e[0] is not None]
    return -sum(lps), len(lps)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=DEFAULT_MODEL or None)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument(
        "--window",
        type=int,
        default=4096,
        help="tokens per window; keep above index_topk=2048 so the sparse path runs",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 means every window")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if not args.model:
        ap.error(
            "no tokenizer: set GLM53_MODEL_PATH or pass --model. "
            "(Sourcing glm53_env.sh exports it.)"
        )
    if args.data is None:
        ap.error(
            "no corpus: set GLM53_EVAL_DIR to a directory holding wikitext/test.parquet, "
            "or pass --data explicitly. (Sourcing glm53_env.sh exports it.)"
        )
    if not args.data.is_file():
        ap.error(f"no corpus at {args.data}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode(load_text(args.data), add_special_tokens=False)

    n_win = len(ids) // args.window
    if args.limit:
        n_win = min(n_win, args.limit)
    LOGGER.info(f"{len(ids)} tokens -> {n_win} windows of {args.window}")

    windows = []
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    for w in range(n_win):
        chunk = ids[w * args.window:(w + 1) * args.window]
        nll, n = window_nll(args.host, args.port, chunk, args.timeout)
        windows.append({"i": w, "nll_sum": nll, "n": n, "ppl": math.exp(nll / n)})
        total_nll += nll
        total_tok += n
        if (w + 1) % 10 == 0 or w + 1 == n_win:
            LOGGER.info(
                f"  {w + 1}/{n_win}  {time.time() - t0:.0f}s  "
                f"running ppl {math.exp(total_nll / total_tok):.4f}"
            )

    ppl = math.exp(total_nll / total_tok)
    LOGGER.info(f"\nperplexity      {ppl:.4f}")
    LOGGER.info(f"mean NLL        {total_nll / total_tok:.6f} nats/token")
    LOGGER.info(f"scored          {total_tok} tokens in {n_win} windows of {args.window}")
    LOGGER.info(f"wall            {time.time() - t0:.0f}s")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "ppl": ppl,
                    "mean_nll": total_nll / total_tok,
                    "total_nll": total_nll,
                    "tokens": total_tok,
                    "window": args.window,
                    "n_windows": n_win,
                    "windows": windows,
                }
            )
        )
        LOGGER.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
