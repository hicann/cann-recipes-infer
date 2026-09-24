#!/usr/bin/env bash
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
#
# Send one request to an already-running server and report decode throughput
# from the first streamed output token onward.  TTFT is reported separately.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NEW_TOKENS=256
PROMPT_TOKENS=1024
PRINT_FULL=0

usage() {
  cat <<'USAGE'
usage: ask.sh [options]

Send one request to the configured running service and measure client-side
decode throughput.  The prompt is real prose from wikitext/test.parquet.

  --new-tokens N      generated tokens; EOS is ignored (default: 256)
  --prompt-tokens N   approximate prompt length (default: 1024)
  --full              print the complete prompt instead of head and tail
  -h, --help          show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --new-tokens)
      [ "$#" -ge 2 ] || { echo "[ask] --new-tokens needs a value" >&2; exit 2; }
      NEW_TOKENS="$2"; shift 2 ;;
    --new-tokens=*) NEW_TOKENS="${1#*=}"; shift ;;
    --prompt-tokens)
      [ "$#" -ge 2 ] || { echo "[ask] --prompt-tokens needs a value" >&2; exit 2; }
      PROMPT_TOKENS="$2"; shift 2 ;;
    --prompt-tokens=*) PROMPT_TOKENS="${1#*=}"; shift ;;
    --full) PRINT_FULL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ask] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${NEW_TOKENS}" in ''|*[!0-9]*|0) echo "[ask] --new-tokens must be positive" >&2; exit 2 ;; esac
[ "${NEW_TOKENS}" -ge 2 ] || { echo "[ask] --new-tokens must be at least 2 to measure decode" >&2; exit 2; }
case "${PROMPT_TOKENS}" in ''|*[!0-9]*|0) echo "[ask] --prompt-tokens must be positive" >&2; exit 2 ;; esac

# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

case "${GLM53_HOST}" in
  0.0.0.0|'') ASK_URL_HOST=127.0.0.1 ;;
  ::|'[::]') ASK_URL_HOST='[::1]' ;;
  '['*']') ASK_URL_HOST="${GLM53_HOST}" ;;
  *:*) ASK_URL_HOST="[${GLM53_HOST}]" ;;
  *) ASK_URL_HOST="${GLM53_HOST}" ;;
esac
case "${GLM53_PORT}" in
  ''|*[!0-9]*) echo "[ask] GLM53_PORT must be an integer" >&2; exit 2 ;;
esac
[ "${GLM53_PORT}" -ge 1 ] && [ "${GLM53_PORT}" -le 65535 ] || {
  echo "[ask] GLM53_PORT must be between 1 and 65535" >&2; exit 2;
}
for _pair in "context:${GLM53_CONTEXT_LENGTH}" "running:${GLM53_MAX_RUNNING_REQUESTS}"; do
  _label="${_pair%%:*}"; _value="${_pair#*:}"
  case "${_value}" in
    ''|*[!0-9]*|0) echo "[ask] resolved ${_label} limit must be a positive integer" >&2; exit 2 ;;
  esac
done
case "${GLM53_MAX_TOTAL_TOKENS:-}" in
  '') ;;
  *[!0-9]*|0) echo "[ask] GLM53_MAX_TOTAL_TOKENS must be empty or a positive integer" >&2; exit 2 ;;
esac
_request_limit="${GLM53_CONTEXT_LENGTH}"
if [ -n "${GLM53_MAX_TOTAL_TOKENS}" ] && [ "${GLM53_MAX_TOTAL_TOKENS}" -lt "${_request_limit}" ]; then
  _request_limit="${GLM53_MAX_TOTAL_TOKENS}"
fi
_request_budget=$((PROMPT_TOKENS + NEW_TOKENS + 32))
[ "${_request_budget}" -le "${_request_limit}" ] || {
  echo "[ask] prompt + generation + guard (${_request_budget}) exceeds the configured request limit (${_request_limit})" >&2
  exit 2
}

CORPUS="${GLM53_EVAL_DIR}/wikitext/test.parquet"
[ -f "${CORPUS}" ] || {
  echo "[ask] evaluation corpus not found: ${CORPUS}" >&2
  echo "[ask] set GLM53_EVAL_DIR to the directory containing wikitext/test.parquet" >&2
  exit 1
}

"${GLM53_PYTHON}" - \
  "${ASK_URL_HOST}" "${GLM53_PORT}" "${NEW_TOKENS}" "${PROMPT_TOKENS}" \
  "${CORPUS}" "${PRINT_FULL}" "${GLM53_MODEL_PATH}" "${_request_limit}" <<'PY'
import json
import pathlib
import sys
import time
import urllib.request

(
    host,
    port,
    new_tokens,
    prompt_tokens,
    corpus,
    print_full,
    model_path,
    request_limit,
) = sys.argv[1:]
new_tokens = int(new_tokens)
prompt_tokens = int(prompt_tokens)
print_full = print_full == "1"
request_limit = int(request_limit)

import pyarrow.parquet as pq

texts = [
    row["text"]
    for row in pq.read_table(str(pathlib.Path(corpus)), columns=["text"]).to_pylist()
    if isinstance(row.get("text"), str) and row["text"].strip()
]
prompt = "\n".join(texts)[: prompt_tokens * 4]
if not prompt:
    raise SystemExit(f"no usable text in {corpus}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
actual_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
if actual_prompt_tokens + new_tokens + 32 > request_limit:
    raise SystemExit(
        "actual prompt + generation + guard exceeds the configured request limit: "
        f"{actual_prompt_tokens} + {new_tokens} + 32 > {request_limit}"
    )

print(
    f"[ask] prompt={len(prompt)} chars ({actual_prompt_tokens} local tokenizer tokens), "
    f"new_tokens={new_tokens}"
)
print("\n================= PROMPT =================")
if print_full or len(prompt) <= 1200:
    print(prompt)
else:
    print(prompt[:700])
    print(f"\n[... {len(prompt) - 1000} chars omitted; pass --full to print all ...]\n")
    print(prompt[-300:])
print("==========================================\n")

body = json.dumps(
    {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
    }
).encode()
request = urllib.request.Request(
    f"http://{host}:{port}/generate",
    data=body,
    headers={"Content-Type": "application/json"},
)

# Local inference must never be routed through a download proxy, even when the
# caller configured one for setup.sh.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
t0 = time.perf_counter()
first = last = None
first_completed = last_completed = None
last_text = ""
meta = {}
events = 0
saw_done = False
with opener.open(request, timeout=3600) as response:
    for raw in response:
        raw = raw.strip()
        if not raw:
            continue
        payload = raw[5:].strip() if raw.startswith(b"data:") else raw
        if payload == b"[DONE]":
            saw_done = True
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            raise SystemExit(f"[ask] server returned an error: {data['error']}")
        current_text = data.get("text")
        current_meta = data.get("meta_info") or {}
        if current_meta:
            meta = current_meta
        if current_text is not None:
            last_text = current_text
        completed = int(current_meta.get("completion_tokens") or events + 1)
        if last_completed is not None and completed <= last_completed:
            continue
        now = time.perf_counter()
        if first is None:
            first = now
            first_completed = completed
        last = now
        last_completed = completed
        events += 1
        if completed > first_completed and completed % 40 == 0:
            rate = (completed - first_completed) / (last - first)
            print(f"[ask] {completed:4d} tokens: {rate:6.2f} token/s")

completion_tokens = int(meta.get("completion_tokens") or last_completed or 0)
print("\n================= OUTPUT =================")
print(last_text or "(empty)")
print("==========================================")
if not saw_done:
    raise SystemExit("[ask] stream ended without [DONE]")
if completion_tokens != new_tokens:
    raise SystemExit(
        f"[ask] requested {new_tokens} output tokens but received {completion_tokens}"
    )
if first is None or last is None or first_completed is None or last_completed is None \
        or last_completed <= first_completed or last <= first:
    raise SystemExit("[ask] too few streamed output tokens to measure")

ttft = first - t0
decode_span = last - first
decode_rate = (last_completed - first_completed) / decode_span
print(f"\nprompt tokens : {meta.get('prompt_tokens')}")
print(f"completion    : {completion_tokens}")
print(f"TTFT          : {ttft:.2f} s")
print(f"decode        : {decode_rate:.2f} token/s ({1000 / decode_rate:.1f} ms/token)")
print("The decode figure excludes TTFT; do not use total wall time / output tokens.")
PY
