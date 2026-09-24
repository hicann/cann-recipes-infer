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
# Contention-gated decode-throughput benchmark.  It starts a dedicated service,
# measures two output lengths so TTFT cancels, writes a JSON result, and stops
# only the service it started.  Benchmark controls are CLI options rather than
# model-specific environment variables.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
NAME=run
PROMPT_TOKENS=630
SHORT_TOKENS=64
LONG_TOKENS=256
WARMUP_TOKENS=256
REPEATS=4
SYNTHETIC=0
ALLOW_CONTAMINATED=0

usage() {
  cat <<'USAGE'
usage: bench.sh [options]

Start a dedicated GLM-5.3 service and measure decode throughput from the median
of paired wall-time differences for two generation lengths on an identical prompt.

  --name TAG               result label (default: run)
  --prompt-tokens N        approximate real-prose prompt length (default: 630)
  --short-tokens N         shorter generation length (default: 64)
  --long-tokens N          longer generation length (default: 256)
  --warmup-tokens N        discarded full warmup length (default: 256)
  --repeats N              even number of paired measurements (default: 4)
  --synthetic-prompt       use repetitive filler instead of WikiText; diagnostic only
  --allow-contaminated     run despite initial host contention, but mark result invalid
  -h, --help               show this help

The configured port must be free.  This script refuses to replace an existing
service.  Results are written below GLM53_LOG_DIR.
USAGE
}

_need_value() {
  [ "$#" -ge 2 ] || { echo "[bench] $1 needs a value" >&2; exit 2; }
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name) _need_value "$@"; NAME="$2"; shift 2 ;;
    --name=*) NAME="${1#*=}"; shift ;;
    --prompt-tokens) _need_value "$@"; PROMPT_TOKENS="$2"; shift 2 ;;
    --prompt-tokens=*) PROMPT_TOKENS="${1#*=}"; shift ;;
    --short-tokens) _need_value "$@"; SHORT_TOKENS="$2"; shift 2 ;;
    --short-tokens=*) SHORT_TOKENS="${1#*=}"; shift ;;
    --long-tokens) _need_value "$@"; LONG_TOKENS="$2"; shift 2 ;;
    --long-tokens=*) LONG_TOKENS="${1#*=}"; shift ;;
    --warmup-tokens) _need_value "$@"; WARMUP_TOKENS="$2"; shift 2 ;;
    --warmup-tokens=*) WARMUP_TOKENS="${1#*=}"; shift ;;
    --repeats) _need_value "$@"; REPEATS="$2"; shift 2 ;;
    --repeats=*) REPEATS="${1#*=}"; shift ;;
    --synthetic-prompt) SYNTHETIC=1; shift ;;
    --allow-contaminated) ALLOW_CONTAMINATED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[bench] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${NAME}" in ''|*[!A-Za-z0-9_.-]*) echo "[bench] --name may contain only A-Z, a-z, 0-9, '.', '_' and '-'" >&2; exit 2 ;; esac
for _pair in \
  "prompt:${PROMPT_TOKENS}" "short:${SHORT_TOKENS}" "long:${LONG_TOKENS}" \
  "warmup:${WARMUP_TOKENS}" "repeats:${REPEATS}"; do
  _label="${_pair%%:*}"; _value="${_pair#*:}"
  case "${_value}" in ''|*[!0-9]*|0) echo "[bench] ${_label} must be a positive integer" >&2; exit 2 ;; esac
done
[ "${LONG_TOKENS}" -gt "${SHORT_TOKENS}" ] || {
  echo "[bench] --long-tokens must be greater than --short-tokens" >&2; exit 2;
}
[ "${REPEATS}" -ge 2 ] && [ $((REPEATS % 2)) -eq 0 ] || {
  echo "[bench] --repeats must be an even integer of at least 2 to balance request order" >&2
  exit 2
}

# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

case "${GLM53_HOST}" in
  0.0.0.0|'') BENCH_CONNECT_HOST=127.0.0.1; BENCH_URL_HOST=127.0.0.1 ;;
  ::|'[::]') BENCH_CONNECT_HOST=::1; BENCH_URL_HOST='[::1]' ;;
  '['*']') BENCH_CONNECT_HOST="${GLM53_HOST#\[}"; BENCH_CONNECT_HOST="${BENCH_CONNECT_HOST%\]}"; BENCH_URL_HOST="${GLM53_HOST}" ;;
  *:*) BENCH_CONNECT_HOST="${GLM53_HOST}"; BENCH_URL_HOST="[${GLM53_HOST}]" ;;
  *) BENCH_CONNECT_HOST="${GLM53_HOST}"; BENCH_URL_HOST="${GLM53_HOST}" ;;
esac
case "${GLM53_PORT}" in
  ''|*[!0-9]*) echo "[bench] GLM53_PORT must be an integer" >&2; exit 2 ;;
esac
case "${GLM53_NPU_DEVICE_ID}" in
  ''|*[!0-9]*) echo "[bench] GLM53_NPU_DEVICE_ID must be a non-negative integer" >&2; exit 2 ;;
esac
[ "${GLM53_PORT}" -ge 1 ] && [ "${GLM53_PORT}" -le 65535 ] || {
  echo "[bench] GLM53_PORT must be between 1 and 65535" >&2; exit 2;
}
for _pair in "context:${GLM53_CONTEXT_LENGTH}" "running:${GLM53_MAX_RUNNING_REQUESTS}"; do
  _label="${_pair%%:*}"; _value="${_pair#*:}"
  case "${_value}" in
    ''|*[!0-9]*|0) echo "[bench] resolved ${_label} limit must be a positive integer" >&2; exit 2 ;;
  esac
done
case "${GLM53_MAX_TOTAL_TOKENS:-}" in
  '' ) ;;
  *[!0-9]*|0) echo "[bench] GLM53_MAX_TOTAL_TOKENS must be empty or a positive integer" >&2; exit 2 ;;
esac

# The prompt length is approximate (the corpus is cut by characters), so retain a
# small guard band.  An oversized ignore_eos request can exhaust the KV pool and
# terminate the NPU scheduler instead of returning a normal request error.
_max_generated="${LONG_TOKENS}"
[ "${WARMUP_TOKENS}" -le "${_max_generated}" ] || _max_generated="${WARMUP_TOKENS}"
_request_budget=$((PROMPT_TOKENS + _max_generated + 32))
_request_limit="${GLM53_CONTEXT_LENGTH}"
if [ -n "${GLM53_MAX_TOTAL_TOKENS}" ] && [ "${GLM53_MAX_TOTAL_TOKENS}" -lt "${_request_limit}" ]; then
  _request_limit="${GLM53_MAX_TOTAL_TOKENS}"
fi
[ "${_request_budget}" -le "${_request_limit}" ] || {
  echo "[bench] prompt + generation + guard (${_request_budget}) exceeds the configured request limit (${_request_limit})" >&2
  exit 2
}

mkdir -p "${GLM53_LOG_DIR}"
OUT="${GLM53_LOG_DIR}/bench_${NAME}_$(date +%Y%m%d-%H%M%S).json"
BENCH_LOG="${GLM53_LOG_DIR}/serve.log"
DIE_IDLE_MIB=6553
LOAD_MAX=8
DIE_WAIT=600
# A cold start may read more than 450 GiB of checkpoint/GGUF data and capture the
# decode graph. Keep this aligned with guide-scripts/06-serve.sh's validated limit.
STARTUP_WAIT=3600
BENCH_STARTED=0
BENCH_PID=""
BENCH_SID=""

_die_used() {
  { npu-smi info 2>&1 | grep -oP '\d+(?=\s*/ 65536)' \
      | sed -n "$((GLM53_NPU_DEVICE_ID + 1))p"; } || true
}
_load() { awk '{print $1}' /proc/loadavg; }
_float_gt() { awk -v left="$1" -v right="$2" 'BEGIN { exit !(left > right) }'; }
_port_open() {
  "${GLM53_PYTHON}" - "${BENCH_CONNECT_HOST}" "${GLM53_PORT}" <<'PY'
import socket
import sys

try:
    connection = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5)
except OSError:
    raise SystemExit(1)
else:
    connection.close()
    raise SystemExit(0)
PY
}
_other_servers() {
  ps -eo pid=,sid=,args= 2>/dev/null | awk -v own="${BENCH_SID}" '
    BEGIN {
      launch = "sglang" ".launch_server"
      scheduler = "sglang" "::scheduler"
    }
    {
      sid = $2
      $1 = ""; $2 = ""
      if ((index($0, launch) || index($0, scheduler)) && (own == "" || sid != own)) {
        count++
      }
    }
    END { print count + 0 }
  '
}
_bench_session_pids() {
  case "${BENCH_SID}" in ''|*[!0-9]*) return 0 ;; esac
  ps -eo pid=,sid= 2>/dev/null | awk -v sid="${BENCH_SID}" '$2 == sid { print $1 }'
}
_contention_report() {
  printf '  load=%s  other_sglang=%s  die%s=%s MiB\n' \
    "$(cut -d' ' -f1-3 /proc/loadavg)" "$(_other_servers)" \
    "${GLM53_NPU_DEVICE_ID}" "$(_die_used)"
}
_cleanup() {
  trap - EXIT INT TERM
  [ "${BENCH_STARTED}" = "1" ] || return 0

  # The benchmark service runs in a private session.  Stop only processes still
  # belonging to that session; never match a shared account's command line or port.
  local _pids _try
  _pids="$(_bench_session_pids)"
  if [ -n "${_pids}" ]; then
    # Intentional word splitting: _bench_session_pids emits validated numeric PIDs.
    kill ${_pids} 2>/dev/null || true
    for _try in 1 2 3 4 5 6 7 8 9 10; do
      sleep 0.5
      _pids="$(_bench_session_pids)"
      [ -n "${_pids}" ] || break
    done
    [ -z "${_pids}" ] || kill -KILL ${_pids} 2>/dev/null || true
  else
    case "${BENCH_PID}" in
      ''|*[!0-9]*) ;;
      *) kill "${BENCH_PID}" 2>/dev/null || true ;;
    esac
  fi
  wait "${BENCH_PID}" 2>/dev/null || true
}
trap _cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "== GLM-5.3 decode benchmark: ${NAME} =="
echo "  die=${GLM53_NPU_DEVICE_ID} port=${GLM53_PORT} pools=${GLM53_THREADPOOL_COUNT} cpuinfer=${GLM53_CPUINFER} resident=${GLM53_NUM_GPU_EXPERTS}"
echo "  prompt=${PROMPT_TOKENS} short=${SHORT_TOKENS} long=${LONG_TOKENS} warmup=${WARMUP_TOKENS} repeats=${REPEATS}"

if _port_open; then
  echo "[bench] port ${GLM53_PORT} already has a service; stop it explicitly or select another port" >&2
  exit 2
fi

CORPUS="${GLM53_EVAL_DIR}/wikitext/test.parquet"
if [ "${SYNTHETIC}" != "1" ] && [ ! -f "${CORPUS}" ]; then
  echo "[bench] real-prose corpus not found: ${CORPUS}" >&2
  exit 1
fi

FOREIGN_BEFORE="$(_other_servers)"
LOAD_BEFORE="$(_load)"
echo "before:"; _contention_report
INITIAL_CONTAMINATED=0
if [ "${FOREIGN_BEFORE}" -gt 0 ] || _float_gt "${LOAD_BEFORE}" "${LOAD_MAX}"; then
  INITIAL_CONTAMINATED=1
  echo "[bench] host is contended (${FOREIGN_BEFORE} other SGLang processes, load ${LOAD_BEFORE})" >&2
  [ "${ALLOW_CONTAMINATED}" = "1" ] || {
    echo "[bench] wait for an idle host; --allow-contaminated records an invalid diagnostic result" >&2
    exit 2
  }
fi

_used="$(_die_used)"
case "${_used}" in ''|*[!0-9]*) echo "[bench] could not read die memory usage from npu-smi" >&2; exit 1 ;; esac
_waited=0
while [ "${_used}" -ge "${DIE_IDLE_MIB}" ] && [ "${_waited}" -lt "${DIE_WAIT}" ]; do
  [ "${_waited}" -eq 0 ] && echo "  waiting for die ${GLM53_NPU_DEVICE_ID} to become idle"
  sleep 15
  _waited=$((_waited + 15))
  _used="$(_die_used)"
done
[ "${_used}" -lt "${DIE_IDLE_MIB}" ] || {
  echo "[bench] die ${GLM53_NPU_DEVICE_ID} still uses ${_used} MiB after ${_waited}s" >&2
  exit 2
}

command -v setsid >/dev/null 2>&1 || {
  echo "[bench] setsid is required to isolate and safely stop the benchmark service" >&2
  exit 1
}
: > "${BENCH_LOG}"
setsid "${_here}/serve.sh" --foreground > "${BENCH_LOG}" 2>&1 &
BENCH_PID=$!
# setsid makes the child's PID its new session ID. Store that expected value before
# enabling the EXIT trap so an interrupt in the following validation window still
# cleans up descendants, not only the leader.
BENCH_SID="${BENCH_PID}"
BENCH_STARTED=1
sleep 1
_actual_sid="$(ps -o sid= -p "${BENCH_PID}" 2>/dev/null | tr -d ' ' || true)"
[ "${_actual_sid}" = "${BENCH_SID}" ] || {
  echo "[bench] could not place the service in its private process session; see ${BENCH_LOG}" >&2
  exit 1
}

_waited=0
until grep -q 'fired up and ready to roll' "${BENCH_LOG}" 2>/dev/null \
      && curl -sf -m3 --noproxy '*' "http://${BENCH_URL_HOST}:${GLM53_PORT}/health" >/dev/null 2>&1; do
  kill -0 "${BENCH_PID}" 2>/dev/null || {
    echo "[bench] service exited during startup; see ${GLM53_LOG_DIR}/serve.log" >&2
    exit 1
  }
  [ "${_waited}" -lt "${STARTUP_WAIT}" ] || {
    echo "[bench] service was not ready after ${STARTUP_WAIT}s" >&2
    exit 1
  }
  sleep 15
  _waited=$((_waited + 15))
done
kill -0 "${BENCH_PID}" 2>/dev/null || {
  echo "[bench] benchmark service exited before measurement; see ${BENCH_LOG}" >&2
  exit 1
}

"${GLM53_PYTHON}" - \
  "${BENCH_URL_HOST}" "${GLM53_PORT}" "${OUT}" "${NAME}" "${CORPUS}" "${PROMPT_TOKENS}" \
  "${SHORT_TOKENS}" "${LONG_TOKENS}" "${WARMUP_TOKENS}" "${REPEATS}" \
  "${SYNTHETIC}" "${GLM53_MODEL_PATH}" "${_request_limit}" <<'PY'
import hashlib
import json
import pathlib
import statistics
import sys
import time
import urllib.request

(
    connect_host,
    port,
    out,
    name,
    corpus,
    prompt_tokens,
    short_tokens,
    long_tokens,
    warmup_tokens,
    repeats,
    synthetic,
    model_path,
    request_limit,
) = sys.argv[1:]
prompt_tokens = int(prompt_tokens)
short_tokens = int(short_tokens)
long_tokens = int(long_tokens)
warmup_tokens = int(warmup_tokens)
repeats = int(repeats)
synthetic = synthetic == "1"
request_limit = int(request_limit)
base_url = f"http://{connect_host}:{port}"

if synthetic:
    prompt = " ".join(
        f"Item {index:04d} is in bay {index % 37}."
        for index in range(max(1, prompt_tokens // 10))
    )
else:
    import pyarrow.parquet as pq

    values = [
        row["text"]
        for row in pq.read_table(
            str(pathlib.Path(corpus)), columns=["text"]
        ).to_pylist()
    ]
    prose = "\n".join(
        value for value in values if isinstance(value, str) and value.strip()
    )
    prompt = prose[: prompt_tokens * 4]
if not prompt:
    raise SystemExit("benchmark prompt is empty")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
actual_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
max_generated = max(short_tokens, long_tokens, warmup_tokens)
if actual_prompt_tokens + max_generated + 32 > request_limit:
    raise SystemExit(
        "actual prompt + generation + guard exceeds the configured request limit: "
        f"{actual_prompt_tokens} + {max_generated} + 32 > {request_limit}"
    )

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def generate(tokens):
    request = urllib.request.Request(
        base_url + "/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with opener.open(request, timeout=3600) as response:
        result = json.load(response)
    return time.perf_counter() - start, result


print(f"  full warmup: {warmup_tokens} tokens")
_, warmup_result = generate(warmup_tokens)
warmup_completion = int(
    (warmup_result.get("meta_info") or {}).get("completion_tokens") or 0
)
if warmup_completion != warmup_tokens:
    raise SystemExit(
        f"warmup requested {warmup_tokens} output tokens but returned {warmup_completion}"
    )

measurements = {short_tokens: [], long_tokens: []}
outputs = {short_tokens: [], long_tokens: []}
prompt_counts = {short_tokens: [], long_tokens: []}
pair_deltas = []
for index in range(repeats):
    # Alternate the within-pair order so a monotonic thermal/cache drift does not
    # systematically favour either arm.  Decode is estimated from each adjacent
    # pair first, then the pair deltas are reduced with a median.
    order = (
        (short_tokens, long_tokens)
        if index % 2 == 0
        else (long_tokens, short_tokens)
    )
    pair_walls = {}
    for tokens in order:
        wall, result = generate(tokens)
        metadata = result.get("meta_info") or {}
        completion = int(metadata.get("completion_tokens") or 0)
        if completion != tokens:
            raise SystemExit(
                f"requested {tokens} output tokens but the server returned {completion}"
            )
        text = result.get("text") or ""
        measurements[tokens].append(wall)
        outputs[tokens].append(text)
        prompt_counts[tokens].append(metadata.get("prompt_tokens"))
        pair_walls[tokens] = wall
        print(f"  output={tokens:4d} pair={index + 1}/{repeats}: {wall:.3f}s")
    pair_deltas.append(pair_walls[long_tokens] - pair_walls[short_tokens])

rows = []
for tokens in (short_tokens, long_tokens):
    ordered = sorted(measurements[tokens])
    texts = outputs[tokens]
    rows.append(
        {
            "requested_tokens": tokens,
            "completion_tokens": tokens,
            "prompt_tokens": prompt_counts[tokens],
            "wall_s": {
                "min": ordered[0],
                "median": statistics.median(ordered),
                "max": ordered[-1],
            },
            "output_sha256": [
                hashlib.sha256(text.encode()).hexdigest() for text in texts
            ],
            "repeats_identical": len(set(texts)) == 1,
        }
    )

short, long = rows
delta_tokens = long_tokens - short_tokens
delta_wall = statistics.median(pair_deltas)
if delta_tokens <= 0 or delta_wall <= 0:
    raise SystemExit("two-length subtraction did not produce a positive decode latency")
decode_ms = delta_wall * 1000 / delta_tokens
non_decode_samples = [
    wall - (short_tokens - 1) * decode_ms / 1000
    for wall in measurements[short_tokens]
]
non_decode_s = statistics.median(non_decode_samples)
prefix_match = all(
    long_text.startswith(outputs[short_tokens][0])
    for long_text in outputs[long_tokens]
)
output_consistent = all(row["repeats_identical"] for row in rows) and prefix_match
prompt_values = [
    value
    for values in prompt_counts.values()
    for value in values
    if value is not None
]
summary = {
    "decode_ms_per_token": decode_ms,
    "decode_tokens_per_second": 1000 / decode_ms,
    "non_decode_s_estimate": non_decode_s,
    "prompt_tokens": prompt_values[0] if prompt_values else None,
    "prompt_tokens_local": actual_prompt_tokens,
    "paired_wall_delta_s": pair_deltas,
    "output_consistent": output_consistent,
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    "long_output_head": outputs[long_tokens][0][:400],
}
payload = {
    "name": name,
    "method": "median of paired two-length wall-time differences",
    "rows": rows,
    "summary": summary,
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
print(f"  decode: {summary['decode_tokens_per_second']:.2f} token/s ({decode_ms:.2f} ms/token)")
print(f"  non-decode estimate: {non_decode_s:.2f}s for {summary['prompt_tokens']} prompt tokens")
print(f"  deterministic prefix check: {'pass' if output_consistent else 'FAIL'}")
PY

FOREIGN_AFTER="$(_other_servers)"
LOAD_AFTER="$(_load)"
INLINE_COUNT="$({ awk '/inline resident/ { count++ } END { print count + 0 }' "${BENCH_LOG}"; } 2>/dev/null || true)"
FALLBACK_COUNT="$({ awk '/streaming failed|hybrid fallback/ { count++ } END { print count + 0 }' "${BENCH_LOG}"; } 2>/dev/null || true)"
INLINE_COUNT="${INLINE_COUNT:-0}"
FALLBACK_COUNT="${FALLBACK_COUNT:-0}"
AVAIL_BEGIN="$(sed -n 's/.*Load weight begin\. avail mem=\([0-9][0-9.]*\).*/\1/p' "${BENCH_LOG}" | tail -1 || true)"
VERDICT=clean
[ "${INITIAL_CONTAMINATED}" = "0" ] || VERDICT=contaminated
[ "${FOREIGN_AFTER}" -eq 0 ] || VERDICT=contaminated
if [ -n "${AVAIL_BEGIN}" ] && _float_gt 58 "${AVAIL_BEGIN}"; then
  VERDICT="contaminated+npu_collision"
elif [ -z "${AVAIL_BEGIN}" ]; then
  VERDICT="${VERDICT}+weight_load_evidence_missing"
fi
if [ "${GLM53_PREFILL_STREAM:-0}" = "1" ]; then
  [ "${INLINE_COUNT}" -gt 0 ] || VERDICT="${VERDICT}+stream_never_engaged"
  [ "${FALLBACK_COUNT}" -eq 0 ] || VERDICT="${VERDICT}+stream_fallback"
fi
OUTPUT_CONSISTENT="$("${GLM53_PYTHON}" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["summary"]["output_consistent"]))' "${OUT}")"
[ "${OUTPUT_CONSISTENT}" = "1" ] || VERDICT="${VERDICT}+output_mismatch"
kill -0 "${BENCH_PID}" 2>/dev/null || VERDICT="${VERDICT}+service_exited"
KT_COMMIT="$(git -C "${KTRANSFORMERS_REPO}" rev-parse HEAD 2>/dev/null || true)"
SGLANG_COMMIT="$(git -C "${SGLANG_REPO}" rev-parse HEAD 2>/dev/null || true)"

"${GLM53_PYTHON}" - \
  "${OUT}" "${VERDICT}" "${LOAD_BEFORE}" "${LOAD_AFTER}" \
  "${FOREIGN_BEFORE}" "${FOREIGN_AFTER}" "${INLINE_COUNT}" "${FALLBACK_COUNT}" \
  "${GLM53_THREADPOOL_COUNT}" "${GLM53_CPUINFER}" "${GLM53_NUM_GPU_EXPERTS}" \
  "${BENCH_PID}" "${PROMPT_TOKENS}" "${SHORT_TOKENS}" "${LONG_TOKENS}" \
  "${WARMUP_TOKENS}" "${REPEATS}" "${SYNTHETIC}" "${INITIAL_CONTAMINATED}" \
  "${BENCH_SID}" "${AVAIL_BEGIN}" "${KT_COMMIT}" "${SGLANG_COMMIT}" \
  "${CANN_VERSION:-}" "${CANN_COMPILER_VERSION:-}" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["verdict"] = sys.argv[2]
payload["contention"] = {
    "initially_contaminated": bool(int(sys.argv[19])),
    "load_before": float(sys.argv[3]),
    "load_after": float(sys.argv[4]),
    "other_servers_before": int(sys.argv[5]),
    "other_servers_after": int(sys.argv[6]),
    "npu_available_at_weight_load_gb": float(sys.argv[21]) if sys.argv[21] else None,
}
payload["streaming"] = {
    "inline_resident": int(sys.argv[7]),
    "fallbacks": int(sys.argv[8]),
}
payload["config"] = {
    "threadpool_count": int(sys.argv[9]),
    "cpuinfer": int(sys.argv[10]),
    "resident_experts": int(sys.argv[11]),
}
payload["benchmark_config"] = {
    "prompt_tokens_requested": int(sys.argv[13]),
    "short_tokens": int(sys.argv[14]),
    "long_tokens": int(sys.argv[15]),
    "warmup_tokens": int(sys.argv[16]),
    "repeats": int(sys.argv[17]),
    "synthetic_prompt": bool(int(sys.argv[18])),
}
service_env = {}
try:
    environment = pathlib.Path(f"/proc/{sys.argv[12]}/environ").read_bytes()
except OSError as error:
    payload["service_environment_error"] = str(error)
    environment = b""
for entry in environment.split(b"\0"):
    if b"=" not in entry:
        continue
    key, value = entry.split(b"=", 1)
    service_env[key.decode(errors="replace")] = value.decode(errors="replace")
keys = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "GLM53_MODEL_PATH",
    "GLM53_CONTEXT_LENGTH",
    "GLM53_MEM_FRACTION",
    "GLM53_PREFILL_STREAM",
    "GLM53_EAGER",
    "GLM53_KT_NUMA_NODES",
    "GLM53_PIN_CORES",
    "GLM53_MAX_RUNNING_REQUESTS",
    "GLM53_MAX_TOTAL_TOKENS",
    "KT_DYNAMIC_RESIDENT",
    "KT_HOT_TAIL_TOKENS",
    "KT_PREFILL_STREAM",
    "KT_PREFILL_STREAM_THRESHOLD",
    "KT_SIDE_STREAM",
    "KT_MXFP4_NZ_CHUNK",
    "KT_MXFP4_GGUF_DEDUP",
    "SGLANG_MAMBA_CONV_DTYPE",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
    "SGLANG_OPT_FP8_WO_A_GEMM",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK",
    "SGLANG_CACHE_DIR",
    "SGLANG_REPO",
)
payload["service_environment"] = {
    key: service_env[key] for key in keys if key in service_env
}
model_path = service_env.get("GLM53_MODEL_PATH")
model_config = pathlib.Path(model_path, "config.json") if model_path else None
if model_config and model_config.is_file():
    payload["model_config_sha256"] = hashlib.sha256(model_config.read_bytes()).hexdigest()

versions = {}
for distribution in (
    "torch",
    "torch-npu",
    "triton-ascend",
    "decorator",
    "scipy",
    "kt-kernel",
    "sgl-kernel-npu",
):
    try:
        versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[distribution] = None
payload["software"] = {
    "versions": versions,
    "ktransformers_commit": sys.argv[22] or None,
    "sglang_commit": sys.argv[23] or None,
    "cann_package_version": sys.argv[24] or None,
    "cann_compiler_version": sys.argv[25] or None,
    "service_pid": int(sys.argv[12]),
    "service_session_id": int(sys.argv[20]),
}
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY

echo "after:"; _contention_report
echo "  weight-load available HBM: ${AVAIL_BEGIN:-unknown} GB"
echo "  streaming: inline_resident=${INLINE_COUNT}, fallbacks=${FALLBACK_COUNT}"
echo "  verdict: ${VERDICT}"
echo "  result: ${OUT}"
[ "${VERDICT}" = "clean" ]
