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
# GPQA-diamond accuracy regression: run the same configuration N times, score each round and keep
# a running mean, then report mean/min/max/sd.
#
# Why repeat: at temp=1 the sampling noise of a single run is large (binomial SE over 198 questions
# is about +-3.3pp), so a single result does not support a conclusion. Measured on one 910B card,
# 3 rounds gave R1/R2/R3 = 69.19/72.73/73.23, mean 71.72% / SD 1.80pp. Chasing a "regression"
# smaller than ~5pp requires averaging about 10 rounds first, otherwise you are chasing noise.
# See the accuracy-evaluation section of the guide (dsv4_flash_single_card_inference_guide.md 9.2).
#
# Prerequisite: the server is up (this script waits for /health=200).
#
# Usage:
#   bash tools/gpqa_accuracy_repeat.sh                       # 10 rounds, port 8020
#   REPEATS=3 PORT=8020 bash tools/gpqa_accuracy_repeat.sh   # only 3 rounds
#   OUT_PREFIX=eval_gpqa_myrun bash tools/gpqa_accuracy_repeat.sh
#
# env (all overridable):
#   REPEATS(10) / PORT(8020) / HOST(127.0.0.1)
#   MODEL_PATH  model under test (must match the server)
#   OUT_DIR     output root (defaults to the repo root)
#   OUT_PREFIX  per-round output directory prefix (default eval_gpqa_R)
#   PY / EVALSCOPE  interpreter and evalscope executable (found on PATH by default)
#
# Failure semantics: a round whose evalscope run fails or whose report is missing logs an ERROR,
# is recorded as '?' and is excluded from the mean; the summary reports how many rounds failed and
# the script exits non-zero -- a "looks fine" score must not hide rounds that never completed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
: "${REPEATS:=10}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8020}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[gpqa][ERROR] MODEL_PATH is not set (W8A8 weight directory, must match the server):" >&2
  echo "[gpqa][ERROR]   MODEL_PATH=/your/path/DeepSeek-V4-Flash-W8A8 bash $0" >&2
  exit 2
fi
: "${OUT_DIR:=$REPO_ROOT}"
: "${OUT_PREFIX:=eval_gpqa_R}"
: "${PY:=$(command -v python3 || echo python3)}"
: "${EVALSCOPE:=$(command -v evalscope || echo evalscope)}"

API="http://${HOST}:${PORT}/v1/chat/completions"
# temp=1 is the standard GPQA-off setting; thinking/high_effort=false selects the non-thinking mode
GEN='{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}'
export no_proxy="${HOST},localhost" NO_PROXY="${HOST},localhost"

command -v "$EVALSCOPE" >/dev/null 2>&1 || { echo "[gpqa] evalscope not found: pip install evalscope, or set EVALSCOPE=" >&2; exit 1; }

echo "[gpqa] Waiting for ${API%/v1*}/health (up to 20 minutes)..."
READY=0
for _ in $(seq 1 240); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" = "200" ]; then
    READY=1
    echo "[gpqa] Server is ready"
    break
  fi
  sleep 5
done
# The timeout must stop the run: otherwise evalscope would go ahead against an unreachable API and
# produce empty results / noise scores.
if [ "$READY" != "1" ]; then
  echo "[gpqa][ERROR] http://${HOST}:${PORT}/health did not return 200 within 20 minutes; not evaluating." >&2
  exit 1
fi

declare -a SCORES
FAILED=0
for i in $(seq 1 "$REPEATS"); do
  WD="$OUT_DIR/${OUT_PREFIX}${i}"
  rm -rf "$WD"
  echo "===================== R$i START $(date +%H:%M:%S) ====================="
  S='?'
  if "$EVALSCOPE" eval --model "$MODEL_PATH" \
      --api-url "$API" --api-key EMPTY \
      --eval-type openai_api --datasets gpqa_diamond \
      --generation-config "$GEN" --eval-batch-size 1 --repeats 1 \
      --work-dir "$WD" 2>&1 | tee "$WD.log"; then
    # `|| true`: evalscope may not have created $WD at all, and under set -e the non-zero exit of
    # find would take the whole script down -- including the ERROR below and the final summary.
    # What is wanted here is "not found", not "abort".
    RPT="$(find "$WD" -name gpqa_diamond.json -path '*reports*' 2>/dev/null | tail -1 || true)"
    if [ -n "$RPT" ] && [ -f "$RPT" ]; then
      S="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('score','?'))" "$RPT")" \
        || { echo "[gpqa][ERROR] R$i could not parse the report: $RPT" >&2; S='?'; }
    else
      echo "[gpqa][ERROR] R$i found no report (*/reports/**/gpqa_diamond.json); recording the round as failed." >&2
    fi
  else
    echo "[gpqa][ERROR] R$i evalscope eval exited non-zero (see $WD.log); recording the round as failed." >&2
  fi
  if [ "$S" = '?' ]; then
    FAILED=$((FAILED + 1))
  fi
  SCORES[i]="$S"
  echo "===================== R$i DONE  $(date +%H:%M:%S)  score=$S ====================="
  "$PY" - "${SCORES[@]}" <<'PYEOF'
import sys
xs=[float(x) for x in sys.argv[1:] if x not in ('','?')]
if xs: print("  >> running (n=%d): mean=%.4f min=%.4f max=%.4f" % (len(xs), sum(xs)/len(xs), min(xs), max(xs)))
PYEOF
done

echo "======================== SUMMARY ========================"
for i in $(seq 1 "$REPEATS"); do echo "R$i = ${SCORES[i]:-?}"; done
"$PY" - "${SCORES[@]}" <<'PYEOF'
import sys, statistics as st
xs=[float(x) for x in sys.argv[1:] if x not in ('','?')]
if xs:
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    print("n=%d  mean=%.4f  min=%.4f  max=%.4f  sd=%.4f" % (len(xs), sum(xs)/len(xs), min(xs), max(xs), sd))
    print("(reference: single 910B card, 3 rounds, mean 71.72%% / SD 1.80pp; "
          "single-run SE is +-3.3pp, so do not conclude from one round)")
PYEOF

if [ "$FAILED" -gt 0 ]; then
  echo "[gpqa][ERROR] $FAILED of $REPEATS rounds failed (see the ERRORs above and the .log files);" >&2
  echo "[gpqa][ERROR] the mean covers only the successful rounds and is not comparable to a full run." >&2
  exit 1
fi
