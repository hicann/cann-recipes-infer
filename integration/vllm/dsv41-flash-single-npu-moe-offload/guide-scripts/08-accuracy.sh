#!/usr/bin/env bash
# Accuracy evaluation. Actions: setup (default) | gsm8k | gpqa
set -euo pipefail
N=精度
: "${DSV41_ROOT:?先 source dsv41.env}"
: "${PORT:?先 source dsv41.env}"
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$S/../accuracy" && pwd)
A=${DSV41_ACC_DIR:-$DSV41_ROOT/accuracy}
AIS_SHA=${AIS_SHA:-52c2ae90629127ecca30b96b446526fc8378f091}
OSS=${AIS_DATASET_URL:-http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data}
ACTION=${1:-setup}

# Fetch AISBench with a reachability probe in front, then retry.
ais_retry() {
  local i j
  for i in $(seq 1 "${AIS_FETCH_TRIES:-8}"); do
    for j in $(seq 1 "${AIS_PROBE_TRIES:-60}"); do
      curl -s --connect-timeout 8 --max-time 15 -o /dev/null https://github.com/ 2>/dev/null && break
      sleep 5
    done
    timeout "${AIS_FETCH_TIMEOUT:-900}" "$@" && return 0
    echo "[$N] WARN 取 AISBench 第 $i 次失败，10 秒后重探"
    sleep 10
  done
  echo "[$N] FAIL 取 AISBench 连续 ${AIS_FETCH_TRIES:-8} 次失败"
  return 1
}

if [ "$ACTION" = setup ]; then
  mkdir -p "$A"
  [ -d "$A/benchmark/.git" ] || git init -q "$A/benchmark"
  git -C "$A/benchmark" remote get-url origin >/dev/null 2>&1 \
    || git -C "$A/benchmark" remote add origin "${AIS_REPO:-https://github.com/AISBench/benchmark.git}"
  ais_retry git -C "$A/benchmark" fetch -q --depth 1 origin "$AIS_SHA" || exit 1
  git -C "$A/benchmark" checkout -q FETCH_HEAD
  echo "[$N] PASS AISBench $(git -C "$A/benchmark" rev-parse HEAD)"
  [ -d "$A/venv" ] || python3 -m venv "$A/venv"
  PIP_OPTS=(--timeout "${PIP_TIMEOUT:-30}" --retries "${PIP_RETRIES:-10}")
  "$A/venv/bin/pip" install -q "${PIP_OPTS[@]}" -U pip
  (cd "$A/benchmark" && "$A/venv/bin/pip" install -q "${PIP_OPTS[@]}" -e . -r requirements/api.txt -r requirements/extra.txt)
  "$A/venv/bin/ais_bench" --help >/dev/null
  echo "[$N] PASS ais_bench 可执行"
  D=$A/benchmark/ais_bench/datasets
  mkdir -p "$D"
  for z in gsm8k gpqa; do
    [ -d "$D/$z" ] && continue
    curl -fsSL --noproxy '*' --connect-timeout 15 --retry 5 --retry-delay 5 -o "$D/$z.zip" "$OSS/$z.zip"
    python3 -m zipfile -e "$D/$z.zip" "$D" && rm -f "$D/$z.zip"
  done
  ck() { if [ -s "$2" ]; then echo "[$N] PASS $1"; else echo "[$N] FAIL 缺 $2"; exit 1; fi; }
  ck "gsm8k 语料"        "$D/gsm8k/test.jsonl"
  ck "gpqa_diamond 语料" "$D/gpqa/gpqa_diamond.csv"
  echo "=== $N 完成 ==="; exit 0
fi

case "$ACTION" in
  gsm8k) CFG=gsm8k.py;         DS=gsm8k;        MIN=5120;;
  gpqa)  CFG=gpqa_diamond.py;  DS=gpqa_diamond; MIN=9216;;
  *) echo "[$N] FAIL 未知动作 $ACTION，可选 setup|gsm8k|gpqa"; exit 1;;
esac
[ -x "$A/venv/bin/ais_bench" ] || { echo "[$N] FAIL 先 bash -l 08-accuracy.sh setup"; exit 1; }
M=$(curl -s --noproxy '*' --max-time 20 "http://127.0.0.1:$PORT/v1/models")
L=$(echo "$M" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["max_model_len"])')
[ "$L" -ge "$MIN" ] || { echo "[$N] FAIL max_model_len $L，$DS 需要 >= $MIN"; exit 1; }
echo "[$N] PASS max_model_len $L >= $MIN"
OUT=$A/outputs/$DS
mkdir -p "$OUT"
cd "$A/benchmark"
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONPATH="$SRC" AISBENCH_ROOT="$A/benchmark" \
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  HOST=127.0.0.1 PORT="$PORT" MODEL_NAME=dsv41 \
  CONCURRENCY="${CONCURRENCY:-8}" THINKING="${THINKING:-0}" \
  ${NUM_SAMPLES:+NUM_SAMPLES=$NUM_SAMPLES} \
  ${AISBENCH_REQUEST_TIMEOUT:+AISBENCH_REQUEST_TIMEOUT=$AISBENCH_REQUEST_TIMEOUT} \
  "$A/venv/bin/ais_bench" "$SRC/configs/$CFG" --mode "${MODE:-all}" -w "$OUT" --dump-eval-details
echo "=== $N 完成 ==="
