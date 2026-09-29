#!/usr/bin/env bash
# Single-request latency and throughput.
set -euo pipefail
N=吞吐
: "${ASCEND_TREE:?先 source dsv41.env}"
cd "$ASCEND_TREE/examples/kt_moe_offload"
PORT=$PORT \
CONCURRENCY=${CONCURRENCY:-1} REQUESTS=${REQUESTS:-5} \
PROMPT_TOKENS=${PROMPT_TOKENS:-4096} GEN_TOKENS=${GEN_TOKENS:-128} \
  bash bench_latency.sh
echo "=== $N 完成 ==="
