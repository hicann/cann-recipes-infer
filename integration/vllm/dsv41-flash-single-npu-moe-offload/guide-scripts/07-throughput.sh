#!/usr/bin/env bash
# Forward benchmark parameters to the source-tree entrypoint.
set -euo pipefail
: "${ASCEND_TREE:?先 source dsv41.env}"
export REQUESTS=${REQUESTS:-5}
export OUTPUT_DIR=${OUTPUT_DIR:-$DSV41_ROOT/throughput/$(date -u +%Y%m%dT%H%M%S%NZ)}
bash "$ASCEND_TREE/examples/kt_moe_offload/bench_latency.sh"
echo "=== 吞吐 完成 ==="
