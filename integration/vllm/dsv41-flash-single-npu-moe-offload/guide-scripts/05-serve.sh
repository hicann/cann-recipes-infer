#!/usr/bin/env bash
# Start or stop the single-card service. Actions: start (default) | stop
set -euo pipefail
N=拉起
: "${ASCEND_TREE:?先 source dsv41.env}"
ACTION=${1:-start}
LOG=$DSV41_ROOT/serve.log

# Session id of this deployment's vllm serve: match --port $PORT in the cmdline, then read the
# session id. start uses setsid, so APIServer, EngineCore and DPCoordinator share one session.
serve_sid() {
  local p cl
  for p in $(pgrep -f "[v]llm serve" 2>/dev/null || true); do
    cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    case "$cl" in *"--port $PORT "*) ps -o sid= -p "$p" 2>/dev/null | tr -d ' '; return 0;; esac
  done
  return 1
}
# Live processes in that session, zombies excluded.
serve_pids() { ps -eo pid=,sid=,stat= 2>/dev/null | awk -v s="$1" '$2==s && $3 !~ /^Z/ {print $1}'; }

if [ "$ACTION" = stop ]; then
  SID=$(serve_sid || true)
  if [ -z "$SID" ]; then
    echo "[$N] PASS 端口 $PORT 上没有本部署的 vllm serve 进程"; echo "=== $N 完成 ==="; exit 0
  fi
  for p in $(serve_pids "$SID"); do kill -TERM "$p" 2>/dev/null || true; done
  for _ in $(seq 1 40); do sleep 5; [ -z "$(serve_pids "$SID")" ] && break; done
  ALIVE=$(serve_pids "$SID" | tr '\n' ' ')
  [ -z "$ALIVE" ] || { echo "[$N] FAIL SIGTERM 后 200 秒仍存活，PID: $ALIVE"; exit 1; }
  echo "[$N] PASS 服务已停（会话 $SID）"
  echo "=== $N 完成 ==="; exit 0
fi

if curl -s --max-time 5 --noproxy '*' "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[$N] FAIL 端口 $PORT 上已有服务在应答。换 --port 重跑第 1 步，或先 bash -l 05-serve.sh stop"; exit 1
fi

profile=${PROFILE:-lowlatency}
if [[ -z ${KT_CONFIG:-} && ( $profile == lowlatency || $profile == throughput ) ]]; then
  KT_CONFIG=$(python3 - "$profile" <<'PY'
import json
import os
from pathlib import Path
import sys

allowed_cpus = os.sched_getaffinity(0)
status = Path("/proc/self/status").read_text().splitlines()
node_ranges = next(line.split(":", 1)[1].strip() for line in status if line.startswith("Mems_allowed_list:"))
allowed_nodes = set()
for interval in node_ranges.split(","):
    first, _, last = interval.partition("-")
    allowed_nodes.update(range(int(first), int(last or first) + 1))

cores_by_node = {}
for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
    node_id = int(node.name[4:])
    if node_id not in allowed_nodes:
        continue
    cores = {
        (cpu / "topology/thread_siblings_list").read_text().strip()
        for cpu in node.glob("cpu[0-9]*")
        if int(cpu.name[3:]) in allowed_cpus
    }
    if cores:
        cores_by_node[node_id] = len(cores)
if not cores_by_node:
    sys.exit("[拉起] FAIL 找不到同时允许 CPU 和内存的 NUMA 节点")

ranked = sorted(cores_by_node, key=lambda node: (-cores_by_node[node], node))[:2]
nodes = sorted(node for node in ranked if cores_by_node[node] > 1) or ranked[:1]
cores_per_pool = min(cores_by_node[node] for node in nodes)
threads_per_pool = max(1, min(48, cores_per_pool * 3 // 4))
lowlatency = sys.argv[1] == "lowlatency"
print(json.dumps({"kt_offload_config": {
    "enabled": True,
    "resident_strategy": "prefix",
    "num_resident_experts": int(os.environ.get("RESIDENT") or 100),
    "cpuinfer_threads": len(nodes) * threads_per_pool,
    "threadpool_count": len(nodes),
    "numa_nodes": nodes,
    "packed_transfer": False,
    "stream_prefill": lowlatency,
    "dynamic_resident": lowlatency,
    "zerocopy_w13": True,
    "zerocopy_node_fraction": [1.0] * len(nodes),
    "zerocopy_streaming_node": nodes[-1],
}}))
PY
)
  export KT_CONFIG
fi

: > "$LOG"
cd "$ASCEND_TREE/examples/kt_moe_offload"
nohup setsid env VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-5}" \
  bash "serve_$profile.sh" > "$LOG" 2>&1 &
T0=$(date +%s)
for _ in $(seq 1 150); do
  sleep 10
  if curl -s --max-time 5 --noproxy '*' "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    SID=$(serve_sid || true)
    [ -n "$SID" ] || { echo "[$N] FAIL 端口 $PORT 有应答，但找不到本部署的 vllm serve 进程"; exit 1; }
    echo "[$N] PASS 服务已就绪，用时 $(( $(date +%s) - T0 )) 秒，会话 $SID"
    grep -m1 -E 'ok: cann_ops_transformer' "$LOG" | sed 's/^/[拉起] /' || true
    grep -m1 -E 'GPU KV cache size' "$LOG" | sed 's/.*INFO/[拉起] INFO/' || true
    echo "=== $N 完成 ==="; exit 0
  fi
  if grep -qE 'Traceback|core dumped' "$LOG"; then
    echo "[$N] FAIL 启动失败，见 $LOG"; grep -nE 'Traceback' "$LOG" | head -3; exit 1
  fi
done
echo "[$N] FAIL 超时，见 $LOG"; exit 1
