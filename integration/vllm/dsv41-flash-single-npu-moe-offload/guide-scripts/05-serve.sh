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

: > "$LOG"
cd "$ASCEND_TREE/examples/kt_moe_offload"
nohup setsid env VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-5}" \
  bash "serve_${PROFILE:-lowlatency}.sh" > "$LOG" 2>&1 &
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
