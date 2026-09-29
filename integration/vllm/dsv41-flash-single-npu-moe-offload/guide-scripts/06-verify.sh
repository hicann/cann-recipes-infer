#!/usr/bin/env bash
# Server-side functional and environment self-check.
set -euo pipefail
N=验收
: "${PORT:?先 source dsv41.env}"
: "${KT_SITE:?先 source dsv41.env}"
P=0; F=0
ck() { if [ "$2" = "$3" ]; then echo "[$N] PASS $1 = $2"; P=$((P+1));
       else echo "[$N] FAIL $1 = $2，期望 $3"; F=$((F+1)); fi; }

M=$(curl -s --noproxy '*' --max-time 20 "http://127.0.0.1:$PORT/v1/models")
ck "model id"        "$(echo "$M" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')" dsv41
ck "max_model_len"   "$(echo "$M" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["max_model_len"])')" 8192

A=$(curl -s --noproxy '*' --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":64,"temperature":0}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
ck "17*23 含 391"    "$(case "$A" in *391*) echo yes;; *) echo no;; esac)" yes

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

SID=$(serve_sid || true)
[ -n "$SID" ] || { echo "[$N] FAIL 端口 $PORT 上找不到本部署的 vllm serve 进程"; exit 1; }
PID=$(ps -eo pid=,sid=,args= 2>/dev/null | awk -v s="$SID" '$2==s && index($0,"VLLM::EngineCore") {print $1; exit}')
[ -n "$PID" ] || { echo "[$N] FAIL 会话 $SID 里没有 VLLM::EngineCore 进程"; exit 1; }
echo "[$N] INFO EngineCore PID $PID（会话 $SID，端口 $PORT）"
ck "cann-9.1.0 映射" "$(grep -c cann-9.1.0 "/proc/$PID/maps" || true)" 0
ck "镜像内包映射"     "$(grep -cE "/usr/local/Ascend/(cann-9.2|ds41_site)" "/proc/$PID/maps" || true)" 589
ck "engram 只读映射"  "$(grep -c "$DSV41_STAGE_DIR" "/proc/$PID/maps" || true)" 4
ck "engram 映射模式"  "$(grep "$DSV41_STAGE_DIR" "/proc/$PID/maps" | awk '{print $2}' | sort -u | tr -d '\n')" "r--s"
ck "kt_kernel 解析"   "$(grep -qE "^[^ ]+ .* $KT_SITE/kt_kernel/" "/proc/$PID/maps" && echo yes || echo no)" yes

L=$DSV41_ROOT/serve.log
ck "算子解析"        "$(grep -c "ok: cann_ops_transformer -> $DS41_SITE" "$L" || true)" 1
ck "engram 未分配"    "$(grep -c 'Engram table not allocated' "$L" || true)" 2
grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens, Maximum concurrency[^$]*' "$L" | sed "s/^/[$N] INFO /"
grep -m1 -oE 'FINAL_of_all_resident=\{[^}]*\} on_target=[0-9.]*%' "$L" | sed "s/^/[$N] INFO numa /" || true
echo "[$N] 自查 $((P+F)) 项，PASS $P，FAIL $F"
[ "$F" = 0 ] || exit 1
echo "=== $N 完成 ==="
