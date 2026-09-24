#!/usr/bin/env bash
# Run verify.sh, parse its output, and enforce three checks omitted from ALL CHECKS PASSED.
set -euo pipefail

# ==== Edit here ====
VERIFY_TIMEOUT=2400            # Maximum verify.sh runtime; normally about seven minutes.
LOAD_WEIGHT_EXPECT=54.11       # Expected Load weight end value in GB; 54.12 is also observed.
LOAD_WEIGHT_TOL=2.5            # GiB tolerance, matching verify.sh.
EXPECT_RESIDENT_EXPERTS=32     # Resident expert count.
MIN_PROMPT_TOKENS=2048         # Minimum long-context request size for the sparse DSA path.
KV_CACHE_REF="0.54"            # Observed KV cache size at 40960 tokens; informational only.
SWIGLU_LIMIT=10                # Expected clamp limit in both SwiGLU log lines.
HEARTBEAT_SEC=60               # Heartbeat interval during silent execution.
# ==== End editable values ====

STAGE="验收"
say()  { echo "[$STAGE] $*"; }
note() { echo "[$STAGE] ....  $*"; }
PASS_N=0
FAIL_N=0
ok()   { echo "[$STAGE] PASS  $*"; PASS_N=$((PASS_N + 1)); }
bad()  { echo "[$STAGE] FAIL  $*"; FAIL_N=$((FAIL_N + 1)); }
die()  { echo "[$STAGE] FAIL  $*" >&2; exit 1; }

# ---- Login shell ----
# Some images add libpython to LD_LIBRARY_PATH only from the login profile.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if ! shopt -q login_shell; then
  if [ -n "${GLM53_RELOGIN:-}" ]; then
    die "不是登录 shell，自动重入 bash -l 也没生效。手工执行 bash -l $SELF"
  fi
  export GLM53_RELOGIN=1
  say "当前不是登录 shell，自动用 bash -l 重新执行本脚本"
  exec bash -l "$SELF" "$@"
fi
unset GLM53_RELOGIN

# ---- Configuration ----
# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${KT_REPO:?配置文件里没有 KT_REPO，先重跑第 1 节}"
: "${GLM53_PORT:?配置文件里没有 GLM53_PORT，先重跑第 1 节}"
: "${GLM53_ARTIFACT_ROOT:?配置文件里没有 GLM53_ARTIFACT_ROOT，先重跑第 1 节}"

TOOLS="$KT_REPO/kt-kernel/tools/ascend_glm53"
[ -d "$TOOLS" ] || die "找不到 $TOOLS。回第 2 节确认取码、打补丁、复制脚本都成功了"
[ -f "$TOOLS/verify.sh" ] || die "找不到 $TOOLS/verify.sh"

LOG_DIR="${GLM53_LOG_DIR:-$GLM53_ARTIFACT_ROOT/logs}"
mkdir -p "$LOG_DIR"
# Resolve the log directory before changing into $TOOLS.
LOG_DIR="$(cd "$LOG_DIR" && pwd)" || die "进不去日志目录 $LOG_DIR"
SERVE_LOG="$LOG_DIR/serve.log"

VERIFY_HOST="${GLM53_HOST:-127.0.0.1}"
case "$VERIFY_HOST" in
  0.0.0.0|'') VERIFY_URL_HOST=127.0.0.1 ;;
  ::|'[::]') VERIFY_URL_HOST='[::1]' ;;
  '['*']') VERIFY_URL_HOST="$VERIFY_HOST" ;;
  *:*) VERIFY_URL_HOST="[$VERIFY_HOST]" ;;
  *) VERIFY_URL_HOST="$VERIFY_HOST" ;;
esac

# verify.sh skips all six log-derived gates when the server log is unreadable,
# so require the log before running the long acceptance request.
[ -r "$SERVE_LOG" ] || die "读不到 $SERVE_LOG。以第 6 节 serve.sh 打的 [serve] log 那一行为准，路径对上了再跑"

# verify.sh does not wait for readiness, so probe /health first.
if curl -sS --fail --noproxy '*' --max-time 15 "http://${VERIFY_URL_HOST}:${GLM53_PORT}/health" >/dev/null 2>&1; then
  ok "服务在端口 $GLM53_PORT 上活着"
else
  die "http://${VERIFY_URL_HOST}:${GLM53_PORT}/health 不通，回第 6 节等就绪。本脚本已用 --noproxy '*' 绕过下载代理"
fi

# ---- Run verify.sh ----
RAW="$LOG_DIR/verify-$(date +%Y%m%d-%H%M%S).log"
say "verify.sh 全程约 7 分钟。/health_generate 之后会发一个 14k token 的长上下文请求，"
say "第一次预填充要把路由专家换进来并选定动态热专家，约 6 分钟，这期间屏幕上一个字都不会出。"
say "这段静默是正常的，不要打断。原始输出留在 $RAW"

# Record the server-log size so stale logs cannot satisfy the SwiGLU and KV checks.
logsize() { wc -c < "$1" 2>/dev/null | tr -d '[:space:]' || true; }
SERVE_LOG_PRE="$SERVE_LOG"
SERVE_SIZE_PRE="$(logsize "$SERVE_LOG")"; SERVE_SIZE_PRE="${SERVE_SIZE_PRE:-0}"

# Use a sentinel-based heartbeat so no orphaned sleep keeps a tee pipeline open.
HB=""
HB_FLAG="$LOG_DIR/.heartbeat.$$"
stop_heartbeat() {
  if [ -n "${HB_FLAG:-}" ]; then rm -f "$HB_FLAG" || true; fi
  if [ -n "${HB:-}" ]; then
    wait "$HB" 2>/dev/null || true
    HB=""
  fi
}
heartbeat() {
  local t0=$SECONDS n=0
  while [ -e "$HB_FLAG" ]; do
    sleep 1
    n=$((n + 1))
    if [ "$n" -ge "$HEARTBEAT_SEC" ]; then
      n=0
      echo "[$STAGE] ....  仍在跑，已 $(( (SECONDS - t0) / 60 )) 分钟，长上下文预填充期间没有输出是正常的"
    fi
  done
}
: > "$HB_FLAG"
heartbeat &
HB=$!
trap stop_heartbeat EXIT

# Archive extraction may drop executable bits.
RUNNER=(./verify.sh)
if [ ! -x "$TOOLS/verify.sh" ]; then
  say "verify.sh 没有可执行位，改用 bash 调用"
  RUNNER=(bash ./verify.sh)
fi
TMO=()
command -v timeout >/dev/null 2>&1 && TMO=(timeout "$VERIFY_TIMEOUT")

cd "$TOOLS"
set +e
${TMO[@]+"${TMO[@]}"} "${RUNNER[@]}" 2>&1 | tee "$RAW" | awk '
  /run_gsm8k\.py/            { next }   # Suppress two hints for scripts outside this delivery.
  /per-token criteria/       { next }
  { print; fflush() }'
RC=${PIPESTATUS[0]}
set -e
stop_heartbeat

if [ "$RC" -eq 124 ]; then
  say "verify.sh 跑满 ${VERIFY_TIMEOUT}s 被 timeout 杀掉，输出是截断的。"
  say "下面大部分 FAIL 都是这一条带出来的，先改 VERIFY_TIMEOUT 再看别的"
fi

# ---- Parse output ----
# Strip ANSI color escapes before matching PASS and FAIL lines.
ESC=$(printf '\033')
CLEAN="$RAW.clean"
sed -e "s/${ESC}\[[0-9;]*[A-Za-z]//g" -e 's/\r$//' "$RAW" > "$CLEAN"

has() { grep -qE -- "$1" "$CLEAN"; }

# Require a specific line to report PASS.
gate() {
  local pat="$1" desc="$2" actual
  if grep -qE -- "^[[:space:]]*PASS[[:space:]]+$pat" "$CLEAN"; then
    ok "$desc"
  else
    actual="$(grep -E -- "$pat" "$CLEAN" | tail -1 || true)"
    bad "$desc 这一行不是 PASS。实际是 ${actual:-没有这一行}"
  fi
}

# Require a specific line regardless of whether it reports PASS or progress.
present() {
  if has "$1"; then ok "$2 这一行在"; else bad "$2 这一行不在，log 派生的六项被跳过了"; fi
}

echo
say "开始逐行核对 verify.sh 的输出"

# Header lines.
if has 'single-die offload: acceptance'; then
  ok "看到验收头部 == GLM-5.3-Flash single-die offload: acceptance =="
else
  bad "没有验收头部那一行，verify.sh 根本没跑起来"
fi

HDR="$(grep -m1 -E '^[[:space:]]*port[[:space:]]+[0-9]+' "$CLEAN" || true)"
if [ -z "$HDR" ]; then
  bad "没有 port / die / experts resident / log 那一行"
else
  HPORT="$(printf '%s\n' "$HDR" | sed -n 's/.*port[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p')"
  HEXP="$(printf '%s\n' "$HDR" | sed -n 's/.*experts resident[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p')"
  HLOG="$(printf '%s\n' "$HDR" | sed -n 's/.*[[:space:]]log[[:space:]]\{1,\}\(.*[^[:space:]]\)[[:space:]]*$/\1/p')"
  if [ "$HPORT" = "$GLM53_PORT" ]; then
    ok "验收打的端口 $HPORT 就是配置里的 GLM53_PORT"
  else
    bad "验收打的端口是 $HPORT，配置里是 $GLM53_PORT，验的不是同一个服务"
  fi
  if [ "$HEXP" = "$EXPECT_RESIDENT_EXPERTS" ]; then
    ok "常驻专家数 $HEXP"
  else
    bad "常驻专家数是 $HEXP，期望 $EXPECT_RESIDENT_EXPERTS"
  fi
  # Use the log path printed by verify.sh for later SwiGLU checks.
  if [ -n "$HLOG" ] && [ -r "$HLOG" ]; then
    SERVE_LOG="$HLOG"
    ok "验收读的日志是 $SERVE_LOG"
  else
    bad "验收头部写的日志路径读不到，实际是 ${HLOG:-空}"
  fi
fi

# Reject a false pass that skipped all six log-derived checks.
if has 'skipping log-derived gates'; then
  bad "verify.sh 打了 skipping log-derived gates，显存 / KV 缓存 / 计算图 / traceback / kt-kernel / 流式六项被整体跳过，这一轮的 ALL CHECKS PASSED 不作数"
fi

say "六项 log 派生检查在不在"
present 'Load weight end'                                    "显存"
present 'KV cache allocated'                                 "KV 缓存"
present 'decode NPU graph captured|no decode-graph capture'  "计算图"
present 'unexplained tracebacks'                             "traceback"
present 'kt-kernel CPU MoE'                                  "kt-kernel"
present 'streaming prefill engaged'                          "流式"

say "十一项门禁逐条看是不是 PASS。这里先看十项，decode NPU graph captured 那项紧接着单独看"
gate 'resident-expert weight load matches the capacity model' "常驻专家权重加载量符合容量模型"
gate 'KV cache allocated'                                     "KV 缓存已分配"
gate 'no unexplained tracebacks'                              "日志里没有无法解释的 traceback"
gate '/health[[:space:]]*$'                                   "/health 存活"
gate '/health_generate[[:space:]]*$'                          "/health_generate 能出词"
gate 'non-empty completion'                                   "补全非空，后面跟的补全内容不必逐字相同"
gate 'greedy output reproducible'                             "贪心输出在 width 1 下可复现"
gate 'kt-kernel CPU MoE is in the log'                        "kt-kernel 的 CPU 侧专家在日志里"
gate 'streaming prefill engaged'                              "流式预填充已启用"
gate 'no streaming fallbacks'                                 "没有回退出流式路径"

# Reject a false pass when decode graph capture is missing.
if grep -qE -- '^[[:space:]]*PASS[[:space:]]+decode NPU graph captured' "$CLEAN"; then
  ok "decode NPU graph captured 是 PASS"
else
  bad "decode NPU graph captured 不是 PASS。verify.sh 不把这一项计入它自己的失败，本脚本计。它意味着退化成了 eager 模式、解码慢约 5 倍。查配置里的 GLM53_EAGER"
fi

# Reject an inconsistent long-context result that verify.sh reports only as progress.
LC="$(grep -E -- 'long-context retrieval' "$CLEAN" | tail -1 || true)"
if [ -z "$LC" ]; then
  bad "没有 long-context retrieval 那一行"
elif printf '%s' "$LC" | grep -qE -- '-- consistent[[:space:]]*$'; then
  ok "long-context retrieval 以 -- consistent 结尾"
else
  bad "long-context retrieval 没有以 -- consistent 结尾，实际是 $LC。答错通常是 GGUF 少一层、那层零个专家"
fi

# Streaming count is cumulative, so require only a value greater than zero.
SPN="$(grep -E -- 'streaming prefill engaged' "$CLEAN" | tail -1 | sed -n 's/.*(\([0-9]\{1,\}\)).*/\1/p' || true)"
if [ -n "$SPN" ] && [ "$SPN" -gt 0 ]; then
  ok "流式预填充计数 $SPN 大于 0，这是累计计数，再跑一次会变大"
else
  bad "流式预填充计数是 ${SPN:-没解析到}，不大于 0"
fi

# Long-context request size.
PT="$(grep -E -- 'prompt_tokens=' "$CLEAN" | tail -1 | sed -n 's/.*prompt_tokens=\([0-9]\{1,\}\).*/\1/p' || true)"
if [ -n "$PT" ] && [ "$PT" -gt "$MIN_PROMPT_TOKENS" ]; then
  ok "长上下文请求 prompt_tokens=$PT，大于 $MIN_PROMPT_TOKENS，走到了 DSA 稀疏路径"
else
  bad "prompt_tokens 是 ${PT:-没解析到}，没超过 $MIN_PROMPT_TOKENS"
fi

say "硬指标对表"

# Confirm that the server log grew during the acceptance requests.
if [ "$SERVE_LOG" = "$SERVE_LOG_PRE" ]; then
  SERVE_SIZE_POST="$(logsize "$SERVE_LOG")"; SERVE_SIZE_POST="${SERVE_SIZE_POST:-0}"
  if [ "$SERVE_SIZE_POST" -gt "$SERVE_SIZE_PRE" ]; then
    ok "验收期间 serve.log 从 $SERVE_SIZE_PRE 长到 $SERVE_SIZE_POST 字节，读的是这个服务当下写的日志"
  else
    bad "验收期间 serve.log 一个字节都没长，还是 $SERVE_SIZE_PRE 字节。它多半是上一轮留下的旧日志，下面 swiglu 与 KV 两项都不作数。回第 6 节按 rm -f 那步清掉日志、重启服务再来"
  fi
else
  note "验收头部的日志路径和本脚本算出来的不是同一个，跳过旧日志判定"
fi

# Load weight end is not bit-stable across deployments; both 54.11 and 54.12 occur.
LW="$(grep -E -- 'Load weight end' "$CLEAN" | tail -1 | sed -n 's/.*Load weight end:[[:space:]]*\([0-9.]\{1,\}\)[[:space:]]*GB.*/\1/p' || true)"
if [ -z "$LW" ]; then
  bad "没解析到 Load weight end 的数值"
elif awk -v v="$LW" -v e="$LOAD_WEIGHT_EXPECT" -v t="$LOAD_WEIGHT_TOL" \
       'BEGIN { d = v - e; if (d < 0) d = -d; exit !(d <= t) }'; then
  ok "Load weight end $LW GB，落在 $LOAD_WEIGHT_EXPECT 正负 $LOAD_WEIGHT_TOL GiB 以内"
else
  bad "Load weight end $LW GB，超出 $LOAD_WEIGHT_EXPECT 正负 $LOAD_WEIGHT_TOL GiB"
fi

# KV cache size is informational because both 0.54 and 0.65 GB have been observed.
KVL="$(grep -iE -- 'kv[ _-]?cache.*GB|GB.*kv[ _-]?cache' "$SERVE_LOG" | tail -2 || true)"
if [ -n "$KVL" ]; then
  note "KV 缓存日志，参考值 $KV_CACHE_REF GB，源码注释写的是 0.65，两个值都见过，不据此判故障"
  printf '%s\n' "$KVL" | sed "s/^/[$STAGE]       /"
else
  note "serve.log 里没找到带 GB 的 KV 缓存行。判据是上面 KV cache allocated 那一项，不是这个数"
fi

# Match both the uppercase [SWIGLU] and lowercase [swiglu] log forms.
SW="$(grep -nE -- '\[SWIGLU\]|\[KT_STREAM\]\[swiglu\]' "$SERVE_LOG" || true)"
if [ -z "$SW" ]; then
  bad "serve.log 里搜不到 swiglu clamp 的日志，CPU 侧专家的数值截断没生效"
else
  N_UP="$(printf '%s\n' "$SW" | grep -cF -- '[SWIGLU]' || true)"
  N_LO="$(printf '%s\n' "$SW" | grep -cF -- '[KT_STREAM][swiglu]' || true)"
  # Require a numeric boundary so limit=100 does not match limit=10.
  N_BAD="$(printf '%s\n' "$SW" | grep -vcE -- "limit=$SWIGLU_LIMIT([^0-9]|\$)" || true)"
  printf '%s\n' "$SW" | sed "s/^/[$STAGE]       /"
  if [ "$N_UP" -ge 1 ] && [ "$N_LO" -ge 1 ]; then
    ok "swiglu clamp 大写 [SWIGLU] 与小写 [KT_STREAM][swiglu] 两条日志都在"
  else
    bad "swiglu clamp 只有一条日志，大写 $N_UP 条、小写 $N_LO 条，两条都要有"
  fi
  if [ "$N_BAD" -eq 0 ]; then
    ok "swiglu clamp 每一条都带 limit=$SWIGLU_LIMIT"
  else
    bad "有 $N_BAD 条 swiglu 日志没带 limit=$SWIGLU_LIMIT"
  fi
fi

# ---- Result ----
say "verify.sh 自己的结论"
if has '^[[:space:]]*ALL CHECKS PASSED[[:space:]]*$'; then
  ok "verify.sh 打了 ALL CHECKS PASSED"
else
  bad "verify.sh 没有打 ALL CHECKS PASSED"
fi
if has 'CHECKS FAILED'; then
  bad "verify.sh 打了 CHECKS FAILED，先看上面哪几行是 FAIL"
fi
if [ "$RC" -eq 0 ]; then
  ok "verify.sh 退出码 0"
elif [ "$RC" -eq 124 ]; then
  bad "verify.sh 跑满 ${VERIFY_TIMEOUT}s 被 timeout 杀掉"
else
  bad "verify.sh 退出码 $RC"
fi

if has 'LD_PRELOAD detected'; then
  note "输出里的 LD_PRELOAD detected 警告来自镜像的 /etc/profile，不影响验收，不用管"
fi
if grep -qE -- 'run_gsm8k\.py' "$RAW"; then
  note "verify.sh 末尾提到的 run_gsm8k.py 不在交付目录里，那两行已经从屏幕上滤掉，忽略即可。它说的 per-token 判定标准就是下一节的困惑度"
fi

echo
say "自查 $((PASS_N + FAIL_N)) 项，PASS $PASS_N，FAIL $FAIL_N"
if [ "$FAIL_N" -eq 0 ]; then
  echo "=== 验收 完成 ==="
else
  say "原始输出 $RAW"
  say "现场只有一处 tail -100 $SERVE_LOG"
  exit 1
fi
