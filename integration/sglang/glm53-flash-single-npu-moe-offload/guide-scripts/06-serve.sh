#!/usr/bin/env bash
# Step 6: start serve.sh in the background, wait for readiness, or stop the configured port.
set -euo pipefail

# ==== Edit here ====
PID_GRACE=120        # Seconds allowed for serve.sh to create serve.log.pid.
READY_TIMEOUT=3600   # Readiness timeout; first startup loads weights, captures graphs,
                     # and reads the GGUF files into the page cache.
POLL_INTERVAL=15     # Poll interval in seconds.
STOP_TIMEOUT=30      # Seconds to wait for all processes on the port to exit.
TAIL_LINES=40        # Log tail length printed on failure.
# ==== End editable values ====

STAGE="06-serve"
say()  { echo "[$STAGE] $*"; }
pass() { echo "[$STAGE] PASS $*"; }
die()  { echo "[$STAGE] FAIL $*" >&2; exit 1; }

# Disable nounset while sourcing the LD_LIBRARY_PATH line appended by step 3.
set +u
# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
set -u

# ---- Login shell ----
# Inspect the parent shell because some images add libpython to LD_LIBRARY_PATH
# only from the login profile.
in_login_shell() {
  if shopt -q login_shell; then return 0; fi

  # Read the parent's argv from NUL-delimited /proc data, or split ps output as a fallback.
  local argv=""
  if [ -r "/proc/$PPID/cmdline" ]; then
    argv="$(tr '\0' '\n' < "/proc/$PPID/cmdline" 2>/dev/null || true)"
  else
    argv="$(ps -o args= -p "$PPID" 2>/dev/null | sed 's/^[[:space:]]*//' | tr ' ' '\n' || true)"
  fi
  local argv0=""
  argv0="$(printf '%s\n' "$argv" | sed -n '1p')"

  # A leading dash in argv[0] is the strongest login-shell signal.
  case "$argv0" in -?*) return 0 ;; esac

  # Otherwise require a shell parent with an explicit -l or --login option.
  case "${argv0##*/}" in
    bash|sh|zsh|ksh|dash|ash) ;;
    *) return 1 ;;
  esac
  # Inspect options only before the script name; later flags belong to the script.
  local w="" first=1
  while IFS= read -r w; do
    if [ "$first" = 1 ]; then first=0; continue; fi
    case "$w" in
      --login) return 0 ;;
      --*)     ;;
      -*l*)    return 0 ;;   # Combined forms such as -l, -il, and -lc.
      -*)      ;;
      "")      ;;
      *)       break ;;
    esac
  done <<< "$argv"
  return 1
}
in_login_shell || die "不是登录 shell。先执行 bash -l 重开会话，或直接用 bash -l 跑本脚本"
pass "登录 shell"

# ---- Configuration availability ----
[ -n "${GLM53_ARTIFACT_ROOT:-}" ] || die "GLM53_ARTIFACT_ROOT 是空的，确认 GLM53_ENV_FILE 指对了"
[ -n "${KT_REPO:-}" ]             || die "KT_REPO 是空的，确认 GLM53_ENV_FILE 指对了"
# Require a port before constructing the pkill pattern.
[ -n "${GLM53_PORT:-}" ]          || die "GLM53_PORT 是空的，配置文件没读到，别在这里瞎杀进程"
pass "配置已读到，端口 $GLM53_PORT"

TOOLS_DIR="$KT_REPO/kt-kernel/tools/ascend_glm53"
LOG="$GLM53_ARTIFACT_ROOT/logs/serve.log"
PIDFILE="$LOG.pid"
LAUNCH_LOG="$GLM53_ARTIFACT_ROOT/logs/serve.launch.log"
READY_LINE="fired up and ready to roll"
# Count only readiness lines added by the current launch.
READY_BASE=0
# The bracketed pattern matches --port without matching the pgrep/pkill command itself.
PORT_PAT="[-]-port $GLM53_PORT"

server_pids() { pgrep -f -- "$PORT_PAT" 2>/dev/null || true; }
one_line()    { echo "$1" | tr '\n' ' ' | sed 's/[[:space:]]*$//'; }
ready_count() { grep -cF -- "$READY_LINE" "$LOG" 2>/dev/null || true; }
log_ready()   { local n; n="$(ready_count)"; [ "${n:-0}" -gt "$READY_BASE" ]; }
alive()       { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

read_pid() {
  local p=""
  p="$(cat "$PIDFILE" 2>/dev/null || true)"
  p="$(printf '%s' "$p" | head -1 | tr -dc '0-9')"
  # Fall back to the pid printed by serve.sh when the pid file is unavailable.
  if [ -z "$p" ] && [ -f "$LAUNCH_LOG" ]; then
    p="$(sed -n 's/^\[serve\] pid=\([0-9][0-9]*\).*/\1/p' "$LAUNCH_LOG" 2>/dev/null | tail -1)"
  fi
  printf '%s' "$p"
}

dump_tail() {
  echo "[$STAGE] ---- $LOG 尾部 $TAIL_LINES 行 ----"
  tail -n "$TAIL_LINES" "$LOG" 2>/dev/null || echo "[$STAGE] （读不到 $LOG）"
  echo "[$STAGE] ---- 尾部结束 ----"
}

show_serve_fatal() {
  local f=""
  f="$(grep -n 'FATAL' "$LAUNCH_LOG" 2>/dev/null || true)"
  if [ -n "$f" ]; then
    echo "[$STAGE] serve.sh 自己打的报错如下"
    printf '%s\n' "$f"
  fi
}

# Wait for readiness and report each failure mode before exiting.
wait_ready() {
  local t0=$SECONDS pid=""
  while ! log_ready; do
    pid="$(read_pid)"

    # A stale pid file is fatal only when no process remains on the configured port.
    if [ -n "$pid" ] && ! alive "$pid" && [ -z "$(server_pids)" ]; then
      say "服务进程 $pid 已退出"
      show_serve_fatal
      dump_tail
      die "服务进程已退出"
    fi

    # A missing pid file after the grace period indicates an early launcher exit.
    if [ -z "$pid" ] && [ $((SECONDS - t0)) -gt "$PID_GRACE" ]; then
      say "等了 ${PID_GRACE}s 还没有 $PIDFILE"
      say "serve.sh 的前置检查失败会直接退出，四种可能"
      say "  GGUF 层数不等于 42"
      say "  sglang 解析到了 \$SGLANG_REPO 之外"
      say "  GLM53_KT_NUMA_NODES 指了不存在的节点"
      say "  设了 GLM53_PIN_CORES 但机器上没有 taskset"
      show_serve_fatal
      say "注意 $LOG 此时可能还是上一轮留下的旧日志，不要照着它排查"
      dump_tail
      die "serve.sh 没能起来"
    fi

    # The process is alive but readiness timed out.
    if [ $((SECONDS - t0)) -gt "$READY_TIMEOUT" ]; then
      say "超过 ${READY_TIMEOUT}s 仍未就绪，进程还活着"
      dump_tail
      die "等待就绪超时"
    fi

    say "等待中，已等 $(( (SECONDS - t0) / 60 )) 分 $(( (SECONDS - t0) % 60 )) 秒"
    sleep "$POLL_INTERVAL"
  done

  # Confirm a live process after observing readiness in the log.
  pid="$(read_pid)"
  if ! alive "$pid" && [ -z "$(server_pids)" ]; then
    say "$LOG 里有就绪行，但端口 $GLM53_PORT 上没有活着的进程，这多半是上一轮的旧日志"
    say "修复办法是 rm -f $LOG $PIDFILE 之后重跑本脚本"
    dump_tail
    die "就绪行来自旧日志"
  fi
  return 0
}

report_ready() {
  pass "服务已就绪，日志里出现了 $READY_LINE"
  say "端口 $GLM53_PORT，日志 $LOG"
  say "接着跑 verify.sh 验收。验收和精度评测都要服务在跑，两步都做完再执行 bash $0 stop"
}

do_start() {
  [ -d "$TOOLS_DIR" ]        || die "找不到 $TOOLS_DIR，先跑完取码打补丁那一步"
  [ -f "$TOOLS_DIR/serve.sh" ] || die "找不到 $TOOLS_DIR/serve.sh"
  # Archive extraction may drop executable bits.
  [ -x "$TOOLS_DIR/serve.sh" ] || die "serve.sh 没有可执行位，先跑 chmod +x $TOOLS_DIR/*.sh"
  mkdir -p "$GLM53_ARTIFACT_ROOT/logs" || die "建不出 $GLM53_ARTIFACT_ROOT/logs，先看这个目录的权限"

  # Do not start a duplicate process when the port is already in use.
  local running=""
  running="$(server_pids)"
  if [ -n "$running" ]; then
    say "端口 $GLM53_PORT 上已有进程 $(one_line "$running")，不重复启动"
    if log_ready; then
      report_ready
      return 0
    fi
    say "还没就绪，直接进入等待"
    wait_ready
    report_ready
    return 0
  fi

  # Remove stale launch files before serve.sh prechecks can exit without replacing them.
  rm -f "$LOG" "$PIDFILE" "$LAUNCH_LOG" || true
  local f=""
  for f in "$LOG" "$PIDFILE" "$LAUNCH_LOG"; do
    if [ -e "$f" ]; then
      die "删不掉 $f，先手工 rm -f 它再重跑"
    fi
  done
  pass "上一轮的 serve.log、serve.log.pid、serve.launch.log 已清掉"

  say "启动 serve.sh，它后台运行并立即返回"
  # Redirect to a file because a background child inheriting a pipe would keep tee open.
  if ! ( cd "$TOOLS_DIR" && ./serve.sh ) >"$LAUNCH_LOG" 2>&1; then
    cat "$LAUNCH_LOG" 2>/dev/null || true
    die "serve.sh 退出非零，报错见上面 [serve] 开头的那几行"
  fi
  cat "$LAUNCH_LOG" 2>/dev/null || true

  # Prefer the log path printed by serve.sh over the derived path.
  local parsed=""
  parsed="$(sed -n 's/^\[serve\] log[[:space:]]*:[[:space:]]*tail -f[[:space:]]*//p' \
            "$LAUNCH_LOG" 2>/dev/null | sed 's/[[:space:]]*$//' | tail -1)"
  if [ -n "$parsed" ] && [ "$parsed" != "$LOG" ]; then
    say "serve.sh 打的日志路径是 $parsed，与配置推出的 $LOG 不一致，改用前者"
    say "刚才清掉的是 $LOG，$parsed 上的旧日志清不掉，只等这一轮新写出来的就绪行"
    LOG="$parsed"
    PIDFILE="$LOG.pid"
  fi
  # A parsed log path may contain old readiness lines, so record its current count.
  READY_BASE="$(ready_count)"
  READY_BASE="${READY_BASE:-0}"

  say "等待就绪，判据是 $LOG 里新出现 $READY_LINE"
  say "首次启动很慢，这期间 /health 返回 503 是正常的，不要拿 /health 当就绪判据"
  wait_ready
  report_ready
}

do_stop() {
  local pids=""
  pids="$(server_pids)"
  if [ -z "$pids" ]; then
    pass "端口 $GLM53_PORT 上没有进程，无需停"
    return 0
  fi
  say "停掉端口 $GLM53_PORT 上的进程 $(one_line "$pids")"
  pkill -f -- "$PORT_PAT" || true

  local t0=$SECONDS
  while [ -n "$(server_pids)" ] && [ $((SECONDS - t0)) -lt "$STOP_TIMEOUT" ]; do
    sleep 1
  done

  local left=""
  left="$(server_pids)"
  if [ -n "$left" ]; then
    say "等了 ${STOP_TIMEOUT}s 还剩进程 $(one_line "$left")"
    die "没停干净，执行 pkill -9 -f -- \"$PORT_PAT\" 再跑一次本脚本"
  fi
  pass "服务已停"
}

ACTION="${1:-start}"
case "$ACTION" in
  start) do_start ;;
  stop)  do_stop ;;
  *)     echo "用法 bash $0 [start|stop]" >&2; exit 2 ;;
esac

echo "=== $STAGE 完成 ==="
