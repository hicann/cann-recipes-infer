#!/usr/bin/env bash
# Step 8: prepare WikiText, run perplexity, and evaluate GSM8K and GPQA.
set -euo pipefail

# ==== Edit here ====
PPL_WINDOWS=32          # Perplexity windows; only 32 can be compared with 3.588836580023618.
SRC_PARQUET=""          # WikiText test.parquet; empty searches locally, then downloads it.
BENCH_LIMIT=5           # Questions per benchmark; 0 runs the full serial dataset.
BENCH_EFFORT=max        # low, high, or max; max follows the template's default branch.
BENCH_DATASETS="gsm8k gpqa"
EVAL_VENV=""            # Dedicated evalscope venv; empty uses $GLM53_ENV_ROOT/.venv-eval.
MS_CACHE=""             # ModelScope cache; empty uses $GLM53_ENV_ROOT/ms_cache.
# ==== End editable values ====

TAG="精度评测"
# Single source for the evalscope version required by the guide.
EVALSCOPE_VERSION=1.11.1
say()  { printf '[%s] %s\n' "$TAG" "$*"; }
pass() { printf '[%s] PASS %s\n' "$TAG" "$*"; }
warn() { printf '[%s] WARN %s\n' "$TAG" "$*"; }
fail() { printf '[%s] FAIL %s\n' "$TAG" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
用法
  bash 08-accuracy.sh [corpus|ppl|bench|all] [--limit N] [--effort low|high|max] [--ppl-windows N]

  corpus  只备语料并校验
  ppl     只跑困惑度
  bench   只跑 GSM8K 与 GPQA
  all     三步都跑，不给子命令时的默认值

  --limit N        每个基准跑前 N 题，0 表示全量。覆盖脚本顶部的 BENCH_LIMIT
  --effort L       reasoning_effort，low / high / max。覆盖脚本顶部的 BENCH_EFFORT
  --ppl-windows N  困惑度窗口数。覆盖脚本顶部的 PPL_WINDOWS，只有 32 能对参考值

  语料路径、evalscope 虚拟环境、ModelScope 缓存没有命令行开关，改脚本顶部那一块
USAGE
}

STAGE=all
while [ $# -gt 0 ]; do
  case "$1" in
    corpus|ppl|bench|all) STAGE="$1"; shift ;;
    --limit)       BENCH_LIMIT="${2:?--limit 后面要跟数字}"; shift 2 ;;
    --effort)      BENCH_EFFORT="${2:?--effort 后面要跟 low/high/max}"; shift 2 ;;
    --ppl-windows) PPL_WINDOWS="${2:?--ppl-windows 后面要跟数字}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) usage >&2; fail "认不出的参数 $1" ;;
  esac
done

case "$BENCH_EFFORT" in
  low|high|max) ;;
  *) fail "BENCH_EFFORT 只能是 low / high / max，现在是 $BENCH_EFFORT" ;;
esac
case "$BENCH_LIMIT" in
  ''|*[!0-9]*) fail "BENCH_LIMIT 必须是非负整数，现在是 $BENCH_LIMIT" ;;
esac
case "$PPL_WINDOWS" in
  ''|*[!0-9]*) fail "PPL_WINDOWS 必须是正整数，现在是 $PPL_WINDOWS" ;;
esac
# Remove leading zeros before the string comparisons below.
BENCH_LIMIT=$((10#$BENCH_LIMIT))
PPL_WINDOWS=$((10#$PPL_WINDOWS))
[ "$PPL_WINDOWS" -gt 0 ] || fail "PPL_WINDOWS 必须是正整数，现在是 $PPL_WINDOWS"

# Disable nounset while sourcing the LD_LIBRARY_PATH line appended by step 3.
# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
# Report an invalid config path explicitly before sourcing it.
[ -r "$GLM53_ENV_FILE" ] || fail "读不到配置文件 $GLM53_ENV_FILE。GLM53_ENV_FILE 要指向第 1 步写出来的 glm53.env"
set +u
# shellcheck disable=SC1090
. "$GLM53_ENV_FILE"
set -u

# Clear positional arguments before vendor set_env.sh parses the caller's arguments.
set --

for v in GLM53_ROOT GLM53_ENV_ROOT GLM53_ARTIFACT_ROOT GLM53_VENV KT_REPO GLM53_PORT GLM53_PYTHON_BIN; do
  [ -n "${!v:-}" ] || fail "配置文件里没有 $v。确认 GLM53_ENV_FILE 指的是第 1 步写出来的那个 glm53.env"
done
unset v

TOOLS_DIR="$KT_REPO/kt-kernel/tools/ascend_glm53"
VENV_PY="$GLM53_VENV/bin/python"
EVAL_DIR="${GLM53_EVAL_DIR:-$GLM53_ENV_ROOT/eval}"
PARQUET="$EVAL_DIR/wikitext/test.parquet"
LOG_DIR="$GLM53_ARTIFACT_ROOT/logs"
[ -n "$EVAL_VENV" ] || EVAL_VENV="$GLM53_ENV_ROOT/.venv-eval"
[ -n "$MS_CACHE" ]  || MS_CACHE="$GLM53_ENV_ROOT/ms_cache"

mkdir -p "$LOG_DIR"

# Check this shell and then its parent for login-shell status.
in_login_shell() {
  if shopt -q login_shell; then return 0; fi

  # Read the parent's argv from NUL-delimited /proc data, or split ps output as a fallback.
  local argv=""
  if [ -r "/proc/${PPID:-$$}/cmdline" ]; then
    argv="$(tr '\0' '\n' < "/proc/${PPID:-$$}/cmdline" 2>/dev/null || true)"
  else
    argv="$(ps -o args= -p "${PPID:-$$}" 2>/dev/null | sed 's/^[[:space:]]*//' | tr ' ' '\n' || true)"
  fi
  local argv0=""
  argv0="$(printf '%s\n' "$argv" | sed -n '1p')"

  # A leading dash in argv[0] is the strongest login-shell signal.
  case "$argv0" in -?*) return 0 ;; esac

  # Otherwise require a shell parent with an explicit login option.
  case "${argv0##*/}" in
    bash|sh|zsh|ksh|dash|ash) ;;
    *) return 1 ;;
  esac
  local w=""
  while IFS= read -r w; do
    case "$w" in
      -l|--login|-lc|-cl|-il|-li) return 0 ;;
    esac
  done <<< "$argv"
  return 1
}
in_login_shell || warn "看起来不是从登录 shell 里跑的。这一版镜像的 python 是 --enable-shared 构建，
       libpython 只有登录 shell 的 profile 才放进 LD_LIBRARY_PATH。出问题就用 bash -l 重进"

[ -x "$VENV_PY" ] || fail "没有 $VENV_PY。先做完第 3 步建虚拟环境"
# ctypes exposes a missing libpython before later commands report misleading errors.
"$VENV_PY" -c 'import ctypes' >/dev/null 2>&1 \
  || fail "$VENV_PY 连 ctypes 都 import 不了，多半是没走登录 shell。用 bash -l 重进再跑"

# Probe the shared service endpoint with a timeout before running any evaluation.
check_service() {
  local url_host
  case "${GLM53_HOST:-127.0.0.1}" in
    0.0.0.0|'') url_host=127.0.0.1 ;;
    ::|'[::]') url_host='[::1]' ;;
    '['*']') url_host="${GLM53_HOST}" ;;
    *:*) url_host="[${GLM53_HOST}]" ;;
    *) url_host="${GLM53_HOST}" ;;
  esac
  if curl -sS --fail --noproxy '*' --connect-timeout 5 --max-time 15 \
       "http://${url_host}:${GLM53_PORT}/health" >/dev/null 2>&1; then
    pass "服务在 ${url_host}:${GLM53_PORT} 上响应"
  else
    say "修复 cd $TOOLS_DIR && ./serve.sh   等它打出 fired up and ready to roll 再回来"
    say "服务刚起来时 /health 返回 503 是正常的，等就绪了再跑本步"
    fail "服务没在 ${url_host}:${GLM53_PORT} 上响应"
  fi
}

# Reuse the step-5 environment while vendor scripts are sourced without nounset.
source_glm53_env() {
  [ -d "$TOOLS_DIR" ] || fail "没有 $TOOLS_DIR。先做完第 2 步取码打补丁"
  cd "$TOOLS_DIR"
  set +eu
  # shellcheck disable=SC1091
  . ./glm53_env.sh
  set -eu
  [ -n "${GLM53_PYTHON:-}" ] && [ -x "${GLM53_PYTHON:-}" ] \
    || fail "source glm53_env.sh 之后 GLM53_PYTHON 还是空的或者不可执行"
}

# Validate the text column, 4358 rows, and 1285622 characters.
check_parquet() {
  GLM53_CHECK_PARQUET="$1" "$VENV_PY" - <<'PY'
import os, sys
try:
    import pyarrow.parquet as pq
except ImportError:
    print("  没有 pyarrow。第 5 步的 setup.sh deps 会装上它"); sys.exit(2)
p = os.environ["GLM53_CHECK_PARQUET"]
try:
    t = pq.read_table(p)
except Exception as e:
    print("  读不出来", e); sys.exit(1)
cols = [f.name for f in t.schema]
s = "".join(r["text"] for r in t.to_pylist()) if "text" in cols else ""
print("  列   ", cols, " 期望 ['text']")
print("  行数 ", t.num_rows, " 期望 4358")
print("  字符 ", len(s), " 期望 1285622")
if s:
    print("  开头 ", repr(s[:40]))
sys.exit(0 if (cols == ["text"] and t.num_rows == 4358 and len(s) == 1285622) else 1)
PY
}

stage_corpus() {
  say "语料目录 $EVAL_DIR/wikitext"
  "$VENV_PY" -c 'import pyarrow' >/dev/null 2>&1 \
    || fail "项目虚拟环境里没有 pyarrow，校验不了语料。回第 5 步重跑 ./setup.sh deps"
  # A shell-only GLM53_EVAL_DIR override is lost before later perplexity runs.
  if [ -n "${GLM53_EVAL_DIR:-}" ] && ! grep -q '^export GLM53_EVAL_DIR=' "$GLM53_ENV_FILE" 2>/dev/null; then
    warn "GLM53_EVAL_DIR 只在当前 shell 里，没写进 $GLM53_ENV_FILE。
       通过 GLM53_EVAL_DIR 指定目录并重跑第 1 步，不要用 >> 追加，第 1 步是截断写"
  fi

  if [ -f "$PARQUET" ] && check_parquet "$PARQUET"; then
    pass "语料已就位，三项与期望一致"
    return 0
  fi

  mkdir -p "$EVAL_DIR/wikitext"
  local src="" cand=""

  if [ -n "$SRC_PARQUET" ]; then
    [ -f "$SRC_PARQUET" ] || fail "SRC_PARQUET 指的文件不存在 $SRC_PARQUET"
    src="$SRC_PARQUET"
  else
    say "全盘找现成的 wikitext test.parquet，最多几分钟"
    local hits=""
    if command -v timeout >/dev/null 2>&1; then
      hits="$(timeout 300 find / -name 'test.parquet' -path '*wikitext*' -type f 2>/dev/null | head -5 || true)"
    else
      hits="$(find / -name 'test.parquet' -path '*wikitext*' -type f 2>/dev/null | head -5 || true)"
    fi
    for cand in $hits; do
      [ "$cand" -ef "$PARQUET" ] 2>/dev/null && continue
      say "候选 $cand"
      if check_parquet "$cand"; then src="$cand"; break; fi
      say "  这份三项对不上，换下一个"
    done
  fi

  if [ -n "$src" ]; then
    say "拷贝 $src"
    cp -f "$src" "$PARQUET"
  else
    say "机器上没有能用的，从 HuggingFace 下"
    if ! GLM53_EVAL_DIR="$EVAL_DIR" "$VENV_PY" - <<'PY'
import os, sys
try:
    from datasets import load_dataset
except ImportError:
    print("没有 datasets 包。第 5 步的 setup.sh deps 会装上它"); sys.exit(1)
d = os.environ["GLM53_EVAL_DIR"] + "/wikitext/test.parquet"
load_dataset("wikitext", "wikitext-2-raw-v1", split="test").to_parquet(d)
print("wrote", d)
PY
    then
      say "受限网络下这一步会报 [Errno 99] Cannot assign requested address ... huggingface.co，重试 5 次后超时。"
      say "修复 把脚本顶部的 SRC_PARQUET 填成你那份 parquet 的路径，再跑一次 bash $0 corpus"
      fail "语料没拿到"
    fi
  fi

  check_parquet "$PARQUET" || fail "语料三项对不上。换一份语料就是换基准，跑出来的困惑度不能和 3.588836580023618 比"
  pass "语料就位 $PARQUET"
}

stage_ppl() {
  [ -f "$PARQUET" ] || fail "没有 $PARQUET。先跑 bash $0 corpus"
  check_service
  source_glm53_env

  # Probe the three dependencies before the long run.
  "$GLM53_PYTHON" -c 'import transformers, pyarrow, requests' >/dev/null 2>&1 \
    || fail "transformers / pyarrow / requests 缺包。回第 5 步重跑 ./setup.sh deps"
  pass "run_ppl.py 的三个依赖都在"

  # Match run_ppl.py's GLM53_EVAL_DIR fallback to the corpus validated above.
  local in_use="${GLM53_EVAL_DIR:-$GLM53_ENV_ROOT/eval}/wikitext/test.parquet"
  [ "$in_use" = "$PARQUET" ] \
    || warn "source glm53_env.sh 之后语料路径变成了 $in_use，run_ppl.py 读的是这一份，不是刚校验的 $PARQUET"

  local out="$LOG_DIR/ppl${PPL_WINDOWS}.json"
  local run_out="${out}.run.$$"
  local log="$LOG_DIR/ppl${PPL_WINDOWS}.stdout.log"

  # Always query the current service; an old result is informational only.
  if [ -f "$out" ]; then
    say "发现旧结果 ${out}（$(date -r "$out" '+%F %T' 2>/dev/null || echo 未知时间)），本轮将重新请求当前服务并覆盖它"
  fi
  # Replace the fixed result only after this process creates a complete temporary result.
  rm -f "$run_out"
  say "跑 $PPL_WINDOWS 窗，32 窗约 10 分钟（实跑 631 秒），每 10 窗打一行进度"
  # Invoke the configured interpreter instead of resolving the script shebang through PATH.
  "$GLM53_PYTHON" run_ppl.py --limit "$PPL_WINDOWS" --out "$run_out" 2>&1 | tee "$log" \
    || fail "run_ppl.py 退出码非零，尾部输出在 ${log}。报 no corpus at ... 就先跑 bash $0 corpus"
  [ -f "$run_out" ] || fail "$run_out 没生成；保留旧结果但本轮不会校验它"
  mv -f "$run_out" "$out"

  if [ "$PPL_WINDOWS" != "32" ]; then
    warn "只有 32 窗能和参考值比。同一配置下 3 窗是 2.2445、12 窗是 3.1449、32 窗是 3.5888，窗口数不同的两个数不能比"
    pass "跑完了，结果在 $out"
    return 0
  fi

  local rc=0
  PPL_JSON="$out" "$VENV_PY" - <<'PY' || rc=$?
import json, os, sys
ref_ppl, ref_nll = 3.588836580023618, 1.2778280775605546
# Ignore per-window values so the flattened lookup selects the top-level metrics.
SKIP = ("windows",)
flat = {}
def walk(o, pre="", depth=0):
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            if depth == 0 and kl in SKIP:
                continue
            key = pre + kl
            if isinstance(v, (dict, list)):
                walk(v, key + ".", depth + 1)
            else:
                flat.setdefault(key, v)
    elif isinstance(o, list):
        for v in o:
            walk(v, pre, depth + 1)
walk(json.load(open(os.environ["PPL_JSON"])))
def pick(names):
    # Prefer an exact top-level key before falling back to nested keys.
    for n in names:
        if n in flat:
            return flat[n]
    for n in names:
        for k, v in flat.items():
            if k.endswith("." + n):
                return v
    return None
ppl, nll = pick(["perplexity", "ppl"]), pick(["mean_nll", "meannll", "nll_mean"])
if ppl is None or nll is None:
    print("  json 里没找到 perplexity / mean_nll，键是", sorted(flat)[:20]); sys.exit(2)
ppl, nll = float(ppl), float(nll)
print(f"  perplexity {ppl!r}  期望 {ref_ppl!r}")
print(f"  mean_nll   {nll!r}  期望 {ref_nll!r}")
if ppl == ref_ppl and nll == ref_nll:
    sys.exit(0)
print(f"  差值 perplexity {ppl - ref_ppl:+.12g}   mean_nll {nll - ref_nll:+.12g}")
sys.exit(1)
PY

  if [ "$rc" = 0 ]; then
    pass "32 窗困惑度与参考值逐位相同"
  elif [ "$rc" = 2 ]; then
    # Fall back to the printed precision when JSON field names do not match.
    warn "$out 里没有 perplexity / mean_nll 字段，改比屏幕输出的位数"
    [ -f "$log" ] || fail "也没有 $log 可比。rm $out 之后重跑一次"
    grep -E 'perplexity|mean NLL' "$log" || true
    grep -qE 'perplexity[[:space:]]+3\.5888' "$log" && grep -qE 'mean NLL[[:space:]]+1\.277828' "$log" \
      && pass "屏幕输出的位数与参考值一致" \
      || fail "屏幕输出与 3.5888 / 1.277828 对不上"
  else
    fail "32 窗困惑度与参考值不符。参考值是本文基线加本补丁跑出来的，噪声下限 0 ULP，对不上就是部署有差异"
  fi
}

stage_bench() {
  check_service

  # Keep evalscope outside the project venv because its dependencies conflict with pinned torch packages.
  local venv_real eval_real
  venv_real="$(cd "$GLM53_VENV" 2>/dev/null && pwd -P || echo "$GLM53_VENV")"
  eval_real="$(cd "$EVAL_VENV" 2>/dev/null && pwd -P || echo "$EVAL_VENV")"
  { [ "$venv_real" != "$eval_real" ] && [ "$GLM53_VENV" != "$EVAL_VENV" ]; } \
    || fail "EVAL_VENV 不能等于项目虚拟环境 $GLM53_VENV，换一个目录"
  if compgen -G "$GLM53_VENV/lib/python*/site-packages/evalscope" >/dev/null 2>&1; then
    warn "项目虚拟环境里有 evalscope，它是被误装进去的。核对一下 $VENV_PY -c 'import torch, torch_npu' 还通不通"
  fi

  local ev_py="$EVAL_VENV/bin/python"
  if [ -d "$EVAL_VENV" ] && [ ! -x "$ev_py" ]; then
    # Recreate only a directory identified as a Python virtual environment.
    if [ ! -f "$EVAL_VENV/pyvenv.cfg" ]; then
      fail "$EVAL_VENV 已经存在，但里面既没有 bin/python 也没有 pyvenv.cfg，不像虚拟环境，本脚本不动它。
       确认这个路径没填错。真要用这个位置就自己先删干净，或者把 EVAL_VENV 换成一个空路径"
    fi
    say "$EVAL_VENV 是个坏掉的虚拟环境（有 pyvenv.cfg 没 bin/python），删掉重建"
    rm -rf "$EVAL_VENV"
  fi
  if [ ! -x "$ev_py" ]; then
    say "建 evalscope 专用虚拟环境 $EVAL_VENV"
    command -v "$GLM53_PYTHON_BIN" >/dev/null 2>&1 || fail "找不到解释器 $GLM53_PYTHON_BIN"
    "$GLM53_PYTHON_BIN" -m venv "$EVAL_VENV" \
      || fail "$GLM53_PYTHON_BIN -m venv $EVAL_VENV 没建起来。看这个解释器有没有 venv 模块、目标目录能不能写"
    [ -x "$ev_py" ] || fail "建完了还是没有 $ev_py"
  fi
  if [ "$("$ev_py" -c 'import evalscope; print(evalscope.__version__)' 2>/dev/null || true)" != "$EVALSCOPE_VERSION" ]; then
    say "装 evalscope==$EVALSCOPE_VERSION，要能连 pypi"
    "$EVAL_VENV/bin/pip" install -q "evalscope==$EVALSCOPE_VERSION" \
      || fail "evalscope 装不上。确认这台机器能连 pypi"
  fi
  local ev_ver
  ev_ver="$("$ev_py" -c 'import evalscope; print(evalscope.__version__)' 2>/dev/null || true)"
  [ "$ev_ver" = "$EVALSCOPE_VERSION" ] || fail "evalscope 版本是 ${ev_ver:-空}，期望 $EVALSCOPE_VERSION"
  pass "evalscope $ev_ver"

  mkdir -p "$MS_CACHE"
  export MODELSCOPE_CACHE="$MS_CACHE"
  say "语料由 evalscope 从 ModelScope 拉，缓存在 $MS_CACHE"

  source_glm53_env
  [ -n "${GLM53_MODEL_PATH:-}" ] || fail "GLM53_MODEL_PATH 是空的。run_bench.py 会直接退出报 no model"

  # Validate numeric KV limits before arithmetic; exhausting the pool is a device error.
  local kv="${GLM53_MAX_TOTAL_TOKENS:-40960}" bs="${GLM53_MAX_RUNNING_REQUESTS:-1}" mt=16384
  case "$kv" in
    ''|*[!0-9]*) fail "GLM53_MAX_TOTAL_TOKENS 不是个数字，是 '$kv'，算不了 KV 预算。
       服务没锁 KV 上限时它就是这样，本脚本不猜。回第 6 步用固定的 --max-total-tokens 起服务，
       或者 export GLM53_MAX_TOTAL_TOKENS=<服务实际的 KV token 数> 再跑本步" ;;
  esac
  case "$bs" in
    ''|*[!0-9]*) fail "GLM53_MAX_RUNNING_REQUESTS 不是个数字，是 '$bs'" ;;
  esac
  if [ "$((bs * mt))" -gt "$kv" ]; then
    fail "并发 $bs × max_tokens $mt = $((bs * mt)) 超过 KV 池 $kv。降并发或降 max_tokens，别硬跑"
  fi
  pass "KV 预算 $bs × $mt = $((bs * mt)) / $kv"

  if [ "$BENCH_EFFORT" = low ]; then
    warn "low effort 下这两个基准还没有参考值，第一次跑出来的就是本部署的基线"
  fi
  if [ "$BENCH_LIMIT" = 0 ]; then
    warn "全量。并发是 $bs，单卡上是串行的，GSM8K 有 1319 题，可能要跑几十个小时。
       evalscope 只在整轮结束时落盘，中途崩了这一轮全丢，不能续跑"
  else
    say "每个基准只跑前 $BENCH_LIMIT 题。--limit 取的是前 N 条不是抽样，这个 score 只用来看跑不跑得通和每题多慢"
  fi

  local ds ds_dir bench_log wall scored score args report stamp
  for ds in $BENCH_DATASETS; do
    case "$ds" in
      gsm8k) ds_dir="gsm8k" ;;
      gpqa|gpqa_diamond) ds_dir="gpqa_diamond" ;;
      *) fail "认不出的数据集 $ds，只支持 gsm8k 和 gpqa" ;;
    esac
    bench_log="$LOG_DIR/bench_${ds_dir}_${BENCH_EFFORT}.log"
    args="--dataset $ds --effort $BENCH_EFFORT"
    [ "$BENCH_LIMIT" = 0 ] || args="$args --limit $BENCH_LIMIT"

    say "跑 $ds effort=$BENCH_EFFORT"
    # Timestamp the run so a stale report cannot be accepted as the current result.
    stamp="$LOG_DIR/.bench_${ds_dir}_${BENCH_EFFORT}.stamp"
    : > "$stamp"

    # shellcheck disable=SC2086
    "$ev_py" run_bench.py $args 2>&1 | tee "$bench_log" \
      || fail "run_bench.py 退出码非零，尾部输出在 $bench_log"

    score="$(awk '$1=="score"{v=$2} END{print v}' "$bench_log")"
    scored="$(awk '$1=="scored"{v=$2} END{print v}' "$bench_log")"
    wall="$(awk '$1=="wall"{v=$2} END{print v}' "$bench_log")"
    report="$(awk '$1=="report"{v=$2} END{print v}' "$bench_log")"
    [ -n "$score" ] || fail "$ds 没打出 score 行，看 $bench_log"
    # Compare nanosecond mtimes because shell -nt has only second-level precision.
    if [ -n "$report" ] && [ -f "$report" ]; then
      "$VENV_PY" - "$report" "$stamp" <<'PY' || fail "$report 比本轮开跑还旧；本轮结果无效，把对应 bench 目录清掉再跑"
import os, sys
sys.exit(0 if os.stat(sys.argv[1]).st_mtime_ns >= os.stat(sys.argv[2]).st_mtime_ns else 1)
PY
    fi
    rm -f "$stamp"
    pass "$ds score=$score scored=$scored wall=$wall"

    if [ -n "$scored" ] && [ -n "$wall" ]; then
      "$VENV_PY" - "$scored" "$wall" "$ds_dir" <<'PY' || true
import re, sys
n = float(re.sub(r"[^0-9.]", "", sys.argv[1]) or 0)
w = float(re.sub(r"[^0-9.]", "", sys.argv[2]) or 0)
if n > 0 and w > 0:
    per = w / n
    line = f"  每题约 {per:.0f} 秒"
    # Only GSM8K has a fixed full-dataset count here.
    if sys.argv[3] == "gsm8k":
        line += f"。GSM8K 全量 1319 题按这个速度约 {per * 1319 / 3600:.1f} 小时"
    print(line)
PY
    fi
    say "结果落在 $GLM53_ARTIFACT_ROOT/bench/${ds_dir}_${BENCH_EFFORT}/，reports 是分数，predictions 是每题的输入输出与 usage"
  done

  if [ "$BENCH_EFFORT" = max ] && [ "$BENCH_LIMIT" = 0 ]; then
    say "GSM8K 的 96.5 到 98.4 是 thinking 全开、1319 题全量下测的，单轮二项标准误 ±0.47pp。
       小于 ±1.3pp（2 西格玛）的差别按噪声看，也不要拿这两个基准去洗困惑度报出来的差异"
  fi
  if [ "$BENCH_EFFORT" != low ]; then
    say "low 之外的 effort 手册里没给这两个基准的参考值，跑出来的数只对本部署自己有意义"
  fi
  # evalscope uses few-shot GSM8K prompts measured at 557-639 tokens on A3,
  # while GPQA-Diamond remains below the 512-token streaming threshold.
  say "evalscope 的 gsm8k 是少样本，prompt 实测 557 到 639 token，过 512 门槛，走流式路径；
       gpqa_diamond 是 156 到 479 token，不过门槛。流式路径的数值判据仍然只有困惑度"
}

case "$STAGE" in
  corpus) stage_corpus ;;
  ppl)    stage_ppl ;;
  bench)  stage_bench ;;
  all)    stage_corpus; stage_ppl; stage_bench ;;
esac

echo "=== $TAG 完成 ==="
