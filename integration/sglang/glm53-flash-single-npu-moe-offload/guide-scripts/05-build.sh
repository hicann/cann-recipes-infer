#!/usr/bin/env bash
# Validate configuration, run the five build stages, repair deep_ep's .pth, and run preflight.
set -euo pipefail

# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"

# ==== Edit here ====
STEP="${1:-all}"                 # all, precheck, deps, sgl-kernel, cann-ops, kt-kernel, gguf, deep-ep, or check.
EXPECT_NUM_GPU_EXPERTS=32        # Expected GLM53_NUM_GPU_EXPERTS from glm53_env.sh --show.
RUN_PROBE=1                      # Set to 0 to skip setup.sh probe, which opens die 0.
# ==== End editable values ====

TAG="[构建]"
TOOLS="${KT_REPO:?配置文件里没有 KT_REPO，回第 1 节重写 glm53.env}/kt-kernel/tools/ascend_glm53"
LOGDIR="${GLM53_ARTIFACT_ROOT:?配置文件里没有 GLM53_ARTIFACT_ROOT，回第 1 节重写 glm53.env}/logs"
SGL_TREE="$(dirname "$KT_REPO")/sgl-kernel-npu"

log()  { echo "$TAG $(date '+%F %T') $*"; }
pass() { echo "$TAG $(date '+%F %T') PASS $*"; }
fail() { echo "$TAG $(date '+%F %T') FAIL $*" >&2; }
die()  { fail "$*"; echo "=== 构建 失败 ===" >&2; exit 1; }
dur()  { printf '%d 分 %d 秒' $(( $1 / 60 )) $(( $1 % 60 )); }
plain() { sed $'s/\033\\[[0-9;]*m//g' "$1"; }  # Strip color codes before matching green ok lines.

# ---------- Login shell ----------
# Some images add libpython for a shared Python build to LD_LIBRARY_PATH only in
# the login profile. Without it, the sgl-kernel CMake probe reports a misleading error.
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if ! shopt -q login_shell; then
  if [ "${GLM53_RELOGIN:-0}" = 1 ]; then
    die "bash -l 重入之后仍不是登录 shell，请手动执行 bash -l \"$_SELF\" $STEP"
  fi
  log "当前不是登录 shell，改用 bash -l 重新执行本脚本"
  export GLM53_RELOGIN=1
  exec bash -l "$_SELF" ${1:+"$@"}
fi
pass "登录 shell"

# ---------- Preconditions ----------
# These paths must be absolute because later cleanup derives targets from KT_REPO.
for _v in KT_REPO SGLANG_REPO GLM53_VENV GLM53_ARTIFACT_ROOT; do
  eval "_t=\${$_v:-}"
  [ -n "$_t" ] || die "配置文件里没有 $_v，回第 1 节重写 $GLM53_ENV_FILE"
  case "$_t" in
    /*) ;;
    *) die "$_v 是相对路径 $_t，回第 1 节改成绝对路径" ;;
  esac
done
unset _v _t
[ -x "$TOOLS/setup.sh" ] || die "$TOOLS/setup.sh 不在或没有可执行位，回第 2 节确认取码、打补丁、复制脚本目录都成功了"
[ -f "$TOOLS/glm53_env.sh" ] || die "$TOOLS/glm53_env.sh 不在，回第 2 节确认取码、打补丁、复制脚本目录都成功了"
mkdir -p "$LOGDIR"

PY="$GLM53_VENV/bin/python"
[ -x "$PY" ] || die "$PY 不在，先跑第 3 节的脚本"

# The virtual-environment Python must import torch and torch_npu.
if ! "$PY" -c 'import ctypes, torch, torch_npu' >/dev/null 2>&1; then
  echo "$TAG   实际回显 $("$PY" -c 'import ctypes, torch, torch_npu' 2>&1 | tail -2)" >&2
  echo "$TAG   报 undefined symbol 时执行" >&2
  echo "$TAG   export LD_LIBRARY_PATH=\"\$($PY -c 'import sys; print(sys.base_prefix)')/lib:\$LD_LIBRARY_PATH\"" >&2
  die "虚拟环境的 python import torch 失败"
fi
pass "虚拟环境 python 能 import torch 与 torch_npu"

# bash -l reruns the image profile, so restore the project venv to the front of PATH.
PATH="${PATH//"$GLM53_VENV/bin:"/}"
PATH="$GLM53_VENV/bin:$PATH"
export PATH

# Vendor build subprocesses use bare python3, so require it to resolve inside the project venv.
_P3="$(command -v python3 2>/dev/null || true)"
case "$_P3" in
  "$GLM53_VENV/bin/"*) pass "python3 落在虚拟环境里 $_P3" ;;
  *) die "python3 是 ${_P3:-<没有>}，不在 $GLM53_VENV/bin 下。虚拟环境里没有 python3，回第 3 节重建" ;;
esac

# ---------- Five glm53_env.sh --show checks ----------
show_line() {
  local v="$1" f="$2" l
  l="$(grep -E "^[[:space:]]*$v([[:space:]]|=|:)" "$f" | tail -1 || true)"
  [ -n "$l" ] || l="$(grep -E "(^|[^A-Za-z0-9_])$v([^A-Za-z0-9_]|$)" "$f" | tail -1 || true)"
  printf '%s' "$l"
}
show_val() {
  show_line "$1" "$2" \
    | sed -E "s/^.*$1[[:space:]]*[:=]?[[:space:]]*//" \
    | sed -E 's/[[:space:]]*\*\*\* MISSING \*\*\*[[:space:]]*$//' \
    | sed -E 's/[[:space:]]+$//'
}

do_show() {
  local out="$LOGDIR/glm53_env-show.log" bad=0 v line val rc=0 t0=$SECONDS
  log "核对配置 bash glm53_env.sh --show"
  # Run all five checks even when --show fails so configuration errors are reported together.
  if ! ( cd "$TOOLS" && bash glm53_env.sh --show ) >"$out" 2>&1; then rc=1; fi
  cat "$out"
  [ -s "$out" ] || die "glm53_env.sh --show 一行输出都没有，日志 $out，回第 2 节确认脚本目录复制完整"
  if [ "$rc" != 0 ]; then fail "glm53_env.sh --show 退出码不是 0，下面五条判据照跑，但结果不可尽信"; bad=1; fi

  # Validate weight paths before the long build stages.
  for v in GLM53_MODEL_PATH GLM53_MXFP4_CKPT; do
    line="$(show_line "$v" "$out")"
    if [ -z "$line" ]; then
      fail "--show 输出里找不到 $v 这一行，自己看 $out"; bad=1
    elif printf '%s' "$line" | grep -q '\*\*\* MISSING \*\*\*'; then
      fail "$v 行尾是 *** MISSING ***，权重路径不对，回第 1 节改 GLM53_MODEL_ROOT，不要往下构建"; bad=1
    else
      pass "$v $(show_val "$v" "$out")"
    fi
  done

  # --show appends capacity text after the value, so compare only the first field.
  read -r val _ <<<"$(show_val GLM53_NUM_GPU_EXPERTS "$out")"
  if [ "$val" = "$EXPECT_NUM_GPU_EXPERTS" ]; then pass "GLM53_NUM_GPU_EXPERTS $val"
  else fail "GLM53_NUM_GPU_EXPERTS 是 ${val:-<空>}，期望 $EXPECT_NUM_GPU_EXPERTS"; bad=1; fi

  val="$(show_val SGLANG_REPO "$out")"
  if [ "$val" = "<NOT FOUND>" ] || [ -z "$val" ]; then
    fail "SGLANG_REPO 是 ${val:-<空>}，第 2 节的 sglang 树没建起来或者软链接断了"; bad=1
  elif [ ! -f "$val/python/sglang/__init__.py" ]; then
    fail "SGLANG_REPO 指向 $val，里面没有 python/sglang/__init__.py"; bad=1
  else pass "SGLANG_REPO $val"; fi

  val="$(show_val GLM53_PYTHON "$out")"
  if [ "$val" = "$PY" ]; then pass "GLM53_PYTHON $val"
  else fail "GLM53_PYTHON 是 ${val:-<空>}，期望 $PY"; bad=1; fi

  val="$(show_val GLM53_LOG_DIR "$out")"
  if [ "$val" = "$LOGDIR" ]; then pass "GLM53_LOG_DIR $val"
  else fail "GLM53_LOG_DIR 是 ${val:-<空>}，期望 $LOGDIR，第 6 节 tail 的就是这个目录"; bad=1; fi

  [ "$bad" = 0 ] || die "--show 有对不上的项，回第 1 节改配置，改完重跑本脚本"
  pass "配置五条都对上了，用时 $(dur $(( SECONDS - t0 )))"
}

do_probe() {
  local out="$LOGDIR/setup-probe.log" t0=$SECONDS
  log "探测环境 setup.sh probe，它会打开 die 0"
  if ! ( cd "$TOOLS" && ./setup.sh probe ) >"$out" 2>&1; then
    cat "$out"
    grep -q 'ZSH_VERSION: unbound variable' "$out" \
      && die "probe 撞上 nnal 的 set_env.sh 未绑定变量，第 2 节的 ktransformers 补丁没打上"
    die "setup.sh probe 失败，日志 $out"
  fi
  cat "$out"
  [ -s "$out" ] || die "probe 一行输出都没有就退出，日志 $out，多半是第 2 节的补丁没打上"
  # Missing components reported by probe are installed by later stages.
  pass "probe 完成，have $(grep -ci 'have' "$out" || true) 项，用时 $(dur $(( SECONDS - t0 )))"
}

# ---------- Build stages ----------
# Verify deps before later stages surface unrelated-looking failures.
require_deps_ok() {
  local dl="$LOGDIR/setup-deps.log"
  if [ ! -f "$dl" ]; then
    log "没有 $dl，本脚本没跑过 deps，核对不了。手动跑过 setup.sh deps 就继续，没跑过先跑 bash \"$_SELF\" deps"
    return 0
  fi
  grep -q 'python deps installed' "$dl" \
    || die "上一次 deps 没装成，先跑 bash \"$_SELF\" deps。在它坏的前提下重跑构建，多少次都是同一个错，日志 $dl"
}

# Install custom operators into the project environment rather than shared CANN.
check_vendor_target() {
  local target="${GLM53_ENV_ROOT}/opp_custom"
  mkdir -p "$target" || die "建不出自定义算子目录 $target"
  [ -w "$target" ] || die "自定义算子目录不可写 $target"
  pass "自定义算子安装目录可写 $target"
}

# sgl-kernel relinks AICore objects in place, so remove an incomplete source tree before rebuilding.
clean_sgl_tree() {
  if ( cd "$TOOLS" && ./setup.sh sgl-kernel --check-only ) >/dev/null 2>&1; then
    pass "sgl-kernel-npu 锁定版本与 API 已安装，不清理源码树"
    return 0
  fi
  if [ -e "$SGL_TREE" ]; then
    log "删掉上一次的 $SGL_TREE，这一步原地重跑会报 unknown file type"
    chmod -R u+w "$SGL_TREE" 2>/dev/null || true
    rm -rf "$SGL_TREE" || die "删不掉 $SGL_TREE，请确认它属于当前用户"
  fi
  pass "sgl-kernel-npu 树是干净的"
}

hint_sgl() {
  local f="$1"
  grep -q 'TORCH_DIR' "$f" && echo "$TAG   日志里有 failed to get ... TORCH_DIR，不是 torch 没装，是 shell 不对，重试多少次都没用" >&2
  grep -qE 'unable to access|Couldn.t connect to server' "$f" && echo "$TAG   克隆断了，重建中转隧道再跑 bash \"$_SELF\" sgl-kernel，别改构建参数" >&2
  grep -qE 'op_kernel/.*\.cpp|gmake.*Error [12]' "$f" && echo "$TAG   AscendC 算子编译偶发失败，直接重跑 bash \"$_SELF\" sgl-kernel，实测第二次即过" >&2
  grep -q 'json not generated' "$f" && echo "$TAG   json not generated 多半是 deps 没装成，先看 $LOGDIR/setup-deps.log 有没有 FAIL pip install failed" >&2
  return 0
}

hint_cann() {
  local f="$1"
  grep -q 'cannot determine the SoC' "$f" && echo "$TAG   当前 shell 没读到配置，确认 GLM53_ENV_FILE 指向 $GLM53_ENV_FILE" >&2
  grep -qE '\b2/3\b|\b3/3\b' "$f" && echo "$TAG   失败在后面的小步时可以只重跑那一步，cd $TOOLS 后 ./setup.sh cann-ops customize|custom_ops|transformer" >&2
  echo "$TAG   import custom_ops 失败不算故障，算子是 vendor 包注册的" >&2
  return 0
}

hint_gguf() {
  local f="$1"
  grep -q 'not enough space' "$f" && echo "$TAG   盘不够，换一个更大的卷，或在配置文件里设 GLM53_GGUF_DIR" >&2
  grep -q 'conversion failed' "$f" && echo "$TAG   看 $LOGDIR/gguf.log，转换带 --skip-existing，修好后重跑只补缺的那几层" >&2
  return 0
}

run_step() {
  local step="$1" what="$2" cost="$3" t0 rc=0
  local slog="$LOGDIR/setup-$step.log"
  # Complete stage prerequisites before announcing the build.
  case "$step" in
    sgl-kernel) require_deps_ok; clean_sgl_tree ;;
    cann-ops)   require_deps_ok; check_vendor_target ;;
    gguf)       require_deps_ok ;;
  esac
  t0=$SECONDS
  log "开始 $step（$what），预计 $cost，日志 $slog"
  if ( cd "$TOOLS" && ./setup.sh "$step" ) 2>&1 | tee "$slog"; then rc=0; else rc=1; fi
  if [ "$rc" != 0 ]; then
    case "$step" in
      sgl-kernel) hint_sgl "$slog" ;;
      cann-ops)   hint_cann "$slog" ;;
      gguf)       hint_gguf "$slog" ;;
      deps)       grep -q 'FAIL pip install failed' "$slog" && echo "$TAG   pip 装不上，先修网络或索引源，后面四步都会跟着坏" >&2 ;;
    esac
    die "$step 失败，用时 $(dur $(( SECONDS - t0 )))，日志 $slog"
  fi
  # The deps success criterion is the explicit "ok python deps installed" line.
  if [ "$step" = deps ]; then
    grep -q 'python deps installed' "$slog" || die "deps 退出码是 0 但没打 ok python deps installed，日志 $slog"
    log "deps 装好了，现在另开一个登录 shell 把第 8 节的评测语料放好，两分钟的事"
  fi
  pass "$step 完成，用时 $(dur $(( SECONDS - t0 )))"
}

# ---------- 5.1 deep_ep .pth ----------
do_deep_ep() {
  local S sl="$LOGDIR/setup-sgl-kernel.log" t0=$SECONDS
  S="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  if [ -f "$sl" ] && grep -q 'already provided' "$sl"; then
    log "sgl-kernel 那步使用了镜像内已锁定版本，通常不用补 .pth"
  fi
  if "$PY" -c 'import torch, torch_npu, deep_ep; print("deep_ep ok")' 2>/dev/null; then
    if [ -f "$S/_deep_ep_cpp_path.pth" ]; then
      pass "补完 .pth 后 deep_ep ok，用时 $(dur $(( SECONDS - t0 )))"
    else
      pass "deep_ep 直接可用，没补 .pth，用时 $(dur $(( SECONDS - t0 )))"
    fi
    return 0
  fi
  # A missing package directory indicates an earlier build failure, not a missing .pth.
  if [ ! -d "$S/deep_ep" ]; then
    echo "$TAG   实际回显 $("$PY" -c 'import deep_ep' 2>&1 | tail -1)" >&2
    echo "$TAG   $S 下面没有 deep_ep 目录，是上一步没产出，不是少了 .pth，看 $sl 与 $LOGDIR/setup-deps.log" >&2
    die "deep_ep 没装上，补 .pth 解决不了"
  fi
  # The locally built wheel needs a path file for its package-local shared library.
  log "import deep_ep 失败，补 $S/_deep_ep_cpp_path.pth"
  echo "$S/deep_ep" > "$S/_deep_ep_cpp_path.pth"
  if "$PY" -c 'import torch, torch_npu, deep_ep; print("deep_ep ok")'; then
    pass "补完 .pth 后 deep_ep ok，用时 $(dur $(( SECONDS - t0 )))"
  else
    echo "$TAG   实际回显 $("$PY" -c 'import deep_ep' 2>&1 | tail -1)" >&2
    echo "$TAG   报 No module named 'deep_ep' 说明上一步没产出，看 $sl 与 $LOGDIR/setup-deps.log" >&2
    die "deep_ep 仍然 import 不了"
  fi
}

# ---------- 5.2 Preflight ----------
do_check() {
  local out="$LOGDIR/setup-check.log" rc=0 n t0=$SECONDS
  log "启动前自检 setup.sh check"
  if ( cd "$TOOLS" && ./setup.sh check ) 2>&1 | tee "$out"; then rc=0; else rc=1; fi
  plain "$out" | grep -q 'host RAM.*is tight' && log "可忽略  warn host RAM is tight，是提醒不是错误"
  if plain "$out" | grep -q 'PREFLIGHT FAILED'; then
    plain "$out" | grep -q 'sglang resolves to' && echo "$TAG   sglang resolves to <none> 是 sglang 依赖缺失，先看 $LOGDIR/setup-deps.log" >&2
    die "PREFLIGHT FAILED，日志 $out"
  fi
  [ "$rc" = 0 ] || die "setup.sh check 退出码不是 0，日志 $out"
  plain "$out" | grep -q 'PREFLIGHT OK' || die "没看到 PREFLIGHT OK，日志 $out"
  # Count detail lines because setup.sh prints PREFLIGHT OK without a total.
  n="$(plain "$out" | grep -cE '^[[:space:]]*ok([[:space:]]|$)' || true)"
  pass "PREFLIGHT OK，ok 共 $n 行，用时 $(dur $(( SECONDS - t0 )))"
}

# ---------- Dispatch ----------
case "$STEP" in
  all)
    do_show
    if [ "$RUN_PROBE" = 1 ]; then do_probe; fi
    log "五步合计约 1.5 到 2.5 小时，中途别关这个 shell"
    run_step deps       "按 pyproject_npu.toml 装 51 个依赖" "几分钟"
    run_step sgl-kernel "sgl_kernel_npu / deep_ep / attentions / torch_memory_saver" "30 到 60 分钟"
    run_step cann-ops   "三个 CANN 自定义算子，打 1/3 2/3 3/3" "30 到 60 分钟"
    run_step kt-kernel  "编 kt-kernel wheel 并安装" "几分钟"
    run_step gguf       "MXFP4 转 42 层 GGUF 并逐位验证，约 151 GiB" "15 分钟"
    do_deep_ep
    do_check
    ;;
  precheck) do_show; if [ "$RUN_PROBE" = 1 ]; then do_probe; fi ;;
  deps)       do_show; run_step deps       "按 pyproject_npu.toml 装 51 个依赖" "几分钟" ;;
  sgl-kernel) do_show; run_step sgl-kernel "sgl_kernel_npu / deep_ep / attentions / torch_memory_saver" "30 到 60 分钟"; do_deep_ep ;;
  cann-ops)   do_show; run_step cann-ops   "三个 CANN 自定义算子，打 1/3 2/3 3/3" "30 到 60 分钟" ;;
  kt-kernel)  do_show; run_step kt-kernel  "编 kt-kernel wheel 并安装" "几分钟" ;;
  gguf)       do_show; run_step gguf       "MXFP4 转 42 层 GGUF 并逐位验证，约 151 GiB" "15 分钟" ;;
  deep-ep)  do_deep_ep ;;
  check)    do_check ;;
  *) die "不认识的步骤 $STEP，可用 all precheck deps sgl-kernel cann-ops kt-kernel gguf deep-ep check" ;;
esac

echo "=== 构建 完成 ==="
