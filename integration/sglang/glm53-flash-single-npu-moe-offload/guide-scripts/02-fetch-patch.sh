#!/usr/bin/env bash
# Fetch ktransformers and SGLang, apply patches in order, copy deployment scripts, and run seven checks.
set -euo pipefail

# ==== Edit here ====
# Upstream baseline commits used by every validation step. Update SG_TREE and
# KT_STAT together when changing either baseline.
KT_BASE=6d460cc10780f2e0ad541b5d58ad28086dd32ef0
SG_BASE=5aab054ec8ce6b6100fbfb7aafe67d632a7df3aa
# Expected SGLang tree after all ten patches.
SG_TREE=f34d08b7f3a447034f2f83cdc69a236b605142b9
# Count only files changed by the patch, excluding the copied ascend_glm53 tree.
KT_STAT="3 files changed, 211 insertions(+), 70 deletions(-)"
# Upstream URLs; replace them when using internal mirrors.
KT_REMOTE=https://github.com/kvcache-ai/ktransformers.git
SG_REMOTE=https://github.com/sgl-project/sglang.git
# ==== End editable values ====

TAG="[取码打补丁]"
FAILED=0

say()  { echo "$TAG $*"; }
pass() { echo "$TAG PASS  $*"; }
fail() { echo "$TAG FAIL  $*"; FAILED=1; }
die()  { echo "$TAG FAIL  $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
用法
  bash -l 02-fetch-patch.sh              取码、打补丁、复制脚本、校验
  bash -l 02-fetch-patch.sh --fresh      先删掉两个仓库再从头来一遍
  bash -l 02-fetch-patch.sh --verify-only  不重新取码打补丁，只跑子模块检查与 7 项校验
USAGE
}

FRESH=0
VERIFY_ONLY=0
for _a in "$@"; do
  case "$_a" in
    --fresh)       FRESH=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "未知参数 $_a" ;;
  esac
done

# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"

# ---- Login shell ----
# Some images add libpython for a shared Python build to LD_LIBRARY_PATH only in
# the login profile. Fail here instead of surfacing a misleading CMake error later.
login_ok=0
if shopt -q login_shell; then
  login_ok=1
else
  # A script launched with bash is not itself a login shell, so inspect its parent.
  if [ -r "/proc/$PPID/cmdline" ]; then
    pcmd="$(tr '\0' ' ' < "/proc/$PPID/cmdline" 2>/dev/null || true)"
  else
    pcmd="$(ps -o args= -p "$PPID" 2>/dev/null || true)"
  fi
  # login(1) and sshd prefix a login shell's argv[0], for example -bash.
  case "$pcmd" in -*) login_ok=1 ;; esac
  for _w in $pcmd; do
    case "$_w" in --login|-[a-zA-Z]*l*) login_ok=1 ;; esac
  done
fi
[ "$login_ok" = 1 ] || die "不是登录 shell。改用 bash -l \"$0\" 重跑，本文所有步骤都要这样跑"
pass "登录 shell"

# ---- Variable and directory prechecks ----
: "${GLM53_ROOT:?glm53.env 里没有 GLM53_ROOT，回第 1 节重写配置}"
: "${KT_REPO:?glm53.env 里没有 KT_REPO，回第 1 节重写配置}"
: "${SGLANG_REPO:?glm53.env 里没有 SGLANG_REPO，回第 1 节重写配置}"
: "${GLM53_PATCH_DIR:?glm53.env 里没有 GLM53_PATCH_DIR，回第 1 节重写配置}"

for _p in "$KT_REPO" "$SGLANG_REPO"; do
  case "$_p" in
    /|/[a-z]|"") die "仓库路径不安全（$_p），检查 glm53.env 里的 GLM53_ROOT" ;;
    /*) ;;
    *) die "仓库路径不是绝对路径（$_p）" ;;
  esac
done

command -v git >/dev/null 2>&1 || die "没有 git"

KT_PATCH="$GLM53_PATCH_DIR/ktransformers-glm53-single-card.patch"
KT_SCRIPTS="$GLM53_PATCH_DIR/ktransformers-scripts/ascend_glm53"
[ -f "$KT_PATCH" ]   || die "找不到 $KT_PATCH，GLM53_PATCH_DIR 填错了，回第 1 节改"
[ -d "$KT_SCRIPTS" ] || die "找不到 $KT_SCRIPTS，GLM53_PATCH_DIR 填错了，回第 1 节改"

shopt -s nullglob
SG_PATCHES=( "$GLM53_PATCH_DIR"/sglang-patches/0*.patch )
shopt -u nullglob
[ "${#SG_PATCHES[@]}" -eq 10 ] || die "sglang-patches 下应当有 10 份补丁，实际 ${#SG_PATCHES[@]} 份"

# Archive extraction may drop executable bits; validate the source before cp -a.
for _f in setup.sh serve.sh verify.sh glm53_env.sh ask.sh bench.sh run_ppl.py run_bench.py; do
  [ -x "$KT_SCRIPTS/$_f" ] || die "$KT_SCRIPTS/$_f 没有可执行位，先跑 chmod +x $KT_SCRIPTS/*.sh $KT_SCRIPTS/run_*.py"
done

LOG_DIR="${GLM53_ARTIFACT_ROOT:-$GLM53_ROOT/artifact}/logs"
mkdir -p "$GLM53_ROOT" "$LOG_DIR"

# ktransformers and SGLang metadata are small, but the non-shallow llama.cpp
# submodule grows $KT_REPO/.git to about 500 MiB. Keep || true so a failed df
# reaches the explicit skip branch under pipefail.
_avail_kb="$(df -Pk "$GLM53_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
case "${_avail_kb:-}" in
  ''|*[!0-9]*) say "读不到 $GLM53_ROOT 的剩余空间，跳过磁盘检查" ;;
  *) [ "$_avail_kb" -ge 2097152 ] || die "$GLM53_ROOT 只剩 $((_avail_kb / 1024)) MiB，不够 2 GiB，子模块那一步会失败" ;;
esac

say "工作目录 $GLM53_ROOT"
say "交付目录 $GLM53_PATCH_DIR"

# ---- Fetch sources and apply patches ----
prepare_repo() {
  # $1 repository, $2 remote URL, $3 baseline commit, $4 branch name.
  _repo="$1"; _url="$2"; _base="$3"; _branch="$4"
  git init -q "$_repo"
  git -C "$_repo" remote remove origin 2>/dev/null || true
  git -C "$_repo" remote add origin "$_url"
  say "浅克隆 $_url @ ${_base:0:12}"
  git -C "$_repo" fetch --depth 1 origin "$_base"
  # Return to the baseline before rerunning so --3way does not see old changes.
  git -C "$_repo" reset -q --hard FETCH_HEAD
  git -C "$_repo" checkout -q -B "$_branch" FETCH_HEAD
  # Remove untracked files from the previous apply while preserving ignored files.
  git -C "$_repo" clean -fdq
}

apply_patch() {
  # $1 repository, $2 patch file.
  _repo="$1"; _patch="$2"
  _name="$(basename "$_patch")"
  _log="$LOG_DIR/apply-$_name.log"
  # Delivery patches are generated with zero context so the repository's outer
  # whitespace checker does not mistake unified-diff context prefixes for trailing
  # whitespace. Both repositories are pinned to exact commits, so line context is not
  # needed; --unidiff-zero is required for git apply to accept that representation.
  if git -C "$_repo" apply --3way --unidiff-zero "$_patch" >"$_log" 2>&1; then
    # Direct-application fallback is expected for new files, gitlinks, and
    # zero-context hunks. Record the count for diagnostics only.
    _fb="$(grep -c 'Falling back to direct application' "$_log" 2>/dev/null || true)"
    say "已应用 $_name（Falling back 提示 ${_fb:-0} 行，正常）"
  else
    echo "$TAG FAIL  $_name 应用失败，日志 $_log" >&2
    tail -20 "$_log" >&2
    exit 1
  fi
}

if [ "$VERIFY_ONLY" = 0 ]; then
  if [ "$FRESH" = 1 ]; then
    say "--fresh 先删掉 $KT_REPO 和 $SGLANG_REPO"
    # rm -rf removes the symlink without following it into SGLANG_REPO.
    rm -rf "$KT_REPO" "$SGLANG_REPO"
  fi

  # Remove the previous symlink before reset restores the gitlink directory.
  rm -rf "$KT_REPO/third_party/sglang" 2>/dev/null || true

  prepare_repo "$KT_REPO" "$KT_REMOTE" "$KT_BASE" glm53-single-card
  apply_patch  "$KT_REPO" "$KT_PATCH"

  # Copy the full script directory with executable bits and hidden files intact.
  # Remove the destination first to avoid creating a nested ascend_glm53 directory.
  rm -rf "$KT_REPO/kt-kernel/tools/ascend_glm53"
  mkdir -p "$KT_REPO/kt-kernel/tools"
  cp -a "$KT_SCRIPTS" "$KT_REPO/kt-kernel/tools/"
  say "已复制 ascend_glm53 到 kt-kernel/tools/"

  prepare_repo "$SGLANG_REPO" "$SG_REMOTE" "$SG_BASE" glm53-npu-offload
  # Apply all ten patches in filename order because later patches have dependencies.
  for _p in "${SG_PATCHES[@]}"; do
    apply_patch "$SGLANG_REPO" "$_p"
  done

  # Initialize only the two required unpatched submodules. An unscoped update would
  # also materialize third_party/sglang and break the symlink created below.
  say "初始化 llama.cpp 与 pybind11 子模块（.git 会涨到约 500 MiB，慢是正常的）"
  git -C "$KT_REPO" submodule update --init third_party/llama.cpp third_party/pybind11

  # Replace the checked-out gitlink directory before linking SGLang into ktransformers.
  rm -rf "$KT_REPO/third_party/sglang"
  ln -s "$SGLANG_REPO" "$KT_REPO/third_party/sglang"
  say "third_party/sglang -> $SGLANG_REPO"
fi

# ---- Seven validation checks ----
# Run before building because generated artifacts would affect git add -A and tree statistics.
say "开始校验"

[ -d "$KT_REPO/.git" ]     || die "$KT_REPO 还不是 git 仓库，先不带 --verify-only 跑一遍"
[ -d "$SGLANG_REPO/.git" ] || die "$SGLANG_REPO 还不是 git 仓库，先不带 --verify-only 跑一遍"

# Use submodule status because rev-parse in an uninitialized empty directory can
# walk up to the parent repository and falsely succeed. Keep this in validation
# so --verify-only detects incomplete previous runs.
_subm="$(git -C "$KT_REPO" submodule status third_party/llama.cpp third_party/pybind11 2>/dev/null || true)"
# Reject empty output before the loop to avoid a false pass.
[ -n "$_subm" ] || die "读不到子模块状态，$KT_REPO 多半没取完，重来一遍 bash -l $(basename "$0") --fresh"
_subm_n=0
while IFS= read -r _line; do
  [ -n "$_line" ] || continue
  _subm_n=$((_subm_n + 1))
  case "$_line" in
    " "*) say "子模块 ${_line# }" ;;
    "-"*) die "子模块没初始化 -> $_line" ;;
    "+"*) die "子模块检出的提交和 gitlink 对不上 -> $_line" ;;
    *)    die "子模块状态异常 -> $_line" ;;
  esac
done <<EOF
$_subm
EOF
[ "$_subm_n" -eq 2 ] || die "子模块状态应当是 2 行，实际 $_subm_n 行 -> $_subm"

# Compare full 40-character commit IDs because abbreviated lengths vary by repository.
# Keep || true so all checks and the final failure summary still run under set -e.
_kt_head="$(git -C "$KT_REPO" rev-parse HEAD 2>/dev/null || true)"
if [ "$_kt_head" = "$KT_BASE" ]; then
  pass "ktransformers HEAD = $KT_BASE"
else
  fail "ktransformers HEAD（实际 ${_kt_head:-读不出来}）"
fi

_sg_head="$(git -C "$SGLANG_REPO" rev-parse HEAD 2>/dev/null || true)"
if [ "$_sg_head" = "$SG_BASE" ]; then
  pass "sglang HEAD = $SG_BASE"
else
  fail "sglang HEAD（实际 ${_sg_head:-读不出来}）"
fi

# Compare the patched SGLang tree with the expected tree.
git -C "$SGLANG_REPO" add -A || true
_tree="$(git -C "$SGLANG_REPO" write-tree 2>/dev/null || true)"
if [ "$_tree" = "$SG_TREE" ]; then
  pass "sglang 树 = $SG_TREE"
else
  fail "sglang 树（实际 ${_tree:-读不出来}），重来一遍 bash -l $(basename "$0") --fresh"
fi

# Compare patch statistics while excluding third_party and the copied ascend_glm53 tree.
git -C "$KT_REPO" add -A || true
_stat="$(git -C "$KT_REPO" diff --stat "$KT_BASE" -- . ':!third_party' ':!kt-kernel/tools/ascend_glm53' 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' || true)"
if [ "$_stat" = "$KT_STAT" ]; then
  pass "补丁改动量 = $KT_STAT"
else
  fail "补丁改动量（实际 ${_stat:-读不出来，KT_BASE 在这个仓库里不存在}）"
fi

# Compare every copied script byte-for-byte with its source.
_src="$GLM53_PATCH_DIR/ktransformers-scripts/ascend_glm53"
_dst="$KT_REPO/kt-kernel/tools/ascend_glm53"
_bad=""
for _f in $(cd "$_src" && ls -A); do
  if ! cmp -s "$_src/$_f" "$_dst/$_f"; then _bad="$_bad $_f"; fi
done
if [ -z "$_bad" ]; then
  pass "部署脚本与 cann-recipes-infer 里的原件逐字节一致"
else
  fail "这几个文件和原件不一致，重跑本节补上 ->$_bad"
fi

if [ -x "$KT_REPO/kt-kernel/tools/ascend_glm53/setup.sh" ]; then
  pass "脚本可执行位"
else
  fail "脚本可执行位丢了，chmod +x $KT_SCRIPTS/*.sh $KT_SCRIPTS/run_*.py 之后重跑"
fi

if [ -L "$KT_REPO/third_party/sglang" ]; then
  pass "third_party/sglang 是符号链接"
else
  fail "third_party/sglang 不是符号链接，多半是 rm -rf 那步没生效"
fi

if [ -f "$KT_REPO/third_party/sglang/python/sglang/__init__.py" ]; then
  pass "符号链接可解析"
else
  fail "符号链接可解析"
fi

if [ "$FAILED" != 0 ]; then
  echo "$TAG 有 FAIL，先照上面的提示修，别往下构建" >&2
  exit 1
fi

echo "=== 取码打补丁 完成 ==="
