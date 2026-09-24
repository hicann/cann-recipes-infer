#!/usr/bin/env bash
# Generate glm53.env for the GLM-5.3 single-card deployment and validate the delivery directory.
set -euo pipefail

# ==== Defaults; override them with the command-line options below. ====
# Ascend installation root.
ASCEND_INSTALL_ROOT="/home/developer/Ascend"
# Workspace containing sources, virtual environments, logs, and the generated config.
GLM53_ROOT="${HOME}/glm53"
# Model root containing the GLM-5.3-Flash-W8A8 and GLM-5.3-Flash-MXFP4 directories.
GLM53_MODEL_ROOT=""
# integration/sglang/glm53-flash-single-npu-moe-offload in cann-recipes-infer.
# Leave empty to use the parent directory of this script.
GLM53_PATCH_DIR=""
# HTTP service port.
GLM53_PORT=30013
# Interpreter used to create the virtual environment; accepts a command or absolute path.
GLM53_PYTHON_BIN=python3
# Evaluation corpus directory; empty uses $GLM53_ROOT/env/eval.
GLM53_EVAL_DIR=""
# ==== End defaults ====

usage() {
  cat <<'USAGE'
用法
  bash -l 01-config.sh [选项]

选项
  --root DIR         工作目录，默认 $HOME/glm53
  --model-root DIR   权重根目录，默认 <root>/models
  --python-bin CMD   建虚拟环境的解释器，默认 python3
  --port PORT        HTTP 服务端口，默认 30013
  --eval-dir DIR     评测语料目录，默认 <root>/env/eval
  --patch-dir DIR    本交付目录；默认按脚本位置推导
  -h, --help         显示帮助
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)       GLM53_ROOT="${2:?--root 后面要跟目录}"; shift 2 ;;
    --model-root) GLM53_MODEL_ROOT="${2:?--model-root 后面要跟目录}"; shift 2 ;;
    --python-bin) GLM53_PYTHON_BIN="${2:?--python-bin 后面要跟命令或路径}"; shift 2 ;;
    --port)       GLM53_PORT="${2:?--port 后面要跟端口}"; shift 2 ;;
    --eval-dir)   GLM53_EVAL_DIR="${2:?--eval-dir 后面要跟目录}"; shift 2 ;;
    --patch-dir)  GLM53_PATCH_DIR="${2:?--patch-dir 后面要跟目录}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[配置] FAIL 不认识的参数 $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$GLM53_MODEL_ROOT" ] || GLM53_MODEL_ROOT="${GLM53_ROOT}/models"

# Derive the delivery directory from the parent of guide-scripts when unset.
if [ -z "$GLM53_PATCH_DIR" ]; then
  GLM53_PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd -P)"
fi

TAG="[配置]"
say()  { printf '%s %s\n' "$TAG" "$*"; }
pass() { printf '%s PASS %s\n' "$TAG" "$*"; }
fail() { printf '%s FAIL %s\n' "$TAG" "$*" >&2; exit 1; }
# An unmatched glob stays literal, so -e rejects it.
has_glob() { [ -e "$1" ]; }

# ---- Check 1: login shell ----
# Some images add libpython for a shared Python build to LD_LIBRARY_PATH only in
# the login profile. Without it, the step-5 CMake probe hides the ImportError and
# reports only "failed to get ENVS like TORCH_DIR".
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
  # $pcmd is intentionally word-split; disable pathname expansion while doing so.
  set -f
  # Inspect only the leading argv options and stop at the first non-option. Scanning
  # the full zsh -c body could mistake an unrelated command such as ls -l for evidence.
  _argv0=1
  for w in $pcmd; do
    if [ "$_argv0" = 1 ]; then
      _argv0=0
      # argv[0] only confirms that the parent is a shell, including forms such as -bash.
      case "${w##*/}" in
        -*|*sh) continue ;;
        *) break ;;
      esac
    fi
    case "$w" in
      --login) login_ok=1 ;;
      --) break ;;
      # Exclude long options such as --norc and --noprofile before checking short flags.
      --*) ;;
      -*l*) login_ok=1 ;;
      -*) ;;
      # The first non-option starts the script name and its arguments.
      *) break ;;
    esac
  done
  unset _argv0
  set +f
fi
[ "$login_ok" = 1 ] || fail "不是登录 shell。改用 bash -l \"$0\" 重跑，本文所有步骤都要这样跑"
pass "登录 shell"

# ---- Write the configuration file ----
# Normalize paths before later string concatenation and comparisons.
for _v in GLM53_ROOT GLM53_MODEL_ROOT GLM53_PATCH_DIR; do
  eval "_t=\$$_v"
  _t="$(printf '%s' "$_t" | sed 's#//*#/#g')"; _t="${_t%/}"
  # Reject empty, relative, and root-only paths after normalization.
  case "$_t" in
    /?*) ;;
    *) fail "$_v 要填绝对路径，现在压平后是 \"$_t\"，回到脚本顶部改这一项" ;;
  esac
  eval "$_v=\"\$_t\""
done
unset _v _t

mkdir -p "$GLM53_ROOT"
GLM53_ENV_FILE="$GLM53_ROOT/glm53.env"

if [ -f "$GLM53_ENV_FILE" ]; then
  _bak="$GLM53_ENV_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  cp -a "$GLM53_ENV_FILE" "$_bak"
  say "旧配置已备份到 $_bak"
fi

# Preserve any LD_LIBRARY_PATH appended by step 3 across the full-file rewrite.
KEEP_LD="$(grep '^export LD_LIBRARY_PATH=' "$GLM53_ENV_FILE" 2>/dev/null || true)"

cat > "$GLM53_ENV_FILE" <<EOF
# ============ Step-1 configuration ============
export ASCEND_INSTALL_ROOT="$ASCEND_INSTALL_ROOT"
source "\$ASCEND_INSTALL_ROOT/ascend-toolkit/set_env.sh"
export GLM53_ROOT="$GLM53_ROOT"
export GLM53_MODEL_ROOT="$GLM53_MODEL_ROOT"
export GLM53_PATCH_DIR="$GLM53_PATCH_DIR"
export GLM53_PORT="$GLM53_PORT"
export GLM53_PYTHON_BIN="$GLM53_PYTHON_BIN"
export GLM53_ENV_FILE="$GLM53_ENV_FILE"
EOF

if [ -n "${GLM53_EVAL_DIR:-}" ]; then
  GLM53_EVAL_DIR="$(printf '%s' "$GLM53_EVAL_DIR" | sed 's#//*#/#g')"; GLM53_EVAL_DIR="${GLM53_EVAL_DIR%/}"
  echo "export GLM53_EVAL_DIR=\"$GLM53_EVAL_DIR\"" >> "$GLM53_ENV_FILE"
fi

cat >> "$GLM53_ENV_FILE" <<'EOF'
# ============ Values derived from the configuration above ============
export GLM53_ARTIFACT_ROOT="$GLM53_ROOT/artifact"            # Logs and build trees, about 10 GiB.
export KT_REPO="$GLM53_ROOT/ktransformers"
export SGLANG_REPO="$GLM53_ROOT/sglang"
export GLM53_ENV_ROOT="$GLM53_ROOT/env"
export GLM53_VENV="$GLM53_ENV_ROOT/.venv-glm53"

# The standard configuration uses streaming prefill and dynamic hot experts.
export GLM53_PREFILL_STREAM=1     # -> MEM_FRACTION 0.95, MAX_TOTAL_TOKENS 40960,
                                  #    CHUNKED_PREFILL_SIZE 6144, KT_DYNAMIC_RESIDENT 1

# Logical device index inside the container, not the Phy-ID shown by npu-smi.
export GLM53_NPU_DEVICE_ID=0

# A3 SoC identifier.
export GLM53_SOC=ascend910_93

# Required.
export PATH="$GLM53_VENV/bin:$PATH"
EOF

if [ -n "$KEEP_LD" ]; then
  # Step 3 may append a bare $LD_LIBRARY_PATH. Add its nounset-safe default before
  # restoring every saved line; the substitution is idempotent.
  KEEP_LD="$(printf '%s\n' "$KEEP_LD" \
       | sed -e 's#\${LD_LIBRARY_PATH}#${LD_LIBRARY_PATH:-}#g' \
             -e 's#\$LD_LIBRARY_PATH#${LD_LIBRARY_PATH:-}#g')"
  printf '%s\n' "$KEEP_LD" >> "$GLM53_ENV_FILE"
  # Print a possibly multiline KEEP_LD value one line at a time.
  while IFS= read -r _ld; do
    if [ -n "$_ld" ]; then say "已保留第 3 步追加的 $_ld"; fi
  done <<INNER
$KEEP_LD
INNER
  unset _ld
fi

# shellcheck source=/dev/null
source "$GLM53_ENV_FILE"
mkdir -p "$GLM53_ENV_ROOT" "$GLM53_ARTIFACT_ROOT" "$GLM53_ARTIFACT_ROOT/logs"

echo "配置文件已写到 $GLM53_ENV_FILE"
echo "  GLM53_ROOT       = $GLM53_ROOT"
echo "  GLM53_MODEL_ROOT = $GLM53_MODEL_ROOT"
echo "  GLM53_PATCH_DIR  = $GLM53_PATCH_DIR"
echo "  GLM53_PORT       = $GLM53_PORT"
pass "配置文件已写入并能 source"

# Create the convenience symlink before later validation can exit the script.
ln -sfn "$GLM53_ENV_FILE" "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)/glm53.env"
pass "脚本目录下的 glm53.env 已指向 $GLM53_ENV_FILE"

# ---- Check 2: delivery directory ----
[ -d "$GLM53_PATCH_DIR" ] || fail "GLM53_PATCH_DIR 不存在 ${GLM53_PATCH_DIR}，回到脚本顶部改这一项"
_miss=""
for f in requirements.txt ktransformers-glm53-single-card.patch sglang-patches ktransformers-scripts; do
  [ -e "$GLM53_PATCH_DIR/$f" ] || _miss="$_miss $f"
done
[ -z "$_miss" ] || fail "GLM53_PATCH_DIR 下缺少${_miss}，这个目录多半指错了层级"
pass "交付目录含 requirements.txt、ktransformers-glm53-single-card.patch、sglang-patches、ktransformers-scripts"

# ---- Check 3: all ten ordered SGLang patches ----
_sgd="$GLM53_PATCH_DIR/sglang-patches"
[ -d "$_sgd" ] || fail "$_sgd 不是目录，这个目录多半指错了层级"
# Let the explicit count check report an unreadable directory instead of pipefail.
_n=$({ find "$_sgd" -mindepth 1 -maxdepth 1 -name '*.patch' 2>/dev/null || true; } | wc -l | tr -d ' ')
[ "$_n" -eq 10 ] || fail "sglang-patches 里应有 10 份 .patch，实际 $_n 份"
for i in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010; do
  has_glob "$_sgd/$i"-*.patch || fail "sglang-patches 里没有 $i 开头的补丁"
done
pass "sglang-patches 里 0001 到 0010 共 10 份"

# ---- Check 4: deployment script directory ----
_ad="$GLM53_PATCH_DIR/ktransformers-scripts/ascend_glm53"
[ -d "$_ad" ] || fail "没有目录 $_ad"
# Check required files individually while allowing extra documentation and helpers.
# Include .gitignore explicitly because wildcard copies silently omit hidden files.
_miss=""
for f in .gitignore README.md setup.sh serve.sh verify.sh glm53_env.sh \
         ask.sh bench.sh run_ppl.py run_bench.py \
         sgl-kernel-npu-ge-header-guards.patch; do
  [ -f "$_ad/$f" ] || _miss="$_miss $f"
done
[ -z "$_miss" ] || fail "ascend_glm53 下缺少${_miss}"
pass "ascend_glm53 的部署、验证、精度和吞吐文件齐全，含 .gitignore"

# ---- Check 5: executable bits, which archive extraction may drop ----
_noexec=""
for f in setup.sh serve.sh verify.sh glm53_env.sh ask.sh bench.sh run_ppl.py run_bench.py; do
  [ -f "$_ad/$f" ] || fail "ascend_glm53 下缺少 $f"
  [ -x "$_ad/$f" ] || _noexec="$_noexec $_ad/$f"
done
if [ -n "$_noexec" ]; then
  printf '%s FAIL 这几个文件没有可执行位\n' "$TAG" >&2
  printf '%s      修复后重跑本脚本  chmod 755%s\n' "$TAG" "$_noexec" >&2
  exit 1
fi
pass "部署、服务、验证、问答、吞吐和精度脚本均可执行"

# ---- Non-blocking warnings ----
if ! command -v "$GLM53_PYTHON_BIN" >/dev/null 2>&1; then
  say "注意 找不到解释器 ${GLM53_PYTHON_BIN}，第 3 步建虚拟环境会失败"
fi
if command -v ss >/dev/null 2>&1; then
  # Capture all ss output before matching; grep -q would close the pipe early and
  # make ss exit 141 under pipefail.
  _ss="$(ss -ltn 2>/dev/null || true)"
  case "$_ss" in
    *":$GLM53_PORT "*|*":$GLM53_PORT"$'\t'*)
      say "注意 端口 $GLM53_PORT 已被占用，不是自己的 glm53 服务就回到脚本顶部换一个" ;;
  esac
  unset _ss
fi

say "在本目录加载脚本生成的内部运行环境，之后执行 02 到 08"
say "  source ./glm53.env"
echo "=== 配置 完成 ==="
