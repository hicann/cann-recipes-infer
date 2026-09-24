#!/usr/bin/env bash
# Create the GLM-5.3 virtual environment, install dependencies, and copy cp312 triton_ascend.
set -euo pipefail

# The config may reference an unset $LD_LIBRARY_PATH, so disable nounset only while sourcing it.
set +u
# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
set -u

# ==== Edit here ====
TRITON_ASCEND_VER=3.2.2          # Package version; public Python 3.11 builds provide 3.2.0.
TRITON_VER=3.2.0                 # Upstream Triton version used by triton-ascend.
TRITON_MIN_MIB=800               # The validated cp312 package is about 971 MiB.
                                 # Set to 0 for a differently sized Python 3.11 pip build.
PIP_INDEX_OVERRIDE=""            # Empty keeps the pip configuration and standard PIP_INDEX_URL.
RECREATE_VENV=0                  # Set to 1 to recreate the project virtual environment.
# ==== End editable values ====

TRITON_SRC=""
usage() {
  cat <<'USAGE'
用法
  bash -l 03-python-env.sh [--triton-src DIR] [--recreate-venv]

  --triton-src DIR   含 triton/ 与 triton_ascend-*.dist-info 的 site-packages
  --recreate-venv    删除本项目已有虚拟环境后重建
  -h, --help         显示帮助
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --triton-src) TRITON_SRC="${2:?--triton-src 后面要跟目录}"; shift 2 ;;
    --recreate-venv) RECREATE_VENV=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[Python 环境] FAIL 不认识的参数 $1" >&2; usage >&2; exit 2 ;;
  esac
done

TAG="[Python 环境]"
FAILED=0

log()  { echo "$TAG $*"; }
pass() { echo "$TAG PASS $*"; }
fail() { echo "$TAG FAIL $*" >&2; FAILED=1; }
die()  { echo "$TAG FAIL $*" >&2; exit 1; }

# This is the only version source for torch, torch_npu, and direct dependencies.
# setup.sh also uses it as constraints when installing SGLang dependencies.
REQUIREMENTS_FILE="${GLM53_PATCH_DIR}/requirements.txt"
[ -f "$REQUIREMENTS_FILE" ] || die "找不到依赖锁定文件 $REQUIREMENTS_FILE"
locked_version() {
  sed -n "s/^${1}==//p" "$REQUIREMENTS_FILE"
}
TORCH_VER="$(locked_version torch)"
TORCH_NPU_VER="$(locked_version torch_npu)"
NUMPY_VER="$(locked_version numpy)"
PYYAML_VER="$(locked_version PyYAML)"
PYBIND11_VER="$(locked_version pybind11)"
DECORATOR_VER="$(locked_version decorator)"
SCIPY_VER="$(locked_version scipy)"
ATTRS_VER="$(locked_version attrs)"
PSUTIL_VER="$(locked_version psutil)"
SAFETENSORS_VER="$(locked_version safetensors)"
HUGGINGFACE_HUB_VER="$(locked_version huggingface_hub)"
MODELSCOPE_VER="$(locked_version modelscope)"
TRANSFORMERS_VER="$(locked_version transformers)"
PYARROW_VER="$(locked_version pyarrow)"
REQUESTS_VER="$(locked_version requests)"
GGUF_VER="$(locked_version gguf)"
for _v in TORCH_VER TORCH_NPU_VER NUMPY_VER PYYAML_VER PYBIND11_VER DECORATOR_VER \
          SCIPY_VER ATTRS_VER PSUTIL_VER SAFETENSORS_VER HUGGINGFACE_HUB_VER \
          MODELSCOPE_VER TRANSFORMERS_VER PYARROW_VER REQUESTS_VER GGUF_VER; do
  eval "_value=\${$_v:-}"
  [ -n "$_value" ] || die "$REQUIREMENTS_FILE 中缺少 $_v 对应的固定版本"
done
unset _v _value

PIP_ARGS=()
[ -n "$PIP_INDEX_OVERRIDE" ] && PIP_ARGS=(--index-url "$PIP_INDEX_OVERRIDE")

# ---------- 1. Login shell ----------
# Some images add libpython for a shared Python build to LD_LIBRARY_PATH only in
# the login profile. Without it, later probes report misleading import or CMake errors.
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if ! shopt -q login_shell; then
  if [ "${GLM53_RELOGIN:-0}" = 1 ]; then
    die "bash -l 重入之后仍不是登录 shell，请手动执行 bash -l $_SELF"
  fi
  log "当前不是登录 shell，改用 bash -l 重新执行本脚本"
  export GLM53_RELOGIN=1
  exec bash -l "$_SELF"
fi
pass "登录 shell"

# ---------- 2. Base interpreter ----------
# Remove the project venv from PATH while locating the base interpreter; otherwise
# a rerun can resolve python3.12 back into the venv that may be recreated.
_base_path=""
_old_ifs="$IFS"; IFS=:
for _d in $PATH; do
  [ -n "$_d" ] || continue
  [ "$_d" != "$GLM53_VENV/bin" ] || continue
  _base_path="${_base_path:+$_base_path:}$_d"
done
IFS="$_old_ifs"; unset _d _old_ifs

BASE_PY="$(PATH="$_base_path" command -v "$GLM53_PYTHON_BIN" 2>/dev/null || true)"
[ -n "$BASE_PY" ] \
  || die "虚拟环境之外找不到 ${GLM53_PYTHON_BIN}；通过 GLM53_PYTHON_BIN 指定解释器后重跑第 1 步"

# Validate the base interpreter before creating the venv.
if ! "$BASE_PY" -c 'import ctypes, sys' >/dev/null 2>&1; then
  _e="$("$BASE_PY" -c 'import ctypes' 2>&1 || true)"
  echo "$TAG   实际回显 $_e" >&2
  die "$BASE_PY 自身不可用；通过 GLM53_PYTHON_BIN 换一个解释器后重跑第 1 步"
fi
pass "基础解释器 $BASE_PY  $("$BASE_PY" -c 'import sys; print(sys.version.split()[0])')"

# ---------- 3. Virtual environment ----------
PY="$GLM53_VENV/bin/python"
if [ "$RECREATE_VENV" = 1 ] && [ -d "$GLM53_VENV" ]; then
  log "RECREATE_VENV=1，删掉 $GLM53_VENV 重建"
  rm -rf "$GLM53_VENV"
fi
if [ -x "$PY" ]; then
  pass "复用已有虚拟环境 $GLM53_VENV"
else
  mkdir -p "$GLM53_ENV_ROOT"
  "$BASE_PY" -m venv "$GLM53_VENV"
  pass "虚拟环境已建 $GLM53_VENV"
fi
[ -x "$PY" ] || die "$PY 不存在，venv 建失败"

# ---------- 4. Dependencies ----------
# Skip installation when every locked dependency is already present.
need_install=1
if "$PY" - "$TORCH_VER" "$TORCH_NPU_VER" "$NUMPY_VER" "$PYYAML_VER" \
  "$PYBIND11_VER" "$DECORATOR_VER" "$SCIPY_VER" "$ATTRS_VER" "$PSUTIL_VER" \
  "$SAFETENSORS_VER" "$HUGGINGFACE_HUB_VER" "$MODELSCOPE_VER" \
  "$TRANSFORMERS_VER" "$PYARROW_VER" "$REQUESTS_VER" "$GGUF_VER" <<'PY' >/dev/null 2>&1
import sys, importlib.metadata as m
names = (
    "torch", "torch_npu", "numpy", "PyYAML", "pybind11", "decorator", "scipy",
    "attrs", "psutil", "safetensors", "huggingface_hub", "modelscope",
    "transformers", "pyarrow", "requests", "gguf",
)
for name, expected in zip(names, sys.argv[1:]):
    actual = m.version(name)
    if name == "torch":
        actual = actual.split("+")[0]
    assert actual == expected, (name, actual, expected)
PY
then
  need_install=0
fi

if [ "$need_install" = 1 ]; then
  log "开始装依赖，torch 与 torch_npu 加起来要下几百 MB"
  "$PY" -m pip install "${PIP_ARGS[@]+"${PIP_ARGS[@]}"}" -r "$REQUIREMENTS_FILE"
  pass "依赖安装完成"
else
  pass "依赖版本已符合，跳过 pip"
fi

# ---------- 5. Import probe and LD_LIBRARY_PATH repair ----------
probe() { "$PY" -c 'import torch, torch_npu, pybind11' >/dev/null 2>&1; }

if probe; then
  pass "探针 import torch, torch_npu, pybind11"
else
  log "探针不通，按 libpython 缺失处理，自动补 LD_LIBRARY_PATH"
  PYPREFIX="$("$PY" -c 'import sys; print(sys.base_prefix)')"
  export LD_LIBRARY_PATH="$PYPREFIX/lib:${LD_LIBRARY_PATH:-}"
  if probe; then
    pass "补上 $PYPREFIX/lib 之后探针通过"
  else
    echo "$TAG   实际回显" >&2
    "$PY" -c 'import torch, torch_npu, pybind11' 2>&1 | tail -5 >&2 || true
    die "补 LD_LIBRARY_PATH 之后探针仍不通"
  fi
  # Keep the exact export prefix so step 1 can preserve this nounset-safe line.
  if ! grep -q "^export LD_LIBRARY_PATH=\"$PYPREFIX/lib" "$GLM53_ENV_FILE"; then
    echo "export LD_LIBRARY_PATH=\"$PYPREFIX/lib:\${LD_LIBRARY_PATH:-}\"" >> "$GLM53_ENV_FILE"
    pass "已追加进 $GLM53_ENV_FILE"
  else
    pass "$GLM53_ENV_FILE 里已有这一行"
  fi
fi

# ---------- 6. Versions and NPU availability ----------
VER_LINE="$("$PY" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())')"
log "版本 $VER_LINE"
read -r _tv _tnv _avail <<<"$VER_LINE"

[ "${_tv%%+*}" = "$TORCH_VER" ] \
  && pass "torch $_tv" \
  || fail "torch 是 $_tv，期望 $TORCH_VER"
[ "$_tnv" = "$TORCH_NPU_VER" ] \
  && pass "torch_npu $_tnv" \
  || fail "torch_npu 是 $_tnv，期望 $TORCH_NPU_VER"
if [ "$_avail" = "True" ]; then
  pass "torch.npu.is_available() = True"
else
  fail "torch.npu.is_available() = False，先用 npu-smi info 确认这个容器里看得见卡，不要往下构建"
fi
[ "$FAILED" = 0 ] || exit 1

# ---------- 7. triton_ascend ----------
# SGLang imports Triton unconditionally. Public indexes lack the required cp312
# build, so copy the complete package from the Ascend image.
DST="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# Validate DST before the two rm -rf operations below.
[ -n "$DST" ] && [ -d "$DST" ] || die "取不到虚拟环境的 site-packages 目录（拿到的是 '$DST'）"
log "目标 site-packages $DST"

triton_ok() {
  "$PY" - "$TRITON_ASCEND_VER" "$TRITON_VER" <<'PY' >/dev/null 2>&1
import sys, importlib.metadata as m, triton
assert m.version("triton_ascend") == sys.argv[1]
assert triton.__version__ == sys.argv[2]
PY
}

if triton_ok; then
  pass "triton_ascend 已就位，跳过拷贝"
else
  # Match candidates by the configured interpreter's actual minor version.
  PY_TAG="python$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ -n "$TRITON_SRC" ]; then
    SRC="$TRITON_SRC"
    log "使用 --triton-src 指定的 $SRC"
  else
    log "在 $GLM53_ROOT 内搜 $PY_TAG 的 triton_ascend"
    SRC=""
    while IFS= read -r d; do
      # Exclude the destination venv so a partial copy is not reused as the source.
      case "$d" in "$GLM53_VENV"/*) continue ;; esac
      [ "$d" != "$DST" ] || continue
      [ -d "$d/triton" ] || continue
      ls -d "$d"/triton_ascend-*.dist-info >/dev/null 2>&1 || continue
      echo "$TAG   候选 $d"
      [ -n "$SRC" ] || SRC="$d"
    done < <(find "$GLM53_ROOT" -maxdepth 8 -type d -name site-packages -print 2>/dev/null \
               | grep "$PY_TAG")
  fi

  if [ -z "$SRC" ] || [ ! -d "$SRC/triton" ] || ! ls -d "$SRC"/triton_ascend-*.dist-info >/dev/null 2>&1; then
    # Report an invalid explicit source separately from a missing local package.
    if [ -n "$TRITON_SRC" ]; then
      echo "$TAG FAIL --triton-src=$TRITON_SRC 下面没有 triton 目录，或者没有 triton_ascend-*.dist-info" >&2
      die "要填的是 site-packages 那一层本身，不是它里面的 triton"
    fi
    echo "$TAG FAIL 本机没有 $PY_TAG 的 triton_ascend，三条路走一条" >&2
    echo "$TAG   1 从别的昇腾镜像取整包（site-packages/triton 加 triton_ascend-*.dist-info，约 971 MiB），" >&2
    echo "$TAG     放到当前工作目录后通过 --triton-src 指定该 site-packages 路径重跑" >&2
    echo "$TAG   2 改走 Python 3.11。公开源上 cp311 的 torch / torch_npu / triton_ascend 3.2.0 都有，" >&2
    echo "$TAG     代价是版本变成 3.2.0 而不是验证过的 3.2.2，且整条流程没在 3.11 上跑过。要动三处，" >&2
    echo "$TAG     只改一处会卡在同一条 FAIL 上。" >&2
    echo "$TAG       a 通过 GLM53_PYTHON_BIN 指定可用的 Python 3.11，重跑 01-config.sh" >&2
    echo "$TAG       b 本脚本顶部 TRITON_ASCEND_VER 改成 3.2.0，TRITON_MIN_MIB 置 0，" >&2
    echo "$TAG         TRITON_VER 按脚本回显里 triton.__version__ 的实际值填" >&2
    echo "$TAG       c 本脚本装不了 triton_ascend，装完依赖后自己补一句" >&2
    echo "$TAG         \"\$GLM53_VENV/bin/python\" -m pip install triton_ascend==3.2.0" >&2
    echo "$TAG         装完不传 --triton-src 重跑本脚本，triton_ok 那步会直接放行" >&2
    echo "$TAG   3 从源码构建 triton-ascend，需要 LLVM，耗时以小时计" >&2
    exit 1
  fi

  # Reject a source inside the destination before removing the destination package.
  case "$SRC" in
    "$DST"|"$GLM53_VENV"/*)
      die "--triton-src 指到了目标虚拟环境自己（$SRC），要填别处那份 site-packages" ;;
  esac

  # Remove the old package before copying to avoid nested directories and stale files.
  rm -rf "$DST/triton"
  rm -rf "$DST"/triton_ascend-*.dist-info
  log "从 $SRC 拷贝，971 MiB 量级，要几分钟"
  cp -a "$SRC/triton" "$SRC"/triton_ascend-*.dist-info "$DST/"
  pass "拷贝完成"
fi

# Convert du failures under pipefail into the explicit size check below.
SIZE_MIB="$(du -sm "$DST/triton" 2>/dev/null | awk 'NR==1{print $1}' || true)"
[ -n "$SIZE_MIB" ] || SIZE_MIB=0
[ "$SIZE_MIB" -ge "$TRITON_MIN_MIB" ] \
  && pass "triton 体积 ${SIZE_MIB} MiB" \
  || fail "triton 在 $DST 下只有 ${SIZE_MIB} MiB，期望 971 MiB 量级，删掉 $DST/triton 重跑本脚本"

# Preserve the explicit failure message when the copied Triton cannot be imported.
TRI_LINE="$("$PY" - 2>/dev/null <<'PY' || true
import importlib.metadata as m, triton
print(m.version("triton_ascend"), triton.__version__)
PY
)"
if [ -z "$TRI_LINE" ]; then
  echo "$TAG   实际回显" >&2
  "$PY" -c 'import importlib.metadata as m, triton; print(m.version("triton_ascend"), triton.__version__)' 2>&1 | tail -5 >&2 || true
  fail "读不出 triton_ascend 与 triton 的版本，$DST/triton 多半没拷全，删掉它重跑本脚本"
else
  read -r _ta _tr <<<"$TRI_LINE"
  [ "$_ta" = "$TRITON_ASCEND_VER" ] \
    && pass "triton_ascend $_ta" \
    || fail "triton_ascend 是 $_ta，期望 $TRITON_ASCEND_VER"
  [ "$_tr" = "$TRITON_VER" ] \
    && pass "triton.__version__ $_tr" \
    || fail "triton.__version__ 是 $_tr，期望 $TRITON_VER"
fi

[ "$FAILED" = 0 ] || exit 1

# Vendor packages exposed by CANN through PYTHONPATH are outside this venv, so
# validate only the two runtime version contracts used by this deployment.
"$PY" - "$TORCH_VER" "$SCIPY_VER" <<'PY' \
  || die "torch_npu/kt-kernel 或 triton_ascend 的运行时版本契约不满足"
import importlib.metadata as m
import sys

torch_ver, scipy_ver = sys.argv[1:]
torch_reqs = [r.replace(" ", "") for r in (m.requires("torch_npu") or [])]
triton_reqs = [r.replace(" ", "") for r in (m.requires("triton_ascend") or [])]
assert m.version("torch").split("+")[0] == torch_ver
assert f"torch=={torch_ver}" in torch_reqs
assert m.version("scipy") == scipy_ver
assert any(r.startswith(f"scipy=={scipy_ver};") for r in triton_reqs)
PY
pass "torch_npu 要求并使用 torch $TORCH_VER"
pass "triton_ascend 要求并使用 scipy $SCIPY_VER"

log "锁定依赖 $REQUIREMENTS_FILE"
echo "=== Python 环境 完成 ==="
