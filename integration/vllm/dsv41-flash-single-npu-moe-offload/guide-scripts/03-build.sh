#!/usr/bin/env bash
# Build vllm_ascend_C.so and the kt-kernel wheel, then install kt-kernel into $KT_SITE.
set -euo pipefail
N=编译
: "${ASCEND_TREE:?先 source dsv41.env}"
ACT=${1:-all}
case "$ACT" in ascend|kt|all) ;; *) echo "用法: bash -l 03-build.sh [ascend|kt|all]"; exit 1;; esac
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
export TMPDIR=${TMPDIR:-$DSV41_ROOT/tmp}
mkdir -p "$TMPDIR"

# Source CANN's set_env.sh with errexit, pipefail and nounset off.
source_cann() {
  set +u +e +o pipefail
  source "$CANN/set_env.sh"
  set -euo pipefail
}

build_ascend() {
  source_cann
  cd "$ASCEND_TREE"
  local T0=$(date +%s)
  SETUPTOOLS_SCM_PRETEND_VERSION=0.19.1rc2.dev1816 \
    COMPILE_CUSTOM_KERNELS=1 SKIP_ACLNN_BUILD=1 \
    python3 setup.py build_ext --inplace > "$DSV41_ROOT/build-ascend.log" 2>&1
  local SO=$ASCEND_TREE/vllm_ascend/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
  [ -f "$SO" ] || { echo "[$N] FAIL 未生成 .so，见 $DSV41_ROOT/build-ascend.log"; exit 1; }
  local SZ=$(stat -c%s "$SO")
  [ "$SZ" = 1112760 ] || echo "[$N] WARN vllm_ascend_C.so 大小 $SZ，参考值 1112760"
  echo "[$N] PASS vllm_ascend_C.so $SZ B，用时 $(( $(date +%s) - T0 )) 秒"
  echo "[$N] PASS 日志 $DSV41_ROOT/build-ascend.log"
}

build_kt() {
  if ! command -v pkg-config >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends pkg-config >/dev/null
    command -v pkg-config >/dev/null || { echo "[$N] FAIL pkg-config 安装失败"; exit 1; }
  fi
  export PKG_CONFIG_PATH=${PKG_CONFIG_PATH:-/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig}
  pkg-config --modversion hwloc >/dev/null || { echo "[$N] FAIL pkg-config 找不到 hwloc"; exit 1; }
  echo "[$N] PASS pkg-config $(pkg-config --version) / hwloc $(pkg-config --modversion hwloc)"

  export CPUINFER_CPU_INSTRUCT=${CPUINFER_CPU_INSTRUCT:-NATIVE}
  export CPUINFER_ENABLE_AMX=${CPUINFER_ENABLE_AMX:-OFF}
  export CPUINFER_ENABLE_AVX512_VNNI=${CPUINFER_ENABLE_AVX512_VNNI:-ON}
  export CPUINFER_ENABLE_AVX512_BF16=${CPUINFER_ENABLE_AVX512_BF16:-ON}
  export CPUINFER_ENABLE_AVX512_VBMI=${CPUINFER_ENABLE_AVX512_VBMI:-ON}
  export CPUINFER_ENABLE_KML=OFF CPUINFER_ENABLE_BLIS=OFF CPUINFER_ENABLE_CPPTRACE=OFF
  export CPUINFER_BUILD_TYPE=Release
  export CPUINFER_PARALLEL=${CPUINFER_PARALLEL:-16}
  export CPUINFER_VERBOSE=1

  case "$KT_SITE" in
    */site-packages|*/site-packages/|*/dist-packages|*/dist-packages/|/usr/*|/lib/*|""|/)
      echo "[$N] FAIL \$KT_SITE 指向系统目录，拒绝清空: $KT_SITE"; exit 1;;
  esac
  rm -rf "$DSV41_KT_WHEEL_DIR" "$KT_SITE"
  mkdir -p "$DSV41_KT_WHEEL_DIR" "$KT_SITE"
  cd "$KT_TREE/kt-kernel"
  local T0=$(date +%s)
  python3 -m pip wheel --no-deps --no-build-isolation -w "$DSV41_KT_WHEEL_DIR" . \
    > "$DSV41_ROOT/build-kt.log" 2>&1 || { echo "[$N] FAIL kt-kernel 构建失败，见 $DSV41_ROOT/build-kt.log"; tail -20 "$DSV41_ROOT/build-kt.log"; exit 1; }
  local WHL=$(ls -1 "$DSV41_KT_WHEEL_DIR"/kt_kernel-*.whl 2>/dev/null | head -1)
  [ -n "$WHL" ] || { echo "[$N] FAIL 未生成 wheel，见 $DSV41_ROOT/build-kt.log"; exit 1; }
  echo "[$N] PASS $(basename "$WHL") $(stat -c%s "$WHL") B，用时 $(( $(date +%s) - T0 )) 秒"

  python3 -m pip install -q --no-deps --no-index --target "$KT_SITE" "$WHL"
  local EXT=$(ls -1 "$KT_SITE"/kt_kernel/kt_kernel_ext*.so 2>/dev/null | head -1)
  [ -n "$EXT" ] || { echo "[$N] FAIL $KT_SITE 下无 kt_kernel_ext*.so"; exit 1; }
  for s in KT_ZEROCOPY_WEIGHTS KT_ZEROCOPY_SCOPE KT_FULLSET_LOAD; do
    strings "$EXT" 2>/dev/null | grep -q "$s" || grep -rqs "$s" "$KT_SITE/kt_kernel" \
      || { echo "[$N] FAIL 产物缺 $s"; exit 1; }
  done
  echo "[$N] PASS 定制开关 KT_ZEROCOPY_WEIGHTS / KT_ZEROCOPY_SCOPE / KT_FULLSET_LOAD 均在产物内"

  local INFO
  INFO=$(PYTHONPATH="$KT_SITE" python3 -c 'import kt_kernel,sys;print(kt_kernel.__version__, getattr(kt_kernel,"__cpu_variant__","?"), kt_kernel.__file__)')
  local KV=$(echo "$INFO" | awk '{print $1}')
  local KC=$(echo "$INFO" | awk '{print $2}')
  local KF=$(echo "$INFO" | awk '{print $3}')
  [ "$KV" = 0.7.0.post4 ] || { echo "[$N] FAIL kt_kernel.__version__ = $KV，期望 0.7.0.post4"; exit 1; }
  case "$KF" in "$KT_SITE"/*) ;; *) echo "[$N] FAIL kt_kernel 解析到 $KF，不在 $KT_SITE 下"; exit 1;; esac
  echo "[$N] PASS kt_kernel $KV，__file__ 在 \$KT_SITE 下"
  echo "[$N] INFO __cpu_variant__ = $KC"
  echo "[$N] PASS 日志 $DSV41_ROOT/build-kt.log"
}

case "$ACT" in
  ascend) build_ascend;;
  kt)     build_kt;;
  all)    build_ascend; build_kt;;
esac
echo "=== $N 完成 ==="
