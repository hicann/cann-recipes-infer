#!/usr/bin/env bash
# Download the vendor FP8 checkpoint and convert it to INT8 W8A8 and MXFP4.
set -euo pipefail

# Disable nounset while sourcing the LD_LIBRARY_PATH line appended by step 3.
set +u
# Environment variables come from source ./glm53.env; only validate them here.
: "${GLM53_ROOT:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
: "${GLM53_ENV_FILE:?环境变量没配。先在脚本所在目录执行  source ./glm53.env}"
set -u

# ==== Edit here ====
FP8_HUB=hf                          # hf uses huggingface.co; ms uses modelscope.cn.
FP8_REPO_HF=zai-org/GLM-5.3-Flash   # 62 shards, 305.8 GiB.
FP8_REPO_MS=ZhipuAI/GLM-5.3-Flash   # ModelScope uses the ZhipuAI namespace.
FP8_DIR=""                          # Empty uses $GLM53_MODEL_ROOT/GLM-5.3-Flash-FP8.
CONV_TOOL_DIR=""                    # Empty uses $GLM53_PATCH_DIR/weight-conversion.
CONV_PYTHON=""                      # Empty selects an interpreter that imports the required packages.
INT8_WORKERS=3                      # Each worker peaks at about 12 GiB of host memory.
MXFP4_WORKERS=8                     # Reduce this when disk writes are slow.
MIN_FREE_GIB=40                     # Free space retained beyond the conversion requirement.
# ==== End editable values ====

TAG="[权重]"
log()  { echo "$TAG $*"; }
pass() { echo "$TAG PASS $*"; }
die()  { echo "$TAG FAIL $*" >&2; exit 1; }

: "${GLM53_ROOT:?glm53.env 里没有 GLM53_ROOT，先跑第 1 步的配置脚本}"
: "${GLM53_MODEL_ROOT:?glm53.env 里没有 GLM53_MODEL_ROOT，先跑第 1 步的配置脚本}"

[ -n "$FP8_DIR" ] || FP8_DIR="$GLM53_MODEL_ROOT/GLM-5.3-Flash-FP8"
[ -n "$CONV_TOOL_DIR" ] || CONV_TOOL_DIR="${GLM53_PATCH_DIR:-}/weight-conversion"
# Keep these directory names aligned with the paths derived in glm53_env.sh.
INT8_DIR="${GLM53_MODEL_PATH:-$GLM53_MODEL_ROOT/GLM-5.3-Flash-W8A8}"
MXFP4_DIR="${GLM53_MXFP4_CKPT:-$GLM53_MODEL_ROOT/GLM-5.3-Flash-MXFP4}"
WORK="$GLM53_ROOT/weights-conv"
LOGDIR="${GLM53_ARTIFACT_ROOT:-$GLM53_ROOT/artifact}/logs"
SRC_VIEW="$WORK/fp8-view"

# Full checkpoint sizes measured on the A3 single-card deployment.
FP8_GIB=306
INT8_GIB=307
MXFP4_GIB=170

case "$FP8_HUB" in
  hf) FP8_REPO="$FP8_REPO_HF" ;;
  ms) FP8_REPO="$FP8_REPO_MS" ;;
  *)  die "FP8_HUB 只能填 hf 或 ms，现在是 $FP8_HUB" ;;
esac

STAGES=("${@:-all}")
if [ "${STAGES[0]}" = all ]; then STAGES=(download int8 mxfp4); fi
for _s in "${STAGES[@]}"; do
  case "$_s" in
    space|download|int8|mxfp4) ;;
    *) die "用法 bash 04-weights.sh [space|download|int8|mxfp4|all]" ;;
  esac
done

mkdir -p "$WORK" "$LOGDIR" || die "建不出 $WORK 或 $LOGDIR，看 GLM53_ROOT 的写权限"

# ---------- Interpreter ----------
# Include NumPy in the probe because safetensors imports it only when saving a shard.
py_ok() { "$1" -c 'import torch, safetensors, numpy' >/dev/null 2>&1; }
PY=""
if [ -n "$CONV_PYTHON" ]; then
  py_ok "$CONV_PYTHON" || die "$CONV_PYTHON 不能同时 import torch、safetensors 与 numpy"
  PY="$CONV_PYTHON"
else
  for _c in "${GLM53_VENV:-}/bin/python" python3; do
    if [ -n "$_c" ] && command -v "$_c" >/dev/null 2>&1 && py_ok "$_c"; then PY="$_c"; break; fi
  done
fi
[ -n "$PY" ] || die "找不到能同时 import torch、safetensors 与 numpy 的解释器，或在脚本顶部填 CONV_PYTHON"
pass "转换用解释器 $PY"

# ---------- Conversion tools ----------
# pipeline_fp8_to_int8.py locates the two upstream tools relative to __file__.
for _f in pipeline_fp8_to_int8.py fp8_to_mxfp4_moe.py tools/fp8_to_bf16.py tools/bf16_to_int8_ct.py; do
  [ -f "$CONV_TOOL_DIR/$_f" ] || die "$CONV_TOOL_DIR 下没有 $_f，把 CONV_TOOL_DIR 指到 cann-recipes-infer 的 weight-conversion 目录"
done
pass "转换工具目录 $CONV_TOOL_DIR"

# ---------- Disk space ----------
existing_dir() { local d="$1"; while [ ! -d "$d" ] && [ "$d" != "/" ]; do d="$(dirname "$d")"; done; printf '%s' "$d"; }
# Ignore du's status because it may print a usable total before reporting an unreadable child.
free_gib() { local kb; kb="$(df -Pk "$(existing_dir "$1")" 2>/dev/null | awk 'NR==2{print $4}' || true)"; printf '%d' "$(( ${kb:-0} / 1048576 ))"; }
have_gib() { local kb=""; if [ -d "$1" ]; then kb="$(du -sk "$1" 2>/dev/null | awk 'NR==1{print $1}' || true)"; fi; printf '%d' "$(( ${kb:-0} / 1048576 ))"; }

# Reject insufficient capacity before starting a multi-hour conversion.
need_space() {  # need_space <directory> <full size GiB> <description>
  local dir="$1" want="$2" what="$3" have free need
  have="$(have_gib "$dir")"; free="$(free_gib "$dir")"
  need=$(( want - have ))
  if [ "$need" -lt 0 ]; then need=0; fi
  need=$(( need + MIN_FREE_GIB ))
  [ "$free" -ge "$need" ] || die "$what 空间不够，需要 $need GiB"
  pass "$what 空间够，需要 $need GiB"
}

stage_space() {
  log "FP8   $FP8_DIR 已占 $(have_gib "$FP8_DIR") GiB，满量 $FP8_GIB GiB"
  log "W8A8  $INT8_DIR 已占 $(have_gib "$INT8_DIR") GiB，满量 $INT8_GIB GiB"
  log "MXFP4 $MXFP4_DIR 已占 $(have_gib "$MXFP4_DIR") GiB，满量 $MXFP4_GIB GiB"
  log "三份并存要 $(( FP8_GIB + INT8_GIB + MXFP4_GIB )) GiB"
}

# ---------- FP8 source ----------
# Present a file-only symlink view because the converters copy every non-shard
# entry and cannot copy downloader-created directories with shutil.copy2.
build_src_view() {
  [ -f "$FP8_DIR/model.safetensors.index.json" ] \
    || die "$FP8_DIR 下没有 model.safetensors.index.json，先跑 bash 04-weights.sh download"
  rm -rf "$SRC_VIEW"
  mkdir -p "$SRC_VIEW"
  find "$FP8_DIR" -mindepth 1 -maxdepth 1 ! -type d -exec ln -s {} "$SRC_VIEW/" \;
  pass "FP8 软链视图就绪 $SRC_VIEW"
}

# ---------- Stages ----------
stage_download() {
  need_space "$FP8_DIR" "$FP8_GIB" "下载 FP8"
  mkdir -p "$FP8_DIR"
  log "下载 $FP8_REPO 到 $FP8_DIR，约 $FP8_GIB GiB，两个下载器都是断点续传，被打断重跑本步就行"
  if [ "$FP8_HUB" = hf ]; then
    HF="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$HF" ] || die "没有 hf 也没有 huggingface-cli；第 3 步的锁定依赖未装完整，先重跑 03-python-env.sh"
    if ! "$HF" download "$FP8_REPO" --local-dir "$FP8_DIR" 2>&1 | tee -a "$LOGDIR/weights-download.log"; then
      die "下载退出码非 0，重跑本步接着续传"
    fi
  else
    command -v modelscope >/dev/null 2>&1 \
      || die "没有 modelscope 命令；第 3 步的锁定依赖未装完整，先重跑 03-python-env.sh"
    if ! modelscope download --model "$FP8_REPO" --local_dir "$FP8_DIR" 2>&1 | tee -a "$LOGDIR/weights-download.log"; then
      die "下载退出码非 0，重跑本步接着续传"
    fi
  fi
  [ -f "$FP8_DIR/model.safetensors.index.json" ] \
    || die "下载器退出码是 0，但 $FP8_DIR 下没有 model.safetensors.index.json，多半是仓库 id 不对"
  pass "FP8 原始权重就绪 $FP8_DIR"
}

stage_int8() {
  # index.json is written last and therefore marks a completed conversion.
  if [ -f "$INT8_DIR/model.safetensors.index.json" ]; then pass "W8A8 已在盘上，跳过"; return 0; fi
  need_space "$INT8_DIR" "$INT8_GIB" "转 W8A8"
  build_src_view
  local t0=$SECONDS
  # Resume by shard so a rerun converts only missing shards.
  if ! "$PY" "$CONV_TOOL_DIR/pipeline_fp8_to_int8.py" \
       --src "$SRC_VIEW" --dst "$INT8_DIR" --workers "$INT8_WORKERS" \
       --min-free-gib "$MIN_FREE_GIB" 2>&1 | tee -a "$LOGDIR/weights-int8.log"; then
    die "转 W8A8 退出码非 0"
  fi
  log "转 W8A8 用时 $(( (SECONDS - t0) / 60 )) 分钟"
  pass "W8A8 就绪 $INT8_DIR"
}

stage_mxfp4() {
  if [ -f "$MXFP4_DIR/model.safetensors.index.json" ]; then pass "MXFP4 已在盘上，跳过"; return 0; fi
  need_space "$MXFP4_DIR" "$MXFP4_GIB" "转 MXFP4"
  build_src_view
  local t0=$SECONDS
  # This converter has no shard-level resume; an interrupted run starts over.
  if ! "$PY" "$CONV_TOOL_DIR/fp8_to_mxfp4_moe.py" --src "$SRC_VIEW" --dst "$MXFP4_DIR" \
       --workers "$MXFP4_WORKERS" 2>&1 | tee -a "$LOGDIR/weights-mxfp4.log"; then
    die "转 MXFP4 退出码非 0"
  fi
  log "转 MXFP4 用时 $(( (SECONDS - t0) / 60 )) 分钟"
  pass "MXFP4 就绪 $MXFP4_DIR"
}

# ---------- Run ----------
for _s in "${STAGES[@]}"; do
  log "--- $_s ---"
  case "$_s" in
    space)    stage_space ;;
    download) stage_download ;;
    int8)     stage_int8 ;;
    mxfp4)    stage_mxfp4 ;;
  esac
done

if [ -f "$INT8_DIR/model.safetensors.index.json" ] \
   && [ -f "$MXFP4_DIR/model.safetensors.index.json" ] && [ -d "$FP8_DIR" ]; then
  log "两份派生权重都在了，FP8 原始权重可以删，能腾出 $(have_gib "$FP8_DIR") GiB"
  log "  确认之后自己执行  rm -rf \"$FP8_DIR\""
  log "  删了之后要重转任何一份，都得先把这 $FP8_GIB GiB 重新下一遍"
fi

echo "=== 权重 完成 ==="
