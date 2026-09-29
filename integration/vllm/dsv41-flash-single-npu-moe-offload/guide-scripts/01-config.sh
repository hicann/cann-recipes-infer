#!/usr/bin/env bash
# Generate dsv41.env. The later steps read only this file.
# --root and --model are required; the other options default to the container's preset paths.
set -euo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)  # directory of this script; dsv41.env is written here
N=配置                                            # log prefix

# ---- required, no defaults ----
ROOT=      # work directory, absolute path, >= 30 GiB free; holds the source trees, build outputs, wheel and logs
MODEL=     # weight directory, absolute path; DeepSeek-V4.1-Flash MXFP4 weights with model.safetensors.index.json

# ---- optional, defaults are the container's preset paths ----
CANN=/usr/local/Ascend/cann-9.2.0-beta.2   # CANN root directory, holds set_env.sh
DS41_SITE=/usr/local/Ascend/ds41_site      # ds41 operator package: cann_ops_transformer / cannbotdsl / cannbot_arena_net_ops
PREBUILT=/usr/local/Ascend/dsv41_prebuilt  # custom_transformer operator package
KT_SITE=                                   # kt-kernel install prefix; empty means <ROOT>/kt_site
KT_SHA=105359b4bf70073af9fab992ad29920666acccc9  # ktransformers baseline commit the 10 patches apply to
CARD=0     # NPU logical index
PORT=8100  # HTTP port the service listens on
# directory holding the 10 patches; defaults to ../ktransformers-patches
KT_PATCH_DIR=$(cd "$S/../ktransformers-patches" 2>/dev/null && pwd || echo "$S/../ktransformers-patches")

usage() {
  cat <<EOU
用法: bash -l 01-config.sh --root DIR --model DIR [其他选项]

必填:
  --root DIR         工作目录，绝对路径，>= 30 GiB 可用
  --model DIR        权重目录，须含 model.safetensors.index.json

选填:
  --cann DIR         CANN 根目录      默认 $CANN
  --ds41-site DIR    ds41 算子包      默认 $DS41_SITE
  --prebuilt DIR     custom_transformer 算子包  默认 $PREBUILT
  --kt-site DIR      kt-kernel 安装目录  默认 <--root>/kt_site
  --kt-sha SHA       ktransformers 基线提交号  默认 $KT_SHA
  --kt-patch-dir DIR ktransformers patch 目录  默认 $KT_PATCH_DIR
  --card N           NPU 逻辑序号     默认 $CARD
  --port N           HTTP 端口        默认 $PORT

例:
  bash -l 01-config.sh --root /root/dsv41 --model /workspace/models/DeepSeek-V4.1-Flash
  bash -l 01-config.sh --root /data/dsv41 --model /data/DeepSeek-V4.1-Flash --card 3 --port 8200
EOU
}
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --cann) CANN=$2; shift 2;;
    --ds41-site) DS41_SITE=$2; shift 2;;
    --prebuilt) PREBUILT=$2; shift 2;;
    --kt-site) KT_SITE=$2; shift 2;;
    --kt-sha) KT_SHA=$2; shift 2;;
    --kt-patch-dir) KT_PATCH_DIR=$2; shift 2;;
    --card) CARD=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "未知选项 $1"; usage; exit 1;;
  esac
done

[ -n "$ROOT" ]  || { echo "[$N] FAIL --root 必填：工作目录没有默认值"; usage; exit 1; }
[ -n "$MODEL" ] || { echo "[$N] FAIL --model 必填：权重目录没有默认值"; usage; exit 1; }
case "$ROOT"  in /*) ;; *) echo "[$N] FAIL --root 需要绝对路径: $ROOT";  exit 1;; esac
case "$MODEL" in /*) ;; *) echo "[$N] FAIL --model 需要绝对路径: $MODEL"; exit 1;; esac

[ -n "$KT_SITE" ] || KT_SITE=$ROOT/kt_site
case "$KT_SITE" in
  */site-packages|*/site-packages/|*/dist-packages|*/dist-packages/|/usr/*|/lib/*|/)
    echo "[$N] FAIL --kt-site 不能指向系统目录: $KT_SITE"; exit 1;;
esac

for p in "$MODEL" "$CANN" "$DS41_SITE" "$PREBUILT"; do
  [ -d "$p" ] || { echo "[$N] FAIL 目录不存在 $p"; exit 1; }
done
[ -f "$CANN/set_env.sh" ] || { echo "[$N] FAIL 缺 $CANN/set_env.sh"; exit 1; }
[ -f "$MODEL/model.safetensors.index.json" ] || { echo "[$N] FAIL 缺 $MODEL/model.safetensors.index.json"; exit 1; }
KT_PATCH_DIR=$(cd "$KT_PATCH_DIR" && pwd)
NP=0
for p in "$KT_PATCH_DIR"/0*.patch; do [ -e "$p" ] || continue; NP=$((NP+1)); done
[ "$NP" = 10 ] || { echo "[$N] FAIL $KT_PATCH_DIR 下 patch 数 $NP，期望 10"; exit 1; }
mkdir -p "$ROOT"

cat > "$S/dsv41.env" <<EOE
export DSV41_ROOT=$ROOT                       # work directory
export DSV41_SCRIPTS=$S                       # this guide-scripts directory
export DSV41_PREBUILT=$PREBUILT               # custom_transformer operator package
export DSV41_STAGE_DIR=/dev/shm/engram_stage  # engram staging directory, fixed in engram_staging.py
export DSV41_VLLM_REPO=https://github.com/wenxuewuhd/vllm.git  # vllm repository
export DSV41_VLLM_BRANCH=dsv41-moe-offload    # vllm branch, for reference only
export DSV41_VLLM_SHA=a9686bc6e6809a49cb3e3c592a337d0721c3619f  # vllm pinned commit
export DSV41_ASCEND_REPO=https://github.com/wenxuewuhd/vllm-ascend.git  # vllm-ascend repository
export DSV41_ASCEND_BRANCH=dsv41-moe-offload-pr  # vllm-ascend branch, for reference only
export DSV41_ASCEND_SHA=59ce1e18a4f93c20a872e4450bd14e891c556cb7  # vllm-ascend pinned commit
export DSV41_KT_REPO=https://github.com/kvcache-ai/ktransformers.git  # ktransformers upstream repository
export DSV41_KT_SHA=$KT_SHA                   # ktransformers baseline commit
export DSV41_KT_TREE_HASH=393d8d5d99afef42449e4e804467c668d71a40c4  # expected git write-tree after the 10 patches
export DSV41_KT_DIFFSTAT="16 files changed, 2080 insertions(+), 161 deletions(-)"  # expected last line of diff --stat after the patches
export DSV41_KT_SUBMODULE_LLAMA=a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f  # expected llama.cpp submodule gitlink
export DSV41_KT_SUBMODULE_PYBIND=bb05e0810b87e74709d9f4c4545f1f57a1b386f5  # expected pybind11 submodule gitlink
export DSV41_KT_PATCH_DIR=$KT_PATCH_DIR       # directory holding the 10 patches
export DSV41_KT_WHEEL_DIR=$ROOT/kt_wheel      # output directory for the kt-kernel wheel
export MODEL=$MODEL                           # weight directory
export CANN=$CANN                             # CANN root directory
export DS41_SITE=$DS41_SITE                   # ds41 operator package
export KT_SITE=$KT_SITE                       # kt-kernel install prefix
export KT_TREE=$ROOT/ktransformers            # ktransformers source tree
export VLLM_TREE=$ROOT/vllm                   # vllm source tree
export ASCEND_TREE=$ROOT/vllm-ascend          # vllm-ascend source tree, holds the serve scripts under examples
export CARD=$CARD                             # NPU logical index
export PORT=$PORT                             # service HTTP port
EOE

echo "[$N] PASS 工作目录 $ROOT"
echo "[$N] PASS 权重 $MODEL"
echo "[$N] PASS CANN $CANN"
echo "[$N] PASS ktransformers patch $NP 个 @ $KT_PATCH_DIR"
echo "[$N] PASS 配置写入 $S/dsv41.env"
echo "=== $N 完成 ==="
