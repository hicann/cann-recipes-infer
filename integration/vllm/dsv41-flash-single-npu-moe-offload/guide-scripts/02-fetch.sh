#!/usr/bin/env bash
# Fetch the three source trees: vllm and vllm-ascend at their pinned SHAs, ktransformers at the
# upstream baseline plus the 10 patches. Then lay down custom_transformer and write two metadata files.
set -euo pipefail
N=取码
: "${DSV41_ROOT:?先 source dsv41.env}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
# Abort an attempt after GIT_HTTP_LOW_SPEED_TIME seconds below GIT_HTTP_LOW_SPEED_LIMIT B/s.
export GIT_HTTP_LOW_SPEED_LIMIT=${GIT_HTTP_LOW_SPEED_LIMIT:-1000}
export GIT_HTTP_LOW_SPEED_TIME=${GIT_HTTP_LOW_SPEED_TIME:-60}
FETCH_TIMEOUT=${DSV41_FETCH_TIMEOUT:-900}
FETCH_TRIES=${DSV41_FETCH_TRIES:-8}
PROBE_TRIES=${DSV41_PROBE_TRIES:-90}

# Wait until github.com:443 accepts a connection.
wait_net() {
  local i
  for i in $(seq 1 "$PROBE_TRIES"); do
    curl -s --connect-timeout 8 --max-time 15 -o /dev/null https://github.com/ 2>/dev/null && return 0
    sleep 5
  done
  echo "[$N] FAIL github.com:443 连续 $PROBE_TRIES 次探测不通"
  return 1
}

retry_net() {
  local what=$1; shift
  local i
  for i in $(seq 1 "$FETCH_TRIES"); do
    wait_net || return 1
    timeout "$FETCH_TIMEOUT" "$@" && return 0
    [ "$i" = "$FETCH_TRIES" ] && { echo "[$N] FAIL $what 连续 $FETCH_TRIES 次失败"; return 1; }
    echo "[$N] WARN $what 第 $i 次失败，10 秒后重探"
    sleep 10
  done
}

# Shallow-fetch a pinned SHA into dir. If HEAD already is that SHA, only reset the tree and index.
fetch_pinned() {
  local dir=$1 url=$2 sha=$3
  if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    git -C "$dir" reset -q --hard HEAD
    git -C "$dir" clean -fdq -e third_party
    echo "[$N] SKIP $(basename "$dir") 已是 $sha，工作区与索引已复位"
    return 0
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  retry_net "取 $url" git -C "$dir" fetch -q --depth 1 --no-tags origin "$sha" || exit 1
  git -C "$dir" checkout -q FETCH_HEAD
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || { echo "[$N] FAIL $dir HEAD 不是 $sha"; exit 1; }
}

fetch_pinned "$VLLM_TREE"   "$DSV41_VLLM_REPO"   "$DSV41_VLLM_SHA"
echo "[$N] PASS vllm        $DSV41_VLLM_SHA"
fetch_pinned "$ASCEND_TREE" "$DSV41_ASCEND_REPO" "$DSV41_ASCEND_SHA"
echo "[$N] PASS vllm-ascend $DSV41_ASCEND_SHA"
fetch_pinned "$KT_TREE"     "$DSV41_KT_REPO"     "$DSV41_KT_SHA"
echo "[$N] PASS ktransformers $DSV41_KT_SHA"

retry_net "取 kt 子模块" git -C "$KT_TREE" submodule update -q --init --depth 1 third_party/llama.cpp third_party/pybind11 || exit 1
SL=$(git -C "$KT_TREE" rev-parse HEAD:third_party/llama.cpp)
SP=$(git -C "$KT_TREE" rev-parse HEAD:third_party/pybind11)
[ "$SL" = "$DSV41_KT_SUBMODULE_LLAMA"  ] || { echo "[$N] FAIL llama.cpp $SL，期望 $DSV41_KT_SUBMODULE_LLAMA"; exit 1; }
[ "$SP" = "$DSV41_KT_SUBMODULE_PYBIND" ] || { echo "[$N] FAIL pybind11 $SP，期望 $DSV41_KT_SUBMODULE_PYBIND"; exit 1; }
[ -f "$KT_TREE/third_party/pybind11/CMakeLists.txt" ] || { echo "[$N] FAIL pybind11 未检出"; exit 1; }
[ -f "$KT_TREE/third_party/llama.cpp/CMakeLists.txt" ] || { echo "[$N] FAIL llama.cpp 未检出"; exit 1; }
echo "[$N] PASS 子模块 llama.cpp $(echo "$SL" | cut -c1-7) / pybind11 $(echo "$SP" | cut -c1-7)"

git -C "$KT_TREE" config user.email dsv41@localhost
git -C "$KT_TREE" config user.name dsv41
APPLIED=0
for p in "$DSV41_KT_PATCH_DIR"/0*.patch; do
  git -C "$KT_TREE" apply --check "$p" || { echo "[$N] FAIL patch 不适用 $(basename "$p")"; exit 1; }
  git -C "$KT_TREE" apply "$p"
  APPLIED=$((APPLIED+1))
done
[ "$APPLIED" = 10 ] || { echo "[$N] FAIL 已应用 $APPLIED 个，期望 10"; exit 1; }
echo "[$N] PASS patch 已应用 $APPLIED 个"

git -C "$KT_TREE" add -A -- ':!third_party'
WT=$(git -C "$KT_TREE" write-tree)
[ "$WT" = "$DSV41_KT_TREE_HASH" ] || { echo "[$N] FAIL write-tree $WT，期望 $DSV41_KT_TREE_HASH"; exit 1; }
echo "[$N] PASS write-tree = $WT"
DS=$(git -C "$KT_TREE" diff --cached --shortstat "$DSV41_KT_SHA" | sed 's/^ *//')
[ "$DS" = "$DSV41_KT_DIFFSTAT" ] || { echo "[$N] FAIL diffstat「$DS」，期望「$DSV41_KT_DIFFSTAT」"; exit 1; }
echo "[$N] PASS diffstat = $DS"

cp -a "$DSV41_PREBUILT/vllm-ascend/vllm_ascend/_cann_ops_custom" "$ASCEND_TREE/vllm_ascend/"
[ -L "$ASCEND_TREE/vllm_ascend/_cann_ops_custom" ] && { echo "[$N] FAIL 算子包是符号链接"; exit 1; }
CT=$(find "$ASCEND_TREE/vllm_ascend/_cann_ops_custom" -path '*custom_transformer*' -type f | wc -l)
[ "$CT" = 786 ] || { echo "[$N] FAIL custom_transformer 文件数 $CT，期望 786"; exit 1; }
echo "[$N] PASS custom_transformer 文件数 = $CT"
echo "[$N] PASS $(cat "$ASCEND_TREE/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/version.info")"

echo "__device_type__ = 'A5'" > "$ASCEND_TREE/vllm_ascend/_build_info.py"
printf "__version__ = version = '0.27.1'\n__version_tuple__ = version_tuple = (0, 27, 1)\n" > "$VLLM_TREE/vllm/_version.py"
echo "[$N] PASS _build_info.py $(stat -c%s "$ASCEND_TREE/vllm_ascend/_build_info.py") B"
echo "[$N] PASS _version.py $(stat -c%s "$VLLM_TREE/vllm/_version.py") B"
echo "=== $N 完成 ==="
