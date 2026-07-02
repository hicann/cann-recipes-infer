#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 单卡 NPU + KT(LLAMAFILE) 启动 SGLang HTTP 服务（DeepSeek-V4-Flash，NPU W8A8 + CPU MXFP4 offload）。
# 底盘为 sglang DSv4 基线（third_party/sglang），KT MoE 在该基线内原生支持；本脚本是单卡 KT 子集
# （省略多卡 HCCL/DeepEP/MTP 等）。
#
# 量化：--quantization compressed-tensors，对齐磁盘 W8A8 (compressed-tensors / int-quantized)。
# 基线已不再读取 SGLANG_APPLY_CONFIG_BACKUP，相关历史变量已移除。
#
# 用法（在任意目录）：
#   bash /path/to/ktransformers-AK/tools/launch_ds4flash_npu.sh
#   bash .../launch_ds4flash_npu.sh 3          # 与 NPU_DEVICE_ID=3 等价，指定物理 NPU 卡号
#
# 常用覆盖（环境变量）：
#   REPO              默认本脚本所在仓库根
#   MODEL_PATH        必填（无默认）：W8A8 权重目录（NPU 侧）
#   KT_GGUF_TEMPLATE  必填（无默认）：MXFP4 GGUF 模板（CPU 侧），如 /path/cache/dsv4_layer{layer_idx}_mxfp4.gguf
#                     （{layer_idx} 是字面量占位符，勿让 shell 展开）
#   PORT              默认 8000
#   ASCEND_TOOLKIT_HOME  默认 /usr/local/Ascend/ascend-toolkit/latest
#   NPU_DEVICE_ID     可选，物理 NPU 序号（如 2）。设置后会 export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICE_ID。
#   CHUNKED_PREFILL_SIZE  默认 2048（必须是 page-size=128 的倍数，且 >= page-size）。
#                         注意：不能传 -1。KT(LLAMAFILE) C++ MoE 内部 fp32 输出
#                         buffer 按 max_possible_qlen()=max(max_len, group_max_len)
#                         分配；-1 会被算成 1，prefill qlen>1 时立刻越界写堆
#                         → glibc tcache abort。详见 Handoff 附录 Z.7。
#   QUANTIZATION      默认 compressed-tensors（与基线一致）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/ensure_kt_kernel.sh
source "${SCRIPT_DIR}/ensure_kt_kernel.sh"
ensure_kt_kernel "$REPO"
export PYTHONPATH="${REPO}/third_party/sglang/python:${REPO}/kt-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

# ---------- 选定 Python 解释器 ----------
# 现象：本镜像默认 PATH 里 ``python3`` 指 ``/usr/bin/python3``（系统 python，没装
# torch_npu/numpy/sglang）。真正可用的解释器在 ``/usr/local/python3.11.14``。
# 用户自定义 shell（例如 conda / 直接 export PATH）会把 3.11.14 提前到前面，
# 但一旦从某个 cron / systemd / clean bash -lic 拉起就会踩坑。下面做一次性探测：
# 1) 允许 ``PYTHON_BIN`` 覆盖；2) 否则按 (``python3`` → ``python3.11`` → 已知绝对
# 路径) 顺序找第一个能 import torch_npu+sglang+numpy 的解释器。
_probe_py() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  PYTHONPATH="$PYTHONPATH" "$bin" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("numpy", "torch", "torch_npu", "sglang"):
    importlib.import_module(m)
PY
}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for _cand in python3 python3.11 /usr/local/python3.11.14/bin/python3.11 \
               /usr/local/python3.11.14/bin/python3 /opt/conda/bin/python3; do
    if _probe_py "$_cand"; then
      PYTHON_BIN="$(command -v "$_cand")"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "[launch][ERROR] 没找到能 import numpy/torch/torch_npu/sglang 的 python。" >&2
  echo "[launch][ERROR] 当前 PATH 中的 python3 = $(command -v python3 || echo none)；" >&2
  echo "[launch][ERROR] 建议显式 export PYTHON_BIN=/usr/local/python3.11.14/bin/python3.11 后重试。" >&2
  exit 2
fi
export PYTHON_BIN
echo "[launch] PYTHON_BIN=${PYTHON_BIN}"

# 可选首个参数：纯数字则视为物理 NPU 卡号（与 NPU_DEVICE_ID 一致）
if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ && -z "${NPU_DEVICE_ID:-}" ]]; then
  NPU_DEVICE_ID="$1"
  shift
fi

# MODEL_PATH 必填：不硬编码环境特定路径（红线 R5）。指向 W8A8 权重目录（NPU 侧）。
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[launch][ERROR] 未设置 MODEL_PATH。请指向 W8A8 权重目录（NPU 侧），例如：" >&2
  echo "[launch][ERROR]   MODEL_PATH=/your/path/DeepSeek-V4-Flash-W8A8 bash $0" >&2
  exit 2
fi
# KT_GGUF_TEMPLATE 必填：MXFP4 GGUF 模板（CPU 侧，批量 convert 输出 dsv4_layer{L}_mxfp4.gguf）。
# 注意 {layer_idx} 是字面量占位符——用单引号，别用 ``${KT_GGUF_TEMPLATE:-...{layer_idx}...}``：
# bash 会把第一个 ``}`` 当成 ``${...:-}`` 的结束符，路径会变成 ``...{layer_idx.gguf}``。
# BF16 回退则把模板换成 ..._bf16.gguf。
if [[ -z "${KT_GGUF_TEMPLATE:-}" ]]; then
  echo "[launch][ERROR] 未设置 KT_GGUF_TEMPLATE。请指向 MXFP4 GGUF 模板（CPU 侧），例如（单引号）：" >&2
  echo "[launch][ERROR]   KT_GGUF_TEMPLATE='/your/cache/dsv4_layer{layer_idx}_mxfp4.gguf' bash $0" >&2
  exit 2
fi
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
# CPU MoE is memory-bandwidth-bound; scale threads to raise effective DDR bandwidth.
# Default is now 128 (16/NUMA). Isolated decode micro-bench (real layer weights, output
# verified) shows effective bandwidth has a KNEE at 128 then a
# noisy plateau: 96(12/NUMA)=88, 112(14)=96, 128(16)=114, 144=109, 160(20)=110, 176=116 GB/s;
# only 192 (24/NUMA = ALL cores) COLLAPSES (no spare core for the NumaJobDistributor spin
# threads + NPU host callback + python/OS -> oversubscription thrash).
# End-to-end server (single card, 32 GPU experts): 96 -> 128 cuts CPU MoE 67.7 -> 55.1
# ms/token, decode 6.84 -> 8.52 tok/s (+24%), output coherent (accuracy preserved). 128 and 160
# give IDENTICAL decode throughput (CPU MoE is ~co-equal/overlapped with NPU past 128), so 128
# wins on safety: 8 cores/NUMA headroom vs 160's 4. The old "<=96, >=128 thrashes" note was a
# live-server-contention artifact, not intrinsic. Override with KT_CPUINFER (160 fine too).
# (profiling: doc/zh/dsv4_single_npu/graph_decode_bandwidth_handoff.md, 2026-06-09)
KT_CPUINFER="${KT_CPUINFER:-128}"
PORT="${PORT:-8000}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME

# 单卡：省略 HCCL/DeepEP/MTP 等多卡项；保留 CPU/Ascend 与融合 kernel。
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export IS_DEEPSEEK_V4="${IS_DEEPSEEK_V4:-1}"
export USE_FUSED_COMPRESSOR="${USE_FUSED_COMPRESSOR:-1}"
export LI_KV_DTYPE_INT8="${LI_KV_DTYPE_INT8:-1}"
export USE_PA_DECODE="${USE_PA_DECODE:-1}"
export USE_PA_PREFILL="${USE_PA_PREFILL:-1}"
export USE_FUSED_HC_POST_ASCENDC="${USE_FUSED_HC_POST_ASCENDC:-1}"
export USE_FUSED_HC_PRE_ASCENDC="${USE_FUSED_HC_PRE_ASCENDC:-1}"
export USE_NPU_MOE_GATING_TOP_K="${USE_NPU_MOE_GATING_TOP_K:-1}"
export USE_FUSED_TRANSPOSE_BATCHMATMUL="${USE_FUSED_TRANSPOSE_BATCHMATMUL:-1}"
export USE_ROPE_PARTIAL_IN_PLACE_ASCENDC="${USE_ROPE_PARTIAL_IN_PLACE_ASCENDC:-1}"
export ASCEND_USE_FIA="${ASCEND_USE_FIA:-1}"

export SGLANG_NPU_PROFILE_ENABLE="${SGLANG_NPU_PROFILE_ENABLE:-0}"
export SGLANG_NPU_PROFILE_DECODE_TOKEN="${SGLANG_NPU_PROFILE_DECODE_TOKEN:-2}"
export SGLANG_NPU_PROFILE_DIR="${SGLANG_NPU_PROFILE_DIR:-./npu_results_dbg}"
export SGLANG_NPU_PROFILE_LEVEL="${SGLANG_NPU_PROFILE_LEVEL:-0}"
export SGLANG_NPU_PROFILE_ANALYSE="${SGLANG_NPU_PROFILE_ANALYSE:-0}"
export SGLANG_NPU_PROFILE_DISABLE_GRAPH="${SGLANG_NPU_PROFILE_DISABLE_GRAPH:-1}"
export SGLANG_NPU_PROFILE_KEEP_EAGER_AFTER="${SGLANG_NPU_PROFILE_KEEP_EAGER_AFTER:-1}"
if [[ "${SGLANG_NPU_PROFILE_ENABLE}" == "1" && "${EXTRA_FLAGS:-}" != *"--disable-cuda-graph"* ]]; then
  EXTRA_FLAGS="${EXTRA_FLAGS:+$EXTRA_FLAGS }--disable-cuda-graph"
  echo "[launch] SGLANG_NPU_PROFILE_ENABLE=1: auto append EXTRA_FLAGS=--disable-cuda-graph"
fi

# CANN toolkit + ATB + 自定义算子 vendor 环境（best-effort）。让本脚本自包含，不再隐式依赖 shell
# profile（.bashrc/profile 才 source 它们）：非交互/非登录 shell 或干净 container 直接拉起也能找到算子。
# 机器无 vendors/config.ini，自定义算子靠 ASCEND_CUSTOM_OPP_PATH（由各 set_env.bash 设），故必须 source。
# ⚠️ 这些 vendor 脚本不是 set -u 干净的（如 atb/set_env.sh 引用未定义的 ZSH_VERSION）→ 在本脚本的
#    `set -euo pipefail` 下会触发 unbound variable 直接退出。故 source 期间临时放开 -e/-u，之后恢复。
set +eu
ASCEND_OPP_VENDORS_DIR="${ASCEND_TOOLKIT_HOME}/opp/vendors"
for _kt_env in \
  "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
  /usr/local/Ascend/nnal/atb/set_env.sh \
  "${ASCEND_OPP_VENDORS_DIR}/customize/bin/set_env.bash" \
  "${ASCEND_OPP_VENDORS_DIR}/custom_transformer/bin/set_env.bash"; do
  if [[ -f "${_kt_env}" ]]; then
    # shellcheck source=/dev/null
    source "${_kt_env}"
  fi
done
unset _kt_env
set -eu

ulimit -n 65536 2>/dev/null || true

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "[launch] 保留环境变量 ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
elif [[ -n "${NPU_DEVICE_ID:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID}"
  echo "[launch] 已设置 ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}（物理卡，进程内为逻辑 npu:0）"
else
  echo "[launch] 提示: 未设置 NPU_DEVICE_ID 且未设置 ASCEND_RT_VISIBLE_DEVICES，将使用系统当前可见的全部 NPU；"
  echo "[launch]       单卡服务通常仍绑定逻辑设备 0（常为物理 0 号卡）。若 0 号卡被占用，请执行:"
  echo "[launch]         NPU_DEVICE_ID=2 bash $0   或   bash $0 2"
fi

echo "[launch] REPO=$REPO"
echo "[launch] PYTHONPATH head: ${PYTHONPATH%%:*}"
echo "[launch] chunked-prefill-size=${CHUNKED_PREFILL_SIZE}（正数须为 page_size 倍数；见脚本头注释）"
echo "[launch] kt-weight-path template=${KT_GGUF_TEMPLATE}"
echo "[launch] quantization=${QUANTIZATION} IS_DEEPSEEK_V4=${IS_DEEPSEEK_V4:-}"
echo "[launch] SGLANG_NPU_PROFILE_ENABLE=${SGLANG_NPU_PROFILE_ENABLE} DECODE_TOKEN=${SGLANG_NPU_PROFILE_DECODE_TOKEN}"
"${PYTHON_BIN}" -c "import sglang; print('[launch] sglang file:', sglang.__file__)"

# EXTRA_FLAGS 用于临时附加任意 sglang.launch_server 参数（不需要改脚本本体），例如：
#   EXTRA_FLAGS="--disable-cuda-graph"   bash tools/launch_ds4flash_npu.sh   # 关图捕获走 eager
#   EXTRA_FLAGS="--cuda-graph-bs 2"      bash tools/launch_ds4flash_npu.sh
# 调试 NPU aicore / aclnn 错误时可再叠加： ASCEND_LAUNCH_BLOCKING=1 bash ...
# EXTRA_FLAGS is a user-supplied string; split it on whitespace into separate args via an array
# (so we can quote the expansion and avoid shellcheck SC2086).
read -ra _EXTRA_FLAGS <<< "${EXTRA_FLAGS:-}"
if [[ ${#_EXTRA_FLAGS[@]} -gt 0 ]]; then
  echo "[launch] EXTRA_FLAGS=${EXTRA_FLAGS:-}"
fi
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --device npu \
  --tensor-parallel-size 1 \
  --page-size 128 \
  --attention-backend ascend \
  --quantization "$QUANTIZATION" \
  --disable-shared-experts-fusion \
  --dtype bfloat16 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --max-prefill-tokens 65535 \
  --context-length 65536 \
  --watchdog-timeout 18000 \
  --skip-server-warmup \
  --kt-method LLAMAFILE \
  --kt-num-gpu-experts 32 \
  --kt-weight-path "$KT_GGUF_TEMPLATE" \
  --kt-threadpool-count 8 \
  --kt-cpuinfer "$KT_CPUINFER" \
  --max-running-requests 1 \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${_EXTRA_FLAGS[@]}"
# cuda-graph 已启用：kt-kernel ACL callback worker + kt_ep_wrapper NPU graph
# host callback（见 kt-kernel/cpu_backend/ascend_callback_worker.*）。
# frequency placement 示例：
#   EXTRA_FLAGS="--kt-expert-placement-strategy frequency --kt-activation-freq-path /path/to/activation_freq.pt"
# 调试同步路径：KT_FORCE_SYNC_SUBMIT=1；回退无 graph：EXTRA_FLAGS="--disable-cuda-graph"
# 生产勿开 KT_DEBUG_HYBRID_MOE / KT_DEBUG_MOE_OUT。graph 性能用 msprof，勿长期开 SGLANG_NPU_PROFILE_ENABLE。
