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
# Start the SGLang HTTP server on a single NPU with KT(LLAMAFILE) offload
# (DeepSeek-V4-Flash: W8A8 on the NPU, MXFP4 experts on the CPU).
#
# Quantisation: --quantization compressed-tensors, matching the on-disk W8A8
# (compressed-tensors / int-quantized).
#
# Usage (from any directory):
#   bash /path/to/ktransformers-AK/tools/launch_ds4flash_npu.sh
#   bash .../launch_ds4flash_npu.sh 3          # same as NPU_DEVICE_ID=3: pick a physical NPU
#
# Common overrides (environment variables):
#   REPO              defaults to the repo root this script lives in
#   MODEL_PATH        required: W8A8 weight directory (NPU side)
#   KT_GGUF_TEMPLATE  required: MXFP4 GGUF template, e.g.
#                     /your/cache/dsv4_layer{layer_idx}_mxfp4.gguf (single-quoted)
#   PORT              default 8020 (matches decode_throughput_test.sh / gpqa_accuracy_repeat.sh)
#   ASCEND_TOOLKIT_HOME  default /usr/local/Ascend/ascend-toolkit/latest
#   NPU_DEVICE_ID     optional physical NPU index (e.g. 2); exports ASCEND_RT_VISIBLE_DEVICES
#   CHUNKED_PREFILL_SIZE  default 2048 (must be a multiple of page-size=128, and >= page-size).
#                         Do NOT pass -1: the KT(LLAMAFILE) C++ MoE sizes its fp32 output buffer
#                         from max_possible_qlen() = max(max_len, group_max_len), and -1 collapses
#                         to 1, so any prefill with qlen>1 writes past the heap allocation and
#                         trips a glibc tcache abort.
#   QUANTIZATION      default compressed-tensors (matches the baseline).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/ensure_kt_kernel.sh
source "${SCRIPT_DIR}/ensure_kt_kernel.sh"
ensure_kt_kernel "$REPO"
export PYTHONPATH="${REPO}/third_party/sglang/python:${REPO}/kt-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

# ---------- select the Python interpreter ----------
# Shares resolve_python_bin with e2e_preflight.sh and asks for the same module list, so the
# preflight resolves the very interpreter this script launches the server with. Selection is by
# importable modules rather than by name: on a clean image the python3 on PATH is the system
# python and has no numpy/torch/torch_npu/sglang.
# shellcheck source=tools/python_env.sh
source "${SCRIPT_DIR}/python_env.sh"
PYTHON_BIN="$(resolve_python_bin launch numpy torch torch_npu sglang)" || exit 2
export PYTHON_BIN
echo "[launch] PYTHON_BIN=${PYTHON_BIN}"

# Optional first argument: a bare number is taken as the physical NPU index (same as NPU_DEVICE_ID)
if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ && -z "${NPU_DEVICE_ID:-}" ]]; then
  NPU_DEVICE_ID="$1"
  shift
fi

# MODEL_PATH is required; environment-specific paths are never hardcoded. Point it at the W8A8
# weight directory (NPU side).
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "[launch][ERROR] MODEL_PATH is not set. Point it at the W8A8 weight directory (NPU side), e.g." >&2
  echo "[launch][ERROR]   MODEL_PATH=/your/path/DeepSeek-V4-Flash-W8A8 bash $0" >&2
  exit 2
fi

# Streaming-prefill checkpoints (only read when KT_PREFILL_STREAM=1). kt_stream_prefill.py
# reads config.json / model.safetensors.index.json from these; its own defaults are the old
# image's /workspace/... paths, so DERIVE them from MODEL_PATH here (any new container just
# needs MODEL_PATH). _CKPT = the W8A8 serving ckpt = MODEL_PATH; _MXFP4_CKPT = the native
# MXFP4 source = sibling dir with the -W8A8 suffix stripped. Explicit env still wins.
export KT_PREFILL_STREAM_CKPT="${KT_PREFILL_STREAM_CKPT:-$MODEL_PATH}"
export KT_MXFP4_CKPT="${KT_MXFP4_CKPT:-${MODEL_PATH%-W8A8}}"

# ---------- KT MoE: everything on by default (each can be overridden explicitly) ----------
# All on = depool + dynamic hot experts + streaming prefill + side stream + GGUF dedup.
# Performance and accuracy figures are in the design note.
# Lightweight prefix-32 baseline (no mxfp4 pool):
#   KT_MXFP4_DEPOOL=0 KT_MXFP4_GGUF_DEDUP=0 KT_DYNAMIC_RESIDENT=0 KT_PREFILL_STREAM=0
export KT_MXFP4_DEPOOL="${KT_MXFP4_DEPOOL:-1}"
# requires depool; the default GGUF template below selects mxfp4
export KT_MXFP4_GGUF_DEDUP="${KT_MXFP4_GGUF_DEDUP:-1}"
export KT_DYNAMIC_RESIDENT="${KT_DYNAMIC_RESIDENT:-1}"
export KT_PREFILL_STREAM="${KT_PREFILL_STREAM:-1}"
export KT_SIDE_STREAM="${KT_SIDE_STREAM:-1}"
# Minimum chunk length that enables streaming prefill (kt_stream_prefill.py:_T): a prefill chunk
# takes the streaming path only when its token count reaches this value. The dynamic hot-expert
# slots (KT_DYNAMIC_RESIDENT) are refreshed on that same streaming path, so a short prefill below
# the threshold runs hybrid and does not update the hot experts. Lower it (e.g. 128) to give
# shorter prompts streaming plus dynamic hot experts; raise it to restrict streaming to longer
# prefills. Default 512, which is the code default (no regression).
export KT_PREFILL_STREAM_THRESHOLD="${KT_PREFILL_STREAM_THRESHOLD:-512}"

# KT_GGUF_TEMPLATE is required: the MXFP4 GGUF template for the CPU side (the batch converter
# writes dsv4_layer{L}_mxfp4.gguf). {layer_idx} is a literal placeholder and must be single-quoted:
# writing "${KT_GGUF_TEMPLATE:-...{layer_idx}.gguf}" makes bash treat the first `}` of
# {layer_idx} as the end of `${...:-}`, turning the path into `...{layer_idx.gguf}`.
# With depool on (the default) the CPU experts should read MXFP4 GGUF (3.4GB/layer, against
# 6.8GB/layer for Q8_0): the CPU MoE is memory-bandwidth bound, and Q8_0 roughly doubles the
# per-token CPU time during decode.
if [[ -z "${KT_GGUF_TEMPLATE:-}" ]]; then
  echo "[launch][ERROR] KT_GGUF_TEMPLATE is not set. Point it at the MXFP4 GGUF template" \
       "(CPU side), single-quoted, e.g." >&2
  echo "[launch][ERROR]   KT_GGUF_TEMPLATE='/your/cache/dsv4_layer{layer_idx}_mxfp4.gguf' bash $0" >&2
  exit 2
fi
# Must be exported: besides --kt-weight-path (CPU MoE), GGUF dedup (KT_MXFP4_GGUF_DEDUP=1) reads
# KT_GGUF_TEMPLATE from os.environ to reuse the CPU's mmapped GGUF. Without the export, dedup
# reports "template empty" and falls back to building a codes pool.
export KT_GGUF_TEMPLATE
# >= typical prompts (GPQA max 2577); the NSA compressor needs one chunk to cover the prompt
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
# CPU MoE is memory-bandwidth-bound; more threads raise effective DDR bandwidth, but the pool
# must fit the host's NUMA topology (see KT_THREADPOOL_COUNT below). Defaults target the
# single-NUMA A3 host: 1 subpool, 32 threads -- portable, and also runs on multi-NUMA
# hosts, just without their extra bandwidth. On an 8-NUMA host use
# `KT_THREADPOOL_COUNT=8 KT_CPUINFER=128` (16 threads/NUMA); do NOT take all cores, the spin
# threads, the NPU host callback and python/OS need headroom or the pool thrashes.
KT_CPUINFER="${KT_CPUINFER:-32}"
# kt-kernel builds one subpool per threadpool_count and binds subpool i to NUMA node i
# (numa_nodes defaults to 0,1,...,count-1), so this MUST NOT exceed the host's NUMA node count,
# otherwise startup fails with "NUMA node N not found" + set_mempolicy errors.
# threads/subpool = KT_CPUINFER / KT_THREADPOOL_COUNT.
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-1}"
PORT="${PORT:-8020}"
# ASCEND_TOOLKIT_HOME is the SINGLE anchor — ATB(nnal), opp vendors, custom-op paths all derive
# from it. Take it from the environment (CANN's set_env.sh exports both ASCEND_TOOLKIT_HOME and
# ASCEND_HOME_PATH); if absent, auto-detect known layouts; if still not found, HARD-FAIL with
# guidance instead of silently using a wrong hardcoded path (that's what broke on new containers).
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH:-}"
fi
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  for _cand in /usr/local/Ascend/ascend-toolkit/latest \
               "${HOME}"/Ascend/ascend-toolkit/latest \
               "${HOME}"/Ascend/cann-* \
               /home/*/Ascend/cann-* ; do
    [[ -f "${_cand}/set_env.sh" ]] && { ASCEND_TOOLKIT_HOME="${_cand}"; break; }
  done
  unset _cand
fi
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" || ! -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  echo "[launch][ERROR] No valid ASCEND_TOOLKIT_HOME (CANN root). Set it explicitly," \
       "or source CANN's set_env.sh first:" >&2
  echo "  export ASCEND_TOOLKIT_HOME=/path/to/Ascend/cann-9.0.0   # use your actual path" >&2
  exit 1
fi
export ASCEND_TOOLKIT_HOME

# NSA compressor calling convention depends on the CANN version:
#   CANN 9.0.0+ -> public 18-arg single-state op   (KT_NSA_COMPRESSOR_MODE=single)
#   CANN 8.5.0  -> private 19-arg split-state op   (KT_NSA_COMPRESSOR_MODE=split, the code default)
# Getting this wrong makes the NSA compressor call fail, so derive it from the installed toolkit
# version unless the user set it explicitly.
if [[ -z "${KT_NSA_COMPRESSOR_MODE:-}" ]]; then
  _cann_ver="$(sed -n 's/^version=\([0-9]\+\).*/\1/p' \
                 "${ASCEND_TOOLKIT_HOME}"/*/ascend_toolkit_install.info 2>/dev/null | head -1)"
  if [[ -n "${_cann_ver}" && "${_cann_ver}" -ge 9 ]]; then
    export KT_NSA_COMPRESSOR_MODE=single
  else
    export KT_NSA_COMPRESSOR_MODE=split
  fi
  echo "[launch] KT_NSA_COMPRESSOR_MODE=${KT_NSA_COMPRESSOR_MODE} (derived from CANN ${_cann_ver:-unknown})"
  unset _cann_ver
else
  echo "[launch] KT_NSA_COMPRESSOR_MODE=${KT_NSA_COMPRESSOR_MODE} (explicitly set)"
fi
echo "[launch] ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME} (ATB and vendor paths derive from it)"
# KML (Kunpeng Math Library) — only on Kunpeng hosts; prepend only if present so a missing
# dir doesn't sit dead in LD_LIBRARY_PATH. Override with KML_LIB_DIR.
_KML_LIB_DIR="${KML_LIB_DIR:-/usr/local/kml/lib}"
[[ -d "${_KML_LIB_DIR}" ]] && export LD_LIBRARY_PATH="${_KML_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Drop any inherited proxy. sglang's startup warmup (SKIP_WARMUP=0) POSTs to the server's OWN
# port; with http_proxy=127.0.0.1:7890 set, that localhost call is intercepted -> 502 ->
# "warmup error: AssertionError res=<Response [502]>" -> Initialization failed (server exits).
# This is exactly why warmup kept being disabled with SKIP_WARMUP=1. A local inference server
# never needs an outbound proxy, so just unset them (also spares every curl the --noproxy dance).
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# Aligned with script/launch_ds4flash_sglang.sh: a single card omits HCCL/DeepEP/MTP and keeps the
# CPU/Ascend settings and the fused kernels.
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

# Source the CANN toolkit, ATB and custom-operator vendor environments here rather than relying on
# the shell profile, so a non-interactive shell or a clean container also finds the operators
# (they are located through ASCEND_CUSTOM_OPP_PATH, which the vendor set_env.bash files set).
# -e/-u are relaxed while sourcing: the vendor scripts are not `set -u` clean and would abort.
set +eu
ASCEND_OPP_VENDORS_DIR="${ASCEND_TOOLKIT_HOME}/opp/vendors"
# ATB (nnal) set_env sits outside the toolkit directory, in the `nnal/atb` sibling of the Ascend
# root, and its exact path varies per image. ATB_SET_ENV overrides the search below.
if [[ -z "${ATB_SET_ENV:-}" ]]; then
  # .../Ascend/cann-9.0.0 and .../Ascend/ascend-toolkit both resolve to .../Ascend
  _ascend_root="$(dirname "${ASCEND_TOOLKIT_HOME}")"
  [[ "$(basename "${_ascend_root}")" == "ascend-toolkit" ]] && _ascend_root="$(dirname "${_ascend_root}")"
  for _atb in "${_ascend_root}/nnal/atb/set_env.sh" \
              /usr/local/Ascend/nnal/atb/set_env.sh; do
    [[ -f "${_atb}" ]] && { ATB_SET_ENV="${_atb}"; break; }
  done
fi
[[ -n "${ATB_SET_ENV:-}" ]] && echo "[launch] ATB set_env: ${ATB_SET_ENV}" || echo "[launch][warn] ATB set_env not found (ATB ops may be unavailable); set ATB_SET_ENV=<path> if needed"
for _kt_env in \
  "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
  "${ATB_SET_ENV:-/nonexistent}" \
  "${ASCEND_OPP_VENDORS_DIR}/customize/bin/set_env.bash" \
  "${ASCEND_OPP_VENDORS_DIR}/custom_transformer/bin/set_env.bash"; do
  if [[ -f "${_kt_env}" ]]; then
    # shellcheck source=/dev/null
    source "${_kt_env}"
  fi
done
unset _kt_env _atb _ascend_root
set -eu

ulimit -n 65536 2>/dev/null || true

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "[launch] Keeping ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} from the environment"
elif [[ -n "${NPU_DEVICE_ID:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID}"
  echo "[launch] Set ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (physical card; logical npu:0 in-process)"
else
  echo "[launch] Note: neither NPU_DEVICE_ID nor ASCEND_RT_VISIBLE_DEVICES is set, so every visible NPU is used;"
  echo "[launch]       a single-card server still binds logical device 0 (usually physical card 0). If card 0 is busy:"
  echo "[launch]         NPU_DEVICE_ID=2 bash $0   or   bash $0 2"
fi

echo "[launch] REPO=$REPO"
echo "[launch] PYTHONPATH head: ${PYTHONPATH%%:*}"
echo "[launch] chunked-prefill-size=${CHUNKED_PREFILL_SIZE}" \
     "(a positive value must be a multiple of page_size; see the header)"
echo "[launch] kt-weight-path template=${KT_GGUF_TEMPLATE}"
echo "[launch] quantization=${QUANTIZATION} IS_DEEPSEEK_V4=${IS_DEEPSEEK_V4:-}"
echo "[launch] SGLANG_NPU_PROFILE_ENABLE=${SGLANG_NPU_PROFILE_ENABLE} DECODE_TOKEN=${SGLANG_NPU_PROFILE_DECODE_TOKEN}"
"${PYTHON_BIN}" -c "import sglang; print('[launch] sglang file:', sglang.__file__)"

# EXTRA_FLAGS appends arbitrary sglang.launch_server arguments without editing this script, e.g.
#   EXTRA_FLAGS="--disable-cuda-graph"   bash tools/launch_ds4flash_npu.sh   # eager, no graph capture
#   EXTRA_FLAGS="--cuda-graph-bs 2"      bash tools/launch_ds4flash_npu.sh
# When debugging NPU aicore / aclnn errors, add ASCEND_LAUNCH_BLOCKING=1 as well.
# Split on whitespace into an array before expanding: expanding the bare variable would let
# pathname expansion rewrite any argument containing * ? [] (e.g.
# --kt-activation-freq-path /path/*.pt), and would need a shellcheck suppression to pass the scan.
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
read -r -a EXTRA_ARGS <<< "${EXTRA_FLAGS}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[launch] EXTRA_FLAGS=${EXTRA_FLAGS}"
fi
# SKIP_WARMUP=1 passes --skip-server-warmup to skip the startup warmup; =0 enables it.
# Streaming prefill never exercises the CPU MoE, so the server would keep a cold kt_kernel and a
# noticeably lower decode rate. With KT_PREFILL_STREAM=1 the warmup is therefore enabled and the
# first pass is forced through the hybrid path (KT_STREAM_WARMUP=1) to warm kt_kernel; one pass is
# enough. Both can be overridden explicitly; non-streaming runs keep the baseline (warmup off).
if [[ "${KT_PREFILL_STREAM:-}" == "1" ]]; then
  SKIP_WARMUP="${SKIP_WARMUP:-0}"
  export KT_STREAM_WARMUP="${KT_STREAM_WARMUP:-1}"
fi
SKIP_WARMUP="${SKIP_WARMUP:-1}"
WARMUP_ARGS=(--skip-server-warmup)
if [[ "${SKIP_WARMUP}" == "0" ]]; then
  WARMUP_ARGS=()
fi
echo "[launch] SKIP_WARMUP=${SKIP_WARMUP} (warmup_flag='${WARMUP_ARGS[*]-}') KT_STREAM_WARMUP=${KT_STREAM_WARMUP:-0}"
# Tunable env:
#   KT_NUM_GPU_EXPERTS  experts kept on the NPU per layer, default 32; each extra one costs about
#       1.0GB of HBM. Measured ceiling at context 65536 is 40; beyond that, lower --context-length.
#   MEM_FRACTION  default 0.81. Sets aside (1 - MEM_FRACTION) of the free HBM measured before the
#       weights load; the KV pool gets what is left, so a lower value leaves more for the prefill
#       activations. 0.81 sizes the pool to ~131k tokens, twice --context-length 65536; the floor
#       is where max_total_num_tokens drops below the context length. A 32k prefill needs it;
#       see the deployment guide for the measured values.
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
  --mem-fraction-static "${MEM_FRACTION:-0.81}" \
  --disable-radix-cache \
  --max-prefill-tokens 65535 \
  --context-length 65536 \
  --watchdog-timeout 18000 \
  "${WARMUP_ARGS[@]}" \
  --kt-method LLAMAFILE \
  --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS:-32}" \
  --kt-weight-path "$KT_GGUF_TEMPLATE" \
  --kt-threadpool-count "$KT_THREADPOOL_COUNT" \
  --kt-cpuinfer "$KT_CPUINFER" \
  --max-running-requests 1 \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
# cuda-graph is enabled (kt-kernel ACL callback worker + the NPU graph host callback in
# kt_ep_wrapper).
# frequency placement example:
#   EXTRA_FLAGS="--kt-expert-placement-strategy frequency --kt-activation-freq-path /path/to/activation_freq.pt"
# To fall back to no graph: EXTRA_FLAGS="--disable-cuda-graph". KT_DEBUG_* and
# SGLANG_NPU_PROFILE_ENABLE are for debugging only and should stay off in production.
