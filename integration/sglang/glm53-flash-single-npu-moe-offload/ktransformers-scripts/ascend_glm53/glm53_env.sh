#!/usr/bin/env bash
# Environment resolution for GLM-5.3-Flash on a single Ascend die with the routed
# experts offloaded to host DDR.  Sourced by setup.sh / serve.sh / verify.sh.
#
# Idempotent and re-sourceable: on a bare image the custom-op vendors do not exist
# when this is first sourced, and setup.sh installs them mid-run.
#
# `bash glm53_env.sh --show` prints everything it resolved.  Read that before serving.

# Callers run with `set -u`; make every variable this file tests safe to read unset.
for _v in ASCEND_INSTALL_ROOT SGLANG_REPO GLM53_THREADPOOL_COUNT GLM53_CPUINFER \
          GLM53_ENV_ROOT GLM53_EVAL_DIR GLM53_VENV GLM53_ARTIFACT_ROOT \
          CANN_VENDORS_DIR CANN_RECIPES_REPO OPS_TRANSFORMER_REPO \
          SGL_KERNEL_NPU_REPO SGL_KERNEL_NPU_TAG \
          GLM53_PYTHON GLM53_GGUF_TEMPLATE GLM53_SOC LD_PRELOAD \
          GLM53_MAX_TOTAL_TOKENS \
          PYTHONPATH LD_LIBRARY_PATH ASCEND_CUSTOM_OPP_PATH; do
  eval ": \"\${${_v}:=}\""
done
unset _v

# ---------------------------------------------------------------- CANN discovery
if [ -z "${ASCEND_INSTALL_ROOT}" ]; then
  # /home/developer is the Ascend container image's own install location, not a personal
  # home -- $HOME on this box is elsewhere, so dropping it breaks CANN discovery outright.
  for _r in "${HOME}/Ascend" /home/developer/Ascend /usr/local/Ascend /opt/Ascend; do
    if [ -r "${_r}/ascend-toolkit/set_env.sh" ]; then ASCEND_INSTALL_ROOT="${_r}"; break; fi
  done
fi
if [ -z "${ASCEND_INSTALL_ROOT}" ]; then
  echo "[glm53_env] FATAL: no CANN toolkit found (looked for ascend-toolkit/set_env.sh under" \
       "\$HOME/Ascend /home/developer/Ascend /usr/local/Ascend /opt/Ascend).' \
       ' Set ASCEND_INSTALL_ROOT to override." >&2
  return 1 2>/dev/null || exit 1
fi
export ASCEND_INSTALL_ROOT

CANN_ROOT="$(readlink -f "${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest" 2>/dev/null)"
case "${CANN_ROOT}" in */aarch64-linux|*/x86_64-linux) CANN_ROOT="$(dirname "${CANN_ROOT}")" ;; esac
export CANN_ROOT
# The package metadata and the compiler component can disagree -- on this image
# ascend_toolkit_install.info says 9.2.0 while the compiler is 9.1.0.  Report both;
# the compiler version is the one the operator packages were built against.
export CANN_VERSION="$(sed -n 's/^version=//p' \
  "${ASCEND_INSTALL_ROOT}/ascend-toolkit/latest/"*/ascend_toolkit_install.info 2>/dev/null | head -1)"
export CANN_COMPILER_VERSION="$(sed -n 's/^Version=//p' \
  "${CANN_ROOT}/compiler/version.info" 2>/dev/null | head -1)"

# ---------------------------------------------------------------- repositories
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
export KTRANSFORMERS_REPO="${KTRANSFORMERS_REPO:-$(cd "${_here}/../../.." && pwd -P)}"
export GLM53_WORKSPACE="${GLM53_WORKSPACE:-$(dirname "${KTRANSFORMERS_REPO}")}"
export GLM53_TOOLS="${_here}"

# The venv, the custom-op vendor packages and the evaluation corpora live in a sibling
# "env" tree, not inside the repo -- they are too big to vendor and are shared between
# checkouts.  Derive it from the repo location: nothing in this file may hardcode a
# path under someone's home, because that makes the tool unrunnable for anyone else.
# Override with GLM53_ENV_ROOT, or more narrowly with GLM53_VENV / GLM53_PYTHON /
# GLM53_OPP_CUSTOM_DIRS / GLM53_EVAL_DIR.
if [ -z "${GLM53_ENV_ROOT}" ]; then
  for _e in "${GLM53_WORKSPACE}/env" "${GLM53_WORKSPACE}"/*/env; do
    if [ -d "${_e}" ]; then GLM53_ENV_ROOT="${_e}"; break; fi
  done
fi
export GLM53_ENV_ROOT
# Real-prose corpus for run_ppl.py.  ⚠ Repetitive filler routes to far fewer
# experts than real text and inflates the resident hit rate, so the numbers stop
# being comparable.
export GLM53_EVAL_DIR="${GLM53_EVAL_DIR:-${GLM53_ENV_ROOT:+${GLM53_ENV_ROOT}/eval}}"

# sglang: prefer the submodule, which is pinned to the branch carrying the GLM-5.3
# NPU port.  A sibling checkout is the fallback for a from-source layout.
if [ -z "${SGLANG_REPO}" ]; then
  for _c in "${KTRANSFORMERS_REPO}/third_party/sglang" "${GLM53_WORKSPACE}/sglang-dllm" \
            "${GLM53_WORKSPACE}/wt-int8-singlecard"; do
    if [ -f "${_c}/python/sglang/__init__.py" ]; then SGLANG_REPO="${_c}"; break; fi
  done
fi
export SGLANG_REPO

# Everything this toolkit writes lands under one root, off /mnt/workspace (which runs
# near full) and off the repo. Logs, the kt-kernel build tree and bench JSON all derive
# from it, so a reader has one path to redirect. (An "artifacts" variable used to be
# exported
# here beside it and was read by nothing at all.)
export GLM53_ARTIFACT_ROOT="${GLM53_ARTIFACT_ROOT:-/var/tmp/glm53}"
export GLM53_LOG_DIR="${GLM53_LOG_DIR:-${GLM53_ARTIFACT_ROOT}/logs}"

# ---------------------------------------------------------------- weights
# Three artifacts, and only two of them are needed to serve.
#
#   GLM53_MODEL_PATH   INT8 W8A8, --model-path.  Everything on the die: attention, the
#                      three dense layers, and the resident experts.  A3 has no fp8
#                      hardware at all, so the vendor FP8 release cannot serve directly.
#   GLM53_MXFP4_CKPT   MXFP4 (compressed-tensors), the SOURCE for the GGUF conversion.
#                      Also read at serve time on the streaming path: serve.sh passes it
#                      as KT_MXFP4_CKPT for the on-device MXFP4 -> W8A8-NZ convert.
#   GLM53_GGUF_DIR     the per-layer MXFP4 GGUF set the CPU MoE memory-maps, produced
#                      from GLM53_MXFP4_CKPT by setup.sh's `gguf` step.
export GLM53_MODEL_ROOT="${GLM53_MODEL_ROOT:-/mnt/workspace/models}"
export GLM53_MODEL_PATH="${GLM53_MODEL_PATH:-${GLM53_MODEL_ROOT}/GLM-5.3-Flash-W8A8}"
export GLM53_MXFP4_CKPT="${GLM53_MXFP4_CKPT:-${GLM53_MODEL_ROOT}/GLM-5.3-Flash-MXFP4}"
export GLM53_GGUF_DIR="${GLM53_GGUF_DIR:-${GLM53_MODEL_ROOT}/GLM-5.3-Flash-MXFP4-gguf}"
export GLM53_GGUF_NAME_PREFIX="${GLM53_GGUF_NAME_PREFIX:-glm53_layer}"
export GLM53_GGUF_NAME_SUFFIX="${GLM53_GGUF_NAME_SUFFIX:-_mxfp4}"
# NOTE: the literal "{layer_idx}" cannot be written inside a ${VAR:-default}; its
# closing brace terminates the parameter expansion and silently mangles the path.
# kt_ep_wrapper.resolve_kt_weight_path_for_layer() formats this per layer.
if [ -z "${GLM53_GGUF_TEMPLATE}" ]; then
  GLM53_GGUF_TEMPLATE="${GLM53_GGUF_DIR}/${GLM53_GGUF_NAME_PREFIX}{layer_idx}${GLM53_GGUF_NAME_SUFFIX}.gguf"
fi
export GLM53_GGUF_TEMPLATE

# Layer range that carries routed experts *and is served*.  GLM-5.3-Flash has 45 layers
# of which 0-2 are dense (first_k_dense_replace=3), so experts live on 3..44.  The
# checkpoint ALSO carries a full 288-expert set for layer 45, the MTP/nextn head, which
# is one past num_hidden_layers and is not served here -- converting it would cost
# another 3.85 GB for nothing.
export GLM53_MOE_LAYER_START="${GLM53_MOE_LAYER_START:-3}"
export GLM53_MOE_LAYER_END="${GLM53_MOE_LAYER_END:-44}"

# ---------------------------------------------------------------- host sizing
# One thread pool per NUMA node.
#
# WARNING about measuring on a development box.  The single-card deployment target is
# one die with ~40 cores, 1 NUMA node and ~200 GB of DDR.  A shared development host
# has 8 NUMA nodes and 320 cores, so these autodetect to 8/128 and any throughput
# number taken with them is NOT the number the target will see: the CPU MoE is host
# memory-bandwidth bound and would get 8 nodes' worth.  Set GLM53_THREADPOOL_COUNT=1
# and GLM53_CPUINFER=32 to measure something comparable.
if [ -z "${GLM53_THREADPOOL_COUNT}" ]; then
  GLM53_THREADPOOL_COUNT="$(find /sys/devices/system/node -maxdepth 1 -type d \
    -name 'node[0-9]*' 2>/dev/null | wc -l)"
  [ "${GLM53_THREADPOOL_COUNT}" -ge 1 ] 2>/dev/null || GLM53_THREADPOOL_COUNT=1
fi
export GLM53_THREADPOOL_COUNT
if [ -z "${GLM53_CPUINFER}" ]; then
  GLM53_CPUINFER=$(( GLM53_THREADPOOL_COUNT * 16 ))
  _cap=$(( $(nproc 2>/dev/null || echo 8) * 3 / 4 ))
  [ "${GLM53_CPUINFER}" -gt "${_cap}" ] && GLM53_CPUINFER="${_cap}"
  [ "${GLM53_CPUINFER}" -lt 1 ] && GLM53_CPUINFER=1
fi
export GLM53_CPUINFER

# ---------------------------------------------------------------- serving knobs
export GLM53_NPU_DEVICE_ID="${GLM53_NPU_DEVICE_ID:-0}"
export GLM53_HOST="${GLM53_HOST:-127.0.0.1}"
export GLM53_PORT="${GLM53_PORT:-30013}"

# Resident experts per layer.  Measured from the checkpoint's own safetensors headers:
# one expert INDEX costs 0.9925 GiB across the 42 SERVED layers, from a fit over three
# measured loads.  (An earlier 1.009 divided by 43 expert-bearing layers, which counted
# the MTP layer 45 that is never served.)  Everything else on the die --
# attention 10.81, embed+lm_head 2.36, the three dense MLPs 1.08, shared experts 1.01,
# MTP 0.20 -- is 15.60 GiB.  So the die holds 15.60 + 0.9925*N GiB of weights, before
# the streaming path's convert slot:
#
#     N=16 -> 31.5 GiB    N=24 -> 39.4 GiB    N=32 -> 47.4 GiB    N=40 -> 55.3 GiB
# Measured fit:  usage(K, slot) = 15.60 + 6.75*[streaming slot] + 0.9925*K  GiB
#
# 64 GiB per die, ~61.3 usable.  32 leaves ~13 GiB for KV, the KDA conv/mamba state and
# graph buffers; 40 leaves ~5 and does not fit with a useful context.
#
# This file is sourced more than once per process -- a caller may source it, then
# serve.sh sources it again -- and the three values below are DERIVED from
# GLM53_PREFILL_STREAM. Once exported they are sticky, so a caller that sources this,
# then sets GLM53_PREFILL_STREAM=1, then launches, would silently keep the hybrid
# budget: 32 resident and 0.85 instead of 28 and 0.95. The server then refuses to start
# ("Raise --mem-fraction-static above 0.887"), which is the lucky case; the unlucky one
# starts and falls back to hybrid without saying so.
#
# So stamp which branch produced them and re-derive if the flag has since changed. Only a
# value we derived may be cleared: remember what we set and drop it only if it is still
# exactly that, so a value the caller chose looks different and survives.
_glm53_undrive() {   # $1 = var name, $2 = name of the stamp holding our last value
  local cur stamp
  eval "cur=\${$1:-}"; eval "stamp=\${$2:-}"
  [ -n "${stamp}" ] && [ "${cur}" = "${stamp}" ] && unset "$1"
  return 0
}
if [ -n "${GLM53_DERIVED_FOR_STREAM:-}" ] \
   && [ "${GLM53_DERIVED_FOR_STREAM}" != "${GLM53_PREFILL_STREAM:-0}" ]; then
  _glm53_undrive GLM53_MEM_FRACTION     _GLM53_DRV_MEMFRAC
  _glm53_undrive GLM53_MAX_TOTAL_TOKENS _GLM53_DRV_KV
  _glm53_undrive GLM53_CHUNKED_PREFILL_SIZE _GLM53_DRV_CHUNK
fi
export GLM53_DERIVED_FOR_STREAM="${GLM53_PREFILL_STREAM:-0}"

# K = 32 on both paths, and the reason it is not branch-dependent is worth keeping.
#
# Streaming prefill reserves a slot big enough to hold ALL 288 experts of one layer as
# W8A8-NZ -- [E,H,2I] + [E,I,H] int8 = 4.83 + 2.42 = 7.25 GiB -- before the KV pool is
# sized. That is 7.2 resident expert indices, so this once ran at K=28 on the streaming
# path: at 32 the server came up and then left 337 MiB free, every layer's convert failed
# its 514 MiB transient, and it fell back to hybrid -- no crash, nothing in the log except
# the [KT_STREAM] log lines (that is a log prefix, not a variable), which is the
# worst way to fail.
#
# What made 32 fit was capping the KV pool (--max-total-tokens 40960) and landing the
# per-chunk H2D. K=32 now measures 54.11 GiB used with 5.50 GiB available after graph
# capture, and passes both streaming gates plus verify.sh on its 14379-token prompt.
#
# ⚠ That last clause holds at chunked-prefill 6144, not at the 8192 this was written
# under, where the same K=32 aborts on the KDA workspace before verify.sh finishes.
# K and the chunk size spend the same headroom; neither can be read on its own.
export GLM53_NUM_GPU_EXPERTS="${GLM53_NUM_GPU_EXPERTS:-32}"

# Static memory fraction, and the KV cap that has to travel with it.
#
# The hybrid path wants 0.85: weights are 47.8 GiB of the 61.3 usable, KV takes what is
# left and there is several GiB of slack for the dynamic allocations (the KDA layers ask
# the Triton runtime for a ~3 GiB workspace on a multi-thousand-token prefill).
#
# Streaming prefill reserves a 7.25 GiB convert-output slot BEFORE KV sizing, and that
# does not fit at 0.85 -- the scheduler refuses to start ("Loaded weights leave no GPU
# memory for the KV cache", minimum viable measured 0.887).  Raising the fraction alone
# swings too far the other way: KV then grows to fill it (190656 tokens on one A3 die,
# far more than a 32768 context with one running request can use) and the first long
# prefill aborts on that Triton workspace.  So streaming needs BOTH -- a higher fraction
# so the slot fits, and a KV pool capped near the context length so the slack survives.
# Measured on one die (61.28 GiB usable): weights+slot 54.11, KV@40960 0.65, decode graph
# ~2.0, leaving ~4.5 GiB dynamic.
#
# ⚠ The chunk size is part of this budget, not an independent knob. The KDA workspace
# scales with the tokens in one forward, and on the streaming path there is only ~5.5 GiB
# left for it. Measured on die 8, K=32, identical in every other respect:
#     chunked-prefill 8192 -> KDA asks 3.00 GiB with 3.03 GiB free -> OutOfMemoryError,
#                             SIGABRT in eager_runner._execute_extend. Twice, and once on
#                             a freshly started server, so it is not fragmentation.
#     chunked-prefill 6144 -> verify.sh ALL CHECKS PASSED on its 14379-token prompt.
# The hybrid path has no 6.75 GiB slot and runs at 0.85, so it keeps 8192.
# Raising it back is the biggest TTFT lever there is (TTFT is dominated by chunk COUNT),
# but it has to be paid for with headroom somewhere else first.
if [ "${GLM53_PREFILL_STREAM:-0}" = "1" ]; then
  export GLM53_MEM_FRACTION="${GLM53_MEM_FRACTION:-0.95}"
  export GLM53_MAX_TOTAL_TOKENS="${GLM53_MAX_TOTAL_TOKENS:-40960}"
  export GLM53_CHUNKED_PREFILL_SIZE="${GLM53_CHUNKED_PREFILL_SIZE:-6144}"
  export _GLM53_DRV_MEMFRAC=0.95 _GLM53_DRV_KV=40960 _GLM53_DRV_CHUNK=6144
else
  export GLM53_MEM_FRACTION="${GLM53_MEM_FRACTION:-0.85}"
  export GLM53_MAX_TOTAL_TOKENS="${GLM53_MAX_TOTAL_TOKENS:-}"
  export GLM53_CHUNKED_PREFILL_SIZE="${GLM53_CHUNKED_PREFILL_SIZE:-8192}"
  export _GLM53_DRV_MEMFRAC=0.85 _GLM53_DRV_KV= _GLM53_DRV_CHUNK=8192
fi
# GLM53_CHUNKED_PREFILL_SIZE is the one set per branch just above, and the constraint on
# it is not the multiple of 128 this comment used to claim: sglang asserts
# chunked_prefill_size % page_size == 0 (arg_groups/validation_hook.py) and the NPU kpool
# indexer wants chunked_prefill_size % index_kpool == 0, so at page-size 64 and
# index_kpool 4 it has to be a positive multiple of 64.  Both shipped values, 6144 and
# 8192, are multiples of 128 and clear that easily.
# Never -1: kt-kernel's LLAMAFILE MoE sizes its fp32 output buffer from the maximum chunk
# length and -1 collapses to 1, so any prefill longer than one token writes past the
# allocation and aborts inside glibc.
export GLM53_CONTEXT_LENGTH="${GLM53_CONTEXT_LENGTH:-32768}"
export GLM53_MAX_RUNNING_REQUESTS="${GLM53_MAX_RUNNING_REQUESTS:-1}"
# ---------------------------------------------------------------- toolchain
if [ -z "${GLM53_PYTHON}" ]; then
  for _p in "${GLM53_VENV:-${GLM53_ENV_ROOT}/.venv-glm53}/bin/python" \
            "$(command -v python3.12)" "$(command -v python3.11)" "$(command -v python3)"; do
    if [ -x "${_p}" ]; then GLM53_PYTHON="${_p}"; break; fi
  done
fi
export GLM53_PYTHON
export GLM53_JOBS="${GLM53_JOBS:-$(( $(nproc 2>/dev/null || echo 8) < 32 ? $(nproc 2>/dev/null || echo 8) : 32 ))}"

# npu-smi prints "Ascend910" for both A2 and A3, so it cannot tell them apart; the
# torch_npu device name can.  Getting this wrong picks the wrong sgl-kernel-npu build
# and fails later in ways that do not mention the SoC.
glm53_soc_from_chip_name() {
  case "$1" in
    *910_93*|*910C*|*910c*|*9362*) echo "ascend910_93" ;;
    *910B*|*910b*)                 echo "ascend910b" ;;
    *)                             echo "" ;;
  esac
}
glm53_detect_soc() {
  local name="" soc=""
  if [ -n "${GLM53_PYTHON}" ]; then
    name="$("${GLM53_PYTHON}" -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))' 2>/dev/null | tail -1)"
    soc="$(glm53_soc_from_chip_name "${name}")"
  fi
  if [ -z "${soc}" ] && command -v npu-smi >/dev/null 2>&1; then
    name="$(npu-smi info 2>/dev/null | awk '/^\| *[0-9]+ +[0-9A-Za-z_]+ +\|/ {print $3; exit}')"
    soc="$(glm53_soc_from_chip_name "${name}")"
  fi
  echo "${soc}"
}
export GLM53_SOC="${GLM53_SOC:-}"

# ---------------------------------------------------------------- CANN env
# Vendor set_env.sh files are not nounset-safe. Pass an empty positional argument
# so their option parsers do not consume the caller's arguments.
_glm53_source_if_readable() {
  [ -r "$1" ] || return 0
  local _f="$1" _u=""
  case "$-" in *u*) _u=1 ;; esac
  set +u
  . "$_f" ""
  [ -n "$_u" ] && set -u
  return 0
}
_glm53_source_if_readable "${ASCEND_INSTALL_ROOT}/ascend-toolkit/set_env.sh"
_glm53_source_if_readable "${ASCEND_INSTALL_ROOT}/nnal/atb/set_env.sh"
_glm53_source_if_readable "${CANN_ROOT}/share/info/ascendnpu-ir/bin/set_env.sh"

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# The custom-op vendors can live either where CANN installs them or in a project-local
# tree (this line has kept them under env/opp_custom since before the CANN packages
# were buildable here).  Search both; first hit wins.
export GLM53_OPP_CUSTOM_DIRS="${GLM53_OPP_CUSTOM_DIRS:-${GLM53_ENV_ROOT}/opp_custom/vendors:${CANN_ROOT}/opp/vendors}"

# ---------------------------------------------------------------- from-scratch build
# Where setup.sh's `cann-ops` step installs the custom operator vendor packages, and the
# two upstream repositories it builds them from.  Pinned to the same commits the
# DeepSeek-V4 recipe in tools/ascend_dsv4 uses: same SoC, same CANN, and GLM-5.3 needs the
# same operator set (compressor / quant_lightning_indexer / sparse_attn_sharedkv for DSA,
# plus the cann-recipes set for mHC and the quantised swiglu/routing kernels -- verified
# against the vendor packages this line has been running on).
export CANN_VENDORS_DIR="${CANN_VENDORS_DIR:-${GLM53_ENV_ROOT}/opp_custom/vendors}"
export CANN_RECIPES_REPO="${CANN_RECIPES_REPO:-${GLM53_WORKSPACE}/cann-recipes-infer}"
export OPS_TRANSFORMER_REPO="${OPS_TRANSFORMER_REPO:-${GLM53_WORKSPACE}/ops-transformer}"
export CANN_RECIPES_URL="${CANN_RECIPES_URL:-https://gitcode.com/cann/cann-recipes-infer.git}"
export OPS_TRANSFORMER_URL="${OPS_TRANSFORMER_URL:-https://gitcode.com/cann/ops-transformer.git}"
export CANN_RECIPES_COMMIT="${CANN_RECIPES_COMMIT:-1c8e6bcc2333d95b3db47d873210f921113d6d11}"
export OPS_TRANSFORMER_COMMIT="${OPS_TRANSFORMER_COMMIT:-8edcd591e83e536e9ee98a9ce0de3af02ea4f3ea}"
export SGL_KERNEL_NPU_REPO="${SGL_KERNEL_NPU_REPO:-${GLM53_WORKSPACE}/sgl-kernel-npu}"
export SGL_KERNEL_NPU_URL="${SGL_KERNEL_NPU_URL:-https://github.com/sgl-project/sgl-kernel-npu.git}"
# 20260826 (commit 146153e5), not 2026.6.2 (commit 9765e275, 2026-06-23): the sglang
# revision this recipe pins reaches sgl_kernel_npu.fla.kda_{chunk_delta_h,gate,prefill,
# target_verify} and mamba.causal_conv1d_verify from ascend_kda_backend.py, and none of
# those exist in the June tag -- the scheduler dies with ModuleNotFoundError after the
# weights are already resident. 20260826 carries all five.
export SGL_KERNEL_NPU_TAG="${SGL_KERNEL_NPU_TAG:-20260826}"

glm53_export_vendor_paths() {
  local _root _vendor _dir
  local _ifs="${IFS}"; IFS=:
  for _root in ${GLM53_OPP_CUSTOM_DIRS}; do
    IFS="${_ifs}"
    for _vendor in customize custom_transformer; do
      _dir="${_root}/${_vendor}"
      [ -d "${_dir}" ] || continue
      case ":${ASCEND_CUSTOM_OPP_PATH}:" in *":${_dir}:"*) continue ;; esac
      _glm53_source_if_readable "${_dir}/bin/set_env.bash"
      case ":${ASCEND_CUSTOM_OPP_PATH}:" in
        *":${_dir}:"*) ;;
        *) export ASCEND_CUSTOM_OPP_PATH="${_dir}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}" ;;
      esac
      case ":${LD_LIBRARY_PATH}:" in
        *":${_dir}/op_api/lib:"*) ;;
        *) export LD_LIBRARY_PATH="${_dir}/op_api/lib:${_dir}/op_proto/lib/linux/$(uname -m)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
    done
    IFS=:
  done
  IFS="${_ifs}"
}
glm53_export_vendor_paths

# ---------------------------------------------------------------- python env
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SGLANG_REPO}/python:${KTRANSFORMERS_REPO}/kt-kernel/python${PYTHONPATH:+:${PYTHONPATH}}"

# torch_npu's FunctionLoader prefers RTLD_DEFAULT and warns on any LD_PRELOAD; the
# system libgomp this image preloads also perturbs it.  Drop it.
unset LD_PRELOAD

# Keep download proxies available to setup.sh, but always bypass them for loopback and
# the configured inference endpoint. Clearing the proxy variables here made setup.sh's documented
# http_proxy / https_proxy support unreachable because that script sources this file
# before it reads them.
_glm53_add_no_proxy() {
  local _name="$1" _value _entry
  eval "_value=\${${_name}:-}"
  for _entry in 127.0.0.1 localhost ::1 "${GLM53_HOST#[}"; do
    _entry="${_entry%]}"
    [ -n "${_entry}" ] || continue
    case ",${_value}," in
      *",${_entry},"*) ;;
      *) _value="${_value:+${_value},}${_entry}" ;;
    esac
  done
  printf -v "${_name}" '%s' "${_value}"
  export "${_name}"
}
_glm53_add_no_proxy no_proxy
_glm53_add_no_proxy NO_PROXY
unset -f _glm53_add_no_proxy

glm53_show_env() {
  cat <<EOF
CANN
  ASCEND_INSTALL_ROOT       ${ASCEND_INSTALL_ROOT}
  CANN_ROOT                 ${CANN_ROOT}
  CANN_VERSION              ${CANN_VERSION} (package)   ${CANN_COMPILER_VERSION:-?} (compiler)
  ASCEND_CUSTOM_OPP_PATH    ${ASCEND_CUSTOM_OPP_PATH:-<none>}
Repositories
  KTRANSFORMERS_REPO        ${KTRANSFORMERS_REPO}
  SGLANG_REPO               ${SGLANG_REPO:-<NOT FOUND>}
  GLM53_LOG_DIR             ${GLM53_LOG_DIR}
Weights
  GLM53_MODEL_PATH          ${GLM53_MODEL_PATH}$([ -d "${GLM53_MODEL_PATH}" ] || echo '   *** MISSING ***')
  GLM53_MXFP4_CKPT          ${GLM53_MXFP4_CKPT}$([ -d "${GLM53_MXFP4_CKPT}" ] || echo '   *** MISSING ***')
  GLM53_GGUF_DIR            ${GLM53_GGUF_DIR}
  GLM53_GGUF_TEMPLATE       ${GLM53_GGUF_TEMPLATE}
  GGUF present              $(ls "${GLM53_GGUF_DIR}"/${GLM53_GGUF_NAME_PREFIX}*${GLM53_GGUF_NAME_SUFFIX}.gguf 2>/dev/null | wc -l) / $(( GLM53_MOE_LAYER_END - GLM53_MOE_LAYER_START + 1 )) layers (${GLM53_MOE_LAYER_START}..${GLM53_MOE_LAYER_END})
Host
  GLM53_SOC                 ${GLM53_SOC:-<unset - detected on demand: $(glm53_detect_soc || true)>}
  NUMA nodes / cores        ${GLM53_THREADPOOL_COUNT} / $(nproc 2>/dev/null || echo '?')
  RAM total                 $(awk '/MemTotal/{printf "%.0f GiB", $2/1048576}' /proc/meminfo 2>/dev/null)
  GLM53_THREADPOOL_COUNT    ${GLM53_THREADPOOL_COUNT}
  GLM53_CPUINFER            ${GLM53_CPUINFER}
Serving
  GLM53_NPU_DEVICE_ID       ${GLM53_NPU_DEVICE_ID}
  GLM53_PORT                ${GLM53_PORT}
  GLM53_NUM_GPU_EXPERTS     ${GLM53_NUM_GPU_EXPERTS}   (weights on die ~ $(awk "BEGIN{printf \"%.1f\", 15.60 + ($([ "${GLM53_PREFILL_STREAM:-0}" = "1" ] && echo 6.75 || echo 0)) + 0.9925*${GLM53_NUM_GPU_EXPERTS}}") GiB of 61.3 usable)
  GLM53_MEM_FRACTION        ${GLM53_MEM_FRACTION}
  GLM53_MAX_TOTAL_TOKENS    ${GLM53_MAX_TOTAL_TOKENS:-<unset, KV takes what mem-fraction allows>}
  GLM53_CONTEXT_LENGTH      ${GLM53_CONTEXT_LENGTH}
  GLM53_CHUNKED_PREFILL_SIZE ${GLM53_CHUNKED_PREFILL_SIZE}
  GLM53_MAX_RUNNING_REQUESTS ${GLM53_MAX_RUNNING_REQUESTS}
  GLM53_PYTHON              ${GLM53_PYTHON}
EOF
}

# ⛔ Only act on --show when this file is EXECUTED, never when it is sourced.
#
# A sourced script shares the caller's positional parameters, so `$1` here used to be
# serve.sh's `$1`. `./serve.sh --show` therefore ran glm53_show_env, which calls
# glm53_detect_soc, which calls torch.npu.get_device_name(0) -- opening die 0 on a box
# where die 0 belongs to somebody else. The A2 byte-identity gate is safe only because it
# passes no arguments, which is not a property anyone should have to know.
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then
  case "${1:-}" in
    --show|show) glm53_show_env ;;
    "") glm53_show_env ;;
    *) echo "usage: bash glm53_env.sh [--show]   (or: source glm53_env.sh)" >&2; exit 2 ;;
  esac
fi
