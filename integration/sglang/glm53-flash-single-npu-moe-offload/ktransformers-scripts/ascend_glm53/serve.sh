#!/usr/bin/env bash
# Launch GLM-5.3-Flash on one Ascend die with the routed experts on the host.
#   ./serve.sh              background, log to $GLM53_LOG_DIR/serve.log
#   ./serve.sh --foreground exec in place
#   ./serve.sh --dry-run    print the launch command without starting the server
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

FOREGROUND=0
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --foreground) FOREGROUND=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
    cat <<'USAGE'
usage: serve.sh [--foreground] [--dry-run]

Launch the GLM-5.3-Flash single-die server. `bash glm53_env.sh --show` prints the
resolved configuration. Supported runtime controls are documented in README.md.

  --foreground   run in this terminal instead of detaching
  --dry-run      print the launch command and exit without touching a die
USAGE
      exit 0 ;;
    *) echo "[serve] unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "${GLM53_LOG_DIR}"
LOG="${GLM53_LOG_DIR}/serve.log"

# SGLang redirects Triton and TorchInductor caches below this framework-owned
# directory. Keep it with the recipe artifacts: a cache compiled by another
# torch ABI under ~/.cache/sglang can otherwise be loaded and fail at startup.
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${GLM53_ARTIFACT_ROOT}/cache/sglang}"
mkdir -p "${SGLANG_CACHE_DIR}"
echo "[serve] cache  = ${SGLANG_CACHE_DIR}"

# An sglang installed in site-packages shadows the repo silently and you then debug a
# tree you are not editing.
_sglang_file="$("${GLM53_PYTHON}" -c 'import sglang; print(sglang.__file__)' 2>/dev/null | tail -1)"
echo "[serve] sglang = ${_sglang_file}"
case "${_sglang_file}" in
  "${SGLANG_REPO}"/python/sglang/__init__.py) ;;
  *) echo "[serve] FATAL: sglang resolves outside ${SGLANG_REPO}. PYTHONPATH was shadowed." >&2
     exit 1 ;;
esac

_ggufs="$(ls "${GLM53_GGUF_DIR}"/${GLM53_GGUF_NAME_PREFIX}*${GLM53_GGUF_NAME_SUFFIX}.gguf 2>/dev/null | wc -l)"
_want=$(( GLM53_MOE_LAYER_END - GLM53_MOE_LAYER_START + 1 ))
if [ "${_ggufs}" -ne "${_want}" ]; then
  echo "[serve] FATAL: ${_ggufs} GGUF files in ${GLM53_GGUF_DIR}, expected ${_want}." >&2
  echo "[serve]        A missing layer does not fail loudly -- kt-kernel loads no experts" >&2
  echo "[serve]        for it and the model answers with garbage. Run: setup.sh gguf" >&2
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${GLM53_NPU_DEVICE_ID}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export INF_NAN_MODE_FORCE_DISABLE="${INF_NAN_MODE_FORCE_DISABLE:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"

# Keep the KDA conv state in the dtype the conv *weights* use.  When the two differ,
# _causal_conv1d_decode loses its fast path and gathers/converts/scatters instead --
# 7 extra kernels per KDA layer per decode step, and GLM has 34 of them.  Accuracy is
# unchanged either way (the conv window holds a copy of a bf16 projection output, so
# the bf16 round trip is exact).
# This is part of the validated recipe rather than a user-facing switch. Letting an
# inherited shell value change it would silently change the decode kernel path.
export SGLANG_MAMBA_CONV_DTYPE=bfloat16

# Carried verbatim from the validated GLM-5.3 single-die INT8 recipe.  These are not
# defaults; each was chosen against a measurement, and changing one silently changes
# what the accuracy baseline means.
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False

# Which NUMA nodes the CPU-MoE subpools bind to. Unset means kt-kernel's default,
# subpool i -> node i, which is right for the single-node deployment target but puts
# every server on a development box onto node 0.
#
# ⚠ This host pairs its nodes (0,1) (2,3) (4,5) (6,7): within a pair a read runs at
# ~150 GB/s, across pairs at ~20 GB/s, and /sys/.../node*/distance reports a uniform
# 20 for every remote and will not warn you. Keep one server's subpools inside one
# pair, and give concurrent servers different pairs.
# ⚠ kt-kernel finds its cores through hwloc, whose view of this box is not the kernel's.
# Asking for a node hwloc cannot enumerate does not fail cleanly: it prints "Core N inside
# NUMA node X not found" and then corrupts the heap -- "free(): invalid next size (fast)",
# SIGABRT during model init, with the real cause buried under a RecursionError in
# torch_npu's excepthook. Measured: nodes 6,7 do this even though /sys shows them with 40
# CPUs each and all 320 CPUs are in our cpuset. Nodes 0,1 work.
#
# So check before launching rather than debugging an abort afterwards.
if [ -n "${GLM53_KT_NUMA_NODES:-}" ]; then
  export KT_NUMA_NODES="${GLM53_KT_NUMA_NODES}"
  for _n in ${KT_NUMA_NODES//,/ }; do
    if [ ! -d "/sys/devices/system/node/node${_n}" ]; then
      echo "[serve] FATAL: NUMA node ${_n} does not exist." >&2; exit 1
    fi
  done
  echo "[serve] CPU-MoE subpools pinned to NUMA nodes ${KT_NUMA_NODES}"
  echo "[serve]   (if startup aborts with 'Core N inside NUMA node ${KT_NUMA_NODES%%,*} not found'"
  echo "[serve]    followed by a heap error, hwloc cannot see that node -- try 0,1)"
fi

# ⚠ One subpool on a multi-node host is a measurement trap, not a bug.
#
# kt-kernel takes its "one subpool owns the whole tensor, so alias the GGUF mmap" path
# at threadpool_count=1 and skips the node-local copy. That is correct and saves 154 GiB
# on the deployment target, a single-NUMA container, where the pages are local anyway.
# Here it leaves the weights wherever the page cache first faulted them -- spread over
# every node -- while the reading threads sit on one. Measured cost on this box: 52 ms
# per token becomes 171. It does not fail, it is just four times slower, and a thread
# sweep across it draws a convincing saturation curve that means nothing.
#
# The warning lives here rather than in the documentation because this failure is silent
# and looks like a hardware limit, so the moment to say it is the moment it is being
# configured.
_glm53_numa_count="$(find /sys/devices/system/node -maxdepth 1 -type d -name 'node[0-9]*' 2>/dev/null | wc -l)"
if [ "${GLM53_THREADPOOL_COUNT}" = "1" ] && [ "${_glm53_numa_count}" -gt 1 ]; then
  echo "[serve] ⚠ --kt-threadpool-count 1 on a ${_glm53_numa_count}-node host: kt-kernel will"
  echo "[serve]   alias the GGUF mmap instead of copying, so the CPU MoE reads pages that"
  echo "[serve]   live on every node. On this box that is ~3x slower and looks like a"
  echo "[serve]   bandwidth ceiling. Fine for correctness and for the single-node target;"
  echo "[serve]   NOT a proxy for the target's performance."
  echo "[serve]   For a representative number: GLM53_THREADPOOL_COUNT=2 GLM53_KT_NUMA_NODES=<a,b>"
  echo "[serve]   with a,b a fast pair -- this host pairs (0,1) (2,3) (4,5) (6,7)."
fi

# ⚠ The KV cap and the concurrency have to be sized together, and getting it wrong does
# not fail cleanly. GLM53_MAX_TOTAL_TOKENS defaults to 40960 on the streaming path
# because the 6.75 GiB convert slot is reserved before KV sizing and the pool would
# otherwise eat the headroom the per-layer convert needs. That figure was chosen for
# --max-running-requests 1. At width 8 with long generations the pool fills, and the
# failure observed is NOT "out of KV": it is
#     full token usage: 1.00
#     error 507035: The vector core execution is abnormal
#     rtDeviceSynchronizeWithTimeout ... reason=vector core exception
# and the scheduler dies. So check the budget here, where it can still be explained.
if [ -n "${GLM53_MAX_TOTAL_TOKENS}" ] && [ "${GLM53_MAX_RUNNING_REQUESTS}" -gt 1 ]; then
  _per_req=$(( GLM53_MAX_TOTAL_TOKENS / GLM53_MAX_RUNNING_REQUESTS ))
  echo "[serve] KV budget: ${GLM53_MAX_TOTAL_TOKENS} tokens / ${GLM53_MAX_RUNNING_REQUESTS} concurrent = ${_per_req} per request"
  if [ "${_per_req}" -lt 4096 ]; then
    echo "[serve] ⚠ only ${_per_req} KV tokens per concurrent request. A generation longer" >&2
    echo "[serve]   than that fills the pool and the scheduler dies with a vector-core" >&2
    echo "[serve]   exception, not a clean out-of-memory. Either lower" >&2
    echo "[serve]   GLM53_MAX_RUNNING_REQUESTS, raise GLM53_MAX_TOTAL_TOKENS, or bound the" >&2
    echo "[serve]   client's max_tokens below ${_per_req}." >&2
  fi
fi

ulimit -n 65536 2>/dev/null || true

# The side stream is INDEPENDENT of streaming prefill, despite both being KT_* knobs:
# kt_ep_wrapper reads KT_SIDE_STREAM at import and uses it in the captured-graph submit
# path, which runs on every decode step whether or not any prefill was streamed. It
# forks the CPU-MoE host callback onto a second stream so the resident-expert
# GroupedMatmul does not sit behind the host round trip. Default it ON: with prefix
# placement only ~32/288 of routed experts are resident, so nearly every token pays a
# CPU round trip and serialising it is the worst case.
export KT_SIDE_STREAM="${KT_SIDE_STREAM:-1}"
[ "${KT_SIDE_STREAM}" = "1" ] && echo "[serve] CPU-MoE side stream ENABLED"

# Streaming prefill is off by default, as it is for DeepSeek-V4: it makes long prompts
# much faster and short ones slower.  It now works on GLM -- the readers probe the
# checkpoint's own index for the tensor spelling instead of assuming DeepSeek's -- but it
# costs HBM, see GLM53_MEM_FRACTION / GLM53_MAX_TOTAL_TOKENS in glm53_env.sh.
#
# ⚠ Every failure inside kt_stream_prefill is caught and falls back to the hybrid path, so
# a broken streaming build looks exactly like a working one from outside.  The only
# evidence is the log: `grep -c 'inline resident'` > 0 and
# `grep -cE 'streaming failed|hybrid fallback'` == 0, on a prompt above the threshold.
if [ "${GLM53_PREFILL_STREAM:-0}" = "1" ]; then
  export KT_PREFILL_STREAM=1
  export KT_PREFILL_STREAM_THRESHOLD="${KT_PREFILL_STREAM_THRESHOLD:-512}"
  export KT_PREFILL_STREAM_CKPT="${KT_PREFILL_STREAM_CKPT:-${GLM53_MODEL_PATH}}"
  export KT_MXFP4_CKPT="${KT_MXFP4_CKPT:-${GLM53_MXFP4_CKPT}}"
  export KT_MXFP4_OP_DIR="${KT_MXFP4_OP_DIR:-${KTRANSFORMERS_REPO}/kt-kernel/tools/ascendc_mxfp4}"
  # KT_MXFP4_DEPOOL is retired: the non-depool W8A8 safetensors reader it selected has
  # been deleted, so the on-device MXFP4 convert is the only path and needs no switch.
  # Experts per chunk in the MXFP4->W8A8-NZ convert.  The wrapper defaults to 32; each
  # chunk costs roughly 4 x chunk x 2I x H bytes of transient HBM (fp16 round trip), which
  # at 32 is ~3 GB and competes with the KDA layers' Triton workspace on the same prefill.
  # 16 halves it and was not measurably slower.
  export KT_MXFP4_NZ_CHUNK="${KT_MXFP4_NZ_CHUNK:-16}"
  export KT_MXFP4_GGUF_DEDUP="${KT_MXFP4_GGUF_DEDUP:-1}"
  export KT_GGUF_TEMPLATE="${KT_GGUF_TEMPLATE:-${GLM53_GGUF_TEMPLATE}}"
  export KT_DYNAMIC_RESIDENT="${KT_DYNAMIC_RESIDENT:-1}"
  # Fit the decode hot set from the last 512 prompt tokens rather than the whole prefill.
  # Measured over 6 arms on a contended box: the 512 arm was never worse than the adjacent
  # whole-prompt arm (5/5 adjacent pairs) but the magnitude is unresolved (2-12%) -- two
  # identical 512 arms differed by 16%. 2048 showed no gain on a 3250-token prompt.
  # Costs nothing (no HBM, one env var), so take the free side of a direction-only result.
  export KT_HOT_TAIL_TOKENS="${KT_HOT_TAIL_TOKENS:-512}"
  export KT_STREAM_WARMUP="${KT_STREAM_WARMUP:-1}"
  echo "[serve] streaming prefill ENABLED (threshold ${KT_PREFILL_STREAM_THRESHOLD} tokens)"
fi

GRAPH_ARGS=( --cuda-graph-max-bs-decode "${GLM53_MAX_RUNNING_REQUESTS}" --disable-prefill-cuda-graph )
# ⚠ --disable-cuda-graph is no_cli=True on this baseline and only reaches argparse through
# a deprecated alias, so it works but prints a deprecation warning.  If the alias is
# dropped upstream, GLM53_EAGER=1 turns into "unrecognized arguments" and the server does
# not start; switch to --cuda-graph-max-bs-decode 0 then.
[ "${GLM53_EAGER:-0}" = "1" ] && GRAPH_ARGS=( --disable-cuda-graph )

CMD=( "${GLM53_PYTHON}" -m sglang.launch_server
  --model-path            "${GLM53_MODEL_PATH}"
  --trust-remote-code
  --device                npu
  # kt_ep_wrapper.create_kt_config_from_server_args refuses to start on NPU unless
  # tp_size == 1, ep_size == 1 and moe_a2a_backend is none (plus kt_num_gpu_experts >= 1).
  # dtype and kv-cache-dtype are NOT checked there; they are our choice, not a gate.
  --tensor-parallel-size  1
  --expert-parallel-size  1
  --moe-a2a-backend       none
  --dtype                 bfloat16
  --kv-cache-dtype        auto
  # 64, not DeepSeek-V4's 128: GLM's DSA pool asserts page_size == 64.
  --page-size             64
  # The shared expert is fused into the routed GroupedMatmul as slot 288 on this
  # branch.  That widens the FusedMoE to 289 while kt_expert_masks sizes its placement
  # tables from n_routed_experts=288; the mismatch makes KTEPWrapperMethod silently
  # discard the placement and pin the always-active shared slot to the CPU, on every
  # token.  Keep the fusion off until that is fixed and A/B'd.
  --disable-shared-experts-fusion
  # causal_conv1d_fn_npu corrupts conv state when a batch mixes has_initial_state,
  # so prefix reuse would poison accuracy runs.
  --disable-radix-cache
  --mem-fraction-static   "${GLM53_MEM_FRACTION}"
  --context-length        "${GLM53_CONTEXT_LENGTH}"
  --chunked-prefill-size  "${GLM53_CHUNKED_PREFILL_SIZE}"
  --max-running-requests  "${GLM53_MAX_RUNNING_REQUESTS}"
  "${GRAPH_ARGS[@]}"
  --watchdog-timeout      18000
  --kt-method             LLAMAFILE
  --kt-num-gpu-experts    "${GLM53_NUM_GPU_EXPERTS}"
  --kt-weight-path        "${GLM53_GGUF_TEMPLATE}"
  --kt-threadpool-count   "${GLM53_THREADPOOL_COUNT}"
  --kt-cpuinfer           "${GLM53_CPUINFER}"
  --host                  "${GLM53_HOST}"
  --port                  "${GLM53_PORT}"
)
# Cap the KV pool rather than letting it absorb every byte --mem-fraction-static allows.
# With one running request and a 32768 context the extra tokens buy nothing, while the
# slack they consume is what the dynamic allocations (Triton workspaces, the streaming
# convert) run out of.  See glm53_env.sh.
[ -n "${GLM53_MAX_TOTAL_TOKENS}" ] && CMD+=( --max-total-tokens "${GLM53_MAX_TOTAL_TOKENS}" )
# ⚠ Pin BEFORE the process starts, never after it is up. Page placement is decided by
# first touch during weight load, so a taskset applied once the server is healthy binds
# the threads and leaves the pages where they already landed. Observed: an instance
# pinned post-load to cores 80-159 (nodes 2,3) had its anon memory on node 1 -- the
# other pair -- and read it across the ~20 GB/s cross-pair cliff.
# sglang and the kt threadpool bind nothing on their own: every thread starts with an
# affinity mask of 0-319, and KT_NUMA_NODES only steers the kt threadpool's own buffers.
# Set GLM53_PIN_CORES to a taskset list (e.g. 0-79) matching GLM53_KT_NUMA_NODES.
if [ -n "${GLM53_PIN_CORES:-}" ]; then
  if ! command -v taskset >/dev/null 2>&1; then
    echo "[serve] FATAL: GLM53_PIN_CORES set but taskset is not installed" >&2; exit 1
  fi
  CMD=( taskset -c "${GLM53_PIN_CORES}" "${CMD[@]}" )
  echo "[serve] pinned to cores ${GLM53_PIN_CORES} from launch (first-touch placement)"
fi

printf '[serve] %s\n' "${CMD[*]}"

# Inspect the resolved command without taking a die.
if [ "${DRY_RUN}" = "1" ]; then
  echo "[serve] --dry-run, not launching"
  exit 0
fi

if [ "${FOREGROUND}" -eq 1 ]; then
  exec "${CMD[@]}"
fi

nohup "${CMD[@]}" > "${LOG}" 2>&1 &
echo $! > "${LOG}.pid"
echo "[serve] pid=$(cat "${LOG}.pid")  die=${GLM53_NPU_DEVICE_ID}  port=${GLM53_PORT}"
echo "[serve] log    : tail -f ${LOG}"
echo "[serve] accept : bash ${_here}/verify.sh"
# Only ever kill by port: this host is shared under one OS account and a global
# pkill on 'sglang.launch_server' takes out other people's runs.
echo "[serve] stop   : pkill -f -- \"[-]-port ${GLM53_PORT}\""
