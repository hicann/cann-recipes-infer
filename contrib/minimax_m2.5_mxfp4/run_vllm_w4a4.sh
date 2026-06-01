#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/set_env.sh"
fi

MODEL_PATH="${MODEL_PATH:-/model/MiniMax-M2.5-MXFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-MiniMax-M2.5}"
TP_SIZE="${TP_SIZE:-16}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
VLLM_ASCEND_ENABLE_FLASHCOMM1="${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}"
VLLM_MXFP4_SKIP_ACT_QDQ="${VLLM_MXFP4_SKIP_ACT_QDQ:-0}"
ENABLE_TOOL_REASONING="${ENABLE_TOOL_REASONING:-1}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"
LOG_DIR="${LOG_DIR:-/data/logs}"
RANK="${RANK:-0}"

if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/minimax-m25-serve_${RANK}.log"

export VLLM_ASCEND_ENABLE_FLASHCOMM1
export VLLM_MXFP4_SKIP_ACT_QDQ

EXPERT_PARALLEL_ARG=()
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
  EXPERT_PARALLEL_ARG+=(--enable-expert-parallel)
fi

ENFORCE_EAGER_ARG=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ENFORCE_EAGER_ARG+=(--enforce-eager)
fi

TOOL_REASONING_ARGS=()
if [[ "${ENABLE_TOOL_REASONING}" == "1" ]]; then
  TOOL_REASONING_ARGS+=(
    --enable-auto-tool-choice
    --tool-call-parser minimax_m2
    --reasoning-parser minimax_m2_append_think
  )
fi

CMD=(
  vllm serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --dtype bfloat16
  --tensor-parallel-size "${TP_SIZE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --compilation-config "${COMPILATION_CONFIG}"
  --port "${PORT}"
)

CMD+=("${EXPERT_PARALLEL_ARG[@]}")
CMD+=("${ENFORCE_EAGER_ARG[@]}")
CMD+=("${TOOL_REASONING_ARGS[@]}")

echo "[INFO] Starting MiniMax-M2.5 MXFP4 W4A4 service"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Port: ${PORT}"

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
  nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo "[INFO] Service started in background, pid=$!"
  echo "[INFO] Use 'tail -f ${LOG_FILE}' to inspect logs"
else
  exec "${CMD[@]}"
fi
