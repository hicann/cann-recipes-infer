#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
DRIVER_LIB_PATH="${DRIVER_LIB_PATH:-/usr/local/Ascend/driver/lib64}"

export ASCEND_TOOLKIT_HOME
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-$ASCEND_TOOLKIT_HOME}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_TOOLKIT_HOME/opp}"

if [[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  set +eu
  source "/usr/local/Ascend/nnal/atb/set_env.sh" || true
  set -eu
fi

BASE_LIB_PATH="${ASCEND_TOOLKIT_HOME}/${ARCH}-linux/lib64"
export LD_LIBRARY_PATH="${DRIVER_LIB_PATH}/driver:${DRIVER_LIB_PATH}/common:${BASE_LIB_PATH}:${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ASCEND_TOOLKIT_HOME}/python/site-packages:${PYTHONPATH:-}"
export PATH="${ASCEND_TOOLKIT_HOME}/bin:${PATH}"

ATB_LIB_DIR="/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib"
if [[ -d "${ATB_LIB_DIR}" ]]; then
  export ATB_HOME_PATH="${ATB_HOME_PATH:-/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1}"
  export LD_LIBRARY_PATH="${ATB_HOME_PATH}/lib:${LD_LIBRARY_PATH}"
fi

export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE="${ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE:-0}"
export ATB_OPSRUNNER_SETUP_CACHE_ENABLE="${ATB_OPSRUNNER_SETUP_CACHE_ENABLE:-1}"
export HCCL_WHITELIST_DISABLE="${HCCL_WHITELIST_DISABLE:-1}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"

echo "[INFO] Ascend vLLM MiniMax environment initialized"
