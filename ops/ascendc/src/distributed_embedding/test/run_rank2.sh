#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source CANN setenv.bash first." >&2
    exit 2
fi
CANN_ROOT=${ASCEND_HOME_PATH}
if [[ -f "${CANN_ROOT}/bin/setenv.bash" ]]; then
    source "${CANN_ROOT}/bin/setenv.bash"
fi

SHMEM_ROOT=${SHMEM_ROOT:-/data/l00975381/turing/ascend/shmem/latest/shmem}
SHMEM_BACKEND=${SHMEM_BACKEND:-950}
if [[ -d "${SHMEM_ROOT}/backends/${SHMEM_BACKEND}" ]]; then
    SHMEM_LIBRARY_DIR=${SHMEM_LIBRARY_DIR:-${SHMEM_ROOT}/backends/${SHMEM_BACKEND}}
else
    SHMEM_LIBRARY_DIR=${SHMEM_LIBRARY_DIR:-${SHMEM_ROOT}/lib}
fi
BUILD_DIR=${BUILD_DIR:-${TEST_DIR}/build}
EXE=${BUILD_DIR}/test_rank2
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
FIRST_DEVICE=${FIRST_DEVICE:-5}
RANK0_DEVICE=${RANK0_DEVICE:-${FIRST_DEVICE}}
RANK1_DEVICE=${RANK1_DEVICE:-$((FIRST_DEVICE + 1))}
SHMEM_IP_PORT=${SHMEM_IP_PORT:-tcp://127.0.0.1:8999}
TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS:-180}
LOG_DIR=${LOG_DIR:-${BUILD_DIR}/rank2_logs}

export LD_LIBRARY_PATH="${SHMEM_LIBRARY_DIR}:${SHMEM_ROOT}/lib:${CANN_ROOT}/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

for required in \
    "${CANN_ROOT}/include/acl/acl.h" \
    "${CANN_ROOT}/x86_64-linux/asc/include/tiling/platform/platform_ascendc.h" \
    "${SHMEM_ROOT}/include/shmem.h" \
    "${SHMEM_LIBRARY_DIR}/libshmem.so"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required dependency: ${required}" >&2
        exit 2
    fi
done

cmake -S "${TEST_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_HOME_PATH="${CANN_ROOT}" \
    -DSHMEM_ROOT="${SHMEM_ROOT}" \
    -DSHMEM_BACKEND="${SHMEM_BACKEND}" \
    -DSHMEM_LIBRARY_DIR="${SHMEM_LIBRARY_DIR}"
cmake --build "${BUILD_DIR}" --target test_rank2 -j"${BUILD_JOBS}"

mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}/rank0.log" "${LOG_DIR}/rank1.log"

launch_rank() {
    local rank=$1
    local device_id=$2
    local log_file=$3
    timeout --foreground "${TEST_TIMEOUT_SECONDS}s" \
        env RANK_ID="${rank}" DEVICE_ID="${device_id}" SHMEM_IP_PORT="${SHMEM_IP_PORT}" \
        "${EXE}" >"${log_file}" 2>&1 &
    RANK_PID=$!
}

launch_rank 0 "${RANK0_DEVICE}" "${LOG_DIR}/rank0.log"
rank0_pid=${RANK_PID}
launch_rank 1 "${RANK1_DEVICE}" "${LOG_DIR}/rank1.log"
rank1_pid=${RANK_PID}

status=0
for pid in "${rank0_pid}" "${rank1_pid}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

for rank in 0 1; do
    echo "===== rank ${rank} log ====="
    cat "${LOG_DIR}/rank${rank}.log"
done

exit "${status}"
