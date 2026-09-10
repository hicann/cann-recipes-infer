#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "${TEST_DIR}/env.sh"
bash "${TEST_DIR}/build.sh" test_rank2
EXE=${BUILD_DIR}/test_rank2

FIRST_DEVICE=${FIRST_DEVICE:-0}
RANK0_DEVICE=${RANK0_DEVICE:-${FIRST_DEVICE}}
RANK1_DEVICE=${RANK1_DEVICE:-$((FIRST_DEVICE + 1))}
SHMEM_IP_PORT=${SHMEM_IP_PORT:-tcp://127.0.0.1:8999}
TEST_TIMEOUT_SECONDS=${TEST_TIMEOUT_SECONDS:-180}
LOG_DIR=${LOG_DIR:-${BUILD_DIR}/rank2_logs}

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
