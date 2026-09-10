#!/usr/bin/env bash
# Shared build/runtime environment for the standalone tests.
CANN_ROOT=${ASCEND_HOME_PATH:-${HOME}/ascend/cann}
if [[ ! -f "${CANN_ROOT}/bin/setenv.bash" ]]; then
    echo "CANN environment script not found: ${CANN_ROOT}/bin/setenv.bash" >&2
    exit 2
fi
# CANN's environment script reads optional, potentially unset variables.
set +u
source "${CANN_ROOT}/bin/setenv.bash"
set -u
export ASCEND_HOME_PATH="${CANN_ROOT}"
export SHMEM_ROOT=${SHMEM_ROOT:-${SHMEM_HOME_PATH:-${HOME}/ascend/shmem/latest}/shmem}
export SHMEM_LIBRARY_DIR=${SHMEM_LIBRARY_DIR:-${SHMEM_ROOT}/lib}
export BUILD_DIR=${BUILD_DIR:-${TEST_DIR}/build}
export BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
export LD_LIBRARY_PATH="${SHMEM_LIBRARY_DIR}:${SHMEM_ROOT}/lib:${CANN_ROOT}/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
