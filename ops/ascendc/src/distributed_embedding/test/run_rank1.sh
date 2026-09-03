#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source CANN setenv.bash first." >&2
    exit 2
fi
CANN_ROOT=${ASCEND_HOME_PATH}
if [[ -f "${CANN_ROOT}/bin/setenv.bash" ]]; then
    # Keep the caller's selected CANN installation as the source of runtime paths.
    source "${CANN_ROOT}/bin/setenv.bash"
fi

# Both SHMEM 1.6.0 and newer packages expose the Ascend950 library at this
# compatibility path. SHMEM_ROOT can still be overridden for isolated tests.
SHMEM_ROOT=${SHMEM_ROOT:-/data/l00975381/turing/ascend/shmem/latest/shmem}
SHMEM_BACKEND=${SHMEM_BACKEND:-950}
SHMEM_BACKEND_DIR=${SHMEM_ROOT}/lib
BUILD_DIR=${BUILD_DIR:-${TEST_DIR}/build}
EXE=${BUILD_DIR}/test_rank1
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

export LD_LIBRARY_PATH="${SHMEM_BACKEND_DIR}:${CANN_ROOT}/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

for required in \
    "${CANN_ROOT}/include/acl/acl.h" \
    "${CANN_ROOT}/x86_64-linux/asc/include/tiling/platform/platform_ascendc.h" \
    "${SHMEM_ROOT}/include/shmem.h" \
    "${SHMEM_BACKEND_DIR}/libshmem.so"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required dependency: ${required}" >&2
        exit 2
    fi
done

cmake -S "${TEST_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_HOME_PATH="${CANN_ROOT}" \
    -DSHMEM_ROOT="${SHMEM_ROOT}" \
    -DSHMEM_BACKEND="${SHMEM_BACKEND}" \
    -DSHMEM_LIBRARY_DIR="${SHMEM_BACKEND_DIR}"
make -C "${BUILD_DIR}" -j"${BUILD_JOBS}" test_rank1

exec "${EXE}"
