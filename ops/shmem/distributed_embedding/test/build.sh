#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${TEST_DIR}/env.sh"

cmake -S "${TEST_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DSHMEM_ROOT="${SHMEM_ROOT}" \
    -DSHMEM_LIBRARY_DIR="${SHMEM_LIBRARY_DIR}"
if [[ $# -eq 0 ]]; then
    set -- test_rank1 test_rank2
fi
cmake --build "${BUILD_DIR}" --target "$@" -j"${BUILD_JOBS}"
