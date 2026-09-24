#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${TEST_DIR}/env.sh"

cmake -S "${TEST_DIR}" -B "${BUILD_DIR}" \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DSHMEM_ROOT="${SHMEM_ROOT}" \
    -DSHMEM_LIBRARY_DIR="${SHMEM_LIBRARY_DIR}"
if [[ $# -eq 0 ]]; then
    set -- d2d_rank1 d2d_rank2 d2h_rank1 d2h_rank2
fi
cmake --build "${BUILD_DIR}" --target "$@" -j"${BUILD_JOBS}"
