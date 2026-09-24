#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "${TEST_DIR}/../env.sh"
bash "${TEST_DIR}/../build.sh" d2d_rank1
EXE=${BUILD_DIR}/d2d_rank1

exec "${EXE}"
