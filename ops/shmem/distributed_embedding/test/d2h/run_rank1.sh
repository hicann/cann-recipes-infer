#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "${TEST_DIR}/../env.sh"
bash "${TEST_DIR}/../build.sh" d2h_rank1
EXE=${BUILD_DIR}/d2h_rank1

status=0
for bits in ${D2H_KEY_BITS:-32 64}; do
    if ! D2H_KEY_BITS="${bits}" "${EXE}"; then
        status=1
    fi
done
exit "${status}"
