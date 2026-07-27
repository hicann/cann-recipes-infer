#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

KIMI_K3_TMP_CANN_ENV="/usr/local/Ascend/cann/set_env.sh"
if [ ! -f "${KIMI_K3_TMP_CANN_ENV}" ]; then
    echo "Kimi K3 A5 CANN environment not found: ${KIMI_K3_TMP_CANN_ENV}" >&2
    return 1
fi

source "${KIMI_K3_TMP_CANN_ENV}"
