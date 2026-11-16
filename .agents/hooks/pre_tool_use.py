#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Enforce model-agent edit boundaries."""

import fnmatch
import json
import logging
import os
import sys

from hook_paths import tool_file_paths

logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# agent_type -> file patterns that role must not modify
PROTECTED_PATTERNS = {
    "model-infer-analyzer": [
        "modeling_*.py",
        "runner_*.py",
        "*.yaml",
        "executor/*",
    ],
    "model-infer-reviewer": [
        "modeling_*.py",
        "runner_*.py",
    ],
}


def matches_any(basename, full_path, patterns):
    for pat in patterns:
        if fnmatch.fnmatch(basename, pat):
            return True
        if "/" in pat and fnmatch.fnmatch(full_path, f"*/{pat}"):
            return True
    return False


def check_role_protection(data):
    """Reject edits to files the acting role is not allowed to touch."""
    agent_type = data.get("agent_type", "")
    agent_id = data.get("agent_id", "")
    if not agent_id:
        return None

    patterns = PROTECTED_PATTERNS.get(agent_type)
    if not patterns:
        return None

    for file_path in tool_file_paths(data):
        basename = os.path.basename(file_path)
        if matches_any(basename, file_path, patterns):
            return f"禁止：{agent_type} 不允许修改 {basename}。模型代码修改请通过 implementer 执行。"

    return None


def main():
    data = json.load(sys.stdin)

    reason = check_role_protection(data)
    if reason:
        logger.error(reason)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
