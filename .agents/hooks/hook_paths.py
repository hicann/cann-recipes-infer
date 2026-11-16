# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Resolve file paths from Claude Code and Codex tool events."""

import os


def tool_file_paths(data):
    """Return all explicit file paths, including both sides of patch moves."""
    tool_input = data.get("tool_input", {})
    paths = []
    if tool_input.get("file_path"):
        paths.append(tool_input["file_path"])
    if data.get("tool_name") == "apply_patch":
        prefixes = ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: ")
        for line in tool_input.get("command", "").splitlines():
            for prefix in prefixes:
                if line.startswith(prefix):
                    paths.append(line[len(prefix):])
                    break
    cwd = data.get("cwd") or os.getcwd()
    return [os.path.abspath(os.path.join(cwd, path)) for path in paths]
