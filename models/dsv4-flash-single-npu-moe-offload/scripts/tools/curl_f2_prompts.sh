#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 整网冒烟：四个 prompt 连贯性验收（需服务已起，PORT 与 launch 一致）。
# 用法：HOST=127.0.0.1 PORT=8020 bash tools/curl_f2_prompts.sh

set -euo pipefail
# export 让内嵌 Python 经 os.environ 读到（默认端口与 launch 脚本对齐为 8020）。
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8020}"
PYBIN="${PYTHON_BIN:-${PYBIN:-/usr/local/python3.11.14/bin/python3.11}}"

"$PYBIN" - <<'PY'
import json
import urllib.request

host = __import__("os").environ.get("HOST", "127.0.0.1")
port = __import__("os").environ.get("PORT", "8020")
base = f"http://{host}:{port}/generate"

prompts = [
    (1, 64, "Below is a Python function to compute Fibonacci numbers:"),
    (
        2,
        128,
        "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n",
    ),
    (3, 80, "请用一句话解释什么是 transformer 模型："),
    (4, 128, "什么是 transformer 模型："),
]

for pid, max_tok, text in prompts:
    print(f"========== prompt {pid} (max_new_tokens={max_tok}) ==========")
    body = {"text": text, "sampling_params": {"max_new_tokens": max_tok, "temperature": 0}}
    req = urllib.request.Request(
        base,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        print(r.read()[:2500].decode(errors="replace"))
    print()

print("[smoke] done — 人工看 text：无 NaN/全感叹号/乱码即通过（base 模型不考核指令遵循）")
PY
