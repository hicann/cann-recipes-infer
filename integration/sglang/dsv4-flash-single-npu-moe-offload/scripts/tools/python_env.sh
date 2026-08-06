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
# Resolve the target Python interpreter, shared by the launcher and the preflight check: both must
# resolve to the same one, or the preflight validates an environment the server does not run in.
# The built kt_kernel_ext.cpython-3XX-*.so is bound to the interpreter ABI and will not load under
# a different version.
#
# setup_dsv4_env_from_clean_cann.sh does not use this file: it runs on a clean image where the
# dependencies are not installed yet, so it can only select by version, not by importable modules.

# _python_has_modules <bin> [module ...]
# With no module arguments it only checks that the interpreter runs.
_python_has_modules() {
  local bin="$1"
  shift
  [[ $# -eq 0 ]] && return 0
  "${bin}" - "$@" <<'PY' >/dev/null 2>&1
import importlib
import sys

for name in sys.argv[1:]:
    importlib.import_module(name)
PY
}

# resolve_python_bin <tag> [module ...]
# Honours PYTHON_BIN when set, without falling back -- a silent fallback would swap in a different
# environment. Otherwise it takes the first candidate that can import every requested module.
# Prints the path on stdout on success; reports the failure and returns 1 otherwise.
resolve_python_bin() {
  local tag="${1:?tag required}"
  shift
  local cand
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  for cand in python3 python3.11 /usr/local/python3.11.14/bin/python3.11 \
              /usr/local/python3.11.14/bin/python3 /opt/conda/bin/python3; do
    command -v "${cand}" >/dev/null 2>&1 || continue
    if _python_has_modules "${cand}" "$@"; then
      command -v "${cand}"
      return 0
    fi
  done
  echo "[${tag}][ERROR] No python interpreter found that can import: ${*}" >&2
  echo "[${tag}][ERROR]   python3 on PATH = $(command -v python3 || echo none)" >&2
  echo "[${tag}][ERROR]   Set it explicitly, e.g. PYTHON_BIN=/usr/local/python3.11.14/bin/python3.11" >&2
  return 1
}
