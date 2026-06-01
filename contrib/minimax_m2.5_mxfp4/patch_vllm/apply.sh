#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_PATCH="${SCRIPT_DIR}/0002-vllm-patch-for-mxfp4.patch"
VLLM_ASCEND_PATCH="${SCRIPT_DIR}/0001-vllm-ascend-patch-for-mxfp4.patch"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
INSTALL_AMD_QUARK="${INSTALL_AMD_QUARK:-1}"

for f in "${VLLM_PATCH}" "${VLLM_ASCEND_PATCH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing patch: ${f}" >&2
    exit 1
  fi
done

if [[ ! -d "${VLLM_DIR}" ]]; then
  echo "[ERROR] Missing directory: ${VLLM_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VLLM_ASCEND_DIR}" ]]; then
  echo "[ERROR] Missing directory: ${VLLM_ASCEND_DIR}" >&2
  exit 1
fi

if ! git -C "${VLLM_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[ERROR] ${VLLM_DIR} is not a git repository" >&2
  exit 1
fi

if ! git -C "${VLLM_ASCEND_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[ERROR] ${VLLM_ASCEND_DIR} is not a git repository" >&2
  exit 1
fi

already_applied() {
  local repo_dir="$1"
  local patch_file="$2"
  git -C "${repo_dir}" apply -p1 --reverse --check "${patch_file}" >/dev/null 2>&1
}

apply_patch_with_git_am() {
  local repo_dir="$1"
  local patch_file="$2"
  local label="$3"

  if already_applied "${repo_dir}" "${patch_file}"; then
    echo "[SKIP] ${label} already applied"
    return 0
  fi

  echo "[CHECK] Dry-run ${label}"
  git -C "${repo_dir}" apply -p1 --check "${patch_file}"

  echo "[APPLY] ${label}"
  git -C "${repo_dir}" am -C1 --ignore-whitespace --3way "${patch_file}"
}

apply_patch_with_git_am "${VLLM_DIR}" "${VLLM_PATCH}" "vllm MXFP4 patch"
apply_patch_with_git_am "${VLLM_ASCEND_DIR}" "${VLLM_ASCEND_PATCH}" "vllm-ascend FP8/MXFP4 patch"

echo "[PATCH] Applying shm memory ordering fix"
SHM_FILE="${VLLM_DIR}/vllm/distributed/device_communicators/shm_broadcast.py"
export SHM_FILE
python3 - <<'PY'
import os
import sys

path = os.environ["SHM_FILE"]
if not os.path.exists(path):
    print(f"[WARN] shm file not found: {path}", file=sys.stderr)
    sys.exit(0)

with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = """                    metadata_buffer[i] = 0
                # mark the block as written
                metadata_buffer[0] = 1"""

new = """                    metadata_buffer[i] = 0
                # Ensure buffer data and flag resets are visible before
                # the written flag (ARM weak memory ordering, vllm PR #32022).
                memory_fence()
                # mark the block as written
                metadata_buffer[0] = 1"""

if new in src:
    print("[SKIP] shm_broadcast.py already contains memory_fence()")
    sys.exit(0)

if old not in src:
    print("[WARN] shm_broadcast.py pattern not found; please check file manually", file=sys.stderr)
    sys.exit(0)

with open(path, "w", encoding="utf-8") as f:
    f.write(src.replace(old, new, 1))

print("[OK] Added memory_fence() before metadata_buffer[0] write")
PY

if [[ "${INSTALL_AMD_QUARK}" == "1" ]]; then
  echo "[INSTALL] python -m pip install amd-quark"
  python -m pip install amd-quark
else
  echo "[SKIP] INSTALL_AMD_QUARK=${INSTALL_AMD_QUARK}"
fi

echo "[OK] Patch apply completed successfully."
