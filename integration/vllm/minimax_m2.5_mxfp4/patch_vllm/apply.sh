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

# Some images ship selected sources with CRLF line endings; normalize so
# LF-based patches apply cleanly. This is a working-tree-only rewrite.
normalize_crlf_files() {
  local repo_dir="$1"
  shift
  local rel_path
  for rel_path in "$@"; do
    local file_path="${repo_dir}/${rel_path}"
    if [[ -f "${file_path}" ]] && grep -q $'\r' "${file_path}"; then
      echo "[NORMALIZE] Strip CRLF: ${rel_path}"
      python3 - "${file_path}" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
path.write_bytes(data)
PY
    fi
  done
}

apply_patch_idempotent() {
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
  git -C "${repo_dir}" apply -p1 "${patch_file}"
}

normalize_crlf_files "${VLLM_ASCEND_DIR}" \
  "vllm_ascend/ops/fused_moe/moe_mlp.py"

apply_patch_idempotent "${VLLM_ASCEND_DIR}" "${VLLM_ASCEND_PATCH}" "vllm-ascend MXFP4 / KV-cache patch"
apply_patch_idempotent "${VLLM_DIR}" "${VLLM_PATCH}" "vllm MXFP4 patch"

if [[ "${INSTALL_AMD_QUARK}" == "1" ]]; then
  if python3 -c "import quark" >/dev/null 2>&1; then
    echo "[SKIP] amd-quark already installed"
  else
    echo "[INSTALL] python -m pip install amd-quark"
    python3 -m pip install amd-quark
  fi
else
  echo "[SKIP] INSTALL_AMD_QUARK=${INSTALL_AMD_QUARK}"
fi

echo "[OK] Patch apply completed successfully."
