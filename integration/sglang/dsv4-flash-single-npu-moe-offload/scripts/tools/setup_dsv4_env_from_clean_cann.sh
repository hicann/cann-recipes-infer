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
# =============================================================================
# DeepSeek-V4-Flash on a single A3 card: bring-up from a clean CANN 9.0.0 image.
# Installs the three custom-operator pieces (custom_ops / customize / custom_transformer),
# the sglang dependencies, sgl_kernel_npu and kt-kernel.
# =============================================================================
# This is an environment/dependency bring-up script, not a model-code patch. It captures the exact
# steps that bring a bare A3 host up from nothing, with the workaround for each known
# pitfall already applied.
#
# Usage:
#   1) Adjust the configurable variables below (paths, Python, CANN version) as needed.
#   2) Run a phase at a time:  bash setup_dsv4_env_from_clean_cann.sh <phase>
#      phase ∈ {all, prereq, torch, triton, sglang_deps,
#               vendor_customize, custom_ops, vendor_transformer,
#               sgl_kernel_npu, kt_kernel, verify}
#      No argument runs `all`, i.e. every phase in order.
#
# The companion document is
#   docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md (the A3 section)
# =============================================================================
set -euo pipefail

# ----------------------------- configurable variables ------------------------
# No machine-specific path is hardcoded: the interpreter is detected on PATH and the workspace root
# defaults to the parent of this repo. All three can be overridden through the environment:
# PYTHON_BIN / CANN_HOME / GITCODE.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python3.11 || command -v python3 || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[setup][ERROR] No python3.11/python3 found. Set it explicitly: PYTHON_BIN=/path/to/python3.11 bash $0" >&2
    exit 2
  fi
fi
: "${CANN_HOME:=$HOME/Ascend/cann-9.0.0}"                         # CANN 9.0.0 as shipped in the clean image
# Clone root for the third-party repositories (ops-transformer / cann-recipes-infer /
# sgl-kernel-npu). Defaults to the parent of the repo holding this script; override with
# GITCODE=/your/workspace elsewhere.
_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GITCODE:=$(cd "$_SETUP_DIR/../.." && pwd)/..}"               # repository root
GITCODE="$(cd "$GITCODE" 2>/dev/null && pwd || echo "$GITCODE")"
: "${REPO:=$GITCODE/ktransformers-AK}"                            # this repo (holds kt-kernel and the sglang submodule)
: "${CC_BIN:=/usr/bin/gcc-13}"                                    # ARM bf16/i8mm + gnu++20 need gcc >= 13
: "${CXX_BIN:=/usr/bin/g++-13}"
: "${SOC:=ascend910_93}"                                          # A3 = ascend910_93
: "${JOBS:=16}"

# Third-party repositories (cloned when missing)
: "${OPS_TF_REPO:=$GITCODE/cann/ops-transformer}"                 # NSA operators (on the master branch)
: "${OPS_TF_WORKTREE:=$GITCODE/cann/ops-transformer-master}"      # clean worktree of master
: "${RECIPES_REPO:=$GITCODE/cann/cann-recipes-infer}"             # customize vendor + custom_ops binding
: "${SGLKNPU_REPO:=$GITCODE/sgl-kernel-npu}"                      # sgl_kernel_npu / deep_ep / attentions

: "${OPS_TF_URL:=https://gitcode.com/cann/ops-transformer.git}"
: "${RECIPES_URL:=https://gitcode.com/cann/cann-recipes-infer.git}"
: "${SGLKNPU_URL:=https://github.com/sgl-project/sgl-kernel-npu.git}"
: "${SGLKNPU_TAG:=2026.6.2}"

# Pinned for reproducibility: none of the three third-party repositories follows a moving branch.
# The NSA operators only exist on ops-transformer's master, which keeps moving, so the commit
# pinned below is the one this project has actually built and run. To upgrade, change it here and
# retest.
: "${OPS_TF_COMMIT:=dd9f31f34}"     # commit on ops-transformer master verified to work
: "${RECIPES_COMMIT:=c5cc95e}"      # cann-recipes-infer (origin/master at the time)

# pip constraints: pin the torch family so no dependency can move them
TORCH_LOCK="$(dirname "$(readlink -f "$0")")/dsv4_torch_lock.txt"

PIP="$PYTHON_BIN -m pip"
export ASCEND_HOME_PATH="$CANN_HOME"

log(){ echo -e "\n\033[1;36m[setup][$(date +%H:%M:%S)] $*\033[0m"; }
die(){ echo -e "\033[1;31m[setup][FATAL] $*\033[0m" >&2; exit 1; }
# The CANN/vendor set_env scripts reference undefined variables (custom_transformer's
# ASCEND_CUSTOM_OPP_PATH, for one), which aborts under set -u, so nounset is disabled while
# sourcing them.
# shellcheck disable=SC1090
source_env(){ set +u; . "$1"; set -u; }

# ----------------------------- 0. prerequisites ------------------------------
phase_prereq(){
  log "phase prereq: toolchain and permission checks"
  # umask 0002 makes the artifacts group-writable and CANN msopgen's security check aborts
  umask 0022
  [ -x "$PYTHON_BIN" ] || die "PYTHON_BIN does not exist: $PYTHON_BIN"
  [ -d "$CANN_HOME" ]  || die "CANN_HOME does not exist: $CANN_HOME"
  # The default gcc-9 cannot compile -march=...+bf16+i8mm or -std=gnu++20, so gcc >= 13 is needed
  if [ ! -x "$CC_BIN" ]; then
    echo "  [warn] $CC_BIN not found. Ubuntu/Debian: apt-get install -y gcc-13 g++-13"
    echo "         Or stay on gcc-9 and build kt-kernel with the ARM extensions off (the kt_kernel phase already does)."
  fi
  # kt-kernel's CMake requires hwloc (pkg_search_module(HWLOC REQUIRED)). Catching it here avoids
  # an opaque CMake FATAL_ERROR much later in the kt_kernel phase. Build time (hwloc.pc) and run
  # time (libhwloc.so.15) come from two different packages, so both must be present.
  pkg-config --exists hwloc 2>/dev/null \
    || die "hwloc development package missing (CMake needs hwloc.pc). Ubuntu/Debian:
  apt-get install -y pkg-config libhwloc-dev libhwloc15"
  ldconfig -p 2>/dev/null | grep -q libhwloc.so.15 \
    || die "libhwloc.so.15 missing (runtime dependency of kt_kernel_ext). Ubuntu/Debian: apt-get install -y libhwloc15"
  "$PYTHON_BIN" -c 'import sys;assert sys.version_info[:2]==(3,11),"need py3.11"' \
    || die "Python must be 3.11 (the torch, torch_npu and custom-operator wheels are all cp311)"
  echo "  CANN=$CANN_HOME  PY=$PYTHON_BIN  CC=$CC_BIN  SOC=$SOC"
  source_env "$CANN_HOME/set_env.sh"
  echo "  ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
}

# ----------------------------- 1. the torch family ---------------------------
phase_torch(){
  log "phase torch: verify or install torch 2.8 + torch_npu 2.8.0.post4"
  # A clean CANN image usually ships torch/torch_npu, but not necessarily the right versions.
  # Check first, install only on mismatch.
  if "$PYTHON_BIN" - <<'PY'
import importlib.metadata as m, sys
want={"torch":"2.8.0","torch_npu":"2.8.0.post4"}
ok=True
for k,v in want.items():
    try:
        got=m.version(k)
    except Exception:
        print(f"  missing {k}"); ok=False; continue
    if not got.startswith(v): print(f"  {k}={got} != {v}"); ok=False
sys.exit(0 if ok else 1)
PY
  then echo "  torch/torch_npu versions OK"; return; fi
  echo "  [!] torch version mismatch. Install from your image's index (the lines below are a reference):"
  cat <<EOF
    $PIP install torch==2.8.0 torchvision==0.23.0 torchaudio==2.11.0 \\
        --index-url https://download.pytorch.org/whl/cpu
    $PIP install torch_npu==2.8.0.post4    # Ascend index / internal pypi
EOF
  die "Install the torch family at the versions in torch-lock.txt, then rerun this phase"
}

# ----------------------------- 2. triton-ascend -------------------------------
phase_triton(){
  log "phase triton: triton-ascend 3.2.1.dev (matching CANN 9.0.0)"
  # triton-ascend==3.2.0 compiles npu_utils.cpp at import time and uses
  # RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE, which CANN 9.0.0 does not have, so the import fails
  # outright. The nightly 3.2.1.dev is required.
  $PIP install "triton-ascend==3.2.1.dev20260530" \
      --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \
      --trusted-host mirrors.huaweicloud.com
  "$PYTHON_BIN" -c "import triton;print('  triton import OK')"
}

# ----------------------------- 3. sglang base deps ---------------------------
phase_sglang_deps(){
  log "phase sglang_deps: install sglang (dsv4 fork) base dependencies, excluding torch and the custom operators"
  local reqs
  reqs="$(dirname "$(readlink -f "$0")")/dsv4_sglang_base_reqs.txt"
  [ -f "$reqs" ] || die "missing $reqs (ships alongside this script)"
  $PIP install -r "$reqs" -c "$TORCH_LOCK"
  # pure-python dependencies of kt-kernel
  $PIP install safetensors gguf -c "$TORCH_LOCK"
}

# ----------------------------- 4. customize vendor ---------------------------
# Fused operators from cann-recipes-infer: HcPre/HcPost/RmsNormDynamicQuant/
# InplacePartialRotaryMul/SwigluClipQuant/MoeGatingTopKHash/... → vendor "customize"
phase_vendor_customize(){
  log "phase vendor_customize: build and install the customize vendor (cann-recipes-infer @ $RECIPES_COMMIT)"
  [ -d "$RECIPES_REPO" ] || git clone "$RECIPES_URL" "$RECIPES_REPO"
  # pinned: does not follow origin/master
  git -C "$RECIPES_REPO" fetch --quiet origin || true
  git -C "$RECIPES_REPO" checkout --quiet --detach "$RECIPES_COMMIT" \
    || die "cann-recipes-infer checkout $RECIPES_COMMIT failed (dirty worktree? check git -C $RECIPES_REPO status)"
  source_env "$CANN_HOME/set_env.sh"; umask 0022
  cd "$RECIPES_REPO/ops/ascendc"
  chmod -R go-w .                                  # keeps msopgen's security check from aborting
  bash build.sh -c "$SOC"                          # builds every operator by default; A3 = ascend910_93
  local run; run=$(ls output/CANN-custom_ops-*-linux.*.run | head -1)
  [ -n "$run" ] || die "no customize .run produced; check the build log"
  chmod +x "$run"
  "./$run" --quiet --install-path="$CANN_HOME/opp"
  echo "  installed vendor: $CANN_HOME/opp/vendors/customize"
}

# ----------------------------- 5. custom_ops torch binding -------------------
# Python bindings for torch.ops.custom.*, exposing the vendor's aclnn operators to torch
phase_custom_ops(){
  log "phase custom_ops: build and install the custom_ops torch binding (cann-recipes-infer)"
  [ -d "$RECIPES_REPO" ] || die "run vendor_customize first (it clones cann-recipes-infer)"
  source_env "$CANN_HOME/set_env.sh"
  cd "$RECIPES_REPO/ops/ascendc/torch_ops_extension"
  USE_NINJA=1 bash build_and_install.sh            # build_ext + bdist_wheel + pip install -I
  # Two things matter when verifying: (1) change to a neutral directory, otherwise the local
  #    custom_ops/ source package shadows the installed wheel (its __init__.py is picked up,
  #    custom_ops_lib is missing, and it surfaces as a bogus "circular import");
  #    (2) import torch/torch_npu first, since the extension links against libc10.so and friends
  #    and needs the torch runtime loaded, otherwise it reports
  #    "libc10.so: cannot open shared object file"。
  ( cd / && "$PYTHON_BIN" -c "import torch, torch_npu; import custom_ops; print('  custom_ops import OK')" )
}

# ----------------------------- 6. custom_transformer vendor ------------------
# NSA/DSA operators: compressor / sparse_attn_sharedkv / quant_lightning_indexer (+metadata).
# These were removed from ops-transformer's 9.0.0 branch and only exist on master under
# experimental/attention/.
# Naming quirk: ops-transformer appends "_transformer" to the vendor name, so passing
# --vendor_name=custom yields the vendor "custom_transformer".
phase_vendor_transformer(){
  log "phase vendor_transformer: build and install the custom_transformer vendor (ops-transformer @ $OPS_TF_COMMIT)"
  [ -d "$OPS_TF_REPO" ] || git clone "$OPS_TF_URL" "$OPS_TF_REPO"
  # Build from a clean worktree, since a dirty master worktree contaminates the artifacts. The
  # worktree detaches straight to the verified commit rather than following master: the NSA
  # operators only live on master, which keeps moving, and following it would build untested
  # operators.
  if [ ! -d "$OPS_TF_WORKTREE" ]; then
    git -C "$OPS_TF_REPO" fetch origin master
    git -C "$OPS_TF_REPO" worktree add --detach "$OPS_TF_WORKTREE" "$OPS_TF_COMMIT" \
      || die "failed to create the ops-transformer worktree @ $OPS_TF_COMMIT"
  fi
  source_env "$CANN_HOME/set_env.sh"; umask 0022
  cd "$OPS_TF_WORKTREE"; chmod -R go-w .
  bash build.sh --pkg --experimental --soc="$SOC" --vendor_name=custom \
    --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,compressor,quant_lightning_indexer,quant_lightning_indexer_metadata \
    --cann_3rd_lib_path="$OPS_TF_REPO/third_party" -j"$JOBS"
  local run="build/cann-ops-transformer-custom_linux-aarch64.run"
  [ -f "$run" ] || die "no custom_transformer .run produced: $run"
  bash "$run" --quiet --install-path="$CANN_HOME/opp"
  echo "  installed vendor: $CANN_HOME/opp/vendors/custom_transformer"
}

# ----------------------------- 7. sgl_kernel_npu -----------------------------
# sgl_kernel_npu / deep_ep / attentions / torch_memory_saver (the NPU attention and MoE kernels)
phase_sgl_kernel_npu(){
  log "phase sgl_kernel_npu: build the sgl_kernel_npu family from source (tag $SGLKNPU_TAG)"
  if [ ! -d "$SGLKNPU_REPO" ]; then
    git clone "$SGLKNPU_URL" "$SGLKNPU_REPO"
    git -C "$SGLKNPU_REPO" checkout "$SGLKNPU_TAG"
    git -C "$SGLKNPU_REPO" submodule update --init --recursive
  fi
  source_env "$CANN_HOME/set_env.sh"; umask 0022
  cd "$SGLKNPU_REPO"; chmod -R go-w .
  # PTAExtensionOPS in csrc/attentions/csrc/CMakeLists.txt is missing -ldl, which surfaces as
  # undefined references to dlopen/dlsym; add ${CMAKE_DL_LIBS}.
  local cm="csrc/attentions/csrc/CMakeLists.txt"
  if [ -f "$cm" ] && ! grep -q "CMAKE_DL_LIBS" "$cm"; then
    echo "  [patch] adding \${CMAKE_DL_LIBS} to target_link_libraries(PTAExtensionOPS ...) in $cm"
    # sed already works line by line, so "to end of line" is just .*: the earlier [^\n] is only
    # read as "not a newline" by GNU sed, while others read it as "not a backslash and not the
    # letter n", truncating the match and inserting ${CMAKE_DL_LIBS} mid-line.
    sed -i 's/\(target_link_libraries(PTAExtensionOPS.*\)/\1 ${CMAKE_DL_LIBS}/' "$cm" || true
    grep -q "CMAKE_DL_LIBS" "$cm" || echo "  [warn] could not add -ldl automatically; add it by hand (see the document)"
  fi
  # The deep_ep vendor installs read-only, so a second build hits Permission denied on rm uninstall.sh
  chmod -R u+w python/deep_ep/deep_ep/vendors 2>/dev/null || true
  rm -rf python/deep_ep/deep_ep/vendors/hwcomputing 2>/dev/null || true
  # Do not rm csrc/attentions/build/: it is a TRACKED source directory, not build output
  bash build.sh                                    # default SOC=Ascend910_9382, i.e. A3
  $PIP install output/{sgl_kernel_npu,deep_ep,attentions,torch_memory_saver}*.whl -c "$TORCH_LOCK"
}

# ----------------------------- 8. kt-kernel ----------------------------------
phase_kt_kernel(){
  log "phase kt_kernel: build kt-kernel (CPU MoE / MXFP4 NEON)"
  # kt-kernel is model code living inside the ktransformers repo, and its Ascend backend comes
  # from a patch, so the repo must be cloned and patched first.
  [ -d "$REPO/kt-kernel" ] || die "$REPO/kt-kernel not found. kt-kernel lives inside the
  ktransformers repo: follow guide section 2 to clone ktransformers-AK into REPO=$REPO,
  section 3 to apply the patches with apply_all.sh, then rerun this phase
  (or override REPO=<your clone path>)."
  cd "$REPO/kt-kernel"
  rm -rf build/temp.linux-aarch64-cpython-311
  # With ARM SVE=ON the MXFP4 CPU MoE takes the SVE branch and reports "llamafile not supported"
  # (moe.hpp:73/77), so SVE/BF16/I8MM must all be off, leaving the verified NEON
  # armv8.2+fp16+dotprod path. The CMakeLists used to force /usr/bin/gcc and now honours CC/CXX.
  CC="$CC_BIN" CXX="$CXX_BIN" CPUINFER_USE_ASCEND_NPU=1 \
    CPUINFER_ARM_SVE=OFF CPUINFER_ARM_BF16=OFF CPUINFER_ARM_I8MM=OFF \
    "$PYTHON_BIN" setup.py build_ext --inplace
  ls python/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so \
    && echo "  kt_kernel_ext.so built OK"
}

# ----------------------------- 9. import gate --------------------------------
phase_verify(){
  log "phase verify: environment import gate (dependencies only; kt-kernel is built and
  checked separately in section 4)"
  source_env "$CANN_HOME/set_env.sh"
  source_env "$CANN_HOME/opp/vendors/custom_transformer/bin/set_env.bash"
  source_env "$CANN_HOME/opp/vendors/customize/bin/set_env.bash"
  # Run from a neutral directory: this block imports custom_ops and must not be shadowed by a
  # directory holding a package of the same name, and torch is imported first so its runtime is
  # loaded. Only the environment side is checked (torch/triton/sgl_kernel_npu/custom_ops plus the
  # torch.ops.custom.* the vendors register), all of which comes from pip/vendor installs, so no
  # PYTHONPATH is set and kt_kernel is not imported.
  cd /
  "$PYTHON_BIN" - <<'PY'
import importlib
import torch, torch_npu; print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
# These three are imported purely for their side effects (triton and sgl_kernel_npu register
# kernels, custom_ops registers torch.ops.custom.*) and the names are never used afterwards. A bare
# import would be reported as unused by flake8/ruff and need a noqa; importlib avoids the suppression.
for _mod in ("triton", "sgl_kernel_npu", "custom_ops"):
    importlib.import_module(_mod); print(_mod, "OK")
# With all three vendors and the binding in place, torch.ops.custom.* resolves. Use raise rather
# than assert: python -O strips assert entirely and the check would silently do nothing.
for op in ("compressor","npu_sparse_attn_sharedkv","npu_quant_lightning_indexer","npu_moe_gating_top_k"):
    if not hasattr(torch.ops.custom, op):
        raise RuntimeError(f"missing torch.ops.custom.{op}")
print("torch.ops.custom.* all present OK")
PY
  echo "  == Environment bring-up and checks complete. Continue with the guide:" \
       "section 2 clone -> 3 apply patches -> 4 build kt-kernel (incl. its import check)" \
       "-> 5 convert GGUF -> 8 start the server. =="
}

# ----------------------------- dispatch --------------------------------------
phase="${1:-all}"
case "$phase" in
  prereq) phase_prereq;;
  torch) phase_prereq; phase_torch;;
  triton) phase_prereq; phase_triton;;
  sglang_deps) phase_prereq; phase_sglang_deps;;
  vendor_customize) phase_prereq; phase_vendor_customize;;
  custom_ops) phase_prereq; phase_custom_ops;;
  vendor_transformer) phase_prereq; phase_vendor_transformer;;
  sgl_kernel_npu) phase_prereq; phase_sgl_kernel_npu;;
  kt_kernel) phase_prereq; phase_kt_kernel;;
  verify) phase_verify;;
  all)
    # Environment and dependency bring-up only: this does not touch the project's own repo
    # (no clone, no patches, no kt-kernel build). kt-kernel is project code and belongs to
    # section 4, built after the clone in section 2 and the patches in section 3.
    phase_prereq
    phase_torch
    phase_triton
    phase_sglang_deps
    phase_vendor_customize
    phase_custom_ops
    phase_vendor_transformer
    phase_sgl_kernel_npu
    phase_verify
    ;;
  *) die "unknown phase: $phase (see the usage in the header)";;
esac
log "phase '$phase' complete"
