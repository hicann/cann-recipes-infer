#!/usr/bin/env bash
# Build and convert everything GLM-5.3-Flash single-die offload needs.
#
#   ./setup.sh probe       report have/build per component, change nothing
#   ./setup.sh submodules  llama.cpp + pybind11 at their pinned commits
#   ./setup.sh kt-kernel   build and install the CPU MoE extension
#   ./setup.sh gguf        convert the MXFP4 checkpoint to the per-layer GGUF set
#   ./setup.sh check       preflight; exit 0 means safe to serve
#   ./setup.sh all         submodules -> deps -> sgl-kernel -> cann-ops -> kt-kernel -> gguf -> check
#
# Every step is idempotent and skips work already done.
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./glm53_env.sh
source "${_here}/glm53_env.sh"

KT="${KTRANSFORMERS_REPO}/kt-kernel"
BUILD_ROOT="${GLM53_ARTIFACT_ROOT:-/var/tmp/glm53}/ktbuild"
mkdir -p "${BUILD_ROOT}/wheels" "${BUILD_ROOT}/tmp" "${GLM53_LOG_DIR}"

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
note() { printf '  ....  %s\n' "$1"; }
hdr()  { printf '\n== %s ==\n' "$1"; }
die()  { bad "$1"; exit 1; }

# `import custom_ops` on its own dies with "libc10.so: cannot open shared object file";
# torch must be imported first. Every probe below therefore imports torch up front.
_py() { "${GLM53_PYTHON}" -c "$1" 2>/dev/null; }
_has_py() { _py "import torch, torch_npu; import $1" >/dev/null 2>&1; }
_locked_deps_ok() {
  local lock="${GLM53_PATCH_DIR:-}/requirements.txt"
  [ -f "${lock}" ] || return 1
  "${GLM53_PYTHON}" - "${lock}" <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import sys

with open(sys.argv[1], encoding="utf-8") as lock_file:
    for raw in lock_file:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = md.version(name)
        if name == "torch":
            actual = actual.split("+")[0]
        assert actual == expected, (name, actual, expected)
PY
}
_sgl_kernel_ok() {
  "${GLM53_PYTHON}" - "${CANN_VERSION}" <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import sys

expected = {
    "sgl-kernel-npu": "2026.6.1",
    "attentions": "0.2",
}
for distribution, version in expected.items():
    actual = md.version(distribution)
    assert actual == version, (distribution, actual, version)
deep_ep_version = md.version("deep-ep")
deep_ep_prefix = f"1.0.0+146153e5.cann.{sys.argv[1]}."
assert deep_ep_version.startswith(deep_ep_prefix), (deep_ep_version, deep_ep_prefix)
import torch, torch_npu, sgl_kernel_npu, attentions
from sgl_kernel_npu.activation.swiglu_oai_quant import swiglu_oai_quant
from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant
PY
}
_ensure_deep_ep_path() {
  _has_py deep_ep && return 0
  local site
  site="$("${GLM53_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  [ -d "${site}/deep_ep" ] || return 1
  # The 20260826 wheel places deep_ep_cpp.so inside the package while __init__.py
  # imports it as a top-level module. Add that one directory to this venv only.
  printf '%s\n' "${site}/deep_ep" > "${site}/_deep_ep_cpp_path.pth"
  _has_py deep_ep
}
_kt_kernel_ok() {
  "${GLM53_PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
import torch, torch_npu
from kt_kernel import kt_kernel_ext as extension, KTMoEWrapper

assert md.version("kt-kernel") == "0.7.0.post1"
assert hasattr(extension, "init_ascend_callback_worker")
torch_version = md.version("torch").split("+")[0]
requirements = [item.replace(" ", "") for item in (md.requires("kt-kernel") or [])]
assert f"torch=={torch_version}" in requirements, (torch_version, requirements)
PY
}

# --------------------------------------------------------------------- probe
step_probe() {
  hdr "probe"
  printf '  %-22s %s\n' "CANN" "${CANN_VERSION} pkg / ${CANN_COMPILER_VERSION:-?} compiler @ ${CANN_ROOT}"
  printf '  %-22s %s\n' "SoC" "$(glm53_detect_soc || echo '?')"
  printf '  %-22s %s\n' "python" "${GLM53_PYTHON}"
  printf '  %-22s %s\n' "gcc" "$(/usr/bin/gcc -dumpfullversion 2>/dev/null || echo '?')"
  printf '  %-22s %s\n' "NUMA / cores" "${GLM53_THREADPOOL_COUNT} / $(nproc)"

  for m in torch torch_npu sglang sgl_kernel_npu deep_ep attentions kt_kernel; do
    _has_py "$m" && printf '  %-22s have\n' "$m" || printf '  %-22s \033[33mbuild\033[0m\n' "$m"
  done
  _has_py custom_ops && printf '  %-22s have\n' "custom_ops (wheel)" \
                      || printf '  %-22s absent (fine if the ops below are registered)\n' "custom_ops (wheel)"

  for d in third_party/llama.cpp third_party/pybind11 third_party/sglang; do
    if [ -n "$(ls -A "${KTRANSFORMERS_REPO}/${d}" 2>/dev/null)" ]; then
      printf '  %-22s have\n' "$(basename "$d") submodule"
    else
      printf '  %-22s \033[33mbuild\033[0m\n' "$(basename "$d") submodule"
    fi
  done

  local have want
  have="$(ls "${GLM53_GGUF_DIR}"/${GLM53_GGUF_NAME_PREFIX}*${GLM53_GGUF_NAME_SUFFIX}.gguf 2>/dev/null | wc -l)"
  want=$(( GLM53_MOE_LAYER_END - GLM53_MOE_LAYER_START + 1 ))
  printf '  %-22s %s/%s layers\n' "MXFP4 GGUF" "${have}" "${want}"
  [ "${have}" -eq "${want}" ] || printf '  %-22s needs ~%s GiB free in %s\n' "" \
      "$(awk "BEGIN{printf \"%.0f\", (${want}-${have})*3.586}")" "${GLM53_GGUF_DIR}"
}

# ---------------------------------------------------------------- submodules
step_submodules() {
  hdr "submodules"
  local need=0 sha actual
  for d in third_party/llama.cpp third_party/pybind11; do
    sha="$(git -C "${KTRANSFORMERS_REPO}" ls-tree HEAD "${d}" | awk '{print $3}')"
    actual=""
    if [ -e "${KTRANSFORMERS_REPO}/${d}/.git" ]; then
      actual="$(git -C "${KTRANSFORMERS_REPO}/${d}" rev-parse HEAD 2>/dev/null || true)"
    fi
    [ -n "${sha}" ] && [ "${actual}" = "${sha}" ] || need=1
  done
  if [ "${need}" -eq 0 ]; then ok "llama.cpp and pybind11 at their pinned commits"; return 0; fi

  # These need github. `--depth 1` alone does NOT work: it fetches the default-branch
  # tip, which does not contain the pinned SHAs, and git's fallback direct-fetch is
  # blocked by the protocol.file hardening (CVE-2022-39253). Fetch the SHAs explicitly.
  # Reuse the standard proxy variables instead of adding a model-specific equivalent.
  local proxy="${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}"
  if [ -n "${proxy}" ]; then
    # Proxy URLs commonly contain credentials. Do not persist them in setup logs.
    note "using a configured HTTP(S) proxy"
  else
    note "fetching submodules directly"
  fi
  ( cd "${KTRANSFORMERS_REPO}" || exit 1
    [ -n "${proxy}" ] && export http_proxy="${proxy}" https_proxy="${proxy}"
    git submodule init third_party/llama.cpp third_party/pybind11
    local sha url
    for d in third_party/pybind11 third_party/llama.cpp; do
      sha="$(git ls-tree HEAD "$d" | awk '{print $3}')"
      url="$(git config -f .gitmodules --get "submodule.$d.url")"
      if [ ! -e "$d/.git" ]; then
        [ -z "$(ls -A "$d" 2>/dev/null)" ] \
          || die "$d 非空但不是 git 子模块；先检查其中内容，本脚本不会覆盖"
        git clone --filter=blob:none --no-checkout "$url" "$d"
      fi
      git -C "$d" fetch --depth 1 origin "$sha"
      git -C "$d" checkout --detach "$sha"
    done ) || { bad "submodule fetch failed"; return 1; }
  ok "llama.cpp and pybind11 at their pinned commits"
}

# ----------------------------------------------------------------- kt-kernel
# ---------------------------------------------------------------- python deps
# The NPU dependency list is SGLang's own (python/pyproject_npu.toml), installed under the
# recipe's checked-in dependency lock.  The same lock creates the base environment in
# 03-python-env.sh, so a later dependency cannot silently replace torch, torch_npu, scipy,
# or another validated direct dependency.
step_deps() {
  hdr "python deps"
  local dry=0; [ "${1:-}" = "--dry-run" ] && dry=1
  local pyproject="${SGLANG_REPO}/python/pyproject_npu.toml"
  [ -f "${pyproject}" ] || die "not found: ${pyproject} -- SGLANG_REPO=${SGLANG_REPO} is not an SGLang checkout"

  local work="${GLM53_ARTIFACT_ROOT}/python-deps"
  mkdir -p "${work}"
  local reqs="${work}/sglang-npu-requirements.txt"
  local lock="${GLM53_PATCH_DIR:?GLM53_PATCH_DIR is not set}/requirements.txt"
  [ -f "${lock}" ] || die "dependency lock not found: ${lock}"

  note "reading the NPU dependency list from ${pyproject}"
  "${GLM53_PYTHON}" - "${pyproject}" "${reqs}" <<'PYDEPS' || die "could not read ${pyproject}"
import sys, tomllib
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]
with open(dst, "w") as f:
    f.write("# Generated from pyproject_npu.toml -- do not edit by hand.\n")
    for d in deps:
        f.write(d + "\n")
print(f"{len(deps)} dependencies -> {dst}")
PYDEPS

  if [ "${dry}" -eq 1 ]; then
    note "dry run; locked by ${lock}; would install:"
    sed -n '2,$p' "${reqs}" | sed 's/^/    /'
    return 0
  fi
  note "installing under lock ${lock}"
  "${GLM53_PYTHON}" -m pip install -r "${reqs}" -c "${lock}" || die "pip install failed"
  "${GLM53_PYTHON}" - "${lock}" <<'PYLOCK' || die "installed versions differ from the recipe lock"
import importlib.metadata as md
import sys

for raw in open(sys.argv[1]):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    actual = md.version(name)
    if name == "torch":
        actual = actual.split("+")[0]
    assert actual == expected, (name, actual, expected)
PYLOCK
  ok "python deps installed"
}

# ---------------------------------------------------------------- sgl-kernel-npu
# Builds sgl_kernel_npu / deep_ep / attentions.  Skipped when the image already provides
# them, which is the common case.  Ported from tools/ascend_dsv4.
step_sgl_kernel() {
  hdr "sgl-kernel-npu"
  case "${1:-}" in
    ''|--check-only) ;;
    *) die "unknown sgl-kernel option '$1' (expected --check-only)" ;;
  esac
  if _sgl_kernel_ok; then
    _ensure_deep_ep_path || die "deep_ep is installed but its extension path cannot be resolved"
    ok "locked sgl-kernel-npu stack already provided"; return 0
  fi
  [ "${1:-}" != "--check-only" ] || return 1
  umask 0022
  if [ ! -d "${SGL_KERNEL_NPU_REPO}/.git" ]; then
    note "cloning ${SGL_KERNEL_NPU_URL} -> ${SGL_KERNEL_NPU_REPO}"
    git clone --progress "${SGL_KERNEL_NPU_URL}" "${SGL_KERNEL_NPU_REPO}" || die "clone failed"
  fi
  git -C "${SGL_KERNEL_NPU_REPO}" fetch --quiet --tags origin || true
  git -C "${SGL_KERNEL_NPU_REPO}" checkout --quiet "${SGL_KERNEL_NPU_TAG}" \
    || die "cannot check out ${SGL_KERNEL_NPU_TAG} in ${SGL_KERNEL_NPU_REPO}"
  git -C "${SGL_KERNEL_NPU_REPO}" submodule update --init --recursive --progress

  local ge_patch="${_here}/sgl-kernel-npu-ge-header-guards.patch"
  [ -f "${ge_patch}" ] || die "not found: ${ge_patch}"
  if git -C "${SGL_KERNEL_NPU_REPO}" apply --unidiff-zero --reverse --check "${ge_patch}" >/dev/null 2>&1; then
    note "GE header-guard compatibility patch already applied"
  elif git -C "${SGL_KERNEL_NPU_REPO}" apply --unidiff-zero --check "${ge_patch}"; then
    git -C "${SGL_KERNEL_NPU_REPO}" apply --unidiff-zero "${ge_patch}" \
      || die "could not apply ${ge_patch}"
    note "applied GE header-guard compatibility patch"
  else
    die "${SGL_KERNEL_NPU_TAG} does not match ${ge_patch}"
  fi

  ( cd "${SGL_KERNEL_NPU_REPO}" || exit 1
    chmod -R go-w . 2>/dev/null || true
    _cm="csrc/attentions/csrc/CMakeLists.txt"
    if [ -f "${_cm}" ] && ! grep -q 'CMAKE_DL_LIBS' "${_cm}"; then
      echo "  ....  patching ${_cm} to link \${CMAKE_DL_LIBS}"
      sed -i 's/\(target_link_libraries(PTAExtensionOPS[^\n]*\)/\1 ${CMAKE_DL_LIBS}/' "${_cm}"
    fi
    chmod -R u+w python/deep_ep/deep_ep/vendors 2>/dev/null || true
    rm -rf python/deep_ep/deep_ep/vendors/hwcomputing 2>/dev/null || true
    if [ -n "${SGL_KERNEL_NPU_SOC:-}" ]; then SOC_VERSION="${SGL_KERNEL_NPU_SOC}" bash build.sh
    else bash build.sh; fi ) || die "sgl-kernel-npu build failed"

  local whls=()
  for pkg in sgl_kernel_npu deep_ep attentions torch_memory_saver; do
    for w in "${SGL_KERNEL_NPU_REPO}"/output/${pkg}*.whl; do [ -e "${w}" ] && whls+=("${w}"); done
  done
  [ "${#whls[@]}" -gt 0 ] || die "no wheels under ${SGL_KERNEL_NPU_REPO}/output -- see the build log above"
  note "installing: ${whls[*]}"
  "${GLM53_PYTHON}" -m pip install --no-deps --force-reinstall "${whls[@]}" || die "wheel install failed"
  _ensure_deep_ep_path || die "deep_ep wheel installed but deep_ep_cpp cannot be imported"
  _sgl_kernel_ok \
    && ok "locked sgl-kernel-npu stack installed" \
    || die "built sgl-kernel-npu stack does not match its version/API contract"
}

# ---------------------------------------------------------------- CANN custom operators
# Three vendor packages: `customize` (cann-recipes-infer: mHC, the quantised swiglu and
# routing kernels), the `custom_ops` torch bindings, and `custom_transformer`
# (ops-transformer: compressor / quant_lightning_indexer / sparse_attn_sharedkv, which the
# DSA layers need).  Both repos are pinned by commit in glm53_env.sh.
#
# Ported from tools/ascend_dsv4.  GLM-5.3 needs the same operator set -- checked against
# the vendor packages this line has been running on, which contain exactly these three
# transformer ops plus the cann-recipes set.
step_cann_ops() {
  hdr "CANN custom operators"
  # ⚠ Look wherever glm53_env.sh actually resolves the vendors, not only under CANN_ROOT.
  # The DSV4 recipe installs into ${CANN_ROOT}/opp and checks there; this line inherited a
  # project-local tree ($GLM53_ENV_ROOT/opp_custom/vendors), which is why
  # GLM53_OPP_CUSTOM_DIRS is a search path.  Checking only CANN_VENDORS_DIR would start a
  # 30-minute build on a box that already has the operators.
  _vendor_found() {
    local _root _ifs="${IFS}"
    IFS=:
    for _root in ${GLM53_OPP_CUSTOM_DIRS}; do
      IFS="${_ifs}"
      [ -d "${_root}/$1" ] && { echo "${_root}/$1"; return 0; }
    done
    IFS="${_ifs}"; return 1
  }
  if [ "${1:-all}" = "all" ] && _cz="$(_vendor_found customize)" && _ct="$(_vendor_found custom_transformer)" \
     && "${GLM53_PYTHON}" -c 'import torch, torch_npu, custom_ops' >/dev/null 2>&1; then
    ok "already provided: ${_cz}"
    ok "already provided: ${_ct}"; return 0
  fi
  umask 0022
  mkdir -p "${GLM53_ARTIFACT_ROOT}/vendor_packages" "${GLM53_ARTIFACT_ROOT}/wheels"
  local vendor_install_root="${CANN_VENDORS_DIR%/vendors}"
  mkdir -p "${vendor_install_root}"
  [ -n "${GLM53_SOC}" ] || GLM53_SOC="$(glm53_detect_soc)"
  [ -n "${GLM53_SOC}" ] || die "cannot determine the SoC; set GLM53_SOC=ascend910_93 (A3) or ascend910b (A2)"
  export GLM53_SOC
  note "SoC ${GLM53_SOC}"

  _clone_pinned() {
    local url="$1" dir="$2" commit="$3"
    [ -d "${dir}/.git" ] || { note "cloning ${url} -> ${dir}"; git clone --progress "${url}" "${dir}" || return 1; }
    git -C "${dir}" fetch --quiet origin || true
    git -C "${dir}" checkout --quiet --detach "${commit}" \
      || die "cannot check out ${commit} in ${dir} (dirty tree? git -C ${dir} status)"
    note "${dir} @ $(git -C "${dir}" rev-parse --short HEAD)"
  }
  _build_customize() {
    note "1/3 customize vendor (cann-recipes-infer @ ${CANN_RECIPES_COMMIT})"
    _clone_pinned "${CANN_RECIPES_URL}" "${CANN_RECIPES_REPO}" "${CANN_RECIPES_COMMIT}" || die "clone failed"
    ( cd "${CANN_RECIPES_REPO}/ops/ascendc" && chmod -R go-w . \
      && OPS_CPU_NUMBER="${GLM53_JOBS}" bash build.sh -c "${GLM53_SOC}" ) || die "customize build failed"
    local run; run="$(ls -1 "${CANN_RECIPES_REPO}/ops/ascendc"/output/CANN-custom_ops-*-linux*.run 2>/dev/null | head -1)"
    [ -n "${run}" ] || die "no .run produced; see the build log above"
    install -m 0755 "${run}" "${GLM53_ARTIFACT_ROOT}/vendor_packages/customize.run"
    bash "${GLM53_ARTIFACT_ROOT}/vendor_packages/customize.run" --quiet --install-path="${vendor_install_root}" \
      || die "customize install failed"
    ok "installed ${CANN_VENDORS_DIR}/customize"
  }
  _build_custom_ops() {
    note "2/3 custom_ops torch bindings"
    [ -d "${CANN_RECIPES_REPO}" ] || die "run the customize step first (it clones the repo)"
    ( cd "${CANN_RECIPES_REPO}/ops/ascendc/torch_ops_extension" && USE_NINJA=1 bash build_and_install.sh ) \
      || die "custom_ops build failed"
    local whl; whl="$(ls -1 "${CANN_RECIPES_REPO}/ops/ascendc/torch_ops_extension"/dist/custom_ops-*-linux_*.whl 2>/dev/null | head -1)"
    [ -n "${whl}" ] || die "no custom_ops wheel produced"
    install -m 0644 "${whl}" "${GLM53_ARTIFACT_ROOT}/wheels/"
    ok "wheel -> ${GLM53_ARTIFACT_ROOT}/wheels/$(basename "${whl}")"
  }
  _build_transformer() {
    note "3/3 custom_transformer vendor (ops-transformer @ ${OPS_TRANSFORMER_COMMIT})"
    _clone_pinned "${OPS_TRANSFORMER_URL}" "${OPS_TRANSFORMER_REPO}" "${OPS_TRANSFORMER_COMMIT}" || die "clone failed"
    ( cd "${OPS_TRANSFORMER_REPO}" && chmod -R go-w . \
      && bash build.sh --pkg --experimental --soc="${GLM53_SOC}" --vendor_name=custom \
           --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,compressor,quant_lightning_indexer,quant_lightning_indexer_metadata \
           --cann_3rd_lib_path="${OPS_TRANSFORMER_REPO}/third_party" -j"${GLM53_JOBS}" ) \
      || die "ops-transformer build failed"
    local run="${OPS_TRANSFORMER_REPO}/build/cann-ops-transformer-custom_linux-$(uname -m).run"
    [ -f "${run}" ] || die "no .run produced at ${run}"
    install -m 0755 "${run}" "${GLM53_ARTIFACT_ROOT}/vendor_packages/custom_transformer.run"
    bash "${GLM53_ARTIFACT_ROOT}/vendor_packages/custom_transformer.run" --quiet --install-path="${vendor_install_root}" \
      || die "custom_transformer install failed"
    ok "installed ${CANN_VENDORS_DIR}/custom_transformer"
  }

  case "${1:-all}" in
    customize)   _build_customize ;;
    custom_ops)  _build_custom_ops ;;
    transformer) _build_transformer ;;
    all)         _build_customize && _build_custom_ops && _build_transformer ;;
    *) die "unknown cann-ops step '${1}' (expected: all | customize | custom_ops | transformer)" ;;
  esac
  glm53_export_vendor_paths
}

step_kt_kernel() {
  hdr "kt-kernel"
  if _kt_kernel_ok; then
    ok "locked kt_kernel Ascend build already installed"; return 0
  fi
  for d in third_party/llama.cpp third_party/pybind11; do
    [ -n "$(ls -A "${KTRANSFORMERS_REPO}/${d}" 2>/dev/null)" ] || { bad "${d} is empty; run: $0 submodules"; return 1; }
  done
  pkg-config --exists hwloc || { bad "hwloc not found (CMake marks it REQUIRED): apt install libhwloc-dev"; return 1; }
  local gccv; gccv="$(/usr/bin/gcc -dumpversion 2>/dev/null | cut -d. -f1)"
  [ "${gccv:-0}" -ge 11 ] || { bad "/usr/bin/gcc is ${gccv}; kt-kernel is C++20 and needs >= 11"; return 1; }

  # Keep the ~22 MB CMake tree under the artifact root by symlinking kt-kernel/build.
  # It is gitignored, so it does not dirty the tree.
  #
  # `[ ! -e ]` is false for a symlink whose target is gone -- and the target lives under
  # /var/tmp, which a reboot clears. That left a dangling link in the source tree that
  # this function then declined to repair, and cmake failed somewhere less obvious.
  # Test the link itself, not what it points at.
  mkdir -p "${BUILD_ROOT}/build"
  if [ -L "${KT}/build" ] && [ ! -d "${KT}/build" ]; then
    warn "kt-kernel/build was a dangling symlink (its target under ${BUILD_ROOT} is gone); relinking"
    rm -f "${KT}/build"
  fi
  if [ ! -e "${KT}/build" ] && [ ! -L "${KT}/build" ]; then
    ln -s "${BUILD_ROOT}/build" "${KT}/build"
  fi

  ( cd "${KT}" || exit 1
    export TMPDIR="${BUILD_ROOT}/tmp"
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    export CPUINFER_USE_ASCEND_NPU=1
    export ASCEND_TOOLKIT_HOME="${CANN_ROOT}"
    export CPUINFER_PARALLEL="${GLM53_JOBS}"
    # The validated DeepSeek-V4 Ascend recipe builds plain NEON. SVE/BF16/I8MM are left
    # off deliberately: they are untested on this path, not known-absent.
    export CPUINFER_ARM_SVE="${CPUINFER_ARM_SVE:-OFF}"
    export CPUINFER_ARM_BF16="${CPUINFER_ARM_BF16:-OFF}"
    export CPUINFER_ARM_I8MM="${CPUINFER_ARM_I8MM:-OFF}"
    "${GLM53_PYTHON}" -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir "${BUILD_ROOT}/wheels" . ) 2>&1 | tail -5
  # Install the wheel we just built by name, never a glob: older builds linger here.
  local whl; whl="$(ls -t "${BUILD_ROOT}"/wheels/kt_kernel-*.whl 2>/dev/null | head -1)"
  [ -n "${whl}" ] || { bad "no wheel produced"; return 1; }
  # Dependencies were installed and checked from the recipe lock in step_deps. Keep this
  # wheel install non-mutating, then verify kt-kernel's torch contract directly. A global
  # pip check would also inspect unrelated CANN packages exposed through PYTHONPATH.
  "${GLM53_PYTHON}" -m pip install -q --no-deps --force-reinstall "${whl}" || { bad "install failed"; return 1; }
  _kt_kernel_ok || { bad "kt-kernel version/API/torch contract does not match the environment"; return 1; }
  ok "installed $(basename "${whl}")"
}

# ---------------------------------------------------------------------- gguf
step_gguf() {
  hdr "gguf"
  local want have need_gib avail_gib
  want=$(( GLM53_MOE_LAYER_END - GLM53_MOE_LAYER_START + 1 ))
  mkdir -p "${GLM53_GGUF_DIR}"
  have="$(ls "${GLM53_GGUF_DIR}"/${GLM53_GGUF_NAME_PREFIX}*${GLM53_GGUF_NAME_SUFFIX}.gguf 2>/dev/null | wc -l)"
  # A matching file count used to return here, so the bit-exact check ran only on the
  # very first conversion -- never on a re-run, never on a fresh clone against a
  # pre-populated volume. Thereafter the whole defence was `ls | wc -l`, which passes on
  # a truncated file, a wrong-content file, or the right count of the wrong layers.
  # Skip the conversion, never the verification.
  if [ "${have}" -eq "${want}" ]; then
    ok "${want}/${want} layers present, skipping conversion -- verifying anyway"
    _glm53_verify_gguf "${want}"; return $?
  fi

  [ -d "${GLM53_MXFP4_CKPT}" ] || { bad "no MXFP4 checkpoint at ${GLM53_MXFP4_CKPT}"; return 1; }
  # 3.586 GiB per layer, measured (3.850 GB): 288 experts x (2 x 2048 x 128 + 4096 x 64)
  # blocks x 17 bytes.
  need_gib="$(awk "BEGIN{printf \"%.0f\", (${want}-${have})*3.586}")"
  avail_gib="$(df -BG --output=avail "${GLM53_GGUF_DIR}" | tail -1 | tr -dc '0-9')"
  note "need ~${need_gib} GiB, ${avail_gib} GiB free on $(df --output=target "${GLM53_GGUF_DIR}" | tail -1)"
  if [ "${avail_gib}" -lt "${need_gib}" ]; then
    bad "not enough space. Free some, or set GLM53_GGUF_DIR to a bigger volume."
    return 1
  fi

  "${GLM53_PYTHON}" "${KT}/tools/mxfp4_gguf/convert_mxfp4_gguf.py" batch \
      --input "${GLM53_MXFP4_CKPT}" --output-dir "${GLM53_GGUF_DIR}" \
      --layer-start "${GLM53_MOE_LAYER_START}" --layer-end "${GLM53_MOE_LAYER_END}" \
      --name-prefix "${GLM53_GGUF_NAME_PREFIX}" --name-suffix "${GLM53_GGUF_NAME_SUFFIX}" \
      --jobs "${GLM53_JOBS}" --skip-existing \
    2>&1 | tee "${GLM53_LOG_DIR}/gguf.log" | grep -vE '^\[convert\]|Writing:'
  [ "${PIPESTATUS[0]}" -eq 0 ] || { bad "conversion failed, see ${GLM53_LOG_DIR}/gguf.log"; return 1; }

  _glm53_verify_gguf "${want}" || return 1
  ok "${want} layers converted and verified bit-exact"
}

# Bit-exact, not approximate: dequantized source safetensors must equal dequantized GGUF
# blocks. This catches a wrong nibble order or a scale misread, both of which otherwise
# surface only as "the model is a bit worse".
_glm53_verify_gguf() {
  local want="$1"
  "${GLM53_PYTHON}" "${KT}/tools/mxfp4_gguf/verify_mxfp4_gguf.py" set \
      --dir "${GLM53_GGUF_DIR}" --name-tpl "${GLM53_GGUF_NAME_PREFIX}{L}${GLM53_GGUF_NAME_SUFFIX}.gguf" \
      --layer-start "${GLM53_MOE_LAYER_START}" --layer-end "${GLM53_MOE_LAYER_END}" \
      --expect-layers "${want}" --deep 3 --model-dir "${GLM53_MXFP4_CKPT}" \
    || { bad "GGUF verification failed"; return 1; }
}

# --------------------------------------------------------------------- check
step_check() {
  hdr "check"
  local fail=0
  _locked_deps_ok && ok "all recipe-locked Python dependencies" \
      || { bad "one or more Python dependencies differ from requirements.txt"; fail=1; }
  _has_py torch      || { bad "torch";      fail=1; }
  _py "import torch, torch_npu; assert torch.npu.is_available()" >/dev/null 2>&1 \
      && ok "torch_npu sees a device" || { bad "torch.npu.is_available() is False"; fail=1; }

  # Probe the registered operators, not the wheel: this line supplies them through
  # ASCEND_CUSTOM_OPP_PATH vendor packages, so `import custom_ops` can legitimately
  # fail while every operator is present.
  local missing
  missing="$(_py "
import torch, torch_npu
try: import custom_ops
except Exception: pass
need = ['npu_moe_gating_top_k','npu_dequant_swiglu_clamp_quant','compressor',
        'npu_sparse_attn_sharedkv','npu_quant_lightning_indexer']
print(' '.join(n for n in need if not hasattr(torch.ops.custom, n)))")"
  if [ -z "${missing}" ]; then ok "custom operators registered"
  else bad "custom operators missing: ${missing} (check ASCEND_CUSTOM_OPP_PATH)"; fail=1; fi

  _sgl_kernel_ok && ok "locked sgl-kernel-npu version/API contract" \
      || { bad "sgl-kernel-npu version/API contract"; fail=1; }
  _has_py deep_ep && ok "deep_ep importable" \
      || { bad "deep_ep import (rerun: ./setup.sh sgl-kernel)"; fail=1; }
  _kt_kernel_ok && ok "locked kt_kernel version/API/torch contract" \
      || { bad "kt_kernel version/API/torch contract"; fail=1; }
  _py "import gguf, sys; sys.exit(0 if int(gguf.GGMLQuantizationType.MXFP4)==39 else 1)" \
      && ok "gguf knows GGML_TYPE_MXFP4 (39)" || { bad "gguf has no MXFP4 type"; fail=1; }

  local sg; sg="$(_py "import sglang; print(sglang.__file__)")"
  case "${sg}" in
    "${SGLANG_REPO}"/python/sglang/__init__.py) ok "sglang resolves to ${SGLANG_REPO}" ;;
    *) bad "sglang resolves to ${sg:-<none>}, not ${SGLANG_REPO}"; fail=1 ;;
  esac
  _py "import sglang.srt.layers.moe.kt_ep_wrapper as m; m.KTEPWrapperMethod" >/dev/null 2>&1 \
      && ok "sglang has the KT offload wrapper" || { bad "sglang lacks kt_ep_wrapper"; fail=1; }

  [ -f "${GLM53_MODEL_PATH}/config.json" ] && ok "INT8 checkpoint present" \
      || { bad "no config.json under ${GLM53_MODEL_PATH}"; fail=1; }
  local have want
  want=$(( GLM53_MOE_LAYER_END - GLM53_MOE_LAYER_START + 1 ))
  have="$(ls "${GLM53_GGUF_DIR}"/${GLM53_GGUF_NAME_PREFIX}*${GLM53_GGUF_NAME_SUFFIX}.gguf 2>/dev/null | wc -l)"
  [ "${have}" -eq "${want}" ] && ok "${want} GGUF layers present" \
      || { bad "${have}/${want} GGUF layers -- a missing layer serves garbage, it does not error"; fail=1; }
  case "${GLM53_GGUF_TEMPLATE}" in *'{layer_idx}'*) ok "--kt-weight-path template has {layer_idx}" ;;
      *) bad "GLM53_GGUF_TEMPLATE has no {layer_idx}: every layer would load the same file"; fail=1 ;; esac

  [ $(( GLM53_CHUNKED_PREFILL_SIZE % 64 )) -eq 0 ] && [ "${GLM53_CHUNKED_PREFILL_SIZE}" -gt 0 ] \
      && ok "chunked-prefill-size ${GLM53_CHUNKED_PREFILL_SIZE}" \
      || { bad "chunked-prefill-size must be a positive multiple of 64"; fail=1; }

  # The CPU MoE keeps the non-resident experts in node-local buffers.
  local ram_gib cpu_gib
  ram_gib="$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)"
  cpu_gib="$(awk "BEGIN{printf \"%.0f\", 3.586*${want}*(288-${GLM53_NUM_GPU_EXPERTS})/288}")"
  [ "${ram_gib}" -gt $(( cpu_gib + 40 )) ] \
      && ok "host RAM ${ram_gib} GiB (offloaded experts need ~${cpu_gib} GiB)" \
      || warn "host RAM ${ram_gib} GiB is tight: offloaded experts alone need ~${cpu_gib} GiB"

  if [ -n "${http_proxy:-}${HTTP_PROXY:-}${https_proxy:-}${HTTPS_PROXY:-}${all_proxy:-}${ALL_PROXY:-}" ]; then
    local _proxy_bypass=",${no_proxy:-},${NO_PROXY:-}," _loopback_ok=1
    case "${_proxy_bypass}" in
      *,127.0.0.1,*) ;;
      *) bad "download proxy is set but NO_PROXY/no_proxy lacks 127.0.0.1"; fail=1; _loopback_ok=0 ;;
    esac
    case "${_proxy_bypass}" in
      *,localhost,*) ;;
      *) bad "download proxy is set but NO_PROXY/no_proxy lacks localhost"; fail=1; _loopback_ok=0 ;;
    esac
    [ "${_loopback_ok}" -eq 0 ] || ok "download proxy configured; NO_PROXY/no_proxy bypasses loopback"
  else
    ok "no download proxy configured"
  fi

  echo
  if [ "${fail}" -eq 0 ]; then echo -e "\033[32mPREFLIGHT OK\033[0m"; else echo -e "\033[31mPREFLIGHT FAILED\033[0m"; fi
  return "${fail}"
}

case "${1:-all}" in
  probe)      step_probe ;;
  submodules) step_submodules ;;
  deps)       step_deps "${2:-}" ;;
  sgl-kernel) step_sgl_kernel "${2:-}" ;;
  cann-ops)   step_cann_ops "${2:-all}" ;;
  kt-kernel)  step_kt_kernel ;;
  gguf)       step_gguf ;;
  check)      step_check ;;
  all)        step_submodules && step_deps && step_sgl_kernel && step_cann_ops \
                && step_kt_kernel && step_gguf && step_check ;;
  *) echo "usage: $0 {probe|submodules|deps|sgl-kernel|cann-ops|kt-kernel|gguf|check|all}" >&2; exit 2 ;;
esac
