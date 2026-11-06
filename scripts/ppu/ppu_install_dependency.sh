#!/bin/bash
# ==============================================================================
# scripts/ppu/ppu_install_dependency.sh -- PPU CI dependency install (GitHub Actions)
# ------------------------------------------------------------------------------
# Callers: ppu-nightly-ci.yml (every area matrix leg) and
#   ppu-{attention,samplers,benchmarks}-pr.yml, running inside the PPU base image container:
#   pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-...-vllm0.27.0-py312-20261001
#
# References:
#   - on the Aone CI side the wheel/dependency install is spread over
#     .aoneci/build-wheel-ppu.yaml and the [setup] section of aone_ci/scripts/test_area_ppu_lora.sh
#   - the layout of sglang-for-sail scripts/ci/ppu/ppu_install_dependency.sh
#
# Core rule (learned from sglang, see its script comments): never touch the PPU stack the
# image ships (torch/vllm/triton/flashinfer and friends built with a +v0.x.0.ppu2.2.0
# local version) - a blind `--force-reinstall` would overwrite the image build with another
# build of the same version, or even downgrade it. Install only what the image lacks.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --retries/--timeout: pip retries only 5 times by default with no connect timeout, so
# tighten both explicitly; one call can then never eat the whole job timeout on a hung network.
PIP_INSTALL="python3 -m pip install --no-cache-dir --retries 5 --timeout 30"
# SAIL SDK v2.2.0 PyPI source
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

# Every install from the private index goes through here.
# Why wrap pip --retries: pip's retry only covers connection-level errors, while an index
# returning an empty version list ("ERROR: ... (from versions: none)") is an application-level
# result that pip treats as a deterministic failure and does not retry - exactly what happens
# when many pods pull packages at once.
# Measurements:
# - 2026-09-09: 4 areas starting pods concurrently all hung on the pytest toolchain install,
#   reporting pytest-asyncio from versions: none; the index page was healthy at the time, so
#   it was treated as a blip.
# - 2026-09-10 (run 34440797915): the failures lasted minutes - the empty pytest-asyncio
#   page for benchmarks/pytorch/cuda stayed >=5min, the old window (4 x ~35s = 2min) could
#   not cover it and all three areas died (the older models-basic/samplers/kernels areas hit
#   the same). The window is therefore widened to ~10min plus a NAS wheelhouse fallback.
# NAS wheelhouse (--find-links always appended): when the mirror returns an empty page pip
# can still resolve versions from the NAS wheels and is immune without waiting; a successful
# install is written back (self-healing seed). A missing directory is skipped, no hard
# dependency. First seeding (run once on any machine that can write to NAS):
#   mkdir -p /nas_aisw/devops/pip-wheelhouse && \
#   python3 -m pip download --no-deps pytest pytest-asyncio tblib \
#     pytest-shard pyyaml -d /nas_aisw/devops/pip-wheelhouse \
#     -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
# The backoff carries random jitter: concurrent pods retrying in lockstep would collide again.
PPU_WHEELHOUSE="${PPU_WHEELHOUSE:-/nas_aisw/devops/pip-wheelhouse}"
# Test toolchain wheels the build-wheel job pre-downloads on a cloud runner with working
# egress, packed into the source tarball under ci-wheel/ and shipped into the pod. A hit means
# no network at all - it dodges both the empty mirror page and the flaky pod egress proxy
# (ptg-green-proxy). A missing directory (dispatch/push builds without wheels, or a failed
# pre-download) simply skips the offline-first path and goes to the index as before.
PPU_TOOLCHAIN_DIR="${PPU_TOOLCHAIN_DIR:-/workspace/source/ci-wheel/toolchain}"
_pip_retry() {
    local max=8 n=1 wait_s
    local wh_args=()
    if [[ -d "${PPU_WHEELHOUSE}" ]]; then
        wh_args=(--find-links "${PPU_WHEELHOUSE}")
    fi
    # Offline first: install from the local toolchain directory with --no-index and return on
    # success (no network, no self-healing write-back). It only succeeds when the requested
    # packages resolve offline; otherwise (vllm and friends are not in the toolchain) pip fails
    # immediately (--no-index never touches the network) and the retry loop below takes over.
    if [[ -d "${PPU_TOOLCHAIN_DIR}" ]] && compgen -G "${PPU_TOOLCHAIN_DIR}/*.whl" > /dev/null 2>&1; then
        # shellcheck disable=SC2068  # $@ must stay word-split into pip arguments (--no-deps etc.)
        if ${PIP_INSTALL} $@ --no-index --find-links "${PPU_TOOLCHAIN_DIR}"; then
            echo "[deps] installed offline from ${PPU_TOOLCHAIN_DIR}: $*"
            return 0
        fi
        echo "[deps] offline toolchain miss, fall back to index: $*" >&2
    fi
    while true; do
        # shellcheck disable=SC2068  # $@ must stay word-split into pip arguments (--no-deps etc.)
        if ${PIP_INSTALL} $@ -i "${PPU_PIP_INDEX}" "${wh_args[@]}"; then
            # Self-healing write-back (best effort, failures do not affect the main flow): the
            # packages that installed fine land in the wheelhouse so later empty mirror pages
            # hit them through --find-links
            if [[ -d "${PPU_WHEELHOUSE}" ]] && touch "${PPU_WHEELHOUSE}/.rw_probe" 2>/dev/null; then
                rm -f "${PPU_WHEELHOUSE}/.rw_probe"
                local pkgs=() a
                for a in "$@"; do [[ "${a}" == -* ]] || pkgs+=("${a}"); done
                python3 -m pip download --no-cache-dir --no-deps -q \
                    "${pkgs[@]}" -d "${PPU_WHEELHOUSE}" -i "${PPU_PIP_INDEX}" \
                    2>/dev/null || true
            fi
            return 0
        fi
        if [[ ${n} -ge ${max} ]]; then
            echo "[deps] ERROR: pip install failed after ${max} attempts: $*" >&2
            return 1
        fi
        wait_s=$(( n * 30 > 90 ? 90 : n * 30 ))
        wait_s=$(( wait_s + RANDOM % 30 ))
        echo "[deps] pip install failed (attempt ${n}/${max}), retry in ${wait_s}s: $*" >&2
        sleep "${wait_s}"
        n=$(( n + 1 ))
    done
}

# flytiger artifactory credentials (injected from GitHub secrets; not needed when the image is public)
if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

# Required to run git as root in the container (setuptools-scm needs it to derive the vllm version)
git config --global --add safe.directory "${REPO_ROOT}"

# ------------------------------------------------------------------------------
# [wheel] install the wheel built by CI
# ------------------------------------------------------------------------------
# dispatch/push -> skipped, the image-provided vllm is kept (current behaviour).
if [[ -n "${PPU_WHEEL_PATH:-}" ]]; then
    if [[ ! -f "${PPU_WHEEL_PATH}" ]]; then
        echo "[wheel] ERROR: PPU_WHEEL_PATH set but not found: ${PPU_WHEEL_PATH}" >&2
        exit 1
    fi
    echo "========== [wheel] install CI-built wheel =========="
    echo "[wheel] ${PPU_WHEEL_PATH}"
    ${PIP_INSTALL} --force-reinstall --no-deps "${PPU_WHEEL_PATH}"
fi

# ------------------------------------------------------------------------------
# [diag] environment inventory: see what the image already has before installing anything
# ------------------------------------------------------------------------------
echo "========== [diag] preinstalled stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
python3 -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm NOT importable"
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer|deep.ep|deep.gemm|tilelang|flash.mla|flash.attn|transformers) " || true

# ------------------------------------------------------------------------------
# [deps] PPU dependencies: prefer the image build, only fill gaps from the flytiger PyPI
# ------------------------------------------------------------------------------
# The area legs first install the v0.28.0 wheel built by CI; without a wheel the v0.27.0
# preinstalled in the SDK 2.2 image is kept. The other components follow the existing PPU CI
# configuration and only fill gaps. The image may carry newer versions than the table below;
# the image always wins and is never downgraded.
_ensure_pip_pkg() {
    local pkg="$1" ver="$2" extra_args="${3:-}"
    local installed
    installed="$(python3 -c "import importlib.metadata as m; print(m.version('${pkg}'))" 2>/dev/null || true)"
    if [[ -n "${installed}" ]]; then
        echo "[deps] ${pkg} already installed: ${installed} — keep image build"
        return 0
    fi
    echo "[deps] installing ${pkg}==${ver}"
    # shellcheck disable=SC2086  # extra_args is intentionally word-split
    _pip_retry "${pkg}==${ver}" ${extra_args}
}

# vllm itself: no --no-deps (its dependency tree must be resolved); other components use --no-deps
_ensure_pip_pkg "vllm" "0.28.0"
_ensure_pip_pkg "flashinfer_python" "0.6.8.post1" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_ep" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_gemm" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "tilelang" "0.1.8" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_mla" "2.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn_3" "2.8.2" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn" "2.7.4.post1" "--no-deps --force-reinstall"

# ------------------------------------------------------------------------------
# [deps] pytest test dependencies (mirrors the [setup] section of test_area_ppu_lora.sh)
# ------------------------------------------------------------------------------
# pytest-asyncio: needed by the asyncio fixtures in tests/conftest.py
# tblib: serialises tracebacks across pytest processes (when a spawn worker reports an error)
# pytest-shard: needed by --shard-id/--num-shards in single mode
# pyyaml: general fallback dependency (the case selection is inlined in ppu_ci_area_lora.sh)
echo "========== [deps] pytest toolchain =========="
_pip_retry pytest pytest-asyncio tblib pytest-shard pyyaml

# ------------------------------------------------------------------------------
# [cext] fill the source tree with compiled artifacts: image vllm C extensions -> REPO_ROOT/vllm
# ------------------------------------------------------------------------------
# pytest uses REPO_ROOT as rootdir and the tests package (tests/__init__.py) prepends
# REPO_ROOT to sys.path, so `import vllm` resolves to the source tree (pure Python, no
# compiled artifacts) instead of the complete image build in site-packages.
# vllm/platforms/ppu.py inherits NvmlCudaPlatform while cuda.py imports
# vllm._C_stable_libtorch at the top level, so conftest fails as soon as it loads because
# the source tree lacks that extension.
# Bring-up approach: copy the compiled artifacts of the preinstalled wheel into the source
# tree, giving the hybrid "Python layer = branch sources + C extensions = image build"
# shape. ABI premise: the branch does not change the csrc/ Python bindings (the current
# changes are pure Python under vllm/lora). The official flow later switches to the wheel
# build (mirroring Aone .aoneci/build-wheel-ppu.yaml), at which point this section can go.
# Gate impact: because the C extensions come from the image rather than the change under
# test, a PR touching csrc/ still exercises the old .so from the image and a green run is a
# false green. That is why the kernels area path filter deliberately excludes csrc/**;
# restore it together with the PPU wheel build.
# Note: locate site-packages with sysconfig instead of `import vllm` - with cwd=REPO_ROOT
# that import goes through the source tree and gives the wrong answer.
echo "========== [cext] borrow compiled extensions from image vllm =========="
SP_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
SP_VLLM_DIR="${SP_SITE}/vllm"
if [ ! -d "${SP_VLLM_DIR}" ]; then
    echo "[cext] ERROR: image vllm not found at ${SP_VLLM_DIR}" >&2
    exit 1
fi
cext_copied=0
# Recursively copy every compiled artifact of the image vllm package, keeping relative
# paths: besides the top-level _C_stable_libtorch/_moe_C_stable_libtorch/..., the
# vllm_flash_attn/ subdirectory also holds _vllm_fa2_C/_vllm_fa3_C (the FA2/FA3
# availability probes, whose absence makes vllm/vllm_flash_attn/__init__.py raise
# ImportError). find copies everything instead of enumerating file names, so the script
# needs no update when the image adds or drops extensions.
while IFS= read -r f; do
    rel="${f#"${SP_VLLM_DIR}"/}"
    dest="${REPO_ROOT}/vllm/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "$f" "${dest}"
    echo "[cext] copied vllm/${rel}"
    cext_copied=1
done < <(find "${SP_VLLM_DIR}" -maxdepth 3 -type f -name '*.so')
if [ -f "${SP_VLLM_DIR}/_version.py" ]; then
    cp -f "${SP_VLLM_DIR}/_version.py" "${REPO_ROOT}/vllm/"
    echo "[cext] copied vllm/_version.py"
    cext_copied=1
fi
# An auditwheel-packed wheel keeps its dependency libraries in vllm.libs/ with RPATH $ORIGIN/../vllm.libs
if [ -d "${SP_VLLM_DIR}.libs" ]; then
    cp -rf "${SP_VLLM_DIR}.libs" "${REPO_ROOT}/"
    echo "[cext] copied vllm.libs/"
    cext_copied=1
fi
# ------------------------------------------------------------------------------
# [cext] vllm_flash_attn directory: borrow the image version wholesale (PPU adaptation)
# ------------------------------------------------------------------------------
# The PPU image does not compile the CUDA _vllm_fa2_C/_vllm_fa3_C extensions (PPU's FA
# goes through the separate flash_attn/flash_attn_3 pip packages), and the upstream
# vllm/vllm_flash_attn/__init__.py in the source tree raises ImportError when both are
# missing; the is_ppu() branch of fa_utils.py imports vllm.vllm_flash_attn at the top
# level, so test collection dies. When the same-named directory in the image
# site-packages is the PPU-adapted version (different .py probes), borrow the whole
# directory including the .py files so the behaviour matches the image. Note: borrowing
# overrides branch changes to that directory, acceptable during bring-up (they are
# concentrated in vllm/lora); this section can go once the wheel build flow is in place.
FA_DIR_SRC="${SP_VLLM_DIR}/vllm_flash_attn"
if [ ! -d "${FA_DIR_SRC}" ]; then
    echo "[cext] ERROR: no vllm_flash_attn dir in image vllm package" >&2
    exit 1
fi
echo "[cext] image vllm_flash_attn contents:"
ls -l "${FA_DIR_SRC}"
# The .so files were already synced by the find loop above, so this diff reflects the .py layer
if ! diff -rq --exclude=__pycache__ "${FA_DIR_SRC}" "${REPO_ROOT}/vllm/vllm_flash_attn" >/dev/null 2>&1; then
    echo "[cext] image version differs from source tree — borrowing image copy:"
    diff -rq --exclude=__pycache__ "${FA_DIR_SRC}" "${REPO_ROOT}/vllm/vllm_flash_attn" || true
    cp -rf "${FA_DIR_SRC}/." "${REPO_ROOT}/vllm/vllm_flash_attn/"
    rm -rf "${REPO_ROOT}/vllm/vllm_flash_attn/__pycache__"
    echo "[cext] copied vllm/vllm_flash_attn/ (image version)"
else
    echo "[cext] image vllm_flash_attn identical to source tree — nothing to borrow;"
    echo "[cext] upstream probe will raise without _vllm_fa2_C/_vllm_fa3_C (see verify step)"
fi
if [ "${cext_copied}" -eq 0 ]; then
    echo "[cext] ERROR: no compiled extensions found in ${SP_VLLM_DIR}" >&2
    exit 1
fi
# Targeted check: the v0.28.0 CUDA/PPU path uses vllm._C_stable_libtorch; the legacy
# vllm._C is only built for ROCm and must not be a required extension of the PPU wheel.
python3 -c "import vllm._C_stable_libtorch as c; print('[cext] vllm._C_stable_libtorch OK from source tree:', c.__file__)"
# Targeted check: after borrowing the PPU-adapted vllm_flash_attn the import must no longer raise
python3 -c "import vllm.vllm_flash_attn as fa; print('[cext] vllm.vllm_flash_attn OK: FA2=%s FA3=%s' % (fa.FA2_AVAILABLE, fa.FA3_AVAILABLE))"

echo "========== [verify] final stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
# In the hybrid shape __version__ comes from the installed wheel's _version.py (0.28.0+...)
# instead of the source tree 'dev' fallback - itself the signal that [cext] worked
python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "[deps] done"
