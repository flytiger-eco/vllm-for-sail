#!/bin/bash
# ==============================================================================
# scripts/ppu/ppu_install_dependency.sh - PPU CI dependency setup (GitHub Actions)
# ------------------------------------------------------------------------------
# Callers: the attention, samplers and benchmarks areas of PR CI, run inside the
# PPU Pod. Installs the wheel built by this CI run first; the rest of the PPU
# stack prefers the components preinstalled in the image.
#
# Core rule (lesson learned in sglang, see its script comments): never touch
# the PPU stack preinstalled in the image (torch/vllm/triton/flashinfer etc.
# carrying a +v0.x.0.ppu2.2.0 local version). A blind `--force-reinstall`
# overwrites the image's build with another build of the same version number,
# or even downgrades it. Only install packages that are actually missing.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --retries/--timeout: by default pip retries only 5 times with no connection
# timeout cap; tightening them explicitly keeps a single call from burning the
# whole job timeout when the network hangs.
PIP_INSTALL="python3 -m pip install --no-cache-dir --retries 5 --timeout 30"
# SAIL SDK v2.2.0 PyPI source
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

# All installs from the private index go through here.
# Why wrap pip --retries in another layer: pip's retries only cover
# connection-level errors. An index returning an empty version list
# ("ERROR: ... (from versions: none)") is an application-level result that pip
# treats as a deterministic failure and does not retry, and that is exactly
# what happens when packages are pulled concurrently.
# Observed evidence:
# - 2026-09-09: with 4 areas starting Pods concurrently, all of them got stuck
#   installing the pytest toolchain with "pytest-asyncio from versions: none";
#   the index page was reachable directly, so it was treated as transient.
# - 2026-09-10 (run 34440797915): the failure grew to minutes. The empty
#   pytest-asyncio page for benchmarks/pytorch/cuda lasted >=5min, the old
#   window (4 tries x ~35s ~= 2min) could not cover it, and all three areas
#   failed (same for the old models-basic/samplers/kernels areas). Hence the
#   window was extended to ~10min with a NAS wheelhouse fallback.
# NAS wheelhouse (always attached via --find-links): when the mirror returns an
# empty page, pip can still resolve versions from the NAS wheels, so it is
# immune without waiting for retries; successful installs are written back
# (self-healing seeding). A missing directory is skipped automatically, so
# there is no hard dependency. Initial seeding (run once on any machine that
# can write to NAS):
#   mkdir -p /nas_aisw/devops/pip-wheelhouse && \
#   python3 -m pip download --no-deps pytest pytest-asyncio tblib \
#     pytest-shard pyyaml -d /nas_aisw/devops/pip-wheelhouse \
#     -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
# Backoff uses random jitter: concurrent Pods retrying in lockstep would
# collide again.
PPU_WHEELHOUSE="${PPU_WHEELHOUSE:-/nas_aisw/devops/pip-wheelhouse}"
# Test toolchain wheels pre-downloaded by the build-wheel job on a cloud runner
# with working egress; they ride along in ci-wheel/ inside the source tarball
# shipped to the Pod. A hit needs no network at all, avoiding both empty mirror
# pages and flakiness of the Pod egress proxy (ptg-green-proxy). If the
# directory is missing (local call without a wheel, or the cloud pre-download
# failed), offline-first is skipped and the index is used as before.
PPU_TOOLCHAIN_DIR="${PPU_TOOLCHAIN_DIR:-/workspace/source/ci-wheel/toolchain}"
_pip_retry() {
    local max=8 n=1 wait_s
    local wh_args=()
    if [[ -d "${PPU_WHEELHOUSE}" ]]; then
        wh_args=(--find-links "${PPU_WHEELHOUSE}")
    fi
    # Offline first: install from the local toolchain dir with --no-index and
    # return on success (no network, no self-healing write-back). This only
    # succeeds when the requested packages resolve offline; otherwise (e.g.
    # vllm is not in the toolchain) pip fails immediately (--no-index never
    # touches the network) and we fall through to the index retry loop below.
    if [[ -d "${PPU_TOOLCHAIN_DIR}" ]] && compgen -G "${PPU_TOOLCHAIN_DIR}/*.whl" > /dev/null 2>&1; then
        # shellcheck disable=SC2068  # $@ must word-split into pip args (incl. --no-deps)
        if ${PIP_INSTALL} $@ --no-index --find-links "${PPU_TOOLCHAIN_DIR}"; then
            echo "[deps] installed offline from ${PPU_TOOLCHAIN_DIR}: $*"
            return 0
        fi
        echo "[deps] offline toolchain miss, fall back to index: $*" >&2
    fi
    while true; do
        # shellcheck disable=SC2068  # $@ must word-split into pip args (incl. --no-deps)
        if ${PIP_INSTALL} $@ -i "${PPU_PIP_INDEX}" "${wh_args[@]}"; then
            # Self-healing write-back (best effort; failures do not affect the
            # main flow): store installed packages in the wheelhouse so that
            # --find-links hits them the next time the mirror returns an empty page
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

# flytiger artifactory credentials (injected from GitHub Secrets; optional when
# the image is public)
if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

# Required to run git as root in the container (setuptools-scm also relies on
# it when deriving the vllm version from git)
git config --global --add safe.directory "${REPO_ROOT}"

# ------------------------------------------------------------------------------
# [wheel] Install the built wheel
# ------------------------------------------------------------------------------
# PR CI always passes this run's wheel; local calls without one keep the
# version preinstalled in the image.
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
# [diag] Inventory: see what the image already has before deciding what to install
# ------------------------------------------------------------------------------
echo "========== [diag] preinstalled stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
python3 -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm NOT importable"
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer|deep.ep|deep.gemm|tilelang|flash.mla|flash.attn|transformers) " || true

# ------------------------------------------------------------------------------
# [deps] PPU dependencies: prefer the image's preinstalled packages and only
# fill gaps from the flytiger PyPI
# ------------------------------------------------------------------------------
# Areas first install the v0.28.0 wheel freshly built by CI; without a wheel
# they keep the v0.27.0 preinstalled in the SDK 2.2 image. Other component
# versions follow the existing PPU CI config and are only a fallback when the
# image lacks a package. If the image's version differs from the table below
# (usually newer), the image wins; never downgrade.
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

# vllm itself: no --no-deps (its dependency tree must be resolved); other PPU
# components use --no-deps per the guide
_ensure_pip_pkg "vllm" "0.28.0"
_ensure_pip_pkg "flashinfer_python" "0.6.8.post1" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_ep" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_gemm" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "tilelang" "0.1.8" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_mla" "2.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn_3" "2.8.2" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn" "2.7.4.post1" "--no-deps --force-reinstall"

# ------------------------------------------------------------------------------
# [deps] pytest test dependencies
# ------------------------------------------------------------------------------
# pytest-asyncio: required by the asyncio fixtures in tests/conftest.py
# tblib: cross-process traceback serialization for pytest (spawn worker errors)
# pytest-shard: required by --shard-id/--num-shards in single mode
# pyyaml: common dependency of the test toolchain
echo "========== [deps] pytest toolchain =========="
_pip_retry pytest pytest-asyncio tblib pytest-shard pyyaml

# ------------------------------------------------------------------------------
# [cext] Copy the installed wheel's compiled artifacts into the source tree so
# pytest can import them.
# ------------------------------------------------------------------------------
# pytest imports vllm from the source tree, so this run's wheel extensions and
# version info must be added there. Locate site-packages via sysconfig so that
# `import vllm` does not resolve to the source tree by mistake.
echo "========== [cext] copy compiled artifacts from installed wheel =========="
SP_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
SP_VLLM_DIR="${SP_SITE}/vllm"
if [ ! -d "${SP_VLLM_DIR}" ]; then
    echo "[cext] ERROR: installed vllm not found at ${SP_VLLM_DIR}" >&2
    exit 1
fi
cext_copied=0
# Recursively copy every compiled artifact in the installed vllm package,
# keeping relative paths: besides the top-level
# _C_stable_libtorch/_moe_C_stable_libtorch/..., the vllm_flash_attn/
# subdirectory also has _vllm_fa2_C/_vllm_fa3_C (FA2/FA3 availability probes;
# vllm/vllm_flash_attn/__init__.py raises ImportError without them).
# Copy everything via find instead of listing file names, so this script needs
# no change when the wheel adds or removes extensions.
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
# Wheels packaged by auditwheel put dependent libraries in vllm.libs/, with
# RPATH pointing at $ORIGIN/../vllm.libs
if [ -d "${SP_VLLM_DIR}.libs" ]; then
    cp -rf "${SP_VLLM_DIR}.libs" "${REPO_ROOT}/"
    echo "[cext] copied vllm.libs/"
    cext_copied=1
fi
# ------------------------------------------------------------------------------
# [cext] Sync the PPU flash attention adaptation from the installed wheel.
# ------------------------------------------------------------------------------
# The wheel installed by the PR comes from the same build as the current
# source, so source changes are never overwritten from an older image.
FA_DIR_SRC="${SP_VLLM_DIR}/vllm_flash_attn"
if [ ! -d "${FA_DIR_SRC}" ]; then
    echo "[cext] ERROR: no vllm_flash_attn dir in installed vllm package" >&2
    exit 1
fi
echo "[cext] installed vllm_flash_attn contents:"
ls -l "${FA_DIR_SRC}"
# The .so files were already synced by the find loop above, so this diff
# effectively reflects .py-level differences
if ! diff -rq --exclude=__pycache__ "${FA_DIR_SRC}" "${REPO_ROOT}/vllm/vllm_flash_attn" >/dev/null 2>&1; then
    echo "[cext] installed wheel differs from source tree, copying the wheel's adaptation dir:"
    diff -rq --exclude=__pycache__ "${FA_DIR_SRC}" "${REPO_ROOT}/vllm/vllm_flash_attn" || true
    cp -rf "${FA_DIR_SRC}/." "${REPO_ROOT}/vllm/vllm_flash_attn/"
    rm -rf "${REPO_ROOT}/vllm/vllm_flash_attn/__pycache__"
    echo "[cext] copied vllm/vllm_flash_attn/ (installed wheel version)"
else
    echo "[cext] installed vllm_flash_attn identical to source tree — nothing to borrow;"
    echo "[cext] upstream probe will raise without _vllm_fa2_C/_vllm_fa3_C (see verify step)"
fi
if [ "${cext_copied}" -eq 0 ]; then
    echo "[cext] ERROR: no compiled extensions found in ${SP_VLLM_DIR}" >&2
    exit 1
fi
# Targeted check: the v0.28.0 CUDA/PPU path uses vllm._C_stable_libtorch; the
# legacy vllm._C is only built for ROCm and must not be required for PPU wheels.
python3 -c "import vllm._C_stable_libtorch as c; print('[cext] vllm._C_stable_libtorch OK from source tree:', c.__file__)"
# Targeted check: after syncing this wheel's PPU adaptation, the import must
# no longer raise
python3 -c "import vllm.vllm_flash_attn as fa; print('[cext] vllm.vllm_flash_attn OK: FA2=%s FA3=%s' % (fa.FA2_AVAILABLE, fa.FA3_AVAILABLE))"

echo "========== [verify] final stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
# In this hybrid layout __version__ comes from the installed wheel's
# _version.py (0.28.0+...) rather than the source tree's 'dev' fallback, which
# by itself signals that [cext] took effect
python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "[deps] done"
