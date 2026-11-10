#!/bin/bash
# Install one verified nightly wheel and the dependencies declared by its release.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PIP_INSTALL="python3 -m pip install --no-cache-dir --retries 5 --timeout 30"
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
PPU_WHEELHOUSE="${PPU_WHEELHOUSE:-/nas_aisw/devops/pip-wheelhouse}"
PPU_TOOLCHAIN_DIR="${PPU_TOOLCHAIN_DIR:-/workspace/source/source/ci-wheel/toolchain}"

: "${PPU_WHEEL_PATH:?nightly requires the wheel built for this release}"
: "${PPU_WHEEL_SHA256:?nightly requires the wheel digest}"
: "${PPU_SOURCE_SHA:?nightly requires the release source SHA}"
: "${PPU_CI_SHA:?nightly requires the CI source SHA}"
: "${PPU_RELEASE:?nightly requires the release name}"
: "${PPU_EXPECTED_VLLM_VERSION:?nightly requires the wheel version}"
[[ "${PPU_WHEEL_PATH}" = /* && -f "${PPU_WHEEL_PATH}" ]]
[[ "${PPU_WHEEL_SHA256}" =~ ^[0-9a-f]{64}$ ]]
[[ "${PPU_SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PPU_CI_SHA}" =~ ^[0-9a-f]{40}$ ]]
case "${PPU_RELEASE}" in v0.26.0|v0.27.0|v0.28.0) ;; *) exit 2 ;; esac
printf '%s  %s\n' "${PPU_WHEEL_SHA256}" "${PPU_WHEEL_PATH}" | sha256sum -c -

_pip_retry() {
    local max=8 n=1 wait_s
    local wh_args=()
    if [[ -d "${PPU_WHEELHOUSE}" ]]; then
        wh_args=(--find-links "${PPU_WHEELHOUSE}")
    fi
    if [[ -d "${PPU_TOOLCHAIN_DIR}" ]] && compgen -G "${PPU_TOOLCHAIN_DIR}/*.whl" > /dev/null 2>&1; then
        # shellcheck disable=SC2068
        if ${PIP_INSTALL} $@ --no-index --find-links "${PPU_TOOLCHAIN_DIR}"; then
            return 0
        fi
    fi
    while true; do
        # shellcheck disable=SC2068
        if ${PIP_INSTALL} $@ -i "${PPU_PIP_INDEX}" "${wh_args[@]}"; then
            return 0
        fi
        if [[ ${n} -ge ${max} ]]; then
            echo "[deps] ERROR: pip install failed after ${max} attempts: $*" >&2
            return 1
        fi
        wait_s=$(( n * 30 > 90 ? 90 : n * 30 ))
        wait_s=$(( wait_s + RANDOM % 30 ))
        echo "[deps] retry ${n}/${max} in ${wait_s}s: $*" >&2
        sleep "${wait_s}"
        n=$(( n + 1 ))
    done
}

if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

git config --global --add safe.directory "${REPO_ROOT}"

echo "========== [wheel] install verified ${PPU_RELEASE} wheel =========="
# Tests prepend the release root to sys.path. Install the complete wheel there so
# Python, native extensions, FlashAttention and _version.py all have one source.
${PIP_INSTALL} --upgrade --no-deps --target "${REPO_ROOT}" "${PPU_WHEEL_PATH}"
python3 - "${PPU_EXPECTED_VLLM_VERSION}" "${REPO_ROOT}" <<'PY'
import pathlib
import sys

expected, root = sys.argv[1:]
import vllm
import vllm._C_stable_libtorch as native
import vllm.vllm_flash_attn as flash_attn

assert vllm.__version__ == expected, (vllm.__version__, expected)
for module in (vllm, native, flash_attn):
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(pathlib.Path(root).resolve()), path
    print(f"[wheel] {module.__name__}: {path}")
print(f"[wheel] verified vllm {expected}")
PY

PPU_STACK_BEFORE="$(python3 -c 'import importlib.metadata as m; print(" ".join(f"{p}=={m.version(p)}" for p in ("torch", "triton")))')"
echo "[deps] preserve image ${PPU_STACK_BEFORE}"

echo "========== [deps] resolve ${PPU_RELEASE} requirements =========="
# pip keeps installed packages that satisfy the release's own constraints. No
# package table from another release is applied to this installation.
_pip_retry -r requirements/ppu.txt
_pip_retry pytest pytest-asyncio tblib pytest-shard pyyaml

# requirements/common.txt leaves numpy unpinned and resolves it to 2.x, which this release
# genuinely needs: opencv-python-headless>=4.13.0 ships no NumPy 1.x wheel, so constraining
# numpy to the version the image carries makes the requirements unresolvable. The image does
# ship scipy and scikit-learn built against the NumPy 1.x C ABI, and transformers imports
# sklearn.metrics while loading, so on 2026-10-29 every area died collecting tests with
# "numpy.core.multiarray failed to import" raised inside the image scipy. Move those two
# forward together with numpy rather than holding numpy back.
_pip_retry --upgrade scipy scikit-learn

# numba must come from the aliyun mirror, not the artifactory: the artifactory's
# numba==0.65.0 is a PPU source wrapper (0.65.0+v0.1.0.ppu2.2.1) whose local
# version out-ranks the community 0.65.0 wheel, so it wins any shared
# resolution, and its setup.py exits with "No installation package compatible
# with the current environment" for cp312/linux_x86_64/ubuntu2404/runtime130,
# taking the whole -r requirements install down with it (PR #57 run 42194854556:
# every area died installing deps, 0 tests run). ngram_proposer only uses the
# CPU jit paths, so the community wheel suffices; 0.65.0 needs numpy<2.5 and
# this env ships 2.3.5. requirements/ppu.txt keeps numba commented for this
# reason. Failure is a warning, not fatal: without numba only the misc area's
# test_ngram collection breaks, while a hard failure would kill every area
# before a single test runs.
# Uninstall first: the image's broken numba carries the same release with a local
# version tag (0.65.0+v0.1.0.ppu2.2.1), which satisfies ==0.65.0 per PEP 440, so
# a plain --upgrade would consider it already installed and keep the broken ABI.
python3 -m pip uninstall -y numba >/dev/null 2>&1 || true
n=1
while ! ${PIP_INSTALL} --upgrade "numba == 0.65.0" -i "https://mirrors.aliyun.com/pypi/simple/"; do
    if [[ ${n} -ge 8 ]]; then
        echo "[deps] WARNING: numba install from the aliyun mirror failed after 8 attempts; the image's numpy-1.x-ABI numba stays and misc test_ngram collection will fail" >&2
        break
    fi
    wait_s=$(( n * 30 > 90 ? 90 : n * 30 ))
    wait_s=$(( wait_s + RANDOM % 30 ))
    echo "[deps] retry ${n}/8 in ${wait_s}s: numba == 0.65.0 (aliyun)" >&2
    sleep "${wait_s}"
    n=$(( n + 1 ))
done
# Diagnostic only: numba importability decides whether misc test_ngram can be collected.
python3 -c "import numba; print(f'[verify] numba {numba.__version__}')" 2>/dev/null \
  || echo "[verify] numba NOT importable - misc test_ngram collection will fail" >&2

# torchaudio is not shipped by the image and requirements/ppu.txt keeps it
# commented (torch/torchaudio/torchvision come with the image), so the
# multimodal processing cases whose remote code imports it (MiniCPM-o) fail with
# "No module named 'torchaudio'". Match the image torch release and install with
# --no-deps so the image torch is never replaced (asserted below). Try the
# artifactory first (a PPU build, if published, carries a local version that still
# satisfies ==<release>), then the aliyun mirror. Failure is a warning, not
# fatal: only the cases that import torchaudio break.
TORCH_RELEASE="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
if python3 -c "import torchaudio" 2>/dev/null; then
    echo "[deps] torchaudio already importable, keep it"
elif ${PIP_INSTALL} --no-deps "torchaudio == ${TORCH_RELEASE}" -i "${PPU_PIP_INDEX}" \
    || ${PIP_INSTALL} --no-deps "torchaudio == ${TORCH_RELEASE}" -i "https://mirrors.aliyun.com/pypi/simple/"; then
    echo "[deps] torchaudio == ${TORCH_RELEASE} installed"
else
    echo "[deps] WARNING: torchaudio == ${TORCH_RELEASE} install failed; multimodal cases importing torchaudio will fail" >&2
fi
# Diagnostic only: a torchaudio built for another torch ABI fails here, not in a test.
python3 -c "import torchaudio; print(f'[verify] torchaudio {torchaudio.__version__}')" 2>/dev/null \
  || echo "[verify] torchaudio NOT importable - MiniCPM-o processing cases will fail" >&2

# Diagnostic only: the remaining image packages are not pinned against this numpy either, so
# print the mismatches instead of letting the next one surface as an unrelated import error.
python3 -m pip check || echo "[deps] WARNING: pip check reports unsatisfied requirements" >&2

python3 - "${PPU_STACK_BEFORE}" <<'PY'
from importlib.metadata import version
import sys

for spec in sys.argv[1].split():
    package, expected = spec.split("==")
    assert version(package) == expected, (package, version(package), expected)
import numpy
import scipy.sparse
import sklearn.metrics
import torch
import vllm

print(f"[verify] numpy {numpy.__version__}, scipy and scikit-learn import")

print(f"[verify] torch {torch.__version__}, devices={torch.cuda.device_count()}")
print(f"[verify] vllm {vllm.__version__}")
PY

echo "[deps] done"
