#!/bin/bash
# Build the PPU wheel: install the SDK + PPU torch inside the PPU build container, then run
# setup.py bdist_wheel.
#
# Ported from .github/workflows/scripts/build_ppu.sh of PR #6 and carried by this branch
# (the branch ships its own CI files, so PR #6's "keep the single copy of the script on the
# default branch" double checkout is not needed).
# Every difference from PR #6 and its reason is in scripts/ppu/docs/ppu-wheel-build.md.
set -ex

# CUDA_VERSION is $3 and is used unconditionally below, so at least three arguments are required.
if [ $# -lt 3 ]; then
  echo "Usage: $0 <TARGET_VERSION> <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

TARGET_VERSION="$1"
PYTHON_VERSION="$2"          # e.g. 3.12
CUDA_VERSION="$3"            # e.g. 13.0
ARCH="${4:-$(uname -m)}"     # optional override; uname -i returns unknown on some distributions

if [ "${ARCH}" = "aarch64" ]; then
  echo "aarch64 is not supported yet: no aarch64 PPU base image / SDK tarball" >&2
  exit 1
else
  # CI passes the image ID already used in the Rust cache key; a direct local call keeps
  # the default image.
  BASE_IMG="${PPU_BUILD_IMAGE:-pkg.flytiger-eco.com/docker_build/pytorch:ubuntu24.04-py312.06}"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

# Host-side ccache directory: restored/saved by the workflow's actions/cache and persistent
# across runs. Inside the container it is always mounted at /root/.ccache (the root's default
# location). The default is a local path so the script also runs on a workstation.
HOST_CCACHE_DIR="${HOST_CCACHE_DIR:-$HOME/.cache/ppu-ccache}"
mkdir -p "${HOST_CCACHE_DIR}"

echo "----------------------------------------"
echo "Build configuration"
echo "TARGET_VERSION:  ${TARGET_VERSION}"
echo "PYTHON_VERSION:  ${PYTHON_VERSION}"
echo "CUDA_VERSION:    ${CUDA_VERSION}"
echo "ARCH:            ${ARCH}"
echo "BASE_IMG:        ${BASE_IMG}"
echo "PYTHON_TAG:      ${PY_TAG}"
echo "MAX_JOBS:        ${MAX_JOBS:-<unset→cpu_count>}"
echo "NVCC_THREADS:    ${NVCC_THREADS:-<unset>}"
echo "HOST_CCACHE_DIR: ${HOST_CCACHE_DIR}"
echo "Output:          ${DIST_DIR}/"
echo "----------------------------------------"

# Use a quoted heredoc (<<'INNER') instead of bash -c '...': the in-container script itself
# contains single quotes (printf '%s\n%s'), which would close the string early and let the
# host shell eat the escapes - PR #6 measured printf '%s\n%s' arriving as printf %sn%s, so
# the version comparison always took the else branch. A quoted heredoc performs no
# host-side expansion at all, every variable arrives through the -e flags below.
# -i is required: the heredoc is fed into the container over stdin.
# MAX_JOBS/NVCC_THREADS go through docker -e and setup.py derives the compile parallelism
# from them; CCACHE_DIR plus the ccache mount make setup.py find a compiler launcher so
# repeat builds hit the cache and skip compilation.
docker run --rm -i \
  --network=host \
  -v "$(pwd):/workspace" \
  -v "${HOST_CCACHE_DIR}:/root/.ccache" \
  -w /workspace \
  -e ARCH="${ARCH}" \
  -e TARGET_VERSION="${TARGET_VERSION}" \
  -e PYTHON_VERSION="${PYTHON_VERSION}" \
  -e CUDA_VERSION="${CUDA_VERSION}" \
  -e VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-}" \
  -e MAX_JOBS="${MAX_JOBS:-}" \
  -e NVCC_THREADS="${NVCC_THREADS:-}" \
  -e CCACHE_DIR="/root/.ccache" \
  -e CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-5G}" \
  "${BASE_IMG}" \
  bash -s <<'INNER'
set -ex

apt update
apt install -y protobuf-compiler ccache
# ccache readiness probe: setup.py decides whether to wire it up based on `which ccache`,
# so fail hard instead of silently falling back to a full compile that looks cached.
command -v ccache
ccache --version | head -n1
ccache --max-size="${CCACHE_MAXSIZE}"
ccache --zero-stats
# Kept in sync with the successful PR #6 version; dropping it is a separate follow-up experiment.
if [ ! -d "$HOME/.cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustc --version
export VLLM_REQUIRE_RUST_FRONTEND=1

export PPU_SDK=/usr/local/PPU_SDK
export PPU_PATH=${PPU_SDK}
export PPU_HOME=${PPU_PATH}
export CUDA_SDK=${PPU_SDK}/CUDA_SDK
export CUDA_TOOLKIT_ROOT=${CUDA_SDK}
export CUDA_PATH=${CUDA_SDK}
export CUDA_HOME=${CUDA_SDK}
export CUDNN_HOME=${CUDA_SDK}
export CUDACXX=${CUDA_SDK}/bin/nvcc
export PATH=${CUDA_SDK}/bin:${PPU_SDK}/bin:${PPU_SDK}/asight/bin:${PPU_SDK}/ppu-smi/bin:${PATH}
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH=${CUDA_SDK}/lib64:${PPU_SDK}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_SDK}/lib64:${PPU_SDK}/lib:${LIBRARY_PATH}

wget -nv https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.2.0/PPU_SDK_cuda-13.0.0-ubuntu2404-2.2.0-663c22.tar.gz -O /tmp/ppu.tar.gz
printf '%s  %s\n' \
  '7c9e85f62104058a8d8c2f5c3fa1c5242f81b66caae3f566974d0e2bbe44ff47' \
  '/tmp/ppu.tar.gz' | sha256sum -c -
mkdir -p /tmp/ppu
tar --extract --file="/tmp/ppu.tar.gz" --directory=/tmp/ppu
mv /tmp/ppu/PPU_SDK /usr/local/
ln -s /usr/local/PPU_SDK/CUDA_SDK /usr/local/cuda-13.0
ln -s /usr/local/cuda-13.0 /usr/local/cuda
echo /usr/local/PPU_SDK/CUDA_SDK/lib >> /etc/ld.so.conf.d/ppu.conf
echo /usr/local/PPU_SDK/CUDA_SDK/lib64 >> /etc/ld.so.conf.d/ppu.conf
ldconfig
ldconfig -p | grep -q libcuda.so
ldconfig -p | grep -q /usr/local/PPU_SDK/CUDA_SDK/lib
source /usr/local/PPU_SDK/envsetup.sh
clang --version
nvcc --version
asys --version
ppu-smi --version
# CCACHE_DIR lives in /root/.ccache and is unaffected by this cleanup
rm -rf /tmp/*

python3 -m pip install https://pkg.flytiger-eco.com/artifactory/pypi_generic/torch/2.13.0%2Bv0.1.0.ppu2.2.0/torch-2.13.0%2Bcu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl --force-reinstall
PPU_TORCH="$(python3 -c 'from importlib.metadata import version; print(version("torch"))')"

# The PPU torch wheel links libnccl.so.2, which neither the build image nor the PPU SDK
# tarball ships: it used to arrive as a side effect of the upstream torch that the release
# requirements pulled in. Depend on it explicitly, pinned to the version the last green
# build resolved, and put its directory on the loader path.
python3 -m pip install nvidia-nccl-cu13==2.30.7
NCCL_LIB_DIR="$(python3 - <<'PY'
import glob
import site
import sys

hits = sorted(glob.glob(f"{site.getsitepackages()[0]}/nvidia/**/libnccl.so.2", recursive=True))
if not hits:
    sys.exit("libnccl.so.2 is not in the installed nvidia wheels")
print(hits[0].rsplit("/", 1)[0])
PY
)"
export LD_LIBRARY_PATH=${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}

# The requirements come after the PPU wheel, not before it. compressed-tensors asks for
# torch>=2.10.0 with no upper bound, so installing them first pulls the current upstream
# torch plus its whole CUDA wheel stack; the PPU wheel then replaces only torch and has to
# live with whatever CUDA minor version pip picked that day. On 2026-10-29 that drift
# (torch 2.14.1 -> 2.15.0, cuda-runtime 13.0 -> 13.2) left libnccl.so.2 unresolvable and
# every build died in setup.py. Installing the PPU wheel first makes torch>=2.10.0 already
# satisfied, so pip never fetches an upstream torch.
echo "Install the PPU requirements declared by this release..."
pip install -r requirements/build/ppu.txt
pip install -r requirements/ppu.txt
pip install numpy==1.26.0

# Fail here, with the reason visible, rather than inside setup.py's import torch.
python3 - "${PPU_TORCH}" <<'PY'
from importlib.metadata import version
import sys

expected = sys.argv[1]
assert version("torch") == expected, (version("torch"), expected)
import torch

print(f"[build] torch {torch.__version__} imports after the requirements install")
PY

export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
export VLLM_REQUIRE_RUST_FRONTEND=0
# Build SM80 kernels only (the same narrowing PR #6 used); consumers on other PPU targets
# will fail, widen it when more hardware support is needed.
export TORCH_CUDA_ARCH_LIST="8.0"

python3 setup.py bdist_wheel

# Cache hit-rate probe: hit/miss is the only hard evidence that ccache is really wired in.
echo "==== ccache stats after build ===="
ccache --show-stats
INNER

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true
