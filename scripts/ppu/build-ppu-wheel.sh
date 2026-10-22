#!/bin/bash
# Build the PPU wheel: install the SDK and PPU torch inside the PPU build
# container, then run setup.py bdist_wheel.
#
# Called by .github/workflows/ppu-build-pr.yml on main.
# This script and the source to build come from the PR's exact head SHA.
set -ex

# CUDA_VERSION is $3 and is used unconditionally below, so require at least
# three arguments.
if [ $# -lt 3 ]; then
  echo "Usage: $0 <TARGET_VERSION> <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

TARGET_VERSION="$1"
PYTHON_VERSION="$2"          # e.g. 3.12
CUDA_VERSION="$3"            # e.g. 13.0
ARCH="${4:-$(uname -m)}"     # optional override; uname -i returns unknown on some distros

if [ "${ARCH}" = "aarch64" ]; then
  echo "aarch64 is not supported yet: no aarch64 PPU base image / SDK tarball" >&2
  exit 1
else
  # CI passes the image ID already used in the Rust cache key; direct local
  # calls still use the default image.
  BASE_IMG="${PPU_BUILD_IMAGE:-pkg.flytiger-eco.com/docker_build/pytorch:ubuntu24.04-py312.06}"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

# Host-side ccache directory: restored/saved by the workflow's actions/cache and
# persisted across runs. It is always mounted at /root/.ccache in the container
# (root's default location). The default is a local path for running locally.
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

# Use a quoted heredoc (<<'INNER') instead of bash -c '...': the in-container
# script contains single quotes (printf '%s\n%s'), which would close a
# single-quoted string early and let the host shell eat the escapes. In PR #6,
# printf '%s\n%s' reached the container as printf %sn%s, so the version
# comparison always took the else branch.
# A quoted heredoc performs no host-side expansion; all variables come from the
# -e flags below.
# -i is required: the heredoc is fed to the container through stdin.
# MAX_JOBS/NVCC_THREADS are passed through docker -e, and setup.py uses them to
# decide build parallelism (container cpu_count when unset); CCACHE_DIR plus
# the ccache mount let setup.py detect the compiler launcher, so repeated
# builds hit the cache and skip compilation.
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
# ccache readiness check: setup.py relies on `which ccache` to wire it in, so
# fail right away if it cannot be installed instead of silently falling back to
# a full build while assuming the cache works.
command -v ccache
ccache --version | head -n1
ccache --max-size="${CCACHE_MAXSIZE}"
ccache --zero-stats
# Keep the Rust environment setup to match PR #6's working version; removing it
# is a separate optimization to verify later.
if [ ! -d "$HOME/.cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustc --version
export VLLM_REQUIRE_RUST_FRONTEND=1

if [ "$(printf '%s\n%s' "$TARGET_VERSION" "0.20.1" | sort -V | head -n1)" = "0.20.1" ]; then
    echo "Target version ($TARGET_VERSION) is >= 0.20.1. Install ppu requirements..."
    pip install -r requirements/build/ppu.txt
    pip install -r requirements/ppu.txt
else
    echo "Target version ($TARGET_VERSION) is < 0.20.1. Install cuda requirements..."
    pip install -r requirements/build/cuda.txt
    pip install -r requirements/cuda.txt
fi
pip install numpy==1.26.0

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

PPU_SDK_SHA256="7c9e85f62104058a8d8c2f5c3fa1c5242f81b66caae3f566974d0e2bbe44ff47"
wget -nv https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.2.0/PPU_SDK_cuda-13.0.0-ubuntu2404-2.2.0-663c22.tar.gz -O /tmp/ppu.tar.gz
echo "${PPU_SDK_SHA256}  /tmp/ppu.tar.gz" | sha256sum -c -
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
# CCACHE_DIR is /root/.ccache and is not affected by this cleanup.
rm -rf /tmp/*

python3 -m pip install https://pkg.flytiger-eco.com/artifactory/pypi_generic/torch/2.13.0%2Bv0.1.0.ppu2.2.0/torch-2.13.0%2Bcu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl --force-reinstall

export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
export VLLM_REQUIRE_RUST_FRONTEND=0
# Only build SM80 kernels (the narrowing from PR #6); consumers outside the PPU
# target hardware will fail. Widen this when broader hardware support is needed.
export TORCH_CUDA_ARCH_LIST="8.0"

python3 setup.py bdist_wheel

# Cache hit-rate check: hit/miss counts are the only hard evidence that ccache
# is actually wired into the build.
echo "==== ccache stats after build ===="
ccache --show-stats
INNER

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true
