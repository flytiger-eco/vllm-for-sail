#!/bin/bash
# 编译 PPU wheel（提速验证版）：在 build-ppu-wheel.sh 基础上叠加两项优化，
# 供 .github/workflows/build-ppu-wheel-test.yml 单独验证，不动已跑通的生产脚本。
#
# 与 scripts/ppu/build-ppu-wheel.sh 的差异（仅此两处）：
#   1. 编译并发真正生效：把 MAX_JOBS / NVCC_THREADS 经 docker -e 透传进容器，
#      setup.py:219-256 据此决定 num_jobs（生产脚本这两行是注释的 no-op）。
#   2. ccache：容器内装 ccache 并挂载持久 CCACHE_DIR；setup.py:283-295 探测到
#      PATH 里的 ccache 后自动 -DCMAKE_*_COMPILER_LAUNCHER=ccache，重复构建命中
#      缓存跳过编译。CCACHE_DIR 由 workflow 用 actions/cache 跨 run 持久化。
# 其余（版本传递、heredoc、SDK/torch 安装、arch 收窄）与生产脚本逐字一致。
set -ex

# CUDA_VERSION 是 $3 且下面无条件使用，所以至少要三个参数。
if [ $# -lt 3 ]; then
  echo "Usage: $0 <TARGET_VERSION> <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

TARGET_VERSION="$1"
PYTHON_VERSION="$2"          # e.g. 3.12
CUDA_VERSION="$3"            # e.g. 13.0
ARCH="${4:-$(uname -m)}"     # optional override；uname -i 在部分发行版返回 unknown

if [ "${ARCH}" = "aarch64" ]; then
  echo "aarch64 is not supported yet: no aarch64 PPU base image / SDK tarball" >&2
  exit 1
else
  BASE_IMG="pkg.flytiger-eco.com/docker_build/pytorch:ubuntu24.04-py312.06"
fi

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

# 宿主侧 ccache 目录：由 workflow 的 actions/cache 恢复/保存，跨 run 持久。
# 容器内固定挂到 /root/.ccache（root 的默认位置）。缺省给一个本地路径便于本机跑。
HOST_CCACHE_DIR="${HOST_CCACHE_DIR:-$HOME/.cache/ppu-ccache}"
mkdir -p "${HOST_CCACHE_DIR}"

echo "----------------------------------------"
echo "Build configuration (TEST / speedup)"
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

# 引号化 heredoc（<<'INNER'）：不做宿主端展开，变量全靠下面的 -e 传入；-i 必需。
# 新增相对生产脚本：-e MAX_JOBS / -e NVCC_THREADS / -e CCACHE_DIR / -v ccache 挂载。
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
# ccache 就绪度自检：setup.py 靠 `which ccache` 决定是否接线，装不上就直接失败，
# 免得静默退回全量编译还以为缓存生效。
command -v ccache
ccache --version | head -n1
ccache --max-size="${CCACHE_MAXSIZE}"
ccache --zero-stats

# 与 PR #6 成功版本一致，先保留 Rust 环境准备；是否删除作为后续独立优化验证。
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

wget --no-check-certificate -nv https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.1.1/PPU_SDK_cuda-13.0.0-ubuntu2404-2.1.1-a5c56e.tar.gz -O /tmp/ppu.tar.gz
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
# 生产脚本这里 `rm -rf /tmp/*`；CCACHE_DIR 在 /root/.ccache，不受影响。
rm -rf /tmp/*

python3 -m pip install https://pkg.flytiger-eco.com/artifactory/pypi_generic/torch/2.11.0%2Bv0.1.0.ppu2.1.1/torch-2.11.0%2Bcu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl --force-reinstall

export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
export VLLM_REQUIRE_RUST_FRONTEND=0
# 只编 SM80 kernel（沿用 PR #6 的收窄）；PPU 目标硬件之外的消费者会失败，
# 如需扩展硬件支持再放开。
export TORCH_CUDA_ARCH_LIST="8.0"

python3 setup.py bdist_wheel

# 缓存命中率自检：hit/miss 是判断 ccache 是否真的接上编译的唯一硬证据。
echo "==== ccache stats after build ===="
ccache --show-stats
INNER

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true
