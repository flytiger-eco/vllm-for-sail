#!/bin/bash
# 编译 PPU wheel：在 PPU 构建容器内装 SDK + PPU torch 后执行 setup.py bdist_wheel。
#
# 移植自 PR #6 的 .github/workflows/scripts/build_ppu.sh，随本分支携带
# （本分支自带 CI 文件，不需要 PR #6 "脚本单点维护在默认分支" 的双 checkout）。
# 与 PR #6 的全部差异及原因见 scripts/ppu/docs/ppu-wheel-build.md。
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

# 并发参数默认值集中在这里兜底；workflow 侧可通过 env 覆盖。
# PR #6 的 docker -e 透传行被注释掉了，workflow 里设的 MAX_JOBS/NVCC_THREADS
# 实际进不了容器，本脚本修正为完整四层传递（workflow env → 脚本 → -e → setup.py）。
MAX_JOBS="${MAX_JOBS:-64}"
NVCC_THREADS="${NVCC_THREADS:-8}"

echo "----------------------------------------"
echo "Build configuration"
echo "TARGET_VERSION: ${TARGET_VERSION}"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "MAX_JOBS:       ${MAX_JOBS}"
echo "NVCC_THREADS:   ${NVCC_THREADS}"
echo "Output:         ${DIST_DIR}/"
echo "----------------------------------------"

# 用带引号的 heredoc（<<'INNER'）而不是 bash -c '...'：容器内脚本自身含单引号
# （printf '%s\n%s'），嵌进单引号串会提前闭合，宿主 shell 会吃掉转义——PR #6
# 实测 printf '%s\n%s' 传到容器里变成 printf %sn%s，版本比较恒走 else 分支。
# 引号化 heredoc 不做任何宿主端展开，变量全靠下面的 -e 传入。
# -i 是必需的：heredoc 走 stdin 喂进容器。
docker run --rm -i \
  --network=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e ARCH="${ARCH}" \
  -e TARGET_VERSION="${TARGET_VERSION}" \
  -e PYTHON_VERSION="${PYTHON_VERSION}" \
  -e CUDA_VERSION="${CUDA_VERSION}" \
  -e VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-}" \
  -e MAX_JOBS="${MAX_JOBS}" \
  -e NVCC_THREADS="${NVCC_THREADS}" \
  "${BASE_IMG}" \
  bash -s <<'INNER'
set -ex

apt update
apt install -y protobuf-compiler

# 不装 rustup：PR #6 里 rustup 安装与结尾的 VLLM_REQUIRE_RUST_FRONTEND=0 并存，
# 前者是纯开销（PR 作者已在风险清单中标注待删）。本脚本确定不强制 Rust frontend，
# 只保留 =0 这一处设置。
export VLLM_REQUIRE_RUST_FRONTEND=0

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
rm -rf /tmp/*

python3 -m pip install https://pkg.flytiger-eco.com/artifactory/pypi_generic/torch/2.11.0%2Bv0.1.0.ppu2.1.1/torch-2.11.0%2Bcu130ubuntu2404oe-cp312-cp312-linux_x86_64.whl --force-reinstall

export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
# 只编 SM80 kernel（沿用 PR #6 的收窄）；PPU 目标硬件之外的消费者会失败，
# 如需扩展硬件支持再放开。
export TORCH_CUDA_ARCH_LIST="8.0"

python3 setup.py bdist_wheel
INNER

echo "Done. Wheels are in ${DIST_DIR}/"
ls -lh "${DIST_DIR}"/*.whl 2>/dev/null || true
