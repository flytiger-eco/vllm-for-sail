
<h3 align="center">
vLLM-for-SAIL Inference Engine
</h3>

---
*Latest News* 🔥

- [2026/10] **vLLM-for-SAIL v0.28.0 is now available as a PREVIEW (release candidate `0.28.0+rc1`).** It has **not** been officially released yet, so there is **no dedicated Docker image and no PyPI wheel** for v0.28.0. To try it, build v0.28.0 from source inside the officially released **v0.27.0** Docker image — see [Usage Guide](#usage-guide) below.

- [2026/10] vLLM-for-SAIL v0.27.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2977).

- [2026/08] vLLM-for-SAIL v0.26.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2912).

- [2026/07] vLLM-for-SAIL v0.23.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=129).

- [2026/06] vLLM-for-SAIL v0.20.1 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=128).

---

> ⚠️ **PREVIEW NOTICE**
>
> v0.28.0 is a **preview / release-candidate** branch and has **not** been officially released.
>
> - There is **no** `vllm0.28.0` Docker image and **no** v0.28.0 wheel on PyPI yet.
> - The **only** supported way to use v0.28.0 right now is to **pull the v0.28.0 source code and build it from source inside the already-released v0.27.0 Docker image**, which provides the matching CUDA / PyTorch / SDK toolchain.
> - Interfaces, environment variables and model support on this preview branch may change before the official release. Once v0.28.0 is officially released, a dedicated Docker image, PyPI wheel and User Guide will be published, and building from source will no longer be required.

## Overview

vLLM-for-SAIL is a customized and optimized version of the [vLLM inference engine](https://github.com/vllm-project/vllm), deeply tailored for the Zhenwu AI accelerator chip. This engine builds upon the core scheduling mechanism and high-performance operators of the vLLM community, while deeply integrating PPU-specific high-performance operators (such as efficient GEMM quantization kernels, DeepEP, and DeepGEMM) to maximize the hardware advantages of PPU based on its architectural characteristics.

## Prerequisites

- Hardware: Zhenwu 810 / Zhenwu 810E / Zhenwu M890
- Operating System: Linux (Ubuntu 24.04 recommended)
- Because v0.28.0 has no published artifacts, you must build it inside the released **v0.27.0** Docker image (see below). The build machine needs network access to GitHub and PyPI.

## Usage Guide

Since v0.28.0 has not been officially released, you cannot install it via Docker or PyPI directly. Instead, start from the released **v0.27.0** Docker image (which already contains the PPU SDK, CUDA, PyTorch and build toolchain), then pull the **v0.28.0** source and build a wheel inside that container.

### Step 1. Launch the v0.27.0 Docker image

Pull and start the officially released v0.27.0 image. This container provides the toolchain used to build v0.28.0.

```bash
# Pull the released v0.27.0 image
docker pull pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-ubuntu24.04-cuda13.0-vllm0.27.0-py312-20261001

# Start a container (mount PPU devices and a workspace as needed)
docker run -it --rm \
  --name vllm-sail-v0280-build \
  -v /path/to/your/workspace:/workspace \
  pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-ubuntu24.04-cuda13.0-vllm0.27.0-py312-20261001 \
  /bin/bash
```

> Note: the image ships vLLM-for-SAIL **0.27.0**. The following steps build and install **0.28.0** on top of it, replacing the pre-installed 0.27.0 wheel.

### Step 2. Build v0.28.0 from source (inside the container)

Run all commands below **inside** the v0.27.0 container started in Step 1. The build steps follow the [vLLM-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2977); the only difference is that you check out the `v0.28.0` branch.

```bash
# 1. Download the vLLM-for-SAIL v0.28.0 source
git clone -b v0.28.0 https://github.com/flytiger-eco/vllm-for-sail

# 2. Build vLLM (the machine needs network access to GitHub)

# 2.1 Set environment variables
export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
export VLLM_REQUIRE_RUST_FRONTEND=0

# 2.2 Install build dependencies
cd vllm-for-sail
pip install -r requirements/build/ppu.txt
pip install -r requirements/ppu.txt
pip install numpy==1.26.0

# 2.3 Build
python setup.py bdist_wheel

# 3. Install the freshly built v0.28.0 wheel (overrides the 0.27.0 in the image)
pip install --force-reinstall --no-deps dist/vllm*.whl
```

### Step 3. Environment Verification

Run the following command inside the container where you just installed vLLM-for-SAIL:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If no errors occur during execution and `0.28.0` is printed in the console, it indicates that the vLLM-for-SAIL v0.28.0 preview has been built and installed successfully.

## Quick Start

Take Qwen3.5-35B-A3B as an example to launch the inference service and send a request:

```bash
# Terminal 1: Start the server
MODEL=<your Qwen3.5-35B-A3B path>
vllm serve ${MODEL} --trust-remote-code --host 0.0.0.0 --port 8999 --dtype bfloat16 --tensor-parallel-size 2
```

```bash
# Terminal 2: Send a request
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

curl -X POST http://0.0.0.0:8999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Briefly introduce what speculative sampling is"}
    ],
    "max_completion_tokens": 300,
    "top_k": 1
  }'
```

## More Information

For the Docker / PyPI installation flow and general usage of the latest **released** version, please refer to the [vLLM-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2977). A dedicated v0.28.0 User Guide, Docker image and PyPI wheel will be published once v0.28.0 is officially released.

## License
See [NOTICE](./NOTICE) and [LICENSE](./LICENSE).
