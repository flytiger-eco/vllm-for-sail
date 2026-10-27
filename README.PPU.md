<!-- markdownlint-disable MD041 -->
<h3 align="center">
vLLM-for-SAIL Inference Engine
</h3>

---
*Latest News* 🔥

- [2026/10] **vLLM-for-SAIL v0.28.0 has been officially released!** You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=11669).

- [2026/10] vLLM-for-SAIL v0.27.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2977).

- [2026/08] vLLM-for-SAIL v0.26.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=2912).

- [2026/07] vLLM-for-SAIL v0.23.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=129).

- [2026/06] vLLM-for-SAIL v0.20.1 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=128).

---

> This is the PPU README of vLLM-for-SAIL. The original vLLM community README is kept in [README.md](https://github.com/flytiger-eco/vllm-for-sail/blob/0.28.0+v0.1.0/README.md).

## Overview

vLLM-for-SAIL is a customized and optimized version of the [vLLM inference engine](https://github.com/vllm-project/vllm), deeply tailored for the Zhenwu AI accelerator chip. This engine builds upon the core scheduling mechanism and high-performance operators of the vLLM community, while deeply integrating PPU-specific high-performance operators (such as efficient GEMM quantization kernels, DeepEP, and DeepGEMM) to maximize the hardware advantages of PPU based on its architectural characteristics.

## Prerequisites

- Hardware: Zhenwu 810 / Zhenwu 810E / Zhenwu M890
- Operating System: Linux (Ubuntu 24.04 recommended)
- SAIL SDK v2.2.0 with PyTorch 2.13.0, CUDA 13.0 and Python 3.12
- Network access to `pkg.flytiger-eco.com` for the PPU PyPI index and Docker images

## Installation

The officially released v0.28.0 wheel, the matching PPU kernel packages and their full installation steps are described in the [vLLM-for-SAIL v0.28.0 User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=11669).

v0.28.0 does not ship a dedicated Docker image: start from the officially released v0.27.0 PPU image and upgrade vLLM inside it, or install the wheel from the PPU PyPI index.

### Upgrade Inside the v0.27.0 Docker Image (Recommended)

Pull and start the v0.27.0 image:

```bash
docker pull pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-ubuntu24.04-cuda13.0-vllm0.27.0-py312-20261001

docker run -it --rm --privileged --network host --shm-size=16g \
  -v /path/to/your/workspace:/workspace \
  pkg.flytiger-eco.com/docker_release/llm:sdk2.2.0-pytorch2.13.0-ubuntu24.04-cuda13.0-vllm0.27.0-py312-20261001
```

> Note: this image ships vLLM-for-SAIL **0.27.0**. The steps below replace the pre-installed 0.27.0 wheel with **0.28.0**.

Run the command below **inside** the container:

```bash
pip install vllm==0.28.0 \
  -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple \
  --force-reinstall --no-deps
```

> Do not upgrade PyTorch. The image keeps `torch==2.13.0`, and any other version breaks the PPU kernel ABI.

### Install the Wheel from the PPU PyPI Index

```bash
pip install vllm==0.28.0 -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
```

Then refresh the PPU kernel packages (`flashinfer_python`, `deep_gemm`, `deep_ep`, `flash_mla`, `flash_attn_3`, `pla`, `mooncake`, ...) to the versions listed in the User Guide.

### Build from Source in the v0.27.0 Docker Image

The build needs the matching SAIL SDK v2.2.0, CUDA 13.0 and PyTorch 2.13.0, so build inside the v0.27.0 image. Start the image if you have not done so yet, then run all commands below **inside** the container:

```bash
# 1. Download the vLLM-for-SAIL v0.28.0 source
git clone -b 0.28.0+v0.1.0 https://github.com/flytiger-eco/vllm-for-sail

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

### Environment Verification

Run the following command inside the container where you just installed vLLM-for-SAIL:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If no errors occur during execution and `0.28.0` is printed in the console, vLLM-for-SAIL v0.28.0 has been installed successfully.

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

## License

See [NOTICE](https://github.com/flytiger-eco/vllm-for-sail/blob/0.28.0+v0.1.0/NOTICE) and [LICENSE](https://github.com/flytiger-eco/vllm-for-sail/blob/0.28.0+v0.1.0/LICENSE).
