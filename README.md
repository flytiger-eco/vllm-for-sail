
<h3 align="center">
vLLM-for-SAIL Inference Engine
</h3>

---
*Latest News* 🔥

- [2026/07] vLLM-for-SAIL v0.23.0 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=129).

- [2026/06] vLLM-for-SAIL v0.20.1 has been officially released! You can start using vLLM-for-SAIL via the [User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=128).

---

## Overview

vLLM-for-SAIL is a customized and optimized version of the [vLLM inference engine](https://github.com/vllm-project/vllm), deeply tailored for T-Head's self-developed AI accelerator chip (PPU). This engine builds upon the core scheduling mechanism and high-performance operators of the vLLM community, while deeply integrating PPU-specific high-performance operators (such as efficient GEMM quantization kernels, DeepEP, and DeepGEMM) to maximize the hardware advantages of PPU based on its architectural characteristics.

## Prerequisites

- Hardware: Zhenwu 810 / Zhenwu 810E / Zhenwu M890
- Operating System: Linux (Ubuntu 24.04 recommended)

## Usage Guide

### Build from Source

If you need to build and install from source, please make sure to build inside the [vLLM-for-SAIL Docker image](https://www.flytiger-eco.com/download?businessType=DOCKER).

```bash
# 1. Download vLLM 0.23.0
git clone -b v0.23.0 https://github.com/flytiger-eco/vllm-for-sail

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

# 3. Install vLLM
pip install dist/vllm*.whl
```

To use vLLM-for-SAIL directly via Docker or install it from PyPI, please refer to the [vLLM-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=129).

### Environment Verification

Run the following command in an environment where vLLM-for-SAIL is installed:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If no errors occur during execution and `0.23.0` is printed in the console, it indicates that vLLM-for-SAIL has been installed successfully.

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
Please refer to the [vLLM-for-SAIL User Guide](https://www.flytiger-eco.com/docs_center/doc_detail/index.html?projectId=6&documentId=129) for more information.


## License
See [NOTICE](./NOTICE) and [LICENSE](./LICENSE).
