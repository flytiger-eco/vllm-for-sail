
<h3 align="center">
vLLM-for-SAIL Inference Engine
</h3>


<p align="center">
<b>English</b> | <a href="./README_zh.md"><b>中文</b></a>
</p>

---
*Latest News* 🔥

- [2026/08] vLLM-for-SAIL v0.25.0 has been officially released! You can start using vLLM-for-SAIL via the User Guide.

- [2026/07] vLLM-for-SAIL v0.23.0 has been officially released! You can start using vLLM-for-SAIL via the User Guide.

- [2026/06] vLLM-for-SAIL v0.20.1 has been officially released! You can start using vLLM-for-SAIL via the User Guide.

---

## Overview

vLLM-for-SAIL is a customized and optimized version of the vLLM inference engine, deeply tailored for T-Head's self-developed AI accelerator chip (PPU). This engine builds upon the core scheduling mechanism and high-performance operators of the vLLM community, while deeply integrating PPU-specific high-performance operators (such as efficient GEMM quantization kernels, DeepEP, and DeepGEMM) to maximize the hardware advantages of PPU based on its architectural characteristics.

## Prerequisites

- Hardware: Zhenwu 810 / Zhenwu 810E / Zhenwu M890
- Operating System: Linux (Ubuntu 24.04 recommended)

## Usage Guide

Please refer to the vLLM-for-SAIL User Guide

### Environment Verification

Run the following command in an environment where vLLM-for-SAIL is installed:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If no errors occur during execution and `0.20.1` is printed in the console, it indicates that vLLM-for-SAIL has been installed successfully.

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
Please refer to the vLLM-for-SAIL User Guide for more information.


## License
See [NOTICE](./NOTICE) and [LICENSE](./LICENSE).
