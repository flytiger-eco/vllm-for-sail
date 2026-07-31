
<h3 align="center">
vLLM-for-SAIL 推理引擎
</h3>


<p align="center">
<a href="./README.md"><b>English</b></a> | <b>中文</b>
</p>

---
*最新消息* 🔥

- [2026/08] vLLM-for-SAIL v0.25.0 正式发布！可通过 用户指南 来使用vLLM-for-SAIL。

- [2026/07] vLLM-for-SAIL v0.23.0 正式发布！可通过 用户指南 来使用vLLM-for-SAIL。

- [2026/06] vLLM-for-SAIL v0.20.1 正式发布！可通过 用户指南 来使用vLLM-for-SAIL。

---

## 简介

vLLM-for-SAIL 是为平头哥自研 AI 加速芯片（PPU）深度定制和优化的 vLLM 推理引擎版本。本引擎在 vLLM 社区核心的调度机制与高性能算子的基础上，针对 PPU 的硬件架构特点，深度集成了 PPU 特有的高性能算子（如高效的 GEMM 量化 kernel、DeepEP 和 DeepGEMM 等），以最大化发挥 PPU 的硬件优势。

## 准备

- 硬件：真武 810 / 真武 810E / 真武 M890
- 操作系统：Linux（推荐 Ubuntu 24.04）

## 使用指南

请参考 vLLM-for-SAIL 用户指南

### 环境验证

在安装有 vLLM-for-SAIL 的环境中执行以下命令：

```bash
python -c "import vllm; print(vllm.__version__)"
```

如果运行过程中没有出现任何报错，并且最终能在控制台中看到输出的 `0.20.1` 字样，则代表 vLLM-for-SAIL 安装成功。

## 快速开始

以 Qwen3.5-35B-A3B 为例，启动推理服务并发送请求：

```bash
# 终端 1: 启动 server
MODEL=<your Qwen3.5-35B-A3B path>
vllm serve ${MODEL} --trust-remote-code --host 0.0.0.0 --port 8999 --dtype bfloat16 --tensor-parallel-size 2
```

```bash
# 终端 2: 发送请求
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

curl -X POST http://0.0.0.0:8999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "简单介绍一下什么是投机采样"}
    ],
    "max_completion_tokens": 300,
    "top_k": 1
  }'
```
## 更多信息
请阅读 vLLM-for-SAIL 用户指南 获取更多信息


## 许可证
详见 [NOTICE](./NOTICE) 与 [LICENSE](./LICENSE)