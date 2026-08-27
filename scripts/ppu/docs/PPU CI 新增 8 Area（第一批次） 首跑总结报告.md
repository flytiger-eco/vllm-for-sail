# PPU CI 新增 8 Area（第一批次） 首跑总结报告

> 分支 `feat/gha-ppu-test`，基线 v0.23.0，触发提交 `5c6d6125d`（"test /ppu path"，即 "Step 1: add 8 important area" 后的首次全量触发），2026-08-26 18:23（CST）push 触发， 8 个 area 均为 run\_number=1 首跑。执行环境：self-hosted PPU runner（单机单 runner 串行）， 容器镜像 `llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-vllm0.23.0-py312`， 离线模式（HF\_HUB\_OFFLINE=1，模型走 /nas\_aisw 预置卷）。

## 总览

| Area | Run | 结论 | 测试 job 耗时 | 首跑 red 规模 |
| --- | --- | --- | --- | --- |
| Models Basic | [32958040040](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958040040) | failure | 48m07s | 6 failed / 362 passed / 24 skipped |
| Attention | [32958040006](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958040006) | failure | 1m23s | 1 failed |
| Model Executor | [32958039974](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039974) | **success** | 1m49s | 无（首跑通过） |
| Kernels | [32958039955](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039955) | failure | 2h14m18s | 3 个测试文件 fail |
| Engine | [32958039950](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039950) | failure | 1h32m08s | 17 failed / 100 passed / 2 skipped（v1\_engine step） |
| Entrypoints LLM | [32958039949](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039949) | failure | 11m34s | 3 failed（同一根因） |
| Entrypoints | [32958039936](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039936) | failure | 1m14s | 2 个 step 级 error（无用例级 fail） |
| Samplers | [32958039926](https://github.com/flytiger-eco/vllm-for-sail/actions/runs/32958039926) | **success** | 9m05s | 无（首跑通过） |

**首跑通过通告：Model Executor、Samplers 两个 area 首跑全绿，无需 triage。** 其余 6 个 area 的首跑 red 用例已全部完成 triage（deselect / ignore / stub / step 停用 + 注释标注），对应修复提交：087c696a8（entrypoints）、d29b68a79 + 工作区改动（models-basic）、 1a73ee2eb（attention）、00452ae3c（kernels）、4122a6a65（engine）、工作区改动（entrypoints-llm）。

## 批次 1：核心链路与用户入口（P1，8 个 area）

目标：调度 → 执行 → 注意力 → 采样 → 入口 → 模型冒烟全链路覆盖，形成 PR 门禁最小闭环。

| 状态 | Area | 规模（组/文件） | 单/多卡 | 说明 |
| --- | --- | --- | --- | --- |
| ✅ | engine | 3 / 11 | 单卡 | 引擎核心：调度器、KV cache、异步 LLM |
| ✅ | attention | 1 / — | 单卡 | 注意力后端，PPU 核心适配点 |
| ✅ | model-executor | 1 / 8 | 单卡 | 模型执行器，连接引擎与模型层 |
| ✅ | samplers | 1 / 3 | 单卡 | 采样正确性，直接决定输出质量 |
| ✅ | entrypoints | 2 / 2 | 单卡 | OpenAI API 服务入口 |
| ✅ | entrypoints-llm | 4 / 3 | 单+多卡 | offline LLM 类接口 |
| ✅ | models-basic | 1 / 7 | 单卡 | 核心模型冒烟底线 |
| ✅ | kernels | 6 / 51 | 单卡 | 算子层最大用例集，可单独排期 |
| ✅ | basic-correctness | 4 / 3 | 单+多卡 | 冒烟底线 |
| ✅ | lora | 6 / 13 | 单+多卡 | LoRA 适配器 |

## 缺少的模型与对应用例

### A. 直接导致首跑 fail 的缺失模型

| 模型 | 影响的用例 | 所属 area | 状态 |
| --- | --- | --- | --- |
| BAAI/bge-base-en-v1.5 | tests/models/test\_transformers.py::test\_pooling\[TransformersEmbeddingModel\] | Models Basic | 用例已 deselect，待入库恢复 |
| papluca/xlm-roberta-base-language-detection | tests/models/test\_transformers.py::test\_pooling\[TransformersForSequenceClassification\] | Models Basic | 用例已 deselect，待入库恢复 |
| meta-llama/Meta-Llama-3-8B（HF gated） | tests/v1/attention/test\_indexer\_deepseek\_v4\_slot\_mapping.py::test\_indexer\_builder\_deepseek\_v4\_compressed\_slot\_mapping\_uses\_storage\_block\_size | Attention | 不在 NAS 缓存；已用 VLLM\_MODEL\_REDIRECT\_PATH config stub 解决（仅需 config 解析），权重本体仍缺 |
| hmellor/Ilama-3.2-1B | tests/models/test\_transformers.py::test\_models\[hmellor/Ilama-3.2-1B-auto\] | Models Basic | 红区清单 MISS（走 Aone 同构路径 symlink）；对拍失败，模型本体是否就位待查 |

### B. 已预知缺失、首跑前已通过 ignore / defer 排除的模型（待入库后放开）

| 模型 | 影响的用例/文件 | 所属 area | 排除原因 |
| --- | --- | --- | --- |
| JackFram/llama-160m | tests/v1/e2e/general/test\_context\_length.py（整文件 ignore） | Engine | GHA 清单与 Aone aliases 双 MISS，待入库 |
| google/gemma-3n-E2B-it | tests/v1/e2e/general/test\_kv\_sharing\_fast\_prefill.py | Engine | 模型较大，首版 defer（Aone yaml 原注释） |
| Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | tests/v1/e2e/general/test\_mamba\_prefix\_cache.py | Engine | PPU 单卡跑不了 80B |
| Qwen/Qwen3-Embedding-0.6B | tests/v1/e2e/general/test\_pooling\_chunked\_prefill.py | Engine | embedding model 首版 defer |
| Meta-Llama-3-8B / Phi-tiny-MoE / embeddinggemma-300m / DeepSeek-V3/R1 等 | tests/v1/attention/ 全部 \*\_correctness 用例族（-k "not \_correctness" 排除） | Attention | 模型未 stage 到红区，离线必挂；stage 后删除 -k 并补 MODEL\_MAP |
| Qwen2/2.5/3-VL、GLM-4.1V 等 6 个多模态模型 | tests/kernels/core/test\_mrope.py（整文件 ignore） | Kernels | 未入库（Aone 快照既有排除），待入库 |
| microsoft/Phi-3.5-vision-instruct、Qwen/Qwen3-0.6B | tests/entrypoints/llm/test\_chat.py::test\_chat\_multi\_image / test\_chat\_extra\_kwargs（-k 排除） | Entrypoints LLM | vision/thinking fixture 仅服务被排除用例，不实例化、无需 symlink |
| BAAI/bge-base-en-v1.5、facebook/opt-125m、GPT-2 等 | test\_model\_load\_with\_params.py / test\_weight\_utils.py / test\_qwen3\_omni.py / test\_qwen3\_vl\_mrope.py / model\_loader 各 loader 目录（ignore）；test\_ep\_weight\_filter 的 GPT-2 用例（-k 排除） | Model Executor | HF 下载类用例首版排除（offline 红区） |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tests/samplers/test\_beam\_search.py（整文件 ignore） | Samplers | 模型已入清单，但 beam search 在 PPU 的正确性未验证，首跑稳定后再放开 |
| distilbert/distilgpt2 | tests/samplers/test\_no\_bad\_words.py（整文件 ignore） | Samplers | NAS 路径为 Aone 同构推测、未经清单证实，保守保持 ignore |

---

## 1. Models Basic（failure，48m07s）

首跑实测：**6 failed / 362 passed / 24 skipped**。6 个 red 用例已逐个 deselect 并注释 （恢复路径见脚本注释），分 3 类：

### 1.1 模型未入库（离线加载失败）×2 

*   `tests/models/test_transformers.py::test_pooling[TransformersEmbeddingModel]` —— 用例模型 BAAI/bge-base-en-v1.5 未入库，HF\_HUB\_OFFLINE=1 下加载失败
    
*   `tests/models/test_transformers.py::test_pooling[TransformersForSequenceClassification]` —— 用例模型 papluca/xlm-roberta-base-language-detection 未入库，同上
    

### 1.2 vLLM vs HF 对拍失败 ×1 —— 待查

*   `tests/models/test_transformers.py::test_models[hmellor/Ilama-3.2-1B-auto]` —— transformers backend 对拍（custom code + trust\_remote\_code 路径）不一致， 具体 mismatch 原因待查；模型走 Aone 同构路径 symlink，本体是否就位也待确认
    

### 1.3 registry 新 arch import 链失败 ×3 —— 原因（代码级推断）

*   `tests/models/test_registry.py::test_registry_imports[Gemma4UnifiedForConditionalGeneration]` —— gemma4\_unified.py 顶层 import `transformers.models.gemma4_unified.*`， 镜像内 transformers 版本不含该子模块（推断，精确报错待查日志）
    
*   `tests/models/test_registry.py::test_registry_imports[MiDashengLMModel]` —— midashenglm.py 顶层 `import torchaudio.functional`，PPU 镜像 torchaudio 缺失/不兼容（推断）
    
*   `tests/models/test_registry.py::test_registry_imports[CohereAsrForConditionalGeneration]` —— 引用链至 vllm/transformers\_utils/processors/cohere\_asr.py 顶层 `from torchaudio.functional import melscale_fbanks`，同上 torchaudio 问题（推断）
    

备注：terratorch 注册的 vllm.general\_plugins 插件（terratorch\_fix）在 EngineCore 刷 ImportError（torchgeo>=0.8 移除 trainers.utils）——已确认非致命（load\_plugins\_by\_group 有 try/except 兜底），属已知无害噪声，保留 terratorch 安装以覆盖 Prithvi/Terratorch arch 注册测试。

## 2. Attention（failure，1m23s）

首跑 **1 failed**（无用例级 error）：

*   `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py::test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_storage_block_size` —— 原因明确：create\_vllm\_config() 构造 ModelConfig(model="meta-llama/Meta-Llama-3-8B") 只解析 HF config，但该 gated 仓库不在 NAS 缓存，HF\_HUB\_OFFLINE=1 下直接 ValidationError。 处置：VLLM\_MODEL\_REDIRECT\_PATH 重定向到本地 config-only stub（zero-diff，不改上游测试）； 修复后复跑 131 用例 100 passed / 31 skipped / 0 failed（45s 量级，skip 均为 SM90/SM100/ ROCm/fp8 等硬件架构不匹配，属预期）。恢复条件：Meta-Llama-3-8B stage 到 /nas\_aisw 后删除 stub。
    

注：test\_gdn\_metadata\_builder.py 的 2 个用例（test\_gdn\_build\_classification / test\_has\_initial\_state\_after\_reclassification）为 Aone 快照既有排除（root cause TBD， F2 跟踪），非本次首跑新增。

## 3. Model Executor（success，1m49s）—— 首跑通过

**首跑全绿，无 fail/error，无需 triage。** 选集为 tests/model\_executor/ 的 mock/合成数据 纯单测子集（CPU-only、ROCm aiter、HF 下载类、外部 loader 包文件均已按 Aone 快照 ignore； 唯一需下载 GPT-2 的 TestSafetensorsWeightsIteratorWithEpFilter 由 -k 排除）， 无真模型依赖，首跑即稳定。

## 4. Kernels（failure，2h14m18s）

首跑 fail 集中在 3 个测试文件，**原因均待查**（已整文件 --ignore + 日期注释， 待调查后可细化到用例级）：

| 文件 | 所属 step | 备注 |
| --- | --- | --- |
| tests/kernels/moe/test\_grouped\_topk.py | kernels\_moe | 2026-08-27 PPU run fail，待调查 |
| tests/kernels/quantization/test\_allspark\_gemm.py | kernels\_quantization | 本次测试失败，待调查 |
| tests/kernels/quantization/test\_nvfp4\_emulation.py | kernels\_quantization | v0.23.0 新增测试文件，实测 fail，待调查 |

顺带清理：删除死 ignore `--ignore=tests/kernels/moe/test_fused_deepgemm_moe_permute_kernel.py` （该文件在 v0.23 已不存在，ignore 静默失效）。

注：test\_causal\_conv1d.py（PPU ptxas SIGSEGV）、acmoe 精度/NaN、flashinfer/SM90+/ SM100+ 专用文件等均为 Aone 快照既有排除（含 device\_conditional\_ignores），非本次首跑新增。

## 5. Engine（failure，1h32m08s）

v1\_engine step（tests/v1/engine/）首跑实测：**17 failed / 100 passed / 2 skipped（2732s）**； engine\_basic 与 v1\_e2e\_general step 无新增 red。17 个 red 用例已逐个 deselect + 分类注释， **具体 root cause 均待查**，按语义分 4 类：

### 5.1 abort 语义 ×6 —— 待查

*   tests/v1/engine/test\_abort\_final\_step.py::test\_abort\_during\_final\_step\[False\]
    
*   tests/v1/engine/test\_abort\_final\_step.py::test\_abort\_during\_final\_step\[True\]
    
*   tests/v1/engine/test\_async\_llm.py::test\_multi\_abort\[RequestOutputKind.DELTA\]
    
*   tests/v1/engine/test\_async\_llm.py::test\_multi\_abort\[RequestOutputKind.FINAL\_ONLY\]
    
*   tests/v1/engine/test\_async\_llm.py::test\_abort\_final\_output\[RequestOutputKind.DELTA\]
    
*   tests/v1/engine/test\_async\_llm.py::test\_abort\_final\_output\[RequestOutputKind.FINAL\_ONLY\]
    

### 5.2 EngineCore 基础 ×4 —— 待查（疑似 OOM/调度相关）

*   tests/v1/engine/test\_engine\_core.py::test\_engine\_core
    
*   tests/v1/engine/test\_engine\_core.py::test\_engine\_core\_advanced\_sampling
    
*   tests/v1/engine/test\_engine\_core.py::test\_engine\_core\_concurrent\_batches
    
*   tests/v1/engine/test\_engine\_core.py::test\_engine\_core\_invalid\_request\_id\_type
    

### 5.3 encoder 零 kv-cache 实例 ×6 —— 待查

*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[False-ec\_producer-0.01-False\]
    
*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[False-ec\_consumer-0.7-True\]
    
*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[False-ec\_consumer-0.7-False\]
    
*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[True-ec\_producer-0.01-False\]
    
*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[True-ec\_consumer-0.7-True\]
    
*   tests/v1/engine/test\_engine\_core.py::test\_encoder\_instance\_zero\_kv\_cache\[True-ec\_consumer-0.7-False\]
    

### 5.4 preprocess 错误处理 ×1 —— 待查

*   tests/v1/engine/test\_preprocess\_error\_handling.py::test\_preprocess\_error\_handling
    

## 6. Entrypoints LLM（failure，11m34s）

首跑 fail **3 个用例，同一根因（已查明）**——FlexAttention 反向块表确定性 OOM：

*   tests/entrypoints/llm/test\_collective\_rpc.py::test\_collective\_rpc\[mp-1\]（single 段：mp-1 failed / 3 skipped）
    
*   tests/entrypoints/llm/test\_collective\_rpc.py::test\_collective\_rpc\[mp-2\]（multi 段）
    
*   tests/entrypoints/llm/test\_collective\_rpc.py::test\_collective\_rpc\[ray-2\]（multi 段）
    

根因：本次 run attention backend 解析到 FLEX\_ATTENTION（ppu.py 候选优先级 FLASH\_ATTN→TRITON\_ATTN→FLEX\_ATTENTION，日志栈确认落到 flex），其 metadata build 阶段 physical\_to\_logical\_mapping 分配 (max\_num\_seqs × num\_gpu\_blocks) 的 int64 反向块表； tiny 模型 + 96 GiB 大显存 → max\_num\_seqs=1024（≥70 GiB 卡的 LLM\_CLASS 默认值）、 KV cache 7.38 亿 tokens ≈ 4614 万 blocks → 单卡需 352 GiB >> 96 GiB，EngineCore warmup 即 OOM（tp=2 双卡 KV cache 更大，单次分配要 702.89 GiB）。上游小显存卡（max\_num\_seqs=256 且 num\_gpu\_blocks 小）不触发。

处置：mp-1 deselect（同 step 余 3 case 全 skip，rc=0）；mp-2/ray-2 全灭，deselect 后 collect=0 会使 pytest 以 rc=5 退出造成 step 误红，故 multi step 整段停用（\_run\_step 调用 已注释，参数数组保留作恢复路径）；FlexAttention 大块表 OOM 修复后取消注释恢复。

其余 step（test\_chat、test\_gpu\_utilization）首跑无 red。

## 7. Entrypoints（failure，1m14s）

无用例级 fail，**2 个 step 级 error，均为 v0.23 rebase 布局地雷（原因明确）**：

1.  **entrypoints\_unit step：collection error**。Aone 快照 ignore 集未同步 v0.23 上游目录 布局：instrumentator/sagemaker 已迁入 serve/ 子目录、rpc/offline\_mode 目录不存在， 4 条 ignore 全为死路径静默失效；v0.23 新增的 tests/entrypoints/speech\_to\_text/ 未被 排除，其 correctness/test\_transcription\_api\_correctness.py:43 有模块级 `get_tokenizer(whisper-large-v3)`，离线环境 collection 阶段即炸，且 collection error 的爆炸半径是整次 pytest 调用（rc=2，全 step 不执行）。
    
2.  **v1\_entrypoints step：pytest usage error（exit 4）**。tests/v1/entrypoints/ 目录在 v0.23 已整体移除（用例迁入 tests/entrypoints/openai/），快照仍引用该路径，必挂。
    

处置：ignore 集对齐上游 .buildkite/test\_areas/entrypoints.yaml unit step（增 ignore serve / speech\_to\_text / generate，删 4 条死路径）；删除 v1\_entrypoints step；MODEL\_MAP 同步清理无消费者条目、补 Qwen3-0.6B（anthropic/test\_messages.py）。

## 8. Samplers（success，9m05s）—— 首跑通过

**首跑全绿，无 fail/error，无需 triage。** tests/samplers/ 选集中 test\_beam\_search.py （TinyLlama 已入清单，但 beam search PPU 正确性未验证）与 test\_no\_bad\_words.py （distilgpt2 路径未经清单证实）两个文件保持 Aone 快照既有 ignore，首跑稳定后可评估放开。

---

## 数据来源与置信度说明

*   GitHub 规定查看/下载 Actions 日志与 artifact 需登录（匿名 API 返回 403/401），本报告 未能直接读取首跑日志。
    
*   首跑 red 用例清单与统计数来自各 scripts/ppu/test-area-ppu-\*.sh 内的首跑 triage 标注 （2026-08-27 实测记录）及 5 个首跑修复提交的 diff，与 run 元数据（结论/耗时）交叉一致。
    
*   标注"推断"的原因（models-basic 的 3 个 registry import 链）为代码级依赖分析，未经日志 核实；engine 17 用例与 kernels 3 文件的具体报错待查。建议后续在已登录环境下载各 run 的 `ppu-*-test-results` artifact（内含 junit test.xml）复核精确报错。