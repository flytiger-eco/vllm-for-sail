# PPU CI Area 迁移进度追踪（Aone CI → GitHub Actions）

> 更新时间：2026-08-27。每完成一个 area 的 GHA 迁移，就把状态列的 `⬜` 改为 `✅`。
> 迁移方法见 skill：`migrate-aoneci-ppu-area-to-gha`。

## 进度总览

- 总计 **29** 个 area（`.aoneci/test-area-ppu-*.yaml`），已完成 **10** 个，剩余 **19** 个。
- 已完成产物：
  - `.github/workflows/test-area-ppu-basic-correctness.yml` + `scripts/ppu/test-area-ppu-basic-correctness.sh`
  - `.github/workflows/test-area-ppu-lora.yml` + `scripts/ppu/test-area-ppu-lora.sh`
  - `.github/workflows/test-area-ppu-engine.yml` + `scripts/ppu/test-area-ppu-engine.sh`
  - `.github/workflows/test-area-ppu-attention.yml` + `scripts/ppu/test-area-ppu-attention.sh`
  - `.github/workflows/test-area-ppu-model-executor.yml` + `scripts/ppu/test-area-ppu-model-executor.sh`
  - `.github/workflows/test-area-ppu-samplers.yml` + `scripts/ppu/test-area-ppu-samplers.sh`
  - `.github/workflows/test-area-ppu-entrypoints.yml` + `scripts/ppu/test-area-ppu-entrypoints.sh`
  - `.github/workflows/test-area-ppu-entrypoints-llm.yml` + `scripts/ppu/test-area-ppu-entrypoints-llm.sh`
  - `.github/workflows/test-area-ppu-models-basic.yml` + `scripts/ppu/test-area-ppu-models-basic.sh`
  - `.github/workflows/test-area-ppu-kernels.yml` + `scripts/ppu/test-area-ppu-kernels.sh`

## 批次 1：核心链路与用户入口（P0，10 个）

目标：冒烟 → 调度 → 执行 → 注意力 → 采样 → 入口 → 模型冒烟 → LoRA 全链路覆盖，形成 PR 门禁最小闭环。

| 状态 | Area | 规模（组/文件） | 单/多卡 | 说明 |
|---|---|---|---|---|
| ✅ | basic-correctness | 4 / 3 | 单+多卡 | 冒烟底线（首轮先行落地） |
| ✅ | engine | 3 / 11 | 单卡 | 引擎核心：调度器、KV cache、异步 LLM |
| ✅ | attention | 1 / — | 单卡 | 注意力后端，PPU 核心适配点 |
| ✅ | model-executor | 1 / 8 | 单卡 | 模型执行器，连接引擎与模型层 |
| ✅ | samplers | 1 / 3 | 单卡 | 采样正确性，直接决定输出质量 |
| ✅ | entrypoints | 2 / 2 | 单卡 | OpenAI API 服务入口 |
| ✅ | entrypoints-llm | 4 / 3 | 单+多卡 | offline LLM 类接口 |
| ✅ | models-basic | 1 / 7 | 单卡 | 核心模型冒烟底线 |
| ✅ | kernels | 6 / 51 | 单卡 | 算子层最大用例集，可单独排期 |
| ✅ | lora | 6 / 13 | 单+多卡 | LoRA 适配器（首轮先行落地） |

## 批次 2：特性、模型族与精度回归（P1/P2，12 个）

目标：补齐量化/分布式/加速特性，扩展模型族与精度回归覆盖。

| 状态 | Area | 规模（组/文件） | 单/多卡 | 说明 |
|---|---|---|---|---|
| ⬜ | quantization | 3 / 21 | 单卡 | 量化方案 |
| ⬜ | weight-loading | 6 / 1 | 多卡 | 权重加载链路（多量化格式 × 多卡） |
| ⬜ | distributed | 1 / 5 | 单+多卡 | 分布式通信/并行基础 |
| ⬜ | spec-decode | 4 / — | 单卡 | 投机解码 |
| ⬜ | expert-parallelism | 2 / 2 | 单+多卡 | MoE EP，DeepSeek 类模型必需 |
| ⬜ | compile | 2 / 2 | 多卡 | torch.compile 管线 |
| ⬜ | model-runner-v2 | 10 / 11 | 单+多卡 | 组数最多的 area |
| ⬜ | models-language | 1 / 1 | 单卡 | 语言模型族（参数化展开实际很多） |
| ⬜ | models-multimodal | 2 / 16 | 单卡 | 多模态模型覆盖 |
| ⬜ | models-distributed | 3 / 2 | 多卡 | 多卡模型验证 |
| ⬜ | lm-eval | 3 / 1 | 单+多卡 | 精度回归（lm-eval-harness） |
| ⬜ | e2e-integration | 3 / 1 | 单+多卡 | 端到端集成 |

## 批次 3：兼容性杂项与性能（P3，7 个，nightly 性质）

| 状态 | Area | 规模（组/文件） | 单/多卡 | 说明 |
|---|---|---|---|---|
| ⬜ | pytorch | 2 / 8 | 单卡 | torch 层兼容性 |
| ⬜ | cuda | 2 / 2 | 单卡 | CUDA 兼容层 |
| ⬜ | misc | 5 / 23 | 单卡 | 杂项工具集，量大但多为边缘功能 |
| ⬜ | plugins | 脚本制 | 单卡 | 无 ppu_extras，用例在 `aone_ci/scripts/test_area_ppu_plugins.sh` |
| ⬜ | ray-compat | 脚本制 | 单卡 | 无 ppu_extras，用例在 `aone_ci/scripts/test_area_ppu_ray_compat.sh` |
| ⬜ | benchmarks | 1 / 2 | 单卡 | 常规性能基准，非阻塞 |
| ⬜ | perf-bench | 两段式 | 4 卡 | build-wheel + benchmark nightly 流水线，结构特殊且依赖内部仓库编译，最后单独处理 |

## 覆盖完整性说明

- 批次 1（含首轮先行落地的 basic-correctness、lora，共 10 个 area）已全部完成，覆盖：**冒烟 → 引擎 → 算子 → 采样 → API 入口 → 模型冒烟 → LoRA**，功能维度覆盖完整，可作为 PR 门禁最小全集（待首跑验证转绿后启用）。
- 批次 2 补齐量化/分布式/加速特性与模型族/精度回归；批次 3 为兼容性杂项与 nightly 性能回归。

## Workflow 体验优化方向（参考其他仓库 CI 设计）

1. **可复用 workflow（`workflow_call`）**：27 个 area 共用一套三段式模板（check-changes → ppu-test → ppu-test-finish），各 area 仅传参（area 名、shards、超时），参考 PyTorch `_linux-test.yml` 模式。
2. **统一门禁聚合 job**：所有 area 汇聚到单个 `ppu-ci-success` required check，分支保护只配一条规则。
3. **PR 路径过滤 + 手动全量**：`dorny/paths-filter` 按需触发（已具备）+ `workflow_dispatch` 支持选 area / shards / 跳过 diff 检查。
4. **结果可读性**：junit XML 上传 artifact + `GITHUB_STEP_SUMMARY` 输出 Markdown 结果表。
5. **flaky 处理**：`nick-fields/retry` 或 pytest-rerunfailures 单次重试；失败自动打包日志片段。
6. **nightly 分层**：PR 跑增量，nightly cron 跑全量 + lm-eval / benchmarks，性能结果做历史趋势对比。
