# vLLM PPU per-PR CI 开发者使用说明

本文档面向在 flytiger-eco/vllm-for-sail 仓库提交 PR 的开发同事，说明 PPU per-PR CI
的触发方式、执行内容与常见问题。

- 适用分支：**试用期从 `feat/gha-ppu-test` 拉分支、也合回
  `feat/gha-ppu-test`**。具体操作与当前限制见 §2.0。
- 编排入口：`.github/workflows/ci.yml` —— 组织统一门禁链与 PPU area 的统一
  编排；area 定义：`.github/workflows/test-area-ppu-*.yml`（11 个 area，PR 上
  由 ci.yml 以 `workflow_call` 调用）+ `ppu-ci-selfcheck.yml`（静态自检）。
- 测试注册机制：`scripts/ppu/test-area-ppu-<area>.sh` —— 每个 area 一份 pytest
  选集脚本。
- 关联文档：[`ppu-ci-usage.md`](ppu-ci-usage.md)（简版）。

## 1. 机制概述

PPU 整机只有**一台** self-hosted runner，所有 PPU job 串行排队。
PR 上的流水线由 `ci.yml` 统一编排，对齐 flytiger-eco 组织门禁链：

```text
precheck（TruffleHog 密钥扫描）
    ↓
smoke-test（静态冒烟） ∥ ai-code-review（Copilot 评审，最长等 8 分钟）
    ↓
快速档 4 个 PPU area（无人工审批，路径命中才跑）
    ↓（PR 带 ppu-full 标签时，门禁链多一道）
human-review（轮询等至少 1 名真人 approve）
    ↓
标签档 6 个 PPU area
    ↓
ci（Full CI 聚合点，skipped 视为通过）
```

三条组织门禁（precheck / ai-code-review / human-review）是组织仓库
`flytiger-eco/.github` 的 reusable workflow，逻辑集中维护、版本升级对本仓库
透明；PPU area 经 `workflow_call` 调用 `test-area-ppu-*.yml`，路径/标签过滤
仍在各自的 `check-changes` 里（被调时继承 PR 事件上下文，过滤逻辑不变）。

为把白天的工作时段留给快速反馈，11 个 area 按耗时分三档：

1. **快速档（4 个，门禁过后自动跑，无人工审批）**：`attention` /
   `model-executor` / `entrypoints` / `samplers`。
   precheck、smoke-test、ai-code-review 通过后自动排队上 PPU；
   `check-changes` 用 `dorny/paths-filter` 判断 diff 是否命中该 area 的依赖路径，命中才放行进 PPU runner。
2. **标签档（6 个，打 `ppu-full` + 真人 approve 才跑）**：`basic-correctness` /
   `entrypoints-llm` / `lora` / `models-basic` / `engine` / `kernels`。
   PR 带 `ppu-full` 标签时，门禁链里出现 human-review（轮询等 approve），
   审批通过后 6 个 area 才放行。此档
   `check-changes` **只看标签不看路径**，打上标签即可跑（即使没有修改代码，只要打上这个标签就会自动跑）。
3. **不在 per-PR 范围（1 个）**：`models-language` 只有 `workflow_dispatch`与nightly
   入口，PR 上无论如何都不会跑；需要时手动触发（§2.2）。

手动触发（`workflow_dispatch`）不受路径过滤、标签闸门与门禁链限制，可以直接跑。

安全闸门：fork 仓库发起的 PR 一律不进 PPU runner —— 测试容器带
`--privileged` + 设备直通 + NAS 挂载 + artifactory 凭证，不能被外部 PR 的代码
驱动。需要 PPU 验证请在本仓库开分支。（fork PR 上门禁链本身照跑，PPU job
被跳过。）

11 个 area 的 job 完全相同：

```text
check-changes（路径/标签门禁）
    ↓
ppu-<area>-test（PPU runner，docker 内装依赖 + 跑 pytest）
    ↓
ppu-<area>-finish（聚合点，skipped 视为通过）
```

## 2. 标准操作流程

### 2.0 从哪拉分支、PR base 填什么

**从 `feat/gha-ppu-test` 拉分支，PR 的 base 也填`feat/gha-ppu-test`。**

```bash
git fetch
git checkout -b <你的分支> origin/feat/gha-ppu-test
# 改代码，push，然后开 PR，base 填 feat/gha-ppu-test
```

为什么不用 `v0.23.0`：PPU CI 还没合进去，从它拉分支拿不到 workflow
文件。以后合进去了，把上面两处换成 `v0.23.0` 就行。

⚠️ base 只能填 `feat/gha-ppu-test` 或 `v0.23.0`。填其他分支时门禁链照跑、
PPU 测试一个不跑，而且不会报错。

### 2.1 提了 PR 之后会发生什么

1. **门禁链先跑（GitHub 托管 runner，不占 PPU 机器）**：precheck 密钥扫描 →
   smoke-test 与 ai-code-review 并行（等 Copilot 评审，最长 8 分钟）。
2. **快速档 4 个 area：门禁过后自动跑。** 只有改动碰到它们的路径
   （清单见 §3）才跑；没碰到就全部 skip，check 记绿。全绿约 14 分钟
   （PPU 排队时间另计）。
3. **标签档 6 个 area：默认不跑，要打标签 + 等审批。** 改了 kernel、engine、
   模型加载这类底层代码时，给 PR 打上 `ppu-full` 标签 → 该 run 的门禁链里出现
   Human Review Gate → 至少 1 名真人 approve 后 6 个 area 排队跑。全开要
   5 小时以上。
4. **结果：** PR 页面上方是门禁链 checks（Pre-check / Smoke Test /
   AI Code Review / Human Review Gate / Full CI）；每个 PPU area 的聚合 check
   `ppu-<area>-finish` 作为 CI Pipeline run 的嵌套 job 显示。

打标签两种方式，效果一样：

| 方式 | 操作 |
| --- | --- |
| GitHub 网页 | PR 右侧 Labels 勾选 `ppu-full` |
| 命令行 | `gh pr edit <PR号> --add-label ppu-full` |

标签就是普通的 GitHub label，需要本仓库 triage 及以上权限。没权限找maintainer 代打。

打标签后注意：

- 标签档 run 与快速档 run 是**两条独立并发线**：打标签不会取消正在跑的
  快速档，push 新提交也不会取消正在等审批/在跑的标签档。
- 标签已在场时再勾一次不会派发新 run；想重跑标签档，先摘掉再重新打上
  （补发 `labeled` 事件）。
- **摘掉标签不会取消已经在跑的 job**，要取消只能去 Actions 页手动 cancel。
- 打标签时若 PR 还没人 approve，Human Review Gate 黄色等待是**预期行为**，
  审批通过后标签档自动开跑。单 job 上限约 6 小时，等不到审批超时失败后，
  摘掉重打标签即可重新派发。作者不能 approve 自己的 PR（GitHub 限制），
  需其他有 read 权限的成员 approve。

### 2.2 手动跑（不开 PR 也能跑）

网页操作：打开 [Actions 页](https://github.com/flytiger-eco/vllm-for-sail/actions)
→ 左侧选一个 area → 右上角 **Run workflow** → 分支选你的分支 → Run。

⚠️ 分支下拉会把所有分支都列出来，但只有 `feat/gha-ppu-test` 系分支上才有
这些 workflow。选 `main` 或 `v0.23.0` 会直接失败。

命令行等价写法：

```bash
gh workflow run test-area-ppu-attention.yml --ref <你的分支>
```

两个可选参数：

- `test_mode`：`all`（默认）/ `single` / `multi` —— 只跑单卡或多卡部分。
- `pytest_args`：透传给 pytest，比如 `-x`（第一个失败就停）。

用途：改动没命中路径清单，但想跑一次。手动触发不经门禁链与人工审批。

## 3. 触发路径范围

`check-changes` 用 `dorny/paths-filter` 判断 PR diff（头分支对基线分支的完整
差异），在 ci.yml 以 `workflow_call` 调用各 area 后仍由各 area 自己判定。
除下表外，每个 area 的清单都还包含自身的三个 CI 文件：
`.github/workflows/test-area-ppu-<area>.yml`、`scripts/ppu/ppu_install_dependency.sh`、`scripts/ppu/test-area-ppu-<area>.sh`。

快速档 —— 命中才跑：

| Area | 触发路径（除 CI 自身文件） |
| --- | --- |
| `attention` | `tests/v1/attention/**`、`vllm/v1/attention/**` |
| `model-executor` | `tests/model_executor/**`、`vllm/model_executor/**` |
| `entrypoints` | `tests/entrypoints/**`、`tests/v1/entrypoints/**`、`vllm/entrypoints/**` |
| `samplers` | `tests/samplers/**`、`tests/conftest.py`、`vllm/v1/sample/**`、`vllm/model_executor/layers/**` |

标签档 —— 打 `ppu-full` 且审批通过后跑，下表路径仅记录不拦截：

| Area | 观测路径（除 CI 自身文件） |
| --- | --- |
| `basic-correctness` | `tests/basic_correctness/**`、`vllm/*` |
| `entrypoints-llm` | `tests/entrypoints/llm/**`、`vllm/*` |
| `lora` | `tests/lora/**`、`vllm/lora/**` |
| `models-basic` | `tests/models/` 下 registry 与 6 个测试文件、`vllm/*`、`vllm/model_executor/models/*` |
| `engine` | `tests/engine/**`、`tests/v1/engine/**`、`tests/v1/e2e/**`、`tests/test_{sequence,logger,vllm_port}.py`、`vllm/engine/**`、`vllm/v1/engine/**`、`vllm/*` |
| `kernels` | `tests/kernels/**`、`tools/install_deepgemm.sh`、`vllm/config.py`、`vllm/config/**`、`vllm/distributed/device_communicators/**`、`vllm/envs.py`、`vllm/model_executor/layers/{attention,fused_moe,quantization}/**`、`vllm/model_executor/layers/mamba/ops/**`、`vllm/platforms/cuda.py`、`vllm/utils/{deep_gemm,import_utils}.py`、`vllm/v1/attention/**` |

精确清单以各 workflow 的 `on.push.paths` / `check-changes` 过滤器为准。

## 4. 测试套件分档

| 档位 | Area | 内容 | 首跑实测 |
| --- | --- | --- | --- |
| 快速档 | `attention` | 注意力后端，PPU 核心适配点 | 1m23s |
| 快速档 | `model-executor` | 模型执行器 | 1m49s |
| 快速档 | `entrypoints` | OpenAI API 服务入口 | 1m14s |
| 快速档 | `samplers` | 采样正确性 | 9m05s |
| 标签档 | `entrypoints-llm` | offline LLM 类接口（单+多卡） | 11m34s |
| 标签档 | `models-basic` | 核心模型冒烟底线 | 48m07s |
| 标签档 | `engine` | 调度器、KV cache、异步 LLM | 1h32m08s |
| 标签档 | `kernels` | 算子层最大用例集 | 2h14m18s |
| 标签档 | `lora` | LoRA 适配器（单+多卡） |  3h4m4s |
| 标签档 | `basic-correctness` | 冒烟底线（单+多卡） | 33m4s |
| 不在 per-PR | `models-language` | 模型语言测试 3 段（仅 dispatch/nightly） | 29m36s |

串行与取消行为：

- 一台 runner 全仓库共享，所有 PPU job 排队串行；标签档全开会把机器占满
  5 小时以上，白天慎用。
- **push 新提交会取消你同 PR 正在跑的门禁链 + 快速档 run**；标签档 run 走
  独立的并发分组（打标签派发），不受 push 取消影响。

## 5. 构建与依赖链路

基础镜像 tag 见任一 workflow 的 `env.PPU_BASE_IMAGE`（当前为
`llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-vllm0.23.0-py312`）；
模型与数据集走 NAS 预置卷（`/nas_aisw`，离线模式）。

## 6. 查看结果与重跑

### 6.1 结果位置

1. **Job Summary**：每个 area 的 job 页面顶部有 markdown 汇总表，按 step/shard
   列出 tests / passed / failed / errors / skipped / time，先看这张表。
2. **PR 注解**：失败用例以注解形式标在 PR 的 Files changed 上
   （junit 报告解析），不用翻日志就能定位挂掉的 case。
3. **Artifact**：job 页面下载 `ppu-<area>-test-results`，含 `test.xml`（合并后
   的 junit）、`summary.md` 与各分片 pytest 输出日志，保留 14 天。
4. **门禁状态**：每个 area 的聚合 check（`ppu-<area>-finish`）收编后显示在
   `CI Pipeline` run 的嵌套 job 里，整条链另有 `Full CI` 聚合 check。试用期这些
   check **不阻塞合并**，先积累信噪比数据；转 required 时按 CI Pipeline 下的
   新 check 路径名配置分支保护。

### 6.2 重跑

- 失败重跑：Actions 页面对应 run 的 **Re-run failed jobs**。⚠️ Re-run 总是用
  原触发提交的代码与 workflow，**不能用来验证你刚推的修复** —— 验证修复请
  push 新提交或重新 dispatch，并核对 run 页的 commit SHA。
- 完整重跑：push 一个提交。注意会取消同 PR 正在跑的门禁链 + 快速档旧
  run（§4）。
- 标签档重跑：摘除 `ppu-full` 后重新添加（补发 `labeled` 事件）。
- human-review 超时失败（约 6 小时上限）：重打标签或 push 触发新 run 即可，
  审批状态不会丢。
- 缩小范围重跑：workflow_dispatch + `pytest_args`（§2.2）。

## 7. 已知问题与注意事项

1. **试用期 base 只能填 `feat/gha-ppu-test`。** base 白名单现在两处：`ci.yml`
   里 PPU 调用 job 的 `if` 条件，与各 area workflow 的 `push.branches`，都是
   `[v0.23.0, feat/gha-ppu-test]`；合入 v0.23.0 后请把试用分支从这两处摘掉。
2. **Run workflow 的分支下拉会列出跑不了的分支。** 默认分支（`main`）与各
   `v0.*` 分支上都没有 PPU workflow，选中它们 dispatch 会失败；请选
   `feat/gha-ppu-test` 系分支（§2.2）。
3. **工作时间排队是常态。** 一台 runner 全仓库串行，包括手动 dispatch 的 run。
   别连打标签、别频繁空 push。
4. **fork PR 不跑 PPU**（安全设计，§1）：门禁链照跑，PPU job 跳过。需要 PPU
   验证请在本仓库开分支提 PR。
5. **门禁链常见问题**：Human Review Gate 黄色等待 = 等真人 approve（预期）；
   单 job 约 6 小时超时后重打标签或 push；AI Code Review 最长等 Copilot
   8 分钟，超时失败可稍后重跑；门禁 job 报 startup_failure 说明组织仓库
   `flytiger-eco/.github` 的 Actions access 未对本仓库放开，联系维护者。
6. **关键字合规不在线上查**：`alibaba-inc.com`、`t-head`、`aone` 等禁用关键字
   由开发者本地 git 钩子在提交前拦截，配置方法见组织门禁接入说明
   （`flytiger-eco/.github` 仓库）。

## 8. 权限与联系人

- 打 `ppu-full` 标签：需要本仓库 triage 及以上权限；无权限请找 maintainer
  代打。
- approve PR（解开 Human Review Gate）：任何有 read 权限的非作者成员。
- workflow_dispatch 手动触发：需要本仓库 write 权限。
- 新增权限、runner/镜像/NAS 等 CI 基础设施问题：联系 PPU CI 维护者（仓库
  maintainer）。
