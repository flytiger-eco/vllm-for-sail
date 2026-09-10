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
PR 事件（opened / synchronize / labeled / …）
                    ↓
     precheck（TruffleHog 密钥扫描）
                    ↓
   smoke-test ∥ ai-code-review（并行）
                    ↓
  ┌─────────────────┴────────────────────┐
  【快速档】无人工介入               【标签档】两处人工动作
  ↓                                       ↓
  ① check-changes 路径门禁               ① 打 ppu-full 标签 ←〔人工〕
     diff 命中该 area 路径才跑              labeled 只认 ppu-full 与
     （路径清单见 §3）                      ppu-review-waived 两个拉起标签
  ↓                                         （需 triage 权限，§8）
  ② attention / model-executor           ↓
     entrypoints / samplers              ② human-review Gate，二选一放行：
     全绿约 14 min                          a〔人工〕真人 approve PR（轮询等，
                                               上限 6h，超时变红）
                                            b〔人工〕打 ppu-review-waived 标签
                                               直接豁免（作者无法 approve 自己
                                               的 PR，单人开发走这条）
                                         ↓
                                         ③basic-correctness / entrypoints-llm
                                            lora / models-basic / engine / kernels
                                            全开 5h+，白天慎用
  └─────────────────┬────────────────────┘
                    ↓
 ci（Full CI 聚合点，skipped 视为通过）
```

两条线共用前置门禁、也共用**同一个 CI run**（单组 concurrency，见 §4），分叉后
各自排队上 PPU，最后汇入 Full CI 聚合点；线 B 的①（打 `ppu-full`）与
②（真人 approve 或打 `ppu-review-waived` 豁免）是两处**必需的人工动作**，
不做就停在原地。

三条组织门禁（precheck / ai-code-review / human-review）是组织仓库
`flytiger-eco/.github` 的 reusable workflow，逻辑集中维护、版本升级对本仓库
透明；PPU area 经 `workflow_call` 调用 `test-area-ppu-*.yml`，路径/标签过滤
仍在各自的 `check-changes` 里（被调时继承 PR 事件上下文，过滤逻辑不变）。

为把白天的工作时段留给快速反馈，11 个 area 按耗时分三档：

1. **快速档（4 个，门禁过后自动跑，无人工审批）**：`attention` /
   `model-executor` / `entrypoints` / `samplers`。
   precheck -> (smoke-test//ai-code-review) 都通过后自动排队上 PPU；
   `check-changes` 用 `dorny/paths-filter` 判断 diff 是否命中该 area 的依赖路径，命中才放行进 PPU runner。
2. **标签档（6 个，打 `ppu-full` + 过审批才跑）**：`basic-correctness` /
   `entrypoints-llm` / `lora` / `models-basic` / `engine` / `kernels`。
   PR 带 `ppu-full` 标签时门禁链里出现 human-review，两条放行通道二选一：
   真人 approve（轮询等待），或再打一个 `ppu-review-waived` 标签直接豁免
   （GitHub 禁止作者 approve 自己的 PR，单人提交时只能走这条，详见 §2.1）。
   放行后 6 个 area 才开跑。此档
   `check-changes` **只看标签不看路径**，打上标签即可跑（即使没有修改代码，只要打上这个标签就会自动跑）。
3. **不在 per-PR 范围（1 个）**：`models-language` 只有 `workflow_dispatch`与nightly
   入口，PR 上不会跑。

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

1. 门禁：precheck  → (smoke-test//ai-code-review)
2. **快速档 4 个 area：门禁通过且改动命中各自的路径（清单见 §3）后跑。** 没碰到就全部 skip，check 记绿。全绿约 14 分钟。
3. **标签档 6 个 area：默认不跑，要门禁 + 打标签(ppu-full) + 过审批。** 改了 kernel、engine、模型加载这类底层代码时，给 PR 打上 `ppu-full` 标签 → 该 run 的门禁链里出现
   Human Review Gate → 放行后 6 个 area 排队跑。全开要
   5 小时以上。放行有两条互斥通道：真人 approve（job 名 `Human Review Gate`），
   或打 `ppu-review-waived` 标签豁免（job 名 `Human Review Gate (Waived)`），
   任一 success 即放行，另一个显示 skipped 属正常。
4. **结果：** PR 页面上方是门禁链 checks（Pre-check / Smoke Test /
   AI Code Review / Human Review Gate 或 Human Review Gate (Waived) /
   Full CI）；每个 PPU area 的聚合 check
   `ppu-<area>-finish` 作为 CI Pipeline run 的嵌套 job 显示。

打标签两种方式，效果一样；想连审批一起免掉就把两个标签一次带上：

| 目的 | 操作 |
| --- | --- |
| 只拉起标签档（仍需真人 approve） | 网页勾选 `ppu-full`，或 `gh pr edit <PR号> --repo flytiger-eco/vllm-for-sail --add-label ppu-full` |
| 拉起标签档 + 豁免审批 | 再勾 `ppu-review-waived`，或 `gh pr edit <PR号> --repo flytiger-eco/vllm-for-sail --add-label ppu-full,ppu-review-waived` |

标签就是普通的 GitHub label，需要本仓库 triage 及以上权限。没权限找maintainer 代打。
建 PR 时也可以直接带上，省一次事件：`gh pr create --label ppu-full --label ppu-review-waived …`
——opened 与两个 labeled 事件会同组收敛成一个 run，不会重复跑（§4）。

打标签后注意：

- **同一 PR 任何时刻只有一个 CI run**：`ppu-full` / `ppu-review-waived` 与 push 类
  事件同属一个并发组，后到者取消先到者（旧版分成两条独立并发线，会让
  6 个标签档重复跑两遍，已修）。存活的那个 run 必然覆盖全部 10 个 area。
- 打**无关**标签（如 `ready`）另属 noop 组，产生的 run 整体 skip，不会误取消
  正在跑的主 run。
- 因此打拉起标签**会**取消正在编译的 build-wheel、从头来一遍；别连着打、
  别打完马上又 push。
- 标签已在场时再勾一次不会派发新 run；想重跑标签档，先摘掉再重新打上
  （补发 `labeled` 事件）。
- **摘掉标签不会取消已经在跑的 job**，要取消只能去 Actions 页手动 cancel。
- 没打 `ppu-review-waived` 时，Human Review Gate 黄色等待是**预期行为**，真人
  approve 后标签档自动开跑；单 job 上限约 6 小时，超时失败后摘掉重打标签
  即可重新派发。**作者不能 approve 自己的 PR**（GitHub 平台限制，网页 / `gh` /
  API 一律返回 "Can not approve your own pull request"），要么找其他有 read
  权限的成员 approve，要么直接打 `ppu-review-waived` 豁免——后者是单人开发
  时的常规做法（打标签需 triage 权限、在 PR 时间线留痕，可审计）。

### 2.2 手动跑（暂未合入默认分支，触发不了）

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
- **单组互斥：同一 PR 任何时刻只有一个 CI Pipeline run。** push 新提交、打
  `ppu-full`、打 `ppu-review-waived` 都归 `ci-<PR号>-main` 组，后到者取消先到
  者；存活的那个 run 必然覆盖全部 10 个 area（快速档已不再排除 labeled
  事件，否则收敛后存活的 labeled run 会丢掉 4 个快速档）。
- 打**无关**标签（既非 `ppu-full` 也非 `ppu-review-waived`）归 `ci-<PR号>-noop`
  组：这种 run 的所有 job 都会被闸门 skip，隔离出来才不会把正在编译 wheel
  的主 run 误取消。

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
- 完整重跑：push 一个提交。注意会取消同 PR 正在跑的整个旧 run（含标签档，
  单组互斥，§4）。
- 标签档重跑：摘除 `ppu-full` 后重新添加（补发 `labeled` 事件）。
- human-review 超时失败（约 6 小时上限）：重打标签或 push 触发新 run 即可，
  审批状态不会丢；不想等审批就直接打 `ppu-review-waived`。
- 缩小范围重跑：workflow_dispatch + `pytest_args`（§2.2）。

## 7. 已知问题与注意事项

1. **试用期 base 填 `feat/gha-ppu-test`。** base 白名单现在两处：`ci.yml`
   里 PPU 调用 job 的 `if` 条件，与各 area workflow 的 `push.branches`，都是
   `[v0.23.0, feat/gha-ppu-test]`；合入 v0.23.0 后请把试用分支从这两处摘掉。
2. **Run workflow 的分支下拉会列出跑不了的分支。** 默认分支（`main`）与各
   `v0.*` 分支上都没有 PPU workflow，选中它们 dispatch 会失败；请选
   `feat/gha-ppu-test` 系分支（§2.2）。
3. **工作时间排队是常态。** 一台 runner 全仓库串行，包括手动 dispatch 的 run。
   别连打标签、别频繁空 push。
4. **fork PR 不跑 PPU**（安全设计，§1）：门禁链照跑，PPU job 跳过。需要 PPU
   验证请在本仓库开分支提 PR。
5. **门禁链常见问题**：`Human Review Gate` 黄色等待 = 等真人 approve（预期），
   单 job 约 6 小时超时后重打标签或 push，或直接打 `ppu-review-waived` 豁免
   （豁免时生效的是 `Human Review Gate (Waived)`，前者显示 skipped 属正常）；
   AI Code Review 最长等 Copilot 8 分钟，超时失败可稍后重跑；门禁 job 报
   startup_failure 说明组织仓库 `flytiger-eco/.github` 的 Actions access 未对
   本仓库放开，联系维护者。
6. **关键字合规不在线上查**：`alibaba-inc.com`、`t-head`、`aone` 等禁用关键字
   由开发者本地 git 钩子在提交前拦截，配置方法见组织门禁接入说明
   （`flytiger-eco/.github` 仓库）。

## 8. 权限与联系人

- 打 `ppu-full` / `ppu-review-waived` 标签：需要本仓库 triage 及以上权限；
  无权限请找 maintainer 代打。
- approve PR（解开 Human Review Gate）：任何有 read 权限的**非作者**成员。
  作者无法 approve 自己的 PR（GitHub 平台限制），单人开发时用
  `ppu-review-waived` 标签豁免替代。
- workflow_dispatch 手动触发：需要本仓库 write 权限。
- 新增权限、runner/镜像/NAS 等 CI 基础设施问题：联系 PPU CI 维护者（仓库
  maintainer）。
