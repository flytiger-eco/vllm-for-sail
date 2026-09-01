# vLLM PPU per-PR CI 开发者使用说明

本文档面向在 flytiger-eco/vllm-for-sail 仓库提交 PR 的开发同事，说明 PPU per-PR CI
的触发方式、执行内容与常见问题。

- 适用分支：**试用期从 `feat/gha-ppu-test` 拉分支、也合回
  `feat/gha-ppu-test`**。具体操作与当前限制见 §2.0。
- 工作流文件：`.github/workflows/test-area-ppu-*.yml`（11 个 area）+
  `ppu-ci-selfcheck.yml`（静态自检）。
- 测试注册机制：`scripts/ppu/test-area-ppu-<area>.sh` —— 每个 area 一份 pytest
  选集脚本。
- 关联文档：[`ppu-ci-usage.md`](ppu-ci-usage.md)（简版）。

## 1. 机制概述

PPU 整机只有**一台** self-hosted runner，所有 PPU job 串行排队。
为把白天的工作时段留给快速反馈，11 个 area 按耗时分三档：

1. **快速档（4 个，PR 自动触发）**：`attention` / `model-executor` /
   `entrypoints` / `samplers`。
   PR 创建/更新时 `check-changes` 用 `dorny/paths-filter` 判断 diff 是否命中该 area 的依赖路径，命中才放行进 PPU runner。
2. **标签档（6 个，打 `ppu-full` 才跑）**：`basic-correctness` /
   `entrypoints-llm` / `lora` / `models-basic` / `engine` / `kernels`。
   PR 事件类型含 `labeled`，打上 `ppu-full` 标签即派发新 run，无需 push 空提交。此档
   `check-changes` **只看标签不看路径**，打上标签即可跑。
3. **不在 per-PR 范围（1 个）**：`models-language` 只有 `workflow_dispatch`与nightly
   入口，PR 上无论如何都不会跑；需要时手动触发（§2.2）。

手动触发（`workflow_dispatch`）不受路径过滤与标签闸门限制，可以直接跑。

安全闸门：fork 仓库发起的 PR 一律不进 PPU runner —— 测试容器带
`--privileged` + 设备直通 + NAS 挂载 + artifactory 凭证，不能被外部 PR 的代码
驱动。需要 PPU 验证请在本仓库开分支。

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

⚠️ base 只能填 `feat/gha-ppu-test` 或 `v0.23.0`。填其他分支时 CI 一个不跑，
而且不会报错。

### 2.1 提了 PR 之后会发生什么

1. **快速档 4 个 area：自动跑。** 只有改动碰到它们的路径
   （清单见 §3）才跑；没碰到就全部 skip，check 记绿。全绿约 14 分钟。
2. **标签档 6 个 area：默认不跑，要自己打标签。** 改了 kernel、engine、
   模型加载这类底层代码时，给 PR 打上 `ppu-full` 标签。全开要 5 小时以上。
3. **结果：** PR 页面上每个 area 对应一个 `ppu-<area>-finish` check。

打标签两种方式，效果一样：

| 方式 | 操作 |
| --- | --- |
| GitHub 网页 | PR 右侧 Labels 勾选 `ppu-full` |
| 命令行 | `gh pr edit <PR号> --add-label ppu-full` |

标签就是普通的 GitHub label，需要本仓库 triage 及以上权限。没权限找maintainer 代打。

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

用途：改动没命中路径清单，但想跑一次。


## 3. 触发路径范围

`check-changes` 用 `dorny/paths-filter` 判断 PR diff（头分支对基线分支的完整
差异）。除下表外，每个 area 的清单都还包含自身的三个 CI 文件：
`.github/workflows/test-area-ppu-<area>.yml`、`scripts/ppu/ppu_install_dependency.sh`、`scripts/ppu/test-area-ppu-<area>.sh`。

快速档 —— 命中才跑：

| Area | 触发路径（除 CI 自身文件） |
| --- | --- |
| `attention` | `tests/v1/attention/**`、`vllm/v1/attention/**` |
| `model-executor` | `tests/model_executor/**`、`vllm/model_executor/**` |
| `entrypoints` | `tests/entrypoints/**`、`tests/v1/entrypoints/**`、`vllm/entrypoints/**` |
| `samplers` | `tests/samplers/**`、`tests/conftest.py`、`vllm/v1/sample/**`、`vllm/model_executor/layers/**` |

标签档 —— 打 `ppu-full` 即跑，下表路径仅记录不拦截：

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
- **push 新提交会取消你正在跑的同 area run**。

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
4. **门禁状态**：每个 area 有独立聚合 check（`ppu-<area>-finish`）。试用期这些
   check **不阻塞合并**，先积累信噪比数据；两周后把快速档 4 个转为 required。

### 6.2 重跑

- 失败重跑：Actions 页面对应 run 的 **Re-run failed jobs**。⚠️ Re-run 总是用
  原触发提交的代码与 workflow，**不能用来验证你刚推的修复** —— 验证修复请
  push 新提交或重新 dispatch，并核对 run 页的 commit SHA。
- 完整重跑：push 一个提交。注意会取消同 area 正在跑的旧 run（§4）。
- 合入 v0.23.0 后：`ppu-full` 标签已在场时再打一次不会派发新 run；只想重跑
  标签档可摘除 `ppu-full` 后重新添加（补发 `labeled` 事件）。
- 缩小范围重跑：workflow_dispatch + `pytest_args`（§2.2）。

## 7. 已知问题与注意事项

1. **试用期 base 只能填 `feat/gha-ppu-test`。** 各 workflow 的 `push` 与
   `pull_request` 过滤已同时列入 `[v0.23.0, feat/gha-ppu-test]`；合入 v0.23.0
   后请把试用分支从这两处摘掉。
2. **Run workflow 的分支下拉会列出跑不了的分支。** 默认分支（`main`）与各
   `v0.*` 分支上都没有 PPU workflow，选中它们 dispatch 会失败；请选
   `feat/gha-ppu-test` 系分支（§2.2）。
3. **工作时间排队是常态。** 一台 runner 全仓库串行，包括手动 dispatch 的 run。
   别连打标签、别频繁空 push。
4. **fork PR 不跑 PPU**（安全设计，§1）。请在本仓库开分支提 PR。

## 8. 权限与联系人

- 打 `ppu-full` 标签：需要本仓库 triage 及以上权限；无权限请找 maintainer
  代打。
- workflow_dispatch 手动触发：需要本仓库 write 权限。
- 新增权限、runner/镜像/NAS 等 CI 基础设施问题：联系 PPU CI 维护者（仓库
  maintainer）。
