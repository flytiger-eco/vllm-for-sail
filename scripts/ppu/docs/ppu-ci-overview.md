# PPU CI 现状总结（feat/gha-ppu-test 分支）

> 整理日期：2026-09-08。分支：`feat/gha-ppu-test`（基线 `v0.23.0`，领先 10 个 PPU 相关 commit）。
> 面向读者：需要快速了解本分支 PPU CI 全貌的维护者与开发同学。
> 各子主题的深入文档见文末「文档导航」。

---

## 1. 一句话总览

本分支在 vLLM v0.23.0 基线上搭建了一套完整的 PPU GitHub Actions CI：**PR 门禁链统一编排（ci.yml）+ 11 个 test-area 分三档触发 + nightly 定时兜底 + 静态自检 + PPU wheel 开发构建**。已从 Aone CI 迁移 11/29 个 area，批次 1（PR 门禁最小闭环）全部落地。

```text
PR 事件（opened / synchronize / labeled / …）
                ↓
  precheck（TruffleHog 密钥扫描，组织 reusable workflow）
                ↓
  smoke-test（PPU 静态冒烟）∥ ai-code-review（Copilot Gate）
                ↓
  ┌─────────────┴──────────────┐
  快速档 ×4（无人工审批）        标签档：ppu-full 标签 → human-review 真人审批
  attention / model-executor     ↓
  entrypoints / samplers         basic-correctness / entrypoints-llm / lora
  （路径过滤命中才跑，≈14min）    models-basic / engine / kernels（全开 5h+）
  └─────────────┬──────────────┘
                ↓
  Full CI 聚合点（skipped 视为通过，供分支保护配单条 required check）
```

---

## 2. 硬件与运行环境约束（一切设计的出发点）

- **PPU 整机仅一台 self-hosted runner**（`runs-on: [self-hosted, ppu]`，板卡 OAM-810E），所有 PPU job 串行排队 → 测试必须分层，重量级 area 不能自动跑。
- 测试容器：统一基础镜像 `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-vllm0.23.0-py312`（各 workflow 的 `env.PPU_BASE_IMAGE`）。
- 容器参数：`--privileged` + `--network host` + `/dev/alixpu_ctl`、`/dev/alixpu` 设备直通 + `--shm-size=8g` + `memlock=-1` + 挂载 `/nas_aisw`（模型/数据集预置，`HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub`，离线模式）。
- **fork PR 一律不进 PPU runner**（特权容器 + 设备 + NAS + artifactory 凭证，不能被外部代码驱动）；fork PR 上门禁链照跑、PPU job 跳过。
- `runs-on` 的 `ppu` 标签已在 `.github/actionlint.yaml` 声明。

---

## 3. Workflow 文件清单（.github/workflows/，共 15 个 PPU 相关）

| 文件 | 作用 | 触发方式 |
| --- | --- | --- |
| `ci.yml` | PR 统一编排入口：组织门禁链 + 10 个 area 的 workflow_call + Full CI 聚合 | pull_request（base 白名单 `v0.23.0` / `feat/gha-ppu-test`） |
| `test-area-ppu-attention.yml` | 快速档：注意力后端 | push（paths）/ workflow_call / dispatch |
| `test-area-ppu-model-executor.yml` | 快速档：模型执行器 | 同上 |
| `test-area-ppu-entrypoints.yml` | 快速档：OpenAI API 入口 | 同上 |
| `test-area-ppu-samplers.yml` | 快速档：采样正确性 | 同上 |
| `test-area-ppu-basic-correctness.yml` | 标签档：冒烟底线（单+多卡） | workflow_call（ppu-full）/ dispatch |
| `test-area-ppu-entrypoints-llm.yml` | 标签档：offline LLM 接口（单+多卡） | 同上 |
| `test-area-ppu-lora.yml` | 标签档：LoRA（单+多卡） | 同上 |
| `test-area-ppu-models-basic.yml` | 标签档：核心模型冒烟 | 同上 |
| `test-area-ppu-engine.yml` | 标签档：调度器/KV cache/异步 LLM | 同上 |
| `test-area-ppu-kernels.yml` | 标签档：算子层最大用例集 | 同上 |
| `test-area-ppu-models-language.yml` | 语言模型族 3 段（仅 dispatch + nightly，**PR 不跑**） | workflow_dispatch |
| `nightly-ppu.yml` | 标签档 6 个 + models-language 的定时兜底回归 | schedule（UTC 18:00）/ dispatch（area_set 可选） |
| `ppu-ci-selfcheck.yml` | 静态自检：`check-exclusions.py` 校验排除项路径存在 | push/PR（paths 过滤）/ dispatch |
| `build-ppu-wheel.yml` | PPU wheel 开发构建 + 构建成功发 prerelease | push（feat/gha-ppu-test，限自身+脚本路径） |

注：`ci.yml` 中的组织门禁（precheck / ai-code-review / human-review）是 `flytiger-eco/.github` 仓库的 reusable workflow，本仓库只做 `uses` + `needs` 编排。

---

## 4. 三档触发机制（单 runner 约束下的分层）

### 4.1 快速档（4 个，PR 自动跑，无人工审批）

| Area | 触发路径（除 CI 自身文件） | 首跑实测 |
| --- | --- | --- |
| attention | `tests/v1/attention/**`、`vllm/v1/attention/**` | 1m23s |
| model-executor | `tests/model_executor/**`、`vllm/model_executor/**` | 1m49s |
| entrypoints | `tests/entrypoints/**`、`tests/v1/entrypoints/**`、`vllm/entrypoints/**` | 1m14s |
| samplers | `tests/samplers/**`、`tests/conftest.py`、`vllm/v1/sample/**`、`vllm/model_executor/layers/**` | 9m05s |

合计约 14 分钟。路径过滤用 `dorny/paths-filter`，命中才进 PPU runner。

### 4.2 标签档（6 个，`ppu-full` 标签 + human-review 真人审批）

| Area | 首跑实测 |
| --- | --- |
| entrypoints-llm | 11m34s |
| models-basic | 48m07s |
| engine | 1h32m08s |
| kernels | 2h14m18s |
| lora | 3h04m（已标定） |
| basic-correctness | 33m（已标定） |

- 全开 5 小时以上，白天慎用；改动触及 kernel / engine / 模型加载时应自觉打标签。
- 标签档 `check-changes` **只看标签不看路径**，但会打印 `[gate] paths_hit=...` 观测日志，用于后续评估哪些 area 可提升为快速档。
- `labeled` 事件只认 `ppu-full` 本身：PR 已带 ppu-full 时打无关标签不会重开 6 个重量级 area。
- 并发设计：标签档 run 与快速档 run 是两条独立并发线（ci.yml 的 concurrency group 按事件类型区分 `label`/`push`），互不取消；摘掉标签不会取消在跑 job。
- human-review 上限约 6h，超时变红后摘掉重打标签即可重派。

### 4.3 定时档（nightly-ppu.yml）

- **工作日 02:00（北京）**：basic-correctness / entrypoints-llm / lora / models-basic（短集）。
- **周六 02:00（北京）**：engine / kernels / models-language（长集）。
- 拆两档的原因：7 个 area 连跑会占用到工作时段、堵住白天 PR 快速档。
- 串行防抢卡：workflow 级 concurrency（`nightly-ppu-run`，排队不取消）+ matrix `max-parallel: 1`，单 area 失败不影响其余排队（`fail-fast: false`）。
- `workflow_dispatch` 支持 `area_set = auto / nightly / weekend / all`。

---

## 5. test-area 标准结构（11 个 area 同构）

每个 area = 1 个 workflow（`.github/workflows/test-area-ppu-<area>.yml`）+ 1 个选集脚本（`scripts/ppu/test-area-ppu-<area>.sh`）。

```text
check-changes（ubuntu-latest，路径/标签门禁）
    ↓
ppu-<area>-test（self-hosted PPU runner）
  Clean workspace（docker root 清理）→ checkout →
  docker 内：ppu_install_dependency.sh（装依赖）→ test-area-ppu-<area>.sh（跑 pytest）
  → Publish test summary（GITHUB_STEP_SUMMARY）→ Upload artifact（14 天）
  → Annotate failed tests（mikepenz/action-junit-report，PR 内联注解）
    ↓
ppu-<area>-finish（聚合点，skipped 视为通过）
```

- **调试输入**：`workflow_dispatch` 与 `workflow_call` 均支持 `test_mode`（all/single/multi）与 `pytest_args`（透传 PYTEST_EXTRA_ARGS，如 `-k test_foo -x`），把 2 小时的 area 压到分钟级，是排障主要手段。
- **结果出口**：Job Summary 汇总表（tests/passed/failed/errors/skipped/time）+ PR 注解 + artifact（合并 junit `test.xml`、`summary.md`、分片日志）。
- 例外：`test-area-ppu-models-language.yml` 无 check-changes/聚合点，单 job，仅 dispatch + nightly。

### 执行脚本侧

- `ppu_install_dependency.sh`（174 行）：`[diag]` 环境盘点 → `[deps]` PPU 依赖（镜像预装优先，缺失走内网 artifactory）→ `[deps]` pytest 依赖 → `[cext]` 从镜像借用预编译 C 扩展 `.so` 到源码树。
- 选集脚本内含 `--ignore` / `--deselect` / `-k` 排除项（规模见 `ppu-ci-exclusions.md`：kernels 56 条 ignore 与 engine 17 条 deselect 是两处主要覆盖缺口），并有 junit 汇总逻辑（含 `tests==0` 空收集视为配置错误的假绿防护）。
- `check-exclusions.py`：纯 Python 静态校验排除项引用的测试路径真实存在（失效的 `--ignore` 会被 pytest 静默吞掉），由 `ppu-ci-selfcheck.yml` 与 ci.yml 的 smoke-test 双路执行。
- `model_alises/`：模型/adapter 清单（checkpoints_cleaned.json 等），供脚本离线解析模型路径。

---

## 6. 其他两条独立流水线

### 6.1 静态自检 ppu-ci-selfcheck.yml

跑在 ubuntu-latest（不占 PPU），paths 过滤（`scripts/ppu/**`、`tests/**`、自身），执行 `check-exclusions.py`。存在意义：把「排除项指向已不存在路径」这类纯静态问题前置到廉价 runner。

### 6.2 PPU wheel 构建 build-ppu-wheel.yml（2026-09-08 新增）

- 移植自 PR #6（nightly 构建），定位差异：本分支开发验证构建，push（合入 main 前唯一触发入口）成功即发 prerelease；脚本随分支携带（`scripts/ppu/build-ppu-wheel.sh`）。
- matrix：py3.12 / cuda13.0 / x86_64（aarch64 无 base image 与 SDK，脚本内直接 exit 1）。
- 版本号 `0.23.0.dev<yyyymmdd>+g<hash>`，经 workflow → 脚本 → docker -e → setup.py 四层传递，构建后对 wheel 文件名做硬校验。
- 安全设计：`release-wheel` 拆独立 job（只转发 artifact，`contents: write` 令牌不暴露给以 root 跑编译的构建 job）；tag 形如 `dev-gha-ppu-test-<date>-<hash>`，与 PR #6 的 nightly tag 隔离。
- 战略意义：接上 wheel 构建后，`ppu_install_dependency.sh` 借用镜像预编译 `.so` 的模式可被替换，**解除「改 csrc/ 假绿」这一最大盲区**（当前 kernels 路径清单刻意不含 `csrc/**`）。

---

## 7. 已知限制与待办

| # | 事项 | 现状/恢复条件 |
| --- | --- | --- |
| 1 | **`csrc/` C/C++ 改动 PPU CI 不验证** | 借用镜像预编译 `.so`，源码改动不重编译 → 假绿。kernels 路径清单刻意不含 `csrc/**`。恢复条件：接 PPU wheel 构建（build-ppu-wheel.yml 已起步） |
| 2 | workflow 模板化未做 | 11 份 workflow/脚本同构拷贝（约 170 行 junit/summary 样板重复），是批次 2（再迁 18 个 area）的**前置硬约束** |
| 3 | 排除项台账待收敛 | 未定位 root cause 的排除：engine 17 条（E-1~E-4）、kernels 3 文件（K-1）、models-basic 4 条（M-1/M-2）、attention 2 条（A-1）；另有上游选集漂移无检测机制（D-4，`check-exclusions.py` 只能查路径存在性） |
| 4 | 试用期 check 不阻塞合并 | 先积累信噪比，计划两周后把快速档 4 个转 required（按 CI Pipeline 下的新 check 路径名配分支保护） |
| 5 | base 白名单含试用分支 | `ci.yml` 与各 area 的 `push.branches` 均为 `[v0.23.0, feat/gha-ppu-test]`，合入 v0.23.0 后需摘除试用分支 |
| 6 | workflow_dispatch 仅默认分支可点 | 未合入默认分支前，网页 Run workflow 按钮不可用；开发期靠 push 触发绕过（build-ppu-wheel 即如此处理） |
| 7 | terratorch 依赖链断裂 | FlyTiger mirror 上 stringzilla 只剩壳包，models-basic 脚本已改为 terratorch 单独安装容错（`|| WARN` 继续）；red 用例用 `--deselect` 精确排除（8 条） |

---

## 8. 分支当前 git 状态

- HEAD：`d9389af55 fix(ppu):fix ppu build wheel`
- 领先 `v0.23.0` 共 10 个 commit，覆盖：11 个 test-area 落地 → nightly 定时 → ci.yml 统一编排（precheck → smoke/ai-review → 快速档 → ppu-full 标签 → human-review → 标签档）→ wheel 构建。
- 工作区（2026-09-08）：存在进行中的未提交修改（ci.yml、部分 test-area workflow、`ppu_install_dependency.sh` 等），以 `git status` 为准。

---

## 9. 文档导航（scripts/ppu/docs/）

| 文档 | 用途 |
| --- | --- |
| [ppu-per-pr-ci-guide.md](ppu-per-pr-ci-guide.md) | **开发者主文档**：触发方式、路径清单、打标签/审批操作、重跑、FAQ、权限 |
| [ppu-ci-usage.md](ppu-ci-usage.md) | 简版使用说明 + 红了怎么办的排查顺序 |
| [ppu-ci-exclusions.md](ppu-ci-exclusions.md) | 排除项台账：规模、待办（root cause 未定位项）、上游偏差、复查节奏 |
| [area-migration.md](area-migration.md) | Aone → GHA 迁移进度（11/29）、批次规划、模板化等优化方向 |
| [ppu-ci-improvements.md](ppu-ci-improvements.md) | 改进方案池（P1 体验 / P2 速度 / P3 对齐厂家实践），部分已实施 |
| [ppu-wheel-build.md](ppu-wheel-build.md) | wheel 构建设计决策与对 PR #6 的差异（staged 未 commit） |
| [first-run-report-batch1.md](first-run-report-batch1.md) / [PPU CI 新增 8 Area（第一批次） 首跑总结报告.md](<PPU CI 新增 8 Area（第一批次） 首跑总结报告.md>) | 批次 1 首跑数据与 red 用例处置记录 |
