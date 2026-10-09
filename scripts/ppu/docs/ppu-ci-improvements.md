# PPU CI Workflow 改进方案与台账

> 整理日期：2026-09-08。基于分支 `feat/gha-ppu-test`(HEAD `d9389af55`，基线 v0.23.0)。
> 参考对象：上游 vLLM CUDA CI(`.buildkite/test_areas/`)、AMD(`.buildkite/hardware_tests/amd.yaml`)、Ascend NPU(`.buildkite/hardware_tests/ascend_npu.yaml`)。
> 本文档滚动更新：已落地条目移入 §2，待办按优先级列在 §3。

---

## 1. 现状架构速览

单 runner 硬约束：PPU 整机仅一台 self-hosted runner，所有 PPU job 串行排队。
全部设计（分档触发、超时收紧、K8s 化探索）都围绕这个约束展开。

### 1.1 Workflow 全景

| 文件 | 角色 | 触发方式 |
| --- | --- | --- |
| `ci.yml` | PR 统一编排入口 | pull_request(opened/synchronize/reopened/ready_for_review/labeled) |
| `test-area-ppu-{attention,model-executor,entrypoints,samplers}.yml` | 快速档 ×4 | ci.yml `workflow_call`（路径过滤命中才跑）+ push + dispatch |
| `test-area-ppu-{basic-correctness,entrypoints-llm,lora,models-basic,engine,kernels}.yml` | 标签档 ×6 | ci.yml `workflow_call`（需 `ppu-full` 标签 + human-review)+ dispatch |
| `test-area-ppu-models-language.yml` | 定时档专属 | 仅 dispatch（nightly 由聚合 workflow 调度） |
| `nightly-ppu.yml` | 定时兜底回归 | cron：工作日 02:00 短集 / 周六 02:00 长集 + dispatch |
| `ppu-ci-selfcheck.yml` | 静态自检（排除项路径有效性） | push/PR(paths 过滤）+ dispatch，跑在云 runner |
| `build-ppu-wheel.yml` | 分支级 wheel 构建验证 | 窄路径 push + dispatch（未注册 main 前仅 push) |
| `ppu-action-smoke.yml` | K8s 调度链路冒烟 | dispatch + 自身路径 push |

### 1.2 PR 门禁链（ci.yml)

```text
PR → precheck(密钥扫描)→ smoke-test ∥ ai-code-review
     ├→ 快速档 ×4(无人工审批，全绿约 14min)
     └→ [ppu-full 标签] → human-review(真人 approve)→ 标签档 ×6(全开 5h+)
     ↓
     ci(Full CI 聚合点,skipped 视为通过)
```

concurrency 按 `ci-<PR号>-<push|label>` 分组：标签档 run 不取消快速档 run。

### 1.3 超时标定（按首跑实测收紧后的现状）

| Area | 档位 | job/step timeout | 首跑实测 |
| --- | --- | --- | --- |
| attention | 快速 | 45/40 | 1m23s |
| model-executor | 快速 | 45/40 | 1m49s |
| entrypoints | 快速 | 45/40 | 1m14s |
| samplers | 快速 | 45/40 | 9m05s |
| basic-correctness | 标签 | 100/90 | 未标定（沿用上限） |
| entrypoints-llm | 标签 | 60/50 | 11m34s |
| lora | 标签 | 100/90 | 未标定（沿用上限） |
| models-basic | 标签 | 120/110 | 48m07s |
| engine | 标签 | 300/290 | 1h32m08s |
| kernels | 标签 | 360/350 | 2h14m18s |
| models-language | 定时 | 300/290 | 未标定 |

---

## 2. 已落地改进（2026-08-26 上一版至今)

对照上一版的 P1/P2/P3 条目：

| 旧条目 | 状态 | 说明 |
| --- | --- | --- |
| P1-2 `workflow_dispatch` 调试输入 | ✅ 已落地 | 11 个 area workflow 均支持 `test_mode`(all/single/multi）与 `pytest_args`（透传 `PYTEST_EXTRA_ARGS`)，且 `workflow_call` 同样暴露这两个 inputs |
| P1-4 JUnit 注解 | ✅ 已落地 | `mikepenz/action-junit-report@v5`(`annotate_only`, `fail_on_failure: false`)，失败用例 inline 注解到 PR Files changed |
| P3-10 artifact 保留期 | ✅ 已落地 | `actions/upload-artifact@v4` 统一 `retention-days: 14` |
| P3-11 actionlint 进 pre-commit | ✅ 已落地 | actionlint + shellcheck 均在 `.pre-commit-config.yaml`;`runs-on` 的 `ppu` 自托管标签在 `.github/actionlint.yaml` 声明 |
| 超时收紧 | ✅ 已落地 | 全部 area 按首跑实测收紧（见 §1.3)，约 2~2.5 倍余量 |
| 并发组 | ✅ 已落地 | 每 area 独立 group(`test-area-ppu-<area>-<ref>`);nightly 用独立组 + matrix `max-parallel: 1`;models-language 用 job 级 group 避免与 nightly 互挤 |
| P3-7 soft_fail 语义 | ◐ 进行中 | 试用期内 `ppu-<area>-finish` / `Full CI` 均不设 required check，不阻塞合并；两周后评估快速档 4 个转 required（见 §3-P3) |

同期落地的其他结构性改进：

- **PR 入口收编 `ci.yml`**:10 个 PR area 的 `pull_request` 直触全部移除，统一走
  组织门禁链（precheck → smoke ∥ AI 评审）后以 `workflow_call` 调用；重量级 6 个
  需 `ppu-full` 标签 + 真人 approve。fork PR 被 `head.repo.full_name` 判定挡在
  self-hosted runner 外（特权容器 + 设备直通 + 凭证，不能被外部代码驱动）。
- **`nightly-ppu.yml` 定时兜底**：标签档 6 个 + models-language 按时长拆两档 cron
  （工作日短集 / 周六长集）,matrix `max-parallel: 1` 串行防抢卡。
- **`ppu-ci-selfcheck.yml` + `check-exclusions.py`**：静态校验 `--ignore`/`--deselect`
  引用的路径真实存在（失效 ignore 会被 pytest 静默吞掉），前置到云 runner,不占 PPU。
- **`build-ppu-wheel.yml` 云端 wheel 构建已跑通**(run 34109691435,2026-09-07,
  真机验证 2026-09-08)：云 runner 编译约 1.5h，产物发 GitHub prerelease;SM80-only、
  x86_64-only。详见 [`ppu-wheel-build.md`](ppu-wheel-build.md)。
- **观测日志**：标签档 area 的 `check-changes` 即使不跑也打印
  `[gate] paths_hit=... ppu_full_label=...`，为后续「哪些 area 值得提升快速档」积累数据。
- **首跑 triage 完成**:8 个新 area 首跑 red 全部处置（deselect/ignore/stub/step 停用
  + 注释标注），台账见 [`ppu-ci-exclusions.md`](ppu-ci-exclusions.md)；首跑报告见
  [`first-run-report-batch1.md`](first-run-report-batch1.md)。

---

## 3. 待办改进

### P0 — 结构性（不解决会持续放大维护成本）

#### 3.1 `workflow_call` 模板化 + 脚本公共段抽取（批次 2 的硬约束）

**现状**:11 份 workflow / 11 份脚本是同构拷贝，单脚本约 170 行
junit/summary/`_run_step` 样板完全重复（实测 attention 与 samplers 仅 22 行不同，
注释措辞已开始分叉）。批次 2/3 还有 18 个 area 待迁移，再添就是 29 份互不同步的拷贝。

**方案**（对位 PyTorch `_linux-test.yml` 模式）:

1. 抽一个 `test-area-ppu-template.yml` reusable workflow，入参：area 名、timeout、
   路径过滤清单、test_mode/pytest_args；各 area 文件只剩 `uses` + 参数。
2. 脚本侧抽 `scripts/ppu/lib/` 公共段（junit 发射、summary 生成、`_run_step`、
   docker run 参数串）,area 脚本只保留选集与 MODEL_MAP。
3. 注意 `ppu-<area>-finish` 聚合 check 名需保持唯一（分支保护按名区分）。

**前置联动**：模板化时把 §3.3(junit 假绿防护）一并做进公共段，只做一次。

#### 3.2 K8s 化（解除单 runner 串行瓶颈的唯一根本解)

**现状**:`ppu-action-smoke.yml` 已验证 `flytiger-eco/ppu-distributed-action` 链路
（ARC scale set `k8s-runner-group-cpu-flytiger` 起 job 容器 → action 向 `ppu-sched`
namespace 提交 Volcano job → worker Pod 占卡执行）。冒烟确认了设备插件注入、shm
emptyDir(64Gi 覆盖原 `--shm-size=8g`)、NAS 挂进 worker Pod 均可用。

**当前唯一卡点**：代码投递。job 容器（ARC 侧）无任何 NAS 挂载，三个候选路径
(`/wl_nas`、`/mnt/wl_nas`、`/nas_aisw`）实测全部不可写，源码送不进 worker Pod。
需集群 owner 在 `arc-runners` namespace 的 `hook-extension-cpu.yaml` 给 job 容器加
`hostPath: /wl_nas`(type 用 `Directory` 而非 `DirectoryOrCreate`,fail loud)。
备选通道（未验证）:Artifactory generic 仓中转源码 tar，或 VCS remote 仓代理 GitHub
按 ref 拉 tarball。

**落地后连锁变化**：单机串行前提消失，需要重新评估：并发组策略（从 GHA 层防抢卡
转为集群配额管理）、nightly 两档拆分是否还有必要、快速档是否扩编。

### P1 — 使用体验

#### 3.3 junit 空收集假绿防护

**问题**:pytest usage error（如路径不存在，rc=4）仍会写出 `tests="0" errors="0"`
的空 junit；当前各脚本的汇总逻辑仅凭 `failures==0 and errors==0` 判定，空收集会被
判成 PASSED（假绿）。v0.23 rebase 已实际踩过（entrypoints 的 `v1_entrypoints` step
引用已删目录）。

**方案**：汇总逻辑加 `tests==0 → FAILED(collected nothing)`；随 §3.1 公共段抽取
一次覆盖全部 area。

#### 3.4 失败 traceback 进 Summary（旧 P1-3，仍欠）

**问题**:`summary.md` 只有统计表；失败时要下载 artifact 翻日志才能看到报错堆栈。

**方案**：脚本 `_emit_junit` 的 EXIT trap 里，对 status=failed/error 的用例从原始
junit XML 提取 `system-out`/`message` 末尾约 30 行，以 `<details>` 折叠块追加进
`test-results/summary.md`（现有 Publish test summary 步骤无需改）。

**收益**:Summary 页直接看到失败堆栈，定位不离开浏览器。

#### 3.5 nightly 未标定 area 的 timeout 回写

`basic-correctness`(100/90)、`lora`(100/90)、`models-language`(300/290）三个
area 仍沿用首跑前的上限值。nightly 首次定时跑完后，按实测把
`nightly-ppu.yml` 的 select-areas 矩阵与各 workflow 的 timeout 一并收紧（对位
entrypoints-llm 11m34s→60/50、models-basic 48m07s→120/110 的既有做法）。

### P2 — 速度与资源

#### 3.6 pip 缓存挂 NAS（旧 P2-5，仍欠）

**问题**:`ppu_install_dependency.sh` 与脚本内 ray 安装每次全量走内网
artifactory(`--no-cache-dir` 仍在），分钟级开销；NAS 卷已挂载但未用于 pip 缓存。

**方案**:docker run 增加 `-e PIP_CACHE_DIR=/nas_aisw/pip_cache`，脚本内去掉
`--no-cache-dir`（或保持、由 env 优先生效，需实测）。多 area 共享同一份缓存。

**收益**：依赖安装从分钟级降到秒级。

#### 3.7 runner 磁盘卫生（旧 P2-6，仍欠）

**问题**:self-hosted 常驻 runner 镜像/容器残留累积,「no space left」是典型故障。

**方案**:Clean workspace 步骤追加：

```bash
docker system prune -f --filter "until=168h"   # 保留 7 天，不动在用基础镜像
df -h | tail -n +2
```

注：K8s 化（§3.2）落地后此条自然消解（Pod 一次性）。

#### 3.8 `csrc/**` 门禁恢复（前置已具备）

**现状**:kernels 路径清单刻意不含 `csrc/**`——`ppu_install_dependency.sh` 的
`[cext]` 段借用镜像预编译 `.so`，改 C++ 源码不会重编译，跑出的绿灯是假绿。
**前置已通**:`build-ppu-wheel.yml` 云端构建已验证（§2)，真机装 wheel + 跑
kernels 抽样 0 failed。

**剩余工作**:① `build-ppu-wheel.yml` 补 `workflow_call` 入口；② 决定 test-area
消费产物的方式（ci.yml 先调构建再调 kernels area，或 kernels 脚本优先装最新
prerelease wheel);③ 恢复 paths-filter 的 `csrc/**`。

### P3 — 治理与锦上添花

#### 3.9 快速档转 required check（试用期收口）

试用期（2026-09-03 ci.yml 上线起）所有 PPU check 不阻塞合并，积累信噪比数据。
两周后评估：快速档 4 个 `ppu-<area>-finish` 转 required。前提：排除项台账
[`ppu-ci-exclusions.md`](ppu-ci-exclusions.md) 第 2 节的未定位 red(engine 17 条、
kernels 3 文件等）至少完成 root cause 定位，否则信噪比无法评估。

#### 3.10 失败诊断收集

```yaml
- name: Collect diagnostics on failure
  if: failure()
  run: |
    docker run --rm --privileged --device=/dev/alixpu_ctl --device=/dev/alixpu \
      ${{ env.PPU_BASE_IMAGE }} \
      bash -c "python -m vllm.collect_env; dmesg | tail -50" || true
```

事后定位不用复现环境。

#### 3.11 flaky 重试

`nick-fields/retry` 或 pytest-rerunfailures 单次重试；硬件 CI 抖动时降低假红。
待试用期信噪比数据出来后再决定是否需要。

#### 3.12 上游选集漂移检测

当前选集是从 `.buildkite/test_areas/<area>.yaml` 手抄的快照，上游 rebase 后无同步
机制（Aone 侧靠 `aone_ci/pipeline_generator` 生成）。`check-exclusions.py` 只能查出
「路径已不存在」，查不出「上游改了排除列表」（已知偏差：台账第 3 节 D-1~D-4)。
方案方向：定期 diff 上游 test_areas yaml 与本地选集，做成 selfcheck 的一个新检查项。

#### 3.13 README CI badge

```markdown
![CI Pipeline](https://github.com/flytiger-eco/vllm-for-sail/actions/workflows/ci.yml/badge.svg?branch=feat/gha-ppu-test)
```

#### 3.14 wheel 构建链路收尾（来自 [`ppu-wheel-build.md`](ppu-wheel-build.md) §8)

- 收紧 480min timeout（实测 1.5h)；清理无效 `MAX_JOBS`/`NVCC_THREADS` env 与
  云 runner 上的 Clean workspace 空操作；每项单独改、单独跑，便于归因。
- Node.js 20 弃用告警：checkout@v4 等被平台强制跑 Node 24，适时升级 action 大版本。
- nightly 构建流水线（PR #6）合入 main 后评估脚本收敛，避免双份漂移。
- terratorch 恢复跟踪：mirror 上 stringzilla 只剩 FlyTiger 壳包（无官方 wheel),
  当前 terratorch 安装走容错分支；FlyTiger 补发产物后恢复并复测
  `test_registry_imports[PrithviGeoSpatialMAE]/[Terratorch]` 两个 deselect。

---

## 4. 已排除项（评估过、确认不做）

- ~~并发组统一 `ppu-device-${{ github.ref }}`~~:2026-08-26 确认当前 PPU 机器整机
  只部署一个 self-hosted runner，单 runner 串行执行 job，各 area 独立 concurrency
  group 即为正确配置，不存在抢卡/虚假 OOM 场景。models-language 与 nightly 不共享组
  （共享组容量 1 会互挤排队任务）。**注意**：若 K8s 化（§3.2）落地或机器增加
  runner，此前提失效，需重新评估。
- ~~镜像与测试分离/内容哈希跳过构建（AMD ci_base 模式）~~：用厂商预置镜像，暂无
  自建镜像管线；wheel 构建链路（§2）已覆盖「编译产物可分发」的核心诉求，pip 缓存
  (§3.6）可进一步缓解依赖安装开销。

---

## 5. 厂家 CI 模式对照

| 实践 | 上游/厂家 | PPU 现状 | 对应条目 |
| --- | --- | --- | --- |
| 路径门禁 | Buildkite `source_file_dependencies` | ✅ 已有（dorny/paths-filter，快速档自动、标签档观测） | — |
| PR 统一编排 + 组织门禁链 | —（组织自定义） | ✅ ci.yml(precheck → smoke ∥ AI 评审 → 分层触发） | — |
| 定时兜底回归 | Buildkite nightly | ✅ nightly-ppu.yml 两档 cron | §3.5 回写 timeout |
| 镜像与测试分离/内容哈希 | AMD ci_base 哈希 | ◐ 厂商预置镜像 + wheel 构建已通 | §3.8 恢复 csrc 门禁 |
| 硬件测试 soft_fail | Ascend `soft_fail: true` | ◐ 试用期不阻塞，聚合点已具备 | §3.9 转 required |
| 超时收紧 | CUDA area 普遍 30min 级（H200) | ✅ 已按实测收紧（§1.3) | §3.5 三个未标定 |
| junit 自动注解 | Buildkite 内建 | ✅ mikepenz/action-junit-report | §3.4 traceback 进 Summary |
| 空 junit 假绿防护 | — | ❌ 未做（failures==0 即过） | §3.3 |
| 跨 workflow 设备串行 | —（各硬件独立 agent) | ✅ 单 runner 天然串行 | §3.2 K8s 化后重估 |
| flaky 重试 | 各厂常见 | ❌ 未做 | §3.11 |

## 6. 关联文档

- [`ppu-ci-usage.md`](ppu-ci-usage.md) / [`ppu-per-pr-ci-guide.md`](ppu-per-pr-ci-guide.md)：开发者使用说明（触发档位、看结果、排障）
- [`area-migration.md`](area-migration.md):29 个 area 的迁移进度与批次规划
- [`ppu-ci-exclusions.md`](ppu-ci-exclusions.md)：排除项台账（待办 red、上游偏差、恢复条件）
- [`first-run-report-batch1.md`](first-run-report-batch1.md)：批次 1 首跑报告与 triage 记录
- [`ppu-wheel-build.md`](ppu-wheel-build.md):wheel 构建链路与已知限制
