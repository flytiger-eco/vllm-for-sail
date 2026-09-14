# PPU CI 现状总结（feat/gha-ppu-test 分支）

> 整理日期：2026-09-11。分支：`feat/gha-ppu-test`(基线 `v0.23.0`)。HEAD：
> `f851df2e7 ci(ppu): 普通 PR 免标签跑全量门禁 + 新增 skip-human-review / run-all 标签`。
> 面向读者：需要快速了解本分支 PPU CI 全貌的维护者与开发同学。
> 各子主题的深入文档见文末「文档导航」。

---

## 1. 一句话总览

本分支在 vLLM v0.23.0 基线上搭建了一套完整的 PPU GitHub Actions CI，并已完成两次关键演进：**运行底座从「单台 self-hosted 整机串行」迁到「K8s worker Pod 按卡隔离、area 间可并行」**，**PR 门禁从「三档触发(快速/标签/nightly)」收敛为「一条链、22 个 area 全部按路径变更自动触发」**，且 **wheel 构建已并入门禁链**——改到 `csrc/` 会真正重编译，堵住了此前最大的假绿盲区。

```text
PR 事件(opened / synchronize / reopened / ready_for_review / labeled)
                ↓
  precheck(TruffleHog 密钥扫描,组织 reusable workflow)
                ↓
  smoke-test(PPU 静态冒烟) ∥ ai-code-review(Copilot Gate) ∥ build-wheel(编 PPU wheel)
                ↓
  human-review(真人审批) ── 或 ──> human-review-waived(skip-human-review 标签豁免通道)
                ↓
  PPU area × 22(wheel 就绪 + 门禁任一通道 success 后统一派发)
     每个 area 在自己的 check-changes 里按路径 diff 决定是否真正上卡;
     路径未命中则 skip 不占资源;PR 带 run-all 标签则 22 个全部强制跑。
                ↓
  Full CI 聚合点(skipped 视为通过,供分支保护配单条 required check)
```

---

## 2. 运行底座：从单机串行到 K8s 按卡隔离(一切设计的出发点)

早期版本所有 PPU job 串行排在**一台** self-hosted 整机上,重量级 area 因此不能自动跑。现已迁到 K8s：

- **编排壳 CPU-only、零占卡**：每个 area 的测试 job `runs-on: k8s-runner-group-cpu-flytiger`,容器 `ghcr.io/actions/actions-runner:latest`,只负责打包源码、调度 worker Pod、回读 summary,不碰 PPU 卡。
- **测试真正跑在 worker Pod**：经 `flytiger-eco/ppu-distributed-action@main` 起一个 PPU Pod(`node_selector: board-type=OAM-810E`、`namespace: ppu-sched`),在 Pod 内 `ppu_install_dependency.sh` 装依赖 → `test-area-ppu-<area>.sh` 跑 pytest。
- **按卡隔离,area 间可并行**：worker Pod **刻意不开 privileged**——特权容器会拿到宿主机全量 `/dev`(实测 16 张卡全暴露),绕过 PPU device plugin。unprivileged 下 device plugin 只注入分配到的 `nproc_per_node` 张卡并**重编号为 `0..N-1`**,正合各 area 脚本 `CUDA_VISIBLE_DEVICES=0..N-1` 的相对语义。K8s extended resource(`alibabacloud.com/ppu`)记账不超卖,超出节点容量的 Pod 自动 Pending 排队。这是「PR 门禁 22 个 area 能并行派发」的前提。
- **容器参数对位**：`host_ipc: true` + `shm_size: "8Gi"` 对位 docker 版的 `--ipc=host --shm-size=8g`;`--ulimit memlock=-1` 无 K8s 等价物,如遇 pinned-memory 类失败需集群侧放宽 `LimitMEMLOCK`。`/nas_aisw`(模型/数据集,`HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub`,离线)与 `/mnt/wl_nas`(summary 回传)由 action 默认挂载。
- **统一基础镜像**：`pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-vllm0.23.0-py312`(各 workflow 的 `env.PPU_BASE_IMAGE`)。
- **fork PR 一律不进 PPU worker Pod**(设备 + NAS + artifactory 凭证,不能被外部代码驱动);fork PR 上门禁链照跑、PPU job 跳过(各 area 的 `head.repo.full_name == github.repository` 判定)。

> 迁移背景与 privileged 对照实测详见 [ppu-k8s.md](ppu-k8s.md)。

---

## 3. Workflow 文件清单(.github/workflows/)

**编排 / 平台层(4 个)**

| 文件 | 作用 | 触发方式 |
| --- | --- | --- |
| `ci.yml` | PR 统一编排入口：组织门禁链 + wheel 构建 + 22 个 area 的 workflow_call + Full CI 聚合 | pull_request(base 白名单 `v0.23.0` / `feat/gha-ppu-test`) |
| `nightly-ppu.yml` | 重量级 + nightly-only area 的定时兜底回归 | schedule(UTC 18:00,工作日/周末两档)/ dispatch(area_set) |
| `build-ppu-wheel.yml` | 编译 PPU wheel,产出 artifact 供 22 个 area 共用 | workflow_call(被 ci.yml 复用)/ dispatch |
| `ppu-ci-selfcheck.yml` | 静态自检：`check-exclusions.py` 校验排除项路径存在 | push/PR(paths 过滤)/ dispatch |

**test-area 层(26 个 `test-area-ppu-*.yml`)**

- **PR 门禁编排的 22 个**(ci.yml 逐个 workflow_call)：`attention`、`model-executor`、`entrypoints`、`samplers`、`basic-correctness`、`entrypoints-llm`、`lora`、`models-basic`、`engine`、`kernels`、`benchmarks`、`compile`、`cuda`、`distributed`、`e2e-integration`、`expert-parallelism`、`lm-eval`、`misc`、`model-runner-v2`、`pytorch`、`quantization`、`weight-loading`。
- **仅 nightly 兜底、不进 PR 门禁的 2 个**：`models-distributed`、`models-multimodal`(重量级回归)。
- **nightly + dispatch 的 1 个**：`models-language`(语言模型族,PR 不跑)。
- **仅 workflow_dispatch 手动入口的 1 个**：`spec-decode`(依赖模型未入库,见该文件头注释)。

> 组织门禁三段(`precheck` / `ai-code-review` / `human-review`)是 `flytiger-eco/.github` 仓库的 reusable workflow,本仓库只做 `uses` + `needs` 编排。

---

## 4. PR 门禁编排(ci.yml)

### 4.1 五个 Stage

1. **precheck** — 组织级密钥扫描。
2. **smoke-test ∥ ai-code-review ∥ build-wheel** 并行：
   - smoke-test：`bash -n` 全部 `scripts/ppu/*.sh` + `py_compile` + `check-exclusions.py`(每个 PR 都跑,与 selfcheck 的 paths 过滤互补)。
   - ai-code-review：组织 Copilot Gate。
   - build-wheel：workflow_call 复用 `build-ppu-wheel.yml`,产出 wheel artifact。
3. **human-review**(真人审批)**或** **human-review-waived**(豁免通道,见 4.3)。
4. **PPU area × 22**：放行条件 = `build-wheel success` **且** 两条审批通道任一 `success`(用 `always()` 兜住另一条必然 skipped 的级联跳过)。每个 area 内部的 `check-changes` 再按路径 diff 决定是否真正上卡。
5. **Full CI 聚合点**：`needs` 全部 job,`skipped` 视为通过——分支保护只需配这一条 required check。

### 4.2 路径驱动,不再分档

不再有「快速档 / 标签档」之分：**普通 PR 无需任何标签即进完整门禁链**,22 个 area 全部被派发,各自的 `check-changes` 用 `dorny/paths-filter` 按路径 diff 决定 `changes_exist`。未命中路径的 area 直接 skip、不占卡;命中的才起 worker Pod。观测日志会打印 `[gate] paths_hit=...`。

### 4.3 两个标签

| 标签 | 作用 |
| --- | --- |
| `skip-human-review` | 跳过真人审批门禁,改走 `human-review-waived` 本地豁免 job 放行。**背景**：GitHub 禁止 PR 作者 approve 自己的 PR,而本仓库日常只有作者一人提交,轮询门禁会空等 6h 后 failure、22 个 area 永不派发;打标签需 triage 权限、在时间线留痕可审计,授权强度不弱于一次 approve。 |
| `run-all` | 无条件执行全部 22 个 area(不跳过 precheck/review 等前置门禁)。各 area 的 `check-changes` 见此标签即强制 `changes_exist=true`。 |

### 4.4 单组并发互斥

`concurrency.group` 单组,`cancel-in-progress: true`——同一 PR 任何时刻只有一个 CI run 覆盖全部 22 个 area。`labeled` 事件只有 `skip-human-review` / `run-all` 两个「会改变编排」的标签归 `main` 组,其余无关标签归 `noop` 组,避免打无关标签重开整条门禁链。

---

## 5. test-area 标准结构(K8s 三段式,26 个 area 同构)

每个 area = 1 个 workflow(`.github/workflows/test-area-ppu-<area>.yml`)+ 1 个选集脚本(`scripts/ppu/test-area-ppu-<area>.sh`)。

```text
check-changes(ubuntu-latest,路径 / run-all 标签门禁)
    ↓
ppu-<area>-test(编排壳 k8s-runner-group-cpu-flytiger,CPU-only)
  checkout → (可选)download wheel artifact →
  Build in-pod payload(把 install+test 脚本 base64 打包,规避多行 command 插值报错)→
  ppu-distributed-action 起 PPU worker Pod：ppu_install_dependency.sh → test-area-ppu-<area>.sh
  → Publish test summary(经 NAS 回读 per-unit 统计表,回退作业元信息表)
    ↓
ppu-<area>-finish(聚合点,skipped 视为通过)
```

- **wheel 消费**：ci.yml 把 build-wheel 的 artifact 名经 `wheel_artifact` 传入;area 侧 `download-artifact` 后设 `PPU_WHEEL_PATH`,脚本优先装该 wheel,空则回退镜像预装 vllm。
- **调试输入**：`workflow_dispatch` / `workflow_call` 均支持 `test_mode`(all/single/multi)与 `pytest_args`(透传 `PYTEST_EXTRA_ARGS`,如 `-k test_foo -x`),把小时级 area 压到分钟级,是排障主要手段。
- **结果出口**：Job Summary(tests/passed/failed/errors/skipped/time)+ artifact + Pod 日志(action 全量透传 stdout)。summary 经 `/mnt/wl_nas` → runner 侧 `/wl_nas` 同一 NAS 回传,路径按 `run_id-attempt/job/(area)` 分层隔离。
- `push` 触发保留(`branches: [v0.23.0, feat/gha-ppu-test]` + paths),供开发期分支验证;PR 入口已收编到 ci.yml,各 area 文件不再直接接受 `pull_request`。

### 执行脚本侧

- `ppu_install_dependency.sh`：`[diag]` 环境盘点 → `[deps]` PPU 依赖(镜像预装优先,缺失走内网 artifactory)→ pytest 依赖 → `[cext]` 借用/安装 C 扩展 `.so`(wheel 就绪后走真编译产物)。
- 选集脚本内含 `--ignore` / `--deselect` / `-k` 排除项(台账见 [ppu-ci-exclusions.md](ppu-ci-exclusions.md)),并有 `tests==0` 空收集视为配置错误的假绿防护。
- `check-exclusions.py`：静态校验排除项引用路径真实存在(失效 `--ignore` 会被 pytest 静默吞掉),由 selfcheck 与 ci.yml 的 smoke-test 双路执行。
- `model_alises/`：模型/adapter 清单,供脚本离线解析模型路径。

---

## 6. 定时档(nightly-ppu.yml)

- **工作日 02:00(北京)**短集：`basic-correctness` / `entrypoints-llm` / `lora` / `models-basic`。
- **周六 02:00(北京)**长集：`engine` / `kernels` / `models-language` + 其余未标定 area(含 nightly-only 的 `models-distributed` / `models-multimodal`;`spec-decode` 为 dispatch-only 不在此列)。
- 拆两档原因：area 需串行,全量连跑会占到工作时段、堵住白天 PR。
- **串行原因已从「抢卡」变为「NAS 暂存目录冲突」**：K8s 按卡隔离后卡已真隔离,但 `ppu-distributed-action` 的 NAS 暂存路径只含 job key、matrix 各 leg 共享,并行时 `tar` 会读到被并发改写的文件而报 `file changed as we read it`。故 workflow 级 concurrency(排队不取消)+ matrix `max-parallel: 1`(`fail-fast: false`,单 area 失败不影响其余排队)。各独立 test-area workflow 是不同 job key,不受此限,可并行。
- `workflow_dispatch` 支持 `area_set = auto / nightly / weekend / all`;每个 area 携带标定的 `nproc` / `timeout` / `run_timeout`。

---

## 7. wheel 构建(build-ppu-wheel.yml)

- **定位**：PR 门禁链内的构建入口,被 ci.yml 经 workflow_call 复用,产出 wheel artifact 供 22 个 area 共用;脚本随分支携带(`scripts/ppu/build-ppu-wheel.sh`)。
- **matrix**：py3.12 / cuda13.0 / x86_64(aarch64 无 base image 与 SDK)。
- **版本号** `0.23.0.dev<yyyymmdd>+g<hash>`,经 workflow → 脚本 → docker `-e` → setup.py 四层传递,构建后对 wheel 文件名硬校验(任一环节断掉会静默产出 setuptools_scm 推导版本)。
- **战略意义**：接上 wheel 构建后,改到 `csrc/` 会真正重编译,**解除了此前「借用镜像预编译 `.so` → csrc 改动假绿」这一最大盲区**。

> 设计决策与对 PR #6 的差异见 [ppu-wheel-build.md](ppu-wheel-build.md)、[ppu-wheel-ci-integration.md](ppu-wheel-ci-integration.md)。

---

## 8. 已知限制与待办

| # | 事项 | 现状 |
| --- | --- | --- |
| 1 | **workflow / 脚本模板化未做** | 26 份 workflow 与脚本大量同构拷贝(K8s payload 打包、summary 回读、聚合点样板重复),是继续扩量与维护的主要成本项 |
| 2 | **排除项台账待收敛** | 仍有未定位 root cause 的排除项,且上游选集漂移无检测机制(`check-exclusions.py` 只能查路径存在性)。详见 [ppu-ci-exclusions.md](ppu-ci-exclusions.md) |
| 3 | **多数新增 area 时长未标定** | nightly 里批次 2 的 area 暂留保守 timeout,首次定时跑完后需按实测回写 `nightly-ppu.yml` |
| 4 | **base 白名单含试用分支** | ci.yml 与各 area 的 `push.branches` / base 白名单均含 `feat/gha-ppu-test`,合入 v0.23.0 后需摘除 |
| 5 | **memlock 无 K8s 等价** | unprivileged worker Pod 无法自行提升 `LimitMEMLOCK`,如遇 pinned-memory 类失败需集群侧放宽 |
| 6 | **spec-decode 未入门禁** | 依赖模型未入库,仅 workflow_dispatch 手动入口 |
| 7 | **required check 待切换** | 试用期先积累信噪比,稳定后把分支保护指向 `Full CI` 单条 required |

---

## 9. 文档导航(scripts/ppu/docs/)

| 文档 | 用途 |
| --- | --- |
| [ppu-per-pr-ci-guide.md](ppu-per-pr-ci-guide.md) | **开发者主文档**：触发方式、路径清单、打标签/审批操作、重跑、FAQ、权限 |
| [ppu-ci-usage.md](ppu-ci-usage.md) | 简版使用说明 + 红了怎么办的排查顺序 |
| [ppu-k8s.md](ppu-k8s.md) | K8s worker Pod 迁移：编排壳/Pod 分层、按卡隔离、privileged 对照实测 |
| [ppu-ci-exclusions.md](ppu-ci-exclusions.md) | 排除项台账：规模、待办、上游偏差、复查节奏 |
| [area-migration.md](area-migration.md) | Aone → GHA 迁移进度、批次规划、模板化等优化方向 |
| [ppu-ci-improvements.md](ppu-ci-improvements.md) | 改进方案池(体验 / 速度 / 对齐厂家实践) |
| [ppu-wheel-build.md](ppu-wheel-build.md) / [ppu-wheel-ci-integration.md](ppu-wheel-ci-integration.md) | wheel 构建设计决策、与 PR #6 的差异、并入门禁链的方式 |
| [first-run-report-batch1.md](first-run-report-batch1.md) / [PPU CI 新增 8 Area(第一批次) 首跑总结报告.md](<PPU CI 新增 8 Area（第一批次） 首跑总结报告.md>) | 批次 1 首跑数据与 red 用例处置记录 |
