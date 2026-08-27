# PPU CI Workflow 改进方案（test-area-ppu-basic-correctness.yml / test-area-ppu-lora.yml)

> 整理日期：2026-08-26。基于分支 `feat/gha-ppu-test`（基线 v0.23.0)。
> 参考对象：上游 vLLM CUDA CI(`.buildkite/test_areas/`)、AMD(`.buildkite/hardware_tests/amd.yaml`)、Ascend NPU(`.buildkite/hardware_tests/ascend_npu.yaml`)。
> 状态：**待评审，未实施**。本文档仅为方案存储。

## 0. 现状基线

当前 `test-area-ppu-basic-correctness.yml` 结构：

- `check-changes`：路径变更门禁（push 走 `on.push.paths` 触发级过滤，PR 走 dorny/paths-filter)
- `ppu-basic-correctness-test`:Clean workspace(docker root 清理)→ checkout@v4 → docker 内装依赖 + 跑 pytest → Publish test summary → Upload artifact
- `ppu-test-finish`：分支保护聚合点

最近一次 CI 实测（2026-08-26 前）:

| unit | tests | passed | skipped | time |
|---|---|---|---|---|
| test_mem | 4 | 4 | 0 | 329s |
| test_basic_correctness | 29 | 17 | 12 | 1265s |
| test_basic_correctness_distributed | 11 | 11 | 0 | 626s |
| **Total** | 44 | 32 | 12 | **2220s (~37min)** |

已完成调整：timeout 由 job 300 / step 290 收紧为 **job 100 / step 90**(约 2.4 倍余量，两个 workflow 同步）。

---

## 已排除项

- ~~并发组统一 `ppu-device-${{ github.ref }}`~~:2026-08-26 确认当前 PPU
  机器整机只部署一个 self-hosted runner，单 runner 串行执行 job，两个 area
  workflow 不会并发运行，不存在抢卡/虚假 OOM 场景，各自独立的
  concurrency group 即为正确配置，无需共享组。若未来机器上增加 runner,
  再重新评估。

---

## P1 — 使用体验（收益最大）

### 2. `workflow_dispatch` 加调试输入

**问题**：调试单个失败用例必须跑全量 37 分钟；脚本侧已支持 `TEST_MODE`
(single/multi/all)，但 workflow 未暴露。

**方案**（两个 workflow 同样适用）:

```yaml
on:
  workflow_dispatch:
    inputs:
      test_mode:
        description: "测试模式"
        type: choice
        options: [all, single, multi]
        default: all
      pytest_args:
        description: "额外 pytest 参数，如 -k test_basic_cumem"
        required: false
        default: ""
```

Run step 透传：

```yaml
-e TEST_MODE="${{ github.event.inputs.test_mode || 'all' }}" \
-e PYTEST_EXTRA_ARGS="${{ github.event.inputs.pytest_args }}" \
```

脚本侧在 pytest 命令尾部追加 `${PYTEST_EXTRA_ARGS:-}`。

**预期收益**：手动触发可 `-k` 精确到单用例，调试反馈从 37min 降到分钟级。

### 3. 失败 traceback 进 Summary

**问题**：summary.md 只有统计表；失败时要下载 artifact 翻日志才能看到报错。

**方案**：在脚本 `_emit_junit` 的 EXIT trap 里，对 status=failed/error 的用例，
从原始 junit XML 提取 `system-out` / `message` 末尾 ~30 行，以 `<details>` 折叠块
追加进 `test-results/summary.md`（现有 Publish test summary 步骤无需改）。

**预期收益**:Summary 页直接看到失败堆栈，定位不离开浏览器。

### 4. JUnit 注解（对齐 Buildkite 自动吃 junit 的体验）

**方案**：利用现有 `test-results/test.xml`，加一步：

```yaml
- name: Annotate test failures
  if: always()
  uses: mikepenz/action-junit-report@v5
  with:
    report_paths: "test-results/test.xml"
    check_name: "PPU Test Results"
```

**预期收益**：失败用例以 inline 注解形式出现在 PR Files changed / Checks 页。

---

## P2 — 速度

### 5. pip 缓存挂 NAS

**问题**:`ppu_install_dependency.sh` 与脚本内 ray 安装每次全量走内网
artifactory，分钟级开销；NAS 卷已挂载但未用于 pip 缓存。

**方案**:docker run 增加：

```yaml
-e PIP_CACHE_DIR=/nas_aisw/pip_cache \
```

脚本内 `pip install` 去掉 `--no-cache-dir`（或保持但 env 优先生效，需实测）。

**预期收益**：依赖安装从分钟级降到秒级；多 area 共享同一份缓存。

### 6. runner 磁盘卫生

**问题**:self-hosted 常驻 runner 镜像/容器残留累积，「no space left」是典型故障。

**方案**:Clean workspace 步骤追加：

```bash
docker system prune -f --filter "until=168h"   # 保留 7 天，不动在用基础镜像
df -h | tail -n +2
```

**预期收益**：防磁盘打满导致的诡异失败；日志里可见水位。

---

## P3 — 对齐厂家实践 / 锦上添花

### 7. soft_fail 语义（参考 Ascend NPU)

Ascend 在 Buildkite 用 `soft_fail: true`：硬件 CI 跑但不阻塞主干。GHA 等价做法：
`ppu-test-finish` **不设为 required check**（分支保护策略层面，非代码改动）。
现状聚合点已具备，仅是保护规则配置问题。

### 8. README CI badge

```markdown
![Basic Correctness (PPU)](https://github.com/<org>/<repo>/actions/workflows/test-area-ppu-basic-correctness.yml/badge.svg?branch=feat/gha-ppu-test)
```

### 9. 失败诊断收集

```yaml
- name: Collect diagnostics on failure
  if: failure()
  run: |
    docker run --rm --privileged --device=/dev/alixpu_ctl --device=/dev/alixpu \
      ${{ env.PPU_BASE_IMAGE }} \
      bash -c "python -m vllm.collect_env; dmesg | tail -50" || true
```

事后定位不用复现环境。

### 10. artifact 保留期

`actions/upload-artifact@v4` 加 `retention-days: 14`，控制存储。

### 11. actionlint 进 pre-commit

静态检查 workflow 语法与常见错误（如 2026-08-26 发现的 `branches: [...]。`
中文句号混入导致的 YAML 语法错误，可在推送前拦截）。

---

## 附：厂家 CI 模式对照

| 实践 | 上游/厂家 | PPU 现状 | 对应条目 |
|---|---|---|---|
| 路径门禁 | Buildkite `source_file_dependencies` | 已有（on.push.paths + dorny) | — |
| 镜像与测试分离/内容哈希跳过构建 | AMD ci_base 哈希 | 用厂商预置镜像，暂无自建 | P2-5 可缓解 |
| 硬件测试 soft_fail | Ascend `soft_fail: true` | ppu-test-finish 聚合点已有 | P3-7 |
| 超时收紧 | CUDA area 普遍 30min 级（H200) | 已收紧至 90/100 | 已完成 |
| junit 自动注解 | Buildkite 内建 | 仅有统计表 | P1-4 |
| 跨 workflow 设备串行 | —（上游各硬件独立 agent) | 单 runner 天然串行，无需共享组 | 已排除 |

## 附：相关工作区未提交状态（2026-08-26)

- `tests/utils.py`、`tests/basic_correctness/test_mem.py`：已还原至 v0.23.0
  零 diff(staged，未 commit)。注意还原后 PPU 上 fork 子进程问题
  ("all HGGC-capable devices are busy or unavailable"）会复发，受影响的是
  test_mem.py 的 4 个用例，需在脚本 deselect 层面处置。
- 两个 workflow:timeout 已改 100/90；basic-correctness 注释精简中混入的
  `。` 语法错误已修复。
