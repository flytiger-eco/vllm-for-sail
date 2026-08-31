# PPU CI 使用说明

面向在 `v0.23.0` 上提 PR 的开发同学。一句话概括：**PR 上默认只跑 4 个轻量
area（约 14 分钟），重的测试需要自己打标签或等夜间定时跑。**

---

## 1. 什么时候会跑、跑什么

PPU 整机只有**一台** self-hosted runner，所有 job 串行排队，所以测试分三档。

### 快速档 —— PR 自动触发

命中对应路径就自动跑，无需任何操作。

| Area | 触发路径（除 CI 自身文件） | 首跑实测 |
| --- | --- | --- |
| `attention` | `tests/v1/attention/**`、`vllm/v1/attention/**` | 1m23s |
| `model-executor` | `tests/model_executor/**`、`vllm/model_executor/**` | 1m49s |
| `entrypoints` | `tests/entrypoints/**`、`tests/v1/entrypoints/**`、`vllm/entrypoints/**` | 1m14s |
| `samplers` | `tests/samplers/**`、`tests/conftest.py`、`vllm/v1/sample/**`、`vllm/model_executor/layers/**` | 9m05s |

四个加起来约 14 分钟（串行）。精确路径清单见各 workflow 的 `on.push.paths`。

### 标签档 —— PR 上打 `ppu-full` 才跑

在 PR 上加 `ppu-full` 标签，以下 6 个 area 会依次排队执行；标签是打上去就触发
（`labeled` 事件），不需要再 push 空提交。

| Area | 首跑实测 |
| --- | --- |
| `basic-correctness` | 未标定 |
| `entrypoints-llm` | 11m34s |
| `lora` | 未标定 |
| `models-basic` | 48m07s |
| `engine` | 1h32m08s |
| `kernels` | 2h14m18s |

为什么不自动跑：全打开需要 5 小时以上串行占用唯一一台机器，白天会把所有人的
PR 都堵住。改动触及 kernel、engine、模型加载这类底层逻辑时，请自觉打标签。

各 area 的依赖路径清单仍保留在 workflow 的 `check-changes` 步骤里，日志中会打印
`[gate] paths_hit=... ppu_full_label=...`，即使没跑也能看到「本次改动是否命中了
该 area 的依赖路径」。这批数据会用于后续决定哪些 area 值得提升到快速档。

### 定时档 —— 无需操作

`nightly-ppu.yml` 把标签档的 7 个 area（上表 6 个 + `models-language`）定时补跑：

- 工作日 02:00（北京）：`basic-correctness` / `entrypoints-llm` / `lora` / `models-basic`
- 周六 02:00（北京）：`engine` / `kernels` / `models-language`

即使没人打标签，也有每日/每周的回归基线。

### 不在门禁范围内的改动

**改 `csrc/` 下的 C/C++ 代码，PPU CI 不会验证。** 当前 `ppu_install_dependency.sh`
借用镜像里预编译好的 `.so`，源码改动不会被重新编译，跑出来的绿灯是假绿。因此
`kernels` 的路径清单刻意不含 `csrc/**`。接上 PPU wheel 构建后会恢复。

---

## 2. 怎么看结果

1. **Job Summary**：每个 area 的 job 页面顶部有一张 markdown 表，按 step/shard 列出
   tests / passed / failed / errors / skipped / time 与总判定，先看这张表。
2. **PR 注解**：失败用例会以注解形式标在 PR 的 Files changed 上（junit 报告解析），
   不用翻日志就能看到是哪个 case 挂了。
3. **Artifact**：job 页面下载 `ppu-<area>-test-results`，含 `test.xml`（合并后的
   junit）、`summary.md` 与各分片的 pytest 输出日志，保留 14 天。
4. **门禁状态**：每个 area 有独立的聚合 check（`ppu-<area>-finish`）。试用期这些
   check **不阻塞合并**，先积累信噪比数据；两周后把快速档的 4 个转为 required。

---

## 3. 红了怎么办

按顺序排查：

1. **先确认是不是已知问题**：查 [`ppu-ci-exclusions.md`](ppu-ci-exclusions.md) 的
   台账和对应 `scripts/ppu/test-area-ppu-<area>.sh` 里的 triage 注释。首跑遗留的
   red 用例（engine 17 条、kernels 3 个文件等）都记在那里。
2. **缩小范围重跑**：走 workflow_dispatch 手动触发（Actions → 选对应 workflow →
   Run workflow），两个输入：
   - `test_mode`：`all`（默认）/ `single` / `multi`，只跑单卡段或多卡段
   - `pytest_args`：追加给 pytest 的参数，例如 `-k test_foo -x`
     ——把 2 小时的 area 压到分钟级，是排障的主要手段。
3. **本地复现**（需要 PPU 机器）：

    ```bash
    docker run --rm -it --privileged --network host \
      --device=/dev/alixpu_ctl --device=/dev/alixpu \
      --ipc=host --shm-size=8g --ulimit memlock=-1 \
      -v "$PWD":/workspace -v /nas_aisw:/nas_aisw \
      -e HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub \
      -e PPU_DEVICE_LABEL=OAM-810E \
      -e PYTEST_EXTRA_ARGS="-k test_foo" \
      <PPU_BASE_IMAGE> bash
    # 容器内：
    cd /workspace
    bash scripts/ppu/ppu_install_dependency.sh
    bash scripts/ppu/test-area-ppu-<area>.sh
    ```

    镜像 tag 见任一 workflow 的 `env.PPU_BASE_IMAGE`。

4. **确认是平台问题而非自己的改动**：翻同一 area 最近一次定时跑的结论；若定时跑
   也是同样的红，属既有问题，请在台账里补一条而不是临时加 `--ignore` 绕过。

---

## 4. 改 CI 本身要注意的

- workflow 与脚本都受 `pre-commit` 门禁（actionlint + shellcheck），提交前跑
  `pre-commit run --all-files` 或 `pre-commit run actionlint --all-files`。
- `runs-on: [self-hosted, ppu]` 的 `ppu` 标签在 `.github/actionlint.yaml` 里声明，
  新增自托管标签要同步加进去，否则 actionlint 报 `label is unknown`。
- 排除项引用的测试路径必须真实存在：`PPU CI Selfcheck` workflow 会跑
  `scripts/ppu/check-exclusions.py` 校验。失效的 `--ignore` 会被 pytest 静默吞掉，
  导致「以为排除了、其实在跑」或「以为在跑、其实没跑」。
- 新增 area 前先做 `workflow_call` 模板化：当前 11 份 workflow / 脚本是同构拷贝，
  再扩张会失控。详见 [`area-migration.md`](area-migration.md)。
- fork 仓库发起的 PR 不会进 PPU runner（特权容器 + 设备直通 + 凭证），需要跑
  PPU 测试的话请在本仓库开分支。
