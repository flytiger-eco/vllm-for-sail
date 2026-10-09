# PPU Wheel 构建集成进 PPU CI 流程 —— 方案与改动清单

> 目标：把 GHA 构建的 PPU dev wheel 变成 **PPU CI 的第一步**，让 10 个 test-area
> 统一消费本次 PR 构建出的 wheel，而不是各自借镜像里预装的 vllm `.so`。
> per-PR 只产 artifact、**不发 Release**；nightly 是另一条独立线，本次完全不碰。

## 1. 背景与需求

mentor 需求（原话转述）：

- 参考现有 build workflow，把它**集成到 CI 流程里**；
- **per-PR 的 build 不发 Release**，构建产物只给这一个 PR 用，把 release 那一步 skip 掉；
- 把 **build 和测试流程串起来**：build 是 CI 的一部分，跑 CI 的第一个步骤，build → 跑测试；
- 目的不是验证 build flow 本身，而是把 build flow 真正接进 CI 主链路。

约束：

- **最小改动**；
- nightly（`release-pypi-ppu-nightly.yml`，自建自发、不走 PR）是另一条逻辑，**本次不动**；
- v0.23.0 zero-diff 基线仍然成立：不改 `cmake/`、`csrc/`、`CMakeLists.txt`、`setup.py`
  （本次唯一涉及 `csrc/**` 的是 kernels area 的 **paths-filter 门禁恢复**，不是改源码）。

本次采用 **方案 B（最小首版）**：不新建 composite action；`workflow_dispatch` 首版
退回镜像预装 vllm（不做"自动取上次 wheel"）。

## 2. 目标链路

```
PR 打开/更新
  └─ ci.yml (pull_request 触发)
       ├─ precheck
       ├─ build-wheel   ──(workflow_call)──> build-ppu-wheel.yml
       │      └─ 产出 artifact: wheel-py3.12-cu13.0-x86_64-v0_23_0
       │         （build-ppu-wheel.yml 不发 Release，release-wheel job 已移除）
       ├─ smoke-test          ┐ 与 build-wheel 并行
       ├─ ai-code-review      ┘
       │
       ├─ 快速档 4 area (needs: smoke + ai-review + build-wheel)
       │      attention / model-executor / entrypoints / samplers
       ├─ human-review（仅带 ppu-full 标签时存在）
       └─ 标签档 6 area (needs: human-review + build-wheel)
              basic-correctness / entrypoints-llm / lora /
              models-basic / engine / kernels

       每个 area：download-artifact(同一 run，按名字取) → 挂进容器
                  → ppu_install_dependency.sh 装本次构建的 wheel
```

其他触发方式：

| 触发 | wheel 来源 | Release |
|---|---|---|
| **PR**（主路径） | build-wheel job 现构建，同 run 共享 artifact | 不发（无 release job） |
| **workflow_dispatch**（手动跑 area） | 不传 `wheel_artifact` → 退回镜像预装 vllm | 不发 |
| **nightly**（独立线） | 不涉及，另一条 workflow | 自建自发 |

> build-ppu-wheel.yml 的 `push` 触发与 `release-wheel` job 已按 mentor 要求整段移除：
> 该 workflow 现在只经 ci.yml 的 `workflow_call`（PR）或 `workflow_dispatch` 触发，
> 只产 artifact，任何路径都不再发 Release。

## 3. 关键设计点

1. **同 run artifact 共享，免传 run_id**：ci.yml 用 `uses:` 调 area（reusable
   workflow），build-wheel 与所有 area 处在同一个 top-level run 内，area 直接
   `actions/download-artifact` 按 **名字**（`wheel_artifact` 传入）取得产物，无需
   跨 run 传 run_id。

2. **per-PR 不发 Release（release-wheel job 整段移除）**：mentor 要求 per-PR 的 build
   不发 Release。由于 build-ppu-wheel.yml 现在只经 PR（workflow_call）/ workflow_dispatch
   触发、不再服务独立发布场景，直接**删除 `release-wheel` job 与 `push` 触发**，比留一个
   `if` 条件更干净；`contents: write` 权限也随之不再需要。

3. **`[cext]` 借 `.so` 机制不改**：`ppu_install_dependency.sh` 先装 dev wheel
   （`--force-reinstall --no-deps`，只换 vllm 本体，torch 等镜像依赖不动），
   使 site-packages 变成本次构建；随后原有 `[cext]` 段从 site-packages 拷 `.so`，
   拷到的就是本次构建的产物 —— `[cext]` 段无需任何改动。

4. **`PPU_WHEEL_PATH` 空则退回镜像（兜底）**：`wheel_artifact` 为空 → download /
   resolve step 被 `if: inputs.wheel_artifact != ''` 跳过 → `PPU_WHEEL_PATH` 未设 →
   `docker run -e PPU_WHEEL_PATH`（无 `=`）不把该变量传入容器 → 脚本 `[wheel]` 段
   跳过，沿用镜像预装 vllm（即现状行为）。

5. **路径复用 workspace 挂载**：`PPU_WHEEL_PATH` 指向 `/workspace/ci-wheel/<file>`，
   `ci-wheel/` 落在 workspace 下，已随现有 `-v ${{ github.workspace }}:/workspace`
   挂进容器，无需额外 `-v`。

6. **镜像不被取代**：镜像继续提供 PPU torch 2.11.0（PyPI 无）、SDK、驱动库、系统
   依赖；wheel 只覆盖 vllm 本体。

## 4. 文件改动清单

### 4.1 `scripts/ppu/ppu_install_dependency.sh`（+1 段）

在 `git config --global --add safe.directory` 之后、`[diag]` 段之前插入 `[wheel]`
段：设了 `PPU_WHEEL_PATH` 就先装该 wheel（`--force-reinstall --no-deps`）；文件不存在
则报错退出；未设则跳过（退回镜像）。

### 4.2 `.github/workflows/build-ppu-wheel.yml`（+workflow_call / -push / -release-wheel）

- `on:` 增加 `workflow_call`，暴露 `outputs`：`artifact_name`、`wheel_version`；
- **删除 `push` 触发**（PR 经 ci.yml → workflow_call 是唯一入口，保留 workflow_dispatch）；
- build-wheel job 的 `outputs` 增加 `artifact_name`；
- version step 输出
  `artifact_name=wheel-py${{ matrix.python-version }}-cu${{ matrix.cuda-version }}-${{ matrix.arch }}-v0_23_0`
  （即 `wheel-py3.12-cu13.0-x86_64-v0_23_0`）；
- **整段删除 `release-wheel` job**（per-PR 不发 Release）；随之清理仅供该 job 使用的
  死输出 `commit_hash`/`commit_sha`/`build_date` 及其 version step 计算。

### 4.3 `.github/workflows/ci.yml`（+build-wheel job / 10 area needs+with / 聚合点依赖）

- 新增 `build-wheel` job（`needs: precheck`，`uses: ./.github/workflows/build-ppu-wheel.yml`）；
- 4 个快速档 area：`needs` 加 `build-wheel`，`with: wheel_artifact: ${{ needs.build-wheel.outputs.artifact_name }}`；
- 6 个标签档 area：`needs` 加 `build-wheel`，同样传 `with.wheel_artifact`；
- Stage 5 聚合点 `needs` 加 `build-wheel`。

### 4.4 10 个 `test-area-ppu-*.yml`（每个 3 处 + kernels 额外 1 处）

覆盖：attention / model-executor / entrypoints / samplers / basic-correctness /
entrypoints-llm / lora / models-basic / engine / kernels。

每个 area 三处标准改动：

- **A. workflow_call.inputs** 增加 `wheel_artifact`（string，default `""`）；
- **B. test-job Checkout 后** 增加两个 step：
  - `Fetch CI-built wheel`（`if: inputs.wheel_artifact != ''`，`download-artifact`
    到 `ci-wheel/`）；
  - `Resolve wheel path`（同 if，把 `PPU_WHEEL_PATH=/workspace/ci-wheel/<file>` 写入
    `$GITHUB_ENV`）；
- **C. docker run env 列表** 增加 `-e PPU_WHEEL_PATH`。

kernels 额外第 4 处：paths-filter 里**放回 `csrc/**`**。接上 wheel 构建后，改 C++
源码会重新编译并参与 kernels 测试，之前"刻意不列 csrc/** 以免假绿"的规避可以撤销。

## 5. 校验

- 所有代码改动已就绪（`git status` 显示 13 个 workflow / 脚本文件 modified）；
- 已 grep 核实 10 个 area 各含 `wheel_artifact` input / `Fetch CI-built wheel` step /
  `-e PPU_WHEEL_PATH` 各 1 处，kernels 含 `csrc/**`。

## 6. 后续项（本次未做）

1. **缓存键控 build**：纯 Python PR 也要等 build（~1.5h）。可用
   `hashFiles('csrc/**','cmake/**','CMakeLists.txt','setup.py','requirements/**','scripts/ppu/build-ppu-wheel.sh')`
   命中缓存时跳过重编。
2. **`workflow_dispatch` 自动取上次 wheel**：需一个 composite action
   （方案 A 的 `.github/actions/ppu-wheel/action.yml`），本次按用户决定**不创建**。
3. **合入 main 后删 push 触发线**（PR-only 后 push 冗余）。
4. **actionlint 校验**：AGENTS.md 要求 `pre-commit run actionlint --all-files`；
   本机 actionlint 不可用，合并前需补。
