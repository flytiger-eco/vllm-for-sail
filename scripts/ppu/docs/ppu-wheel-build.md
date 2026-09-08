# PPU Wheel 构建（feat/gha-ppu-test）：链路与使用说明

> 本文以 `.github/workflows/build-ppu-wheel.yml` + `scripts/ppu/build-ppu-wheel.sh`
> 的**当前内容**为准，描述这条已跑通的 PPU wheel 构建链路：怎么触发、在哪编、
> 怎么编、产物去哪。与 nightly 构建流水线（PR #6）的分工见 §3。

## 1. 用途

1. **分支级编译验证**：改动 `setup.py` / `csrc/` / `requirements/` 时，在合入
   维护分支前确认"编得出 wheel"。
2. **产出可分发的 dev wheel**：构建成功即发布 GitHub prerelease，内部环境直接
   `pip install`，免去每台机器重复源码编译。
3. **kernels area `csrc/**` 路径门禁的恢复前提**：`test-area-ppu-kernels.yml`
   刻意排除了 `csrc/**`（CI 里改 C++ 源码不会重编译，绿灯是假绿），接上
   wheel 构建后才可放回。

## 2. 链路总览

```text
触发：push(branches=[feat/gha-ppu-test], paths=[workflow 自身, 构建脚本])
      workflow_dispatch（已声明；未合入 main 前平台不注册，见 §5）

concurrency: ${{ github.workflow }}-${{ github.ref }}，cancel-in-progress: false
permissions: contents: read（workflow 级最小权限）

build-wheel  [ubuntu-latest]  timeout 480min
  if: github.repository == 'flytiger-eco/vllm-for-sail'   # fork 不白烧配额
  matrix: python=3.12, cuda=13.0, arch=x86_64（当前各 1 值）
  outputs: wheel_version / commit_hash / commit_sha / build_date
  steps:
    1. Clean workspace       alpine:3 一次性容器清 workspace（云 runner 是全新
                             VM，实际为空操作，沿用已验证路径先保留）
    2. Checkout code         actions/checkout@v4，submodules=recursive
    3. Compute wheel version 生成 0.23.0.dev<YYYYMMDD>+g<short-sha> 写入 outputs
    4. Build wheel           env: VLLM_VERSION_OVERRIDE（MAX_JOBS/NVCC_THREADS
                             配了但不进容器，见 §4.5）；
                             调 build-ppu-wheel.sh "0.23.0" <py> <cuda> <arch>
    5. Verify wheel version  dist/ 下必须恰好 1 个 wheel 且版本段 == 期望值
    6. Upload artifact       wheel-py3.12-cu13.0-x86_64-v0_23_0（默认保留期）

release-wheel  [ubuntu-latest]  needs: build-wheel（构建成功即发布）
  permissions: contents: write（写权限只落在本 job，不给跑编译的 job）
  steps:
    1. Download wheel        pattern=wheel-*-v0_23_0，merge-multiple
    2. List downloaded wheels
    3. Create GitHub Release（softprops/action-gh-release@v2）
       tag: dev-gha-ppu-test-<YYYY-MM-DD>-<short-sha>，prerelease: true
       target_commitish: 被编译的 commit；fail_on_unmatched_files: true
```

## 3. 与 nightly 构建流水线的分工

与 [PR #6](https://github.com/flytiger-eco/vllm-for-sail/pull/6)
（`release-pypi-ppu-nightly.yml`，拟合入 main）职责互补：

| 维度 | nightly 流水线 | 本 workflow |
| --- | --- | --- |
| 定位 | 维护分支每日构建 + 自动发 Release | 开发分支验证构建 |
| 构建对象 | 固定 pin `v0.23.0` / `v0.20.1` | 触发 run 的分支本身 |
| 脚本位置 | 单点维护在默认分支，双 checkout | 随分支携带（`scripts/ppu/`），单 checkout |
| 触发 | schedule + repository_dispatch + workflow_dispatch | 窄路径 push（合入 main 前唯一入口） |
| Release tag | `nightly-v0.23.0-*` / `nightly-v0.20.1-*` | `dev-gha-ppu-test-*`，只构建 v0.23.0 |

nightly 合入 main 后以 main 版本为准，本 workflow 只服务开发验证；两边脚本
若漂移需做 diff 同步。

## 4. 关键设计

1. **build job 跑 `ubuntu-latest`（GitHub 托管云 runner），不占 PPU 整机**。
   编译是纯 CPU 任务，不需要物理 PPU；且这是能编过的**唯一**环境：在带 PPU
   驱动的机器（self-hosted runner）上，PPU 用户态库（UKI）会向 **stdout**
   打 WARN，而 `cmake/utils.cmake` 的 `get_torch_gpu_compiler_flags` 用
   `execute_process(OUTPUT_VARIABLE)` 捕获 `torch.utils.cpp_extension` 打印的
   nvcc flags——WARN 文本被粘进 flags 首项，整串加引号后分号失去 list 分隔
   语义，`-DENABLE_FP8` 静默失效，fp8 头文件编译报错。云 runner 没有 PPU
   驱动，UKI 不进设备探测路径，flags 干净。**若将来回迁 self-hosted，必须先
   解决 UKI stdout 污染（如设备直通或改 cmake 捕获方式），否则会复现该
   失败。**
2. **版本号四层传递 + 产物硬校验**。版本 `0.23.0.dev<YYYYMMDD>+g<short-sha>`
   经 `VLLM_VERSION_OVERRIDE` 穿过 workflow → 脚本 → `docker -e` → setup.py；
   任一环节断掉会静默产出 setuptools_scm 推导版本。`Verify wheel version`
   解析最终 wheel 文件名版本段与期望值比对，无 wheel / 多 wheel / 不匹配均
   报错退出。
3. **引号化 heredoc `bash -s <<'INNER'`**（脚本侧）。容器内脚本含单引号
   （`printf '%s\n%s'`），嵌进 `bash -c '...'` 会被宿主 shell 吃掉转义。
   引号化 heredoc 不做宿主端展开，变量全靠 `docker -e` 传入；`-i` 必需
   （stdin 喂入）。
4. **build / release 拆两个 job**：`contents: write` 令牌不暴露给在容器里以
   root 执行编译的 build job。
5. **`MAX_JOBS=64` / `NVCC_THREADS=8` 当前是 no-op**：脚本里对应的
   `docker -e` 透传行保持注释（与已验证成功的配置一致），setup.py 实际按
   容器 cpu_count 决定并发。workflow 侧这两个 env 暂留，待单独清理。
6. 构建容器只挂 workspace + `--network=host`，不加 `--privileged` / 设备
   直通。构建镜像（`docker_build/pytorch:ubuntu24.04-py312.06`）、PPU SDK
   tarball（v2.1.1 / cuda-13.0.0）与 PPU torch wheel（2.11.0+cu130）均可从
   云 runner 匿名拉取。
7. **`if: github.repository == 'flytiger-eco/vllm-for-sail'`**：fork 仓库不
   白烧约 1.5h 的 Actions 构建配额。
8. 其他：`fail_on_unmatched_files: true`（glob 未命中不发布空 Release）；
   `prerelease: true`；`target_commitish` 指向实际被编译的 commit。

## 5. 触发与产物获取

```bash
# 未合入 main 前（现状）：push 触碰触发。改动必须真实命中两个 paths 之一
# （空提交不命中 paths 过滤）：
#   .github/workflows/build-ppu-wheel.yml
#   scripts/ppu/build-ppu-wheel.sh
git push origin feat/gha-ppu-test

# 合入 main 后：workflow_dispatch 可用（UI 按钮与 gh CLI 均可）
gh workflow run build-ppu-wheel.yml --ref feat/gha-ppu-test
# 构建成功即自动发布 prerelease，无输入参数
```

> `workflow_dispatch`（UI 按钮、gh CLI、REST API）都要求 workflow 已注册到
> 默认分支 main；合入前手动触发不可用（gh 报 `could not find any workflows`），
> 所以触发器只配了窄路径 push——只在刻意改构建管线时触发，业务代码 push
> 不会误触发小时级构建。合入 main 后 dispatch 选 `feat/gha-ppu-test` ref，
> 执行的仍是该分支上的 workflow 与脚本版本。

产物：

- **artifact**：`wheel-py3.12-cu13.0-x86_64-v0_23_0`（Actions 默认保留期）
- **Release**：Releases 页 Pre-releases 分组，tag
  `dev-gha-ppu-test-<YYYY-MM-DD>-<short-sha>`；wheel 挂为资产，
  `gh release download <tag>` 或直链下载（私有库需凭证）

## 6. 验证状态

**云端真实构建已通过**（run 34109691435，2026-09-07，commit `d9389af`）：

| 项 | 结果 |
| --- | --- |
| build-wheel | 1h29m3s（编译段 [1/89]→[89/89] 约 1h10m），success |
| release-wheel | 32s，success |
| 环境洁净度 | 日志中 `alixpu` 0 次、UKI WARN 0 次、无编译错误；nvcc flags 拆词正常 |
| wheel | `vllm-0.23.0.dev20260907+gd9389af-cp312-cp312-linux_x86_64.whl`，Verify wheel version 通过 |
| Release | `dev-gha-ppu-test-2026-09-07-d9389af`（prerelease），wheel 已挂载 |

**真机验证已通过**（ppu2，2026-09-08，同一 wheel）：

| 项 | 结果 |
| --- | --- |
| import 冒烟 | 版本号 / `vllm._C` / 16 卡可见正常 |
| serve 冒烟 | Qwen2.5-0.5B-Instruct（/nas_aisw）加载，`/v1/models` 就绪 |
| kernels 抽样 | core+mamba+moe 1531 passed/18 skipped；quantization 190 passed/531 deselected（CI 同款过滤）；合计 0 failed，~11min |

测试栈 = 分支源码 Python 层 + dev wheel 全部编译产物：先装 wheel 再跑
`ppu_install_dependency.sh`，其 `[cext]` 段从 site-packages（即 dev wheel）拷
`.so` 与 PPU 适配层进源码树。wheel 自带 PPU 适配 vllm_flash_attn（setup.py
`copy_ppu_flash_attn_overrides`：PPU_SDK 存在时把 `vllm/vllm_flash_attn/ppu/**`
覆盖到包顶层），纯 wheel 安装即可用。未覆盖：`chat/completions` 真实生成
（采样/detokenize 路径）、全量 kernels area（5000+ 例，留待 CI 接通）。

本地静态检查（2026-09-08 按当前文件重跑）：`bash -n` 脚本与 heredoc 容器内
脚本（65 行）通过；requirements 版本分支判定（≥0.20.1→ppu / 否则→cuda）正确；
workflow YAML 解析与本文 §2 描述一致；5 个 run 块（`${{ }}` 占位符替换后）
`bash -n` 全部通过；版本校验四场景（无 wheel→1 / 匹配→0 / 不匹配→1 /
多 wheel→1）符合预期。actionlint 本机不可用，合并前补
`pre-commit run actionlint --all-files`。

## 7. 已知限制

1. **`TORCH_CUDA_ARCH_LIST="8.0"`**：wheel 只含 SM80 kernel，其它架构硬件的
   消费者会失败，确认目标硬件范围前不要放开。
2. **aarch64 不支持**（无对应 base image / SDK tarball），脚本显式
   `exit 1`；扩展需同时补 base image、SDK tarball，且 matrix 多实例时 job
   outputs 会被最后完成的实例覆盖，届时需拆独立 job。
3. **self-hosted PPU 整机当前不可用于本构建**（UKI stdout 污染，见 §4.1）。
4. **配额与时长**：单次约 1.5h；仓库为 private 时消耗 Actions 分钟。timeout
   480min 相对实测有较大富余，可收紧。
5. **无效配置待清理**：`MAX_JOBS`/`NVCC_THREADS`（§4.5）、云 runner 上的
   Clean workspace 空操作、Rust 环境准备（容器内先 `=1` 后 `=0`）均为
   "与已验证成功配置保持一致"而暂留，各自单独评估后再动。
6. **双份维护**：与 nightly 流水线脚本同源，后者合入 main 后需 diff 同步，
   或评估本版退役、统一走 main。

## 8. 后续项

- [ ] 分别评估删除 Clean workspace / Rust 准备、收紧 timeout、清理无效
      MAX_JOBS/NVCC_THREADS 配置；每项单独改、单独跑，便于归因
- [ ] `pre-commit run actionlint --all-files`
- [ ] 处理 Node.js 20 弃用告警：checkout@v4 / upload-artifact@v4 /
      download-artifact@v4 / action-gh-release@v2 目前被平台强制跑 Node 24，
      暂不影响运行，适时升级 action 大版本
- [ ] 恢复 `test-area-ppu-kernels.yml` paths-filter 的 `csrc/**`：届时需决定
      构建产物如何被 test-area 消费——候选方案是 ci.yml 以 `workflow_call`
      先调本 workflow 再调 kernels area（本文件当前只有 push 与
      workflow_dispatch 两个触发器，走该方案需先补 `workflow_call` 入口）
- [ ] nightly 流水线合入 main 后评估脚本收敛（见 §7.6）
