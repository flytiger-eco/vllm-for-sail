# PPU K8s 分布式作业文档

## 容器运行时选项（container_options）

通过 container_options 传入一段 YAML（可多行），可调整 worker pod 的容器运行时行为。支持的键及其映射如下（留空时使用安全默认值，行为不变）：

| 键 | 示例值 | 映射 | 说明 |
| --- | --- | --- | --- |
| privileged | true / false | 容器 securityContext.privileged | 特权模式 |
| host_ipc | true / false | pod spec.hostIPC | 共享宿主机 IPC 命名空间 |
| host_network | true / false | pod spec.hostNetwork | 使用宿主机网络；置 true 时 dnsPolicy 自动补为 ClusterFirstWithHostNet |
| dns_policy | ClusterFirst / ClusterFirstWithHostNet / Default | pod spec.dnsPolicy | 显式指定时覆盖上述自动补齐 |
| shm_size | "8Gi" | /dev/shm emptyDir 的 sizeLimit | 默认 64Gi |
| cap_add | "SYS_PTRACE,IPC_LOCK" | securityContext.capabilities.add | 逗号分隔 |

注意：当 host_network=true 使用宿主机网络时，Action 会自动将 dnsPolicy 补齐为 ClusterFirstWithHostNet，否则注入的 MASTER_ADDR（Headless Service DNS）无法解析；若显式传入 dns_policy 则以显式值为准。

使用示例：

```yaml
steps:
  - uses: flytiger-eco/ppu-distributed-action@main
    with:
      image: your-image:tag
      command: |
        python train.py
      container_options: |
        privileged: true
        host_ipc: true
        host_network: true          # dnsPolicy 自动补为 ClusterFirstWithHostNet
        shm_size: "16Gi"            # /dev/shm 大小，默认 64Gi
        cap_add: "SYS_PTRACE,IPC_LOCK"
```

## 示例一：890P 单卡（1 PPU）

最简场景，适合功能验证、环境检查。

```yaml
jobs:
  test-890p-single:
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 单卡 (1 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 1
          node_selector: "board-type=ZW-M890P"
          host_volumes: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"  # 这些已默认自动挂载，此处为示意，实际使用可省略
          image: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
          command: |
            echo '=== PPU Verification ===' &&
            echo "Hostname: $(hostname), Node: $NODE_NAME" &&
            echo "RANK=$RANK NPROC_PER_NODE=$NPROC_PER_NODE" &&
            ppu-smi &&
            echo '--- ls /nas_aisw ---' && ls /nas_aisw &&
            echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas &&
            echo '=== DONE ==='
          namespace: ppu-sched
          timeout_minutes: 10
```

## 示例二：890P 多卡（8 PPU）

单节点多 PPU 卡，适合数据并行、张量并行训练。

```yaml
jobs:
  test-890p-multi-card:
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 多卡 (8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 8
          node_selector: "board-type=ZW-M890P"
          host_volumes: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"
          image: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
          command: |
            echo '=== PPU Verification ===' &&
            echo "Hostname: $(hostname), Node: $NODE_NAME" &&
            echo "RANK=$RANK NPROC_PER_NODE=$NPROC_PER_NODE" &&
            ppu-smi &&
            echo '--- ls /nas_aisw ---' && ls /nas_aisw &&
            echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas &&
            echo '=== DONE ==='
          namespace: ppu-sched
          timeout_minutes: 10
```

## 示例三：890P 多机（2 nodes × 8 PPU）

跨节点分布式训练，适合大模型预训练。

```yaml
jobs:
  test-890p-multi-node:
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 多机 (2 nodes × 8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 2
          nproc_per_node: 8
          node_selector: "board-type=ZW-M890P"
          host_volumes: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"
          image: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
          command: |
            echo '=== PPU Verification ===' &&
            echo "Hostname: $(hostname), Node: $NODE_NAME" &&
            echo "RANK=$RANK NODE_RANK=$NODE_RANK WORLD_SIZE=$WORLD_SIZE" &&
            echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT" &&
            ppu-smi &&
            echo '--- ls /nas_aisw ---' && ls /nas_aisw &&
            echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas &&
            echo '=== DONE ==='
          namespace: ppu-sched
          timeout_minutes: 10
```

要点：

* `nnodes: 2` 创建 2 个 worker Pod，每个 8 PPU，共 16 卡
* 自动创建 PodGroup（minMember=2），gang 调度确保两个 Pod 同时获得资源
* 自动创建 Headless Service，`$MASTER_ADDR` 指向 worker-0 的稳定 DNS
* 跨节点分布由 gang 调度（PodGroup）与资源约束自然达成

## 示例四：810E 单卡（1 PPU）

通过 node_selector 切换到 810E 节点。

```yaml
jobs:
  test-810e-single:
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "810E 单卡 (1 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 1
          node_selector: "board-type=OAM-810E"
          host_volumes: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"
          image: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
          command: |
            echo '=== PPU Verification ===' &&
            echo "Hostname: $(hostname), Node: $NODE_NAME" &&
            echo "RANK=$RANK NPROC_PER_NODE=$NPROC_PER_NODE" &&
            ppu-smi &&
            echo '--- ls /nas_aisw ---' && ls /nas_aisw &&
            echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas &&
            echo '=== DONE ==='
          namespace: ppu-sched
          timeout_minutes: 10
```

## 示例五：810E 多卡（8 PPU）

810E 节点有 16 张 PPU，可同时运行多个 8 卡任务。

```yaml
jobs:
  test-810e-multi-card:
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "810E 多卡 (8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 8
          node_selector: "board-type=OAM-810E"
          host_volumes: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"
          image: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
          command: |
            echo '=== PPU Verification ===' &&
            echo "Hostname: $(hostname), Node: $NODE_NAME" &&
            echo "RANK=$RANK NPROC_PER_NODE=$NPROC_PER_NODE" &&
            ppu-smi &&
            echo '--- ls /nas_aisw ---' && ls /nas_aisw &&
            echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas &&
            echo '=== DONE ==='
          namespace: ppu-sched
          timeout_minutes: 10
```

## 完整 Workflow 示例

以下是支持按场景选择执行的完整验证 workflow https://github.com/flytiger-eco/sglang-for-test/actions/runs/33373326501

```yaml
name: "PPU Distributed Scheduler Test"

on:
  workflow_dispatch:
    inputs:
      test_scenario:
        description: "测试场景"
        required: true
        type: choice
        options:
          - 890p-single
          - 890p-multi-card
          - 890p-multi-node
          - 810e-single
          - 810e-multi-card
          - all

env:
  IMAGE: "reg.docker.alibaba-inc.com/ai-tmp/llm:pytorch2.9.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312-ppu202605281910"
  NAMESPACE: "ppu-sched"
  HOST_VOLUMES: "/nas_aisw:/nas_aisw,/wl_nas:/mnt/wl_nas"
  COMMAND: "echo '=== PPU Verification ===' && echo \"Hostname: $(hostname), Node: $NODE_NAME\" && echo \"RANK=$RANK NODE_RANK=$NODE_RANK WORLD_SIZE=$WORLD_SIZE NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT\" && ppu-smi && echo '--- ls /nas_aisw ---' && ls /nas_aisw && echo '--- ls /mnt/wl_nas ---' && ls /mnt/wl_nas && echo '=== DONE ==='"

jobs:
  test-890p-single:
    if: ${{ inputs.test_scenario == '890p-single' || inputs.test_scenario == 'all' }}
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 单卡 (1 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 1
          node_selector: "board-type=ZW-M890P"
          host_volumes: ${{ env.HOST_VOLUMES }}
          image: ${{ env.IMAGE }}
          command: ${{ env.COMMAND }}
          namespace: ${{ env.NAMESPACE }}
          timeout_minutes: 10

  test-890p-multi-card:
    if: ${{ inputs.test_scenario == '890p-multi-card' || inputs.test_scenario == 'all' }}
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 多卡 (8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 8
          node_selector: "board-type=ZW-M890P"
          host_volumes: ${{ env.HOST_VOLUMES }}
          image: ${{ env.IMAGE }}
          command: ${{ env.COMMAND }}
          namespace: ${{ env.NAMESPACE }}
          timeout_minutes: 10

  test-890p-multi-node:
    if: ${{ inputs.test_scenario == '890p-multi-node' || inputs.test_scenario == 'all' }}
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "890P 多机 (2 nodes × 8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 2
          nproc_per_node: 8
          node_selector: "board-type=ZW-M890P"
          host_volumes: ${{ env.HOST_VOLUMES }}
          image: ${{ env.IMAGE }}
          command: ${{ env.COMMAND }}
          namespace: ${{ env.NAMESPACE }}
          timeout_minutes: 10

  test-810e-single:
    if: ${{ inputs.test_scenario == '810e-single' || inputs.test_scenario == 'all' }}
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "810E 单卡 (1 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 1
          node_selector: "board-type=OAM-810E"
          host_volumes: ${{ env.HOST_VOLUMES }}
          image: ${{ env.IMAGE }}
          command: ${{ env.COMMAND }}
          namespace: ${{ env.NAMESPACE }}
          timeout_minutes: 10

  test-810e-multi-card:
    if: ${{ inputs.test_scenario == '810e-multi-card' || inputs.test_scenario == 'all' }}
    runs-on: k8s-runner-group-cpu-flytiger
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - name: "810E 多卡 (8 PPU)"
        uses: flytiger-eco/ppu-distributed-action@main
        with:
          nnodes: 0
          nproc_per_node: 8
          node_selector: "board-type=OAM-810E"
          host_volumes: ${{ env.HOST_VOLUMES }}
          image: ${{ env.IMAGE }}
          command: ${{ env.COMMAND }}
          namespace: ${{ env.NAMESPACE }}
          timeout_minutes: 10
```

## 输出（Outputs）

| 输出 | 说明 |
| --- | --- |
| job_name | 作业标识名 |
| job_status | 最终状态：succeeded / failed / timeout |
| duration_seconds | 运行时长（秒） |
| task_pods | 所有 Pod 名称（逗号分隔） |
| log_artifact_name | 上传的日志 artifact 名称 |

## 日志

* Action 实时流式输出 worker-0 的日志到 GitHub Actions console
* 多机场景仅打印 worker-0 日志，其他 worker 可通过 kubectl logs 查看
* 作业失败时建议设置 `cleanup_policy: on_success` 保留 Pod 以便排查

## 清理策略

| 策略 | 行为 |
| --- | --- |
| always | 无论成功失败均清理（默认） |
| on_success | 成功后自动清理；失败保留用于调试 |
| on_failure | 失败时清理；成功时保留资源 |
| never | 从不自动清理（依赖集群 TTL 兑底） |

## t-head 组织使用说明

t-head 组织的仓库可以直接使用 ppu-distributed-action 提交 PPU 训练作业。flytiger-eco/ppu-distributed-action 是 public repo，t-head 组织的 workflow 可跨组织引用。

### runs-on 标签

t-head 组织使用专属的 CPU Runner Scale Set：

```yaml
runs-on: k8s-runner-group-cpu-thead
```

注意：ARC Runner Scale Set 不需要 self-hosted 标签，直接使用 `k8s-runner-group-cpu-thead` 即可。

### Action 引用

```yaml
uses: flytiger-eco/ppu-distributed-action@main
```

### 基本 Workflow 示例

```yaml
name: PPU Training

on:
  workflow_dispatch:

jobs:
  train:
    runs-on: k8s-runner-group-cpu-thead
    container:
      image: ghcr.io/actions/actions-runner:latest
    steps:
      - uses: actions/checkout@v4
      - uses: flytiger-eco/ppu-distributed-action@main
        with:
          image: <训练镜像>
          command: <训练命令>
          nnodes: 0
          nproc_per_node: 1
```

### 可用节点

| 板卡类型 | 标签 | PPU 数量/节点 | 节点数 |
| --- | --- | --- | --- |
| 890P | board-type=ZW-M890P | 8 PPU | 16 台 |
| 810E | board-type=OAM-810E | 16 PPU | 1 台 |

通过 node_selector 参数指定板卡类型：

```yaml
node_selector: "board-type=ZW-M890P"    # 890P 节点
node_selector: "board-type=OAM-810E"    # 810E 节点
```

### 完整示例

https://github.com/t-head/FlashMLA-for-sail/actions/runs/34199019544

### 注意事项

* 所有 Action 参数、环境变量契约、NAS 自动挂载、源码自动传递、容器运行时选项（container_options）等功能与 flytiger-eco 组织完全一致，参考上方文档各章节
* 唯一差异是 runs-on 标签使用 `k8s-runner-group-cpu-thead`
* Action 版本建议固定为 `@main`，或指定具体 tag/commit SHA
